# POGO Multi-Actor (JAX implementation)
# JAX 기반 알고리즘에 POGO Multi-Actor 구조 적용
# - Actor0: 원래 알고리즘 loss만 사용 (W2 penalty 없음)
# - Actor1+: 원래 알고리즘 loss + W2 distance to previous actor

import os

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # For reproducibility

import math
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import chex
import d4rl  # noqa
import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pyrallis
import wandb
from flax.core import FrozenDict
from flax.training.train_state import TrainState
from ott.geometry import pointcloud
from ott.solvers.linear import solve as sinkhorn_solve
# JAX utilities and ReBRAC imports (for now, can be extended to other algorithms)
from .rebrac import (
    Config as BaseConfig,
    CriticTrainState,
    ActorTrainState,
    Metrics,
    ReplayBuffer,
    update_critic,
    update_actor,
    update_td3,
    ReBRACAlgorithm,
)
# Environment utilities (PyTorch의 utils_pytorch와 대응)
from .utils_jax import wrap_env, evaluate, qlearning_dataset
# Network classes and utilities
from algorithms.networks.critics_jax import DetActor, Critic, EnsembleCritic
from algorithms.networks.mlp_jax import pytorch_init, uniform_init, identity, compute_mean_std, normalize_states
# JAX Actor implementations
from algorithms.networks.actors_jax import (
    GaussianMLP,
    TanhGaussianMLP,
    StochasticMLP,
    DeterministicMLP,
)
# AlgorithmInterface and ActorConfig
from .utils_jax import AlgorithmInterface, ActorConfig

default_kernel_init = nn.initializers.lecun_normal()
default_bias_init = nn.initializers.zeros


@dataclass
class Config(BaseConfig):
    """POGO Multi-Actor Config (JAX)"""
    # POGO Multi-Actor 설정
    # w2_weights: Actor1부터의 가중치 리스트 (Actor0는 W2 penalty 없음)
    w2_weights: List[float] = field(default_factory=lambda: [10.0, 10.0])
    num_actors: Optional[int] = None
    actor_configs: Optional[List[dict]] = None  # [{"type": "gaussian"|"stochastic"|"deterministic"}, ...]
    
    # Sinkhorn 설정 (Actor1+용, Gaussian이 아닌 경우에만 사용)
    sinkhorn_K: int = 4
    sinkhorn_blur: float = 0.05
    sinkhorn_backend: str = "auto"  # JAX에서는 ott-jax 사용 (실제로 OTT 사용)
    
    def __post_init__(self):
        super().__post_init__()
        if self.num_actors is None:
            # w2_weights는 Actor1부터이므로 num_actors = len(w2_weights) + 1
            self.num_actors = len(self.w2_weights) + 1
        # w2_weights는 Actor1부터이므로 num_actors - 1 길이여야 함
        expected_len = self.num_actors - 1
        if len(self.w2_weights) < expected_len:
            w = self.w2_weights[-1] if self.w2_weights else 10.0
            self.w2_weights = self.w2_weights + [w] * (expected_len - len(self.w2_weights))
        self.w2_weights = self.w2_weights[:expected_len]
        
        # actor_configs 기본값 설정
        if self.actor_configs is None:
            self.actor_configs = [{"type": "deterministic"} for _ in range(self.num_actors)]
        elif len(self.actor_configs) < self.num_actors:
            # 부족한 경우 마지막 설정으로 채움
            last_config = self.actor_configs[-1] if self.actor_configs else {"type": "deterministic"}
            self.actor_configs = self.actor_configs + [last_config] * (self.num_actors - len(self.actor_configs))
        self.actor_configs = self.actor_configs[:self.num_actors]


# Actor classes are now imported from algorithms.networks.actors_jax


def sinkhorn_distance_jax(
    x: jax.Array,
    y: jax.Array,
    blur: float = 0.05,
    num_iterations: int = 100,
) -> jax.Array:
    """
    Sinkhorn distance 계산 (OTT-jax 사용)
    
    Args:
        x: [B, K, action_dim] 첫 번째 분포의 샘플
        y: [B, K, action_dim] 두 번째 분포의 샘플 (detached)
        blur: regularization parameter (epsilon)
        num_iterations: Sinkhorn 알고리즘 반복 횟수 (OTT에서는 max_iterations로 전달)
    
    Returns:
        [B] 각 state에 대한 Sinkhorn distance
    """
    B, K, action_dim = x.shape
    
    # Uniform weights for each point cloud
    a = jnp.ones((B, K)) / K  # [B, K]
    b = jnp.ones((B, K)) / K  # [B, K]
    
    def compute_sinkhorn_for_batch(x_i, y_i, a_i, b_i):
        """Single batch Sinkhorn computation using OTT"""
        # Create geometry object
        geom = pointcloud.PointCloud(x_i, y_i, epsilon=blur)
        
        # Solve Sinkhorn
        out = sinkhorn_solve(geom, a_i, b_i, max_iterations=num_iterations)
        
        # Extract distance (transport cost)
        return out.reg_ot_cost
    
    # Vectorize over batch dimension
    distances = jax.vmap(compute_sinkhorn_for_batch)(x, y, a, b)  # [B]
    
    return distances


def closed_form_w2_gaussian(
    mean1: jax.Array,
    std1: jax.Array,
    mean2: jax.Array,
    std2: jax.Array,
) -> jax.Array:
    """
    Closed form W2 distance for Gaussian distributions
    W2² = ||μ1 - μ2||² + ||σ1 - σ2||²
    
    Args:
        mean1: [B, action_dim] mean of first Gaussian
        std1: [B, action_dim] std of first Gaussian
        mean2: [B, action_dim] mean of second Gaussian
        std2: [B, action_dim] std of second Gaussian
    
    Returns:
        [B] per-state W2² distance
    """
    mean_diff = mean1 - mean2  # [B, action_dim]
    std_diff = std1 - std2  # [B, action_dim]
    
    # W2² = ||μ1 - μ2||² + ||σ1 - σ2||²
    w2_squared = jnp.sum(mean_diff ** 2, axis=-1) + jnp.sum(std_diff ** 2, axis=-1)  # [B]
    return w2_squared


def per_state_sinkhorn(
    actor_i_config: ActorConfig,
    ref_actor_config: ActorConfig,
    states: jax.Array,
    key: jax.random.PRNGKey,
    K: int = 4,
    blur: float = 0.05,
) -> jax.Array:
    """
    Per-state distance 계산 (Gaussian closed form 또는 Sinkhorn 또는 L2)
    
    Args:
        actor_i_config: 현재 actor 설정
        ref_actor_config: 참조 actor 설정
        states: [B, state_dim]
        key: PRNG key
        K: 샘플 수 (Gaussian이 아닌 경우에만 사용)
        blur: Sinkhorn regularization (Gaussian이 아닌 경우에만 사용)
    
    Returns:
        평균 distance (scalar)
    """
    # Both Gaussian: use closed form W2
    if actor_i_config.is_gaussian and ref_actor_config.is_gaussian:
        mean_i, std_i = actor_i_config.module.get_mean_std(actor_i_config.params, states)  # [B, action_dim]
        mean_ref, std_ref = ref_actor_config.module.get_mean_std(ref_actor_config.params, states)  # [B, action_dim]
        mean_ref = jax.lax.stop_gradient(mean_ref)
        std_ref = jax.lax.stop_gradient(std_ref)
        
        w2_squared = closed_form_w2_gaussian(mean_i, std_i, mean_ref, std_ref)  # [B]
        return w2_squared.mean()
    
    # At least one is not Gaussian: use sampling-based methods
    key1, key2 = jax.random.split(key)
    
    # Sample from current actor
    a = actor_i_config.module.sample_actions(actor_i_config.params, states, key1, K)  # [B, K, action_dim]
    
    # Sample from reference actor (stop_gradient)
    b = ref_actor_config.module.sample_actions(ref_actor_config.params, states, key2, K)  # [B, K, action_dim]
    b = jax.lax.stop_gradient(b)
    
    if actor_i_config.is_stochastic and ref_actor_config.is_stochastic:
        # Both stochastic (but not Gaussian): use Sinkhorn
        distances = sinkhorn_distance_jax(a, b, blur=blur)  # [B]
        return distances.mean()
    else:
        # At least one deterministic: use L2
        a_det = a[:, 0, :]  # [B, action_dim] - take first sample
        b_det = b[:, 0, :]  # [B, action_dim]
        distances = jnp.sum((a_det - b_det) ** 2, axis=-1)  # [B]
        return distances.mean()


def update_multi_actor_gaussian(
    key: jax.random.PRNGKey,
    actors: List[ActorTrainState],
    critic: CriticTrainState,
    batch: Dict[str, jax.Array],
    metrics: Metrics,
    actor_modules: List[nn.Module],
    actor_is_gaussian: List[bool],
    w2_weights: List[float],
    beta: float,
    gamma: float,
    tau: float,
    policy_noise: float,
    noise_clip: float,
    normalize_q: bool,
) -> Tuple[jax.random.PRNGKey, List[ActorTrainState], CriticTrainState, Metrics]:
    """
    Multi-actor 업데이트 (Gaussian policy용 - closed form W2)
    현재는 ReBRAC 구조를 사용하지만, 다른 알고리즘으로 확장 가능
    """
    num_actors = len(actors)
    new_actors = []
    new_metrics = metrics
    
    # Critic 업데이트는 Actor0만 사용
    key, new_critic, new_metrics = update_critic(
        key,
        actors[0],
        critic,
        batch,
        gamma=gamma,
        beta=beta,
        tau=tau,
        policy_noise=policy_noise,
        noise_clip=noise_clip,
        metrics=new_metrics,
    )
    
    # Multi-actor 업데이트
    for i in range(num_actors):
        actor_i = actors[i]
        actor_module_i = actor_modules[i]
        
        key, actor_key = jax.random.split(key)
        
        def actor_loss_fn(params: FrozenDict) -> Tuple[jax.Array, Metrics]:
            # GaussianMLP: use mean
            mean_i, std_i = actor_module_i.get_mean_std(params, batch["states"])
            
            bc_penalty = ((mean_i - batch["actions"]) ** 2).sum(-1)
            q_values = new_critic.apply_fn(new_critic.params, batch["states"], mean_i).min(0)
            lmbda = 1.0
            if normalize_q:
                lmbda = jax.lax.stop_gradient(1.0 / jnp.abs(q_values).mean())
            
            if i == 0:
                # Actor0: ReBRAC loss만 사용
                loss = (beta * bc_penalty - lmbda * q_values).mean()
                actor_metrics = new_metrics.update({
                    f"actor_{i}_loss": loss,
                    f"actor_{i}_bc_mse": bc_penalty.mean(),
                })
                return loss, actor_metrics
            else:
                # Actor1+: Closed form W2
                ref_actor = actors[i - 1]
                ref_actor_module = actor_modules[i - 1]
                w2_weight_i = w2_weights[i - 1]
                
                rebrac_loss = (beta * bc_penalty - lmbda * q_values).mean()
                
                # Closed form W2
                mean_ref, std_ref = ref_actor_module.get_mean_std(ref_actor.params, batch["states"])
                mean_ref = jax.lax.stop_gradient(mean_ref)
                std_ref = jax.lax.stop_gradient(std_ref)
                
                w2_dist = closed_form_w2_gaussian(mean_i, std_i, mean_ref, std_ref).mean()
                
                loss = rebrac_loss + w2_weight_i * w2_dist
                
                actor_metrics = new_metrics.update({
                    f"actor_{i}_loss": loss,
                    f"actor_{i}_bc_mse": bc_penalty.mean(),
                    f"w2_{i}_distance": w2_dist,
                })
                return loss, actor_metrics
        
        grads, actor_metrics = jax.grad(actor_loss_fn, has_aux=True)(actor_i.params)
        new_actor_i = actor_i.apply_gradients(grads=grads)
        
        new_actor_i = new_actor_i.replace(
            target_params=optax.incremental_update(
                new_actor_i.params, actor_i.target_params, tau
            )
        )
        
        new_actors.append(new_actor_i)
        new_metrics = actor_metrics
    
    new_critic = new_critic.replace(
        target_params=optax.incremental_update(
            new_critic.params, critic.target_params, tau
        )
    )
    
    return key, new_actors, new_critic, new_metrics


def update_multi_actor_stochastic(
    key: jax.random.PRNGKey,
    actors: List[ActorTrainState],
    critic: CriticTrainState,
    batch: Dict[str, jax.Array],
    metrics: Metrics,
    actor_modules: List[nn.Module],
    actor_is_stochastic: List[bool],
    w2_weights: List[float],
    sinkhorn_K: int,
    sinkhorn_blur: float,
    beta: float,
    gamma: float,
    tau: float,
    policy_noise: float,
    noise_clip: float,
    normalize_q: bool,
) -> Tuple[jax.random.PRNGKey, List[ActorTrainState], CriticTrainState, Metrics]:
    """
    Multi-actor 업데이트 (Stochastic policy용 - Sinkhorn)
    현재는 ReBRAC 구조를 사용하지만, 다른 알고리즘으로 확장 가능
    """
    num_actors = len(actors)
    new_actors = []
    new_metrics = metrics
    
    # Critic 업데이트는 Actor0만 사용
    key, new_critic, new_metrics = update_critic(
        key,
        actors[0],
        critic,
        batch,
        gamma=gamma,
        beta=beta,
        tau=tau,
        policy_noise=policy_noise,
        noise_clip=noise_clip,
        metrics=new_metrics,
    )
    
    # Multi-actor 업데이트
    for i in range(num_actors):
        actor_i = actors[i]
        actor_module_i = actor_modules[i]
        
        key, actor_key = jax.random.split(key)
        
        def actor_loss_fn(params: FrozenDict) -> Tuple[jax.Array, Metrics]:
            # Get deterministic actions
            if actor_is_stochastic[i]:
                z_zero = jnp.zeros((batch["states"].shape[0], actor_module_i.action_dim), dtype=batch["states"].dtype)
                actions = actor_module_i.apply(params, batch["states"], z_zero)
            else:
                actions = actor_module_i.apply(params, batch["states"])
            
            bc_penalty = ((actions - batch["actions"]) ** 2).sum(-1)
            q_values = new_critic.apply_fn(new_critic.params, batch["states"], actions).min(0)
            lmbda = 1.0
            if normalize_q:
                lmbda = jax.lax.stop_gradient(1.0 / jnp.abs(q_values).mean())
            
            if i == 0:
                # Actor0: ReBRAC loss만 사용
                loss = (beta * bc_penalty - lmbda * q_values).mean()
                actor_metrics = new_metrics.update({
                    f"actor_{i}_loss": loss,
                    f"actor_{i}_bc_mse": bc_penalty.mean(),
                })
                return loss, actor_metrics
            else:
                # Actor1+: Sinkhorn or L2
                ref_actor = actors[i - 1]
                ref_actor_module = actor_modules[i - 1]
                w2_weight_i = w2_weights[i - 1]
                
                rebrac_loss = (beta * bc_penalty - lmbda * q_values).mean()
                
                # W2 distance (Sinkhorn or L2) using ActorConfig
                key_w2, _ = jax.random.split(actor_key)
                actor_i_config = ActorConfig(
                    params=params,
                    module=actor_module_i,
                    is_stochastic=actor_is_stochastic[i],
                    is_gaussian=False,  # Stochastic이므로 Gaussian 아님
                )
                ref_actor_config = ActorConfig(
                    params=ref_actor.params,
                    module=ref_actor_module,
                    is_stochastic=actor_is_stochastic[i - 1],
                    is_gaussian=False,  # Stochastic이므로 Gaussian 아님
                )
                w2_dist = per_state_sinkhorn(
                    actor_i_config=actor_i_config,
                    ref_actor_config=ref_actor_config,
                    states=batch["states"],
                    key=key_w2,
                    K=sinkhorn_K,
                    blur=sinkhorn_blur,
                )
                
                loss = rebrac_loss + w2_weight_i * w2_dist
                
                actor_metrics = new_metrics.update({
                    f"actor_{i}_loss": loss,
                    f"actor_{i}_bc_mse": bc_penalty.mean(),
                    f"w2_{i}_distance": w2_dist,
                })
                return loss, actor_metrics
        
        grads, actor_metrics = jax.grad(actor_loss_fn, has_aux=True)(actor_i.params)
        new_actor_i = actor_i.apply_gradients(grads=grads)
        
        new_actor_i = new_actor_i.replace(
            target_params=optax.incremental_update(
                new_actor_i.params, actor_i.target_params, tau
            )
        )
        
        new_actors.append(new_actor_i)
        new_metrics = actor_metrics
    
    new_critic = new_critic.replace(
        target_params=optax.incremental_update(
            new_critic.params, critic.target_params, tau
        )
    )
    
    return key, new_actors, new_critic, new_metrics


@pyrallis.wrap()
def main(config: Config):
    """POGO Multi-Actor 메인 함수 (JAX)"""
    dict_config = asdict(config)
    dict_config["mlc_job_name"] = os.environ.get("PLATFORM_JOB_NAME")

    wandb.init(
        config=dict_config,
        project=config.project,
        group=config.group,
        name=config.name,
        id=str(uuid.uuid4()),
    )
    wandb.mark_preempting()
    
    buffer = ReplayBuffer()
    buffer.create_from_d4rl(
        config.dataset_name, config.normalize_reward, config.normalize_states
    )

    key = jax.random.PRNGKey(seed=config.train_seed)
    key, actor_keys, critic_key = jax.random.split(key, config.num_actors + 2)
    actor_keys = jax.random.split(actor_keys, config.num_actors)

    eval_env = gym.make(config.dataset_name)
    eval_env.seed(config.eval_seed)
    eval_env.action_space.seed(config.eval_seed)
    eval_env.observation_space.seed(config.eval_seed)
    eval_env = wrap_env(eval_env, buffer.mean, buffer.std)
    init_state = buffer.data["states"][0][None, ...]
    init_action = buffer.data["actions"][0][None, ...]
    
    action_dim = init_action.shape[-1]
    max_action = float(eval_env.action_space.high[0])

    # Create multiple actors
    actors = []
    actor_modules = []
    actor_is_stochastic = []
    actor_is_gaussian = []
    
    if config.actor_configs is None:
        config.actor_configs = [{"type": "deterministic"} for _ in range(config.num_actors)]
    
    for i in range(config.num_actors):
        actor_config = config.actor_configs[i]
        actor_type = actor_config.get("type", "deterministic")  # "tanh_gaussian", "gaussian", "stochastic", or "deterministic"
        
        if actor_type == "gaussian":
            # Gaussian: mean에 tanh 적용된 상태에서 샘플링
            actor_module = GaussianMLP(
                action_dim=action_dim,
                max_action=max_action,
                hidden_dim=config.hidden_dim,
                layernorm=config.actor_ln,
                n_hiddens=config.actor_n_hiddens,
            )
            init_params = actor_module.init(actor_keys[i], init_state)
            init_target_params = actor_module.init(actor_keys[i], init_state)
        elif actor_type == "tanh_gaussian":
            # TanhGaussian: unbounded Gaussian에서 샘플링 후 tanh 적용
            actor_module = TanhGaussianMLP(
                action_dim=action_dim,
                max_action=max_action,
                hidden_dim=config.hidden_dim,
                layernorm=config.actor_ln,
                n_hiddens=config.actor_n_hiddens,
            )
            init_params = actor_module.init(actor_keys[i], init_state)
            init_target_params = actor_module.init(actor_keys[i], init_state)
        elif actor_type == "stochastic":
            actor_module = StochasticMLP(
                action_dim=action_dim,
                max_action=max_action,
                hidden_dim=config.hidden_dim,
                layernorm=config.actor_ln,
                n_hiddens=config.actor_n_hiddens,
            )
            # StochasticMLP: need state and z for init
            init_z = jnp.zeros((1, action_dim), dtype=init_state.dtype)
            init_params = actor_module.init(actor_keys[i], init_state, init_z)
            init_target_params = actor_module.init(actor_keys[i], init_state, init_z)
        else:  # deterministic
            actor_module = DeterministicMLP(
                action_dim=action_dim,
                max_action=max_action,
                hidden_dim=config.hidden_dim,
                layernorm=config.actor_ln,
                n_hiddens=config.actor_n_hiddens,
            )
            # DeterministicMLP: only state for init
            init_params = actor_module.init(actor_keys[i], init_state)
            init_target_params = actor_module.init(actor_keys[i], init_state)
        
        # Policy 클래스의 속성에서 자동으로 가져옴
        is_stochastic = getattr(actor_module, 'is_stochastic', False)
        is_gaussian = getattr(actor_module, 'is_gaussian', False)
        
        actor = ActorTrainState.create(
            apply_fn=actor_module.apply,
            params=init_params,
            target_params=init_target_params,
            tx=optax.adam(learning_rate=config.actor_learning_rate),
        )
        
        actors.append(actor)
        actor_modules.append(actor_module)
        actor_is_stochastic.append(is_stochastic)
        actor_is_gaussian.append(is_gaussian)

    # Create critic (EnsembleCritic 사용)
    critic_module = EnsembleCritic(
        hidden_dim=config.hidden_dim,
        num_critics=2,
        layernorm=config.critic_ln,
        n_hiddens=config.critic_n_hiddens,
    )
    critic = CriticTrainState.create(
        apply_fn=critic_module.apply,
        params=critic_module.init(critic_key, init_state, init_action),
        target_params=critic_module.init(critic_key, init_state, init_action),
        tx=optax.adam(learning_rate=config.critic_learning_rate),
    )

    # Gaussian과 Stochastic 구분
    use_gaussian = any(actor_is_gaussian)
    
    if use_gaussian:
        # Gaussian: closed form W2
        update_multi_actor_partial = partial(
            update_multi_actor_gaussian,
            actor_modules=actor_modules,
            actor_is_gaussian=actor_is_gaussian,
            w2_weights=config.w2_weights,
            beta=config.actor_bc_coef,
            gamma=config.gamma,
            tau=config.tau,
            policy_noise=config.policy_noise,
            noise_clip=config.noise_clip,
            normalize_q=config.normalize_q,
        )
    else:
        # Stochastic: Sinkhorn
        update_multi_actor_partial = partial(
            update_multi_actor_stochastic,
            actor_modules=actor_modules,
            actor_is_stochastic=actor_is_stochastic,
            w2_weights=config.w2_weights,
            sinkhorn_K=config.sinkhorn_K,
            sinkhorn_blur=config.sinkhorn_blur,
            beta=config.actor_bc_coef,
            gamma=config.gamma,
            tau=config.tau,
            policy_noise=config.policy_noise,
            noise_clip=config.noise_clip,
            normalize_q=config.normalize_q,
        )

    def loop_update_step(i: int, carry: Dict) -> Dict:
        key, batch_key = jax.random.split(carry["key"])
        batch = carry["buffer"].sample_batch(batch_key, batch_size=config.batch_size)

        key, new_actors, new_critic, new_metrics = update_multi_actor_partial(
            key=key,
            actors=carry["actors"],
            critic=carry["critic"],
            batch=batch,
            metrics=carry["metrics"],
        )

        carry.update({
            "key": key,
            "actors": new_actors,
            "critic": new_critic,
            "metrics": new_metrics,
        })
        return carry

    # Metrics
    bc_metrics_to_log = [
        "critic_loss",
        "q_min",
    ] + [f"actor_{i}_loss" for i in range(config.num_actors)] + \
      [f"actor_{i}_bc_mse" for i in range(config.num_actors)] + \
      [f"w2_{i}_distance" for i in range(1, config.num_actors)]
    
    # Shared carry for update loops
    update_carry = {
        "key": key,
        "actors": actors,
        "critic": critic,
        "buffer": buffer,
    }

    def make_actor_action_fn(actor_idx: int):
        """Create action function for a specific actor"""
        actor_module = actor_modules[actor_idx]
        is_gaussian = actor_is_gaussian[actor_idx]
        is_stoch = actor_is_stochastic[actor_idx]
        
        @jax.jit
        def _action_fn(params: FrozenDict, obs: jax.Array) -> jax.Array:
            if is_gaussian:
                # GaussianMLP: use mean
                mean, _ = actor_module.apply(params, obs)
                return mean
            elif is_stoch:
                # StochasticMLP: use z=0 for deterministic
                z_zero = jnp.zeros((obs.shape[0], actor_module.action_dim), dtype=obs.dtype)
                return actor_module.apply(params, obs, z_zero)
            else:
                # DeterministicMLP: state only
                return actor_module.apply(params, obs)
        
        return _action_fn

    print("---------------------------------------")
    print(f"Training POGO Multi-Actor (JAX), Env: {config.dataset_name}, Seed: {config.train_seed}")
    print(f"  Actors: {config.num_actors}, W2 weights (Actor1+): {config.w2_weights}")
    print("  Actor0: 원래 알고리즘 loss만 사용")
    print("  Actor1+: 알고리즘 loss + W2 distance")
    print("    - Gaussian actors: Closed form W2 (||μ1-μ2||² + ||σ1-σ2||²)")
    print("    - TanhGaussian/Stochastic actors: Sinkhorn distance")
    print("    - Stochastic (non-Gaussian): Sinkhorn distance")
    print("    - Deterministic: L2 distance")
    print(f"  Actor types: {[config.actor_configs[i].get('type', 'deterministic') for i in range(config.num_actors)]}")
    print("---------------------------------------")

    for epoch in range(config.num_epochs):
        # Reset metrics every epoch
        update_carry["metrics"] = Metrics.create(bc_metrics_to_log)

        update_carry = jax.lax.fori_loop(
            lower=0,
            upper=config.num_updates_on_epoch,
            body_fun=loop_update_step,
            init_val=update_carry,
        )
        
        # Log mean over epoch for each metric
        mean_metrics = update_carry["metrics"].compute()
        wandb.log(
            {"epoch": epoch, **{f"POGO_Multi_JAX/{k}": v for k, v in mean_metrics.items()}}
        )

        if epoch % config.eval_every == 0 or epoch == config.num_epochs - 1:
            # Evaluate each actor
            for actor_idx in range(config.num_actors):
                action_fn = make_actor_action_fn(actor_idx)
                eval_returns = evaluate(
                    eval_env,
                    update_carry["actors"][actor_idx].params,
                    action_fn,
                    config.eval_episodes,
                    seed=config.eval_seed,
                )
                normalized_score = eval_env.get_normalized_score(eval_returns) * 100.0
                wandb.log(
                    {
                        "epoch": epoch,
                        f"eval/actor_{actor_idx}_return_mean": np.mean(eval_returns),
                        f"eval/actor_{actor_idx}_return_std": np.std(eval_returns),
                        f"eval/actor_{actor_idx}_normalized_score_mean": np.mean(normalized_score),
                        f"eval/actor_{actor_idx}_normalized_score_std": np.std(normalized_score),
                    }
                )


if __name__ == "__main__":
    main()
