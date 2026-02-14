# source: https://github.com/tinkoff-ai/ReBRAC
# https://arxiv.org/abs/2305.09836

import os

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # For reproducibility

import math
import uuid
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import chex
import gym
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pyrallis
import wandb
from flax.core import FrozenDict
from flax.training.train_state import TrainState

# Network classes are now imported from algorithms.networks.critics_jax
from algorithms.networks.critics_jax import DetActor, Critic, EnsembleCritic
# Utility functions are now imported from algorithms.networks.mlp_jax
from algorithms.networks.mlp_jax import compute_mean_std, normalize_states
# Environment utilities are now imported from utils_jax (PyTorch의 utils_pytorch와 대응)
from .utils_jax import wrap_env, evaluate, qlearning_dataset, seed_env_spaces, set_global_seed


@dataclass
class Config:
    # wandb params
    entity: str = "seonvin0319"
    project: str = "PORL"  # 기본값은 PORL
    group: str = "rebrac"
    name: str = "rebrac"
    # model params
    actor_learning_rate: float = 1e-3
    critic_learning_rate: float = 1e-3
    hidden_dim: int = 256
    actor_n_hiddens: int = 3
    critic_n_hiddens: int = 3
    gamma: float = 0.99
    tau: float = 5e-3
    actor_bc_coef: float = 1.0
    critic_bc_coef: float = 1.0
    actor_ln: bool = False
    critic_ln: bool = True
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    normalize_q: bool = True
    # training params
    dataset_name: str = "halfcheetah-medium-v2"
    batch_size: int = 1024
    num_epochs: int = 1000
    num_updates_on_epoch: int = 1000
    normalize_reward: bool = False
    normalize_states: bool = False
    # evaluation params
    eval_episodes: int = 10
    eval_freq: int = int(5e3)  # Update step 기반 평가 주기
    # general params
    seed: Optional[int] = None
    train_seed: int = 0
    eval_seed: int = 42

    def __post_init__(self):
        if self.seed is not None:
            self.train_seed = self.seed
            self.eval_seed = self.seed
        self.name = f"{self.name}-{self.dataset_name}-{str(uuid.uuid4())[:8]}"


# Network classes (DetActor, Critic, EnsembleCritic) are now in algorithms.networks.critics_jax
# Utility functions (compute_mean_std, normalize_states) are now in algorithms.networks.mlp_jax
# Environment utilities (wrap_env, evaluate, qlearning_dataset) are now in utils_jax (PyTorch의 utils_pytorch와 대응)


@chex.dataclass
class ReplayBuffer:
    data: Dict[str, jax.Array] = None
    mean: float = 0
    std: float = 1

    def create_from_d4rl(
        self,
        dataset_name: str,
        normalize_reward: bool = False,
        is_normalize: bool = False,
    ):
        d4rl_data = qlearning_dataset(gym.make(dataset_name))
        buffer = {
            "states": jnp.asarray(d4rl_data["observations"], dtype=jnp.float32),
            "actions": jnp.asarray(d4rl_data["actions"], dtype=jnp.float32),
            "rewards": jnp.asarray(d4rl_data["rewards"], dtype=jnp.float32),
            "next_states": jnp.asarray(
                d4rl_data["next_observations"], dtype=jnp.float32
            ),
            "next_actions": jnp.asarray(d4rl_data["next_actions"], dtype=jnp.float32),
            "dones": jnp.asarray(d4rl_data["terminals"], dtype=jnp.float32),
        }
        if is_normalize:
            self.mean, self.std = compute_mean_std(buffer["states"], eps=1e-3)
            buffer["states"] = normalize_states(buffer["states"], self.mean, self.std)
            buffer["next_states"] = normalize_states(
                buffer["next_states"], self.mean, self.std
            )
        if normalize_reward:
            buffer["rewards"] = ReplayBuffer.normalize_reward(
                dataset_name, buffer["rewards"]
            )
        self.data = buffer

    @property
    def size(self) -> int:
        # WARN: It will use len of the dataclass, i.e. number of fields.
        return self.data["states"].shape[0]

    def sample_batch(
        self, key: jax.random.PRNGKey, batch_size: int
    ) -> Dict[str, jax.Array]:
        indices = jax.random.randint(
            key, shape=(batch_size,), minval=0, maxval=self.size
        )
        batch = jax.tree.map(lambda arr: arr[indices], self.data)
        return batch

    def get_moments(self, modality: str) -> Tuple[jax.Array, jax.Array]:
        mean = self.data[modality].mean(0)
        std = self.data[modality].std(0)
        return mean, std

    @staticmethod
    def normalize_reward(dataset_name: str, rewards: jax.Array) -> jax.Array:
        if "antmaze" in dataset_name:
            return rewards * 100.0  # like in LAPO
        else:
            raise NotImplementedError(
                "Reward normalization is implemented only for AntMaze yet!"
            )


@chex.dataclass(frozen=True)
class Metrics:
    """순수 PyTree 기반 Metrics (deepcopy 제거, JAX 친화적)"""
    accumulators: Dict[str, Tuple[jax.Array, jax.Array]]

    @staticmethod
    def create(metrics: Sequence[str]) -> "Metrics":
        init_metrics = {key: (jnp.array(0.0), jnp.array(0.0)) for key in metrics}
        return Metrics(accumulators=init_metrics)

    def update(self, updates: Dict[str, jax.Array]) -> "Metrics":
        """순수 함수형 업데이트 (deepcopy 제거)"""
        new_accumulators = {}
        for key in self.accumulators.keys():
            acc, steps = self.accumulators[key]
            if key in updates:
                new_accumulators[key] = (acc + updates[key], steps + 1)
            else:
                new_accumulators[key] = (acc, steps)
        return self.replace(accumulators=new_accumulators)

    def compute(self) -> Dict[str, np.ndarray]:
        # cumulative_value / total_steps
        return {k: np.array(v[0] / v[1]) if v[1] > 0 else np.array(0.0) for k, v in self.accumulators.items()}


# normalize, make_env, wrap_env, evaluate are now in utils_jax_env


class CriticTrainState(TrainState):
    target_params: FrozenDict


class ActorTrainState(TrainState):
    target_params: FrozenDict


def update_actor(
    key: jax.random.PRNGKey,
    actor: TrainState,
    critic: TrainState,
    batch: Dict[str, jax.Array],
    beta: float,
    tau: float,
    normalize_q: bool,
    metrics: Metrics,
) -> Tuple[jax.random.PRNGKey, TrainState, TrainState, Metrics]:
    key, random_action_key = jax.random.split(key, 2)

    def actor_loss_fn(params: jax.Array) -> Tuple[jax.Array, Metrics]:
        actions = actor.apply_fn(params, batch["states"])

        bc_penalty = ((actions - batch["actions"]) ** 2).sum(-1)
        q_values = critic.apply_fn(critic.params, batch["states"], actions).min(0)
        lmbda = 1
        if normalize_q:
            lmbda = jax.lax.stop_gradient(1 / jax.numpy.abs(q_values).mean())

        loss = (beta * bc_penalty - lmbda * q_values).mean()

        # logging stuff
        random_actions = jax.random.uniform(
            random_action_key, shape=batch["actions"].shape, minval=-1.0, maxval=1.0
        )
        new_metrics = metrics.update(
            {
                "actor_loss": loss,
                "bc_mse_policy": bc_penalty.mean(),
                "bc_mse_random": ((random_actions - batch["actions"]) ** 2)
                .sum(-1)
                .mean(),
                "action_mse": ((actions - batch["actions"]) ** 2).mean(),
            }
        )
        return loss, new_metrics

    grads, new_metrics = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    new_actor = actor.apply_gradients(grads=grads)

    new_actor = new_actor.replace(
        target_params=optax.incremental_update(new_actor.params, actor.target_params, tau)
    )
    new_critic = critic.replace(
        target_params=optax.incremental_update(critic.params, critic.target_params, tau)
    )

    return key, new_actor, new_critic, new_metrics


def update_critic(
    key: jax.random.PRNGKey,
    actor: TrainState,
    critic: CriticTrainState,
    batch: Dict[str, jax.Array],
    gamma: float,
    beta: float,
    tau: float,
    policy_noise: float,
    noise_clip: float,
    metrics: Metrics,
) -> Tuple[jax.random.PRNGKey, TrainState, Metrics]:
    key, actions_key = jax.random.split(key)

    next_actions = actor.apply_fn(actor.target_params, batch["next_states"])
    noise = jax.numpy.clip(
        (jax.random.normal(actions_key, next_actions.shape) * policy_noise),
        -noise_clip,
        noise_clip,
    )
    next_actions = jax.numpy.clip(next_actions + noise, -1, 1)
    bc_penalty = ((next_actions - batch["next_actions"]) ** 2).sum(-1)
    next_q = critic.apply_fn(
        critic.target_params, batch["next_states"], next_actions
    ).min(0)
    next_q = next_q - beta * bc_penalty

    target_q = batch["rewards"] + (1 - batch["dones"]) * gamma * next_q

    def critic_loss_fn(critic_params: jax.Array) -> Tuple[jax.Array, jax.Array]:
        # [N, batch_size] - [1, batch_size]
        q = critic.apply_fn(critic_params, batch["states"], batch["actions"])
        q_min = q.min(0).mean()
        loss = ((q - target_q[None, ...]) ** 2).mean(1).sum(0)
        return loss, q_min

    (loss, q_min), grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(
        critic.params
    )
    new_critic = critic.apply_gradients(grads=grads)
    new_metrics = metrics.update(
        {
            "critic_loss": loss,
            "q_min": q_min,
        }
    )
    return key, new_critic, new_metrics


def update_td3(
    key: jax.random.PRNGKey,
    actor: TrainState,
    critic: CriticTrainState,
    batch: Dict[str, Any],
    metrics: Metrics,
    gamma: float,
    actor_bc_coef: float,
    critic_bc_coef: float,
    tau: float,
    policy_noise: float,
    noise_clip: float,
    normalize_q: bool,
) -> Tuple[jax.random.PRNGKey, TrainState, TrainState, Metrics]:
    key, new_critic, new_metrics = update_critic(
        key,
        actor,
        critic,
        batch,
        gamma,
        critic_bc_coef,
        tau,
        policy_noise,
        noise_clip,
        metrics,
    )
    key, new_actor, new_critic, new_metrics = update_actor(
        key, actor, new_critic, batch, actor_bc_coef, tau, normalize_q, new_metrics
    )
    return key, new_actor, new_critic, new_metrics


def update_td3_no_targets(
    key: jax.random.PRNGKey,
    actor: TrainState,
    critic: CriticTrainState,
    batch: Dict[str, Any],
    gamma: float,
    metrics: Metrics,
    critic_bc_coef: float,
    tau: float,
    policy_noise: float,
    noise_clip: float,
) -> Tuple[jax.random.PRNGKey, TrainState, TrainState, Metrics]:
    key, new_critic, new_metrics = update_critic(
        key,
        actor,
        critic,
        batch,
        gamma,
        critic_bc_coef,
        tau,
        policy_noise,
        noise_clip,
        metrics,
    )
    return key, actor, new_critic, new_metrics


# action_fn은 main 함수에서만 사용되므로 여기서 정의
def action_fn(actor: TrainState) -> Callable:
    @jax.jit
    def _action_fn(obs: jax.Array) -> jax.Array:
        action = actor.apply_fn(actor.params, obs)
        return action

    return _action_fn


@pyrallis.wrap()
def main(config: Config):
    if config.seed is not None:
        config.train_seed = config.seed
        config.eval_seed = config.seed
    key = set_global_seed(config.train_seed)

    dict_config = asdict(config)
    dict_config["mlc_job_name"] = os.environ.get("PLATFORM_JOB_NAME")

    wandb.init(
        config=dict_config,
        entity=config.entity,
        project=config.project,
        group=config.group,
        name=config.name,
        id=str(uuid.uuid4()),
    )
    try:
        wandb.mark_preempting()
    except Exception:
        pass
    buffer = ReplayBuffer()
    buffer.create_from_d4rl(
        config.dataset_name, config.normalize_reward, config.normalize_states
    )

    key, actor_key, critic_key = jax.random.split(key, 3)

    eval_env = gym.make(config.dataset_name)
    seed_env_spaces(eval_env, config.eval_seed)
    eval_env = wrap_env(eval_env, buffer.mean, buffer.std)
    init_state = buffer.data["states"][0][None, ...]
    init_action = buffer.data["actions"][0][None, ...]

    actor_module = DetActor(
        action_dim=init_action.shape[-1],
        hidden_dim=config.hidden_dim,
        layernorm=config.actor_ln,
        n_hiddens=config.actor_n_hiddens,
    )
    actor = ActorTrainState.create(
        apply_fn=actor_module.apply,
        params=actor_module.init(actor_key, init_state),
        target_params=actor_module.init(actor_key, init_state),
        tx=optax.adam(learning_rate=config.actor_learning_rate),
    )

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

    update_td3_partial = partial(
        update_td3,
        gamma=config.gamma,
        actor_bc_coef=config.actor_bc_coef,
        critic_bc_coef=config.critic_bc_coef,
        tau=config.tau,
        policy_noise=config.policy_noise,
        noise_clip=config.noise_clip,
        normalize_q=config.normalize_q,
    )

    update_td3_no_targets_partial = partial(
        update_td3_no_targets,
        gamma=config.gamma,
        critic_bc_coef=config.critic_bc_coef,
        tau=config.tau,
        policy_noise=config.policy_noise,
        noise_clip=config.noise_clip,
    )


    # metrics (batch_entropy 제거: 실제로 업데이트되지 않음)
    bc_metrics_to_log = [
        "critic_loss",
        "q_min",
        "actor_loss",
        "bc_mse_policy",
        "bc_mse_random",
        "action_mse",
    ]
    @jax.jit
    def actor_action_fn(params: jax.Array, obs: jax.Array):
        """Actor action function with shape handling"""
        # obs를 배치로 변환 (1D -> 2D)
        obs_batch = obs[None, :] if obs.ndim == 1 else obs
        action_batch = actor.apply_fn(params, obs_batch)
        return action_batch[0] if action_batch.shape[0] == 1 else action_batch

    current_key = key
    current_actor = actor
    current_critic = critic

    for epoch in range(config.num_epochs):
        # metrics for accumulation during epoch and logging to wandb
        # Python for 루프에서 누적 (JAX 친화적)
        epoch_metrics = Metrics.create(bc_metrics_to_log)

        # Python for 루프로 변경 (buffer 접근을 JIT 밖으로)
        for i in range(config.num_updates_on_epoch):
            # 배치 샘플링 (Python side)
            current_key, batch_key = jax.random.split(current_key)
            batch = buffer.sample_batch(batch_key, batch_size=config.batch_size)

            # 업데이트 (JAX side)
            should_update_actor = (i % config.policy_freq == 0)
            if should_update_actor:
                current_key, current_actor, current_critic, temp_metrics = update_td3_partial(
                    key=current_key,
                    actor=current_actor,
                    critic=current_critic,
                    batch=batch,
                    metrics=epoch_metrics,
                )
            else:
                current_key, current_actor, current_critic, temp_metrics = update_td3_no_targets_partial(
                    key=current_key,
                    actor=current_actor,
                    critic=current_critic,
                    batch=batch,
                    metrics=epoch_metrics,
                )
            epoch_metrics = temp_metrics

        # log mean over epoch for each metric
        mean_metrics = epoch_metrics.compute()
        wandb.log(
            {"epoch": epoch, **{f"ReBRAC/{k}": v for k, v in mean_metrics.items()}}
        )

        # Update step 기반 평가 (오프라인 RL에서는 gradient update step 기준)
        global_update_step = (epoch + 1) * config.num_updates_on_epoch
        if epoch == 0 or (global_update_step % config.eval_freq == 0) or epoch == config.num_epochs - 1:
            eval_returns = evaluate(
                eval_env,
                current_actor.params,
                actor_action_fn,
                config.eval_episodes,
                seed=config.eval_seed,
            )
            normalized_score = eval_env.get_normalized_score(eval_returns) * 100.0
            wandb.log(
                {
                    "epoch": epoch,
                    "eval/return_mean": np.mean(eval_returns),
                    "eval/return_std": np.std(eval_returns),
                    "eval/normalized_score_mean": np.mean(normalized_score),
                    "eval/normalized_score_std": np.std(normalized_score),
                }
            )


# ============================================================================
# ReBRACAlgorithm for POGO Multi-Actor
# ============================================================================

from .utils_jax import AlgorithmInterface


class ReBRACAlgorithm(AlgorithmInterface):
    """ReBRAC 알고리즘 구현체
    
    ReBRAC (Regularized Behavior Cloning) 알고리즘을 POGO Multi-Actor 구조에 통합.
    """
    def __init__(
        self,
        beta: float,
        gamma: float,
        tau: float,
        policy_noise: float,
        noise_clip: float,
        normalize_q: bool,
    ):
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.normalize_q = normalize_q
    
    def update_critic(
        self,
        key: jax.random.PRNGKey,
        actor: ActorTrainState,
        critic: CriticTrainState,
        batch: Dict[str, jax.Array],
        actor_module: Optional[nn.Module] = None,
        metrics: Optional[Metrics] = None,
        **kwargs
    ) -> Tuple[jax.random.PRNGKey, CriticTrainState, Metrics]:
        """ReBRAC critic 업데이트"""
        if metrics is None:
            metrics = Metrics.create([])
        return update_critic(
            key,
            actor,
            critic,
            batch,
            gamma=self.gamma,
            beta=self.beta,
            tau=self.tau,
            policy_noise=self.policy_noise,
            noise_clip=self.noise_clip,
            metrics=metrics,
        )
    
    def compute_actor_loss(
        self,
        actor_params: FrozenDict,
        actor_module: nn.Module,
        critic: CriticTrainState,
        batch: Dict[str, jax.Array],
        actor_idx: int = 0,
        **kwargs
    ) -> jax.Array:
        """ReBRAC actor loss 계산
        
        Args:
            actor_params: Actor parameters
            actor_module: Actor module
            critic: Critic train state
            batch: Batch data
            actor_idx: Actor index (not used for ReBRAC, but kept for interface consistency)
            **kwargs: Additional arguments
        """
        # Actor0의 actions 사용 (통합 인터페이스 사용)
        from .utils_jax import AlgorithmInterface
        actions = AlgorithmInterface._deterministic_actions(
            actor_params, actor_module, batch["states"]
        )
        
        bc_penalty = jnp.sum((actions - batch["actions"]) ** 2, axis=-1)
        q_values = critic.apply_fn(critic.params, batch["states"], actions).min(0)
        lmbda = 1.0
        if self.normalize_q:
            lmbda = jax.lax.stop_gradient(1.0 / jnp.abs(q_values).mean())
        return (self.beta * bc_penalty - lmbda * q_values).mean()
    
    def compute_energy_function(
        self,
        actor_params: FrozenDict,
        actor_module: nn.Module,
        critic: CriticTrainState,
        batch: Dict[str, jax.Array],
        actor_idx: int = 1,
        **kwargs
    ) -> jax.Array:
        """ReBRAC energy function for Actor1+
        
        ReBRAC의 energy function: -Q(state, π(state))
        
        Args:
            actor_params: Actor parameters
            actor_module: Actor module (Actor1+)
            critic: Critic train state
            batch: Batch data
            actor_idx: Actor index (should be >= 1 for Actor1+)
            **kwargs: Additional arguments
        
        Returns:
            Energy function value (scalar)
        """
        # Deterministic actions 사용
        from .utils_jax import AlgorithmInterface
        actions = AlgorithmInterface._deterministic_actions(
            actor_params, actor_module, batch["states"]
        )
        
        q_values = critic.apply_fn(critic.params, batch["states"], actions).min(0)
        lmbda = 1.0
        if self.normalize_q:
            lmbda = jax.lax.stop_gradient(1.0 / jnp.abs(q_values).mean())
        
        # Energy function: -Q (Q를 최대화하려면 -Q를 최소화)
        energy = -lmbda * q_values.mean()
        return energy


if __name__ == "__main__":
    main()
