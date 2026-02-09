# source: https://github.com/tinkoff-ai/ReBRAC
# https://arxiv.org/abs/2305.09836

import os

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # For reproducibility

import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict
from flax.training.train_state import TrainState

# Import utilities from utils_jax
from .utils_jax import (
    ActorTrainState,
    AlgorithmInterface,
    CriticTrainState,
    Metrics,
    ReplayBuffer,
    compute_mean_std,
    evaluate,
    identity,
    make_env,
    normalize_states,
    pytorch_init,
    qlearning_dataset,
    uniform_init,
    wrap_env,
)

default_kernel_init = nn.initializers.lecun_normal()
default_bias_init = nn.initializers.zeros


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
    train_seed: int = 0
    eval_seed: int = 42

    def __post_init__(self):
        self.name = f"{self.name}-{self.dataset_name}-{str(uuid.uuid4())[:8]}"


class DetActor(nn.Module):
    action_dim: int
    hidden_dim: int = 256
    layernorm: bool = True
    n_hiddens: int = 3

    @nn.compact
    def __call__(self, state: jax.Array) -> jax.Array:
        s_d, h_d = state.shape[-1], self.hidden_dim
        # Initialization as in the EDAC paper
        layers = [
            nn.Dense(
                self.hidden_dim,
                kernel_init=pytorch_init(s_d),
                bias_init=nn.initializers.constant(0.1),
            ),
            nn.relu,
            nn.LayerNorm() if self.layernorm else identity,
        ]
        for _ in range(self.n_hiddens - 1):
            layers += [
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=pytorch_init(h_d),
                    bias_init=nn.initializers.constant(0.1),
                ),
                nn.relu,
                nn.LayerNorm() if self.layernorm else identity,
            ]
        layers += [
            nn.Dense(
                self.action_dim,
                kernel_init=uniform_init(1e-3),
                bias_init=uniform_init(1e-3),
            ),
            nn.tanh,
        ]
        net = nn.Sequential(layers)
        actions = net(state)
        return actions


class Critic(nn.Module):
    hidden_dim: int = 256
    layernorm: bool = True
    n_hiddens: int = 3

    @nn.compact
    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        s_d, a_d, h_d = state.shape[-1], action.shape[-1], self.hidden_dim
        # Initialization as in the EDAC paper
        layers = [
            nn.Dense(
                self.hidden_dim,
                kernel_init=pytorch_init(s_d + a_d),
                bias_init=nn.initializers.constant(0.1),
            ),
            nn.relu,
            nn.LayerNorm() if self.layernorm else identity,
        ]
        for _ in range(self.n_hiddens - 1):
            layers += [
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=pytorch_init(h_d),
                    bias_init=nn.initializers.constant(0.1),
                ),
                nn.relu,
                nn.LayerNorm() if self.layernorm else identity,
            ]
        layers += [
            nn.Dense(1, kernel_init=uniform_init(3e-3), bias_init=uniform_init(3e-3))
        ]
        network = nn.Sequential(layers)
        state_action = jnp.hstack([state, action])
        out = network(state_action).squeeze(-1)
        return out


class EnsembleCritic(nn.Module):
    hidden_dim: int = 256
    num_critics: int = 10
    layernorm: bool = True
    n_hiddens: int = 3

    @nn.compact
    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        ensemble = nn.vmap(
            target=Critic,
            in_axes=None,
            out_axes=0,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            axis_size=self.num_critics,
        )
        q_values = ensemble(self.hidden_dim, self.layernorm, self.n_hiddens)(
            state, action
        )
        return q_values




class ReBRACAlgorithm(AlgorithmInterface):
    """ReBRAC 알고리즘 구현체
    
    ReBRAC (Regularized Behavior Cloning) 알고리즘을 POGO Multi-Actor 구조에 통합.
    """
    def __init__(
        self,
        actor_bc_coef: float,  # actor loss에 사용
        critic_bc_coef: float,  # critic 업데이트에 사용
        gamma: float,
        tau: float,
        policy_noise: float,
        noise_clip: float,
        normalize_q: bool,
    ):
        self.actor_bc_coef = actor_bc_coef
        self.critic_bc_coef = critic_bc_coef
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
        
        key, actions_key = jax.random.split(key)

        next_actions = actor.apply_fn(actor.target_params, batch["next_states"])
        noise = jax.numpy.clip(
            (jax.random.normal(actions_key, next_actions.shape) * self.policy_noise),
            -self.noise_clip,
            self.noise_clip,
        )
        next_actions = jax.numpy.clip(next_actions + noise, -1, 1)
        bc_penalty = ((next_actions - batch["next_actions"]) ** 2).sum(-1)
        next_q = critic.apply_fn(
            critic.target_params, batch["next_states"], next_actions
        ).min(0)
        next_q = next_q - self.critic_bc_coef * bc_penalty

        target_q = batch["rewards"] + (1 - batch["dones"]) * self.gamma * next_q

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
    
    def update_actor0(
        self,
        key: jax.random.PRNGKey,
        actor: ActorTrainState,
        actor_module: nn.Module,
        critic: CriticTrainState,
        batch: Dict[str, jax.Array],
        flow_policy_state: Optional[ActorTrainState] = None,
        tau: float = 0.005,
        metrics: Optional[Metrics] = None,
    ) -> Tuple[jax.random.PRNGKey, ActorTrainState, Optional[ActorTrainState], Metrics]:
        """ReBRAC Actor0 업데이트
        
        Args:
            key: PRNG key
            actor: Actor0 train state
            actor_module: Actor0 module
            critic: Critic train state
            batch: Batch data
            flow_policy_state: Flow policy train state (not used for ReBRAC)
            tau: Target network update rate
            metrics: Metrics 객체
        
        Returns:
            업데이트된 key, actor, None (flow_policy_state), metrics
        """
        if metrics is None:
            metrics = Metrics.create([])
        
        def actor0_loss_fn(params: FrozenDict) -> Tuple[jax.Array, Metrics]:
            # Actor0의 actions 사용 (통합 인터페이스 사용)
            from .utils_jax import AlgorithmInterface
            actions = AlgorithmInterface._deterministic_actions(
                params, actor_module, batch["states"]
            )
            
            bc_penalty = jnp.sum((actions - batch["actions"]) ** 2, axis=-1)
            q_values = critic.apply_fn(critic.params, batch["states"], actions).min(0)
            lmbda = 1.0
            if self.normalize_q:
                lmbda = jax.lax.stop_gradient(1.0 / jnp.abs(q_values).mean())
            loss = (self.actor_bc_coef * bc_penalty - lmbda * q_values).mean()
            
            m = metrics.update({"actor_0_loss": loss})
            return loss, m
        
        (loss0, updated_metrics), grads0 = jax.value_and_grad(actor0_loss_fn, has_aux=True)(actor.params)
        new_actor = actor.apply_gradients(grads=grads0)
        
        # Target network update는 상위(update_multi_actor)에서 일괄 처리
        # 여기서는 params만 업데이트
        
        return key, new_actor, None, updated_metrics
    
    def compute_energy_function(
        self,
        actor_params: FrozenDict,
        actor_module: nn.Module,
        critic: CriticTrainState,
        batch: Dict[str, jax.Array],
        **kwargs
    ) -> jax.Array:
        """Energy function: -Q(state, π(state)) (Actor1+용)
        
        Args:
            actor_params: Actor parameters
            actor_module: Actor module
            critic: Critic train state
            batch: Batch data
            **kwargs: Additional arguments
        
        Returns:
            Energy function value: -Q.mean() (scalar)
        """
        from .utils_jax import AlgorithmInterface
        # Actor1+의 actions 추출
        actions = AlgorithmInterface._deterministic_actions(
            actor_params, actor_module, batch["states"]
        )
        
        # Q value 계산
        q_values = critic.apply_fn(critic.params, batch["states"], actions).min(0)
        
        # Energy function: -Q
        return -q_values.mean()


