# FQL Algorithm for POGO Multi-Actor integration
# Adapted from https://github.com/seohongpark/fql
# Note: FQLFlowPolicy는 pogo_policies_jax.py에 정의되어 있음

from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from flax import linen as nn

from .utils_jax import AlgorithmInterface
from .rebrac import ActorTrainState, CriticTrainState, Metrics
from .pogo_policies_jax import FQLFlowPolicy


class FQLAlgorithm(AlgorithmInterface):
    """FQL Algorithm implementation for POGO Multi-Actor
    
    FQL (Flow Q-Learning) 알고리즘을 POGO Multi-Actor 구조에 통합.
    ReBRAC와 동일한 방식으로 AlgorithmInterface를 구현.
    
    FQL의 multi-step flow matching 구조:
    - actor_bc_flow: multi-step flow matching (velocity field 학습)
    - actor_onestep_flow: one-step policy (빠른 샘플링)
    
    Actor loss:
    - Actor0: BC flow loss만 (BC policy)
    - Actor1+: Q loss만 (-Q, W2는 _update_single_actor에서 자동 추가됨)
    """
    def __init__(
        self,
        gamma: float = 0.99,
        tau: float = 0.005,
        q_agg: str = 'mean',
        normalize_q_loss: bool = False,
        alpha: float = 10.0,
        flow_steps: int = 10,
    ):
        """
        Args:
            gamma: Discount factor
            tau: Target network update rate
            q_agg: Q value aggregation method ('mean' or 'min')
            normalize_q_loss: Whether to normalize Q loss
            alpha: Distillation loss coefficient (BC flow loss는 항상 1.0)
            flow_steps: Number of flow steps for BC flow
        """
        self.gamma = gamma
        self.tau = tau
        self.q_agg = q_agg
        self.normalize_q_loss = normalize_q_loss
        self.alpha = alpha
        self.flow_steps = flow_steps
    
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
        """FQL critic 업데이트
        
        Args:
            key: PRNG key
            actor: Actor train state
            critic: Critic train state
            batch: Batch data
            actor_module: Actor module (required for FQL)
            metrics: Metrics 객체
            **kwargs: Additional arguments
        
        Returns:
            업데이트된 key, critic, metrics
        """
        if actor_module is None:
            raise ValueError("FQL requires actor_module to be provided")
        if metrics is None:
            metrics = Metrics.create([])
        
        key, sample_rng = jax.random.split(key)
        
        # Next actions 샘플링 (Actor0 사용)
        next_actions = actor_module.sample_actions(
            actor.params,
            batch["next_states"],
            sample_rng,
            K=1
        )  # [B, 1, action_dim]
        next_actions = next_actions.squeeze(1)  # [B, action_dim]
        next_actions = jnp.clip(next_actions, -1, 1)
        
        # Target Q 계산
        next_qs = critic.apply_fn(
            critic.target_params,
            batch["next_states"],
            next_actions
        )  # [num_critics, B]
        if self.q_agg == 'min':
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)
        
        target_q = batch["rewards"] + self.gamma * (1 - batch["dones"]) * next_q
        
        # Critic loss
        def critic_loss_fn(params):
            qs = critic.apply_fn(params, batch["states"], batch["actions"])  # [num_critics, B]
            q = qs.mean(axis=0) if self.q_agg == 'mean' else qs.min(axis=0)
            loss = jnp.square(q - target_q).mean()
            return loss
        
        grads = jax.grad(critic_loss_fn)(critic.params)
        new_critic = critic.apply_gradients(grads=grads)
        
        # Target network update
        new_target_params = jax.tree.map(
            lambda p, tp: p * self.tau + tp * (1 - self.tau),
            new_critic.params,
            critic.target_params,
        )
        new_critic = new_critic.replace(target_params=new_target_params)
        
        # Metrics
        qs = critic.apply_fn(critic.params, batch["states"], batch["actions"])
        q = qs.mean(axis=0) if self.q_agg == 'mean' else qs.min(axis=0)
        critic_loss = jnp.square(q - target_q).mean()
        
        return key, new_critic, metrics.update({'critic_loss': critic_loss})
    
    def compute_actor_loss(
        self,
        actor_params: FrozenDict,
        actor_module: nn.Module,
        critic: CriticTrainState,
        batch: Dict[str, jax.Array],
        actor_idx: int = 0,
        key: Optional[jax.random.PRNGKey] = None,
        **kwargs
    ) -> jax.Array:
        """FQL actor loss 계산
        
        FQL의 actor loss:
        - Actor0: BC flow loss만 (BC policy)
        - Actor1+: Q loss만 (-Q, W2는 _update_single_actor에서 자동 추가됨)
        
        Args:
            actor_params: Actor parameters
            actor_module: FQLFlowPolicy module (Actor0) or other policy (Actor1+)
            critic: Critic train state
            batch: Batch data
            actor_idx: Actor index (0 for Actor0, 1+ for Actor1+)
            key: PRNG key (required for FQL)
            **kwargs: Additional arguments
        
        Returns:
            Actor loss (scalar)
        """
        if key is None:
            raise ValueError("FQL requires key to be provided")
        # Actor0: BC flow loss만 사용 (BC policy)
        if actor_idx == 0:
            # FQLFlowPolicy 사용
            batch_size, action_dim = batch['actions'].shape
            key, x_rng, t_rng = jax.random.split(key, 3)
            
            # BC flow loss: actor_bc_flow의 velocity 예측
            x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
            x_1 = batch['actions'] / actor_module.max_action  # Normalize to [-1, 1]
            t = jax.random.uniform(t_rng, (batch_size, 1))
            x_t = (1 - t) * x_0 + t * x_1
            vel = x_1 - x_0  # True velocity
            
            # Predict velocity using actor_bc_flow
            bc_flow_params = actor_params['actor_bc_flow']
            pred_vel = actor_module.actor_bc_flow.apply(
                {'params': bc_flow_params},
                batch['states'],
                x_t,
                t,
                is_encoded=False,
            )
            bc_flow_loss = jnp.mean((pred_vel - vel) ** 2)
            return bc_flow_loss
        
        # Actor1+: Q loss만 (-Q, W2는 _update_single_actor에서 추가됨)
        # Actor1+는 다른 policy 타입 사용 가능 (stochastic, deterministic 등)
        key, sample_rng = jax.random.split(key)
        
        # Actions 추출 (통합 인터페이스 사용)
        # Note: FQL Actor1+는 deterministic actions를 사용 (Q loss 계산용)
        from .utils_jax import AlgorithmInterface
        actions = AlgorithmInterface._deterministic_actions(
            actor_params, actor_module, batch['states']
        )
        
        actions = jnp.clip(actions, -actor_module.max_action, actor_module.max_action)
        
        # Q loss: Q value 최대화 (-Q)
        qs = critic.apply_fn(critic.params, batch["states"], actions)  # [num_critics, B]
        q = qs.mean(axis=0) if self.q_agg == 'mean' else qs.min(axis=0)
        
        q_loss = -q.mean()
        if self.normalize_q_loss:
            lam = jax.lax.stop_gradient(1.0 / jnp.abs(q).mean())
            q_loss = lam * q_loss
        
        return q_loss
