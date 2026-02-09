# FQL Algorithm for POGO Multi-Actor integration
# Adapted from https://github.com/seohongpark/fql
# Note: FQLFlowPolicy는 pogo_policies_jax.py에 정의되어 있음 (참고용, 실제로는 사용하지 않음)

from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax.core import FrozenDict
from flax import linen as nn

from .utils_jax import AlgorithmInterface
from .rebrac import ActorTrainState, CriticTrainState, Metrics


class FQLAlgorithm(AlgorithmInterface):
    """FQL Algorithm implementation for POGO Multi-Actor
    
    FQL (Flow Q-Learning) 알고리즘을 POGO Multi-Actor 구조에 통합.
    ReBRAC와 동일한 방식으로 AlgorithmInterface를 구현.
    
    FQL의 구조:
    - flow_policy (ActorVectorField): multi-step flow matching (velocity field 학습, actor_bc_flow)
    - actor_module (StochasticMLP): one-step policy (빠른 샘플링, actor_onestep_flow 역할)
    
    Actor loss:
    - Actor0: flow matching loss + (-Q + alpha(actor_0 - pi_flow)^2)
      - actor_0: StochasticMLP에서 샘플링 (one-step BC policy)
      - pi_flow: flow_policy에서 flow matching으로 계산된 action
    - Actor1+: Q loss만 (-Q, W2는 pogo_multi_jax에서 자동 추가됨)
    """
    def __init__(
        self,
        gamma: float = 0.99,
        tau: float = 0.005,
        q_agg: str = 'mean',
        normalize_q_loss: bool = False,
        alpha: float = 10.0,
        flow_steps: int = 10,
        flow_policy: Optional[nn.Module] = None,
    ):
        """
        Args:
            gamma: Discount factor
            tau: Target network update rate
            q_agg: Q value aggregation method ('mean' or 'min')
            normalize_q_loss: Whether to normalize Q loss
            alpha: Distillation loss coefficient (BC flow loss는 항상 1.0)
            flow_steps: Number of flow steps for BC flow
            flow_policy: Flow policy network (ActorVectorField, actor_bc_flow 역할, 별도 네트워크)
            Note: flow_policy_state를 외부에서 관리하고 update_actor0에 전달
        """
        self.gamma = gamma
        self.tau = tau
        self.q_agg = q_agg
        self.normalize_q_loss = normalize_q_loss
        self.alpha = alpha
        self.flow_steps = flow_steps
        self.flow_policy = flow_policy
    
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
        
        # FQL: StochasticMLP의 경우 deterministic_actions를 명시적으로 호출
        if hasattr(actor_module, 'deterministic_actions'):
            next_actions = actor_module.deterministic_actions(actor.params, batch["next_states"])
        else:
            next_actions = AlgorithmInterface._deterministic_actions(
                actor.params, actor_module, batch["next_states"]
            )  # [B, action_dim]
        
        # max_action: JIT 안전성을 위해 jnp.asarray로 처리
        max_action = getattr(actor_module, "max_action", 1.0)
        max_action = jnp.asarray(max_action, dtype=jnp.float32)
        next_actions = jnp.clip(next_actions, -max_action, max_action)
        
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
        
        # Metrics
        qs = critic.apply_fn(critic.params, batch["states"], batch["actions"])
        q = qs.mean(axis=0) if self.q_agg == 'mean' else qs.min(axis=0)
        critic_loss = jnp.square(q - target_q).mean()
        
        return key, new_critic, metrics.update({'critic_loss': critic_loss})
    
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
        """FQL Actor0 업데이트: flow_policy_params와 actor_params를 동시에 업데이트
        
        Args:
            key: PRNG key
            actor: Actor0 train state
            actor_module: Actor0 module (StochasticMLP for FQL)
            critic: Critic train state
            batch: Batch data
            flow_policy_state: Flow policy train state (필수)
            tau: Target network update rate
            metrics: Metrics 객체
        
        Returns:
            업데이트된 key, actor, flow_policy_state, metrics
        """
        if metrics is None:
            metrics = Metrics.create([])
        
        if flow_policy_state is None:
            raise ValueError("flow_policy_state must be provided for FQL")
        
        B = batch['states'].shape[0]
        action_dim = batch["actions"].shape[-1]
        max_action = getattr(actor_module, "max_action", 1.0)
        max_action = jnp.asarray(max_action, dtype=jnp.float32)
        
        # Loss 함수: flow_policy_params와 actor_params를 동시에 업데이트
        def loss_fn(flow_params: FrozenDict, actor_params: FrozenDict, loss_key: jax.random.PRNGKey) -> Tuple[jax.Array, Metrics]:
            loss_key, t_key, noise_key, flow_noise_key = jax.random.split(loss_key, 4)
            
            # 1. Flow matching loss: flow_policy (ActorVectorField, actor_bc_flow 역할) 학습
            x0 = jax.random.normal(noise_key, (B, action_dim))
            x1 = batch['actions'] / max_action
            x1 = jnp.clip(x1, -1.0, 1.0)
            t = jax.random.uniform(t_key, (B, 1), minval=0.0, maxval=1.0)
            x_t = (1 - t) * x0 + t * x1
            v_target = x1 - x0
            v_pred = self.flow_policy.apply(
                flow_params,
                batch['states'],
                x_t,
                times=t,
            )
            flow_loss = jnp.mean((v_pred - v_target) ** 2)
            
            # 2. One-step BC policy loss: -Q + alpha(actor_0 - pi_flow)^2
            # actor_0: StochasticMLP (actor_onestep_flow 역할)에서 z를 샘플링
            loss_key, z_key = jax.random.split(loss_key)
            z = jax.random.normal(z_key, (B, action_dim))
            actor_0 = actor_module.apply(actor_params, batch['states'], z)
            actor_0 = jnp.clip(actor_0, -max_action, max_action)
            
            # pi_flow: flow_policy (ActorVectorField)에서 flow matching으로 계산된 action
            x0_flow = jax.random.normal(flow_noise_key, (B, action_dim))
            x_t_flow = x0_flow
            for step in range(self.flow_steps):
                t_flow = jnp.full((B, 1), step / self.flow_steps)
                v_t = self.flow_policy.apply(
                    flow_params,
                    batch['states'],
                    x_t_flow,
                    times=t_flow,
                )
                x_t_flow = x_t_flow + v_t / self.flow_steps
            pi_flow = jnp.clip(x_t_flow, -1.0, 1.0) * max_action
            
            # Q loss
            qs = critic.apply_fn(critic.params, batch["states"], actor_0)
            q = qs.mean(axis=0) if self.q_agg == 'mean' else qs.min(axis=0)
            q_loss = -q.mean()
            if self.normalize_q_loss:
                lam = jax.lax.stop_gradient(1.0 / jnp.abs(q).mean())
                q_loss = lam * q_loss
            
            # Distillation loss
            actor_0_norm = actor_0 / max_action
            pi_flow_norm = pi_flow / max_action
            distil_loss = self.alpha * jnp.mean((actor_0_norm - pi_flow_norm) ** 2)
            
            total_loss = flow_loss + q_loss + distil_loss
            m = metrics.update({
                'actor_0_loss': total_loss,
                'flow_loss': flow_loss,
                'q_loss': q_loss,
                'distil_loss': distil_loss,
            })
            return total_loss, m
        
        # flow_policy_params와 actor_params를 동시에 업데이트
        key, loss_key = jax.random.split(key)
        (loss, updated_metrics), (flow_grads, actor_grads) = jax.value_and_grad(
            lambda fp, ap: loss_fn(fp, ap, loss_key), argnums=(0, 1), has_aux=True
        )(flow_policy_state.params, actor.params)
        
        new_flow_policy_state = flow_policy_state.apply_gradients(grads=flow_grads)
        new_actor = actor.apply_gradients(grads=actor_grads)
        
        # Target network updates는 update_multi_actor의 do_policy_update에서 통합적으로 처리
        return key, new_actor, new_flow_policy_state, updated_metrics
    
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
        # Actor1+의 actions 추출
        # key를 kwargs에서 받아서 전달
        key = kwargs.get("key", None)
        is_stochastic = kwargs.get("is_stochastic", None)
        
        if key is None:
            # key가 없으면 deterministic actions 사용
            actions = AlgorithmInterface._deterministic_actions(
                actor_params, actor_module, batch["states"],
                is_stochastic=is_stochastic
            )
        else:
            # key가 있으면 sample_actions 사용 (fallback 포함)
            key, sample_key = jax.random.split(key)
            actions = AlgorithmInterface.sample_actions(
                actor_params,
                actor_module,
                batch["states"],
                sample_key,
                K=1,
                is_stochastic=is_stochastic
            )
            # K=1일 때 shape이 [B, 1, A]일 수 있으므로 squeeze
            if actions.ndim == 3:
                actions = actions[:, 0, :]  # [B, 1, A] -> [B, A]
        
        # max_action: JIT 안전성을 위해 jnp.asarray로 처리
        max_action = getattr(actor_module, "max_action", 1.0)
        max_action = jnp.asarray(max_action, dtype=jnp.float32)
        actions = jnp.clip(actions, -max_action, max_action)
        
        # Q value 계산
        qs = critic.apply_fn(critic.params, batch["states"], actions)  # [num_critics, B]
        q = qs.mean(axis=0) if self.q_agg == 'mean' else qs.min(axis=0)
        
        # Energy function: -Q
        return -q.mean()
