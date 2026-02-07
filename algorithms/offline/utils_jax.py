# POGO Multi-Actor 공통 유틸리티
# AlgorithmInterface와 관련 클래스들을 정의

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from flax import linen as nn

from .rebrac import (
    CriticTrainState,
    ActorTrainState,
    Metrics,
    update_critic,
)


@dataclass
class ActorConfig:
    """Actor 설정을 그룹화하는 dataclass"""
    params: FrozenDict
    module: nn.Module
    is_stochastic: bool
    is_gaussian: bool


class AlgorithmInterface:
    """알고리즘별 인터페이스 정의
    
    다른 알고리즘(ReBRAC, FQL 외)으로 확장할 때 이 인터페이스를 구현하면 됨.
    PyTorch 버전의 PyTorchAlgorithmInterface와 유사한 구조.
    """
    @staticmethod
    def _deterministic_actions(
        actor_params: FrozenDict,
        actor_module: nn.Module,
        states: jax.Array,
    ) -> jax.Array:
        """Actor로부터 deterministic actions 추출 (통합 인터페이스)
        
        Policy 타입에 관계없이 일관된 방식으로 deterministic actions를 추출합니다.
        hasattr 분기를 제거하여 JIT trace 안정성과 가독성을 향상시킵니다.
        
        Args:
            actor_params: Actor parameters
            actor_module: Actor module
            states: [B, state_dim] states
        
        Returns:
            [B, action_dim] deterministic actions
        """
        if hasattr(actor_module, 'get_mean_std'):
            # Gaussian/TanhGaussian policy: use mean
            mean, _ = actor_module.get_mean_std(actor_params, states)
            return mean
        elif hasattr(actor_module, 'deterministic_actions'):
            # Deterministic/Stochastic policy: use deterministic_actions method
            return actor_module.deterministic_actions(actor_params, states)
        else:
            # Fallback: direct forward pass
            return actor_module.apply(actor_params, states)
    
    def update_critic(
        self,
        key: jax.random.PRNGKey,
        actor: ActorTrainState,
        critic: CriticTrainState,
        batch: Dict[str, jax.Array],
        actor_module: Optional[nn.Module] = None,
        **kwargs
    ) -> Tuple[jax.random.PRNGKey, CriticTrainState, Metrics]:
        """Critic 업데이트
        
        Args:
            key: PRNG key
            actor: Actor train state
            critic: Critic train state
            batch: Batch data
            actor_module: Actor module (optional, for algorithms that need it)
            **kwargs: Additional arguments
        """
        raise NotImplementedError
    
    def compute_actor_loss(
        self,
        actor_params: FrozenDict,
        actor_module: nn.Module,
        critic: CriticTrainState,
        batch: Dict[str, jax.Array],
        actor_idx: int = 0,
        **kwargs
    ) -> jax.Array:
        """Actor loss 계산
        
        Args:
            actor_params: Actor parameters
            actor_module: Actor module
            critic: Critic train state
            batch: Batch data
            actor_idx: Actor index (0 for Actor0, 1+ for Actor1+)
            **kwargs: Additional arguments
        """
        raise NotImplementedError


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
        actions = AlgorithmInterface._deterministic_actions(
            actor_params, actor_module, batch["states"]
        )
        
        bc_penalty = jnp.sum((actions - batch["actions"]) ** 2, axis=-1)
        q_values = critic.apply_fn(critic.params, batch["states"], actions).min(0)
        lmbda = 1.0
        if self.normalize_q:
            lmbda = jax.lax.stop_gradient(1.0 / jnp.abs(q_values).mean())
        return (self.beta * bc_penalty - lmbda * q_values).mean()
