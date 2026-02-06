# POGO Multi-Actor Main Script
# 각 알고리즘(IQL, CQL, TD3_BC 등)의 구조를 그대로 사용하되,
# Actor만 multi-actor로 교체하여 학습
# Config에서 algorithm 선택 가능

import copy
import math
import os
import random
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import d4rl
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
# wandb removed - no longer used
from geomloss import SamplesLoss

# 각 알고리즘 import
from .iql import ImplicitQLearning, TwinQ, ValueFunction, GaussianPolicy, DeterministicPolicy
from .td3_bc import TD3_BC, Actor, Critic
from .cql import ContinuousCQL, TanhGaussianPolicy, FullyConnectedQFunction, Scalar
from .awac import AdvantageWeightedActorCritic, soft_update
from .sac_n import SACN, Actor as SACNActor, VectorizedCritic
from .edac import EDAC
from .pogo_policies import BaseActor, DeterministicMLP, StochasticMLP, GaussianMLP, TanhGaussianMLP

TensorBatch = List[torch.Tensor]


@dataclass
class ActorConfig:
    """Actor 설정을 그룹화하는 dataclass (PyTorch 버전)
    JAX 버전과 일관성을 유지하기 위해 동일한 구조 사용
    """
    actor: nn.Module
    is_stochastic: bool
    is_gaussian: bool
    
    @classmethod
    def from_actor(cls, actor: nn.Module) -> 'ActorConfig':
        """Actor 객체로부터 ActorConfig 생성"""
        return cls(
            actor=actor,
            is_stochastic=getattr(actor, 'is_stochastic', False),
            is_gaussian=getattr(actor, 'is_gaussian', False),
        )


class EnsembleCritic(nn.Module):
    """Ensemble Critic for PyTorch (ReBRAC style)
    Multiple independent critics, returns [num_critics, batch_size] Q values
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_critics: int = 2,
        layernorm: bool = True,
        n_hiddens: int = 3,
    ):
        super().__init__()
        self.num_critics = num_critics
        self.hidden_dim = hidden_dim
        self.layernorm = layernorm
        
        # Create multiple independent critics
        self.critics = nn.ModuleList([
            self._make_critic(state_dim, action_dim, hidden_dim, layernorm, n_hiddens)
            for _ in range(num_critics)
        ])
    
    def _make_critic(self, state_dim, action_dim, hidden_dim, layernorm, n_hiddens):
        """Create a single critic network"""
        layers = []
        # First layer
        layers.append(nn.Linear(state_dim + action_dim, hidden_dim))
        layers.append(nn.ReLU())
        if layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        
        # Hidden layers
        for _ in range(n_hiddens - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            if layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, 1))
        
        # Initialize as in ReBRAC (EDAC style)
        net = nn.Sequential(*layers)
        for i, layer in enumerate(net):
            if isinstance(layer, nn.Linear):
                if i == 0:
                    # First layer: constant bias 0.1
                    nn.init.constant_(layer.bias, 0.1)
                    # Kaiming init for weight
                    nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
                elif i == len([l for l in net if isinstance(l, nn.Linear)]) - 1:
                    # Last layer: uniform init [-3e-3, 3e-3]
                    nn.init.uniform_(layer.weight, -3e-3, 3e-3)
                    nn.init.uniform_(layer.bias, -3e-3, 3e-3)
                else:
                    # Hidden layers: constant bias 0.1
                    nn.init.constant_(layer.bias, 0.1)
                    nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
        
        return net
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Args:
            state: [batch_size, state_dim]
            action: [batch_size, action_dim]
        Returns:
            q_values: [num_critics, batch_size]
        """
        state_action = torch.cat([state, action], dim=-1)  # [batch_size, state_dim + action_dim]
        q_values = torch.stack([
            critic(state_action).squeeze(-1)  # [batch_size]
            for critic in self.critics
        ], dim=0)  # [num_critics, batch_size]
        return q_values


@dataclass
class TrainConfig:
    # Experiment
    device: str = "cuda"
    env: str = "halfcheetah-medium-v2"
    seed: int = 0
    eval_freq: int = int(5e3)
    n_episodes: int = 10
    max_timesteps: int = int(1e6)
    checkpoints_path: Optional[str] = None
    load_model: str = ""

    # Algorithm 선택 (필수)
    algorithm: str = "iql"  # "iql", "cql", "td3_bc", "awac" 등

    # POGO Multi-Actor 설정
    # w2_weights: Actor1부터의 가중치 리스트 (Actor0는 W2 penalty 없음)
    w2_weights: List[float] = field(default_factory=lambda: [10.0, 10.0])
    num_actors: Optional[int] = None
    actor_configs: Optional[List[dict]] = None

    # Sinkhorn
    sinkhorn_K: int = 4
    sinkhorn_blur: float = 0.05
    sinkhorn_backend: str = "tensorized"

    # Algorithm별 파라미터는 각 알고리즘의 config에서 가져옴
    # IQL
    iql_tau: float = 0.7
    beta: float = 3.0
    vf_lr: float = 3e-4
    qf_lr: float = 3e-4
    actor_lr: float = 3e-4
    iql_deterministic: bool = False
    actor_dropout: Optional[float] = None

    # TD3_BC
    alpha: float = 2.5
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2

    # CQL
    cql_alpha: float = 10.0
    cql_n_actions: int = 10
    target_entropy: Optional[float] = None
    
    # AWAC
    awac_lambda: float = 1.0
    exp_adv_max: float = 100.0
    
    # SAC-N
    num_critics: int = 10
    alpha_learning_rate: float = 3e-4
    
    # EDAC
    eta: float = 1.0

    # 공통
    batch_size: int = 256
    discount: float = 0.99
    tau: float = 0.005
    buffer_size: int = 2_000_000
    normalize: bool = True
    normalize_reward: bool = False

    # Wandb
    project: str = "CORL"
    group: str = "POGO-Multi"
    name: str = "POGO-Multi"

    def __post_init__(self):
        if self.num_actors is None:
            # w2_weights는 Actor1부터이므로 num_actors = len(w2_weights) + 1
            self.num_actors = len(self.w2_weights) + 1
        # w2_weights는 Actor1부터이므로 num_actors - 1 길이여야 함
        expected_len = self.num_actors - 1
        if len(self.w2_weights) < expected_len:
            w = self.w2_weights[-1] if self.w2_weights else 10.0
            self.w2_weights = self.w2_weights + [w] * (expected_len - len(self.w2_weights))
        self.w2_weights = self.w2_weights[:expected_len]
        self.name = f"{self.name}-{self.algorithm}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


def _per_state_sinkhorn(
    policy,
    ref,
    states: torch.Tensor,
    K: int = 4,
    blur: float = 0.05,
    sinkhorn_loss=None,
    seed: Optional[int] = None,
):
    if sinkhorn_loss is None:
        sinkhorn_loss = SamplesLoss(loss="sinkhorn", p=2, blur=blur, backend="tensorized")
    a = policy.sample_actions(states, K=K, seed=None if seed is None else seed + 0)
    with torch.no_grad():
        b_detached = ref.sample_actions(states, K=K, seed=None if seed is None else seed + 10000).detach()
    loss = sinkhorn_loss(a, b_detached)
    return loss.mean() if loss.dim() > 0 else loss


def _create_multi_actors(
    state_dim: int,
    action_dim: int,
    max_action: float,
    num_actors: int,
    actor_configs: Optional[List[dict]],
    algorithm: str,
    device: str,
    lr: float,
):
    """Multi-actor 생성: 모든 알고리즘에서 POGO policies 사용 (일관성)
    Policy 클래스의 is_gaussian, is_stochastic 속성을 자동으로 사용
    """
    actors = []
    actor_targets = []
    actor_optimizers = []
    actor_is_stochastic = []
    actor_is_gaussian = []

    if actor_configs is None:
        actor_configs = [{} for _ in range(num_actors)]

    for i in range(num_actors):
        config = actor_configs[i]
        actor_type = config.get("type", "stochastic") if isinstance(config, dict) else getattr(config, "type", "stochastic")

        # 모든 알고리즘에서 POGO policies 사용 (sample_actions, deterministic_actions 지원)
        if actor_type == "gaussian":
            # Gaussian: mean에 tanh 적용된 상태에서 샘플링
            actor = GaussianMLP(state_dim, action_dim, max_action).to(device)
        elif actor_type == "tanh_gaussian":
            # TanhGaussian: unbounded Gaussian에서 샘플링 후 tanh 적용
            actor = TanhGaussianMLP(state_dim, action_dim, max_action).to(device)
        elif actor_type == "deterministic":
            actor = DeterministicMLP(state_dim, action_dim, max_action).to(device)
        else:  # stochastic
            actor = StochasticMLP(state_dim, action_dim, max_action).to(device)

        # Policy 클래스의 속성에서 자동으로 가져옴
        is_stochastic = getattr(actor, 'is_stochastic', False)
        is_gaussian = getattr(actor, 'is_gaussian', False)

        actor_target = copy.deepcopy(actor)
        actors.append(actor)
        actor_targets.append(actor_target)
        actor_optimizers.append(torch.optim.Adam(actor.parameters(), lr=lr))
        actor_is_stochastic.append(is_stochastic)
        actor_is_gaussian.append(is_gaussian)

    return actors, actor_targets, actor_optimizers, actor_is_stochastic, actor_is_gaussian


def _closed_form_w2_gaussian(mean1, std1, mean2, std2):
    """Closed form W2 distance for Gaussian distributions: ||μ1-μ2||² + ||σ1-σ2||²"""
    mean_diff = mean1 - mean2
    std_diff = std1 - std2
    w2_squared = torch.sum(mean_diff ** 2, dim=-1) + torch.sum(std_diff ** 2, dim=-1)
    return w2_squared.mean()


def _compute_w2_distance(
    actor_i_config: ActorConfig,
    ref_actor_config: ActorConfig,
    states: torch.Tensor,
    sinkhorn_K: int = 4,
    sinkhorn_blur: float = 0.05,
    sinkhorn_loss=None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    공통 W2 distance 계산 함수 (JAX 버전과 일관성 유지)
    Policy 타입에 따라 자동으로 적절한 방법 선택:
    - Both Gaussian → closed form W2
    - Both Stochastic (not Gaussian) → Sinkhorn
    - At least one Deterministic → L2
    
    Args:
        actor_i_config: 현재 actor 설정
        ref_actor_config: 참조 actor 설정
        states: [B, state_dim]
        sinkhorn_K: 샘플 수 (Gaussian이 아닌 경우에만 사용)
        sinkhorn_blur: Sinkhorn regularization (Gaussian이 아닌 경우에만 사용)
        sinkhorn_loss: Sinkhorn loss 객체 (None이면 자동 생성)
        seed: 랜덤 시드
    
    Returns:
        평균 distance (scalar)
    """
    # Both Gaussian: use closed form W2
    if actor_i_config.is_gaussian and ref_actor_config.is_gaussian:
        mean_i, std_i = actor_i_config.actor.get_mean_std(states)
        with torch.no_grad():
            mean_ref, std_ref = ref_actor_config.actor.get_mean_std(states)
        return _closed_form_w2_gaussian(mean_i, std_i, mean_ref, std_ref)
    
    # At least one is not Gaussian: use sampling-based methods
    if actor_i_config.is_stochastic and ref_actor_config.is_stochastic:
        # Both stochastic (but not Gaussian): use Sinkhorn
        return _per_state_sinkhorn(
            actor_i_config.actor, ref_actor_config.actor, states,
            K=sinkhorn_K, blur=sinkhorn_blur,
            sinkhorn_loss=sinkhorn_loss,
            seed=seed,
        )
    else:
        # At least one deterministic: use L2
        pi_i = actor_i_config.actor.deterministic_actions(states)
        with torch.no_grad():
            ref_action = ref_actor_config.actor.deterministic_actions(states)
        return torch.sum((pi_i - ref_action) ** 2, dim=-1).mean()


def _compute_actor_loss_with_w2(
    base_loss_fn: Callable[[nn.Module], torch.Tensor],
    actor_i_config: ActorConfig,
    ref_actor_config: Optional[ActorConfig],
    states: torch.Tensor,
    w2_weight: float,
    sinkhorn_K: int = 4,
    sinkhorn_blur: float = 0.05,
    sinkhorn_loss=None,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    공통 multi-actor loss 계산 함수 (JAX 버전과 일관성 유지)
    Actor0: base loss만 사용
    Actor1+: base loss + w2_weight * w2_distance
    
    Args:
        base_loss_fn: Base loss 계산 함수
        actor_i_config: 현재 actor 설정
        ref_actor_config: 참조 actor 설정 (Actor0인 경우 None)
        states: [B, state_dim]
        w2_weight: W2 distance 가중치
        sinkhorn_K: 샘플 수
        sinkhorn_blur: Sinkhorn regularization
        sinkhorn_loss: Sinkhorn loss 객체
        seed: 랜덤 시드
    
    Returns:
        actor_loss: 계산된 loss
        w2_distance: W2 distance (Actor0인 경우 None)
    """
    base_loss = base_loss_fn(actor_i_config.actor)
    
    if ref_actor_config is None:
        # Actor0: W2 penalty 없음
        return base_loss, None
    
    # Actor1+: W2 distance 계산
    w2_distance = _compute_w2_distance(
        actor_i_config=actor_i_config,
        ref_actor_config=ref_actor_config,
        states=states,
        sinkhorn_K=sinkhorn_K,
        sinkhorn_blur=sinkhorn_blur,
        sinkhorn_loss=sinkhorn_loss,
        seed=seed,
    )
    
    return base_loss + w2_weight * w2_distance, w2_distance


def _train_iql_multi_actor_gaussian(
    trainer: ImplicitQLearning,
    actors: List[nn.Module],
    actor_targets: List[nn.Module],
    actor_optimizers: List[torch.optim.Optimizer],
    actor_is_gaussian: List[bool],
    w2_weights: List[float],
    batch: TensorBatch,
    seed_base: int,
) -> Dict[str, float]:
    """IQL trainer에 multi-actor 적용 (Gaussian policy용 - closed form W2)"""
    trainer.total_it += 1
    observations, actions, rewards, next_observations, dones = batch
    log_dict = {}

    # Base IQL의 V, Q 업데이트
    with torch.no_grad():
        next_v = trainer.vf(next_observations)
    adv = trainer._update_v(observations, actions, log_dict)
    rewards = rewards.squeeze(dim=-1)
    dones = dones.squeeze(dim=-1)
    trainer._update_q(next_v, observations, actions, rewards, dones, log_dict)

    # Multi-actor 업데이트
    exp_adv = torch.exp(trainer.beta * adv.detach()).clamp(max=100.0)

    for i in range(len(actors)):
        actor_i = actors[i]

        # Base loss 계산 함수 (클로저로 i와 exp_adv 캡처)
        def base_loss_fn(actor):
            mean_i, _ = actor.get_mean_std(observations)
            if i == 0:
                # Actor0: IQL BC loss
                bc_losses = torch.sum((mean_i - actions) ** 2, dim=1)
                return torch.mean(exp_adv * bc_losses)
            else:
                # Actor1+: Q 기반 loss
                Q_i = trainer.qf(observations, mean_i)
                return -Q_i.mean()

        # W2 distance 계산 (ActorConfig 사용)
        actor_i_config = ActorConfig.from_actor(actor_i)
        ref_actor_config = ActorConfig.from_actor(actors[i - 1]) if i > 0 else None
        w2_weight_i = w2_weights[i - 1] if i > 0 else 0.0
        
        actor_loss_i, w2_i = _compute_actor_loss_with_w2(
            base_loss_fn=base_loss_fn,
            actor_i_config=actor_i_config,
            ref_actor_config=ref_actor_config,
            states=observations,
            w2_weight=w2_weight_i,
            sinkhorn_K=4,  # Gaussian이므로 사용 안 됨
            sinkhorn_blur=0.05,
            sinkhorn_loss=None,
            seed=seed_base + 100 + i,
        )

        opt = actor_optimizers[i]
        opt.zero_grad()
        actor_loss_i.backward()
        opt.step()

        log_dict[f"actor_{i}_loss"] = float(actor_loss_i.item())
        if i > 0 and w2_i is not None:
            log_dict[f"w2_{i}_distance"] = float(w2_i.item())

    # Target update
    for actor, actor_target in zip(actors, actor_targets):
        for p, tp in zip(actor.parameters(), actor_target.parameters()):
            tp.data.copy_(trainer.tau * p.data + (1 - trainer.tau) * tp.data)

    return log_dict


def _train_iql_multi_actor_stochastic(
    trainer: ImplicitQLearning,
    actors: List[nn.Module],
    actor_targets: List[nn.Module],
    actor_optimizers: List[torch.optim.Optimizer],
    actor_is_stochastic: List[bool],
    w2_weights: List[float],
    sinkhorn_K: int,
    sinkhorn_blur: float,
    sinkhorn_backend: str,
    batch: TensorBatch,
    sinkhorn_loss,
    seed_base: int,
) -> Dict[str, float]:
    """IQL trainer에 multi-actor 적용 (Stochastic policy용 - Sinkhorn)"""
    trainer.total_it += 1
    observations, actions, rewards, next_observations, dones = batch
    log_dict = {}

    # Base IQL의 V, Q 업데이트
    with torch.no_grad():
        next_v = trainer.vf(next_observations)
    adv = trainer._update_v(observations, actions, log_dict)
    rewards = rewards.squeeze(dim=-1)
    dones = dones.squeeze(dim=-1)
    trainer._update_q(next_v, observations, actions, rewards, dones, log_dict)

    # Multi-actor 업데이트
    exp_adv = torch.exp(trainer.beta * adv.detach()).clamp(max=100.0)

    for i in range(len(actors)):
        actor_i = actors[i]

        # Base loss 계산 함수
        def base_loss_fn(actor):
            if hasattr(actor, 'forward') and not actor_is_stochastic[i]:
                pi_i = actor.forward(observations)
            else:
                pi_i = actor.sample_actions(observations, K=1, seed=seed_base + i)[:, 0, :]
            
            if i == 0:
                # Actor0: IQL BC loss
                bc_losses = torch.sum((pi_i - actions) ** 2, dim=1)
                return torch.mean(exp_adv * bc_losses)
            else:
                # Actor1+: Q 기반 loss
                Q_i = trainer.qf(observations, pi_i)
                return -Q_i.mean()

        # W2 distance 계산 (ActorConfig 사용)
        actor_i_config = ActorConfig.from_actor(actor_i)
        ref_actor_config = ActorConfig.from_actor(actors[i - 1]) if i > 0 else None
        w2_weight_i = w2_weights[i - 1] if i > 0 else 0.0
        
        actor_loss_i, w2_i = _compute_actor_loss_with_w2(
            base_loss_fn=base_loss_fn,
            actor_i_config=actor_i_config,
            ref_actor_config=ref_actor_config,
            states=observations,
            w2_weight=w2_weight_i,
            sinkhorn_K=sinkhorn_K,
            sinkhorn_blur=sinkhorn_blur,
            sinkhorn_loss=sinkhorn_loss,
            seed=seed_base + 100 + i,
        )

        opt = actor_optimizers[i]
        opt.zero_grad()
        actor_loss_i.backward()
        opt.step()

        log_dict[f"actor_{i}_loss"] = float(actor_loss_i.item())
        if i > 0 and w2_i is not None:
            log_dict[f"w2_{i}_distance"] = float(w2_i.item())

    # Target update
    for actor, actor_target in zip(actors, actor_targets):
        for p, tp in zip(actor.parameters(), actor_target.parameters()):
            tp.data.copy_(trainer.tau * p.data + (1 - trainer.tau) * tp.data)

    return log_dict


def _train_cql_multi_actor(
    trainer: ContinuousCQL,
    cql_actor: nn.Module,  # CQL의 원래 TanhGaussianPolicy (Q loss 계산용)
    actors: List[nn.Module],  # POGO multi-actors
    actor_targets: List[nn.Module],
    actor_optimizers: List[torch.optim.Optimizer],
    actor_is_stochastic: List[bool],
    w2_weights: List[float],
    sinkhorn_K: int,
    sinkhorn_blur: float,
    sinkhorn_backend: str,
    batch: TensorBatch,
    sinkhorn_loss,
    seed_base: int,
) -> Dict[str, float]:
    """CQL trainer에 multi-actor 적용 - CQL의 원래 Q loss 그대로 사용"""
    trainer.total_it += 1
    observations, actions, rewards, next_observations, dones = batch
    log_dict = {}

    # Base CQL의 Q loss 계산 (CQL 구조 그대로 - 원래 _q_loss 메서드 사용)
    # CQL의 _q_loss는 TanhGaussianPolicy의 repeat 파라미터가 필요하므로,
    # trainer.actor는 원래 TanhGaussianPolicy를 사용
    new_actions, log_pi = cql_actor(observations)
    alpha, alpha_loss = trainer._alpha_and_alpha_loss(observations, log_pi)
    
    # CQL의 원래 _q_loss 메서드 호출 (그대로 사용)
    qf_loss, alpha_prime, alpha_prime_loss = trainer._q_loss(
        observations, actions, next_observations, rewards, dones, alpha, log_dict
    )
    
    # Q 업데이트
    trainer.critic_1_optimizer.zero_grad()
    trainer.critic_2_optimizer.zero_grad()
    qf_loss.backward(retain_graph=True)
    trainer.critic_1_optimizer.step()
    trainer.critic_2_optimizer.step()
    
    # Alpha 업데이트
    if trainer.use_automatic_entropy_tuning:
        trainer.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        trainer.alpha_optimizer.step()

    # Multi-actor 업데이트 (POGO policies 사용)
    for i in range(len(actors)):
        actor_i = actors[i]
        
        # POGO policies 사용
        new_actions_i = actor_i.deterministic_actions(observations)
        log_pi_i = torch.zeros(observations.size(0), device=observations.device)
        alpha_i, _ = trainer._alpha_and_alpha_loss(observations, log_pi_i)
        
        if i == 0:
            # Actor0: CQL policy loss만 사용 (W2 penalty 없음)
            # CQL policy loss: alpha * log_pi - Q (BC steps 후)
            if trainer.total_it <= trainer.bc_steps:
                # BC 단계: log_prob 대신 L2 사용
                bc_loss = ((new_actions_i - actions) ** 2).sum(dim=1).mean()
                actor_loss_i = (alpha_i * log_pi_i.mean() - bc_loss).mean()
            else:
                q_new = torch.min(
                    trainer.critic_1(observations, new_actions_i),
                    trainer.critic_2(observations, new_actions_i)
                )
                actor_loss_i = (alpha_i * log_pi_i.mean() - q_new).mean()
        else:
            # Actor1+: Sinkhorn to previous actor
            ref_actor = actors[i - 1]
            w2_weight_i = w2_weights[i - 1]  # w2_weights는 Actor1부터 시작
            is_i_stoch = actor_is_stochastic[i]
            is_ref_stoch = actor_is_stochastic[i - 1]
            
            if is_i_stoch and is_ref_stoch:
                w2_i = _per_state_sinkhorn(
                    actor_i, ref_actor, observations,
                    K=sinkhorn_K, blur=sinkhorn_blur,
                    sinkhorn_loss=sinkhorn_loss,
                    seed=seed_base + 100 + i,
                )
            else:
                with torch.no_grad():
                    ref_action = ref_actor.deterministic_actions(observations)
                w2_i = ((new_actions_i - ref_action) ** 2).mean()
            
            # CQL policy loss + Sinkhorn
            if trainer.total_it <= trainer.bc_steps:
                bc_loss = ((new_actions_i - actions) ** 2).sum(dim=1).mean()
                policy_loss_base = (alpha_i * log_pi_i.mean() - bc_loss).mean()
            else:
                q_new = torch.min(
                    trainer.critic_1(observations, new_actions_i),
                    trainer.critic_2(observations, new_actions_i)
                )
                policy_loss_base = (alpha_i * log_pi_i.mean() - q_new).mean()
            actor_loss_i = policy_loss_base + w2_weight_i * w2_i
        
        opt = actor_optimizers[i]
        opt.zero_grad()
        actor_loss_i.backward()
        opt.step()
        
        log_dict[f"actor_{i}_loss"] = float(actor_loss_i.item())
        if i > 0:
            log_dict[f"w2_{i}_distance"] = float(w2_i.item())
    
    # Target update
    if trainer.total_it % trainer.target_update_period == 0:
        trainer.update_target_network(trainer.soft_target_update_rate)
        for actor, actor_target in zip(actors, actor_targets):
            for p, tp in zip(actor.parameters(), actor_target.parameters()):
                tp.data.copy_(trainer.soft_target_update_rate * p.data + (1 - trainer.soft_target_update_rate) * tp.data)
    
    return log_dict


def _train_awac_multi_actor(
    trainer: AdvantageWeightedActorCritic,
    actors: List[nn.Module],
    actor_targets: List[nn.Module],
    actor_optimizers: List[torch.optim.Optimizer],
    actor_is_stochastic: List[bool],
    w2_weights: List[float],
    sinkhorn_K: int,
    sinkhorn_blur: float,
    sinkhorn_backend: str,
    batch: TensorBatch,
    sinkhorn_loss,
    seed_base: int,
) -> Dict[str, float]:
    """AWAC trainer에 multi-actor 적용"""
    if not hasattr(trainer, "total_it"):
        trainer.total_it = 0
    trainer.total_it += 1
    states, actions, rewards, next_states, dones = batch
    log_dict = {}

    # Base AWAC의 Critic 업데이트 (AWAC 구조 그대로)
    trainer._actor = actors[0]  # 임시로 첫 번째 actor 연결
    critic_loss = trainer._update_critic(states, actions, rewards, dones, next_states)
    log_dict["critic_loss"] = critic_loss

    # Multi-actor 업데이트
    for i in range(len(actors)):
        actor_i = actors[i]
        
        trainer._actor = actor_i  # 현재 actor로 교체
        
        if i == 0:
            # Actor0: AWAC loss만 사용 (W2 penalty 없음)
            # AWAC의 actor loss 계산
            with torch.no_grad():
                pi_action = actor_i.deterministic_actions(states)
                v = torch.min(
                    trainer._critic_1(states, pi_action),
                    trainer._critic_2(states, pi_action)
                )
                q = torch.min(
                    trainer._critic_1(states, actions),
                    trainer._critic_2(states, actions)
                )
                adv = q - v
                weights = torch.clamp_max(
                    torch.exp(adv / trainer._awac_lambda), trainer._exp_adv_max
                )
            
            # POGO policies는 log_prob 대신 L2 사용
            # Gradient가 필요하므로 forward() 직접 사용
            if hasattr(actor_i, 'forward') and not actor_is_stochastic[i]:
                pi_i = actor_i.forward(states)
            else:
                # StochasticMLP: sample_actions 사용
                pi_i = actor_i.sample_actions(states, K=1, seed=seed_base + i)[:, 0, :]
            bc_losses = torch.sum((pi_i - actions) ** 2, dim=1)
            actor_loss_i = torch.mean(weights * bc_losses)
        else:
            # Actor1+: Sinkhorn to previous actor
            ref_actor = actors[i - 1]
            w2_weight_i = w2_weights[i - 1]  # w2_weights는 Actor1부터 시작
            is_i_stoch = actor_is_stochastic[i]
            is_ref_stoch = actor_is_stochastic[i - 1]
            
            if is_i_stoch and is_ref_stoch:
                w2_i = _per_state_sinkhorn(
                    actor_i, ref_actor, states,
                    K=sinkhorn_K, blur=sinkhorn_blur,
                    sinkhorn_loss=sinkhorn_loss,
                    seed=seed_base + 100 + i,
                )
            else:
                with torch.no_grad():
                    if hasattr(ref_actor, 'forward') and not is_ref_stoch:
                        ref_action = ref_actor.forward(states)
                    else:
                        ref_action = ref_actor.deterministic_actions(states)
                # Gradient가 필요한 경우 forward() 직접 사용
                if hasattr(actor_i, 'forward') and not is_i_stoch:
                    pi_i = actor_i.forward(states)
                else:
                    pi_i = actor_i.deterministic_actions(states)
                w2_i = ((pi_i - ref_action) ** 2).mean()
            
            # Q 기반 loss (gradient 필요)
            if hasattr(actor_i, 'forward') and not is_i_stoch:
                pi_i = actor_i.forward(states)
            else:
                # StochasticMLP: sample_actions 사용
                pi_i = actor_i.sample_actions(states, K=1, seed=seed_base + i)[:, 0, :]
            Q_i = torch.min(
                trainer._critic_1(states, pi_i),
                trainer._critic_2(states, pi_i)
            )
            actor_loss_i = -Q_i.mean() + w2_weight_i * w2_i
        
        opt = actor_optimizers[i]
        opt.zero_grad()
        actor_loss_i.backward()
        opt.step()
        
        log_dict[f"actor_{i}_loss"] = float(actor_loss_i.item())
        if i > 0:
            log_dict[f"w2_{i}_distance"] = float(w2_i.item())
    
    # Target update
    soft_update(trainer._target_critic_1, trainer._critic_1, trainer._tau)
    soft_update(trainer._target_critic_2, trainer._critic_2, trainer._tau)
    for actor, actor_target in zip(actors, actor_targets):
        for p, tp in zip(actor.parameters(), actor_target.parameters()):
            tp.data.copy_(trainer._tau * p.data + (1 - trainer._tau) * tp.data)
    
    # 첫 번째 actor를 다시 연결
    trainer._actor = actors[0]
    
    return log_dict


def _train_sac_n_multi_actor(
    trainer: SACN,
    actors: List[nn.Module],
    actor_targets: List[nn.Module],
    actor_optimizers: List[torch.optim.Optimizer],
    actor_is_stochastic: List[bool],
    w2_weights: List[float],
    sinkhorn_K: int,
    sinkhorn_blur: float,
    sinkhorn_backend: str,
    batch: TensorBatch,
    sinkhorn_loss,
    seed_base: int,
) -> Dict[str, float]:
    """SAC-N trainer에 multi-actor 적용 - SAC-N의 원래 구조 그대로 사용"""
    if not hasattr(trainer, "total_it"):
        trainer.total_it = 0
    trainer.total_it += 1
    state, action, reward, next_state, done = [arr.to(trainer.device) for arr in batch]
    log_dict = {}

    # Base SAC-N의 Alpha, Critic 업데이트 (SAC-N 구조 그대로)
    trainer.actor = actors[0]  # 임시로 첫 번째 actor 연결
    
    # Alpha update
    alpha_loss = trainer._alpha_loss(state)
    trainer.alpha_optimizer.zero_grad()
    alpha_loss.backward()
    trainer.alpha_optimizer.step()
    trainer.alpha = trainer.log_alpha.exp().detach()
    
    # Critic update (SAC-N의 원래 _critic_loss 사용)
    critic_loss = trainer._critic_loss(state, action, reward, next_state, done)
    trainer.critic_optimizer.zero_grad()
    critic_loss.backward()
    trainer.critic_optimizer.step()
    
    log_dict["alpha_loss"] = float(alpha_loss.item())
    log_dict["critic_loss"] = float(critic_loss.item())
    log_dict["alpha"] = float(trainer.alpha.item())

    # Multi-actor 업데이트 (POGO policies 사용)
    for i in range(len(actors)):
        actor_i = actors[i]
        
        trainer.actor = actor_i  # 현재 actor로 교체
        
        if i == 0:
            # Actor0: SAC-N loss만 사용 (W2 penalty 없음)
            # SAC-N의 actor loss: alpha * log_pi - Q_min
            # POGO policies는 log_prob를 직접 제공하지 않으므로 근사
            pi_i = actor_i.deterministic_actions(state)
            q_value_dist = trainer.critic(state, pi_i)
            q_value_min = q_value_dist.min(0).values
            # log_prob 근사 (작은 값 사용)
            log_pi_i = torch.zeros(state.size(0), device=state.device)
            actor_loss_i = (trainer.alpha * log_pi_i - q_value_min).mean()
        else:
            # Actor1+: Sinkhorn to previous actor
            ref_actor = actors[i - 1]
            w2_weight_i = w2_weights[i - 1]  # w2_weights는 Actor1부터 시작
            is_i_stoch = actor_is_stochastic[i]
            is_ref_stoch = actor_is_stochastic[i - 1]
            
            if is_i_stoch and is_ref_stoch:
                w2_i = _per_state_sinkhorn(
                    actor_i, ref_actor, state,
                    K=sinkhorn_K, blur=sinkhorn_blur,
                    sinkhorn_loss=sinkhorn_loss,
                    seed=seed_base + 100 + i,
                )
            else:
                with torch.no_grad():
                    ref_action = ref_actor.deterministic_actions(state)
                pi_i = actor_i.deterministic_actions(state)
                w2_i = ((pi_i - ref_action) ** 2).mean()
            
            # SAC-N loss + Sinkhorn
            pi_i = actor_i.deterministic_actions(state)
            q_value_dist = trainer.critic(state, pi_i)
            q_value_min = q_value_dist.min(0).values
            log_pi_i = torch.zeros(state.size(0), device=state.device)
            actor_loss_base = (trainer.alpha * log_pi_i - q_value_min).mean()
            actor_loss_i = actor_loss_base + w2_weight_i * w2_i
        
        opt = actor_optimizers[i]
        opt.zero_grad()
        actor_loss_i.backward()
        opt.step()
        
        log_dict[f"actor_{i}_loss"] = float(actor_loss_i.item())
        if i > 0:
            log_dict[f"w2_{i}_distance"] = float(w2_i.item())
    
    # Target update
    soft_update(trainer.target_critic, trainer.critic, tau=trainer.tau)
    for actor, actor_target in zip(actors, actor_targets):
        for p, tp in zip(actor.parameters(), actor_target.parameters()):
            tp.data.copy_(trainer.tau * p.data + (1 - trainer.tau) * tp.data)
    
    # 첫 번째 actor를 다시 연결
    trainer.actor = actors[0]
    
    return log_dict


def _train_edac_multi_actor(
    trainer: EDAC,
    actors: List[nn.Module],
    actor_targets: List[nn.Module],
    actor_optimizers: List[torch.optim.Optimizer],
    actor_is_stochastic: List[bool],
    w2_weights: List[float],
    sinkhorn_K: int,
    sinkhorn_blur: float,
    sinkhorn_backend: str,
    batch: TensorBatch,
    sinkhorn_loss,
    seed_base: int,
) -> Dict[str, float]:
    """EDAC trainer에 multi-actor 적용 - EDAC의 원래 구조 그대로 사용"""
    if not hasattr(trainer, "total_it"):
        trainer.total_it = 0
    trainer.total_it += 1
    state, action, reward, next_state, done = [arr.to(trainer.device) for arr in batch]
    log_dict = {}

    # Base EDAC의 Alpha, Critic 업데이트 (EDAC 구조 그대로)
    trainer.actor = actors[0]  # 임시로 첫 번째 actor 연결
    
    # Alpha update
    alpha_loss = trainer._alpha_loss(state)
    trainer.alpha_optimizer.zero_grad()
    alpha_loss.backward()
    trainer.alpha_optimizer.step()
    trainer.alpha = trainer.log_alpha.exp().detach()
    
    # Critic update (EDAC의 원래 _critic_loss 사용 - diversity loss 포함)
    critic_loss = trainer._critic_loss(state, action, reward, next_state, done)
    trainer.critic_optimizer.zero_grad()
    critic_loss.backward()
    trainer.critic_optimizer.step()
    
    log_dict["alpha_loss"] = float(alpha_loss.item())
    log_dict["critic_loss"] = float(critic_loss.item())
    log_dict["alpha"] = float(trainer.alpha.item())

    # Multi-actor 업데이트 (POGO policies 사용)
    for i in range(len(actors)):
        actor_i = actors[i]
        
        trainer.actor = actor_i  # 현재 actor로 교체
        
        if i == 0:
            # Actor0: EDAC loss만 사용 (W2 penalty 없음)
            # EDAC의 actor loss: alpha * log_pi - Q_min
            pi_i = actor_i.deterministic_actions(state)
            q_value_dist = trainer.critic(state, pi_i)
            q_value_min = q_value_dist.min(0).values
            log_pi_i = torch.zeros(state.size(0), device=state.device)
            actor_loss_i = (trainer.alpha * log_pi_i - q_value_min).mean()
        else:
            # Actor1+: Sinkhorn to previous actor
            ref_actor = actors[i - 1]
            w2_weight_i = w2_weights[i - 1]  # w2_weights는 Actor1부터 시작
            is_i_stoch = actor_is_stochastic[i]
            is_ref_stoch = actor_is_stochastic[i - 1]
            
            if is_i_stoch and is_ref_stoch:
                w2_i = _per_state_sinkhorn(
                    actor_i, ref_actor, state,
                    K=sinkhorn_K, blur=sinkhorn_blur,
                    sinkhorn_loss=sinkhorn_loss,
                    seed=seed_base + 100 + i,
                )
            else:
                with torch.no_grad():
                    ref_action = ref_actor.deterministic_actions(state)
                pi_i = actor_i.deterministic_actions(state)
                w2_i = ((pi_i - ref_action) ** 2).mean()
            
            # EDAC loss + Sinkhorn
            pi_i = actor_i.deterministic_actions(state)
            q_value_dist = trainer.critic(state, pi_i)
            q_value_min = q_value_dist.min(0).values
            log_pi_i = torch.zeros(state.size(0), device=state.device)
            actor_loss_base = (trainer.alpha * log_pi_i - q_value_min).mean()
            actor_loss_i = actor_loss_base + w2_weight_i * w2_i
        
        opt = actor_optimizers[i]
        opt.zero_grad()
        actor_loss_i.backward()
        opt.step()
        
        log_dict[f"actor_{i}_loss"] = float(actor_loss_i.item())
        if i > 0:
            log_dict[f"w2_{i}_distance"] = float(w2_i.item())
    
    # Target update
    soft_update(trainer.target_critic, trainer.critic, tau=trainer.tau)
    for actor, actor_target in zip(actors, actor_targets):
        for p, tp in zip(actor.parameters(), actor_target.parameters()):
            tp.data.copy_(trainer.tau * p.data + (1 - trainer.tau) * tp.data)
    
    # 첫 번째 actor를 다시 연결
    trainer.actor = actors[0]
    
    return log_dict


def _train_td3_bc_multi_actor(
    trainer: TD3_BC,
    actors: List[nn.Module],
    actor_targets: List[nn.Module],
    actor_optimizers: List[torch.optim.Optimizer],
    actor_is_stochastic: List[bool],
    w2_weights: List[float],
    sinkhorn_K: int,
    sinkhorn_blur: float,
    sinkhorn_backend: str,
    batch: TensorBatch,
    sinkhorn_loss,
    seed_base: int,
) -> Dict[str, float]:
    """TD3_BC trainer에 multi-actor 적용"""
    trainer.total_it += 1  # Base trainer의 total_it 업데이트
    state, action, reward, next_state, done = batch
    not_done = 1.0 - done
    log_dict = {}

    # Critic 업데이트 (base trainer 방식)
    with torch.no_grad():
        noise = (torch.randn_like(action) * trainer.policy_noise).clamp(
            -trainer.noise_clip, trainer.noise_clip
        )
        next_action = (actor_targets[0](next_state) + noise).clamp(
            -trainer.max_action, trainer.max_action
        )
        target_q1 = trainer.critic_1_target(next_state, next_action)
        target_q2 = trainer.critic_2_target(next_state, next_action)
        target_q = reward + not_done * trainer.discount * torch.min(target_q1, target_q2)

    current_q1 = trainer.critic_1(state, action)
    current_q2 = trainer.critic_2(state, action)
    critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
    trainer.critic_1_optimizer.zero_grad()
    trainer.critic_2_optimizer.zero_grad()
    critic_loss.backward()
    trainer.critic_1_optimizer.step()
    trainer.critic_2_optimizer.step()
    log_dict["critic_loss"] = float(critic_loss.item())

    # Multi-actor 업데이트 (policy_freq마다)
    if trainer.total_it % trainer.policy_freq == 0:
        for i in range(len(actors)):
            actor_i = actors[i]

            # POGO policies 사용
            pi_i = actor_i.deterministic_actions(state)
            q = trainer.critic_1(state, pi_i)
            lmbda = trainer.alpha / q.abs().mean().detach()

            if i == 0:
                # Actor0: TD3_BC loss만 사용 (W2 penalty 없음)
                actor_loss_i = -lmbda * q.mean() + trainer.alpha * ((pi_i - action) ** 2).mean()
            else:
                # Actor1+: Sinkhorn to previous actor
                ref_actor = actors[i - 1]
                w2_weight_i = w2_weights[i - 1]  # w2_weights는 Actor1부터 시작
                is_i_stoch = actor_is_stochastic[i]
                is_ref_stoch = actor_is_stochastic[i - 1]

                if is_i_stoch and is_ref_stoch:
                    w2_i = _per_state_sinkhorn(
                        actor_i, ref_actor, state,
                        K=sinkhorn_K, blur=sinkhorn_blur,
                        sinkhorn_loss=sinkhorn_loss,
                        seed=seed_base + 100 + i,
                    )
                else:
                    with torch.no_grad():
                        ref_action = ref_actor.deterministic_actions(state)
                    w2_i = ((pi_i - ref_action) ** 2).mean()

                actor_loss_i = -lmbda * q.mean() + w2_weight_i * w2_i

            opt = actor_optimizers[i]
            opt.zero_grad()
            actor_loss_i.backward()
            opt.step()

            log_dict[f"actor_{i}_loss"] = float(actor_loss_i.item())
            if i > 0:
                log_dict[f"w2_{i}_distance"] = float(w2_i.item())

        # Target update
        for actor, actor_target in zip(actors, actor_targets):
            for p, tp in zip(actor.parameters(), actor_target.parameters()):
                tp.data.copy_(trainer.tau * p.data + (1 - trainer.tau) * tp.data)
        for p, tp in zip(trainer.critic_1.parameters(), trainer.critic_1_target.parameters()):
            tp.data.copy_(trainer.tau * p.data + (1 - trainer.tau) * tp.data)
        for p, tp in zip(trainer.critic_2.parameters(), trainer.critic_2_target.parameters()):
            tp.data.copy_(trainer.tau * p.data + (1 - trainer.tau) * tp.data)

    return log_dict


# 나머지 함수들 (compute_mean_std, normalize_states, wrap_env, modify_reward, ReplayBuffer, set_seed, wandb_init, eval_actor)은
# 기존 pogo_multi.py에서 가져오거나 각 알고리즘 파일에서 import

@pyrallis.wrap()
def train(config: TrainConfig):
    """통합 학습 함수: 각 알고리즘의 구조를 그대로 사용하되 Actor만 multi-actor"""
    from .iql import (
        compute_mean_std,
        normalize_states,
        wrap_env,
        modify_reward,
        ReplayBuffer,
        set_seed,
        # wandb_init removed,
        eval_actor as eval_actor_base,
    )

    import sys
    sys.stderr.write("=" * 60 + "\n")
    sys.stderr.write("POGO Multi-Actor Training Started\n")
    sys.stderr.write(f"Algorithm: {config.algorithm}\n")
    sys.stderr.write(f"Environment: {config.env}\n")
    sys.stderr.write("=" * 60 + "\n")
    sys.stderr.flush()
    
    print(f"[DEBUG] Creating environment: {config.env}", flush=True)
    env = gym.make(config.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    print(f"[DEBUG] State dim: {state_dim}, Action dim: {action_dim}, Max action: {max_action}", flush=True)

    print(f"[DEBUG] Loading dataset...", flush=True)
    dataset = d4rl.qlearning_dataset(env)
    print(f"[DEBUG] Dataset loaded: {len(dataset['observations'])} transitions", flush=True)
    if config.normalize_reward:
        modify_reward(dataset, config.env)

    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = np.zeros(state_dim), np.ones(state_dim)
    dataset["observations"] = normalize_states(dataset["observations"], state_mean, state_std)
    dataset["next_observations"] = normalize_states(dataset["next_observations"], state_mean, state_std)
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)

    replay_buffer = ReplayBuffer(state_dim, action_dim, config.buffer_size, config.device)
    replay_buffer.load_d4rl_dataset(dataset)

    if config.checkpoints_path:
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    set_seed(config.seed, env)

    # Multi-actor 생성
    actors, actor_targets, actor_optimizers, actor_is_stochastic, actor_is_gaussian = _create_multi_actors(
        state_dim, action_dim, max_action, config.num_actors,
        config.actor_configs, config.algorithm, config.device, config.actor_lr
    )

    # Base trainer 생성 (각 알고리즘별)
    sinkhorn_loss = SamplesLoss(loss="sinkhorn", p=2, blur=config.sinkhorn_blur, backend=config.sinkhorn_backend)

    if config.algorithm == "iql":
        q_network = TwinQ(state_dim, action_dim).to(config.device)
        v_network = ValueFunction(state_dim).to(config.device)
        v_optimizer = torch.optim.Adam(v_network.parameters(), lr=config.vf_lr)
        q_optimizer = torch.optim.Adam(q_network.parameters(), lr=config.qf_lr)

        trainer = ImplicitQLearning(
            max_action=max_action,
            actor=actors[0],  # 첫 번째 actor를 base trainer에 연결
            actor_optimizer=actor_optimizers[0],
            q_network=q_network,
            q_optimizer=q_optimizer,
            v_network=v_network,
            v_optimizer=v_optimizer,
            iql_tau=config.iql_tau,
            beta=config.beta,
            max_steps=config.max_timesteps,
            discount=config.discount,
            tau=config.tau,
            device=config.device,
        )

        # Gaussian과 Stochastic 구분
        use_gaussian = any(actor_is_gaussian)
        
        def train_fn(batch):
            seed_base = (config.seed if config.seed else 0) * 1000000 + trainer.total_it * 1000
            if use_gaussian:
                return _train_iql_multi_actor_gaussian(
                    trainer, actors, actor_targets, actor_optimizers, actor_is_gaussian,
                    config.w2_weights, batch, seed_base
                )
            else:
                return _train_iql_multi_actor_stochastic(
                    trainer, actors, actor_targets, actor_optimizers, actor_is_stochastic,
                    config.w2_weights, config.sinkhorn_K, config.sinkhorn_blur,
                    config.sinkhorn_backend, batch, sinkhorn_loss, seed_base
                )

    elif config.algorithm == "td3_bc":
        critic_1 = Critic(state_dim, action_dim).to(config.device)
        critic_2 = Critic(state_dim, action_dim).to(config.device)
        critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=config.qf_lr)
        critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=config.qf_lr)

        trainer = TD3_BC(
            max_action=max_action,
            actor=actors[0],
            actor_optimizer=actor_optimizers[0],
            critic_1=critic_1,
            critic_1_optimizer=critic_1_optimizer,
            critic_2=critic_2,
            critic_2_optimizer=critic_2_optimizer,
            discount=config.discount,
            tau=config.tau,
            policy_noise=config.policy_noise * max_action,
            noise_clip=config.noise_clip * max_action,
            policy_freq=config.policy_freq,
            alpha=config.alpha,
            device=config.device,
        )

        def train_fn(batch):
            seed_base = (config.seed if config.seed else 0) * 1000000 + trainer.total_it * 1000
            return _train_td3_bc_multi_actor(
                trainer, actors, actor_targets, actor_optimizers, actor_is_stochastic,
                config.w2_weights, config.sinkhorn_K, config.sinkhorn_blur,
                config.sinkhorn_backend, batch, sinkhorn_loss, seed_base
            )

    elif config.algorithm == "cql":
        from .cql import Scalar, TanhGaussianPolicy
        critic_1 = FullyConnectedQFunction(state_dim, action_dim).to(config.device)
        critic_2 = FullyConnectedQFunction(state_dim, action_dim).to(config.device)
        critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=config.qf_lr)
        critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=config.qf_lr)

        target_entropy_val = config.target_entropy
        if target_entropy_val is None:
            target_entropy_val = -np.prod(env.action_space.shape).item()

        # CQL의 Q loss는 TanhGaussianPolicy의 repeat 파라미터가 필요하므로,
        # CQL trainer에는 원래 TanhGaussianPolicy 사용
        cql_actor = TanhGaussianPolicy(
            state_dim, action_dim, max_action,
            log_std_multiplier=1.0,
            log_std_offset=-1.0,
            orthogonal_init=False,
        ).to(config.device)
        cql_actor_optimizer = torch.optim.Adam(cql_actor.parameters(), lr=config.actor_lr)

        trainer = ContinuousCQL(
            critic_1=critic_1,
            critic_1_optimizer=critic_1_optimizer,
            critic_2=critic_2,
            critic_2_optimizer=critic_2_optimizer,
            actor=cql_actor,  # CQL의 원래 actor 사용 (Q loss 계산용)
            actor_optimizer=cql_actor_optimizer,
            target_entropy=target_entropy_val,
            discount=config.discount,
            alpha_multiplier=1.0,
            use_automatic_entropy_tuning=True,
            backup_entropy=False,
            policy_lr=config.actor_lr,
            qf_lr=config.qf_lr,
            soft_target_update_rate=config.tau,
            bc_steps=100000,
            target_update_period=1,
            cql_n_actions=config.cql_n_actions,
            cql_importance_sample=True,
            cql_lagrange=False,
            cql_target_action_gap=-1.0,
            cql_temp=1.0,
            cql_alpha=config.cql_alpha,
            cql_max_target_backup=False,
            cql_clip_diff_min=-np.inf,
            cql_clip_diff_max=np.inf,
            device=config.device,
        )

        def train_fn(batch):
            seed_base = (config.seed if config.seed else 0) * 1000000 + trainer.total_it * 1000
            return _train_cql_multi_actor(
                trainer, cql_actor, actors, actor_targets, actor_optimizers, actor_is_stochastic,
                config.w2_weights, config.sinkhorn_K, config.sinkhorn_blur,
                config.sinkhorn_backend, batch, sinkhorn_loss, seed_base
            )

    elif config.algorithm == "awac":
        from .awac import Critic as AWACCritic
        critic_1 = AWACCritic(state_dim, action_dim, hidden_dim=256).to(config.device)
        critic_2 = AWACCritic(state_dim, action_dim, hidden_dim=256).to(config.device)
        critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=config.actor_lr)
        critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=config.actor_lr)

        trainer = AdvantageWeightedActorCritic(
            actor=actors[0],
            actor_optimizer=actor_optimizers[0],
            critic_1=critic_1,
            critic_1_optimizer=critic_1_optimizer,
            critic_2=critic_2,
            critic_2_optimizer=critic_2_optimizer,
            gamma=config.discount,
            tau=config.tau,
            awac_lambda=getattr(config, "awac_lambda", 1.0),
            exp_adv_max=getattr(config, "exp_adv_max", 100.0),
        )

        def train_fn(batch):
            seed_base = (config.seed if config.seed else 0) * 1000000 + getattr(trainer, "total_it", 0) * 1000
            if not hasattr(trainer, "total_it"):
                trainer.total_it = 0
            return _train_awac_multi_actor(
                trainer, actors, actor_targets, actor_optimizers, actor_is_stochastic,
                config.w2_weights, config.sinkhorn_K, config.sinkhorn_blur,
                config.sinkhorn_backend, batch, sinkhorn_loss, seed_base
            )

    elif config.algorithm == "sac_n":
        # SAC-N: VectorizedCritic 사용 (여러 critic ensemble)
        critic = VectorizedCritic(state_dim, action_dim, getattr(config, "hidden_dim", 256), config.num_critics).to(config.device)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.actor_lr)

        trainer = SACN(
            actor=actors[0],  # 첫 번째 actor를 base trainer에 연결
            actor_optimizer=actor_optimizers[0],
            critic=critic,
            critic_optimizer=critic_optimizer,
            gamma=config.discount,
            tau=config.tau,
            alpha_learning_rate=config.alpha_learning_rate,
            device=config.device,
        )

        def train_fn(batch):
            seed_base = (config.seed if config.seed else 0) * 1000000 + getattr(trainer, "total_it", 0) * 1000
            if not hasattr(trainer, "total_it"):
                trainer.total_it = 0
            return _train_sac_n_multi_actor(
                trainer, actors, actor_targets, actor_optimizers, actor_is_stochastic,
                config.w2_weights, config.sinkhorn_K, config.sinkhorn_blur,
                config.sinkhorn_backend, batch, sinkhorn_loss, seed_base
            )

    elif config.algorithm == "edac":
        # EDAC: VectorizedCritic 사용 + diversity loss
        critic = VectorizedCritic(state_dim, action_dim, getattr(config, "hidden_dim", 256), config.num_critics).to(config.device)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.actor_lr)

        trainer = EDAC(
            actor=actors[0],  # 첫 번째 actor를 base trainer에 연결
            actor_optimizer=actor_optimizers[0],
            critic=critic,
            critic_optimizer=critic_optimizer,
            gamma=config.discount,
            tau=config.tau,
            eta=config.eta,
            alpha_learning_rate=config.alpha_learning_rate,
            device=config.device,
        )

        def train_fn(batch):
            seed_base = (config.seed if config.seed else 0) * 1000000 + getattr(trainer, "total_it", 0) * 1000
            if not hasattr(trainer, "total_it"):
                trainer.total_it = 0
            return _train_edac_multi_actor(
                trainer, actors, actor_targets, actor_optimizers, actor_is_stochastic,
                config.w2_weights, config.sinkhorn_K, config.sinkhorn_blur,
                config.sinkhorn_backend, batch, sinkhorn_loss, seed_base
            )

    else:
        raise ValueError(f"Unsupported algorithm: {config.algorithm}")

    print("---------------------------------------", flush=True)
    print(f"Training POGO Multi-Actor ({config.algorithm.upper()}), Env: {config.env}, Seed: {config.seed}", flush=True)
    print(f"  Actors: {config.num_actors}, W2 weights (Actor1+): {config.w2_weights}", flush=True)
    print("  Actor0: 원래 알고리즘 loss만 사용, Actor1+: Sinkhorn", flush=True)
    print("---------------------------------------", flush=True)

    evaluations = {i: [] for i in range(config.num_actors)}
    all_logs = []  # 모든 로그 저장
    
    for t in range(int(config.max_timesteps)):
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        log_dict = train_fn(batch)
        log_dict["timestep"] = t + 1
        all_logs.append(log_dict)

        if (t + 1) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            for i in range(config.num_actors):
                # 평가는 각 actor마다
                actor_i = actors[i]
                try:
                    env.seed(config.seed + 100 + i)
                    env.action_space.seed(config.seed + 100 + i)
                except Exception:
                    pass
                actor_i.eval()
                episode_rewards = []
                for ep in range(config.n_episodes):
                    state, done = env.reset(), False
                    if isinstance(state, tuple):
                        state = state[0]
                    ep_ret = 0.0
                    while not done:
                        action = actor_i.act(state, config.device)
                        step_out = env.step(action)
                        if len(step_out) == 5:
                            state, reward, terminated, truncated, _ = step_out
                            done = terminated or truncated
                        else:
                            state, reward, done, _ = step_out
                        ep_ret += reward
                    episode_rewards.append(ep_ret)
                actor_i.train()
                scores = np.asarray(episode_rewards)
                norm_score = env.get_normalized_score(scores.mean()) * 100.0
                evaluations[i].append(norm_score)
                log_dict[f"eval_actor_{i}"] = norm_score
                print(f"  Actor {i} eval (norm): {norm_score:.1f}", flush=True)
                
                # 평가 결과를 로그에 추가
                eval_log = {
                    "timestep": t + 1,
                    "actor": i,
                    "eval_score": float(norm_score),
                    "raw_score": float(scores.mean()),
                    "std": float(scores.std()),
                }
                all_logs.append(eval_log)

    # 학습 완료 후 최종 평가
    print("\n" + "=" * 60, flush=True)
    print("Training completed!", flush=True)
    print("=" * 60, flush=True)
    for i in range(config.num_actors):
        if evaluations[i]:
            final_score = evaluations[i][-1]
            best_score = max(evaluations[i])
            print(f"Actor {i}: Final={final_score:.1f}, Best={best_score:.1f}", flush=True)
    
    # 로그 저장: results/{algorithm}/{env}/seed_{seed}/logs/
    import json
    import datetime
    env_name = config.env.replace("-", "_")  # halfcheetah-medium-v2 -> halfcheetah_medium_v2
    log_dir = os.path.join("results", config.algorithm, env_name, f"seed_{config.seed}", "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{config.algorithm}_{env_name}_seed{config.seed}_{timestamp}.json")
    
    # 평가 결과만 정리해서 저장
    eval_summary = {
        "algorithm": config.algorithm,
        "env": config.env,
        "seed": config.seed,
        "num_actors": config.num_actors,
        "w2_weights": config.w2_weights,
        "max_timesteps": config.max_timesteps,
        "evaluations": {str(i): evaluations[i] for i in range(config.num_actors)},
        "final_scores": {str(i): evaluations[i][-1] if evaluations[i] else None for i in range(config.num_actors)},
        "best_scores": {str(i): max(evaluations[i]) if evaluations[i] else None for i in range(config.num_actors)},
    }
    
    with open(log_file, "w") as f:
        json.dump(eval_summary, f, indent=2)
    
    print(f"\n로그 저장 완료: {log_file}", flush=True)
    
    # Checkpoint 저장: results/{algorithm}/{env}/seed_{seed}/checkpoints/
    env_name = config.env.replace("-", "_")
    checkpoint_dir = os.path.join("results", config.algorithm, env_name, f"seed_{config.seed}", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    ckpt = {}
    if hasattr(trainer, "critic_1"):
        ckpt["critic_1"] = trainer.critic_1.state_dict()
        ckpt["critic_2"] = trainer.critic_2.state_dict()
    if hasattr(trainer, "qf"):
        ckpt["qf"] = trainer.qf.state_dict()
        ckpt["vf"] = trainer.vf.state_dict()
    if hasattr(trainer, "critic"):
        ckpt["critic"] = trainer.critic.state_dict()
    for i in range(config.num_actors):
        ckpt[f"actor_{i}"] = actors[i].state_dict()
        if actor_targets[i] is not None:
            ckpt[f"actor_{i}_target"] = actor_targets[i].state_dict()
    
    checkpoint_file = os.path.join(checkpoint_dir, f"model_{timestamp}.pt")
    torch.save(ckpt, checkpoint_file)
    print(f"Checkpoint 저장 완료: {checkpoint_file}", flush=True)

    return trainer, actors


if __name__ == "__main__":
    train()
