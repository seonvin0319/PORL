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
import wandb
from geomloss import SamplesLoss

# 각 알고리즘 import
from .iql import ImplicitQLearning, TwinQ, ValueFunction
from .td3_bc import TD3_BC, Critic
from .networks import (
    DeterministicMLP,
    DeterministicMLP as Actor,
    TanhGaussianMLP as TanhGaussianPolicy,
    BaseActor,
    StochasticMLP,
    GaussianMLP,
    TanhGaussianMLP,
    build_mlp,
)
from .cql import ContinuousCQL, FullyConnectedQFunction, Scalar
from .awac import AdvantageWeightedActorCritic, soft_update
from .sac_n import SACN, VectorizedCritic
from .edac import EDAC
# SAC-N/EDAC Actor는 각 파일에서 TanhGaussianMLP를 사용하도록 통합됨

TensorBatch = List[torch.Tensor]


# PyTorchAlgorithmInterface와 관련 클래스들은 utils_pytorch.py로 이동
from .utils_pytorch import (
    PyTorchAlgorithmInterface,
    ActorConfig,
)

# 각 알고리즘의 인터페이스 구현은 각 알고리즘 파일에서 직접 import (순환 참조 방지)
from .iql import IQLAlgorithm
from .td3_bc import TD3BCAlgorithm
from .cql import CQLAlgorithm
from .awac import AWACAlgorithm
from .sac_n import SACNAlgorithm
from .edac import EDACAlgorithm


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
        """Create a single critic network using build_mlp"""
        layers = build_mlp(
            state_dim + action_dim, hidden_dim, n_hiddens, layernorm=layernorm
        )
        
        # Initialize as in ReBRAC (EDAC style)
        net = nn.Sequential(*layers)
        linear_layers = [layer for layer in net if isinstance(layer, nn.Linear)]
        for i, layer in enumerate(linear_layers):
            if i == 0:
                # First layer: constant bias 0.1
                nn.init.constant_(layer.bias, 0.1)
                nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
            elif i == len(linear_layers) - 1:
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
    wandb_project: str = "PORL"
    wandb_entity: Optional[str] = "seonvin0319"
    wandb_name: Optional[str] = None
    use_wandb: bool = True
    group: str = "POGO-Multi"  # 하위 호환성 유지
    
    # Logging / checkpoint
    log_interval: int = 1  # train log 주기 (step 기준)
    checkpoint_freq: int = int(5e5)  # 500k
    save_train_logs: bool = True
    
    # Final evaluation
    final_eval_runs: int = 5
    final_eval_episodes: int = 10

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
        # wandb 이름: 환경-알고리즘-seed{seed}-w{} (JAX 버전과 동일한 형식)
        # 정수면 소수점 제거, 같은 값이면 하나만 표시
        if len(self.w2_weights) > 0:
            # 모든 값이 같은지 확인
            if len(set(self.w2_weights)) == 1:
                # 같은 값이면 하나만 사용
                w = self.w2_weights[0]
                # 정수면 소수점 제거
                w2_str = str(int(w)) if w == int(w) else str(w)
            else:
                # 다른 값이면 모두 표시하되 정수면 소수점 제거
                w2_str = "_".join([str(int(w)) if w == int(w) else str(w) for w in self.w2_weights])
        else:
            w2_str = "10"
        default_name = f"{self.env}-{self.algorithm}-seed{self.seed}-w{w2_str}"
        if self.wandb_name is None:
            self.wandb_name = default_name
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, default_name)


def _per_state_sinkhorn(
    policy,
    ref,
    states: torch.Tensor,
    K: int = 4,
    blur: float = 0.05,
    sinkhorn_loss=None,
    seed: Optional[int] = None,
    backend: str = "tensorized",
):
    """
    Per-state Sinkhorn distance 계산
    
    주의: SamplesLoss 출력이 상황별로 scalar/tensor일 수 있으므로
    loss.dim() > 0 체크로 안전하게 처리합니다.
    per-state 기대를 정확히 원하면 sample_actions의 반환 shape가
    "point cloud" semantics를 따르는지 확인해야 합니다:
    - sample_actions는 [B, K, action_dim] 형태를 반환해야 합니다.
    - sinkhorn_loss는 [B, K] point cloud 간 거리를 계산합니다.
    """
    if sinkhorn_loss is None:
        sinkhorn_loss = SamplesLoss(loss="sinkhorn", p=2, blur=blur, backend=backend)
    a = policy.sample_actions(states, K=K, seed=None if seed is None else seed + 0)
    with torch.no_grad():
        b_detached = ref.sample_actions(states, K=K, seed=None if seed is None else seed + 10000).detach()
    loss = sinkhorn_loss(a, b_detached)
    # SamplesLoss 출력이 상황별로 scalar/tensor일 수 있으므로 안전하게 처리
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
            # CQL과 동일한 구조를 위해 3개의 hidden layer 사용 (기본값은 2)
            n_hiddens = config.get("n_hiddens", 3) if isinstance(config, dict) else getattr(config, "n_hiddens", 3)
            actor = TanhGaussianMLP(state_dim, action_dim, max_action, hidden_dim=256, n_hiddens=n_hiddens).to(device)
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
    backend: str = "tensorized",
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
            backend=backend,
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
    backend: str = "tensorized",
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
        backend=backend,
    )
    
    return base_loss + w2_weight * w2_distance, w2_distance


# ============================================================================
# 통합된 Multi-Actor 업데이트 함수
# ============================================================================
# ============================================================================
# 통합된 Multi-Actor 업데이트 함수
# ============================================================================

def _update_single_actor_pytorch(
    actor: nn.Module,
    actor_target: nn.Module,
    actor_optimizer: torch.optim.Optimizer,
    actor_config: ActorConfig,
    ref_actor: Optional[nn.Module],
    ref_actor_config: Optional[ActorConfig],
    w2_weight: Optional[float],
    batch: TensorBatch,
    algorithm: PyTorchAlgorithmInterface,
    actor_idx: int,
    actor_is_stochastic: bool,
    sinkhorn_K: int,
    sinkhorn_blur: float,
    sinkhorn_loss,
    seed_base: int,
    tau: float,
    backend: str = "tensorized",
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    단일 actor 업데이트 헬퍼 함수 (JAX 버전의 _update_single_actor와 유사)
    
    Args:
        actor: 현재 actor network
        actor_target: 현재 actor target network
        actor_optimizer: 현재 actor optimizer
        actor_config: 현재 actor config
        ref_actor: 참조 actor network (Actor1+인 경우)
        ref_actor_config: 참조 actor config (Actor1+인 경우)
        w2_weight: W2 distance 가중치 (Actor1+인 경우)
        batch: 배치 데이터
        algorithm: 알고리즘 인터페이스
        actor_idx: Actor 인덱스
        actor_is_stochastic: Actor가 stochastic인지 여부
        sinkhorn_K: Sinkhorn 샘플 수
        sinkhorn_blur: Sinkhorn regularization
        sinkhorn_loss: Sinkhorn loss 객체
        seed_base: 랜덤 시드 베이스
        tau: Target network 업데이트 계수
    
    Returns:
        actor_loss: 계산된 loss
        w2_distance: W2 distance (Actor0인 경우 None)
    """
    # Base actor loss 계산
    base_loss = algorithm.compute_actor_loss(
        algorithm.trainer,
        actor,
        batch,
        actor_idx=actor_idx,
        actor_is_stochastic=actor_is_stochastic,
        seed_base=seed_base,
    )
    
    if actor_idx == 0:
        # Actor0: Base loss만 사용 (W2 penalty 없음)
        actor_loss = base_loss
        w2_distance = None
    else:
        # Actor1+: Base loss + W2 distance
        w2_distance = _compute_w2_distance(
            actor_i_config=actor_config,
            ref_actor_config=ref_actor_config,
            states=batch[0],  # observations
            sinkhorn_K=sinkhorn_K,
            sinkhorn_blur=sinkhorn_blur,
            sinkhorn_loss=sinkhorn_loss,
            seed=seed_base + 100 + actor_idx,
            backend=backend,
        )
        actor_loss = base_loss + w2_weight * w2_distance
    
    # Actor 업데이트
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()
    
    # Target network 업데이트
    for p, tp in zip(actor.parameters(), actor_target.parameters()):
        tp.data.copy_(tau * p.data + (1 - tau) * tp.data)
    
    return actor_loss, w2_distance


def update_multi_actor_pytorch(
    trainer: Any,
    actors: List[nn.Module],
    actor_targets: List[nn.Module],
    actor_optimizers: List[torch.optim.Optimizer],
    actor_is_stochastic: List[bool],
    actor_is_gaussian: List[bool],
    w2_weights: List[float],
    batch: TensorBatch,
    algorithm: PyTorchAlgorithmInterface,
    sinkhorn_K: int,
    sinkhorn_blur: float,
    sinkhorn_backend: str,
    sinkhorn_loss,
    seed_base: int,
    tau: float,
) -> Dict[str, float]:
    """
    통합된 multi-actor 업데이트 함수 (JAX 버전의 update_multi_actor와 유사)
    
    Args:
        trainer: Base trainer 객체
        actors: Actor networks
        actor_targets: Target actor networks
        actor_optimizers: Actor optimizers
        actor_is_stochastic: Actor stochastic 여부 리스트
        actor_is_gaussian: Actor gaussian 여부 리스트
        w2_weights: W2 distance 가중치 리스트 (Actor1+용)
        batch: 배치 데이터
        algorithm: 알고리즘 인터페이스
        sinkhorn_K: Sinkhorn 샘플 수
        sinkhorn_blur: Sinkhorn regularization
        sinkhorn_backend: Sinkhorn backend
        sinkhorn_loss: Sinkhorn loss 객체
        seed_base: 랜덤 시드 베이스
        tau: Target network 업데이트 계수
    
    Returns:
        log_dict: 로깅용 딕셔너리
    """
    log_dict = {}
    
    # Critic/V/Q 업데이트 (AlgorithmInterface 사용)
    log_dict = algorithm.update_critic(
        trainer,
        batch,
        log_dict,
        actors=actors,
        actor_targets=actor_targets,
    )
    
    # Multi-actor 업데이트
    num_actors = len(actors)
    for i in range(num_actors):
        actor_i = actors[i]
        actor_target_i = actor_targets[i]
        actor_optimizer_i = actor_optimizers[i]
        
        # ActorConfig 생성
        actor_config_i = ActorConfig.from_actor(actor_i)
        
        # 참조 actor 설정 (Actor1+인 경우)
        if i == 0:
            ref_actor = None
            ref_actor_config = None
            w2_weight_i = None
        else:
            ref_actor = actors[i - 1]
            ref_actor_config = ActorConfig.from_actor(ref_actor)
            w2_weight_i = w2_weights[i - 1]
        
        # 단일 actor 업데이트
        actor_loss_i, w2_i = _update_single_actor_pytorch(
            actor=actor_i,
            actor_target=actor_target_i,
            actor_optimizer=actor_optimizer_i,
            actor_config=actor_config_i,
            ref_actor=ref_actor,
            ref_actor_config=ref_actor_config,
            w2_weight=w2_weight_i,
            batch=batch,
            algorithm=algorithm,
            actor_idx=i,
            actor_is_stochastic=actor_is_stochastic[i],
            sinkhorn_K=sinkhorn_K,
            sinkhorn_blur=sinkhorn_blur,
            sinkhorn_loss=sinkhorn_loss,
            seed_base=seed_base,
            tau=tau,
            backend=sinkhorn_backend,
        )
        
        log_dict[f"actor_{i}_loss"] = float(actor_loss_i.item())
        if i > 0 and w2_i is not None:
            log_dict[f"w2_{i}_distance"] = float(w2_i.item())
    
    # 추가 target network 업데이트 (알고리즘별)
    algorithm.update_target_networks(
        trainer,
        actors,
        actor_targets,
        tau,
    )
    
    return log_dict


# ============================================================================
# 평가 및 로깅 유틸리티
# ============================================================================

def eval_policy_multi(
    actor: nn.Module,
    env: gym.Env,
    base_seed: int,
    n_episodes: int = 10,
    device: str = "cuda",
    state_mean: Optional[np.ndarray] = None,
    state_std: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, float]:
    """
    Multi-actor 정책 평가: deterministic과 stochastic 모두 평가하고 결과 반환
    
    Args:
        actor: 평가할 actor network
        env: 평가 환경
        base_seed: 기본 시드 (pogogo.py 스타일: config.seed + 100 + i * 100)
        n_episodes: 평가 episode 수
        device: 디바이스
        state_mean: 상태 정규화 평균 (None이면 정규화 안 함)
        state_std: 상태 정규화 표준편차 (None이면 정규화 안 함)
    
    Returns:
        tuple: (det_avg, det_score, stoch_avg, stoch_score)
            - det_avg: deterministic 평균 리워드
            - det_score: deterministic D4RL 정규화 점수
            - stoch_avg: stochastic 평균 리워드
            - stoch_score: stochastic D4RL 정규화 점수
    """
    actor.eval()
    
    # Deterministic 평가
    det_total = 0.0
    for ep in range(n_episodes):
        ep_seed = base_seed + ep
        np.random.seed(ep_seed)
        try:
            env.action_space.seed(ep_seed)
            reset_result = env.reset(seed=ep_seed)
        except (TypeError, AttributeError):
            try:
                reset_result = env.reset(seed=ep_seed)
            except TypeError:
                if hasattr(env, 'seed'):
                    env.seed(ep_seed)
                reset_result = env.reset()
        
        state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        done = False
        ep_ret = 0.0
        while not done:
            # 상태 정규화
            if state_mean is not None and state_std is not None:
                nstate = (np.asarray(state).reshape(1, -1) - state_mean) / state_std
            else:
                nstate = np.asarray(state).reshape(1, -1)
            
            action = actor.act(nstate, device)
            step_out = env.step(action)
            if len(step_out) == 5:
                state, reward, terminated, truncated, _ = step_out
                done = terminated or truncated
            else:
                state, reward, done, _ = step_out
            ep_ret += reward
        det_total += ep_ret
    
    # Stochastic 평가 (재현성을 위해 시드 고정)
    stoch_total = 0.0
    step_count = 0
    for ep in range(n_episodes):
        ep_seed = base_seed + ep
        np.random.seed(ep_seed)
        try:
            env.action_space.seed(ep_seed)
            reset_result = env.reset(seed=ep_seed)
        except (TypeError, AttributeError):
            try:
                reset_result = env.reset(seed=ep_seed)
            except TypeError:
                if hasattr(env, 'seed'):
                    env.seed(ep_seed)
                reset_result = env.reset()
        
        state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        done = False
        ep_ret = 0.0
        while not done:
            # 상태 정규화
            if state_mean is not None and state_std is not None:
                nstate = (np.asarray(state).reshape(1, -1) - state_mean) / state_std
            else:
                nstate = np.asarray(state).reshape(1, -1)
            
            state_tensor = torch.tensor(nstate, device=device, dtype=torch.float32)
            # 재현성을 위해 episode와 step 기반 시드 사용 (pogogo.py 스타일)
            eval_seed = base_seed * 10000 + ep * 1000 + step_count
            actions = actor.sample_actions(state_tensor, K=1, seed=eval_seed)
            action = actions[0].cpu().numpy().flatten()
            step_out = env.step(action)
            if len(step_out) == 5:
                state, reward, terminated, truncated, _ = step_out
                done = terminated or truncated
            else:
                state, reward, done, _ = step_out
            ep_ret += reward
            step_count += 1
        stoch_total += ep_ret
    
    actor.train()
    
    # 결과 계산
    det_avg = det_total / n_episodes
    stoch_avg = stoch_total / n_episodes
    det_score = env.get_normalized_score(det_avg) * 100.0
    stoch_score = env.get_normalized_score(stoch_avg) * 100.0
    
    return det_avg, det_score, stoch_avg, stoch_score


def log_train_wandb(train_out: dict, step: int):
    """학습 메트릭 wandb 로깅 (pogogo.py 스타일)"""
    log_dict = {}
    for k, v in train_out.items():
        if k == "timestep":
            continue
        if isinstance(v, (int, float, np.floating)):
            log_dict[f"train/{k}"] = float(v)
    log_dict["train/global_step"] = int(step)
    wandb.log(log_dict, step=int(step))


def log_eval_wandb(actor_results: list, step: int):
    """평가 메트릭 wandb 로깅 (pogogo.py 스타일)"""
    eval_log_dict = {"eval/global_step": int(step)}
    for i, r in enumerate(actor_results):
        eval_log_dict[f"eval/actor_{i}/det_score"] = float(r["det_score"])
        eval_log_dict[f"eval/actor_{i}/det_avg"] = float(r["det_avg"])
        eval_log_dict[f"eval/actor_{i}/stoch_score"] = float(r["stoch_score"])
        eval_log_dict[f"eval/actor_{i}/stoch_avg"] = float(r["stoch_avg"])
    wandb.log(eval_log_dict, step=int(step))


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

    # Base trainer 생성 및 AlgorithmInterface 인스턴스 생성
    sinkhorn_loss = SamplesLoss(loss="sinkhorn", p=2, blur=config.sinkhorn_blur, backend=config.sinkhorn_backend)
    algorithm = None

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

        algorithm = IQLAlgorithm(trainer)
        
        def train_fn(batch):
            seed_base = (config.seed if config.seed else 0) * 1000000 + trainer.total_it * 1000
            return update_multi_actor_pytorch(
                trainer=trainer,
                actors=actors,
                actor_targets=actor_targets,
                actor_optimizers=actor_optimizers,
                actor_is_stochastic=actor_is_stochastic,
                actor_is_gaussian=actor_is_gaussian,
                w2_weights=config.w2_weights,
                batch=batch,
                algorithm=algorithm,
                sinkhorn_K=config.sinkhorn_K,
                sinkhorn_blur=config.sinkhorn_blur,
                sinkhorn_backend=config.sinkhorn_backend,
                sinkhorn_loss=sinkhorn_loss,
                seed_base=seed_base,
                tau=config.tau,
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
        
        algorithm = TD3BCAlgorithm(trainer)

        def train_fn(batch):
            seed_base = (config.seed if config.seed else 0) * 1000000 + trainer.total_it * 1000
            # TD3_BC는 policy_freq마다만 actor 업데이트
            if trainer.total_it % trainer.policy_freq == 0:
                return update_multi_actor_pytorch(
                    trainer=trainer,
                    actors=actors,
                    actor_targets=actor_targets,
                    actor_optimizers=actor_optimizers,
                    actor_is_stochastic=actor_is_stochastic,
                    actor_is_gaussian=actor_is_gaussian,
                    w2_weights=config.w2_weights,
                    batch=batch,
                    algorithm=algorithm,
                    sinkhorn_K=config.sinkhorn_K,
                    sinkhorn_blur=config.sinkhorn_blur,
                    sinkhorn_backend=config.sinkhorn_backend,
                    sinkhorn_loss=sinkhorn_loss,
                    seed_base=seed_base,
                    tau=config.tau,
                )
            else:
                # Critic만 업데이트
                log_dict = algorithm.update_critic(
                    trainer,
                    batch,
                    {},
                    actors=actors,
                    actor_targets=actor_targets,
                )
                return log_dict

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
        # CQL trainer에는 원래 TanhGaussianPolicy 사용 (3개의 hidden layer)
        cql_actor = TanhGaussianPolicy(
            state_dim, action_dim, max_action,
            hidden_dim=256, n_hiddens=3,
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
        
        algorithm = CQLAlgorithm(trainer, cql_actor)

        def train_fn(batch):
            seed_base = (config.seed if config.seed else 0) * 1000000 + trainer.total_it * 1000
            return update_multi_actor_pytorch(
                trainer=trainer,
                actors=actors,
                actor_targets=actor_targets,
                actor_optimizers=actor_optimizers,
                actor_is_stochastic=actor_is_stochastic,
                actor_is_gaussian=actor_is_gaussian,
                w2_weights=config.w2_weights,
                batch=batch,
                algorithm=algorithm,
                sinkhorn_K=config.sinkhorn_K,
                sinkhorn_blur=config.sinkhorn_blur,
                sinkhorn_backend=config.sinkhorn_backend,
                sinkhorn_loss=sinkhorn_loss,
                seed_base=seed_base,
                tau=config.tau,
            )

    elif config.algorithm == "awac":
        from .awac import Critic as AWACCritic
        critic_1 = AWACCritic(state_dim, action_dim, hidden_dim=256).to(config.device)
        critic_2 = AWACCritic(state_dim, action_dim, hidden_dim=256).to(config.device)
        critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=config.qf_lr)
        critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=config.qf_lr)

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
        
        if not hasattr(trainer, "total_it"):
            trainer.total_it = 0
        
        algorithm = AWACAlgorithm(trainer)
        
        def train_fn(batch):
            seed_base = (config.seed if config.seed else 0) * 1000000 + trainer.total_it * 1000
            return update_multi_actor_pytorch(
                trainer=trainer,
                actors=actors,
                actor_targets=actor_targets,
                actor_optimizers=actor_optimizers,
                actor_is_stochastic=actor_is_stochastic,
                actor_is_gaussian=actor_is_gaussian,
                w2_weights=config.w2_weights,
                batch=batch,
                algorithm=algorithm,
                sinkhorn_K=config.sinkhorn_K,
                sinkhorn_blur=config.sinkhorn_blur,
                sinkhorn_backend=config.sinkhorn_backend,
                sinkhorn_loss=sinkhorn_loss,
                seed_base=seed_base,
                tau=config.tau,
            )

    elif config.algorithm == "sac_n":
        # SAC-N: VectorizedCritic 사용 (여러 critic ensemble)
        critic = VectorizedCritic(state_dim, action_dim, getattr(config, "hidden_dim", 256), config.num_critics).to(config.device)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.qf_lr)

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
        
        if not hasattr(trainer, "total_it"):
            trainer.total_it = 0
        
        algorithm = SACNAlgorithm(trainer)
        
        def train_fn(batch):
            seed_base = (config.seed if config.seed else 0) * 1000000 + trainer.total_it * 1000
            return update_multi_actor_pytorch(
                trainer=trainer,
                actors=actors,
                actor_targets=actor_targets,
                actor_optimizers=actor_optimizers,
                actor_is_stochastic=actor_is_stochastic,
                actor_is_gaussian=actor_is_gaussian,
                w2_weights=config.w2_weights,
                batch=batch,
                algorithm=algorithm,
                sinkhorn_K=config.sinkhorn_K,
                sinkhorn_blur=config.sinkhorn_blur,
                sinkhorn_backend=config.sinkhorn_backend,
                sinkhorn_loss=sinkhorn_loss,
                seed_base=seed_base,
                tau=config.tau,
            )

    elif config.algorithm == "edac":
        # EDAC: VectorizedCritic 사용 + diversity loss
        critic = VectorizedCritic(state_dim, action_dim, getattr(config, "hidden_dim", 256), config.num_critics).to(config.device)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.qf_lr)

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
        
        if not hasattr(trainer, "total_it"):
            trainer.total_it = 0
        
        algorithm = EDACAlgorithm(trainer)
        
        def train_fn(batch):
            seed_base = (config.seed if config.seed else 0) * 1000000 + trainer.total_it * 1000
            return update_multi_actor_pytorch(
                trainer=trainer,
                actors=actors,
                actor_targets=actor_targets,
                actor_optimizers=actor_optimizers,
                actor_is_stochastic=actor_is_stochastic,
                actor_is_gaussian=actor_is_gaussian,
                w2_weights=config.w2_weights,
                batch=batch,
                algorithm=algorithm,
                sinkhorn_K=config.sinkhorn_K,
                sinkhorn_blur=config.sinkhorn_blur,
                sinkhorn_backend=config.sinkhorn_backend,
                sinkhorn_loss=sinkhorn_loss,
                seed_base=seed_base,
                tau=config.tau,
            )

    else:
        raise ValueError(f"Unsupported algorithm: {config.algorithm}")

    print("---------------------------------------", flush=True)
    print(f"Training POGO Multi-Actor ({config.algorithm.upper()}), Env: {config.env}, Seed: {config.seed}", flush=True)
    print(f"  Actors: {config.num_actors}, W2 weights (Actor1+): {config.w2_weights}", flush=True)
    print("  Actor0: 원래 알고리즘 loss만 사용, Actor1+: Sinkhorn", flush=True)
    print("---------------------------------------", flush=True)

    evaluations = {i: [] for i in range(config.num_actors)}
    
    train_logs = []
    eval_logs = []
    last_eval_step = 0
    
    # wandb 초기화
    import wandb
    if config.use_wandb:
        wandb.init(
            config=asdict(config),
            entity=config.wandb_entity,
            project=config.wandb_project,
            group=config.group,
            name=config.wandb_name,
            id=str(uuid.uuid4()),
        )
    
    global_step = 0  # 마지막 step 저장용
    for t in range(int(config.max_timesteps)):
        global_step = t + 1
        
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        train_out = train_fn(batch)
        
        # wandb 로깅: 학습 메트릭 (pogogo.py 형식)
        if config.use_wandb:
            log_train_wandb(train_out, global_step)
        
        # 공통 메타 필드 부착 (파일 저장용)
        train_log = {
            "update_step": int(global_step),
            "epoch": None,  # 필요하면 채우기
            **{k: float(v) if isinstance(v, (int, float, np.floating)) else v 
               for k, v in train_out.items() 
               if k != "timestep"},
        }
        
        # 내부 timestep 키 제거/통일
        train_log.pop("timestep", None)
        
        # train 로그 저장/출력
        if config.save_train_logs and (global_step % config.log_interval == 0):
            train_logs.append(train_log)
        
        # 체크포인트
        if global_step % config.checkpoint_freq == 0:
            checkpoint_dir = os.path.join("results", config.algorithm, config.env.replace("-", "_"), f"seed_{config.seed}", "checkpoints")
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
            checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_{global_step}.pt")
            torch.save(ckpt, checkpoint_file)
        
        # 진행 상황 출력 (10k step마다)
        if global_step % 10000 == 0 or global_step >= config.max_timesteps:
            print(f"Steps: {global_step}", flush=True)

        # 평가
        if global_step % config.eval_freq == 0:
            print(f"  Evaluation at step {global_step}:", flush=True)
            actor_results = []
            for i in range(config.num_actors):
                # pogogo.py 스타일: base_seed = config.seed + 100 + i * 100
                base_seed = config.seed + 100 + i * 100
                det_avg, det_score, stoch_avg, stoch_score = eval_policy_multi(
                    actor=actors[i],
                    env=env,
                    base_seed=base_seed,
                    n_episodes=config.n_episodes,
                    device=config.device,
                    state_mean=state_mean,
                    state_std=state_std,
                )
                
                actor_results.append({
                    'det_avg': det_avg,
                    'det_score': det_score,
                    'stoch_avg': stoch_avg,
                    'stoch_score': stoch_score
                })
                
                print(f"    Actor {i} - Deterministic: {det_avg:.3f}, D4RL score: {det_score:.3f}", flush=True)
                print(f"    Actor {i} - Stochastic: {stoch_avg:.3f}, D4RL score: {stoch_score:.3f}", flush=True)
                
                evaluations[i].append(det_score)
                
                e = {
                    "update_step": int(global_step),
                    "actor": int(i),
                    "det_score": det_score,
                    "det_avg": det_avg,
                    "stoch_score": stoch_score,
                    "stoch_avg": stoch_avg,
                }
                eval_logs.append(e)
            
            # wandb 로깅: 평가 메트릭 (pogogo.py 형식)
            if config.use_wandb:
                log_eval_wandb(actor_results, global_step)
            
            last_eval_step = global_step

    # 학습 완료 후 최종 평가 (pogogo.py 형식)
    print("\n" + "=" * 60, flush=True)
    print("Training completed!", flush=True)
    print("=" * 60, flush=True)
    
    # 최종 평가: 각 actor마다 deterministic과 stochastic 평가 (pogogo.py 스타일)
    for i in range(config.num_actors):
        print(f"\n======== Final Evaluation: Actor {i} ========", flush=True)
        
        det_scores, stoch_scores = [], []
        for r in range(config.final_eval_runs):
            # Deterministic 평가: base_seed = 1000 + 100 * r (pogogo.py 스타일)
            base_seed_det = 1000 + 100 * r
            _, det_score, _, _ = eval_policy_multi(
                actor=actors[i],
                env=env,
                base_seed=base_seed_det,
                n_episodes=config.final_eval_episodes,
                device=config.device,
                state_mean=state_mean,
                state_std=state_std,
            )
            det_scores.append(det_score)
            
            # Stochastic 평가: base_seed = 2000 + 100 * r (pogogo.py 스타일)
            base_seed_stoch = 2000 + 100 * r
            _, _, _, stoch_score = eval_policy_multi(
                actor=actors[i],
                env=env,
                base_seed=base_seed_stoch,
                n_episodes=config.final_eval_episodes,
                device=config.device,
                state_mean=state_mean,
                state_std=state_std,
            )
            stoch_scores.append(stoch_score)
        
        det_scores = np.array(det_scores, dtype=np.float32)
        stoch_scores = np.array(stoch_scores, dtype=np.float32)
        
        print(f"[FINAL] Deterministic: mean={det_scores.mean():.3f}, std={det_scores.std():.3f} over {config.final_eval_runs}x{config.final_eval_episodes}", flush=True)
        print(f"[FINAL] Stochastic:   mean={stoch_scores.mean():.3f}, std={stoch_scores.std():.3f} over {config.final_eval_runs}x{config.final_eval_episodes}", flush=True)
        
        # wandb 로깅: 최종 평가 결과 (pogogo.py 형식)
        if config.use_wandb:
            wandb.log({
                f"final_actor_{i}/det_mean": float(det_scores.mean()),
                f"final_actor_{i}/det_std": float(det_scores.std()),
                f"final_actor_{i}/stoch_mean": float(stoch_scores.mean()),
                f"final_actor_{i}/stoch_std": float(stoch_scores.std()),
            }, step=global_step)
        
        if evaluations[i]:
            final_score = evaluations[i][-1]
            best_score = max(evaluations[i])
            print(f"Actor {i}: Final={final_score:.1f}, Best={best_score:.1f}", flush=True)
    
    # 로그 저장: results/{algorithm}/{env}/seed_{seed}/logs/
    import json
    import datetime
    
    env_name = config.env.replace("-", "_")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_dir = os.path.join("results", config.algorithm, env_name, f"seed_{config.seed}", "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    train_log_file = os.path.join(
        log_dir, f"train_{config.algorithm}_{env_name}_seed{config.seed}_{timestamp}.jsonl"
    )
    eval_log_file = os.path.join(
        log_dir, f"eval_{config.algorithm}_{env_name}_seed{config.seed}_{timestamp}.jsonl"
    )
    summary_file = os.path.join(
        log_dir, f"summary_{config.algorithm}_{env_name}_seed{config.seed}_{timestamp}.json"
    )
    
    # JSONL (line-delimited)
    if config.save_train_logs:
        with open(train_log_file, "w") as f:
            for row in train_logs:
                f.write(json.dumps(row) + "\n")
    
    with open(eval_log_file, "w") as f:
        for row in eval_logs:
            f.write(json.dumps(row) + "\n")
    
    eval_summary = {
        "algorithm": config.algorithm,
        "env": config.env,
        "seed": int(config.seed),
        "num_actors": int(config.num_actors),
        "w2_weights": [float(w) for w in config.w2_weights],
        "max_timesteps": int(config.max_timesteps),
        "log_interval": int(config.log_interval),
        "eval_freq": int(config.eval_freq),
        "checkpoint_freq": int(config.checkpoint_freq),
        "final_step": int(global_step),
        "evaluations": {str(i): [float(x) for x in evaluations[i]] for i in range(config.num_actors)},
        "final_scores": {
            str(i): (float(evaluations[i][-1]) if len(evaluations[i]) > 0 else None)
            for i in range(config.num_actors)
        },
        "best_scores": {
            str(i): (float(max(evaluations[i])) if len(evaluations[i]) > 0 else None)
            for i in range(config.num_actors)
        },
        "files": {
            "train_jsonl": train_log_file if config.save_train_logs else None,
            "eval_jsonl": eval_log_file,
        },
    }
    
    with open(summary_file, "w") as f:
        json.dump(eval_summary, f, indent=2)
    
    print(f"\n로그 저장 완료:", flush=True)
    if config.save_train_logs:
        print(f"  train:   {train_log_file}", flush=True)
    print(f"  eval:    {eval_log_file}", flush=True)
    print(f"  summary: {summary_file}", flush=True)
    
    # wandb 종료
    if config.use_wandb:
        wandb.finish()
    
    # Checkpoint 저장: results/{algorithm}/{env}/seed_{seed}/checkpoints/
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
    
    checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_{global_step}.pt")
    torch.save(ckpt, checkpoint_file)
    print(f"Checkpoint 저장 완료: {checkpoint_file}", flush=True)

    return trainer, actors


if __name__ == "__main__":
    train()
