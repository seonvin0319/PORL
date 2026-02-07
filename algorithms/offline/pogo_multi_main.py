# POGO Multi-Actor Main Script
# 각 알고리즘(IQL, CQL, TD3_BC 등)의 구조를 그대로 사용하되,
# Actor만 multi-actor로 교체하여 학습
# Config에서 algorithm 선택 가능

import copy
import os
import random
import sys
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
from .iql import ImplicitQLearning, TwinQ, ValueFunction, GaussianPolicy, DeterministicPolicy
from .td3_bc import TD3_BC, Actor, Critic
from .cql import ContinuousCQL, TanhGaussianPolicy
from .awac import AdvantageWeightedActorCritic
from .utils_pytorch import soft_update
from .sac_n import SACN, Actor as SACNActor, VectorizedCritic
from .edac import EDAC, Actor as EDACActor
from algorithms.networks import BaseActor, DeterministicMLP, StochasticMLP, GaussianMLP, TanhGaussianMLP
from .utils_pytorch import ActorConfig, action_for_loss
from utils.policy_call import get_action, act_for_eval, sample_K_actions

TensorBatch = List[torch.Tensor]


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
    use_wandb: bool = True
    project: str = "PORL"
    group: str = "POGO-Multi"
    name: str = "POGO-Multi"

    def __post_init__(self):
        # w2_weights는 config에서 직접 설정 (자동 설정 제거)
        # 기본값이 없으면 [10.0, 10.0] 사용
        if self.w2_weights is None or len(self.w2_weights) == 0:
            self.w2_weights = [10.0, 10.0]
        
        if self.num_actors is None:
            # w2_weights는 Actor1부터이므로 num_actors = len(w2_weights) + 1
            self.num_actors = len(self.w2_weights) + 1
        # w2_weights는 Actor1부터이므로 num_actors - 1 길이여야 함
        expected_len = self.num_actors - 1
        if len(self.w2_weights) < expected_len:
            w = self.w2_weights[-1] if self.w2_weights else 10.0
            self.w2_weights = self.w2_weights + [w] * (expected_len - len(self.w2_weights))
        self.w2_weights = self.w2_weights[:expected_len]
        # wandb 이름: 환경-알고리즘-seed{seed}-w{weight}
        weight_str = str(int(self.w2_weights[0])) if self.w2_weights else "10"
        self.name = f"{self.env}-{self.algorithm}-seed{self.seed}-w{weight_str}"
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
    # forward(obs, deterministic=False) 또는 sample_actions로 K개 샘플
    a = sample_K_actions(policy, states, K=K, deterministic=False, seed=None if seed is None else seed + 0)
    with torch.no_grad():
        b_detached = sample_K_actions(
            ref, states, K=K, deterministic=False,
            seed=None if seed is None else seed + 10000,
        ).detach()
    loss = sinkhorn_loss(a, b_detached)
    return loss.mean() if loss.dim() > 0 else loss


@dataclass
class ActorCreationConfig:
    """알고리즘별 actor 생성 설정. 각 알고리즘 블록에서 config를 만들어 전달."""
    create_actor0: Callable[[], Tuple[nn.Module, bool, bool]]
    pogo_default_type: str = "stochastic"
    pogo_default_tanh_mean: bool = True


@dataclass
class Actor0Config:
    """Actor0 구조 정보. 각 trainer에 저장하여 재사용/로깅용."""
    type: str  # "gaussian", "tanh_gaussian", "deterministic", "stochastic"
    kwargs: dict  # tanh_mean 등
    class_name: str
    is_stochastic: bool
    is_gaussian: bool


def _build_actor0_config(actor0: nn.Module, is_stochastic: bool, is_gaussian: bool) -> Actor0Config:
    """Actor0로부터 Actor0Config 생성. trainer.actor0_config에 저장용."""
    pogo_type, kwargs = _infer_actor0_type(actor0)
    return Actor0Config(
        type=pogo_type,
        kwargs=kwargs,
        class_name=type(actor0).__name__,
        is_stochastic=is_stochastic,
        is_gaussian=is_gaussian,
    )


def _infer_actor0_type(actor0: nn.Module) -> Tuple[str, dict]:
    """Actor0로부터 POGO 타입 및 kwargs 추론. type 미지정 시 Actor1+ 초기화용."""
    if isinstance(actor0, GaussianMLP):
        return "gaussian", {"tanh_mean": getattr(actor0, "tanh_mean", True)}
    if isinstance(actor0, TanhGaussianMLP):
        return "tanh_gaussian", {}
    if isinstance(actor0, DeterministicMLP):
        return "deterministic", {}
    if isinstance(actor0, StochasticMLP) or isinstance(actor0, BaseActor):
        return "stochastic", {}
    # td3_bc.Actor 등 외부 클래스: is_stochastic로 판단
    stoch = getattr(actor0, "is_stochastic", False)
    gauss = getattr(actor0, "is_gaussian", False)
    if not stoch:
        return "deterministic", {}
    if gauss:
        return "gaussian", {"tanh_mean": getattr(actor0, "tanh_mean", True)}
    return "tanh_gaussian", {}  # stochastic + not gaussian -> tanh_gaussian 대체


def _create_actors(
    state_dim: int,
    action_dim: int,
    max_action: float,
    num_actors: int,
    actor_configs: Optional[List[dict]],
    actor_creation: ActorCreationConfig,
    device: str,
    lr: float,
    base_config: Optional[Actor0Config] = None,
) -> Tuple[List[nn.Module], List[Optional[nn.Module]], List[torch.optim.Optimizer], List[bool], List[bool], Optional[Actor0Config]]:
    """config로 actor 생성: actor0=create_actor0(), actor1+=POGO policies.
    actor1+ type 미지정 시 base_config(저장된 config) 사용. base_config 없으면 actor0에서 추론.
    """
    actors = []
    actor_targets = []
    actor_optimizers = []
    actor_is_stochastic = []
    actor_is_gaussian = []

    # actor0: 알고리즘에서 준 create_actor0 호출
    if num_actors >= 1:
        actor0, stoch0, gauss0 = actor_creation.create_actor0()
        actor0 = actor0.to(device)
        actors.append(actor0)
        actor_targets.append(copy.deepcopy(actor0))
        actor_optimizers.append(torch.optim.Adam(actor0.parameters(), lr=lr))
        actor_is_stochastic.append(stoch0)
        actor_is_gaussian.append(gauss0)

    stored_config = base_config
    if stored_config is None and num_actors >= 1:
        stored_config = _build_actor0_config(actors[0], actor_is_stochastic[0], actor_is_gaussian[0])

    # actor1+: POGO policies
    num_pogo = num_actors - 1
    if num_pogo <= 0:
        return actors, actor_targets, actor_optimizers, actor_is_stochastic, actor_is_gaussian, stored_config

    default_type = actor_creation.pogo_default_type
    default_tanh_mean = actor_creation.pogo_default_tanh_mean
    actor0_ref = actors[0]
    # type 미지정 시 저장된 config 사용
    fallback_type = stored_config.type if stored_config else default_type
    fallback_kw = stored_config.kwargs if stored_config else {}

    cfg_1plus = actor_configs[1:] if actor_configs and len(actor_configs) > 1 else None
    if cfg_1plus is None or len(cfg_1plus) == 0:
        cfg_1plus = [{"type": fallback_type, "_from_stored": True, **fallback_kw} for _ in range(num_pogo)]
    else:
        for i in range(num_pogo):
            if i >= len(cfg_1plus) or not cfg_1plus[i]:
                cfg_1plus.append({"type": fallback_type, "_from_stored": True, **fallback_kw})
            else:
                cfg_1plus[i] = cfg_1plus[i] or {}
                if "type" not in cfg_1plus[i] or cfg_1plus[i].get("type") is None:
                    cfg_1plus[i]["type"] = fallback_type
                    cfg_1plus[i]["_from_stored"] = True
                    cfg_1plus[i].update(fallback_kw)
                else:
                    actor_type = cfg_1plus[i].get("type", default_type)
                    if actor_type == "gaussian" and "tanh_mean" not in cfg_1plus[i]:
                        cfg_1plus[i]["tanh_mean"] = default_tanh_mean

    for i in range(num_pogo):
        c = cfg_1plus[i]
        actor_type = c.get("type", default_type) if isinstance(c, dict) else getattr(c, "type", default_type)
        from_stored = c.get("_from_stored", False) if isinstance(c, dict) else False

        if actor_type == "gaussian":
            tanh_mean = c.get("tanh_mean", True) if isinstance(c, dict) else getattr(c, "tanh_mean", True)
            actor = GaussianMLP(state_dim, action_dim, max_action, tanh_mean=tanh_mean).to(device)
        elif actor_type == "tanh_gaussian":
            actor = TanhGaussianMLP(state_dim, action_dim, max_action).to(device)
        elif actor_type == "deterministic":
            actor = DeterministicMLP(state_dim, action_dim, max_action).to(device)
        else:
            actor = StochasticMLP(state_dim, action_dim, max_action).to(device)

        if from_stored and actor0_ref is not None:
            try:
                src_sd = actor0_ref.state_dict()
                tgt_sd = actor.state_dict()
                if set(src_sd.keys()) == set(tgt_sd.keys()):
                    actor.load_state_dict(src_sd, strict=True)
                else:
                    common = {k: v for k, v in src_sd.items() if k in tgt_sd and v.shape == tgt_sd[k].shape}
                    if common:
                        actor.load_state_dict({**tgt_sd, **common}, strict=False)
            except Exception:
                pass

        actors.append(actor)
        actor_targets.append(copy.deepcopy(actor))
        actor_optimizers.append(torch.optim.Adam(actor.parameters(), lr=lr))
        actor_is_stochastic.append(getattr(actor, "is_stochastic", False))
        actor_is_gaussian.append(getattr(actor, "is_gaussian", False))

    return actors, actor_targets, actor_optimizers, actor_is_stochastic, actor_is_gaussian, stored_config


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
    - Both Gaussian (not TanhGaussian) → closed form W2
    - Both Stochastic (TanhGaussian 포함) → Sinkhorn
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
        # At least one deterministic: use L2 (미분 가능한 action 사용)
        pi_i = action_for_loss(actor_i_config.actor, actor_i_config, states, seed=seed)
        with torch.no_grad():
            ref_a = action_for_loss(
                ref_actor_config.actor, ref_actor_config, states,
                seed=None if seed is None else seed + 999,
            )
        return ((pi_i - ref_a) ** 2).sum(dim=-1).mean()


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


def _train_multi_actor(
    trainer: Any,
    actors: List[nn.Module],
    actor_targets: List[nn.Module],
    actor_optimizers: List[torch.optim.Optimizer],
    actor_is_stochastic: List[bool],
    actor_is_gaussian: List[bool],
    w2_weights: List[float],
    sinkhorn_K: int,
    sinkhorn_blur: float,
    sinkhorn_backend: str,
    batch: TensorBatch,
    sinkhorn_loss,
    seed_base: int,
    *,
    call_trainer_update_target_network: bool = False,
    target_update_period: Optional[int] = None,
) -> Dict[str, float]:
    """통합 multi-actor train: trainer.train/update + Actor1+ W2.
    Target update (boolean으로 명시):
      - call_trainer_update_target_network: trainer.update_target_network(tau) 호출 여부
      - target_update_period: 위 호출 주기 (None이면 호출 안 함)
    actor_targets는 모두 여기서 관리 (TD3_BC actor_target 포함).
    """
    if hasattr(trainer, "train"):
        log_dict = trainer.train(batch)
    else:
        if not hasattr(trainer, "total_it"):
            trainer.total_it = 0
        trainer.total_it += 1
        log_dict = trainer.update(batch)

    state = batch[0]
    actions = batch[1] if len(batch) > 1 else None
    policy_freq = getattr(trainer, "policy_freq", 1)
    tau = getattr(trainer, "tau", None) or getattr(trainer, "_tau", None) or getattr(trainer, "soft_target_update_rate", 0.005)

    if trainer.total_it % policy_freq == 0:
        for i in range(1, len(actors)):
            def base_loss_fn(actor, idx=i):
                return trainer.compute_actor_base_loss(actor, state, actions, seed=seed_base + 100 + idx)

            actor_i_config = ActorConfig.from_actor(actors[i])
            ref_actor_config = ActorConfig.from_actor(actors[i - 1])
            w2_weight_i = w2_weights[i - 1]
            actor_loss_i, w2_i = _compute_actor_loss_with_w2(
                base_loss_fn=base_loss_fn,
                actor_i_config=actor_i_config,
                ref_actor_config=ref_actor_config,
                states=state,
                w2_weight=w2_weight_i,
                sinkhorn_K=sinkhorn_K,
                sinkhorn_blur=sinkhorn_blur,
                sinkhorn_loss=sinkhorn_loss,
                seed=seed_base + 100 + i,
            )
            actor_optimizers[i].zero_grad()
            actor_loss_i.backward()
            actor_optimizers[i].step()
            log_dict[f"actor_{i}_loss"] = float(actor_loss_i.item())
            if w2_i is not None:
                log_dict[f"w2_{i}_distance"] = float(w2_i.item())

        # Target update (actor_targets는 모두 여기서 관리)
        do_periodic_critic = (
            call_trainer_update_target_network
            and target_update_period is not None
            and trainer.total_it % target_update_period == 0
        )
        if do_periodic_critic:
            trainer.update_target_network(tau)
        for actor, actor_target in zip(actors, actor_targets):
            if actor_target is not None:
                for p, tp in zip(actor.parameters(), actor_target.parameters()):
                    tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

    return log_dict


def _parse_env_name(env_name: str) -> Tuple[str, str]:
    """환경 이름을 파싱하여 (env, task) 반환
    예: 'halfcheetah-medium-v2' -> ('halfcheetah', 'medium_v2')
        'hopper-medium-expert-v2' -> ('hopper', 'medium_expert_v2')
    """
    parts = env_name.split("-")
    if len(parts) < 2:
        return parts[0], "-".join(parts[1:]) if len(parts) > 1 else "default"
    
    # 첫 번째 부분이 환경 이름 (halfcheetah, hopper, walker2d 등)
    env = parts[0]
    # 나머지가 task (medium-v2, medium-expert-v2 등)
    task = "_".join(parts[1:])
    
    return env, task


def _get_config_path(config: TrainConfig) -> str:
    """config 파일 경로 생성: configs/pogo/{algorithm}/{env}/{task}/seed_{seed}/config.yaml"""
    env, task = _parse_env_name(config.env)
    config_dir = Path("configs/pogo") / config.algorithm / env / task / f"seed_{config.seed}"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.yaml"
    return str(config_path)


def _create_config_file(config: TrainConfig) -> str:
    """config 파일을 자동 생성하고 경로 반환"""
    config_path = _get_config_path(config)
    
    # 이미 파일이 있으면 생성하지 않음
    if os.path.exists(config_path):
        print(f"[INFO] Config 파일이 이미 존재합니다: {config_path}", flush=True)
        return config_path
    
    # config 파일 생성
    print(f"[INFO] Config 파일 생성: {config_path}", flush=True)
    
    # 기본값으로 config 생성
    config_dict = asdict(config)
    
    # YAML 파일로 저장 (주석 포함)
    import yaml
    
    # 알고리즘별 주석 템플릿
    algorithm_comments = {
        "iql": "# POGO Multi-Actor with IQL algorithm\n# IQL의 구조(V, Q, Actor)를 그대로 사용하되, Actor만 multi-actor",
        "td3_bc": "# POGO Multi-Actor with TD3_BC algorithm",
        "cql": "# POGO Multi-Actor with CQL algorithm",
        "awac": "# POGO Multi-Actor with AWAC algorithm",
        "sac_n": "# POGO Multi-Actor with SAC-N algorithm",
        "edac": "# POGO Multi-Actor with EDAC algorithm",
    }
    
    comment = algorithm_comments.get(config.algorithm, f"# POGO Multi-Actor with {config.algorithm.upper()} algorithm")
    
    with open(config_path, "w") as f:
        f.write(f"{comment}\n\n")
        f.write(f"algorithm: {config.algorithm}  # 필수: 사용할 알고리즘 선택\n\n")
        f.write("# 환경 설정\n")
        f.write(f"env: {config.env}\n")
        f.write(f"seed: {config.seed}\n")
        f.write(f"eval_freq: {config.eval_freq}\n")
        f.write(f"n_episodes: {config.n_episodes}\n")
        f.write(f"max_timesteps: {config.max_timesteps}\n\n")
        
        f.write("# POGO Multi-Actor\n")
        f.write("# w2_weights: Actor1부터의 가중치 리스트 (Actor0는 W2 penalty 없음)\n")
        w2_str = "[" + ", ".join(str(w) for w in config.w2_weights) + "]"
        f.write(f"w2_weights: {w2_str}\n")
        f.write(f"num_actors: {config.num_actors}\n\n")
        
        f.write("# Sinkhorn\n")
        f.write(f"sinkhorn_K: {config.sinkhorn_K}\n")
        f.write(f"sinkhorn_blur: {config.sinkhorn_blur}\n")
        f.write(f'sinkhorn_backend: "{config.sinkhorn_backend}"\n\n')
        
        # 알고리즘별 파라미터
        if config.algorithm == "iql":
            f.write("# IQL 파라미터 (IQL 구조 그대로 사용)\n")
            f.write(f"iql_tau: {config.iql_tau}\n")
            f.write(f"beta: {config.beta}\n")
            f.write(f"vf_lr: {config.vf_lr}\n")
            f.write(f"qf_lr: {config.qf_lr}\n")
            f.write(f"actor_lr: {config.actor_lr}\n")
            f.write(f"iql_deterministic: {str(config.iql_deterministic).lower()}\n")
            f.write(f"actor_dropout: {config.actor_dropout if config.actor_dropout is not None else 'null'}\n\n")
        elif config.algorithm == "td3_bc":
            f.write("# TD3_BC 파라미터\n")
            f.write(f"alpha: {config.alpha}\n")
            f.write(f"policy_noise: {config.policy_noise}\n")
            f.write(f"noise_clip: {config.noise_clip}\n")
            f.write(f"policy_freq: {config.policy_freq}\n\n")
        elif config.algorithm == "cql":
            f.write("# CQL 파라미터\n")
            f.write(f"cql_alpha: {config.cql_alpha}\n")
            f.write(f"cql_n_actions: {config.cql_n_actions}\n")
            f.write(f"target_entropy: {config.target_entropy if config.target_entropy is not None else 'null'}\n")
            f.write(f"qf_lr: {config.qf_lr}\n")
            f.write(f"actor_lr: {config.actor_lr}\n\n")
        elif config.algorithm == "awac":
            f.write("# AWAC 파라미터\n")
            f.write(f"awac_lambda: {config.awac_lambda}\n")
            f.write(f"exp_adv_max: {config.exp_adv_max}\n\n")
        elif config.algorithm in ["sac_n", "edac"]:
            f.write(f"# {config.algorithm.upper()} 파라미터\n")
            f.write(f"num_critics: {config.num_critics}\n")
            f.write(f"alpha_learning_rate: {config.alpha_learning_rate}\n")
            if config.algorithm == "edac":
                f.write(f"eta: {config.eta}\n")
            f.write("\n")
        
        f.write("# 공통\n")
        f.write(f"batch_size: {config.batch_size}\n")
        f.write(f"discount: {config.discount}\n")
        f.write(f"tau: {config.tau}\n")
        f.write(f"buffer_size: {config.buffer_size}\n")
        f.write(f"normalize: {str(config.normalize).lower()}\n")
        f.write(f"normalize_reward: {str(config.normalize_reward).lower()}\n\n")
        
        f.write("# Wandb\n")
        f.write(f"use_wandb: {str(config.use_wandb).lower()}\n")
        f.write(f"project: {config.project}\n")
        f.write(f"group: {config.group}\n")
        f.write(f"name: {config.name}\n")
    
    print(f"[INFO] Config 파일 생성 완료: {config_path}", flush=True)
    return config_path


def wandb_init(config: TrainConfig) -> None:
    """wandb 초기화. config use_wandb=true면 기본 활성화, mode='online' 고정."""
    if not config.use_wandb:
        return
    wandb.init(
        config=asdict(config),
        project=config.project,
        group=config.group,
        name=config.name,
        id=str(uuid.uuid4()),
        mode="online",
    )
    print(f"[Wandb] 초기화 완료: project={config.project}, run={config.name}", flush=True)


# 나머지 함수들 (compute_mean_std, normalize_states, wrap_env, modify_reward, ReplayBuffer, set_seed, eval_actor)은
# 기존 pogo_multi.py에서 가져오거나 각 알고리즘 파일에서 import

def _parse_args_with_no_wandb():
    """커스텀 argparse 파서: --no_wandb 옵션 지원"""
    import argparse
    import sys
    
    # 기본 pyrallis 파서 생성
    parser = argparse.ArgumentParser()
    
    # --no_wandb 옵션 추가
    parser.add_argument('--no_wandb', action='store_true', 
                       help='wandb를 사용하지 않음 (--use_wandb False와 동일)')
    
    # 나머지 인자는 pyrallis가 처리하도록 전달
    args, remaining = parser.parse_known_args()
    
    # --no_wandb가 설정되면 --use_wandb false로 변환 (config override)
    if args.no_wandb:
        sys.argv = [sys.argv[0]] + ['--use_wandb', 'false'] + remaining
    else:
        sys.argv = [sys.argv[0]] + remaining


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
        eval_actor as eval_actor_base,
    )

    print("=" * 60, flush=True)
    print("POGO Multi-Actor Training Started", flush=True)
    print(f"Algorithm: {config.algorithm}", flush=True)
    print(f"Environment: {config.env}", flush=True)
    print("=" * 60, flush=True)
    print()

    # Seed 설정 (모든 랜덤 소스 초기화)
    # 환경 생성 전에 seed 설정하여 완전한 재현성 보장
    set_seed(config.seed, deterministic_torch=False)
    print(f"Seed set to: {config.seed} (for reproducibility)", flush=True)
    print()

    print(f"Creating environment: {config.env}...", flush=True)
    env = gym.make(config.env)
    # 환경 seed 설정 (gym.make 이후)
    set_seed(config.seed, env=env, deterministic_torch=False)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    print(f"State dimension: {state_dim}", flush=True)
    print(f"Action dimension: {action_dim}", flush=True)
    print(f"Max action: {max_action}", flush=True)
    print()

    print(f"Loading D4RL dataset for {config.env}...", flush=True)
    # D4RL dataset 로딩 (seed가 설정된 상태에서 로딩)
    dataset = d4rl.qlearning_dataset(env)
    print(f"Dataset size: {len(dataset['observations']):,}", flush=True)
    if config.normalize_reward:
        modify_reward(dataset, config.env)

    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = np.zeros(state_dim), np.ones(state_dim)
    print(f"State normalization: mean shape={state_mean.shape}, std shape={state_std.shape}", flush=True)
    print()
    dataset["observations"] = normalize_states(dataset["observations"], state_mean, state_std)
    dataset["next_observations"] = normalize_states(dataset["next_observations"], state_mean, state_std)
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)

    # ReplayBuffer에 seed 전달하여 재현성 보장
    replay_buffer = ReplayBuffer(state_dim, action_dim, config.buffer_size, config.device, seed=config.seed)
    replay_buffer.load_d4rl_dataset(dataset)

    if config.checkpoints_path:
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Seed는 이미 위에서 설정했으므로 중복 호출 제거
    # set_seed는 환경 생성 전과 후에 이미 호출됨

    # wandb 초기화 (기본 활성화, --no_wandb 시 비활성화)
    if config.use_wandb:
        wandb_init(config)
    else:
        print("[Wandb] 비활성화됨 (--no_wandb)", flush=True)

    # Base trainer 생성 (각 알고리즘별)
    sinkhorn_loss = SamplesLoss(loss="sinkhorn", p=2, blur=config.sinkhorn_blur, backend=config.sinkhorn_backend)

    if config.algorithm == "iql":
        def _iql_actor0():
            det = config.iql_deterministic
            drop = config.actor_dropout
            if det:
                a = DeterministicPolicy(state_dim, action_dim, max_action, dropout=drop)
                return a, False, False
            a = GaussianPolicy(state_dim, action_dim, max_action, dropout=drop)
            return a, True, False

        actors, actor_targets, actor_optimizers, actor_is_stochastic, actor_is_gaussian, stored_config = _create_actors(
            state_dim, action_dim, max_action, config.num_actors,
            config.actor_configs,
            ActorCreationConfig(create_actor0=_iql_actor0, pogo_default_type="gaussian", pogo_default_tanh_mean=True),
            config.device, config.actor_lr,
        )
        base_actor0 = actors[0]

        q_network = TwinQ(state_dim, action_dim).to(config.device)
        v_network = ValueFunction(state_dim).to(config.device)
        v_optimizer = torch.optim.Adam(v_network.parameters(), lr=config.vf_lr)
        q_optimizer = torch.optim.Adam(q_network.parameters(), lr=config.qf_lr)

        trainer = ImplicitQLearning(
            max_action=max_action,
            actor=base_actor0,
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
        trainer.actor0_config = stored_config

        def train_fn(batch):
            seed_base = (config.seed if config.seed else 0) * 1000000 + getattr(trainer, "total_it", 0) * 1000
            return _train_multi_actor(
                trainer, actors, actor_targets, actor_optimizers, actor_is_stochastic,
                actor_is_gaussian, config.w2_weights, config.sinkhorn_K, config.sinkhorn_blur,
                config.sinkhorn_backend, batch, sinkhorn_loss, seed_base,
                call_trainer_update_target_network=False,
                target_update_period=None,
            )

    elif config.algorithm == "td3_bc":
        def _td3_actor0():
            a = Actor(state_dim, action_dim, max_action)
            return a, False, False

        actors, actor_targets, actor_optimizers, actor_is_stochastic, actor_is_gaussian, stored_config = _create_actors(
            state_dim, action_dim, max_action, config.num_actors,
            config.actor_configs,
            ActorCreationConfig(create_actor0=_td3_actor0, pogo_default_type="deterministic"),
            config.device, config.actor_lr,
        )
        base_actor0 = actors[0]

        critic_1 = Critic(state_dim, action_dim).to(config.device)
        critic_2 = Critic(state_dim, action_dim).to(config.device)
        critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=config.qf_lr)
        critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=config.qf_lr)

        trainer = TD3_BC(
            max_action=max_action,
            actor=base_actor0,
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
            actor_target=actor_targets[0],
        )
        trainer.actor0_config = stored_config

        def train_fn(batch):
            seed_base = (config.seed if config.seed else 0) * 1000000 + getattr(trainer, "total_it", 0) * 1000
            return _train_multi_actor(
                trainer, actors, actor_targets, actor_optimizers, actor_is_stochastic,
                actor_is_gaussian, config.w2_weights, config.sinkhorn_K, config.sinkhorn_blur,
                config.sinkhorn_backend, batch, sinkhorn_loss, seed_base,
                call_trainer_update_target_network=False,
                target_update_period=None,
            )

    elif config.algorithm == "cql":
        from algorithms.networks import FullyConnectedQFunction
        def _cql_actor0():
            a = TanhGaussianPolicy(state_dim, action_dim, max_action)
            return a, True, False

        actors, actor_targets, actor_optimizers, actor_is_stochastic, actor_is_gaussian, stored_config = _create_actors(
            state_dim, action_dim, max_action, config.num_actors,
            config.actor_configs,
            ActorCreationConfig(create_actor0=_cql_actor0, pogo_default_type="tanh_gaussian"),
            config.device, config.actor_lr,
        )
        base_actor0 = actors[0]

        critic_1 = FullyConnectedQFunction(state_dim, action_dim).to(config.device)
        critic_2 = FullyConnectedQFunction(state_dim, action_dim).to(config.device)
        critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=config.qf_lr)
        critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=config.qf_lr)

        target_entropy_val = config.target_entropy
        if target_entropy_val is None:
            target_entropy_val = -np.prod(env.action_space.shape).item()

        trainer = ContinuousCQL(
            critic_1=critic_1,
            critic_1_optimizer=critic_1_optimizer,
            critic_2=critic_2,
            critic_2_optimizer=critic_2_optimizer,
            actor=base_actor0,
            actor_optimizer=actor_optimizers[0],
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
        trainer.actor0_config = stored_config

        def train_fn(batch):
            seed_base = (config.seed if config.seed else 0) * 1000000 + getattr(trainer, "total_it", 0) * 1000
            return _train_multi_actor(
                trainer, actors, actor_targets, actor_optimizers, actor_is_stochastic,
                actor_is_gaussian, config.w2_weights, config.sinkhorn_K, config.sinkhorn_blur,
                config.sinkhorn_backend, batch, sinkhorn_loss, seed_base,
                call_trainer_update_target_network=False,
                target_update_period=None,
            )

    elif config.algorithm == "awac":
        from .awac import Actor as AWACActor, Critic as AWACCritic
        hd = getattr(config, "hidden_dim", 256)

        def _awac_actor0():
            a = AWACActor(state_dim, action_dim, hd, min_action=-max_action, max_action=max_action)
            return a, True, False

        actors, actor_targets, actor_optimizers, actor_is_stochastic, actor_is_gaussian, stored_config = _create_actors(
            state_dim, action_dim, max_action, config.num_actors,
            config.actor_configs,
            ActorCreationConfig(create_actor0=_awac_actor0, pogo_default_type="gaussian", pogo_default_tanh_mean=False),
            config.device, config.actor_lr,
        )
        base_actor0 = actors[0]

        critic_1 = AWACCritic(state_dim, action_dim, hidden_dim=hd).to(config.device)
        critic_2 = AWACCritic(state_dim, action_dim, hidden_dim=hd).to(config.device)
        critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=config.actor_lr)
        critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=config.actor_lr)

        trainer = AdvantageWeightedActorCritic(
            actor=base_actor0,
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
        trainer.actor0_config = stored_config

        def train_fn(batch):
            seed_base = (config.seed if config.seed else 0) * 1000000 + getattr(trainer, "total_it", 0) * 1000
            return _train_multi_actor(
                trainer, actors, actor_targets, actor_optimizers, actor_is_stochastic,
                actor_is_gaussian, config.w2_weights, config.sinkhorn_K, config.sinkhorn_blur,
                config.sinkhorn_backend, batch, sinkhorn_loss, seed_base,
                call_trainer_update_target_network=False,
                target_update_period=None,
            )

    elif config.algorithm == "sac_n":
        hd = getattr(config, "hidden_dim", 256)

        def _sacn_actor0():
            a = SACNActor(state_dim, action_dim, hd, max_action=max_action)
            return a, True, False

        actors, actor_targets, actor_optimizers, actor_is_stochastic, actor_is_gaussian, stored_config = _create_actors(
            state_dim, action_dim, max_action, config.num_actors,
            config.actor_configs,
            ActorCreationConfig(create_actor0=_sacn_actor0, pogo_default_type="tanh_gaussian"),
            config.device, config.actor_lr,
        )
        base_actor0 = actors[0]

        critic = VectorizedCritic(state_dim, action_dim, hd, config.num_critics).to(config.device)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.actor_lr)

        trainer = SACN(
            actor=base_actor0,
            actor_optimizer=actor_optimizers[0],
            critic=critic,
            critic_optimizer=critic_optimizer,
            gamma=config.discount,
            tau=config.tau,
            alpha_learning_rate=config.alpha_learning_rate,
            device=config.device,
        )
        trainer.actor0_config = stored_config

        def train_fn(batch):
            seed_base = (config.seed if config.seed else 0) * 1000000 + getattr(trainer, "total_it", 0) * 1000
            return _train_multi_actor(
                trainer, actors, actor_targets, actor_optimizers, actor_is_stochastic,
                actor_is_gaussian, config.w2_weights, config.sinkhorn_K, config.sinkhorn_blur,
                config.sinkhorn_backend, batch, sinkhorn_loss, seed_base,
                call_trainer_update_target_network=False,
                target_update_period=None,
            )

    elif config.algorithm == "edac":
        hd = getattr(config, "hidden_dim", 256)

        def _edac_actor0():
            a = EDACActor(state_dim, action_dim, hd, max_action=max_action)
            return a, True, False

        actors, actor_targets, actor_optimizers, actor_is_stochastic, actor_is_gaussian, stored_config = _create_actors(
            state_dim, action_dim, max_action, config.num_actors,
            config.actor_configs,
            ActorCreationConfig(create_actor0=_edac_actor0, pogo_default_type="tanh_gaussian"),
            config.device, config.actor_lr,
        )
        base_actor0 = actors[0]

        critic = VectorizedCritic(state_dim, action_dim, hd, config.num_critics).to(config.device)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.actor_lr)

        trainer = EDAC(
            actor=base_actor0,
            actor_optimizer=actor_optimizers[0],
            critic=critic,
            critic_optimizer=critic_optimizer,
            gamma=config.discount,
            tau=config.tau,
            eta=config.eta,
            alpha_learning_rate=config.alpha_learning_rate,
            device=config.device,
        )
        trainer.actor0_config = stored_config

        def train_fn(batch):
            seed_base = (config.seed if config.seed else 0) * 1000000 + getattr(trainer, "total_it", 0) * 1000
            return _train_multi_actor(
                trainer, actors, actor_targets, actor_optimizers, actor_is_stochastic,
                actor_is_gaussian, config.w2_weights, config.sinkhorn_K, config.sinkhorn_blur,
                config.sinkhorn_backend, batch, sinkhorn_loss, seed_base,
                call_trainer_update_target_network=False,
                target_update_period=None,
            )

    else:
        raise ValueError(f"Unsupported algorithm: {config.algorithm}")

    # W2 distance 계산 방법 결정 (Gaussian policy 사용 시 closed form W2)
    use_gaussian = any(actor_is_gaussian)
    w2_method = "Wasserstein (closed form)" if use_gaussian else "Sinkhorn"
    
    print("---------------------------------------", flush=True)
    print(f"Training POGO Multi-Actor ({config.algorithm.upper()}), Env: {config.env}, Seed: {config.seed}", flush=True)
    print(f"  Actors: {config.num_actors}, W2 weights (Actor1+): {config.w2_weights}", flush=True)
    print(f"  Actor0: 원래 알고리즘 loss만 사용, Actor1+: {w2_method}", flush=True)
    ac0 = getattr(trainer, "actor0_config", None)
    if ac0 is not None:
        print(f"  Actor0 config: type={ac0.type}, class={ac0.class_name}, kwargs={ac0.kwargs}", flush=True)
        if config.use_wandb:
            wandb.config.update({"actor0_config": asdict(ac0)}, allow_val_change=True)
    print("---------------------------------------", flush=True)

    evaluations = {i: [] for i in range(config.num_actors)}
    all_logs = []  # 모든 로그 저장
    
    # Checkpoint 저장 디렉토리 미리 생성 (env 변수명 충돌 방지)
    env_name, task_name = _parse_env_name(config.env)
    checkpoint_dir = os.path.join("results", config.algorithm, env_name, task_name, f"seed_{config.seed}", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(timestep: int, suffix: str = ""):
        """체크포인트 저장 함수"""
        ckpt = {}
        if hasattr(trainer, "actor0_config") and trainer.actor0_config is not None:
            ckpt["actor0_config"] = asdict(trainer.actor0_config)
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
        
        checkpoint_file = os.path.join(checkpoint_dir, f"model_step{timestep}{suffix}.pt")
        torch.save(ckpt, checkpoint_file)
        print(f"Checkpoint 저장 완료: {checkpoint_file}", flush=True)
    
    for t in range(int(config.max_timesteps)):
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        log_dict = train_fn(batch)
        log_dict["timestep"] = t + 1
        all_logs.append(log_dict)
        
        # wandb 로깅
        if config.use_wandb:
            wandb.log(log_dict, step=t + 1)
        
        # 500k timestep마다 체크포인트 저장
        if (t + 1) % 500000 == 0:
            save_checkpoint(t + 1, f"_step{t+1}")

        if (t + 1) % config.eval_freq == 0:
            print(f"[Training] Time steps: {t + 1}", flush=True)
            actor_results = []
            for i in range(config.num_actors):
                # 평가는 각 actor마다
                # deterministic policy → deterministic eval, stochastic policy → stochastic eval
                actor_i = actors[i]
                use_deterministic_eval = not actor_is_stochastic[i]
                
                # 평가용 seed 설정 (actor별, episode별로 일관된 seed 사용)
                eval_seed_base = config.seed + 10000 + i * 1000
                
                # 환경 seed 설정 (재현성 보장)
                try:
                    env.seed(eval_seed_base)
                    env.action_space.seed(eval_seed_base)
                    if hasattr(env, 'observation_space'):
                        env.observation_space.seed(eval_seed_base)
                except Exception:
                    pass
                
                actor_i.eval()
                episode_rewards = []
                for ep in range(config.n_episodes):
                    # 각 episode마다 다른 seed 사용 (하지만 일관성 유지)
                    episode_seed = eval_seed_base + ep
                    try:
                        env.seed(episode_seed)
                        env.action_space.seed(episode_seed)
                    except Exception:
                        pass
                    
                    state, done = env.reset(), False
                    if isinstance(state, tuple):
                        state = state[0]
                    ep_ret = 0.0
                    while not done:
                        action = act_for_eval(
                            actor_i, state, config.device,
                            deterministic=use_deterministic_eval,
                        )
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
                raw_avg = float(scores.mean())
                evaluations[i].append(norm_score)
                log_dict[f"eval_actor_{i}"] = norm_score
                actor_results.append({"raw_avg": raw_avg, "norm_score": norm_score})

                # 평가 결과를 로그에 추가
                eval_log = {
                    "timestep": t + 1,
                    "actor": i,
                    "eval_score": float(norm_score),
                    "raw_score": float(scores.mean()),
                    "std": float(scores.std()),
                }
                all_logs.append(eval_log)

                # wandb에 평가 결과 로깅
                if config.use_wandb:
                    wandb.log({
                        f"eval/actor_{i}/score": float(norm_score),
                        f"eval/actor_{i}/raw_score": float(scores.mean()),
                        f"eval/actor_{i}/std": float(scores.std()),
                    }, step=t + 1)

            print("---------------------------------------", flush=True)
            print(f"Evaluation over {config.n_episodes} episodes:", flush=True)
            for i in range(config.num_actors):
                r = actor_results[i]
                print(f"  Actor {i} - Raw: {r['raw_avg']:.3f}, D4RL score: {r['norm_score']:.1f}", flush=True)
            print("---------------------------------------", flush=True)

    # 학습 완료 후 final eval 전에 체크포인트 저장
    print("\n" + "=" * 60, flush=True)
    print("Training completed! Saving final checkpoint before evaluation...", flush=True)
    print("=" * 60, flush=True)
    save_checkpoint(config.max_timesteps, "_final")
    
    # 학습 완료 후 최종 평가
    print("\n" + "=" * 60, flush=True)
    print("======== Final Evaluation (trained weights) ========", flush=True)
    print("=" * 60, flush=True)
    for i in range(config.num_actors):
        if evaluations[i]:
            final_score = evaluations[i][-1]
            best_score = max(evaluations[i])
            print(f"[FINAL] Actor {i}: Final={final_score:.1f}, Best={best_score:.1f}", flush=True)
    
    # 로그 저장: results/{algorithm}/{env}/{task}/seed_{seed}/logs/
    import json
    import datetime
    env_name, task_name = _parse_env_name(config.env)  # halfcheetah-medium-v2 -> (halfcheetah, medium_v2)
    log_dir = os.path.join("results", config.algorithm, env_name, task_name, f"seed_{config.seed}", "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{config.algorithm}_{config.env.replace('-', '_')}_seed{config.seed}_{timestamp}.json")
    
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
    
    # wandb 종료
    if config.use_wandb:
        wandb.finish()

    return trainer, actors


if __name__ == "__main__":
    _parse_args_with_no_wandb()
    train()
