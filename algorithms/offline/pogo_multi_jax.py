# POGO Multi-Actor (JAX implementation)
# JAX 기반 알고리즘에 POGO Multi-Actor 구조 적용
# - Actor0: 원래 알고리즘 loss만 사용 (W2 penalty 없음)
# - Actor1+: 원래 알고리즘 loss + W2 distance to previous actor

import os

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # For reproducibility

import uuid
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Dict, List, Optional, Tuple

import d4rl  # noqa
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import ott
import pyrallis
import wandb
from flax.core import FrozenDict
from ott.geometry import pointcloud
# OTT-JAX 0.6.0+ API 사용
try:
    from ott.solvers.linear import sinkhorn as sinkhorn_solver
    from ott.problems.linear import linear_problem
    OTT_AVAILABLE = True
except ImportError:
    # Fallback for older versions
    try:
        from ott.tools import sinkhorn as sinkhorn_solver
        OTT_AVAILABLE = True
    except ImportError:
        OTT_AVAILABLE = False

# JAX utilities and ReBRAC imports (for now, can be extended to other algorithms)
from .rebrac import (
    Config as BaseConfig,
    EnsembleCritic,
    CriticTrainState,
    ActorTrainState,
    Metrics,
    ReplayBuffer,
    update_critic,
    make_env,
    wrap_env,
    evaluate,
)

# POGO Policies
from .pogo_policies_jax import (
    GaussianMLP,
    TanhGaussianMLP,
    StochasticMLP,
    DeterministicMLP,
    FQLFlowPolicy,
)

default_kernel_init = nn.initializers.lecun_normal()
default_bias_init = nn.initializers.zeros

# AlgorithmInterface와 관련 클래스들은 utils_jax.py로 이동
from .utils_jax import ActorConfig, AlgorithmInterface, ReBRACAlgorithm


@dataclass
class Config(BaseConfig):
    """POGO Multi-Actor Config (JAX)"""
    # Algorithm 선택 (필수)
    algorithm: str = "rebrac"  # "rebrac" or "fql"
    
    # Wandb 설정
    use_wandb: bool = True
    
    # Logging / checkpoint
    log_interval: int = 1  # train log 주기 (env-step 기준)
    checkpoint_freq: int = int(5e5)  # 500k
    save_train_logs: bool = True
    
    # POGO Multi-Actor 설정
    # w2_weights: Actor1부터의 가중치 리스트 (Actor0는 W2 penalty 없음)
    w2_weights: List[float] = field(default_factory=lambda: [10.0, 10.0])
    num_actors: Optional[int] = None
    actor_configs: Optional[List[dict]] = None  # [{"type": "gaussian"|"stochastic"|"deterministic"|"flow"}, ...]
    
    # Sinkhorn 설정 (Actor1+용, Gaussian이 아닌 경우에만 사용)
    sinkhorn_K: int = 4
    sinkhorn_blur: float = 0.05  # 하한 1e-4로 자동 설정됨 (수치 안정성)
    sinkhorn_backend: str = "jax"  # "jax" (pure JAX) or "ott" (OTT-JAX). 기본값은 "jax" (K가 작아서 충분)
    
    # FQL 설정 (FQL 알고리즘 사용 시)
    fql_alpha: float = 10.0  # Distillation loss coefficient
    fql_flow_steps: int = 10  # Number of flow steps for BC flow
    fql_q_agg: str = "mean"  # Q aggregation: "mean" or "min"
    fql_normalize_q_loss: bool = False
    
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
        
        # wandb run 이름 형식: 환경-알고리즘이름-seed{}-w{}
        # 정수면 소수점 제거, 같은 값이면 하나만 표시
        # 예: halfcheetah-medium-v2-rebrac-seed0-w10 (w2_weights=[10.0, 10.0]인 경우)
        if len(self.w2_weights) > 0:
            # 모든 값이 같은지 확인
            if len(set(self.w2_weights)) == 1:
                # 같은 값이면 하나만 사용
                w = self.w2_weights[0]
                # 정수면 소수점 제거
                w_str = str(int(w)) if w == int(w) else str(w)
            else:
                # 다른 값이면 모두 표시하되 정수면 소수점 제거
                w_str = "_".join([str(int(w)) if w == int(w) else str(w) for w in self.w2_weights])
        else:
            w_str = "10"
        self.name = f"{self.dataset_name}-{self.algorithm}-seed{self.train_seed}-w{w_str}"


# import-time probe 제거: sinkhorn_backend는 Config에서 명시적으로 선택


def _sinkhorn_pure_jax(cost_matrix: jax.Array, blur: float, num_iterations: int) -> jax.Array:
    """
    Pure JAX implementation of Sinkhorn algorithm using lax.scan
    Log-domain implementation for numerical stability
    
    Args:
        cost_matrix: [K, K] cost matrix
        blur: regularization parameter (epsilon), should be >= 1e-4 for stability
        num_iterations: number of Sinkhorn iterations
    
    Returns:
        Regularized OT cost (scalar)
    """
    K = cost_matrix.shape[0]
    
    # Cost shift for numerical stability: subtract min to avoid overflow
    cost_min = jnp.min(cost_matrix)
    cost_shifted = cost_matrix - cost_min
    
    # Blur 하한 설정 (너무 작으면 underflow 발생)
    blur_safe = jnp.maximum(blur, 1e-4)
    
    # Log-domain Sinkhorn: log(K_matrix) = -cost_shifted / blur_safe
    log_K = -cost_shifted / blur_safe
    
    def sinkhorn_step_log(carry, _):
        log_u, log_v = carry
        
        # Update log_u: log_u = -logsumexp(log_K + log_v, axis=1)
        # logsumexp(a) = log(sum(exp(a))) = max(a) + log(sum(exp(a - max(a))))
        log_Kv = log_K + log_v[None, :]  # [K, K]
        log_u_new = -jax.scipy.special.logsumexp(log_Kv, axis=1)  # [K]
        # Normalize: subtract log(sum(exp(log_u)))
        log_u_new = log_u_new - jax.scipy.special.logsumexp(log_u_new)
        
        # Update log_v: log_v = -logsumexp(log_u + log_K, axis=0)
        log_uK = log_u_new[:, None] + log_K  # [K, K]
        log_v_new = -jax.scipy.special.logsumexp(log_uK, axis=0)  # [K]
        # Normalize: subtract log(sum(exp(log_v)))
        log_v_new = log_v_new - jax.scipy.special.logsumexp(log_v_new)
        
        return (log_u_new, log_v_new), None
    
    # Initialize in log domain: log(1/K) = -log(K)
    log_K_init = jnp.log(K)
    log_u = jnp.zeros(K) - log_K_init
    log_v = jnp.zeros(K) - log_K_init
    
    # Scan over iterations
    (log_u, log_v), _ = jax.lax.scan(sinkhorn_step_log, (log_u, log_v), None, length=num_iterations)
    
    # Compute transport plan in log domain and convert back
    # transport[i,j] = exp(log_u[i] + log_K[i,j] + log_v[j])
    log_transport = log_u[:, None] + log_K + log_v[None, :]
    transport = jnp.exp(log_transport)
    
    # Compute cost: sum(transport * cost_matrix)
    # Note: cost_matrix is used (not cost_shifted) since transport sums to 1
    return (transport * cost_matrix).sum()


def sinkhorn_distance_jax(
    x: jax.Array,
    y: jax.Array,
    blur: float = 0.05,
    num_iterations: int = 100,
    backend: str = "jax",
) -> jax.Array:
    """
    Sinkhorn distance 계산 (pure JAX 또는 OTT-JAX)
    
    Args:
        x: [B, K, action_dim] 첫 번째 분포의 샘플
        y: [B, K, action_dim] 두 번째 분포의 샘플 (detached)
        blur: regularization parameter (epsilon)
        num_iterations: Sinkhorn 알고리즘 반복 횟수
        backend: "jax" (pure JAX, 기본값) or "ott" (OTT-JAX)
    
    Returns:
        [B] 각 state에 대한 Sinkhorn distance
    """
    B, K, action_dim = x.shape
    
    if backend == "jax":
        # Pure JAX 경로 (log-domain 구현으로 수치 안정성 확보)
        # Blur 하한 설정 (수치 안정성)
        blur_safe = max(blur, 1e-4)
        
        def compute_sinkhorn_for_batch(x_batch: jax.Array, y_batch: jax.Array) -> jax.Array:
            """단일 배치에 대한 Sinkhorn distance 계산 (pure JAX, log-domain)"""
            # Cost matrix 계산: [K, K]
            cost = jnp.sum((x_batch[:, None, :] - y_batch[None, :, :]) ** 2, axis=-1)
            return _sinkhorn_pure_jax(cost, blur_safe, num_iterations)
        
        # vmap을 사용하여 배치 처리
        distances = jax.vmap(compute_sinkhorn_for_batch)(x, y)  # [B]
        return distances
    
    elif backend == "ott":
        # OTT-JAX 경로 (solver 재생성 최소화를 위해 외부에서 생성)
        if not OTT_AVAILABLE:
            raise RuntimeError("OTT-JAX backend requested but OTT is not available. Use backend='jax' instead.")
        
        # Solver를 외부에서 생성하여 재사용 (vmap 밖에서, closure로 캡처)
        # num_iterations는 함수 파라미터이므로 여기서 사용 가능
        solver = sinkhorn_solver.Sinkhorn(threshold=1e-3, max_iterations=num_iterations)
        
        def compute_sinkhorn_for_batch_ott(x_batch: jax.Array, y_batch: jax.Array) -> jax.Array:
            """단일 배치에 대한 Sinkhorn distance 계산 (OTT-JAX, solver는 외부에서 캡처)"""
            # Blur 하한 설정 (수치 안정성)
            blur_safe = max(blur, 1e-4)
            geom = pointcloud.PointCloud(x_batch, y_batch, epsilon=blur_safe)
            a = jnp.ones(K) / K
            b = jnp.ones(K) / K
            prob = linear_problem.LinearProblem(geom, a=a, b=b)
            output = solver(prob)  # 외부에서 생성된 solver 재사용
            # 결과 추출
            if hasattr(output, 'reg_ot_cost'):
                return output.reg_ot_cost
            elif hasattr(output, 'transport_cost'):
                return output.transport_cost
            elif hasattr(output, 'ent_reg_cost'):
                return output.ent_reg_cost
            else:
                # Fallback to pure JAX if output format is unexpected
                cost = jnp.sum((x_batch[:, None, :] - y_batch[None, :, :]) ** 2, axis=-1)
                return _sinkhorn_pure_jax(cost, blur_safe, num_iterations)
        
        # vmap을 사용하여 배치 처리
        distances = jax.vmap(compute_sinkhorn_for_batch_ott)(x, y)  # [B]
        return distances
    
    else:
        raise ValueError(f"Unknown sinkhorn_backend: {backend}. Must be 'jax' or 'ott'.")


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


def _compute_w2_distance_jax(
    actor_i_params: FrozenDict,
    actor_i_module: nn.Module,
    actor_i_is_gaussian: bool,
    actor_i_is_stochastic: bool,
    ref_actor_params: FrozenDict,
    ref_actor_module: nn.Module,
    ref_actor_is_gaussian: bool,
    ref_actor_is_stochastic: bool,
    states: jax.Array,
    key: jax.random.PRNGKey,
    sinkhorn_K: int = 4,
    sinkhorn_blur: float = 0.05,
    sinkhorn_backend: str = "jax",
) -> jax.Array:
    """
    W2 distance 계산 헬퍼 함수 (JAX-friendly)
    ActorConfig 객체 대신 필요한 정보만 직접 전달하여 객체 조작 최소화
    
    Args:
        actor_i_params: 현재 actor parameters
        actor_i_module: 현재 actor module
        actor_i_is_gaussian: 현재 actor가 Gaussian인지
        actor_i_is_stochastic: 현재 actor가 stochastic인지
        ref_actor_params: 참조 actor parameters
        ref_actor_module: 참조 actor module
        ref_actor_is_gaussian: 참조 actor가 Gaussian인지
        ref_actor_is_stochastic: 참조 actor가 stochastic인지
        states: [B, state_dim]
        key: PRNG key
        sinkhorn_K: 샘플 수 (Gaussian이 아닌 경우에만 사용)
        sinkhorn_blur: Sinkhorn regularization (Gaussian이 아닌 경우에만 사용)
        sinkhorn_backend: "jax" (pure JAX) or "ott" (OTT-JAX)
    
    Returns:
        평균 distance (scalar)
    """
    # Both Gaussian: use closed form W2
    if actor_i_is_gaussian and ref_actor_is_gaussian:
        mean_i, std_i = actor_i_module.get_mean_std(actor_i_params, states)  # [B, action_dim]
        mean_ref, std_ref = ref_actor_module.get_mean_std(ref_actor_params, states)  # [B, action_dim]
        mean_ref = jax.lax.stop_gradient(mean_ref)
        std_ref = jax.lax.stop_gradient(std_ref)
        
        w2_squared = closed_form_w2_gaussian(mean_i, std_i, mean_ref, std_ref)  # [B]
        return w2_squared.mean()
    
    # At least one is not Gaussian: use sampling-based methods
    key1, key2 = jax.random.split(key)
    
    # Sample from current actor
    a = actor_i_module.sample_actions(actor_i_params, states, key1, sinkhorn_K)  # [B, K, action_dim]
    
    # Sample from reference actor (stop_gradient)
    b = ref_actor_module.sample_actions(ref_actor_params, states, key2, sinkhorn_K)  # [B, K, action_dim]
    b = jax.lax.stop_gradient(b)
    
    if actor_i_is_stochastic and ref_actor_is_stochastic:
        # Both stochastic (but not Gaussian): use Sinkhorn
        distances = sinkhorn_distance_jax(a, b, blur=sinkhorn_blur, backend=sinkhorn_backend)  # [B]
        return distances.mean()
    else:
        # At least one deterministic: use L2
        a_det = a[:, 0, :]  # [B, action_dim] - take first sample
        b_det = b[:, 0, :]  # [B, action_dim]
        distances = jnp.sum((a_det - b_det) ** 2, axis=-1)  # [B]
        return distances.mean()


def _update_single_actor(
    actor: ActorTrainState,
    actor_module: nn.Module,
    actor_config: ActorConfig,
    ref_actor: Optional[ActorTrainState],
    ref_actor_config: Optional[ActorConfig],
    ref_actor_module: Optional[nn.Module],
    w2_weight: Optional[float],
    batch: Dict[str, jax.Array],
    critic: CriticTrainState,
    algorithm: AlgorithmInterface,
    key: jax.random.PRNGKey,
    sinkhorn_K: int,
    sinkhorn_blur: float,
    sinkhorn_backend: str,
    tau: float,
    actor_idx: int,
    metrics: Metrics,
) -> Tuple[ActorTrainState, Metrics]:
    """
    단일 actor 업데이트 헬퍼 함수
    
    Args:
        actor: 현재 actor train state
        actor_module: 현재 actor module
        actor_config: 현재 actor config
        ref_actor: 참조 actor train state (Actor1+인 경우)
        ref_actor_config: 참조 actor config (Actor1+인 경우)
        ref_actor_module: 참조 actor module (Actor1+인 경우)
        w2_weight: W2 distance 가중치 (Actor1+인 경우)
        batch: 배치 데이터
        critic: Critic train state
        algorithm: 알고리즘 인터페이스
        key: PRNG key
        sinkhorn_K: Sinkhorn 샘플 수
        sinkhorn_blur: Sinkhorn regularization
        sinkhorn_backend: "jax" (pure JAX) or "ott" (OTT-JAX)
        tau: Target network 업데이트 계수
        actor_idx: Actor 인덱스
        metrics: Metrics 객체
    
    Returns:
        업데이트된 actor와 metrics
    """
    def actor_loss_fn(params: FrozenDict) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        # Base actor loss는 반드시 algorithm.compute_actor_loss를 통해 계산
        # 알고리즘별 올바른 gradient를 보장하기 위함
        base_loss = algorithm.compute_actor_loss(
            actor_params=params,
            actor_module=actor_module,
            critic=critic,
            batch=batch,
            actor_idx=actor_idx,
        )
        
        if actor_idx == 0:
            # Actor0: Base loss만 사용
            log_dict = {
                f"actor_{actor_idx}_loss": base_loss,
            }
            return base_loss, log_dict
        else:
            # Actor1+: Base loss + W2 distance
            # W2 distance 계산
            key_w2, _ = jax.random.split(key)
            w2_dist = _compute_w2_distance_jax(
                actor_i_params=params,
                actor_i_module=actor_module,
                actor_i_is_gaussian=actor_config.is_gaussian,
                actor_i_is_stochastic=actor_config.is_stochastic,
                ref_actor_params=ref_actor_config.params,
                ref_actor_module=ref_actor_config.module,
                ref_actor_is_gaussian=ref_actor_config.is_gaussian,
                ref_actor_is_stochastic=ref_actor_config.is_stochastic,
                states=batch["states"],
                key=key_w2,
                sinkhorn_K=sinkhorn_K,
                sinkhorn_blur=sinkhorn_blur,
                sinkhorn_backend=sinkhorn_backend,
            )
            
            loss = base_loss + w2_weight * w2_dist
            
            log_dict = {
                f"actor_{actor_idx}_loss": loss,
                f"w2_{actor_idx}_distance": w2_dist,
            }
            return loss, log_dict
    
    grads, log_dict = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    # 바깥에서 metrics 업데이트 (JAX trace 밖에서)
    actor_metrics = metrics.update(log_dict)
    new_actor = actor.apply_gradients(grads=grads)
    
    new_actor = new_actor.replace(
        target_params=optax.incremental_update(
            new_actor.params, actor.target_params, tau
        )
    )
    
    return new_actor, actor_metrics


def update_multi_actor(
    key: jax.random.PRNGKey,
    actors: List[ActorTrainState],
    critic: CriticTrainState,
    batch: Dict[str, jax.Array],
    metrics: Metrics,
    actor_modules: List[nn.Module],
    actor_is_stochastic: List[bool],
    actor_is_gaussian: List[bool],
    w2_weights: List[float],
    sinkhorn_K: int,
    sinkhorn_blur: float,
    sinkhorn_backend: str,
    algorithm: AlgorithmInterface,
    tau: float,
) -> Tuple[jax.random.PRNGKey, List[ActorTrainState], CriticTrainState, Metrics]:
    """
    통합된 multi-actor 업데이트 함수 (JAX-friendly)
    Python for loop를 최소화하고 객체 조작을 줄임
    
    Args:
        key: PRNG key
        actors: Actor train state 리스트
        critic: Critic train state
        batch: 배치 데이터
        metrics: Metrics 객체
        actor_modules: Actor module 리스트
        actor_is_stochastic: Actor stochastic 여부 리스트
        actor_is_gaussian: Actor gaussian 여부 리스트
        w2_weights: W2 distance 가중치 리스트 (Actor1+용)
        sinkhorn_K: Sinkhorn 샘플 수
        sinkhorn_blur: Sinkhorn regularization
        sinkhorn_backend: "jax" (pure JAX) or "ott" (OTT-JAX)
        algorithm: 알고리즘 인터페이스
        tau: Target network 업데이트 계수
    
    Returns:
        업데이트된 key, actors, critic, metrics
    """
    num_actors = len(actors)
    new_metrics = metrics
    
    # Critic 업데이트는 Actor0만 사용 (AlgorithmInterface 사용)
    key, new_critic, new_metrics = algorithm.update_critic(
        key,
        actors[0],
        critic,
        batch,
        actor_module=actor_modules[0],
        metrics=new_metrics,
    )
    
    # Multi-actor 업데이트: Python for loop를 사용하되 객체 생성 최소화
    # (actor_modules가 Python list이므로 완전한 JAX화는 어려움)
    new_actors = []
    for i in range(num_actors):
        actor_i = actors[i]
        actor_module_i = actor_modules[i]
        
        key, actor_key = jax.random.split(key)
        
        # 참조 actor 설정 (Actor1+인 경우)
        if i == 0:
            ref_actor = None
            ref_actor_config = None
            ref_actor_module = None
            w2_weight_i = None
        else:
            ref_actor = actors[i - 1]
            ref_actor_module = actor_modules[i - 1]
            # ActorConfig는 필요한 경우에만 생성
            ref_actor_config = ActorConfig(
                params=ref_actor.params,
                module=ref_actor_module,
                is_stochastic=actor_is_stochastic[i - 1],
                is_gaussian=actor_is_gaussian[i - 1],
            )
            w2_weight_i = w2_weights[i - 1]
        
        # ActorConfig는 _update_single_actor 내부에서 생성하지 않고 여기서 생성
        actor_config_i = ActorConfig(
            params=actor_i.params,
            module=actor_module_i,
            is_stochastic=actor_is_stochastic[i],
            is_gaussian=actor_is_gaussian[i],
        )
        
        new_actor_i, new_metrics = _update_single_actor(
            actor=actor_i,
            actor_module=actor_module_i,
            actor_config=actor_config_i,
            ref_actor=ref_actor,
            ref_actor_config=ref_actor_config,
            ref_actor_module=ref_actor_module,
            w2_weight=w2_weight_i,
            batch=batch,
            critic=new_critic,
            algorithm=algorithm,
            key=actor_key,
            sinkhorn_K=sinkhorn_K,
            sinkhorn_blur=sinkhorn_blur,
            sinkhorn_backend=sinkhorn_backend,
            tau=tau,
            actor_idx=i,
            metrics=new_metrics,
        )
        
        new_actors.append(new_actor_i)
    
    # Critic target network 업데이트
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

    if config.use_wandb:
        wandb.init(
            config=dict_config,
            entity=config.entity,
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
    # Split keys: one for actors, one for critic
    keys = jax.random.split(key, config.num_actors + 2)
    key = keys[0]
    actor_keys = keys[1:config.num_actors+1]
    critic_key = keys[config.num_actors+1]

    eval_env = make_env(config.dataset_name, seed=config.eval_seed)
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
    actor_types = []  # "flow", "tanh_gaussian", "gaussian", "stochastic", "deterministic"
    
    if config.actor_configs is None:
        config.actor_configs = [{"type": "deterministic"} for _ in range(config.num_actors)]
    
    for i in range(config.num_actors):
        actor_config = config.actor_configs[i]
        actor_type = actor_config.get("type", "deterministic")  # "flow", "tanh_gaussian", "gaussian", "stochastic", or "deterministic"
        
        if actor_type == "flow":
            # FQLFlowPolicy: FQL의 multi-step flow matching 구조 사용
            # actor_bc_flow (multi-step) + actor_onestep_flow (one-step)
            actor_module = FQLFlowPolicy(
                action_dim=action_dim,
                max_action=max_action,
                hidden_dim=config.hidden_dim,
                layernorm=config.actor_ln,
                n_hiddens=config.actor_n_hiddens,
            )
            # FQLFlowPolicy: need state, noise, and times for init
            init_noise = jnp.zeros((1, action_dim), dtype=init_state.dtype)
            init_times = jnp.zeros((1, 1), dtype=init_state.dtype)
            # Initialize both actor_bc_flow and actor_onestep_flow
            init_params = actor_module.init(
                actor_keys[i],
                init_state,
                init_noise,
                times=init_times,
            )
            init_target_params = actor_module.init(
                actor_keys[i],
                init_state,
                init_noise,
                times=init_times,
            )
        elif actor_type == "gaussian":
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
        actor_types.append(actor_type)

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

    # Algorithm 선택 및 인스턴스 생성
    if config.algorithm == "rebrac":
        algorithm = ReBRACAlgorithm(
            beta=config.actor_bc_coef,
            gamma=config.gamma,
            tau=config.tau,
            policy_noise=config.policy_noise,
            noise_clip=config.noise_clip,
            normalize_q=config.normalize_q,
        )
    elif config.algorithm == "fql":
        from .fql import FQLAlgorithm
        algorithm = FQLAlgorithm(
            gamma=config.gamma,
            tau=config.tau,
            q_agg=config.fql_q_agg,
            normalize_q_loss=config.fql_normalize_q_loss,
            alpha=config.fql_alpha,
            flow_steps=config.fql_flow_steps,
        )
        # FQL: Actor0만 flow 타입이어야 함 (BC policy)
        # Actor1+는 다른 policy 타입 사용 가능 (stochastic, deterministic 등)
        if config.actor_configs[0].get("type") != "flow":
            raise ValueError("FQL algorithm requires Actor0 to be of type 'flow' (BC policy)")
    else:
        raise ValueError(f"Unsupported algorithm: {config.algorithm}")
    
    # Sinkhorn backend 체크 (시작 시점에 한 번만)
    if config.sinkhorn_backend == "ott":
        if not OTT_AVAILABLE:
            raise RuntimeError(
                "sinkhorn_backend='ott' requested but OTT-JAX is not available. "
                "Please install ott-jax or set sinkhorn_backend='jax'."
            )
        # OTT 사용 가능 여부 테스트 (시작 시점에 한 번만)
        try:
            test_x = jnp.ones((4, 2))
            test_y = jnp.ones((4, 2))
            test_geom = pointcloud.PointCloud(test_x, test_y, epsilon=0.05)
            test_a = jnp.ones(4) / 4
            test_b = jnp.ones(4) / 4
            test_prob = linear_problem.LinearProblem(test_geom, a=test_a, b=test_b)
            test_solver = sinkhorn_solver.Sinkhorn(threshold=1e-3, max_iterations=10)
            test_output = test_solver(test_prob)
        except Exception as e:
            raise RuntimeError(
                f"sinkhorn_backend='ott' requested but OTT-JAX test failed: {e}. "
                "Please check OTT installation or set sinkhorn_backend='jax'."
            )
    
    # 통합된 multi-actor 업데이트 함수 사용
    update_multi_actor_partial = partial(
        update_multi_actor,
        actor_modules=actor_modules,
        actor_is_stochastic=actor_is_stochastic,
        actor_is_gaussian=actor_is_gaussian,
        w2_weights=config.w2_weights,
        sinkhorn_K=config.sinkhorn_K,
        sinkhorn_blur=config.sinkhorn_blur,
        sinkhorn_backend=config.sinkhorn_backend,
        algorithm=algorithm,
        tau=config.tau,
    )

    # Metrics
    bc_metrics_to_log = [
        "critic_loss",
        "q_min",
    ] + [f"actor_{i}_loss" for i in range(config.num_actors)] + \
      [f"w2_{i}_distance" for i in range(1, config.num_actors)]
    
    # JAX-friendly state (Python 객체 제외)
    # buffer는 fori_loop 밖에서 Python for로 처리
    current_key = key
    current_actors = actors
    current_critic = critic

    def eval_policy_jax(
        actor_params: FrozenDict,
        action_fn_det,
        action_fn_stoch,
        eval_env,
        base_seed: int,
        eval_episodes: int = 10,
    ):
        """Evaluate policy (deterministic and stochastic) - pogogo.py 형식"""
        # Deterministic
        det_total = 0.0
        for ep in range(eval_episodes):
            ep_seed = base_seed + ep
            np.random.seed(ep_seed)
            try:
                eval_env.action_space.seed(ep_seed)
                reset_result = eval_env.reset(seed=ep_seed)
            except (TypeError, AttributeError):
                try:
                    reset_result = eval_env.reset(seed=ep_seed)
                except TypeError:
                    if hasattr(eval_env, "seed"):
                        eval_env.seed(ep_seed)
                    reset_result = eval_env.reset()

            obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            obs = np.asarray(obs)
            done = False
            ep_ret = 0.0
            while not done:
                obs_batch = obs[None, :] if obs.ndim == 1 else obs
                action_batch = action_fn_det(actor_params, obs_batch)
                action = np.asarray(jax.device_get(action_batch[0] if action_batch.shape[0] == 1 else action_batch))
                step_out = eval_env.step(action)
                if len(step_out) == 5:
                    obs, reward, terminated, truncated, _ = step_out
                    done = terminated or truncated
                else:
                    obs, reward, done, _ = step_out
                ep_ret += reward
            det_total += ep_ret

        # Stochastic (pogogo 규칙: base_seed*10000 + ep*1000 + step_count)
        stoch_total = 0.0
        step_count = 0
        for ep in range(eval_episodes):
            ep_seed = base_seed + ep
            np.random.seed(ep_seed)
            try:
                eval_env.action_space.seed(ep_seed)
                reset_result = eval_env.reset(seed=ep_seed)
            except (TypeError, AttributeError):
                try:
                    reset_result = eval_env.reset(seed=ep_seed)
                except TypeError:
                    if hasattr(eval_env, "seed"):
                        eval_env.seed(ep_seed)
                    reset_result = eval_env.reset()

            obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            obs = np.asarray(obs)
            done = False
            ep_ret = 0.0
            while not done:
                eval_seed = base_seed * 10000 + ep * 1000 + step_count
                eval_key = jax.random.PRNGKey(eval_seed)

                obs_batch = obs[None, :] if obs.ndim == 1 else obs
                action_batch = action_fn_stoch(actor_params, obs_batch, eval_key)
                action = np.asarray(jax.device_get(action_batch[0] if action_batch.shape[0] == 1 else action_batch))

                step_out = eval_env.step(action)
                if len(step_out) == 5:
                    obs, reward, terminated, truncated, _ = step_out
                    done = terminated or truncated
                else:
                    obs, reward, done, _ = step_out
                ep_ret += reward
                step_count += 1
            stoch_total += ep_ret

        det_avg = det_total / eval_episodes
        stoch_avg = stoch_total / eval_episodes
        det_score = eval_env.get_normalized_score(det_avg) * 100.0
        stoch_score = eval_env.get_normalized_score(stoch_avg) * 100.0
        return det_avg, det_score, stoch_avg, stoch_score

    def make_actor_action_fn(actor_idx: int, deterministic: bool = True):
        """Create action function for a specific actor"""
        actor_module = actor_modules[actor_idx]
        actor_type = actor_types[actor_idx]
        is_gaussian = actor_is_gaussian[actor_idx]
        is_stoch = actor_is_stochastic[actor_idx]
        
        @jax.jit
        def _action_fn_det(params: FrozenDict, obs: jax.Array) -> jax.Array:
            if actor_type == "flow":
                # FQLFlowPolicy: noise=0으로 one-step deterministic
                noise_zero = jnp.zeros((obs.shape[0], actor_module.action_dim), dtype=obs.dtype)
                return actor_module.apply(params, obs, noise_zero, times=None, use_onestep=True)
            elif is_gaussian:
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
        
        @jax.jit
        def _action_fn_stoch(params: FrozenDict, obs: jax.Array, key: jax.random.PRNGKey) -> jax.Array:
            if actor_type == "flow":
                # FQLFlowPolicy: noise ~ N(0,I) for stochastic evaluation
                noise = jax.random.normal(key, (obs.shape[0], actor_module.action_dim))
                return actor_module.apply(params, obs, noise, times=None, use_onestep=True)
            elif actor_type == "gaussian":
                # GaussianMLP: sample from distribution (mean, log_std) -> (mean, std)
                mean, log_std = actor_module.apply(params, obs)
                std = jnp.exp(log_std)
                action = mean + std * jax.random.normal(key, mean.shape)
                return jnp.clip(action, -actor_module.max_action, actor_module.max_action)
            elif actor_type == "tanh_gaussian":
                # TanhGaussianMLP: unbounded Gaussian 샘플링 후 tanh 적용
                mean, log_std = actor_module.apply(params, obs)
                std = jnp.exp(log_std)
                action_unbounded = mean + std * jax.random.normal(key, mean.shape)
                return jnp.tanh(action_unbounded) * actor_module.max_action
            elif is_stoch:
                # StochasticMLP: sample z
                z = jax.random.normal(key, (obs.shape[0], actor_module.action_dim))
                return actor_module.apply(params, obs, z)
            else:
                # DeterministicMLP: same as deterministic
                return actor_module.apply(params, obs)
        
        return _action_fn_det if deterministic else _action_fn_stoch

    print("---------------------------------------")
    print(f"Training POGO Multi-Actor (JAX), Algorithm: {config.algorithm.upper()}")
    print(f"  Env: {config.dataset_name}, Seed: {config.train_seed}")
    print(f"  Actors: {config.num_actors}, W2 weights (Actor1+): {config.w2_weights}")
    print("  Actor0: 원래 알고리즘 loss만 사용")
    print("  Actor1+: 알고리즘 loss + W2 distance")
    print("    - Gaussian actors: Closed form W2 (||μ1-μ2||² + ||σ1-σ2||²)")
    print("    - TanhGaussian/Stochastic actors: Sinkhorn distance")
    print("    - Stochastic (non-Gaussian): Sinkhorn distance")
    print("    - Deterministic: L2 distance")
    print(f"  Actor types: {[config.actor_configs[i].get('type', 'deterministic') for i in range(config.num_actors)]}")
    if config.algorithm == "fql":
        print(f"  FQL settings: alpha={config.fql_alpha}, flow_steps={config.fql_flow_steps}, q_agg={config.fql_q_agg}")
    print("---------------------------------------")

    # 로그 저장을 위한 초기화
    import json
    import datetime
    
    # 전역 카운터 (gradient update step 기준, pogogo.py와 동일)
    global_update_step = 0  # gradient update count
    
    train_logs = []
    eval_logs = []
    evaluations = {i: [] for i in range(config.num_actors)}
    last_eval_step = 0
    
    updates_per_epoch = config.num_updates_on_epoch
    max_updates = config.num_epochs * updates_per_epoch
    
    for epoch in range(config.num_epochs):
        # Python for loop로 배치 샘플링 + 업데이트
        # buffer는 Python 객체이므로 fori_loop 밖에서 처리
        for i in range(config.num_updates_on_epoch):
            # 배치 샘플링 (Python side)
            current_key, batch_key = jax.random.split(current_key)
            batch = buffer.sample_batch(batch_key, batch_size=config.batch_size)
            
            # 업데이트 (JAX side)
            # 매 업데이트마다 metrics를 새로 생성하여 현재 배치의 메트릭만 계산
            temp_metrics = Metrics.create(bc_metrics_to_log)
            current_key, current_actors, current_critic, temp_metrics = update_multi_actor_partial(
                key=current_key,
                actors=current_actors,
                critic=current_critic,
                batch=batch,
                metrics=temp_metrics,
            )
            
            # 현재 update step 계산 (pogogo.py와 동일: gradient update step 기준)
            global_update_step += 1
            
            # PyTorch 버전과 동일하게 매 update마다 로깅
            train_out = temp_metrics.compute()
            
            # wandb 로깅: 학습 메트릭 (pogogo.py 형식)
            if config.use_wandb and (global_update_step % config.log_interval == 0):
                log_dict = {f"train/{k}": float(v.item()) if isinstance(v, (np.ndarray, jnp.ndarray)) else float(v) 
                           for k, v in train_out.items() 
                           if k != "timestep" and isinstance(v, (int, float, np.floating, np.ndarray, jnp.ndarray))}
                log_dict["train/global_step"] = global_update_step
                wandb.log(log_dict, step=global_update_step)
            
            # 공통 메타 필드 부착 (파일 저장용)
            train_log = {
                "update_step": int(global_update_step),
                "epoch": int(epoch),
                **{k: float(v.item()) if isinstance(v, (np.ndarray, jnp.ndarray)) else float(v) 
                   for k, v in train_out.items() 
                   if k != "timestep" and isinstance(v, (int, float, np.floating, np.ndarray, jnp.ndarray))},
            }
            
            # 내부 timestep 키 제거/통일
            train_log.pop("timestep", None)
            
            # train 로그 저장/출력
            if config.save_train_logs and (global_update_step % config.log_interval == 0):
                train_logs.append(train_log)
            
            # 체크포인트
            if global_update_step % config.checkpoint_freq == 0:
                # TODO: 체크포인트 저장 구현
                pass
            
            # 진행 상황 출력 (pogogo.py와 동일하게 update step 기준)
            if global_update_step % 10000 == 0 or global_update_step >= max_updates:
                print(f"Update steps: {global_update_step}", flush=True)

            # 평가 (pogogo.py와 동일: gradient update step 기준)
            if (global_update_step % config.eval_freq == 0) or global_update_step >= max_updates:
                print(f"[Training] Time steps: {global_update_step}", flush=True)
                # Evaluate each actor (deterministic and stochastic)
                actor_results = []
                for actor_idx in range(config.num_actors):
                    action_fn_det = make_actor_action_fn(actor_idx, deterministic=True)
                    action_fn_stoch = make_actor_action_fn(actor_idx, deterministic=False)
                    
                    det_avg, det_score, stoch_avg, stoch_score = eval_policy_jax(
                        current_actors[actor_idx].params,
                        action_fn_det,
                        action_fn_stoch,
                        eval_env,
                        base_seed=config.eval_seed + actor_idx * 100,
                        eval_episodes=config.eval_episodes,
                    )
                    
                    actor_results.append({
                        'det_avg': det_avg,
                        'det_score': det_score,
                        'stoch_avg': stoch_avg,
                        'stoch_score': stoch_score
                    })
                    
                    evaluations[actor_idx].append(det_score)
                    
                    e = {
                        "update_step": int(global_update_step),
                        "actor": int(actor_idx),
                        "det_score": det_score,
                        "det_avg": det_avg,
                        "stoch_score": stoch_score,
                        "stoch_avg": stoch_avg,
                    }
                    eval_logs.append(e)
                
                # 평가 결과 출력 (pogogo.py 형식)
                print("---------------------------------------", flush=True)
                print("Evaluation over 10 episodes:", flush=True)
                for i in range(config.num_actors):
                    r = actor_results[i]
                    print(f"  Actor {i} - Deterministic: {r['det_avg']:.3f}, D4RL score: {r['det_score']:.3f}", flush=True)
                    print(f"  Actor {i} - Stochastic: {r['stoch_avg']:.3f}, D4RL score: {r['stoch_score']:.3f}", flush=True)
                print("---------------------------------------", flush=True)
                
                # wandb 로깅: 평가 메트릭 (pogogo.py 형식)
                if config.use_wandb:
                    eval_log_dict = {}
                    for i in range(config.num_actors):
                        r = actor_results[i]
                        eval_log_dict[f"eval/actor_{i}/det_score"] = r['det_score']
                        eval_log_dict[f"eval/actor_{i}/det_avg"] = r['det_avg']
                        eval_log_dict[f"eval/actor_{i}/stoch_score"] = r['stoch_score']
                        eval_log_dict[f"eval/actor_{i}/stoch_avg"] = r['stoch_avg']
                    eval_log_dict["eval/global_step"] = global_update_step
                    wandb.log(eval_log_dict, step=global_update_step)
                
                last_eval_step = global_update_step
    
    # 학습 완료 후 최종 평가 (pogogo.py 형식)
    print("\n" + "=" * 60, flush=True)
    print("Training completed!", flush=True)
    print("=" * 60, flush=True)
    
    # 최종 평가: 각 actor마다 deterministic과 stochastic 평가
    for i in range(config.num_actors):
        print(f"\n======== Final Evaluation: Actor {i} ========", flush=True)
        action_fn_det = make_actor_action_fn(i, deterministic=True)
        action_fn_stoch = make_actor_action_fn(i, deterministic=False)
        
        det_scores, stoch_scores = [], []
        for r in range(5):  # 5 runs
            _, det_score, _, stoch_score = eval_policy_jax(
                current_actors[i].params,
                action_fn_det,
                action_fn_stoch,
                eval_env,
                base_seed=1000 + 100 * r + i * 1000,
                eval_episodes=config.eval_episodes,
            )
            det_scores.append(float(det_score))
            stoch_scores.append(float(stoch_score))
        
        det_scores = np.array(det_scores, dtype=np.float32)
        stoch_scores = np.array(stoch_scores, dtype=np.float32)
        
        print(f"[FINAL] Deterministic: mean={det_scores.mean():.3f}, std={det_scores.std():.3f} over 5x{config.eval_episodes}", flush=True)
        print(f"[FINAL] Stochastic:   mean={stoch_scores.mean():.3f}, std={stoch_scores.std():.3f} over 5x{config.eval_episodes}", flush=True)
        
        # wandb 로깅: 최종 평가 결과 (pogogo.py 형식)
        if config.use_wandb:
            actor_suffix = f"_actor_{i}" if i is not None else ""
            wandb.log({
                f"final{actor_suffix}/det_mean": float(det_scores.mean()),
                f"final{actor_suffix}/det_std": float(det_scores.std()),
                f"final{actor_suffix}/stoch_mean": float(stoch_scores.mean()),
                f"final{actor_suffix}/stoch_std": float(stoch_scores.std()),
            })
        
        if len(evaluations[i]) > 0:
            print(f"Actor {i}: Final={evaluations[i][-1]:.1f}, Best={max(evaluations[i]):.1f}", flush=True)
    
    # 로그 저장: results/{algorithm}/{env}/seed_{seed}/logs/
    env_name = config.dataset_name.replace("-", "_")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_dir = os.path.join("results", config.algorithm, env_name, f"seed_{config.train_seed}", "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    train_log_file = os.path.join(
        log_dir, f"train_{config.algorithm}_{env_name}_seed{config.train_seed}_{timestamp}.jsonl"
    )
    eval_log_file = os.path.join(
        log_dir, f"eval_{config.algorithm}_{env_name}_seed{config.train_seed}_{timestamp}.jsonl"
    )
    summary_file = os.path.join(
        log_dir, f"summary_{config.algorithm}_{env_name}_seed{config.train_seed}_{timestamp}.json"
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
        "env": config.dataset_name,
        "seed": int(config.train_seed),
        "num_actors": int(config.num_actors),
        "w2_weights": [float(w) for w in config.w2_weights],
        "max_updates": int(max_updates),
        "log_interval": int(config.log_interval),
        "eval_freq": int(config.eval_freq),
        "checkpoint_freq": int(config.checkpoint_freq),
        "final_update_step": int(global_update_step),
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


if __name__ == "__main__":
    main()
