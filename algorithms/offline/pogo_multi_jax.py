# POGO Multi-Actor (JAX implementation)
# JAX 기반 알고리즘에 POGO Multi-Actor 구조 적용
# - Actor0: 원래 알고리즘 loss만 사용 (W2 penalty 없음)
# - Actor1+: 원래 알고리즘 loss + W2 distance to previous actor

import os

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # For reproducibility

import math
import pickle
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
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
# JAX utilities
from .utils_jax import (
    ActorConfig,
    ActorTrainState,
    AlgorithmInterface,
    CriticTrainState,
    Metrics,
    ReplayBuffer,
    W2DistanceCalculator,
    compute_mean_std,
    evaluate,
    identity,
    normalize_states,
    pytorch_init,
    qlearning_dataset,
    uniform_init,
    wrap_env,
)
# ReBRAC imports
from .rebrac import (
    Config as BaseConfig,
    EnsembleCritic,
    ReBRACAlgorithm,
)
# FQL imports
from .fql import FQLAlgorithm
from algorithms.networks.actors_jax import ActorVectorField
# JAX Actor implementations
from algorithms.networks.actors_jax import (
    GaussianMLP,
    TanhGaussianMLP,
    StochasticMLP,
    DeterministicMLP,
)


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
    
    # Wandb 설정
    use_wandb: bool = True  # wandb 사용 여부
    
    # 알고리즘 타입
    algorithm: str = "rebrac"  # "rebrac" or "fql"
    
    # FQL 파라미터 (FQL 사용 시)
    q_agg: str = "mean"  # Q value aggregation: "mean" or "min"
    normalize_q_loss: bool = False  # Whether to normalize Q loss
    alpha: float = 10.0  # Distillation loss coefficient
    flow_steps: int = 10  # Number of flow steps for BC flow
    
    def __post_init__(self):
        super().__post_init__()
        
        # train_seed와 eval_seed가 None이면 기본값 설정 (config 파일에서 None/null로 설정된 경우 대비)
        if self.train_seed is None:
            self.train_seed = 0
        if self.eval_seed is None:
            self.eval_seed = 42
        
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


def update_multi_actor(
    key: jax.random.PRNGKey,
    actors: List[ActorTrainState],
    critic: CriticTrainState,
    batch: Dict[str, jax.Array],
    metrics: Metrics,
    algorithm: AlgorithmInterface,
    actor_modules: List[nn.Module],
    actor_is_gaussian: List[bool],
    actor_is_stochastic: List[bool],
    w2_weights: List[float],
    w2_calculator: W2DistanceCalculator,
    gamma: float,
    tau: float,
    total_steps: int = 0,
    policy_freq: int = 1,
    flow_policy_state: Optional[ActorTrainState] = None,  # FQL용
    **kwargs,
) -> Tuple[jax.random.PRNGKey, List[ActorTrainState], CriticTrainState, Metrics, Optional[ActorTrainState]]:

    num_actors = len(actors)

    # 1) Critic 업데이트는 매 스텝 (원본 ReBRAC과 동일)
    # critic_bc_coef는 algorithm 객체에 이미 저장되어 있으므로 전달 불필요
    key, new_critic, new_metrics = algorithm.update_critic(
        key,
        actors[0],
        critic,
        batch,
        actor_module=actor_modules[0],
        metrics=metrics if metrics is not None else Metrics.create([]),
    )
    # 2) Actor 및 Target 업데이트는 policy_freq 주기로만 실행
    should_update = (total_steps % policy_freq) == 0

    def do_policy_update(carry):
        """policy_freq 주기: Actor(0..end) 업데이트 + 모든 Target 네트워크 업데이트"""
        key, actors, critic, metrics, flow_policy_state = carry

        updated_key = key
        updated_metrics = metrics
        updated_actors = [None] * num_actors

        # ---------- Actor0 업데이트 ----------
        actor0 = actors[0]
        actor0_module = actor_modules[0]
        updated_key, actor0_key = jax.random.split(updated_key)
        
        updated_key, new_actor0, new_flow_policy_state, updated_metrics = algorithm.update_actor0(
            updated_key,
            actor0,
            actor0_module,
            critic,
            batch,
            flow_policy_state=flow_policy_state,
            tau=tau,
            metrics=updated_metrics,
        )
        
        new_actor0 = new_actor0.replace(
            target_params=optax.incremental_update(new_actor0.params, actor0.target_params, tau)
        )
        
        if new_flow_policy_state is not None:
            new_flow_policy_state = new_flow_policy_state.replace(
                target_params=optax.incremental_update(
                    new_flow_policy_state.params, flow_policy_state.target_params, tau
                )
            )
            flow_policy_state = new_flow_policy_state
        
        updated_actors[0] = new_actor0

        # ---------- Actor1+ 업데이트 ----------
        actor0_params_for_1plus = new_actor0.params
        actor0_module_for_1plus = actor_modules[0]
        
        for i in range(1, num_actors):
            actor_i = actors[i]
            actor_module_i = actor_modules[i]
            ref_actor = updated_actors[i - 1]
            ref_actor_module = actor_modules[i - 1]
            w2_weight_i = w2_weights[i - 1]

            updated_key, energy_key, w2_key = jax.random.split(updated_key, 3)

            def actor_i_loss_fn(params: FrozenDict) -> Tuple[jax.Array, Metrics]:
                # energy (Actor1+)
                energy = algorithm.compute_energy_function(
                    params,
                    actor_module_i,
                    critic,      # new_critic를 넣어야 함
                    batch,
                    key=energy_key,  # Sampling 기반 energy를 위한 key 전달
                    is_stochastic=actor_is_stochastic[i],  # Actor 타입 정보 전달
                    actor0_params=actor0_params_for_1plus,  # FQL용: Actor0 params 전달
                    actor0_module=actor0_module_for_1plus,   # FQL용: Actor0 module 전달
                )

                # w2 (기존 로직 유지)
                if actor_is_gaussian[i] and actor_is_gaussian[i - 1]:
                    mean_i, std_i = actor_module_i.get_mean_std(params, batch["states"])
                    mean_ref, std_ref = ref_actor_module.get_mean_std(ref_actor.params, batch["states"])
                    mean_ref = jax.lax.stop_gradient(mean_ref)
                    std_ref = jax.lax.stop_gradient(std_ref)
                    w2_dist = w2_calculator.closed_form_w2_gaussian(mean_i, std_i, mean_ref, std_ref).mean()

                elif actor_is_stochastic[i] and actor_is_stochastic[i - 1]:
                    key_w2, _ = jax.random.split(w2_key)
                    actor_i_config = ActorConfig(params=params, module=actor_module_i, is_stochastic=True, is_gaussian=False)
                    ref_actor_config = ActorConfig(params=ref_actor.params, module=ref_actor_module, is_stochastic=True, is_gaussian=False)
                    w2_dist = w2_calculator.compute_distance(
                        actor_i_config=actor_i_config,
                        ref_actor_config=ref_actor_config,
                        states=batch["states"],
                        key=key_w2,
                    )
                else:
                    # deterministic/L2 fallback
                    if actor_is_gaussian[i]:
                        actions_i, _ = actor_module_i.get_mean_std(params, batch["states"])
                    elif actor_is_stochastic[i]:
                        actions_i = actor_module_i.deterministic_actions(params, batch["states"])
                    else:
                        actions_i = actor_module_i.apply(params, batch["states"])

                    if actor_is_gaussian[i - 1]:
                        actions_ref, _ = ref_actor_module.get_mean_std(ref_actor.params, batch["states"])
                    elif actor_is_stochastic[i - 1]:
                        actions_ref = ref_actor_module.deterministic_actions(ref_actor.params, batch["states"])
                    else:
                        actions_ref = ref_actor_module.apply(ref_actor.params, batch["states"])

                    actions_ref = jax.lax.stop_gradient(actions_ref)
                    w2_dist = ((actions_i - actions_ref) ** 2).sum(-1).mean()

                loss = energy + w2_weight_i * w2_dist
                m = updated_metrics.update({f"actor_{i}_loss": loss, f"w2_{i}_distance": w2_dist})
                return loss, m

            (loss_i, updated_metrics), grads_i = jax.value_and_grad(actor_i_loss_fn, has_aux=True)(actor_i.params)
            new_actor_i = actor_i.apply_gradients(grads=grads_i)
            new_actor_i = new_actor_i.replace(
                target_params=optax.incremental_update(new_actor_i.params, actor_i.target_params, tau)
            )
            updated_actors[i] = new_actor_i

        # ---------- Critic target update도 policy update 시점에만 ----------
        # Note: critic은 do_policy_update carry로 들어온 값 (바깥에서 이미 new_critic로 업데이트됨)
        # target_params만 업데이트 (TD3 스타일: policy update 시점에만)
        new_critic2 = critic.replace(
            params=critic.params,  # 현재 critic.params는 이미 업데이트된 값
            target_params=optax.incremental_update(critic.params, critic.target_params, tau),
        )

        return updated_key, tuple(updated_actors), new_critic2, updated_metrics, flow_policy_state

    def skip_policy_update(carry):
        """non-policy step: actor/targets는 그대로, critic은 이미 업데이트된 params만 유지"""
        key, actors, critic, metrics, flow_policy_state = carry
        # 원본 ReBRAC처럼: 이 스텝에서는 target들을 건드리지 않음
        return key, tuple(actors), critic, metrics, flow_policy_state

    # cond 입력 carry: (key, actors, new_critic, new_metrics, flow_policy_state)
    key, new_actors, new_critic_final, new_metrics_final, new_flow_policy_state = jax.lax.cond(
        should_update,
        do_policy_update,
        skip_policy_update,
        (key, actors, new_critic, new_metrics, flow_policy_state),
    )

    return key, list(new_actors), new_critic_final, new_metrics_final, new_flow_policy_state


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
def main(config: Config):
    """POGO Multi-Actor 메인 함수 (JAX)"""
    dict_config = asdict(config)
    dict_config["mlc_job_name"] = os.environ.get("PLATFORM_JOB_NAME")

    if config.use_wandb:
        wandb.init(
            config=dict_config,
            project=config.project,
            group=config.group,
            name=config.name,
            id=str(uuid.uuid4()),
        )
        wandb.mark_preempting()
    else:
        print("[Wandb] 비활성화됨 (--no_wandb 또는 use_wandb: false)", flush=True)
    
    buffer = ReplayBuffer()
    buffer.create_from_d4rl(
        config.dataset_name, config.normalize_reward, config.normalize_states
    )

    key = jax.random.PRNGKey(seed=config.train_seed)
    keys = jax.random.split(key, config.num_actors + 2)
    actor_keys = keys[:config.num_actors]
    critic_key = keys[config.num_actors]
    key = keys[config.num_actors + 1]  # 다음 split을 위한 key

    eval_env = gym.make(config.dataset_name)
    eval_env.seed(config.eval_seed)
    eval_env.action_space.seed(config.eval_seed)
    eval_env.observation_space.seed(config.eval_seed)
    eval_env = wrap_env(eval_env, buffer.mean, buffer.std)
    init_state = buffer.data["states"][0][None, ...]
    init_action = buffer.data["actions"][0][None, ...]
    
    action_dim = init_action.shape[-1]
    max_action = float(eval_env.action_space.high[0])

    # 알고리즘 판단: config.algorithm으로 직접 지정
    is_fql = config.algorithm.lower() == "fql"
    
    # Create multiple actors
    actors = []
    actor_modules = []
    actor_is_stochastic = []
    actor_is_gaussian = []
    
    # Actor0 생성
    if is_fql:
        # FQL의 경우: StochasticMLP 사용 (GELU activation)
        actor0_module = StochasticMLP(
            action_dim=action_dim,
            max_action=max_action,
            hidden_dim=config.hidden_dim,
            layernorm=config.actor_ln,
            n_hiddens=config.actor_n_hiddens,
            activation="gelu",  # FQL: GELU activation
        )
        # StochasticMLP 초기화: state와 z 필요
        init_z = jnp.zeros((1, action_dim), dtype=init_state.dtype)
        actor0_params = actor0_module.init(actor_keys[0], init_state, init_z)
        actor0_target_params = actor0_module.init(actor_keys[0], init_state, init_z)
    else:
        # ReBRAC의 경우: DeterministicMLP 사용
        actor0_module = DeterministicMLP(
            action_dim=action_dim,
            max_action=max_action,
            hidden_dim=config.hidden_dim,
            layernorm=config.actor_ln,
            n_hiddens=config.actor_n_hiddens,
        )
        actor0_params = actor0_module.init(actor_keys[0], init_state)
        actor0_target_params = actor0_module.init(actor_keys[0], init_state)
    
    actor0_is_stochastic = getattr(actor0_module, 'is_stochastic', False)
    actor0_is_gaussian = getattr(actor0_module, 'is_gaussian', False)
    
    # Actor0 타입 추론
    if isinstance(actor0_module, GaussianMLP):
        actor0_type = "gaussian"
    elif isinstance(actor0_module, TanhGaussianMLP):
        actor0_type = "tanh_gaussian"
    elif isinstance(actor0_module, StochasticMLP):
        actor0_type = "stochastic"
    else:
        actor0_type = "deterministic"
    
    # actor_configs가 None이거나 부족하면 Actor0 타입으로 채우기
    if config.actor_configs is None:
        config.actor_configs = [{"type": actor0_type} for _ in range(config.num_actors)]
    else:
        # 부족한 경우 Actor0 타입으로 채우기
        while len(config.actor_configs) < config.num_actors:
            config.actor_configs.append({"type": actor0_type})
        config.actor_configs = config.actor_configs[:config.num_actors]
    
    # Actor0의 타입을 확실히 설정 (config.actor_configs[0]에 명시적으로 설정)
    config.actor_configs[0]["type"] = actor0_type
    
    # Actor0 추가
    actor0 = ActorTrainState.create(
        apply_fn=actor0_module.apply,
        params=actor0_params,
        target_params=actor0_target_params,
        tx=optax.adam(learning_rate=config.actor_learning_rate),
    )
    actors.append(actor0)
    actor_modules.append(actor0_module)
    actor_is_stochastic.append(actor0_is_stochastic)
    actor_is_gaussian.append(actor0_is_gaussian)
    
    # Actor1+ 생성
    for i in range(1, config.num_actors):
        actor_config = config.actor_configs[i]
        actor_type = actor_config.get("type", actor0_type)  # 기본값은 Actor0 타입
        
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
        
        # Actor1+이고 Actor0와 같은 타입이면 params 복사 (PyTorch 버전과 동일)
        if i > 0 and actor_type == actor0_type:
            # 같은 타입이면 Actor0의 params 복사
            try:
                actor0_params = actors[0].params
                # FrozenDict를 unfreeze()로 변환 (FrozenDict인 경우만)
                if isinstance(actor0_params, FrozenDict):
                    actor0_dict = actor0_params.unfreeze()
                else:
                    actor0_dict = dict(actor0_params)
                
                if isinstance(init_params, FrozenDict):
                    init_dict = init_params.unfreeze()
                else:
                    init_dict = dict(init_params)
                
                # 공통 키만 복사 (shape이 같은 경우만)
                def copy_params_recursive(src_dict, tgt_dict):
                    """Recursively copy params with shape checking"""
                    result = {}
                    for key in tgt_dict.keys():
                        if key in src_dict:
                            src_val = src_dict[key]
                            tgt_val = tgt_dict[key]
                            
                            if isinstance(tgt_val, dict) and isinstance(src_val, dict):
                                # Nested dict (예: params['Dense_0']['kernel'])
                                result[key] = copy_params_recursive(src_val, tgt_val)
                            elif hasattr(tgt_val, 'shape') and hasattr(src_val, 'shape'):
                                # JAX array - check shape
                                if src_val.shape == tgt_val.shape:
                                    result[key] = src_val
                                else:
                                    result[key] = tgt_val
                            else:
                                result[key] = tgt_val
                        else:
                            result[key] = tgt_val
                    return result
                
                new_params_dict = copy_params_recursive(actor0_dict, init_dict)
                init_params = FrozenDict(new_params_dict)
                init_target_params = FrozenDict(new_params_dict)
            except Exception as e:
                # 복사 실패 시 초기화된 params 사용
                print(f"Warning: Failed to copy Actor0 params to Actor{i}: {e}", flush=True)
        
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

    # 알고리즘 객체 생성
    if is_fql:
        # Flow policy 생성 (ActorVectorField, actor_bc_flow 역할)
        flow_policy = ActorVectorField(
            hidden_dim=config.hidden_dim,
            action_dim=action_dim,
            n_hiddens=config.actor_n_hiddens,
            layer_norm=config.actor_ln,
            activation="gelu",  # FQL: GELU activation
        )
        # Flow policy 초기화: state, action, time 필요
        init_noise_flow = jnp.zeros((1, action_dim), dtype=init_state.dtype)
        init_time_flow = jnp.zeros((1, 1), dtype=init_state.dtype)
        flow_policy_params = flow_policy.init(actor_keys[0], init_state, init_noise_flow, init_time_flow)
        flow_policy_target_params = flow_policy.init(actor_keys[0], init_state, init_noise_flow, init_time_flow)
        
        # Flow policy train state 생성
        flow_policy_state = ActorTrainState.create(
            apply_fn=flow_policy.apply,
            params=flow_policy_params,
            target_params=flow_policy_target_params,
            tx=optax.adam(learning_rate=config.actor_learning_rate),
        )
        
        algorithm = FQLAlgorithm(
            gamma=config.gamma,
            tau=config.tau,
            q_agg=config.q_agg,
            normalize_q_loss=config.normalize_q_loss,
            alpha=config.alpha,
            flow_steps=config.flow_steps,
            flow_policy=flow_policy,
        )
    else:
        flow_policy_state = None
        algorithm = ReBRACAlgorithm(
            actor_bc_coef=config.actor_bc_coef,  # actor loss에 사용
            critic_bc_coef=config.critic_bc_coef,  # critic 업데이트에 사용
            gamma=config.gamma,
            tau=config.tau,
            policy_noise=config.policy_noise,
            noise_clip=config.noise_clip,
            normalize_q=config.normalize_q,
        )
    
    # W2 Distance 계산기 생성
    w2_calculator = W2DistanceCalculator(
        sinkhorn_blur=config.sinkhorn_blur,
        sinkhorn_K=config.sinkhorn_K,
    )
    update_multi_actor_partial = partial(
        update_multi_actor,
        algorithm=algorithm,
        actor_modules=actor_modules,
        actor_is_gaussian=actor_is_gaussian,
        actor_is_stochastic=actor_is_stochastic,
        w2_weights=config.w2_weights,
        w2_calculator=w2_calculator,
        gamma=config.gamma,
        tau=config.tau,
        policy_freq=getattr(config, "policy_freq", 1),
    )

    def loop_update_step(i: int, carry: Dict) -> Dict:
        key, batch_key = jax.random.split(carry["key"])
        batch = carry["buffer"].sample_batch(batch_key, batch_size=config.batch_size)

        key, new_actors, new_critic, new_metrics, new_flow_policy_state = update_multi_actor_partial(
            key=key,
            actors=carry["actors"],
            critic=carry["critic"],
            batch=batch,
            metrics=carry["metrics"],
            total_steps=carry["total_steps"],
            flow_policy_state=carry.get("flow_policy_state", None),  # FQL용: carry에서 최신 state 전달
        )

        # total_steps 증가
        total_steps = carry["total_steps"] + 1

        carry.update({
            "key": key,
            "actors": new_actors,
            "critic": new_critic,
            "metrics": new_metrics,
            "total_steps": total_steps,
            "flow_policy_state": new_flow_policy_state,  # FQL용
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
        "total_steps": 0,  # 전체 update step 수 추적
        "flow_policy_state": flow_policy_state,  # FQL용: 초기 flow_policy_state 설정
    }

    def make_actor_action_fn(actor_idx: int):
        """Create action function for a specific actor"""
        actor_module = actor_modules[actor_idx]
        is_gaussian = actor_is_gaussian[actor_idx]
        is_stoch = actor_is_stochastic[actor_idx]
        
        # StochasticMLP의 경우 key를 내부에서 생성 (evaluation용)
        if is_stoch:
            base_key = jax.random.PRNGKey(42)  # Evaluation용 base key
            
            @jax.jit
            def _action_fn(params: FrozenDict, obs: jax.Array) -> jax.Array:
                # obs가 1D인 경우 2D로 변환
                if obs.ndim == 1:
                    obs = obs[None, :]  # [1, state_dim]
                    squeeze_output = True
                else:
                    squeeze_output = False
                
                # StochasticMLP: z를 샘플링 (one-step BC policy)
                # obs 기반으로 key 생성 (deterministic evaluation을 위해)
                obs_hash = jnp.sum(obs).astype(jnp.uint32)
                key = jax.random.fold_in(base_key, obs_hash)
                z = jax.random.normal(key, (obs.shape[0], actor_module.action_dim))
                actions = actor_module.apply(params, obs, z)
                
                if squeeze_output:
                    actions = actions.squeeze(0)  # [action_dim]
                return actions
        else:
            @jax.jit
            def _action_fn(params: FrozenDict, obs: jax.Array) -> jax.Array:
                # obs가 1D인 경우 2D로 변환
                if obs.ndim == 1:
                    obs = obs[None, :]  # [1, state_dim]
                    squeeze_output = True
                else:
                    squeeze_output = False
                
                if is_gaussian:
                    # GaussianMLP: use mean
                    mean, _ = actor_module.apply(params, obs)
                    actions = mean
                else:
                    # DeterministicMLP: state only
                    actions = actor_module.apply(params, obs)
                
                if squeeze_output:
                    actions = actions.squeeze(0)  # [action_dim]
                return actions
        
        return _action_fn

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

    # Checkpoint 저장 디렉토리 미리 생성
    env_name, task_name = _parse_env_name(config.dataset_name)
    checkpoint_dir = os.path.join("results", config.algorithm, env_name, task_name, f"seed_{config.train_seed}", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(carry: Dict[str, Any], timestep: int, suffix: str = "", 
                        evaluations: Optional[Dict[int, List[float]]] = None,
                        metrics: Optional[Dict[str, np.ndarray]] = None):
        """체크포인트 저장 함수"""
        ckpt = {
            "config": asdict(config),
            "actors": [actor.params for actor in carry["actors"]],
            "actor_targets": [actor.target_params for actor in carry["actors"]],
            "critic": carry["critic"].params,
            "total_steps": timestep,
        }
        
        # FQL의 경우 flow_policy_state도 저장
        if is_fql and carry.get("flow_policy_state") is not None:
            ckpt["flow_policy_state"] = carry["flow_policy_state"].params
            ckpt["flow_policy_target"] = carry["flow_policy_state"].target_params
        
        # 평가 결과 저장
        if evaluations is not None:
            ckpt["evaluations"] = {str(k): v for k, v in evaluations.items()}  # JAX array를 list로 변환
        
        # 학습 메트릭 저장
        if metrics is not None:
            ckpt["metrics"] = {k: float(v) if isinstance(v, (np.ndarray, jnp.ndarray)) else v 
                              for k, v in metrics.items()}
        
        checkpoint_file = os.path.join(checkpoint_dir, f"model_step{timestep}{suffix}.pkl")
        with open(checkpoint_file, "wb") as f:
            pickle.dump(ckpt, f)
        print(f"Checkpoint 저장 완료: {checkpoint_file}", flush=True)

    print("---------------------------------------")
    print(f"Training POGO Multi-Actor (JAX), Algorithm: {config.algorithm.upper()}, Env: {config.dataset_name}, Seed: {config.train_seed}")
    print(f"  Actors: {config.num_actors}, W2 weights (Actor1+): {config.w2_weights}")
    print("  Actor0: 원래 알고리즘 loss만 사용")
    print("  Actor1+: Energy Function + W2 distance")
    print("    - Gaussian actors: Closed form W2 (||μ1-μ2||² + ||σ1-σ2||²)")
    print("    - TanhGaussian/Stochastic actors: Sinkhorn distance")
    print("    - Stochastic (non-Gaussian): Sinkhorn distance")
    print("    - Deterministic: L2 distance")
    print(f"  Actor types: {[config.actor_configs[i].get('type', 'deterministic') for i in range(config.num_actors)]}")
    print("---------------------------------------")

    # 평가 결과 저장용 리스트
    evaluations = {i: [] for i in range(config.num_actors)}

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
        total_steps = update_carry["total_steps"]
        
        if config.use_wandb:
            wandb.log(
                {"epoch": epoch, "total_steps": total_steps, **{f"POGO_Multi_JAX/{k}": v for k, v in mean_metrics.items()}}
            )

        # 500k timestep마다 체크포인트 저장
        if total_steps % 500000 == 0 and total_steps > 0:
            save_checkpoint(update_carry, total_steps, "_mid", 
                          evaluations=evaluations, metrics=mean_metrics)

        # eval_freq는 update step 기준 (PyTorch 버전과 동일)
        if total_steps % config.eval_freq == 0 or (epoch == config.num_epochs - 1):
            # Evaluate each actor
            actor_results = []
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
                raw_avg = float(np.mean(eval_returns))
                norm_score = float(np.mean(normalized_score))
                raw_std = float(np.std(eval_returns))
                norm_std = float(np.std(normalized_score))
                
                evaluations[actor_idx].append(norm_score)
                
                actor_results.append({
                    "raw_avg": raw_avg,
                    "raw_std": raw_std,
                    "norm_score": norm_score,
                    "norm_std": norm_std,
                })
                
                # wandb 로깅
                if config.use_wandb:
                    wandb.log(
                        {
                            f"eval/actor_{actor_idx}/score": norm_score,
                            f"eval/actor_{actor_idx}/raw_score": raw_avg,
                            f"eval/actor_{actor_idx}/std": norm_std,
                        },
                        step=total_steps,
                    )
            
            # 평가 결과를 한 번에 출력 (PyTorch 버전과 동일한 형식)
            print("---------------------------------------", flush=True)
            print(f"[Training] Algorithm: {config.algorithm.upper()}, Env: {config.dataset_name}, Time steps: {total_steps}", flush=True)
            print(f"Evaluation over {config.eval_episodes} episodes:", flush=True)
            for actor_idx in range(config.num_actors):
                r = actor_results[actor_idx]
                print(f"  Actor {actor_idx} - Raw: {r['raw_avg']:.3f} ± {r['raw_std']:.3f}, D4RL score: {r['norm_score']:.1f} ± {r['norm_std']:.1f}", flush=True)
            print("---------------------------------------", flush=True)

    # 학습 완료 후 final 체크포인트 저장
    print("\n" + "=" * 60, flush=True)
    print("Training completed! Saving final checkpoint before evaluation...", flush=True)
    print("=" * 60, flush=True)
    final_steps = update_carry["total_steps"]
    final_metrics = update_carry["metrics"].compute() if update_carry.get("metrics") is not None else None
    save_checkpoint(update_carry, final_steps, "_final", 
                  evaluations=evaluations, metrics=final_metrics)
    
    # 학습 완료 후 최종 평가
    print("\n" + "=" * 60, flush=True)
    print("======== Final Evaluation (trained weights) ========", flush=True)
    print("=" * 60, flush=True)
    for actor_idx in range(config.num_actors):
        if evaluations[actor_idx]:
            final_score = evaluations[actor_idx][-1]
            best_score = max(evaluations[actor_idx])
            print(f"[FINAL] Actor {actor_idx}: Final={final_score:.1f}, Best={best_score:.1f}", flush=True)


if __name__ == "__main__":
    _parse_args_with_no_wandb()
    main()
