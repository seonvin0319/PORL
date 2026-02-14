# POGO Multi-Actor 공통 유틸리티
# AlgorithmInterface와 관련 클래스들을 정의

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple
import os
import random

import gym
import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict
from flax import linen as nn

import d4rl  # noqa

# 순환 참조 방지를 위해 타입 힌트만 사용
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .rebrac import CriticTrainState, ActorTrainState, Metrics


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
        actor: "ActorTrainState",
        critic: "CriticTrainState",
        batch: Dict[str, jax.Array],
        actor_module: Optional[nn.Module] = None,
        **kwargs
    ) -> Tuple[jax.random.PRNGKey, "CriticTrainState", "Metrics"]:
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
        critic: "CriticTrainState",
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


# ReBRACAlgorithm은 algorithms.offline.rebrac에 정의되어 있음
# 순환 참조 방지를 위해 여기서는 import하지 않음


# ============================================================================
# 환경 및 평가 유틸리티 (PyTorch의 utils_pytorch.py와 대응)
# ============================================================================

def set_global_seed(seed: int) -> jax.Array:
    """Seed Python/NumPy and return the base JAX PRNG key."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return jax.random.PRNGKey(seed)


def make_eval_seeds(seed: int, n_episodes: int):
    """Return deterministic eval seeds for each episode."""
    return [seed + 10000 + ep for ep in range(n_episodes)]


def seed_env_spaces(env: gym.Env, seed: int) -> None:
    """Seed env, action space, and observation space where supported."""
    if hasattr(env, "seed"):
        try:
            env.seed(seed)
        except Exception:
            pass

    if hasattr(env, "action_space") and hasattr(env.action_space, "seed"):
        try:
            env.action_space.seed(seed)
        except Exception:
            pass

    if hasattr(env, "observation_space") and hasattr(env.observation_space, "seed"):
        try:
            env.observation_space.seed(seed)
        except Exception:
            pass


def reset_env_with_seed(env: gym.Env, seed: int):
    """Reset env with a deterministic seed, supporting old/new gym APIs."""
    try:
        return env.reset(seed=seed)
    except (TypeError, AttributeError):
        seed_env_spaces(env, seed)
        return env.reset()

def qlearning_dataset(
    env: gym.Env,
    dataset: Dict = None,
    terminate_on_end: bool = False,
    **kwargs,
) -> Dict:
    """D4RL qlearning dataset 로딩"""
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    N = dataset["rewards"].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    next_action_ = []
    reward_ = []
    done_ = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = "timeouts" in dataset

    episode_step = 0
    for i in range(N - 1):
        obs = dataset["observations"][i].astype(np.float32)
        new_obs = dataset["observations"][i + 1].astype(np.float32)
        action = dataset["actions"][i].astype(np.float32)
        new_action = dataset["actions"][i + 1].astype(np.float32)
        reward = dataset["rewards"][i].astype(np.float32)
        done_bool = bool(dataset["terminals"][i])

        if use_timeouts:
            final_timestep = dataset["timeouts"][i]
        else:
            final_timestep = episode_step == env._max_episode_steps - 1
        if (not terminate_on_end) and final_timestep:
            # Skip this transition
            episode_step = 0
            continue
        if done_bool or final_timestep:
            episode_step = 0

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        next_action_.append(new_action)
        reward_.append(reward)
        done_.append(done_bool)
        episode_step += 1

    return {
        "observations": np.array(obs_),
        "actions": np.array(action_),
        "next_observations": np.array(next_obs_),
        "next_actions": np.array(next_action_),
        "rewards": np.array(reward_),
        "terminals": np.array(done_),
    }


def wrap_env(
    env: gym.Env,
    state_mean: float = 0.0,
    state_std: float = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    """환경을 normalization wrapper로 감싸기 (PyTorch의 wrap_env와 동일)"""
    def normalize_state(state: np.ndarray) -> np.ndarray:
        return (state - state_mean) / state_std  # epsilon should be already added in std.

    def scale_reward(reward: float) -> float:
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


def evaluate(
    env: gym.Env,
    params: jax.Array,
    action_fn: Callable,
    num_episodes: int,
    seed: int,
) -> np.ndarray:
    """Evaluate policy over multiple episodes (PyTorch의 eval_actor와 대응)
    
    Args:
        env: Gym environment
        params: Policy parameters
        action_fn: Action function (params, obs) -> action
        num_episodes: Number of episodes to evaluate
        seed: Base seed for evaluation
    
    Returns:
        Array of episode returns [num_episodes]
    """
    returns = []
    eval_seeds = make_eval_seeds(seed, num_episodes)
    for ep_seed in eval_seeds:
        # antmaze 등 reset 시 전역 np.random을 참조하는 환경이 있어 고정
        np.random.seed(ep_seed)
        seed_env_spaces(env, ep_seed)
        reset_result = reset_env_with_seed(env, ep_seed)
        
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        done = False
        total_reward = 0.0
        while not done:
            action = np.asarray(jax.device_get(action_fn(params, obs)))
            step_out = env.step(action)
            if len(step_out) == 5:
                obs, reward, terminated, truncated, _ = step_out
                done = terminated or truncated
            else:
                obs, reward, done, _ = step_out
            total_reward += reward
        returns.append(total_reward)

    return np.array(returns)
