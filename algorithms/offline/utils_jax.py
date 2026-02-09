# POGO Multi-Actor 공통 유틸리티
# AlgorithmInterface와 관련 클래스들을 정의

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import chex
import d4rl  # noqa
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict
from flax import linen as nn
from flax.training.train_state import TrainState
from ott.geometry import pointcloud
from ott.solvers.linear import solve as sinkhorn_solve


# TrainState classes
class CriticTrainState(TrainState):
    target_params: FrozenDict


class ActorTrainState(TrainState):
    target_params: FrozenDict


# Utility functions
def pytorch_init(fan_in: float) -> Callable:
    """
    Default init for PyTorch Linear layer weights and biases:
    https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    """
    bound = math.sqrt(1 / fan_in)

    def _init(key: jax.random.PRNGKey, shape: Tuple, dtype: type) -> jax.Array:
        return jax.random.uniform(
            key, shape=shape, minval=-bound, maxval=bound, dtype=dtype
        )

    return _init


def uniform_init(bound: float) -> Callable:
    def _init(key: jax.random.PRNGKey, shape: Tuple, dtype: type) -> jax.Array:
        return jax.random.uniform(
            key, shape=shape, minval=-bound, maxval=bound, dtype=dtype
        )

    return _init


def identity(x: Any) -> Any:
    return x


def qlearning_dataset(
    env: gym.Env,
    dataset: Dict = None,
    terminate_on_end: bool = False,
    **kwargs,
) -> Dict:
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


def compute_mean_std(states: jax.Array, eps: float) -> Tuple[jax.Array, jax.Array]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: jax.Array, mean: jax.Array, std: jax.Array) -> jax.Array:
    return (states - mean) / std


@chex.dataclass
class ReplayBuffer:
    data: Dict[str, jax.Array] = None
    mean: float = 0
    std: float = 1

    def create_from_d4rl(
        self,
        dataset_name: str,
        normalize_reward: bool = False,
        is_normalize: bool = False,
    ):
        d4rl_data = qlearning_dataset(gym.make(dataset_name))
        buffer = {
            "states": jnp.asarray(d4rl_data["observations"], dtype=jnp.float32),
            "actions": jnp.asarray(d4rl_data["actions"], dtype=jnp.float32),
            "rewards": jnp.asarray(d4rl_data["rewards"], dtype=jnp.float32),
            "next_states": jnp.asarray(
                d4rl_data["next_observations"], dtype=jnp.float32
            ),
            "next_actions": jnp.asarray(d4rl_data["next_actions"], dtype=jnp.float32),
            "dones": jnp.asarray(d4rl_data["terminals"], dtype=jnp.float32),
        }
        if is_normalize:
            self.mean, self.std = compute_mean_std(buffer["states"], eps=1e-3)
            buffer["states"] = normalize_states(buffer["states"], self.mean, self.std)
            buffer["next_states"] = normalize_states(
                buffer["next_states"], self.mean, self.std
            )
        if normalize_reward:
            buffer["rewards"] = ReplayBuffer.normalize_reward(
                dataset_name, buffer["rewards"]
            )
        self.data = buffer

    @property
    def size(self) -> int:
        # WARN: It will use len of the dataclass, i.e. number of fields.
        return self.data["states"].shape[0]

    def sample_batch(
        self, key: jax.random.PRNGKey, batch_size: int
    ) -> Dict[str, jax.Array]:
        indices = jax.random.randint(
            key, shape=(batch_size,), minval=0, maxval=self.size
        )
        batch = jax.tree.map(lambda arr: arr[indices], self.data)
        return batch

    def get_moments(self, modality: str) -> Tuple[jax.Array, jax.Array]:
        mean = self.data[modality].mean(0)
        std = self.data[modality].std(0)
        return mean, std

    @staticmethod
    def normalize_reward(dataset_name: str, rewards: jax.Array) -> jax.Array:
        if "antmaze" in dataset_name:
            return rewards * 100.0  # like in LAPO
        else:
            raise NotImplementedError(
                "Reward normalization is implemented only for AntMaze yet!"
            )


@chex.dataclass(frozen=True)
class Metrics:
    """순수 PyTree 기반 Metrics (deepcopy 제거, JAX 친화적)"""
    accumulators: Dict[str, Tuple[jax.Array, jax.Array]]

    @staticmethod
    def create(metrics: Sequence[str]) -> "Metrics":
        init_metrics = {key: (jnp.array(0.0), jnp.array(0.0)) for key in metrics}
        return Metrics(accumulators=init_metrics)

    def update(self, updates: Dict[str, jax.Array]) -> "Metrics":
        """순수 함수형 업데이트 (deepcopy 제거)"""
        new_accumulators = {}
        for key in self.accumulators.keys():
            acc, steps = self.accumulators[key]
            if key in updates:
                new_accumulators[key] = (acc + updates[key], steps + 1)
            else:
                new_accumulators[key] = (acc, steps)
        return self.replace(accumulators=new_accumulators)

    def compute(self) -> Dict[str, np.ndarray]:
        # cumulative_value / total_steps
        return {k: np.array(v[0] / v[1]) if v[1] > 0 else np.array(0.0) for k, v in self.accumulators.items()}


def normalize(
    arr: jax.Array, mean: jax.Array, std: jax.Array, eps: float = 1e-8
) -> jax.Array:
    return (arr - mean) / (std + eps)


def make_env(env_name: str, seed: int) -> gym.Env:
    env = gym.make(env_name)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state: np.ndarray) -> np.ndarray:
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

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
    """Evaluate policy over multiple episodes
    
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
    for ep in range(num_episodes):
        # 각 episode마다 시드 재설정 (pogogo.py와 동일)
        ep_seed = seed + ep
        np.random.seed(ep_seed)  # antmaze 등 reset 시 전역 np.random 쓰는 env용
        try:
            env.action_space.seed(ep_seed)
            reset_result = env.reset(seed=ep_seed)
        except (TypeError, AttributeError):
            try:
                reset_result = env.reset(seed=ep_seed)
            except TypeError:
                if hasattr(env, 'seed'):
                    env.seed(ep_seed)  # gym 0.23 등 reset(seed=) 미지원 시
                reset_result = env.reset()
        
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
        is_gaussian: Optional[bool] = None,
        is_stochastic: Optional[bool] = None,
    ) -> jax.Array:
        """Actor로부터 deterministic actions 추출 (통합 인터페이스)
        
        Policy 타입에 관계없이 일관된 방식으로 deterministic actions를 추출합니다.
        is_gaussian/is_stochastic 플래그로 분기하여 JIT trace 안정성을 보장합니다.
        
        Args:
            actor_params: Actor parameters
            actor_module: Actor module
            states: [B, state_dim] states
            is_gaussian: Whether actor is Gaussian (if None, inferred from module)
            is_stochastic: Whether actor is stochastic (if None, inferred from module)
        
        Returns:
            [B, action_dim] deterministic actions
        """
        # 플래그가 제공되지 않으면 hasattr로 fallback (하위 호환성)
        if is_gaussian is None:
            is_gaussian = hasattr(actor_module, 'get_mean_std')
        if is_stochastic is None:
            is_stochastic = getattr(actor_module, 'is_stochastic', False)
        
        if is_gaussian:
            # Gaussian/TanhGaussian policy: use mean
            mean, _ = actor_module.get_mean_std(actor_params, states)
            return mean
        elif hasattr(actor_module, 'deterministic_actions'):
            # Deterministic/Stochastic policy: use deterministic_actions method
            return actor_module.deterministic_actions(actor_params, states)
        else:
            # Fallback: direct forward pass
            return actor_module.apply(actor_params, states)
    
    @staticmethod
    def sample_actions(
        actor_params: FrozenDict,
        actor_module: nn.Module,
        states: jax.Array,
        key: jax.random.PRNGKey,
        K: int = 1,
        is_stochastic: Optional[bool] = None,
    ) -> jax.Array:
        """Actor로부터 sampled actions 추출 (통합 인터페이스)
        
        sample_actions 메서드가 있으면 사용하고, 없으면 deterministic_actions로 fallback합니다.
        
        Args:
            actor_params: Actor parameters
            actor_module: Actor module
            states: [B, state_dim] states
            key: PRNG key for sampling
            K: Number of samples per state
            is_stochastic: Whether actor is stochastic (if None, inferred from module)
        
        Returns:
            [B, K, action_dim] sampled actions (K=1인 경우 [B, action_dim]로 squeeze)
        """
        # is_stochastic 플래그 확인
        if is_stochastic is None:
            is_stochastic = getattr(actor_module, 'is_stochastic', False)
        
        # sample_actions 메서드가 있으면 사용
        if hasattr(actor_module, 'sample_actions') and is_stochastic:
            actions = actor_module.sample_actions(actor_params, states, key, K=K)
            if K == 1:
                return actions[:, 0, :]  # [B, 1, action_dim] -> [B, action_dim]
            return actions  # [B, K, action_dim]
        else:
            # Fallback: deterministic actions 사용
            deterministic_actions = AlgorithmInterface._deterministic_actions(
                actor_params, actor_module, states
            )
            if K == 1:
                return deterministic_actions  # [B, action_dim]
            # K > 1인 경우 동일한 action을 K번 반복
            return jnp.expand_dims(deterministic_actions, axis=1).repeat(K, axis=1)  # [B, K, action_dim]
    
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
        """Actor0 업데이트 (통합 인터페이스)
        
        Args:
            key: PRNG key
            actor: Actor0 train state
            actor_module: Actor0 module
            critic: Critic train state
            batch: Batch data
            flow_policy_state: Flow policy train state (optional, for FQL)
            tau: Target network update rate (deprecated: 타겟 업데이트는 update_multi_actor에서 처리)
            metrics: Metrics 객체
        
        Returns:
            업데이트된 key, actor, flow_policy_state (optional), metrics
        """
        raise NotImplementedError


# W2 Distance 계산 클래스
class W2DistanceCalculator:
    """Wasserstein-2 distance 계산 클래스"""
    
    def __init__(
        self,
        sinkhorn_blur: float = 0.05,
        sinkhorn_num_iterations: int = 100,
        sinkhorn_K: int = 4,
    ):
        """
        Args:
            sinkhorn_blur: Sinkhorn regularization parameter (epsilon)
            sinkhorn_num_iterations: Sinkhorn 알고리즘 반복 횟수
            sinkhorn_K: 샘플 수 (Gaussian이 아닌 경우에만 사용)
        """
        self.sinkhorn_blur = sinkhorn_blur
        self.sinkhorn_num_iterations = sinkhorn_num_iterations
        self.sinkhorn_K = sinkhorn_K
    
    def sinkhorn_distance(
        self,
        x: jax.Array,
        y: jax.Array,
    ) -> jax.Array:
        """
        Sinkhorn distance 계산 (OTT-jax 사용)
        
        Args:
            x: [B, K, action_dim] 첫 번째 분포의 샘플
            y: [B, K, action_dim] 두 번째 분포의 샘플 (detached)
        
        Returns:
            [B] 각 state에 대한 Sinkhorn distance
        """
        B, K, action_dim = x.shape
        
        # Uniform weights for each point cloud
        a = jnp.ones((B, K)) / K  # [B, K]
        b = jnp.ones((B, K)) / K  # [B, K]
        
        def compute_sinkhorn_for_batch(x_i, y_i, a_i, b_i):
            """Single batch Sinkhorn computation using OTT"""
            # Create geometry object
            geom = pointcloud.PointCloud(x_i, y_i, epsilon=self.sinkhorn_blur)
            
            # Solve Sinkhorn
            out = sinkhorn_solve(geom, a_i, b_i, max_iterations=self.sinkhorn_num_iterations)
            
            # Extract distance (transport cost)
            return out.reg_ot_cost
        
        # Vectorize over batch dimension
        distances = jax.vmap(compute_sinkhorn_for_batch)(x, y, a, b)  # [B]
        
        return distances
    
    @staticmethod
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
    
    def compute_distance(
        self,
        actor_i_config: ActorConfig,
        ref_actor_config: ActorConfig,
        states: jax.Array,
        key: jax.random.PRNGKey,
    ) -> jax.Array:
        """
        Per-state distance 계산 (Gaussian closed form 또는 Sinkhorn 또는 L2)
        
        Args:
            actor_i_config: 현재 actor 설정
            ref_actor_config: 참조 actor 설정
            states: [B, state_dim]
            key: PRNG key
        
        Returns:
            평균 distance (scalar)
        """
        # Both Gaussian: use closed form W2
        if actor_i_config.is_gaussian and ref_actor_config.is_gaussian:
            mean_i, std_i = actor_i_config.module.get_mean_std(actor_i_config.params, states)  # [B, action_dim]
            mean_ref, std_ref = ref_actor_config.module.get_mean_std(ref_actor_config.params, states)  # [B, action_dim]
            mean_ref = jax.lax.stop_gradient(mean_ref)
            std_ref = jax.lax.stop_gradient(std_ref)
            
            w2_squared = self.closed_form_w2_gaussian(mean_i, std_i, mean_ref, std_ref)  # [B]
            return w2_squared.mean()
        
        # At least one is not Gaussian: use sampling-based methods
        key1, key2 = jax.random.split(key)
        
        # Sample from current actor
        a = actor_i_config.module.sample_actions(actor_i_config.params, states, key1, self.sinkhorn_K)  # [B, K, action_dim]
        
        # Sample from reference actor (stop_gradient)
        b = ref_actor_config.module.sample_actions(ref_actor_config.params, states, key2, self.sinkhorn_K)  # [B, K, action_dim]
        b = jax.lax.stop_gradient(b)
        
        if actor_i_config.is_stochastic and ref_actor_config.is_stochastic:
            # Both stochastic (but not Gaussian): use Sinkhorn
            distances = self.sinkhorn_distance(a, b)  # [B]
            return distances.mean()
        else:
            # At least one deterministic: use L2
            a_det = a[:, 0, :]  # [B, action_dim] - take first sample
            b_det = b[:, 0, :]  # [B, action_dim]
            distances = jnp.sum((a_det - b_det) ** 2, axis=-1)  # [B]
            return distances.mean()
