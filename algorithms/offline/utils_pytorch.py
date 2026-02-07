# POGO Multi-Actor 공통 유틸리티 (PyTorch 버전)
# PyTorchAlgorithmInterface와 관련 클래스들을 정의

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import uuid

TensorBatch = Tuple[torch.Tensor, ...]

# 각 알고리즘의 인터페이스 구현은 각 알고리즘 파일에서 정의됨
# 순환 참조 방지를 위해 여기서는 import하지 않음
# 대신 pogo_multi_main.py에서 직접 import하도록 변경


class PyTorchAlgorithmInterface:
    """PyTorch 알고리즘별 인터페이스 정의
    JAX 버전의 AlgorithmInterface와 유사한 구조
    다른 알고리즘으로 확장할 때 이 인터페이스를 구현하면 됨
    """
    def update_critic(
        self,
        trainer: Any,
        batch: TensorBatch,
        log_dict: Dict[str, float],
        **kwargs
    ) -> Dict[str, float]:
        """Critic/V/Q 업데이트
        
        Args:
            trainer: Base trainer 객체
            batch: Batch data
            log_dict: 로깅용 딕셔너리
            **kwargs: Additional arguments
        
        Returns:
            업데이트된 log_dict
        """
        raise NotImplementedError
    
    def compute_actor_loss(
        self,
        trainer: Any,
        actor: nn.Module,
        batch: TensorBatch,
        actor_idx: int,
        actor_is_stochastic: bool,
        seed_base: int,
        **kwargs
    ) -> torch.Tensor:
        """Actor loss 계산
        
        Args:
            trainer: Base trainer 객체
            actor: Actor network
            batch: Batch data
            actor_idx: Actor index (0 for Actor0, 1+ for Actor1+)
            actor_is_stochastic: Actor가 stochastic인지 여부
            seed_base: 랜덤 시드 베이스
            **kwargs: Additional arguments
        
        Returns:
            Actor loss (scalar tensor)
        """
        raise NotImplementedError
    
    def update_target_networks(
        self,
        trainer: Any,
        actors: List[nn.Module],
        actor_targets: List[nn.Module],
        tau: float,
        **kwargs
    ) -> None:
        """Target network 업데이트
        
        Args:
            trainer: Base trainer 객체
            actors: Actor networks
            actor_targets: Target actor networks
            tau: Soft update coefficient
            **kwargs: Additional arguments
        """
        raise NotImplementedError


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


def action_for_loss(
    actor: nn.Module,
    cfg: ActorConfig,
    states: torch.Tensor,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """미분 가능한 action getter (학습 loss / W2 penalty용).
    deterministic_actions()는 @torch.no_grad() 때문에 gradient가 끊기므로 사용하지 않음.
    IQL GaussianPolicy처럼 forward가 Distribution을 반환하는 경우도 처리.
    """
    if cfg.is_gaussian and hasattr(actor, "get_mean_std"):
        return actor.get_mean_std(states)[0]
    if cfg.is_stochastic and hasattr(actor, "sample_actions"):
        return actor.sample_actions(states, K=1, seed=seed)[:, 0, :]
    out = actor.forward(states)
    if isinstance(out, tuple):
        return out[0]
    if isinstance(out, torch.distributions.Distribution):
        return out.rsample() if out.has_rsample else out.sample()
    return out


# ============================================================================
# ReplayBuffer
# ============================================================================

class ReplayBuffer:
    """D4RL 데이터셋을 위한 Replay Buffer"""
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
        seed: Optional[int] = None,
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device
        
        # NumPy Generator for reproducible sampling
        # 전역 numpy random state 대신 독립적인 generator 사용
        self._rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        """Loads data in d4rl format, i.e. from Dict[str, np.array]."""
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> List[torch.Tensor]:
        """Sample a batch of transitions
        
        재현성을 위해 내부 numpy Generator 사용 (전역 numpy random state와 독립적)
        """
        max_idx = min(self._size, self._pointer)
        indices = self._rng.integers(0, max_idx, size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def add_transition(self):
        """Use this method to add new data into the replay buffer during fine-tuning."""
        raise NotImplementedError


# ============================================================================
# 공통 유틸리티 함수들
# ============================================================================

def soft_update(target: nn.Module, source: nn.Module, tau: float):
    """Soft update target network parameters"""
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    """Set random seeds for reproducibility
    
    모든 랜덤 소스에 seed를 설정하여 완전한 재현성 보장:
    - Python random
    - NumPy random
    - PyTorch random (CPU & CUDA)
    - 환경 (gym) seed
    - Python hash seed
    """
    import os
    import random
    
    # Python hash seed (dict 등에서 사용)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # Python random
    random.seed(seed)
    
    # NumPy random
    np.random.seed(seed)
    
    # PyTorch random (CPU)
    torch.manual_seed(seed)
    
    # PyTorch random (CUDA) - 모든 GPU에 적용
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # CUDA 연산의 재현성 보장 (성능 저하 가능)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # PyTorch deterministic algorithms
    torch.use_deterministic_algorithms(deterministic_torch)
    
    # 환경 seed 설정
    if env is not None:
        try:
            env.seed(seed)
        except AttributeError:
            pass
        try:
            env.action_space.seed(seed)
        except AttributeError:
            pass
        try:
            env.observation_space.seed(seed)
        except AttributeError:
            pass


def wandb_init(config: dict) -> None:
    """Initialize wandb logging"""
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int
) -> np.ndarray:
    """Evaluate actor on environment
    
    재현성을 위해 각 episode마다 일관된 seed 사용
    """
    actor.eval()
    # 환경 seed 설정 (재현성 보장)
    try:
        env.seed(seed)
        env.action_space.seed(seed)
        if hasattr(env, 'observation_space'):
            env.observation_space.seed(seed)
    except Exception:
        pass
    
    episode_rewards = []
    for ep in range(n_episodes):
        # 각 episode마다 다른 seed 사용 (하지만 일관성 유지)
        episode_seed = seed + ep
        try:
            env.seed(episode_seed)
            env.action_space.seed(episode_seed)
        except Exception:
            pass
        
        state, done = env.reset(), False
        if isinstance(state, tuple):
            state = state[0]
        episode_reward = 0.0
        while not done:
            action = actor.act(state, device)
            step_out = env.step(action)
            if len(step_out) == 5:
                state, reward, terminated, truncated, _ = step_out
                done = terminated or truncated
            else:
                state, reward, done, _ = step_out
            episode_reward += reward
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and std of states"""
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    """Normalize states"""
    return (states - mean) / std


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    """Wrap environment with normalization"""
    def normalize_state(state):
        return (state - state_mean) / state_std

    def scale_reward(reward):
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


# 각 알고리즘의 인터페이스 구현은 각 알고리즘 파일에서 정의됨
# 여기서는 import만 수행
