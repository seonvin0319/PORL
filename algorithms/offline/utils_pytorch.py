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
        """Sample a batch of transitions"""
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
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

def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    """Set random seeds for reproducibility"""
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    import os
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


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
    """Evaluate actor on environment"""
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        while not done:
            action = actor.act(state, device)
            state, reward, done, _ = env.step(action)
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
