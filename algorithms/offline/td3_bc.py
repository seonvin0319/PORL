# source: https://github.com/sfujim/TD3_BC
# https://arxiv.org/pdf/2106.06860.pdf
import copy
import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import d4rl
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from .networks import DeterministicMLP, build_mlp
from .utils_pytorch import (
    ReplayBuffer,
    set_seed,
    wandb_init,
    eval_actor,
    compute_mean_std,
    normalize_states,
    wrap_env,
)

TensorBatch = List[torch.Tensor]


@dataclass
class TrainConfig:
    # Experiment
    device: str = "cuda"
    env: str = "halfcheetah-medium-expert-v2"  # OpenAI gym environment name
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_freq: int = int(5e3)  # How often (time steps) we evaluate
    n_episodes: int = 10  # How many episodes run during evaluation
    max_timesteps: int = int(1e6)  # Max time steps to run environment
    checkpoints_path: Optional[str] = None  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load
    # TD3
    buffer_size: int = 2_000_000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount ffor
    expl_noise: float = 0.1  # Std of Gaussian exploration noise
    tau: float = 0.005  # Target network update rate
    policy_noise: float = 0.2  # Noise added to target actor during critic update
    noise_clip: float = 0.5  # Range to clip target actor noise
    policy_freq: int = 2  # Frequency of delayed actor updates
    # TD3 + BC
    alpha: float = 2.5  # Coefficient for Q function in actor loss
    normalize: bool = True  # Normalize states
    normalize_reward: bool = False  # Normalize reward
    # Wandb logging
    project: str = "CORL"
    group: str = "TD3_BC-D4RL"
    name: str = "TD3_BC"

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env




def return_reward_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def modify_reward(dataset, env_name, max_episode_steps=1000):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0


# Actor는 pogo_policies.py의 DeterministicMLP로 통합됨
Actor = DeterministicMLP




class Critic(nn.Module):
    """Critic network for TD3_BC (networks.py의 build_mlp 사용)"""
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        layers = build_mlp(state_dim + action_dim, hidden_dim=256, n_hiddens=3)
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], dim=-1)
        return self.net(sa)


class TD3_BC:
    def __init__(
        self,
        max_action: float,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        critic_1: nn.Module,
        critic_1_optimizer: torch.optim.Optimizer,
        critic_2: nn.Module,
        critic_2_optimizer: torch.optim.Optimizer,
        discount: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        alpha: float = 2.5,
        device: str = "cpu",
    ):
        self.actor = actor
        self.actor_target = copy.deepcopy(actor)
        self.actor_optimizer = actor_optimizer
        self.critic_1 = critic_1
        self.critic_1_target = copy.deepcopy(critic_1)
        self.critic_1_optimizer = critic_1_optimizer
        self.critic_2 = critic_2
        self.critic_2_target = copy.deepcopy(critic_2)
        self.critic_2_optimizer = critic_2_optimizer

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha

        self.total_it = 0
        self.device = device

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        """기존 TD3_BC 학습 메서드 (단일 actor용)"""
        log_dict = {}
        state, action, reward, next_state, done = batch
        not_done = 1.0 - done
        
        # Critic 업데이트
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = (self.actor_target.deterministic_actions(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )
            target_q1 = self.critic_1_target(next_state, next_action)
            target_q2 = self.critic_2_target(next_state, next_action)
            target_q = reward + not_done * self.discount * torch.min(target_q1, target_q2)
        
        current_q1 = self.critic_1(state, action)
        current_q2 = self.critic_2(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()
        log_dict["critic_loss"] = float(critic_loss.item())
        
        # Actor 업데이트 (policy_freq마다)
        if self.total_it % self.policy_freq == 0:
            pi = self.actor.deterministic_actions(state)
            q = self.critic_1(state, pi)
            lmbda = self.alpha / q.abs().mean().detach()
            actor_loss = -lmbda * q.mean() + self.alpha * ((pi - action) ** 2).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            log_dict["actor_loss"] = float(actor_loss.item())
            
            # Target network 업데이트
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        
        self.total_it += 1
        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "critic_1": self.critic_1.state_dict(),
            "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.critic_1.load_state_dict(state_dict["critic_1"])
        self.critic_1_optimizer.load_state_dict(state_dict["critic_1_optimizer"])
        self.critic_1_target = copy.deepcopy(self.critic_1)

        self.critic_2.load_state_dict(state_dict["critic_2"])
        self.critic_2_optimizer.load_state_dict(state_dict["critic_2_optimizer"])
        self.critic_2_target = copy.deepcopy(self.critic_2)

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_target = copy.deepcopy(self.actor)

        self.total_it = state_dict["total_it"]


@pyrallis.wrap()
def train(config: TrainConfig):
    env = gym.make(config.env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    dataset = d4rl.qlearning_dataset(env)

    if config.normalize_reward:
        modify_reward(dataset, config.env)

    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
    )
    replay_buffer.load_d4rl_dataset(dataset)

    max_action = float(env.action_space.high[0])

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Set seeds
    seed = config.seed
    set_seed(seed, env)

    actor = Actor(state_dim, action_dim, max_action).to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)

    critic_1 = Critic(state_dim, action_dim).to(config.device)
    critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=3e-4)
    critic_2 = Critic(state_dim, action_dim).to(config.device)
    critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=3e-4)

    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "critic_1": critic_1,
        "critic_1_optimizer": critic_1_optimizer,
        "critic_2": critic_2,
        "critic_2_optimizer": critic_2_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": config.device,
        # TD3
        "policy_noise": config.policy_noise * max_action,
        "noise_clip": config.noise_clip * max_action,
        "policy_freq": config.policy_freq,
        # TD3 + BC
        "alpha": config.alpha,
    }

    print("---------------------------------------")
    print(f"Training TD3 + BC, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize actor
    trainer = TD3_BC(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    wandb_init(asdict(config))

    evaluations = []
    for t in range(int(config.max_timesteps)):
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        log_dict = trainer.train(batch)
        wandb.log(log_dict, step=trainer.total_it)
        # Evaluate episode
        if (t + 1) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            eval_scores = eval_actor(
                env,
                actor,
                device=config.device,
                n_episodes=config.n_episodes,
                seed=config.seed,
            )
            eval_score = eval_scores.mean()
            normalized_eval_score = env.get_normalized_score(eval_score) * 100.0
            evaluations.append(normalized_eval_score)
            print("---------------------------------------")
            print(
                f"Evaluation over {config.n_episodes} episodes: "
                f"{eval_score:.3f} , D4RL score: {normalized_eval_score:.3f}"
            )
            print("---------------------------------------")

            if config.checkpoints_path is not None:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                )

            wandb.log(
                {"d4rl_normalized_score": normalized_eval_score},
                step=trainer.total_it,
            )


# ============================================================================
# POGO Multi-Actor Interface 구현
# ============================================================================

from typing import Any, Dict, List

# 순환 참조 방지를 위해 파일 끝에서 import
from .utils_pytorch import PyTorchAlgorithmInterface


class TD3BCAlgorithm(PyTorchAlgorithmInterface):
    """TD3_BC 알고리즘의 POGO Multi-Actor 인터페이스 구현"""
    def __init__(self, trainer: 'TD3_BC'):
        self.trainer = trainer
    
    def update_critic(
        self,
        trainer: Any,
        batch: TensorBatch,
        log_dict: Dict[str, float],
        **kwargs
    ) -> Dict[str, float]:
        """TD3_BC의 Critic 업데이트"""
        trainer.total_it += 1
        state, action, reward, next_state, done = batch
        not_done = 1.0 - done
        
        with torch.no_grad():
            noise = (torch.randn_like(action) * trainer.policy_noise).clamp(
                -trainer.noise_clip, trainer.noise_clip
            )
            next_action = (kwargs['actor_targets'][0](next_state) + noise).clamp(
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
        return log_dict
    
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
        """TD3_BC actor loss 계산"""
        state, action, reward, next_state, done = batch
        pi_i = actor.deterministic_actions(state)
        q = trainer.critic_1(state, pi_i)
        lmbda = trainer.alpha / q.abs().mean().detach()
        
        if actor_idx == 0:
            # Actor0: TD3_BC loss
            return -lmbda * q.mean() + trainer.alpha * ((pi_i - action) ** 2).mean()
        else:
            # Actor1+: Q 기반 loss만 (W2는 외부에서 추가)
            return -lmbda * q.mean()
    
    def update_target_networks(
        self,
        trainer: Any,
        actors: List[nn.Module],
        actor_targets: List[nn.Module],
        tau: float,
        **kwargs
    ) -> None:
        """TD3_BC target network 업데이트"""
        for actor, actor_target in zip(actors, actor_targets):
            for p, tp in zip(actor.parameters(), actor_target.parameters()):
                tp.data.copy_(trainer.tau * p.data + (1 - trainer.tau) * tp.data)
        for p, tp in zip(trainer.critic_1.parameters(), trainer.critic_1_target.parameters()):
            tp.data.copy_(trainer.tau * p.data + (1 - trainer.tau) * tp.data)
        for p, tp in zip(trainer.critic_2.parameters(), trainer.critic_2_target.parameters()):
            tp.data.copy_(trainer.tau * p.data + (1 - trainer.tau) * tp.data)


if __name__ == "__main__":
    train()
