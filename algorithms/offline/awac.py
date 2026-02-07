import os
import random
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import d4rl
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional
import wandb
from tqdm import trange
from .networks import GaussianMLP, build_mlp
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
    project: str = "CORL"
    group: str = "AWAC-D4RL"
    name: str = "AWAC"
    checkpoints_path: Optional[str] = None

    env_name: str = "halfcheetah-medium-expert-v2"
    seed: int = 42
    test_seed: int = 69
    deterministic_torch: bool = False
    device: str = "cuda"

    buffer_size: int = 2_000_000
    num_train_ops: int = 1_000_000
    batch_size: int = 256
    eval_frequency: int = 1000
    n_test_episodes: int = 10
    normalize_reward: bool = False

    hidden_dim: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 5e-3
    awac_lambda: float = 1.0

    def __post_init__(self):
        self.name = f"{self.name}-{self.env_name}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


# Actor는 pogo_policies.py의 GaussianMLP로 통합됨 (tanh_mean=False로 AWAC 방식 사용)
# AWAC는 unbounded mean에서 샘플링 후 clamp 사용
class Actor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        min_action: float = -1.0,
        max_action: float = 1.0,
    ):
        super().__init__()
        # AWAC는 3개의 hidden layer 사용 (256 -> 256 -> 256)
        # tanh_mean=False: unbounded mean에서 샘플링 후 clamp (AWAC 방식)
        self.policy = GaussianMLP(
            state_dim, action_dim, max_action,
            hidden_dim=hidden_dim, n_hiddens=3, tanh_mean=False
        )
        self._min_action = min_action
        self._max_action = max_action

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.policy.log_prob(state, action)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.policy(state)

    def act(self, state: np.ndarray, device: str) -> np.ndarray:
        state_t = torch.tensor(state[None], dtype=torch.float32, device=device)
        action_t, _ = self.policy(state_t)
        return action_t[0].cpu().numpy()




def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


class Critic(nn.Module):
    """Critic network for AWAC (networks.py의 build_mlp 사용)"""
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        layers = build_mlp(state_dim + action_dim, hidden_dim=hidden_dim, n_hiddens=3)
        self._mlp = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q_value = self._mlp(torch.cat([state, action], dim=-1))
        return q_value


class AdvantageWeightedActorCritic:
    def __init__(
        self,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        critic_1: nn.Module,
        critic_1_optimizer: torch.optim.Optimizer,
        critic_2: nn.Module,
        critic_2_optimizer: torch.optim.Optimizer,
        gamma: float = 0.99,
        tau: float = 5e-3,  # parameter for the soft target update,
        awac_lambda: float = 1.0,
        exp_adv_max: float = 100.0,
    ):
        self._actor = actor
        self._actor_optimizer = actor_optimizer

        self._critic_1 = critic_1
        self._critic_1_optimizer = critic_1_optimizer
        self._target_critic_1 = deepcopy(critic_1)

        self._critic_2 = critic_2
        self._critic_2_optimizer = critic_2_optimizer
        self._target_critic_2 = deepcopy(critic_2)

        self._gamma = gamma
        self._tau = tau
        self._awac_lambda = awac_lambda
        self._exp_adv_max = exp_adv_max

    def _actor_loss(self, states, actions):
        with torch.no_grad():
            pi_action, _ = self._actor(states)
            v = torch.min(
                self._critic_1(states, pi_action), self._critic_2(states, pi_action)
            )

            q = torch.min(
                self._critic_1(states, actions), self._critic_2(states, actions)
            )
            adv = q - v
            weights = torch.clamp_max(
                torch.exp(adv / self._awac_lambda), self._exp_adv_max
            )

        action_log_prob = self._actor.log_prob(states, actions)
        loss = (-action_log_prob * weights).mean()
        return loss

    def _critic_loss(self, states, actions, rewards, dones, next_states):
        with torch.no_grad():
            next_actions, _ = self._actor(next_states)

            q_next = torch.min(
                self._target_critic_1(next_states, next_actions),
                self._target_critic_2(next_states, next_actions),
            )
            q_target = rewards + self._gamma * (1.0 - dones) * q_next

        q1 = self._critic_1(states, actions)
        q2 = self._critic_2(states, actions)

        q1_loss = nn.functional.mse_loss(q1, q_target)
        q2_loss = nn.functional.mse_loss(q2, q_target)
        loss = q1_loss + q2_loss
        return loss

    def _update_critic(self, states, actions, rewards, dones, next_states):
        loss = self._critic_loss(states, actions, rewards, dones, next_states)
        self._critic_1_optimizer.zero_grad()
        self._critic_2_optimizer.zero_grad()
        loss.backward()
        self._critic_1_optimizer.step()
        self._critic_2_optimizer.step()
        return loss.item()

    def _update_actor(self, states, actions):
        loss = self._actor_loss(states, actions)
        self._actor_optimizer.zero_grad()
        loss.backward()
        self._actor_optimizer.step()
        return loss.item()

    def update(self, batch: TensorBatch) -> Dict[str, float]:
        states, actions, rewards, next_states, dones = batch
        critic_loss = self._update_critic(states, actions, rewards, dones, next_states)
        actor_loss = self._update_actor(states, actions)

        soft_update(self._target_critic_1, self._critic_1, self._tau)
        soft_update(self._target_critic_2, self._critic_2, self._tau)

        result = {"critic_loss": critic_loss, "actor_loss": actor_loss}
        return result

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self._actor.state_dict(),
            "critic_1": self._critic_1.state_dict(),
            "critic_2": self._critic_2.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self._actor.load_state_dict(state_dict["actor"])
        self._critic_1.load_state_dict(state_dict["critic_1"])
        self._critic_2.load_state_dict(state_dict["critic_2"])


def return_reward_range(dataset, max_episode_steps):
    """Return reward range for reward normalization"""
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
    """Modify reward for specific environments"""
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0


@pyrallis.wrap()
def train(config: TrainConfig):
    env = gym.make(config.env_name)
    set_seed(config.seed, env, deterministic_torch=config.deterministic_torch)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    dataset = d4rl.qlearning_dataset(env)

    if config.normalize_reward:
        modify_reward(dataset, config.env_name)

    state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
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

    actor_critic_kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_dim": config.hidden_dim,
    }

    actor = Actor(**actor_critic_kwargs)
    actor.to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.learning_rate)
    critic_1 = Critic(**actor_critic_kwargs)
    critic_2 = Critic(**actor_critic_kwargs)
    critic_1.to(config.device)
    critic_2.to(config.device)
    critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=config.learning_rate)
    critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=config.learning_rate)

    awac = AdvantageWeightedActorCritic(
        actor=actor,
        actor_optimizer=actor_optimizer,
        critic_1=critic_1,
        critic_1_optimizer=critic_1_optimizer,
        critic_2=critic_2,
        critic_2_optimizer=critic_2_optimizer,
        gamma=config.gamma,
        tau=config.tau,
        awac_lambda=config.awac_lambda,
    )
    wandb_init(asdict(config))

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    for t in trange(config.num_train_ops, ncols=80):
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        update_result = awac.update(batch)
        wandb.log(update_result, step=t)
        if (t + 1) % config.eval_frequency == 0:
            eval_scores = eval_actor(
                env, actor, config.device, config.n_test_episodes, config.test_seed
            )

            wandb.log({"eval_score": eval_scores.mean()}, step=t)
            if hasattr(env, "get_normalized_score"):
                normalized_eval_scores = env.get_normalized_score(eval_scores) * 100.0
                wandb.log(
                    {"d4rl_normalized_score": normalized_eval_scores.mean()}, step=t
                )

            if config.checkpoints_path is not None:
                torch.save(
                    awac.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                )

    wandb.finish()


# ============================================================================
# POGO Multi-Actor Interface 구현
# ============================================================================

from typing import Any, Dict, List

# 순환 참조 방지를 위해 파일 끝에서 import
from .utils_pytorch import PyTorchAlgorithmInterface


class AWACAlgorithm(PyTorchAlgorithmInterface):
    """AWAC 알고리즘의 POGO Multi-Actor 인터페이스 구현"""
    def __init__(self, trainer: 'AdvantageWeightedActorCritic'):
        self.trainer = trainer
    
    def update_critic(
        self,
        trainer: Any,
        batch: TensorBatch,
        log_dict: Dict[str, float],
        **kwargs
    ) -> Dict[str, float]:
        """AWAC의 Critic 업데이트"""
        if not hasattr(trainer, "total_it"):
            trainer.total_it = 0
        trainer.total_it += 1
        states, actions, rewards, next_states, dones = batch
        trainer._actor = kwargs['actors'][0]  # 임시로 첫 번째 actor 연결
        critic_loss = trainer._update_critic(states, actions, rewards, dones, next_states)
        log_dict["critic_loss"] = critic_loss
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
        """AWAC actor loss 계산"""
        states, actions, rewards, next_states, dones = batch
        
        if actor_idx == 0:
            # Actor0: AWAC loss
            with torch.no_grad():
                pi_action = actor.deterministic_actions(states)
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
            
            if hasattr(actor, 'forward') and not actor_is_stochastic:
                pi_i = actor.forward(states)
            else:
                pi_i = actor.sample_actions(states, K=1, seed=seed_base)[:, 0, :]
            bc_losses = torch.sum((pi_i - actions) ** 2, dim=1)
            return torch.mean(weights * bc_losses)
        else:
            # Actor1+: Q 기반 loss
            if hasattr(actor, 'forward') and not actor_is_stochastic:
                pi_i = actor.forward(states)
            else:
                pi_i = actor.sample_actions(states, K=1, seed=seed_base)[:, 0, :]
            Q_i = torch.min(
                trainer._critic_1(states, pi_i),
                trainer._critic_2(states, pi_i)
            )
            return -Q_i.mean()
    
    def update_target_networks(
        self,
        trainer: Any,
        actors: List[nn.Module],
        actor_targets: List[nn.Module],
        tau: float,
        **kwargs
    ) -> None:
        """AWAC target network 업데이트"""
        soft_update(trainer._target_critic_1, trainer._critic_1, trainer._tau)
        soft_update(trainer._target_critic_2, trainer._critic_2, trainer._tau)
        for actor, actor_target in zip(actors, actor_targets):
            for p, tp in zip(actor.parameters(), actor_target.parameters()):
                tp.data.copy_(trainer._tau * p.data + (1 - trainer._tau) * tp.data)


if __name__ == "__main__":
    train()
