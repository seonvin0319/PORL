# POGO Multi-Actor: POGO_sv baseline + CORL integration
# - Actor0: W2 (L2 to dataset actions)
# - Actor1+: Sinkhorn distance to previous actor
# - Config에서 base_algorithm으로 다른 알고리즘 config 참조 가능

import copy
import os
import random
import uuid
from dataclasses import asdict, dataclass, field
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
from geomloss import SamplesLoss

# POGO Policy (from POGO_sv) - inline for standalone execution
try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol


class Policy(Protocol):
    action_dim: int
    max_action: float

    def sample_actions(self, states: torch.Tensor, K: int = 1, seed: Optional[int] = None) -> torch.Tensor: ...
    def deterministic_actions(self, states: torch.Tensor) -> torch.Tensor: ...


class BaseActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super().__init__()
        self.action_dim = action_dim
        self.max_action = max_action

    def _randn(self, shape: Tuple[int, ...], device, dtype, seed: Optional[int]):
        if seed is None:
            return torch.randn(shape, device=device, dtype=dtype)
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        return torch.randn(shape, device=device, dtype=dtype, generator=g)

    @torch.no_grad()
    def deterministic_actions(self, states: torch.Tensor) -> torch.Tensor:
        B = states.size(0)
        z = torch.zeros((B, self.action_dim), device=states.device, dtype=states.dtype)
        return self.forward(states, z)

    def sample_actions(self, states: torch.Tensor, K: int = 1, seed: Optional[int] = None) -> torch.Tensor:
        B = states.size(0)
        z = self._randn((B, K, self.action_dim), device=states.device, dtype=states.dtype, seed=seed)
        states_flat = states[:, None, :].expand(B, K, states.size(1)).reshape(B * K, -1)
        z_flat = z.reshape(B * K, self.action_dim)
        return self.forward(states_flat, z_flat).reshape(B, K, self.action_dim)

    def forward(self, states: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class StochasticMLP(BaseActor):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__(state_dim, action_dim, max_action)
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.head = nn.Linear(256, action_dim)

    def forward(self, states, z):
        x = torch.cat([states, z], dim=1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return torch.tanh(self.head(x)) * self.max_action


class DeterministicMLP(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.action_dim = action_dim
        self.max_action = max_action
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.head = nn.Linear(256, action_dim)

    def forward(self, states):
        x = F.relu(self.l1(states))
        x = F.relu(self.l2(x))
        return torch.tanh(self.head(x)) * self.max_action

    @torch.no_grad()
    def deterministic_actions(self, states):
        return self.forward(states)

    def sample_actions(self, states, K: int = 1, seed: Optional[int] = None):
        a = self.forward(states)
        return a[:, None, :].expand(a.size(0), K, a.size(1)).contiguous()

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

    # Config에서 algorithm 선택: 다른 알고리즘의 config를 참조
    # 예: base_algorithm=iql -> configs/offline/iql/{env_domain}/{env_task}.yaml
    base_algorithm: Optional[str] = None  # iql, cql, td3_bc, rebrac 등
    base_config_path: Optional[str] = None  # 직접 config 경로 지정

    # POGO Multi-Actor
    w2_weights: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    num_actors: Optional[int] = None  # None이면 len(w2_weights)
    actor_configs: Optional[List[dict]] = None  # [{"deterministic": bool}, ...]

    # Sinkhorn (actor1+용)
    sinkhorn_K: int = 4
    sinkhorn_blur: float = 0.05
    sinkhorn_backend: str = "tensorized"

    # TD3/Critic
    buffer_size: int = 2_000_000
    batch_size: int = 256
    discount: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    lr: float = 3e-4

    # Data
    normalize: bool = True
    normalize_reward: bool = False

    # Wandb
    project: str = "CORL"
    group: str = "POGO-Multi"
    name: str = "POGO-Multi"

    def __post_init__(self):
        if self.num_actors is None:
            self.num_actors = len(self.w2_weights)
        if len(self.w2_weights) < self.num_actors:
            w = self.w2_weights[0] if self.w2_weights else 1.0
            self.w2_weights = self.w2_weights + [w] * (self.num_actors - len(self.w2_weights))
        self.w2_weights = self.w2_weights[: self.num_actors]
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


def _per_state_sinkhorn(
    policy: Policy,
    ref: Policy,
    states: torch.Tensor,
    K: int = 4,
    blur: float = 0.05,
    p: int = 2,
    backend: str = "tensorized",
    sinkhorn_loss=None,
    seed: Optional[int] = None,
):
    if sinkhorn_loss is None:
        sinkhorn_loss = SamplesLoss(loss="sinkhorn", p=p, blur=blur, backend=backend)
    a = policy.sample_actions(states, K=K, seed=None if seed is None else seed + 0)
    with torch.no_grad():
        b_detached = ref.sample_actions(states, K=K, seed=None if seed is None else seed + 10000).detach()
    b = b_detached
    loss = sinkhorn_loss(a, b)
    return loss.mean() if loss.dim() > 0 else loss


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], dim=1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class POGOMultiAgent:
    """
    Multi-actor POGO:
    - Actor0: W2 (L2 to dataset actions)
    - Actor1+: Sinkhorn distance to actor[i-1]
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        w2_weights: List[float],
        discount: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        lr: float = 3e-4,
        seed: Optional[int] = None,
        actor_configs: Optional[List[dict]] = None,
        sinkhorn_K: int = 4,
        sinkhorn_blur: float = 0.05,
        sinkhorn_backend: str = "tensorized",
        device: str = "cuda",
    ):
        self.num_actors = len(w2_weights)
        self.w2_weights = w2_weights
        self.seed = seed
        self.action_dim = action_dim
        self.max_action = max_action
        self.device = torch.device(device)

        if actor_configs is None:
            actor_configs = [{} for _ in range(self.num_actors)]
        assert len(actor_configs) == self.num_actors

        self.actors = []
        self.actor_targets = []
        self.actor_optimizers = []
        self.actor_is_stochastic = []

        for i in range(self.num_actors):
            config = actor_configs[i]
            deterministic = config.get("deterministic", False)
            if deterministic:
                actor_cls = DeterministicMLP
                is_stochastic = False
            else:
                actor_cls = StochasticMLP
                is_stochastic = True
            actor = actor_cls(state_dim, action_dim, max_action).to(self.device)
            actor_target = copy.deepcopy(actor)
            self.actors.append(actor)
            self.actor_targets.append(actor_target)
            self.actor_optimizers.append(torch.optim.Adam(actor.parameters(), lr=lr))
            self.actor_is_stochastic.append(is_stochastic)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0
        self._sinkhorn_loss = SamplesLoss(
            loss="sinkhorn", p=2, blur=sinkhorn_blur, backend=sinkhorn_backend
        )
        self.sinkhorn_K = sinkhorn_K
        self.sinkhorn_blur = sinkhorn_blur
        self.sinkhorn_backend = sinkhorn_backend

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = True,
        actor_idx: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).reshape(1, -1)
        if actor_idx is None:
            actor_idx = self.num_actors - 1
        actor = self.actors[actor_idx]
        if deterministic:
            action = actor.deterministic_actions(state)
        else:
            action = actor.sample_actions(state, K=1, seed=seed)[:, 0, :]
        return action.detach().cpu().numpy().flatten()

    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        return self.select_action(state, deterministic=True, actor_idx=self.num_actors - 1)

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        state, action, reward, next_state, done = batch
        not_done = 1.0 - done

        base_seed = self.seed if self.seed is not None else 0
        seed_base = base_seed * 1000000 + self.total_it * 1000

        # Critic
        with torch.no_grad():
            actor0_target = self.actor_targets[0]
            next_action = actor0_target.sample_actions(next_state, K=1, seed=seed_base + 1)[:, 0, :]
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = reward + not_done * self.discount * torch.min(target_Q1, target_Q2)
            gen = torch.Generator(device=self.device).manual_seed(seed_base + 2)
            action_noise = self.policy_noise * torch.randn(
                action.shape, dtype=action.dtype, device=self.device, generator=gen
            ).clamp(-self.noise_clip, self.noise_clip)
            action_noise = action_noise * self.max_action
            noisy_next_action = (next_action + action_noise).clamp(
                -self.max_action, self.max_action
            )
            nq1, nq2 = self.critic_target(next_state, noisy_next_action)
            target_Q = reward + not_done * self.discount * torch.min(nq1, nq2)

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        metrics = {"critic_loss": float(critic_loss.item())}

        # Actor (policy_freq마다)
        if self.total_it % self.policy_freq == 0:
            for a in self.actors:
                a.train()
            for i in range(self.num_actors):
                actor_i = self.actors[i]
                w2_weight_i = self.w2_weights[i]
                pi_i = actor_i.sample_actions(state, K=1, seed=seed_base + 10 + i)[:, 0, :]

                if i == 0:
                    for p in self.critic.parameters():
                        p.requires_grad_(True)
                elif i == 1:
                    for p in self.critic.parameters():
                        p.requires_grad_(False)

                Q_i = self.critic.Q1(state, pi_i)

                if i == 0:
                    w2_i = ((pi_i - action) ** 2).mean()
                else:
                    ref_actor = self.actors[i - 1]
                    is_i_stoch = self.actor_is_stochastic[i]
                    is_ref_stoch = self.actor_is_stochastic[i - 1]
                    if is_i_stoch and is_ref_stoch:
                        w2_i = _per_state_sinkhorn(
                            actor_i,
                            ref_actor,
                            state,
                            K=self.sinkhorn_K,
                            blur=self.sinkhorn_blur,
                            p=2,
                            backend=self.sinkhorn_backend,
                            sinkhorn_loss=self._sinkhorn_loss,
                            seed=seed_base + 100 + i,
                        )
                    else:
                        with torch.no_grad():
                            ref_action = ref_actor.deterministic_actions(state)
                        w2_i = ((pi_i - ref_action) ** 2).mean()

                actor_loss_i = -(Q_i.mean()) + w2_weight_i * w2_i
                opt = self.actor_optimizers[i]
                opt.zero_grad()
                actor_loss_i.backward()
                opt.step()

                metrics[f"actor_{i}_loss"] = float(actor_loss_i.item())
                metrics[f"w2_{i}_distance"] = float(w2_i.item())

            for p in self.critic.parameters():
                p.requires_grad_(True)
            for actor, actor_target in zip(self.actors, self.actor_targets):
                for p, tp in zip(actor.parameters(), actor_target.parameters()):
                    tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return metrics


def _merge_base_config(config: TrainConfig) -> TrainConfig:
    """base_algorithm 또는 base_config_path로 참조 config 로드 후 merge"""
    import yaml
    base_path = None
    if config.base_config_path:
        p = Path(config.base_config_path)
        if not p.is_absolute():
            p = Path(__file__).resolve().parents[2] / p
        if p.exists():
            base_path = p
    elif config.base_algorithm:
        env_parts = config.env.split("-", 1)
        if len(env_parts) == 2:
            domain, task = env_parts[0], env_parts[1].replace("-", "_")
            p = Path(__file__).resolve().parents[2] / "configs" / "offline" / config.base_algorithm / domain / f"{task}.yaml"
            if p.exists():
                base_path = p
    if base_path:
        with open(base_path) as f:
            base_cfg = yaml.safe_load(f) or {}
        for k in ["batch_size", "discount", "tau", "lr", "buffer_size", "normalize", "normalize_reward"]:
            if k in base_cfg and hasattr(config, k):
                setattr(config, k, base_cfg[k])
        if "env" in base_cfg:
            config.env = base_cfg["env"]
        if "actor_lr" in base_cfg and hasattr(config, "lr"):
            config.lr = base_cfg["actor_lr"]
        if "actor_learning_rate" in base_cfg:
            config.lr = base_cfg["actor_learning_rate"]
    return config


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wrap_env(env: gym.Env, state_mean, state_std, reward_scale: float = 1.0):
    def normalize_state(state):
        return (state - state_mean) / state_std

    def scale_reward(reward):
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


def return_reward_range(dataset, max_episode_steps=1000):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)
    return min(returns), max(returns)


def modify_reward(dataset, env_name, max_episode_steps=1000):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, buffer_size, device="cpu"):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0
        self._states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self._actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=device)
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        n = data["observations"].shape[0]
        self._states[:n] = torch.tensor(data["observations"], dtype=torch.float32, device=self._device)
        self._actions[:n] = torch.tensor(data["actions"], dtype=torch.float32, device=self._device)
        self._rewards[:n] = torch.tensor(data["rewards"][..., None], dtype=torch.float32, device=self._device)
        self._next_states[:n] = torch.tensor(data["next_observations"], dtype=torch.float32, device=self._device)
        self._dones[:n] = torch.tensor(data["terminals"][..., None], dtype=torch.float32, device=self._device)
        self._size = n
        self._pointer = n
        print(f"Dataset size: {n}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, self._size, size=batch_size)
        return [
            self._states[indices],
            self._actions[indices],
            self._rewards[indices],
            self._next_states[indices],
            self._dones[indices],
        ]


def set_seed(seed: int, env=None, deterministic_torch: bool = False):
    if env is not None:
        try:
            env.seed(seed)
            env.action_space.seed(seed)
        except Exception:
            pass
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def wandb_init(config: dict):
    if os.getenv("WANDB_MODE") == "disabled":
        return
    wandb.init(config=config, project=config["project"], group=config["group"], name=config["name"], id=str(uuid.uuid4()))
    wandb.run.save()


@torch.no_grad()
def eval_actor(agent: POGOMultiAgent, env: gym.Env, device: str, n_episodes: int, seed: int, actor_idx: Optional[int] = None) -> np.ndarray:
    try:
        env.seed(seed)
        env.action_space.seed(seed)
    except Exception:
        pass
    agent.actors[agent.num_actors - 1 if actor_idx is None else actor_idx].eval()
    episode_rewards = []
    for _ in range(n_episodes):
        state, done = env.reset(), False
        if isinstance(state, tuple):
            state = state[0]
        ep_ret = 0.0
        while not done:
            action = agent.select_action(state, deterministic=True, actor_idx=actor_idx)
            step_out = env.step(action)
            if len(step_out) == 5:
                state, reward, terminated, truncated, _ = step_out
                done = terminated or truncated
            else:
                state, reward, done, _ = step_out
            ep_ret += reward
        episode_rewards.append(ep_ret)
    agent.actors[agent.num_actors - 1 if actor_idx is None else actor_idx].train()
    return np.asarray(episode_rewards)


@pyrallis.wrap()
def train(config: TrainConfig):
    config = _merge_base_config(config)
    env = gym.make(config.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    dataset = d4rl.qlearning_dataset(env)
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
    agent = POGOMultiAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        w2_weights=config.w2_weights,
        discount=config.discount,
        tau=config.tau,
        policy_noise=config.policy_noise * max_action,
        noise_clip=config.noise_clip * max_action,
        policy_freq=config.policy_freq,
        lr=config.lr,
        seed=config.seed,
        actor_configs=config.actor_configs,
        sinkhorn_K=config.sinkhorn_K,
        sinkhorn_blur=config.sinkhorn_blur,
        sinkhorn_backend=config.sinkhorn_backend,
        device=config.device,
    )

    if config.load_model:
        state_dict = torch.load(config.load_model, map_location=config.device)
        if "critic" in state_dict:
            agent.critic.load_state_dict(state_dict["critic"])
            agent.critic_target = copy.deepcopy(agent.critic)
        for i in range(agent.num_actors):
            key = f"actor_{i}"
            if key in state_dict:
                agent.actors[i].load_state_dict(state_dict[key])
                agent.actor_targets[i] = copy.deepcopy(agent.actors[i])

    wandb_init(asdict(config))

    print("---------------------------------------")
    print(f"Training POGO Multi-Actor, Env: {config.env}, Seed: {config.seed}")
    print(f"  Actors: {agent.num_actors}, W2 weights: {config.w2_weights}")
    print("  Actor0: W2 to dataset, Actor1+: Sinkhorn")
    print("---------------------------------------")

    evaluations = {i: [] for i in range(agent.num_actors)}
    for t in range(int(config.max_timesteps)):
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        log_dict = agent.train(batch)
        wandb.log(log_dict, step=agent.total_it)

        if (t + 1) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            for i in range(agent.num_actors):
                scores = eval_actor(agent, env, config.device, config.n_episodes, config.seed + 100 + i, actor_idx=i)
                norm_score = env.get_normalized_score(scores.mean()) * 100.0
                evaluations[i].append(norm_score)
                log_dict[f"eval_actor_{i}"] = norm_score
                print(f"  Actor {i} eval (norm): {norm_score:.1f}")
            wandb.log(log_dict, step=agent.total_it)

    if config.checkpoints_path:
        ckpt = {
            "critic": agent.critic.state_dict(),
            "critic_target": agent.critic_target.state_dict(),
        }
        for i in range(agent.num_actors):
            ckpt[f"actor_{i}"] = agent.actors[i].state_dict()
        torch.save(ckpt, os.path.join(config.checkpoints_path, "model.pt"))
    return agent
