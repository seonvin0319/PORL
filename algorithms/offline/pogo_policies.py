# POGO Policy implementations (from POGO_sv)
# Policy Protocol: sample_actions, deterministic_actions
# BaseActor: z를 사용하는 stochastic actor
# StochasticMLP, DeterministicMLP, PolicyAdapter

import inspect
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol


class Policy(Protocol):
    action_dim: int
    max_action: float

    def sample_actions(
        self,
        states: torch.Tensor,
        K: int = 1,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Return [B, K, action_dim]."""

    def deterministic_actions(self, states: torch.Tensor) -> torch.Tensor:
        """Return [B, action_dim]."""


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
        a = self.forward(states_flat, z_flat).reshape(B, K, self.action_dim)
        return a

    def forward(self, states: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class StochasticMLP(BaseActor):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__(state_dim, action_dim, max_action)
        self.is_gaussian = False
        self.is_stochastic = True
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.head = nn.Linear(256, action_dim)

    def _forward_with_z(self, states, z):
        x = torch.cat([states, z], dim=1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return torch.tanh(self.head(x)) * self.max_action

    def forward(self, states, z=None, deterministic: bool = False):
        """BaseActor 호환: z가 있으면 사용. z=None이면 deterministic으로 z=0 또는 샘플."""
        if z is not None:
            return self._forward_with_z(states, z)
        if deterministic:
            z = torch.zeros((states.size(0), self.action_dim), device=states.device, dtype=states.dtype)
        else:
            z = torch.randn((states.size(0), self.action_dim), device=states.device, dtype=states.dtype)
        return self._forward_with_z(states, z)

    def action_and_log_prob(self, obs):
        """StochasticMLP는 명시적 분포 없음 → log_prob 0"""
        action = self.forward(obs, deterministic=False)
        log_prob = torch.zeros(obs.size(0), 1, device=action.device, dtype=action.dtype)
        return action, log_prob

    @torch.no_grad()
    def act(self, state, device: str = "cpu"):
        """CORL 알고리즘과 호환되는 act 메서드"""
        import numpy as np
        state_tensor = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        action_tensor = self.deterministic_actions(state_tensor)
        return action_tensor.cpu().numpy().flatten()


class GaussianMLP(nn.Module):
    """Gaussian Policy: mean에 tanh 적용 여부를 선택 가능
    tanh_mean=True: mean = tanh(...) * max_action (bounded mean, IQL처럼)
    tanh_mean=False: mean = ... (unbounded mean, AWAC처럼)
    mean, std로 Gaussian 샘플링 (둘 다 closed form W2 사용 가능)
    """
    LOG_STD_MIN = -20.0
    LOG_STD_MAX = 2.0

    def __init__(self, state_dim, action_dim, max_action, tanh_mean=True):
        super().__init__()
        self.action_dim = action_dim
        self.max_action = max_action
        self.tanh_mean = tanh_mean
        self.is_gaussian = True  # Closed form W2 사용 가능
        self.is_stochastic = True
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.head = nn.Linear(256, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))

    def _mean_logstd(self, obs):
        x = F.relu(self.l1(obs))
        x = F.relu(self.l2(x))
        mean = self.head(x)
        if self.tanh_mean:
            mean = torch.tanh(mean) * self.max_action
        log_std = self.log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std).unsqueeze(0).expand(mean.size(0), -1)
        return mean, std

    def forward(self, obs, deterministic: bool = False):
        """action만 반환"""
        mean, std = self._mean_logstd(obs)
        if deterministic:
            return mean.clamp(-self.max_action, self.max_action)
        eps = torch.randn_like(mean)
        action = mean + std * eps
        return action.clamp(-self.max_action, self.max_action)

    def action_and_log_prob(self, obs):
        mean, std = self._mean_logstd(obs)
        dist = torch.distributions.Normal(mean, std)
        pre = dist.rsample()
        action = pre.clamp(-self.max_action, self.max_action)
        log_prob = dist.log_prob(pre).sum(dim=-1, keepdim=True)
        return action, log_prob

    def get_mean_std(self, states):
        """Get mean and std: (mean, std)"""
        return self._mean_logstd(states)

    def log_prob_actions(self, states, actions, keepdim: bool = True) -> torch.Tensor:
        """Normal(mean, std)에서 actions의 log_prob을 action_dim에 대해 합산."""
        mean, std = self._mean_logstd(states)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        return log_prob.unsqueeze(-1) if keepdim else log_prob

    @torch.no_grad()
    def deterministic_actions(self, states):
        """Deterministic action: mean"""
        return self.forward(states, deterministic=True)

    def sample_actions(self, states, K: int = 1, seed: Optional[int] = None):
        """Sample K actions: [B, K, action_dim]"""
        mean, std = self._mean_logstd(states)
        B = states.size(0)
        if seed is None:
            noise = torch.randn(B, K, self.action_dim, device=states.device, dtype=states.dtype)
        else:
            g = torch.Generator(device=states.device)
            g.manual_seed(seed)
            noise = torch.randn(B, K, self.action_dim, device=states.device, dtype=states.dtype, generator=g)
        mean_expanded = mean.unsqueeze(1).expand(B, K, self.action_dim)
        std_expanded = std.unsqueeze(1).expand(B, K, self.action_dim)
        actions = (mean_expanded + std_expanded * noise).clamp(-self.max_action, self.max_action)
        return actions

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        """CORL 알고리즘과 호환되는 act 메서드"""
        state_tensor = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        action_tensor = self.deterministic_actions(state_tensor)
        return action_tensor.cpu().numpy().flatten()


class TanhGaussianMLP(nn.Module):
    """TanhGaussian Policy: unbounded Gaussian에서 샘플링 후 tanh 적용
    mean = ... (unbounded)
    mean, std로 Gaussian 샘플링 (unbounded space)
    그 다음 tanh를 적용하여 bounded로 만듦
    Closed form W2 사용 불가 (Sinkhorn 사용)
    """
    LOG_STD_MIN = -20.0
    LOG_STD_MAX = 2.0

    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.action_dim = action_dim
        self.max_action = max_action
        self.is_gaussian = False  # TanhGaussian은 closed form W2 사용 불가
        self.is_stochastic = True
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.head = nn.Linear(256, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))

    def _mean_logstd(self, obs):
        x = F.relu(self.l1(obs))
        x = F.relu(self.l2(x))
        mean = self.head(x)  # Unbounded
        log_std = self.log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std).unsqueeze(0).expand(mean.size(0), -1)
        return mean, std

    def forward(self, obs, deterministic: bool = False):
        """action만 반환"""
        mean, std = self._mean_logstd(obs)
        if deterministic:
            return torch.tanh(mean) * self.max_action
        eps = torch.randn_like(mean)
        pre_tanh = mean + std * eps
        return torch.tanh(pre_tanh) * self.max_action

    def action_and_log_prob(self, obs):
        mean, std = self._mean_logstd(obs)
        eps = torch.randn_like(mean)
        pre_tanh = mean + std * eps
        tanh_a = torch.tanh(pre_tanh)
        action = tanh_a * self.max_action
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(pre_tanh).sum(dim=-1, keepdim=True)
        log_prob -= torch.log(1 - tanh_a.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        return action, log_prob

    def get_mean_std(self, states):
        """Get mean and std: (mean, std). mean은 unbounded space."""
        return self._mean_logstd(states)

    def log_prob_actions(self, states, actions, keepdim: bool = True) -> torch.Tensor:
        """TanhGaussian: bounded action을 u=atanh(a/max)로 역변환 후 log_prob."""
        mean, std = self._mean_logstd(states)
        a_norm = (actions / self.max_action).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        u = torch.atanh(a_norm)
        dist = torch.distributions.Normal(mean, std)
        log_prob_u = dist.log_prob(u).sum(dim=-1)
        log_prob_u = log_prob_u - torch.log(1.0 - a_norm.pow(2) + 1e-6).sum(dim=-1)
        return log_prob_u.unsqueeze(-1) if keepdim else log_prob_u

    @torch.no_grad()
    def deterministic_actions(self, states):
        """Deterministic action: tanh(mean) * max_action"""
        return self.forward(states, deterministic=True)

    def sample_actions(self, states, K: int = 1, seed: Optional[int] = None):
        """Sample K actions: [B, K, action_dim]. unbounded Gaussian → tanh."""
        mean, std = self._mean_logstd(states)
        B = states.size(0)
        if seed is None:
            noise = torch.randn(B, K, self.action_dim, device=states.device, dtype=states.dtype)
        else:
            g = torch.Generator(device=states.device)
            g.manual_seed(seed)
            noise = torch.randn(B, K, self.action_dim, device=states.device, dtype=states.dtype, generator=g)
        mean_expanded = mean.unsqueeze(1).expand(B, K, self.action_dim)
        std_expanded = std.unsqueeze(1).expand(B, K, self.action_dim)
        actions_unbounded = mean_expanded + std_expanded * noise
        return torch.tanh(actions_unbounded) * self.max_action

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        """CORL 알고리즘과 호환되는 act 메서드"""
        state_tensor = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        action_tensor = self.deterministic_actions(state_tensor)
        return action_tensor.cpu().numpy().flatten()


class DeterministicMLP(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.action_dim = action_dim
        self.max_action = max_action
        self.is_gaussian = False
        self.is_stochastic = False
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.head = nn.Linear(256, action_dim)

    def _forward_action(self, obs):
        x = F.relu(self.l1(obs))
        x = F.relu(self.l2(x))
        return torch.tanh(self.head(x)) * self.max_action

    def forward(self, obs, deterministic: bool = False):
        """action만 반환. deterministic flag는 시그니처 통일용."""
        return self._forward_action(obs)

    @torch.no_grad()
    def deterministic_actions(self, states):
        return self._forward_action(states)

    def sample_actions(self, states, K: int = 1, seed: Optional[int] = None):
        a = self._forward_action(states)
        return a[:, None, :].expand(a.size(0), K, a.size(1)).contiguous()

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        """CORL 알고리즘과 호환되는 act 메서드"""
        import numpy as np
        state_tensor = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        action_tensor = self.deterministic_actions(state_tensor)
        return action_tensor.cpu().numpy().flatten()


class PolicyAdapter(nn.Module):
    """외부 actor를 Policy 인터페이스로 감싸기"""

    def __init__(
        self,
        actor: nn.Module,
        action_dim: int,
        max_action: float,
        uses_z: Optional[bool] = None,
    ):
        super().__init__()
        self.actor = actor
        self.action_dim = action_dim
        self.max_action = max_action

        if uses_z is None:
            sig = inspect.signature(actor.forward)
            param_names = list(sig.parameters.keys())
            z_keywords = {"z", "noise", "latent", "random"}
            self.uses_z = any(
                name.lower() in z_keywords or "z" in name.lower() or "noise" in name.lower()
                for name in param_names[1:]
            )
        else:
            self.uses_z = uses_z

    def _randn(self, shape, device, dtype, seed):
        if seed is None:
            return torch.randn(shape, device=device, dtype=dtype)
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        return torch.randn(shape, device=device, dtype=dtype, generator=g)

    @torch.no_grad()
    def deterministic_actions(self, states):
        if self.uses_z:
            z = torch.zeros((states.size(0), self.action_dim), device=states.device, dtype=states.dtype)
            return self.actor(states, z)
        return self.actor(states)

    def sample_actions(self, states, K: int = 1, seed: Optional[int] = None):
        B = states.size(0)
        if not self.uses_z:
            if K == 1:
                a = self.actor(states)
                return a[:, None, :]
            actions_list = [self.actor(states) for _ in range(K)]
            return torch.stack(actions_list, dim=1)

        z = self._randn((B, K, self.action_dim), device=states.device, dtype=states.dtype, seed=seed)
        states_flat = states[:, None, :].expand(B, K, states.size(1)).reshape(B * K, -1)
        z_flat = z.reshape(B * K, self.action_dim)
        a = self.actor(states_flat, z_flat).reshape(B, K, self.action_dim)
        return a
