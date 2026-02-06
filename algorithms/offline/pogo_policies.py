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

    def forward(self, states, z=None):
        """Forward pass. If z is None, uses deterministic_actions (z=0)"""
        if z is None:
            # AWAC 등에서 states만 전달하는 경우를 위해 z=0 사용
            z = torch.zeros((states.size(0), self.action_dim), device=states.device, dtype=states.dtype)
        x = torch.cat([states, z], dim=1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return torch.tanh(self.head(x)) * self.max_action

    def __call__(self, states, *args, **kwargs):
        """AWAC 등과 호환: (states) -> (action, log_prob)"""
        if len(args) == 0 and len(kwargs) == 0:
            # states만 전달된 경우 (AWAC 등)
            action = self.deterministic_actions(states)
            # log_prob는 0으로 반환 (deterministic이므로)
            log_prob = torch.zeros(action.size(0), device=action.device)
            return action, log_prob
        else:
            # 원래 forward 호출
            return super().__call__(states, *args, **kwargs)

    @torch.no_grad()
    def act(self, state, device: str = "cpu"):
        """CORL 알고리즘과 호환되는 act 메서드"""
        import numpy as np
        state_tensor = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        action_tensor = self.deterministic_actions(state_tensor)
        return action_tensor.cpu().numpy().flatten()


class GaussianMLP(nn.Module):
    """Gaussian Policy: mean에 tanh가 적용된 상태에서 샘플링
    mean = tanh(...) * max_action (bounded mean)
    그 mean, std로 Gaussian 샘플링
    Closed form W2 distance 사용 가능
    """
    LOG_STD_MIN = -20.0
    LOG_STD_MAX = 2.0

    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.action_dim = action_dim
        self.max_action = max_action
        self.is_gaussian = True  # Closed form W2 사용 가능
        self.is_stochastic = True
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.head = nn.Linear(256, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))

    def forward(self, states):
        """Forward pass: state -> mean (with tanh)"""
        x = F.relu(self.l1(states))
        x = F.relu(self.l2(x))
        mean = torch.tanh(self.head(x)) * self.max_action
        return mean

    def get_mean_std(self, states):
        """Get mean and std: (mean, std)
        mean은 이미 tanh space에 있음 (bounded)
        """
        mean = self.forward(states)
        log_std = self.log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        # Broadcast std to batch size
        std = std.unsqueeze(0).expand(mean.size(0), -1)
        return mean, std

    @torch.no_grad()
    def deterministic_actions(self, states):
        """Deterministic action: mean"""
        return self.forward(states)

    def sample_actions(self, states, K: int = 1, seed: Optional[int] = None):
        """Sample K actions: [B, K, action_dim]
        mean에 tanh가 적용된 상태에서 샘플링
        """
        mean, std = self.get_mean_std(states)  # [B, action_dim]
        B = states.size(0)
        
        # Sample noise
        if seed is None:
            noise = torch.randn(B, K, self.action_dim, device=states.device, dtype=states.dtype)
        else:
            g = torch.Generator(device=states.device)
            g.manual_seed(seed)
            noise = torch.randn(B, K, self.action_dim, device=states.device, dtype=states.dtype, generator=g)
        
        # Expand mean and std: [B, K, action_dim]
        mean_expanded = mean.unsqueeze(1).expand(B, K, self.action_dim)
        std_expanded = std.unsqueeze(1).expand(B, K, self.action_dim)
        
        # Sample actions (mean은 이미 bounded)
        actions = mean_expanded + std_expanded * noise
        actions = torch.clamp(actions, -self.max_action, self.max_action)
        return actions

    def __call__(self, states, *args, **kwargs):
        """AWAC 등과 호환: (states) -> (action, log_prob)"""
        if len(args) == 0 and len(kwargs) == 0:
            # states만 전달된 경우 (AWAC 등)
            action = self.deterministic_actions(states)
            # log_prob는 0으로 반환 (간단화)
            log_prob = torch.zeros(action.size(0), device=action.device)
            return action, log_prob
        else:
            # 원래 forward 호출
            return super().__call__(states, *args, **kwargs)

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

    def forward(self, states):
        """Forward pass: state -> mean (unbounded)"""
        x = F.relu(self.l1(states))
        x = F.relu(self.l2(x))
        mean = self.head(x)  # Unbounded
        return mean

    def get_mean_std(self, states):
        """Get mean and std: (mean, std)
        mean은 unbounded space에 있음
        """
        mean = self.forward(states)
        log_std = self.log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        # Broadcast std to batch size
        std = std.unsqueeze(0).expand(mean.size(0), -1)
        return mean, std

    @torch.no_grad()
    def deterministic_actions(self, states):
        """Deterministic action: tanh(mean) * max_action"""
        mean = self.forward(states)
        return torch.tanh(mean) * self.max_action

    def sample_actions(self, states, K: int = 1, seed: Optional[int] = None):
        """Sample K actions: [B, K, action_dim]
        unbounded Gaussian에서 샘플링 후 tanh 적용
        """
        mean, std = self.get_mean_std(states)  # [B, action_dim]
        B = states.size(0)
        
        # Sample noise
        if seed is None:
            noise = torch.randn(B, K, self.action_dim, device=states.device, dtype=states.dtype)
        else:
            g = torch.Generator(device=states.device)
            g.manual_seed(seed)
            noise = torch.randn(B, K, self.action_dim, device=states.device, dtype=states.dtype, generator=g)
        
        # Expand mean and std: [B, K, action_dim]
        mean_expanded = mean.unsqueeze(1).expand(B, K, self.action_dim)
        std_expanded = std.unsqueeze(1).expand(B, K, self.action_dim)
        
        # Sample actions from unbounded Gaussian
        actions_unbounded = mean_expanded + std_expanded * noise
        
        # Apply tanh to make bounded
        actions = torch.tanh(actions_unbounded) * self.max_action
        return actions

    def __call__(self, states, *args, **kwargs):
        """AWAC 등과 호환: (states) -> (action, log_prob)"""
        if len(args) == 0 and len(kwargs) == 0:
            action = self.deterministic_actions(states)
            log_prob = torch.zeros(action.size(0), device=action.device)
            return action, log_prob
        else:
            return super().__call__(states, *args, **kwargs)

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

    def forward(self, states):
        x = F.relu(self.l1(states))
        x = F.relu(self.l2(x))
        return torch.tanh(self.head(x)) * self.max_action

    def __call__(self, states, *args, **kwargs):
        """AWAC 등과 호환: (states) -> (action, log_prob)"""
        if len(args) == 0 and len(kwargs) == 0:
            # states만 전달된 경우 (AWAC 등)
            action = self.forward(states)
            # log_prob는 0으로 반환 (deterministic이므로)
            log_prob = torch.zeros(action.size(0), device=action.device)
            return action, log_prob
        else:
            # 원래 forward 호출
            return super().__call__(states, *args, **kwargs)

    @torch.no_grad()
    def deterministic_actions(self, states):
        return self.forward(states)

    def sample_actions(self, states, K: int = 1, seed: Optional[int] = None):
        a = self.forward(states)
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
