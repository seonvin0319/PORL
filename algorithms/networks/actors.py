# Actor Networks - 행동 분포/샘플링/로그확률
# log_prob_actions를 모든 actor에 제공 (deterministic은 0 반환)
# __call__ 오버라이드 최소화, shape 통일, seed 생성기 공통화

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .mlp import build_mlp_layers, create_generator, init_module_weights


# ============================================================================
# Base Classes
# ============================================================================

class BaseActor(nn.Module):
    """Base class for actors using z (noise) input"""
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super().__init__()
        self.action_dim = action_dim
        self.max_action = max_action

    def _randn(self, shape: Tuple[int, ...], device, dtype, seed: Optional[int]):
        """Seed 생성기 공통화"""
        if seed is None:
            return torch.randn(shape, device=device, dtype=dtype)
        g = create_generator(device, seed)
        return torch.randn(shape, device=device, dtype=dtype, generator=g)

    @torch.no_grad()
    def deterministic_actions(self, states: torch.Tensor) -> torch.Tensor:
        """Return [B, A] - deterministic actions"""
        B = states.size(0)
        z = torch.zeros((B, self.action_dim), device=states.device, dtype=states.dtype)
        return self.forward(states, z)

    def sample_actions(
        self, states: torch.Tensor, K: int = 1, seed: Optional[int] = None
    ) -> torch.Tensor:
        """Return [B, K, A] - sampled actions"""
        B = states.size(0)
        z = self._randn((B, K, self.action_dim), device=states.device, dtype=states.dtype, seed=seed)
        states_flat = states[:, None, :].expand(B, K, states.size(1)).reshape(B * K, -1)
        z_flat = z.reshape(B * K, self.action_dim)
        a = self.forward(states_flat, z_flat).reshape(B, K, self.action_dim)
        return a

    def forward(self, states: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def log_prob_actions(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Return [B] - BaseActor는 log_prob 제공 불가 (0 반환)"""
        return torch.zeros(actions.size(0), device=actions.device)

    @torch.no_grad()
    def act(self, state, device: str = "cpu"):
        """CORL 알고리즘과 호환되는 act 메서드"""
        state_tensor = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        action_tensor = self.deterministic_actions(state_tensor)
        return action_tensor.cpu().numpy().flatten()


class BasePolicyMLP(nn.Module):
    """Base class for policy MLPs. build_mlp_layers로 base Linear만 받고, activation은 forward에서 명시."""
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hiddens: int = 2,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.max_action = max_action
        self.n_hiddens = n_hiddens
        self.dropout = dropout

        base_layers, head_input_dim = build_mlp_layers(state_dim, hidden_dim, n_hiddens)
        for i, layer in enumerate(base_layers):
            setattr(self, f'l{i+1}', layer)
        self.head_input_dim = head_input_dim

    def _forward_base(self, x: torch.Tensor) -> torch.Tensor:
        """Base layers: Linear + ReLU (policy는 항상 F.relu 사용, layernorm/activation 옵션 없음)"""
        for i in range(self.n_hiddens):
            x = F.relu(getattr(self, f'l{i+1}')(x))
            if self.dropout is not None:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        """CORL 알고리즘과 호환되는 act 메서드"""
        state_tensor = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        action_tensor = self.deterministic_actions(state_tensor)
        return action_tensor.cpu().numpy().flatten()


# ============================================================================
# Actor Implementations
# ============================================================================

class StochasticMLP(BaseActor):
    """Stochastic actor using z (noise) input"""
    is_stochastic = True
    is_gaussian = False

    def __init__(
        self, state_dim, action_dim, max_action, hidden_dim: int = 256, n_hiddens: int = 2
    ):
        super().__init__(state_dim, action_dim, max_action)

        # 공통 MLP 레이어 빌딩 함수 사용
        input_dim = state_dim + action_dim
        base_layers, head_input_dim = build_mlp_layers(input_dim, hidden_dim, n_hiddens)
        for i, layer in enumerate(base_layers):
            setattr(self, f'l{i+1}', layer)
        self.head = nn.Linear(head_input_dim, action_dim)
        self.n_hiddens = n_hiddens

    def forward(self, states, z=None):
        """Forward pass. If z is None, uses deterministic_actions (z=0)"""
        if z is None:
            z = torch.zeros((states.size(0), self.action_dim), device=states.device, dtype=states.dtype)
        x = torch.cat([states, z], dim=1)
        # 공통 레이어 통과
        for i in range(self.n_hiddens):
            x = F.relu(getattr(self, f'l{i+1}')(x))
        return torch.tanh(self.head(x)) * self.max_action

    def log_prob_actions(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Return [B] - StochasticMLP는 log_prob 제공 불가 (0 반환)"""
        return torch.zeros(actions.size(0), device=actions.device)


class DeterministicMLP(BasePolicyMLP):
    """Deterministic policy"""
    is_stochastic = False
    is_gaussian = False

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        hidden_dim: int = 256,
        n_hiddens: int = 2,
        dropout: Optional[float] = None,
    ):
        super().__init__(state_dim, action_dim, max_action, hidden_dim, n_hiddens, dropout)
        self.head = nn.Linear(self.head_input_dim, action_dim)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Return [B, A]"""
        x = self._forward_base(states)
        return torch.tanh(self.head(x)) * self.max_action

    @torch.no_grad()
    def deterministic_actions(self, states: torch.Tensor) -> torch.Tensor:
        """Return [B, A]"""
        return self.forward(states)

    def sample_actions(
        self, states: torch.Tensor, K: int = 1, seed: Optional[int] = None
    ) -> torch.Tensor:
        """Return [B, K, A] - shape 통일"""
        a = self.forward(states)  # [B, A]
        return a.unsqueeze(1).expand(-1, K, -1).contiguous()  # [B, K, A]

    def log_prob_actions(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Return [B] - deterministic은 0 반환"""
        return torch.zeros(actions.size(0), device=actions.device)


class GaussianMLP(BasePolicyMLP):
    """Gaussian Policy: mean에 tanh 적용 여부를 선택 가능
    tanh_mean=True (기본값): mean = tanh(...) * max_action (bounded mean)
    tanh_mean=False (AWAC용): mean = ... (unbounded mean, 샘플링 후 clamp)
    Closed form W2 distance 사용 가능
    """
    LOG_STD_MIN = -20.0
    LOG_STD_MAX = 2.0
    is_stochastic = True
    is_gaussian = True  # Closed form W2 사용 가능

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        hidden_dim: int = 256,
        n_hiddens: int = 2,
        tanh_mean: bool = True,
        dropout: Optional[float] = None,
    ):
        super().__init__(state_dim, action_dim, max_action, hidden_dim, n_hiddens, dropout)
        self.tanh_mean = tanh_mean
        self.head = nn.Linear(self.head_input_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))

    def _get_mean(self, states: torch.Tensor) -> torch.Tensor:
        """Internal: state -> mean (bounded if tanh_mean else unbounded)"""
        x = self._forward_base(states)
        mean_raw = self.head(x)
        if self.tanh_mean:
            mean = torch.tanh(mean_raw) * self.max_action
        else:
            mean = mean_raw
        return mean

    def get_mean_std(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get mean and std: (mean, std)"""
        mean = self._get_mean(states)
        log_std = self.log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        std = std.unsqueeze(0).expand(mean.size(0), -1)
        return mean, std

    def action_and_log_prob(
        self, states: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward: state -> (action, log_prob)
        
        Args:
            states: [B, state_dim]
            deterministic: True면 mean 기반 action
        
        Returns:
            action: [B, action_dim]
            log_prob: [B]
        """
        mean, std = self.get_mean_std(states)
        policy_dist = Normal(mean, std)

        if deterministic:
            action = mean.clamp(-self.max_action, self.max_action)
        else:
            action = policy_dist.rsample()
            action = action.clamp(-self.max_action, self.max_action)

        log_prob = policy_dist.log_prob(action).sum(-1)
        return action, log_prob

    @torch.no_grad()
    def deterministic_actions(self, states: torch.Tensor) -> torch.Tensor:
        """Return [B, A]"""
        return self._get_mean(states)

    def sample_actions(
        self, states: torch.Tensor, K: int = 1, seed: Optional[int] = None
    ) -> torch.Tensor:
        """Return [B, K, A] - shape 통일"""
        mean, std = self.get_mean_std(states)  # [B, action_dim]
        B = states.size(0)

        # Sample noise - seed 생성기 공통화
        if seed is None:
            noise = torch.randn(B, K, self.action_dim, device=states.device, dtype=states.dtype)
        else:
            g = create_generator(states.device, seed)
            noise = torch.randn(B, K, self.action_dim, device=states.device, dtype=states.dtype, generator=g)

        # Expand mean and std: [B, K, action_dim]
        mean_expanded = mean.unsqueeze(1).expand(B, K, self.action_dim)
        std_expanded = std.unsqueeze(1).expand(B, K, self.action_dim)

        # Sample actions
        actions = mean_expanded + std_expanded * noise
        actions = torch.clamp(actions, -self.max_action, self.max_action)
        return actions

    def log_prob_actions(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Return [B] - shape 통일 (내부에서 squeeze)"""
        mean, std = self.get_mean_std(states)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(-1)  # [B]
        return log_prob


class TanhGaussianMLP(BasePolicyMLP):
    """TanhGaussian Policy: unbounded Gaussian에서 샘플링 후 tanh 적용
    mean = ... (unbounded)
    mean, std로 Gaussian 샘플링 (unbounded space)
    그 다음 tanh를 적용하여 bounded로 만듦
    Closed form W2 사용 불가 (Sinkhorn 사용)
    """
    LOG_STD_MIN = -20.0
    LOG_STD_MAX = 2.0
    is_stochastic = True
    is_gaussian = False

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        hidden_dim: int = 256,
        n_hiddens: int = 2,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        separate_mu_logstd: bool = False,
        log_std_multiplier: float = 1.0,
        log_std_offset: float = 0.0,
        sac_edac_init: bool = False,
        cql_style: bool = False,
        orthogonal_init: bool = False,
    ):
        super().__init__(state_dim, action_dim, max_action, hidden_dim, n_hiddens)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.separate_mu_logstd = separate_mu_logstd
        self.log_std_multiplier = log_std_multiplier
        self.log_std_offset = log_std_offset
        self.cql_style = cql_style

        if cql_style:
            self.head = nn.Linear(self.head_input_dim, 2 * action_dim)
        elif separate_mu_logstd:
            self.mu = nn.Linear(self.head_input_dim, action_dim)
            self.log_std_head = nn.Linear(self.head_input_dim, action_dim)
        else:
            self.head = nn.Linear(self.head_input_dim, action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))

        # 초기화 정책 중앙화
        if orthogonal_init:
            for layer in [getattr(self, f'l{i+1}') for i in range(n_hiddens)]:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=1.0)
                    nn.init.constant_(layer.bias, 0.0)
            if self.cql_style or (hasattr(self, 'head') and not self.separate_mu_logstd):
                nn.init.orthogonal_(self.head.weight, gain=1e-2)
                nn.init.constant_(self.head.bias, 0.0)
        elif sac_edac_init:
            for layer in [getattr(self, f'l{i+1}') for i in range(n_hiddens)]:
                if isinstance(layer, nn.Linear):
                    torch.nn.init.constant_(layer.bias, 0.1)
            if self.separate_mu_logstd:
                torch.nn.init.uniform_(self.mu.weight, -1e-3, 1e-3)
                torch.nn.init.uniform_(self.mu.bias, -1e-3, 1e-3)
                torch.nn.init.uniform_(self.log_std_head.weight, -1e-3, 1e-3)
                torch.nn.init.uniform_(self.log_std_head.bias, -1e-3, 1e-3)

    def _get_mean_logstd(
        self, states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get mean and log_std from states (internal method)"""
        x = self._forward_base(states)

        if self.cql_style:
            out = self.head(x)
            mean, log_std = out.chunk(2, dim=-1)
            log_std = self.log_std_multiplier * log_std + self.log_std_offset
            log_std = log_std.clamp(self.log_std_min, self.log_std_max)
        elif self.separate_mu_logstd:
            # SAC-N/EDAC 방식: mu와 log_std를 별도로 출력
            mean = self.mu(x)
            log_std = self.log_std_head(x)
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        else:
            mean = self.head(x)
            log_std = self.log_std.clamp(self.log_std_min, self.log_std_max)
            if log_std.dim() == 1:
                log_std = log_std.unsqueeze(0).expand(mean.size(0), -1)
            if self.log_std_multiplier != 1.0 or self.log_std_offset != 0.0:
                log_std = self.log_std_multiplier * log_std + self.log_std_offset
                log_std = log_std.clamp(self.log_std_min, self.log_std_max)

        return mean, log_std

    def get_mean_std(
        self, states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get mean and std: (mean, std)
        mean은 unbounded space에 있음
        """
        mean, log_std = self._get_mean_logstd(states)
        std = torch.exp(log_std)
        return mean, std

    def action_and_log_prob(
        self, states: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward: state -> (action, log_prob)
        
        Args:
            states: [B, state_dim]
            deterministic: True면 mean 사용, False면 샘플링
        
        Returns:
            action: [B, action_dim]
            log_prob: [B]
        """
        mean, log_std = self._get_mean_logstd(states)
        std = torch.exp(log_std)
        policy_dist = Normal(mean, std)

        if deterministic:
            action_unbounded = mean
        else:
            action_unbounded = policy_dist.rsample()

        tanh_action = torch.tanh(action_unbounded)
        action = tanh_action * self.max_action

        # SAC paper, appendix C, eq 21: change of variables formula
        log_prob = policy_dist.log_prob(action_unbounded).sum(axis=-1)
        log_prob = log_prob - torch.log(1 - tanh_action.pow(2) + 1e-6).sum(axis=-1)

        return action, log_prob

    @torch.no_grad()
    def deterministic_actions(self, states: torch.Tensor) -> torch.Tensor:
        """Return [B, A]"""
        mean, _ = self._get_mean_logstd(states)
        return torch.tanh(mean) * self.max_action

    def sample_actions(
        self, states: torch.Tensor, K: int = 1, seed: Optional[int] = None
    ) -> torch.Tensor:
        """Return [B, K, A] - shape 통일"""
        mean, log_std = self._get_mean_logstd(states)  # [B, action_dim]
        log_std = log_std.clamp(self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        B = states.size(0)

        # Sample noise - seed 생성기 공통화
        if seed is None:
            noise = torch.randn(B, K, self.action_dim, device=states.device, dtype=states.dtype)
        else:
            g = create_generator(states.device, seed)
            noise = torch.randn(B, K, self.action_dim, device=states.device, dtype=states.dtype, generator=g)

        # Expand mean and std: [B, K, action_dim]
        mean_expanded = mean.unsqueeze(1).expand(B, K, self.action_dim)
        std_expanded = std.unsqueeze(1).expand(B, K, self.action_dim)

        # Sample actions from unbounded Gaussian
        actions_unbounded = mean_expanded + std_expanded * noise

        # Apply tanh to make bounded
        actions = torch.tanh(actions_unbounded) * self.max_action
        return actions

    def log_prob_actions(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Return [B] - shape 통일 (내부에서 squeeze)"""
        mean, log_std = self._get_mean_logstd(states)
        std = torch.exp(log_std.clamp(self.log_std_min, self.log_std_max))
        # actions: [-max_action, max_action] -> unbounded: atanh(actions/max_action)
        actions_scaled = actions / (self.max_action + 1e-8)
        actions_scaled = actions_scaled.clamp(-0.9999, 0.9999)
        action_unbounded = 0.5 * (torch.log1p(actions_scaled) - torch.log1p(-actions_scaled))
        log_p = Normal(mean, std).log_prob(action_unbounded).sum(-1)
        log_p = log_p - torch.log(1 - actions_scaled.pow(2) + 1e-6).sum(-1)
        return log_p

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        """CORL 알고리즘과 호환되는 act 메서드
        SAC-N/EDAC 호환: deterministic = not self.training
        """
        state_tensor = torch.tensor(state, device=device, dtype=torch.float32)
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
        deterministic = not self.training
        action, _ = self.action_and_log_prob(state_tensor, deterministic=deterministic)
        return action[0].cpu().numpy()
