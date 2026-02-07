# POGO Network implementations (Policy and Critic)
# Policy Protocol: sample_actions, deterministic_actions
# BaseActor: z를 사용하는 stochastic actor
# StochasticMLP, DeterministicMLP, PolicyAdapter
# Critic networks: TwinQ, ValueFunction, Critic, FullyConnectedQFunction, VectorizedCritic, EnsembleCritic

import inspect
import math
from typing import Optional, Tuple, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol


# ============================================================================
# 공통 유틸리티 함수
# ============================================================================

def build_mlp_layers(
    input_dim: int,
    hidden_dim: int = 256,
    n_hiddens: int = 2,
    output_dim: Optional[int] = None,
    layernorm: bool = False,
    activation: type = nn.ReLU,
):
    """MLP 레이어 빌딩 함수.
    
    - Policy용 (output_dim=None): base Linear layers만 반환. activation은 _forward_base에서 F.relu로 명시.
    - Critic용 (output_dim 설정): Sequential용 레이어 리스트 반환. activation은 nn.Module 클래스로 받음.
    
    Args:
        input_dim: 입력 차원
        hidden_dim: Hidden layer 차원
        n_hiddens: Hidden layer 개수
        output_dim: 출력 차원 (None이면 policy용 base layers만)
        layernorm: LayerNorm 사용 여부 (critic용)
        activation: nn.Module 클래스 (기본 nn.ReLU). F.relu 같은 함수는 사용 불가.
    
    Returns:
        output_dim이 None이면: (base_layers, head_input_dim)
        output_dim이 있으면: 완전한 레이어 리스트 (activation, layernorm 포함)
    """
    layers = []
    current_dim = input_dim

    for _ in range(n_hiddens):
        layers.append(nn.Linear(current_dim, hidden_dim))
        current_dim = hidden_dim

    if output_dim is None:
        return layers, current_dim

    # Critic용: Sequential에 넣을 레이어 (activation은 nn.Module 클래스로 인스턴스화)
    result_layers = []
    for layer in layers:
        result_layers.append(layer)
        result_layers.append(activation())
        if layernorm:
            result_layers.append(nn.LayerNorm(hidden_dim))

    result_layers.append(nn.Linear(hidden_dim, output_dim))
    return result_layers


def build_mlp(
    input_dim: int,
    hidden_dim: int = 256,
    n_hiddens: int = 3,
    output_dim: int = 1,
    layernorm: bool = False,
    activation: type = nn.ReLU,
) -> list:
    """MLP 레이어 빌딩 유틸리티 (Critic 및 기타 네트워크용)
    build_mlp_layers를 사용하여 구현
    
    Args:
        input_dim: 입력 차원
        hidden_dim: Hidden layer 차원
        n_hiddens: Hidden layer 개수
        output_dim: 출력 차원 (기본값: 1)
        layernorm: LayerNorm 사용 여부
        activation: 활성화 함수 클래스 (기본값: nn.ReLU)
    
    Returns:
        레이어 리스트
    """
    return build_mlp_layers(
        input_dim, hidden_dim, n_hiddens, output_dim, layernorm, activation
    )


# ============================================================================
# Policy Protocol 및 Base Classes
# ============================================================================

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
    """Base class for actors using z (noise) input"""
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

    @torch.no_grad()
    def act(self, state, device: str = "cpu"):
        """CORL 알고리즘과 호환되는 act 메서드"""
        import numpy as np
        state_tensor = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        action_tensor = self.deterministic_actions(state_tensor)
        return action_tensor.cpu().numpy().flatten()


class BasePolicyMLP(nn.Module):
    """Base class for policy MLPs. build_mlp_layers로 base Linear만 받고, activation은 forward에서 명시."""
    def __init__(self, state_dim: int, action_dim: int, max_action: float,
                 hidden_dim: int = 256, n_hiddens: int = 2, dropout: Optional[float] = None):
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


class StochasticMLP(BaseActor):
    """Stochastic actor using z (noise) input"""
    def __init__(self, state_dim, action_dim, max_action, hidden_dim: int = 256, n_hiddens: int = 2):
        super().__init__(state_dim, action_dim, max_action)
        self.is_gaussian = False
        self.is_stochastic = True
        
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
            # AWAC 등에서 states만 전달하는 경우를 위해 z=0 사용
            z = torch.zeros((states.size(0), self.action_dim), device=states.device, dtype=states.dtype)
        x = torch.cat([states, z], dim=1)
        # 공통 레이어 통과
        for i in range(self.n_hiddens):
            x = F.relu(getattr(self, f'l{i+1}')(x))
        return torch.tanh(self.head(x)) * self.max_action

    def __call__(self, states, *args, **kwargs):
        """AWAC 등과 호환: (states) -> (action, log_prob)"""
        if len(args) == 0 and len(kwargs) == 0:
            action = self.deterministic_actions(states)
            log_prob = torch.zeros(action.size(0), device=action.device)
            return action, log_prob
        return super().__call__(states, *args, **kwargs)


class GaussianMLP(BasePolicyMLP):
    """Gaussian Policy: mean에 tanh 적용 여부를 선택 가능
    tanh_mean=True (기본값): mean = tanh(...) * max_action (bounded mean)
    tanh_mean=False (AWAC용): mean = ... (unbounded mean, 샘플링 후 clamp)
    Closed form W2 distance 사용 가능
    """
    LOG_STD_MIN = -20.0
    LOG_STD_MAX = 2.0

    def __init__(self, state_dim, action_dim, max_action, hidden_dim: int = 256, n_hiddens: int = 2,
                 tanh_mean: bool = True, dropout: Optional[float] = None):
        super().__init__(state_dim, action_dim, max_action, hidden_dim, n_hiddens, dropout)
        self.tanh_mean = tanh_mean
        self.is_gaussian = True  # Closed form W2 사용 가능
        self.is_stochastic = True
        
        self.head = nn.Linear(self.head_input_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))

    def _get_mean(self, states):
        """Internal: state -> mean (bounded if tanh_mean else unbounded)"""
        x = self._forward_base(states)
        mean_raw = self.head(x)
        if self.tanh_mean:
            mean = torch.tanh(mean_raw) * self.max_action
        else:
            mean = mean_raw
        return mean

    def forward(self, states, deterministic: bool = False, need_log_prob: bool = False):
        """Forward: state -> (action, log_prob). TanhGaussianMLP와 동일 시그니처.
        deterministic=True → mean 기반 action
        deterministic=False → rsample 기반 action
        """
        mean, std = self.get_mean_std(states)
        from torch.distributions import Normal
        policy_dist = Normal(mean, std)

        if deterministic:
            action = mean.clamp(-self.max_action, self.max_action)
        else:
            action = policy_dist.rsample()
            action = action.clamp(-self.max_action, self.max_action)

        log_prob = None
        if need_log_prob:
            log_prob = policy_dist.log_prob(action).sum(-1)

        return action, log_prob

    def get_mean_std(self, states):
        """Get mean and std: (mean, std)"""
        mean = self._get_mean(states)
        log_std = self.log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        std = std.unsqueeze(0).expand(mean.size(0), -1)
        return mean, std

    @torch.no_grad()
    def deterministic_actions(self, states):
        """Deterministic action: mean"""
        return self._get_mean(states)

    def sample_actions(self, states, K: int = 1, seed: Optional[int] = None):
        """Sample K actions: [B, K, action_dim]
        tanh_mean=True: mean에 tanh가 적용된 상태에서 샘플링
        tanh_mean=False: unbounded mean에서 샘플링 후 clamp (AWAC 방식)
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
        
        # Sample actions
        actions = mean_expanded + std_expanded * noise
        # tanh_mean=False인 경우 unbounded mean에서 샘플링했으므로 clamp 필요
        # tanh_mean=True인 경우에도 안전을 위해 clamp
        actions = torch.clamp(actions, -self.max_action, self.max_action)
        return actions

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """AWAC와 호환: log_prob 계산"""
        mean, std = self.get_mean_std(state)
        from torch.distributions import Normal
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        return log_prob
    
    def __call__(self, states, *args, deterministic=False, need_log_prob=False, **kwargs):
        """AWAC 등과 호환: (states) -> (action, log_prob). forward 위임."""
        if len(args) == 0:
            return self.forward(states, deterministic=deterministic, need_log_prob=need_log_prob)
        return super().__call__(states, *args, deterministic=deterministic, need_log_prob=need_log_prob, **kwargs)


class TanhGaussianMLP(BasePolicyMLP):
    """TanhGaussian Policy: unbounded Gaussian에서 샘플링 후 tanh 적용
    mean = ... (unbounded)
    mean, std로 Gaussian 샘플링 (unbounded space)
    그 다음 tanh를 적용하여 bounded로 만듦
    Closed form W2 사용 불가 (Sinkhorn 사용)
    """
    LOG_STD_MIN = -20.0
    LOG_STD_MAX = 2.0

    def __init__(self, state_dim, action_dim, max_action, hidden_dim: int = 256, n_hiddens: int = 2,
                 log_std_min: float = -20.0, log_std_max: float = 2.0, separate_mu_logstd: bool = False,
                 log_std_multiplier: float = 1.0, log_std_offset: float = 0.0,
                 sac_edac_init: bool = False, cql_style: bool = False, orthogonal_init: bool = False):
        super().__init__(state_dim, action_dim, max_action, hidden_dim, n_hiddens)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.separate_mu_logstd = separate_mu_logstd
        self.log_std_multiplier = log_std_multiplier
        self.log_std_offset = log_std_offset
        self.cql_style = cql_style
        self.is_gaussian = False
        self.is_stochastic = True

        if cql_style:
            self.head = nn.Linear(self.head_input_dim, 2 * action_dim)
        elif separate_mu_logstd:
            self.mu = nn.Linear(self.head_input_dim, action_dim)
            self.log_std_head = nn.Linear(self.head_input_dim, action_dim)
        else:
            self.head = nn.Linear(self.head_input_dim, action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))

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

    def _get_mean_logstd(self, states):
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

    def _get_mean(self, states):
        """Get mean from states (internal method, backward compatibility)"""
        mean, _ = self._get_mean_logstd(states)
        return mean

    def forward(self, states, deterministic: bool = False, need_log_prob: bool = False):
        """Forward pass: state -> (action, log_prob)
        
        Args:
            states: [B, state_dim]
            deterministic: True면 mean 사용, False면 샘플링
            need_log_prob: True면 log_prob 계산 (SAC-N/EDAC용)
        
        Returns:
            action: [B, action_dim]
            log_prob: [B] (need_log_prob=True일 때만, None일 수 있음)
        """
        mean, log_std = self._get_mean_logstd(states)
        std = torch.exp(log_std)
        
        from torch.distributions import Normal
        policy_dist = Normal(mean, std)
        
        if deterministic:
            action_unbounded = mean
        else:
            action_unbounded = policy_dist.rsample()
        
        tanh_action = torch.tanh(action_unbounded)
        action = tanh_action * self.max_action
        
        log_prob = None
        if need_log_prob:
            # SAC paper, appendix C, eq 21: change of variables formula
            log_prob = policy_dist.log_prob(action_unbounded).sum(axis=-1)
            log_prob = log_prob - torch.log(1 - tanh_action.pow(2) + 1e-6).sum(axis=-1)
        
        return action, log_prob

    def get_mean_std(self, states):
        """Get mean and std: (mean, std)
        mean은 unbounded space에 있음
        """
        mean, log_std = self._get_mean_logstd(states)
        std = torch.exp(log_std)
        return mean, std

    @torch.no_grad()
    def deterministic_actions(self, states):
        """Deterministic action: tanh(mean) * max_action"""
        mean = self._get_mean(states)
        return torch.tanh(mean) * self.max_action

    def sample_actions(self, states, K: int = 1, seed: Optional[int] = None):
        """Sample K actions: [B, K, action_dim]
        unbounded Gaussian에서 샘플링 후 tanh 적용
        """
        mean, log_std = self._get_mean_logstd(states)  # [B, action_dim]
        log_std = log_std.clamp(self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
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

    def log_prob(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """CQL 호환: TanhGaussian log_prob (bounded action space)"""
        mean, log_std = self._get_mean_logstd(observations)
        std = torch.exp(log_std.clamp(self.log_std_min, self.log_std_max))
        # actions: [-max_action, max_action] -> unbounded: atanh(actions/max_action)
        actions_scaled = actions / (self.max_action + 1e-8)
        actions_scaled = actions_scaled.clamp(-0.9999, 0.9999)
        action_unbounded = 0.5 * (torch.log1p(actions_scaled) - torch.log1p(-actions_scaled))
        from torch.distributions import Normal
        log_p = Normal(mean, std).log_prob(action_unbounded).sum(-1)
        log_p = log_p - torch.log(1 - actions_scaled.pow(2) + 1e-6).sum(-1)
        return log_p
    
    def __call__(self, states, *args, repeat=None, deterministic=False, need_log_prob=False, **kwargs):
        """AWAC, CQL, SAC-N, EDAC 등과 호환: (states) -> (action, log_prob)
        
        Args:
            states: [B, state_dim]
            repeat: CQL에서 사용하는 파라미터 (여러 action 샘플링)
            deterministic: SAC-N/EDAC용
            need_log_prob: SAC-N/EDAC용 (tanh 변환에 대한 log_prob 보정 포함)
        """
        if repeat is not None:
            actions = self.sample_actions(states, K=repeat)  # [B, repeat, action_dim]
            B = actions.size(0)
            actions_flat = actions.reshape(B * repeat, -1)
            states_repeated = states.unsqueeze(1).expand(B, repeat, states.size(1)).reshape(B * repeat, -1)
            log_probs = self.log_prob(states_repeated, actions_flat)
            return actions_flat, log_probs
        
        # SAC-N/EDAC 호환: deterministic, need_log_prob 파라미터 지원
        # args가 비어있거나 deterministic/need_log_prob가 명시적으로 전달된 경우
        if len(args) == 0 or deterministic is not False or need_log_prob is not False:
            return self.forward(states, deterministic=deterministic, need_log_prob=need_log_prob)
        else:
            return super().__call__(states, *args, **kwargs)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        """CORL 알고리즘과 호환되는 act 메서드
        SAC-N/EDAC 호환: deterministic = not self.training
        """
        state_tensor = torch.tensor(state, device=device, dtype=torch.float32)
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
        deterministic = not self.training
        action, _ = self(state_tensor, deterministic=deterministic)
        return action[0].cpu().numpy()


class DeterministicMLP(BasePolicyMLP):
    """Deterministic policy"""
    def __init__(self, state_dim, action_dim, max_action, hidden_dim: int = 256, n_hiddens: int = 2,
                 dropout: Optional[float] = None):
        super().__init__(state_dim, action_dim, max_action, hidden_dim, n_hiddens, dropout)
        self.is_gaussian = False
        self.is_stochastic = False
        
        self.head = nn.Linear(self.head_input_dim, action_dim)

    def forward(self, states):
        x = self._forward_base(states)
        return torch.tanh(self.head(x)) * self.max_action

    def __call__(self, states, *args, deterministic=False, **kwargs):
        """AWAC 등과 호환: (states) -> (action, log_prob). deterministic은 무시."""
        if len(args) == 0:
            action = self.forward(states)
            log_prob = torch.zeros(action.size(0), device=action.device)
            return action, log_prob
        return super().__call__(states, *args, deterministic=deterministic, **kwargs)

    @torch.no_grad()
    def deterministic_actions(self, states):
        return self.forward(states)

    def sample_actions(self, states, K: int = 1, seed: Optional[int] = None):
        a = self.forward(states)
        return a[:, None, :].expand(a.size(0), K, a.size(1)).contiguous()


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


# ============================================================================
# Critic Networks
# ============================================================================

class EnsembleCritic(nn.Module):
    """Ensemble Critic for PyTorch (ReBRAC style)
    Multiple independent critics, returns [num_critics, batch_size] Q values
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_critics: int = 2,
        layernorm: bool = True,
        n_hiddens: int = 3,
    ):
        super().__init__()
        self.num_critics = num_critics
        self.hidden_dim = hidden_dim
        self.layernorm = layernorm
        
        self.critics = nn.ModuleList([
            self._make_critic(state_dim, action_dim, hidden_dim, layernorm, n_hiddens)
            for _ in range(num_critics)
        ])
    
    def _make_critic(self, state_dim, action_dim, hidden_dim, layernorm, n_hiddens):
        """Create a single critic network"""
        layers = []
        layers.append(nn.Linear(state_dim + action_dim, hidden_dim))
        layers.append(nn.ReLU())
        if layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        
        for _ in range(n_hiddens - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            if layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
        
        layers.append(nn.Linear(hidden_dim, 1))
        net = nn.Sequential(*layers)
        linear_layers = [m for m in net.modules() if isinstance(m, nn.Linear)]
        for i, layer in enumerate(linear_layers):
            if i == len(linear_layers) - 1:
                nn.init.uniform_(layer.weight, -3e-3, 3e-3)
                nn.init.uniform_(layer.bias, -3e-3, 3e-3)
            else:
                nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
                nn.init.constant_(layer.bias, 0.1)
        return net
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Returns q_values: [num_critics, batch_size]"""
        state_action = torch.cat([state, action], dim=-1)
        q_values = torch.stack([
            critic(state_action).squeeze(-1) for critic in self.critics
        ], dim=0)
        return q_values


# ============================================================================
# MLP Building Utilities (for Critic and other networks)
# ============================================================================
# build_mlp는 위에 build_mlp_layers와 함께 정의됨


class Squeeze(nn.Module):
    """Squeeze layer for MLP"""
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


class VectorizedLinear(nn.Module):
    """Vectorized Linear layer for ensemble critics (SAC-N, EDAC, LB-SAC용 기본 골격)"""
    def __init__(self, in_features: int, out_features: int, ensemble_size: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size

        self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters"""
        for layer in range(self.ensemble_size):
            nn.init.kaiming_uniform_(self.weight[layer], a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: [ensemble_size, batch_size, input_size] -> [ensemble_size, batch_size, out_size]"""
        return x @ self.weight + self.bias
