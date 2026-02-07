# Critic Networks - Q/V 계열
# TwinQ, ValueFunction, Critic, FullyConnectedQFunction, VectorizedCritic, EnsembleCritic

import math
from typing import Tuple

import torch
import torch.nn as nn

from .mlp import build_mlp, extend_and_repeat, init_module_weights


# ============================================================================
# Critic Utilities
# ============================================================================

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


# ============================================================================
# Critic Networks
# ============================================================================

class FullyConnectedQFunction(nn.Module):
    """Fully connected Q-function for CQL algorithm
    
    Supports multiple actions (3D tensors) and orthogonal initialization.
    """
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        orthogonal_init: bool = False,
        n_hidden_layers: int = 3,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        layers = [
            nn.Linear(observation_dim + action_dim, 256),
            nn.ReLU(),
        ]
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(256, 256))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(256, 1))

        self.network = nn.Sequential(*layers)
        # 초기화 정책 중앙화
        init_module_weights(
            self.network,
            init_type="orthogonal" if orthogonal_init else "xavier",
            orthogonal_gain=1.0,
            last_layer_gain=1e-2,
        )

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Forward pass
        
        Args:
            observations: [batch_size, obs_dim] or [batch_size, ...]
            actions: [batch_size, action_dim] or [batch_size, n_actions, action_dim]
        
        Returns:
            q_values: [batch_size] or [batch_size, n_actions]
        """
        multiple_actions = False
        batch_size = observations.shape[0]
        if actions.ndim == 3 and observations.ndim == 2:
            multiple_actions = True
            observations = extend_and_repeat(observations, 1, actions.shape[1]).reshape(
                -1, observations.shape[-1]
            )
            actions = actions.reshape(-1, actions.shape[-1])
        input_tensor = torch.cat([observations, actions], dim=-1)
        q_values = torch.squeeze(self.network(input_tensor), dim=-1)
        if multiple_actions:
            q_values = q_values.reshape(batch_size, -1)
        return q_values


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
        
        # 초기화 정책 중앙화
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


class Scalar(nn.Module):
    """Scalar parameter module (for learnable temperature/alpha parameters)"""
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> nn.Parameter:
        return self.constant
