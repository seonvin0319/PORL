# Actor Interface (Protocol) and Adapters
# 인터페이스(계약) 먼저 고정

import inspect
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol


class ActorAPI(Protocol):
    """Actor 인터페이스 계약 - 모든 actor가 구현해야 하는 메서드"""
    action_dim: int
    max_action: float
    is_stochastic: bool
    is_gaussian: bool

    def deterministic_actions(self, states: torch.Tensor) -> torch.Tensor:
        """Return [B, A] - deterministic actions"""
        ...

    def sample_actions(
        self, states: torch.Tensor, K: int = 1, seed: Optional[int] = None
    ) -> torch.Tensor:
        """Return [B, K, A] - sampled actions"""
        ...

    def log_prob_actions(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Return [B] - log probabilities (deterministic은 0 반환)"""
        ...


class Policy(Protocol):
    """Legacy Protocol for backward compatibility"""
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


class PolicyAdapter(nn.Module):
    """외부 actor를 ActorAPI 인터페이스로 감싸기 (레거시 호환용)"""

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
        self.is_stochastic = getattr(actor, "is_stochastic", False)
        self.is_gaussian = getattr(actor, "is_gaussian", False)

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
        from .mlp import create_generator
        if seed is None:
            return torch.randn(shape, device=device, dtype=dtype)
        g = create_generator(device, seed)
        return torch.randn(shape, device=device, dtype=dtype, generator=g)

    @torch.no_grad()
    def deterministic_actions(self, states: torch.Tensor) -> torch.Tensor:
        """Return [B, A]"""
        if self.uses_z:
            z = torch.zeros((states.size(0), self.action_dim), device=states.device, dtype=states.dtype)
            return self.actor(states, z)
        return self.actor(states)

    def sample_actions(
        self, states: torch.Tensor, K: int = 1, seed: Optional[int] = None
    ) -> torch.Tensor:
        """Return [B, K, A]"""
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

    def log_prob_actions(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Return [B] - 레거시 actor의 log_prob 메서드 호출"""
        if hasattr(self.actor, "log_prob"):
            lp = self.actor.log_prob(states, actions)
            if lp.dim() > 1:
                lp = lp.squeeze(-1)
            return lp
        elif hasattr(self.actor, "log_prob_actions"):
            lp = self.actor.log_prob_actions(states, actions)
            if lp.dim() > 1:
                lp = lp.squeeze(-1)
            return lp
        else:
            # Deterministic fallback
            return torch.zeros(actions.size(0), device=actions.device)
