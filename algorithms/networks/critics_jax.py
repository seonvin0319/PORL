# JAX Critic implementations
# Critic, EnsembleCritic, DetActor (ReBRAC용)

import math
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

from .mlp_jax import pytorch_init, uniform_init, identity


class DetActor(nn.Module):
    """ReBRAC Deterministic Actor
    
    ReBRAC 알고리즘에서 사용하는 deterministic actor.
    POGO Multi-Actor에서도 사용 가능.
    """
    action_dim: int
    hidden_dim: int = 256
    layernorm: bool = True
    n_hiddens: int = 3

    @nn.compact
    def __call__(self, state: jax.Array) -> jax.Array:
        s_d, h_d = state.shape[-1], self.hidden_dim
        # Initialization as in the EDAC paper
        layers = [
            nn.Dense(
                self.hidden_dim,
                kernel_init=pytorch_init(s_d),
                bias_init=nn.initializers.constant(0.1),
            ),
            nn.relu,
            nn.LayerNorm() if self.layernorm else identity,
        ]
        for _ in range(self.n_hiddens - 1):
            layers += [
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=pytorch_init(h_d),
                    bias_init=nn.initializers.constant(0.1),
                ),
                nn.relu,
                nn.LayerNorm() if self.layernorm else identity,
            ]
        layers += [
            nn.Dense(
                self.action_dim,
                kernel_init=uniform_init(1e-3),
                bias_init=uniform_init(1e-3),
            ),
            nn.tanh,
        ]
        net = nn.Sequential(layers)
        actions = net(state)
        return actions


class Critic(nn.Module):
    """Single Critic Network for JAX
    
    ReBRAC 및 기타 알고리즘에서 사용하는 critic network.
    """
    hidden_dim: int = 256
    layernorm: bool = True
    n_hiddens: int = 3

    @nn.compact
    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        s_d, a_d, h_d = state.shape[-1], action.shape[-1], self.hidden_dim
        # Initialization as in the EDAC paper
        layers = [
            nn.Dense(
                self.hidden_dim,
                kernel_init=pytorch_init(s_d + a_d),
                bias_init=nn.initializers.constant(0.1),
            ),
            nn.relu,
            nn.LayerNorm() if self.layernorm else identity,
        ]
        for _ in range(self.n_hiddens - 1):
            layers += [
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=pytorch_init(h_d),
                    bias_init=nn.initializers.constant(0.1),
                ),
                nn.relu,
                nn.LayerNorm() if self.layernorm else identity,
            ]
        layers += [
            nn.Dense(1, kernel_init=uniform_init(3e-3), bias_init=uniform_init(3e-3))
        ]
        network = nn.Sequential(layers)
        state_action = jnp.hstack([state, action])
        out = network(state_action).squeeze(-1)
        return out


class EnsembleCritic(nn.Module):
    """Ensemble Critic Network for JAX
    
    여러 개의 Critic을 ensemble로 묶은 네트워크.
    ReBRAC, EDAC 등에서 사용.
    """
    hidden_dim: int = 256
    num_critics: int = 10
    layernorm: bool = True
    n_hiddens: int = 3

    @nn.compact
    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        ensemble = nn.vmap(
            target=Critic,
            in_axes=None,
            out_axes=0,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            axis_size=self.num_critics,
        )
        q_values = ensemble(self.hidden_dim, self.layernorm, self.n_hiddens)(
            state, action
        )
        return q_values
