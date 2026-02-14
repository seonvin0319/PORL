# JAX MLP utilities
# MLP building and initialization utilities for JAX/Flax

import math
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn


def pytorch_init(fan_in: float) -> Callable:
    """
    Default init for PyTorch Linear layer weights and biases:
    https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    """
    bound = math.sqrt(1 / fan_in)

    def _init(key: jax.random.PRNGKey, shape: Tuple, dtype: type) -> jax.Array:
        return jax.random.uniform(
            key, shape=shape, minval=-bound, maxval=bound, dtype=dtype
        )

    return _init


def uniform_init(bound: float) -> Callable:
    def _init(key: jax.random.PRNGKey, shape: Tuple, dtype: type) -> jax.Array:
        return jax.random.uniform(
            key, shape=shape, minval=-bound, maxval=bound, dtype=dtype
        )

    return _init


def identity(x: Any) -> Any:
    return x


def compute_mean_std(states: jax.Array, eps: float = 1e-3) -> Tuple[jax.Array, jax.Array]:
    """Compute mean and std of states"""
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: jax.Array, mean: jax.Array, std: jax.Array) -> jax.Array:
    """Normalize states using mean and std"""
    return (states - mean) / std


def build_mlp_layers(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    n_hiddens: int,
    layernorm: bool = False,
    final_activation=None,
) -> list:
    """공통 MLP 레이어 빌딩 함수
    
    Args:
        input_dim: 입력 차원 (kernel init scale 계산용, 실제 Dense shape 추론은 Flax가 자동 수행)
        hidden_dim: Hidden layer 차원
        output_dim: 출력 차원
        n_hiddens: Hidden layer 개수
        layernorm: LayerNorm 적용 여부
        final_activation: 마지막 레이어 activation (None이면 적용 안 함)
    
    Returns:
        레이어 리스트
    """
    s_d, h_d = input_dim, hidden_dim
    layers = [
        nn.Dense(
            hidden_dim,
            kernel_init=pytorch_init(s_d),
            bias_init=nn.initializers.constant(0.1),
        ),
        nn.relu,
        nn.LayerNorm() if layernorm else identity,
    ]
    for _ in range(n_hiddens - 1):
        layers += [
            nn.Dense(
                hidden_dim,
                kernel_init=pytorch_init(h_d),
                bias_init=nn.initializers.constant(0.1),
            ),
            nn.relu,
            nn.LayerNorm() if layernorm else identity,
        ]
    layers += [
        nn.Dense(
            output_dim,
            kernel_init=uniform_init(1e-3),
            bias_init=uniform_init(1e-3),
        ),
    ]
    if final_activation is not None:
        layers.append(final_activation)
    return layers


def build_base_network(
    input_dim: int,
    hidden_dim: int,
    n_hiddens: int,
    layernorm: bool = False,
) -> nn.Module:
    """Base network builder for Gaussian/TanhGaussian policies
    
    Args:
        input_dim: Input dimension (kernel init scale 계산용)
        hidden_dim: Hidden layer dimension
        n_hiddens: Number of hidden layers
        layernorm: Whether to apply layer normalization
    
    Returns:
        Base network Sequential module (output: hidden_dim)
    """
    layers = []
    # First layer
    layers.append(
        nn.Dense(
            hidden_dim,
            kernel_init=pytorch_init(input_dim),
            bias_init=nn.initializers.constant(0.1),
        )
    )
    layers.append(nn.relu)
    if layernorm:
        layers.append(nn.LayerNorm())
    
    # Hidden layers
    for _ in range(n_hiddens - 1):
        layers.append(
            nn.Dense(
                hidden_dim,
                kernel_init=pytorch_init(hidden_dim),
                bias_init=nn.initializers.constant(0.1),
            )
        )
        layers.append(nn.relu)
        if layernorm:
            layers.append(nn.LayerNorm())
    
    return nn.Sequential(layers)
