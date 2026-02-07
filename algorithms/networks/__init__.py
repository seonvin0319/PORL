# Networks package - Re-export for backward compatibility
from .adapters import ActorAPI, Policy, PolicyAdapter
from .mlp import (
    build_mlp_layers,
    build_mlp,
    init_module_weights,
    extend_and_repeat,
    create_generator,
)
from .actors import (
    BaseActor,
    BasePolicyMLP,
    StochasticMLP,
    GaussianMLP,
    TanhGaussianMLP,
    DeterministicMLP,
)
from .critics import (
    FullyConnectedQFunction,
    Scalar,
    EnsembleCritic,
    VectorizedLinear,
    Squeeze,
)

__all__ = [
    # Protocols
    "ActorAPI",
    "Policy",
    # Adapters
    "PolicyAdapter",
    # MLP utilities
    "build_mlp_layers",
    "build_mlp",
    "init_module_weights",
    "extend_and_repeat",
    "create_generator",
    # Actors
    "BaseActor",
    "BasePolicyMLP",
    "StochasticMLP",
    "GaussianMLP",
    "TanhGaussianMLP",
    "DeterministicMLP",
    # Critics
    "FullyConnectedQFunction",
    "Scalar",
    "EnsembleCritic",
    "VectorizedLinear",
    "Squeeze",
]
