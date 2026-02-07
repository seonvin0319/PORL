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
from .policy_call import (
    get_action,
    act_for_eval,
    sample_actions,
    sample_K_actions,
    sample_actions_with_log_prob,
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
    # Policy call utilities
    "get_action",
    "act_for_eval",
    "sample_actions",
    "sample_K_actions",
    "sample_actions_with_log_prob",
]
