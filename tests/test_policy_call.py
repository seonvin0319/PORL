"""
Minimal validation for policy_call utilities.
StochasticMLP / GaussianMLP / TanhGaussianMLP / DeterministicMLP 각각에 대해
get_action이 action=[B,A]를, sample_actions가 [B,K,A]를 반환하는지 확인.
"""
import sys

import torch

# Add project root for imports
sys.path.insert(0, ".")

from algorithms.offline.pogo_policies import (
    DeterministicMLP,
    GaussianMLP,
    StochasticMLP,
    TanhGaussianMLP,
)
from algorithms.networks import act_for_eval, get_action, sample_actions


def test_get_action_deterministic():
    """get_action(..., deterministic=True) -> action [B,A], log_prob or None"""
    B, S, A, max_a = 4, 17, 6, 1.0
    states = torch.randn(B, S)

    for policy_cls, name in [
        (StochasticMLP, "StochasticMLP"),
        (GaussianMLP, "GaussianMLP"),
        (TanhGaussianMLP, "TanhGaussianMLP"),
        (DeterministicMLP, "DeterministicMLP"),
    ]:
        policy = policy_cls(S, A, max_a)
        action, log_prob = get_action(policy, states, deterministic=True)
        assert action.shape == (B, A), f"{name}: expected action {B,A}, got {action.shape}"
        assert log_prob is None or log_prob.shape == (B,), f"{name}: log_prob shape"
        print(f"OK {name}: get_action(deterministic=True) -> action {action.shape}")


def test_get_action_stochastic():
    """get_action(..., deterministic=False, need_log_prob=True) -> (action, log_prob)"""
    B, S, A, max_a = 4, 17, 6, 1.0
    states = torch.randn(B, S)

    for policy_cls, name in [
        (StochasticMLP, "StochasticMLP"),
        (GaussianMLP, "GaussianMLP"),
        (TanhGaussianMLP, "TanhGaussianMLP"),
        (DeterministicMLP, "DeterministicMLP"),
    ]:
        policy = policy_cls(S, A, max_a)
        action, log_prob = get_action(
            policy, states, deterministic=False, need_log_prob=True
        )
        assert action.shape == (B, A), f"{name}: expected action {B,A}, got {action.shape}"
        assert log_prob is not None, f"{name}: need log_prob"
        assert log_prob.shape == (B,) or log_prob.numel() == B
        print(f"OK {name}: get_action(deterministic=False, need_log_prob=True) -> action {action.shape}")


def test_sample_actions():
    """sample_actions(policy, states, K) -> [B, K, A]"""
    B, S, A, K, max_a = 4, 17, 6, 3, 1.0
    states = torch.randn(B, S)

    for policy_cls, name in [
        (StochasticMLP, "StochasticMLP"),
        (GaussianMLP, "GaussianMLP"),
        (TanhGaussianMLP, "TanhGaussianMLP"),
        (DeterministicMLP, "DeterministicMLP"),
    ]:
        policy = policy_cls(S, A, max_a)
        actions = sample_actions(policy, states, K=K)
        assert actions.shape == (B, K, A), f"{name}: expected {B,K,A}, got {actions.shape}"
        print(f"OK {name}: sample_actions(K={K}) -> {actions.shape}")


def test_act_for_eval():
    """act_for_eval(policy, state_numpy, device) -> numpy action [A]"""
    import numpy as np

    S, A, max_a = 17, 6, 1.0
    state_np = np.random.randn(S).astype(np.float32)

    for policy_cls, name in [
        (StochasticMLP, "StochasticMLP"),
        (GaussianMLP, "GaussianMLP"),
        (TanhGaussianMLP, "TanhGaussianMLP"),
        (DeterministicMLP, "DeterministicMLP"),
    ]:
        policy = policy_cls(S, A, max_a)
        action = act_for_eval(policy, state_np, device="cpu")
        assert isinstance(action, np.ndarray), f"{name}: expected ndarray"
        assert action.shape == (A,), f"{name}: expected ({A},), got {action.shape}"
        print(f"OK {name}: act_for_eval -> numpy {action.shape}")


if __name__ == "__main__":
    print("Testing policy_call utilities...")
    test_get_action_deterministic()
    test_get_action_stochastic()
    test_sample_actions()
    test_act_for_eval()
    print("All tests passed.")
