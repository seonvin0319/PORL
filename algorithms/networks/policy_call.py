"""
Policy 공통 인터페이스: forward(obs, deterministic=False), action_and_log_prob(obs)
get_action / act_for_eval는 이 인터페이스만 사용.
"""

from typing import Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    import numpy as np


def _ensure_policy(policy: nn.Module, action_dim: Optional[int] = None, max_action: Optional[float] = None) -> nn.Module:
    """Policy 인터페이스가 없으면 PolicyAdapter로 감싸서 반환."""
    if hasattr(policy, "forward") and getattr(policy.forward, "__code__", None):
        try:
            import inspect
            sig = inspect.signature(policy.forward)
            if "deterministic" in sig.parameters:
                return policy
        except Exception:
            pass
    if hasattr(policy, "deterministic_actions") and hasattr(policy, "sample_actions"):
        return policy
    try:
        import inspect
        sig = inspect.signature(getattr(policy, "forward", policy.__call__))
        params = sig.parameters
        if "repeat" in params or "need_log_prob" in params:
            return policy  # CQL/SAC-N/EDAC-style, wrap 하지 않음
    except Exception:
        pass
    from algorithms.offline.pogo_policies import PolicyAdapter

    ad = getattr(policy, "action_dim", action_dim)
    ma = getattr(policy, "max_action", max_action)
    if ad is None or ma is None:
        raise ValueError("PolicyAdapter requires action_dim and max_action.")
    return PolicyAdapter(policy, ad, ma)


def get_action(
    policy: nn.Module,
    states: torch.Tensor,
    deterministic: bool = True,
    *,
    need_log_prob: bool = False,
    action_dim: Optional[int] = None,
    max_action: Optional[float] = None,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    policy에서 action을 얻음. (action [B,A], log_prob or None) 반환.
    공통 인터페이스: forward(obs, deterministic=False), action_and_log_prob(obs)
    """
    policy = _ensure_policy(policy, action_dim, max_action)

    if need_log_prob:
        if hasattr(policy, "action_and_log_prob"):
            action, log_prob = policy.action_and_log_prob(states)
            if log_prob is not None and log_prob.dim() > 1:
                log_prob = log_prob.squeeze(-1)
            return action, log_prob
        action = _forward_action(policy, states, deterministic, seed=seed)
        log_prob = torch.zeros(states.size(0), device=action.device, dtype=action.dtype)
        return action, log_prob

    action = _forward_action(policy, states, deterministic, seed=seed)
    return action, None


def _forward_action(policy: nn.Module, states: torch.Tensor, deterministic: bool, seed: Optional[int] = None) -> torch.Tensor:
    """forward(obs, deterministic=...) 시그니처 통일된 policy에서 action만 반환."""
    try:
        out = policy(states, deterministic=deterministic)
        return out[0] if isinstance(out, (tuple, list)) else out
    except TypeError:
        # 시그니처 불일치(deterministic 미지원)만 fallback, 나머지 에러는 그대로 raise
        pass
    if hasattr(policy, "deterministic_actions") and deterministic:
        return policy.deterministic_actions(states)
    if hasattr(policy, "sample_actions"):
        return policy.sample_actions(states, K=1, seed=seed)[:, 0, :]
    out = policy(states)
    return out[0] if isinstance(out, (tuple, list)) else out


def sample_actions(
    policy: nn.Module,
    states: torch.Tensor,
    K: int = 1,
    seed: Optional[int] = None,
    *,
    action_dim: Optional[int] = None,
    max_action: Optional[float] = None,
) -> torch.Tensor:
    """policy에서 K개 action 샘플: [B, K, action_dim] 반환."""
    policy = _ensure_policy(policy, action_dim, max_action)
    if hasattr(policy, "sample_actions"):
        return policy.sample_actions(states, K=K, seed=seed)
    # forward만 있는 policy: K번 샘플링
    actions_list = []
    for _ in range(K):
        a = _forward_action(policy, states, deterministic=False)
        actions_list.append(a)
    return torch.stack(actions_list, dim=1)


def sample_K_actions(
    policy: nn.Module,
    states: torch.Tensor,
    K: int,
    deterministic: bool = False,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """forward(obs, deterministic=...)로 K개 샘플. [B, K, A] 반환."""
    if hasattr(policy, "sample_actions") and not deterministic:
        return policy.sample_actions(states, K=K, seed=seed)
    actions = []
    for _ in range(K):
        actions.append(_forward_action(policy, states, deterministic))
    return torch.stack(actions, dim=0).transpose(0, 1)  # [K,B,A] -> [B,K,A]


def sample_actions_with_log_prob(
    policy: nn.Module,
    states: torch.Tensor,
    K: int = 1,
    seed: Optional[int] = None,
    *,
    action_dim: Optional[int] = None,
    max_action: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """K개 action 샘플 + log_prob: (actions [B,K,A], log_probs [B,K]) 반환."""
    policy = _ensure_policy(policy, action_dim, max_action)
    try:
        out = policy(states, deterministic=False, repeat=K)
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            actions, log_probs = out[0], out[1]
            if log_probs.dim() > 2:
                log_probs = log_probs.squeeze(-1)
            return actions, log_probs
    except TypeError:
        pass
    actions = policy.sample_actions(states, K=K, seed=seed)
    if hasattr(policy, "log_prob_actions"):
        B, Kk, A = actions.shape
        states_exp = states.unsqueeze(1).expand(B, Kk, states.size(-1)).reshape(B * Kk, -1)
        actions_flat = actions.reshape(B * Kk, A)
        lp = policy.log_prob_actions(states_exp, actions_flat, keepdim=False)
        return actions, lp.reshape(B, Kk)
    log_probs = torch.zeros(actions.size(0), actions.size(1), device=actions.device, dtype=actions.dtype)
    return actions, log_probs


def act_for_eval(
    policy: nn.Module,
    state: "np.ndarray",
    device: str = "cpu",
    *,
    action_dim: Optional[int] = None,
    max_action: Optional[float] = None,
    deterministic: bool = True,
    seed: Optional[int] = None,
):
    """평가/롤아웃용: actor.act(state, device) 대체.
    deterministic: True면 mean/고정값, False면 policy에서 샘플링 (stochastic eval).
    seed: stochastic eval 시 재현성을 위한 seed (None이면 랜덤).
    """
    import numpy as np

    policy = _ensure_policy(policy, action_dim, max_action)
    if hasattr(policy, "act"):
        return policy.act(state, device)
    state_t = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
    action, _ = get_action(policy, state_t, deterministic=deterministic, seed=seed)
    return action.cpu().numpy().flatten()
