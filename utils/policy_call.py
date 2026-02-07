"""
Backward compatibility: Re-export from algorithms.networks
이 파일은 하위 호환성을 위해 유지되며, 새로운 코드는 algorithms.networks를 직접 사용하세요.
"""

from algorithms.networks import (
    get_action,
    act_for_eval,
    sample_actions,
    sample_K_actions,
    sample_actions_with_log_prob,
)

__all__ = [
    "get_action",
    "act_for_eval",
    "sample_actions",
    "sample_K_actions",
    "sample_actions_with_log_prob",
]
