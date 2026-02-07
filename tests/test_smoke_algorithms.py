"""
엔드투엔드 스모크 테스트: TD3+BC, CQL, SAC-N, EDAC, AWAC 각 200~500 step 실행.
체크 포인트:
- get_action(deterministic=True) 평가 루프
- action_and_log_prob() 반환 shape [B,1]
- sample_K_actions() stochastic policy에서 K번 서로 다른 샘플 (variance)
"""
import os
import subprocess
import sys

# Add project root for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from algorithms.offline.pogo_policies import GaussianMLP, TanhGaussianMLP
from algorithms.networks import get_action, sample_K_actions


def test_action_and_log_prob_shape():
    """action_and_log_prob() 반환 log_prob shape가 [B,1] 인지 확인"""
    B, S, A, max_a = 8, 17, 6, 1.0
    states = torch.randn(B, S)

    for policy_cls, name in [(GaussianMLP, "GaussianMLP"), (TanhGaussianMLP, "TanhGaussianMLP")]:
        policy = policy_cls(S, A, max_a)
        action, log_prob = policy.action_and_log_prob(states)
        assert action.shape == (B, A), f"{name}: action {action.shape}"
        assert log_prob is not None, f"{name}: need log_prob"
        assert log_prob.shape == (B, 1) or (log_prob.dim() == 2 and log_prob.shape[-1] == 1), (
            f"{name}: log_prob shape should be [B,1], got {log_prob.shape}"
        )
        print(f"OK {name}: action_and_log_prob -> log_prob {log_prob.shape}")


def test_sample_K_actions_variance():
    """sample_K_actions가 stochastic policy에서 K개 서로 다른 샘플 뽑는지 (variance > 0)"""
    B, S, A, K, max_a = 4, 17, 6, 5, 1.0
    states = torch.randn(B, S)
    policy = TanhGaussianMLP(S, A, max_a)
    policy.eval()

    actions = sample_K_actions(policy, states, K=K, deterministic=False, seed=None)
    assert actions.shape == (B, K, A), f"expected (B,K,A), got {actions.shape}"

    # 각 state별로 K개 샘플의 분산이 0이 아니어야 함 (stochastic)
    for b in range(B):
        var_per_dim = actions[b].var(dim=0)
        assert var_per_dim.sum() > 1e-6, f"state {b}: K samples should differ (stochastic)"
    print("OK sample_K_actions: stochastic policy produces K different samples")


def test_get_action_deterministic_eval():
    """get_action(deterministic=True) 평가 루프용"""
    B, S, A, max_a = 4, 17, 6, 1.0
    states = torch.randn(B, S)
    policy = TanhGaussianMLP(S, A, max_a)
    policy.eval()

    action, log_prob = get_action(policy, states, deterministic=True)
    assert action.shape == (B, A)
    assert log_prob is None or log_prob.shape == (B,)
    # deterministic이면 같은 state에서 같은 action
    action2, _ = get_action(policy, states, deterministic=True)
    assert torch.allclose(action, action2), "deterministic should give same action"
    print("OK get_action(deterministic=True) for eval loop")


CONFIG_MAP = {
    "td3_bc": "configs/offline/pogo_multi/halfcheetah/medium_v2_td3_bc.yaml",
    "cql": "configs/offline/cql_pogo_base.yaml",
    "sac_n": "configs/offline/sac_n_pogo_base.yaml",
    "edac": "configs/offline/edac_pogo_base.yaml",
    "awac": "configs/offline/awac_pogo_base.yaml",
}


def run_smoke_training(algorithm: str, steps: int = 300) -> bool:
    """pogo_multi_main으로 알고리즘 steps만큼 학습 실행"""
    config_path = CONFIG_MAP.get(algorithm, "configs/offline/cql_pogo_base.yaml")
    env = os.environ.copy()
    env["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
    env["MUJOCO_GL"] = "egl"
    env["WANDB_MODE"] = "disabled"

    cmd = [
        sys.executable, "-u", "-m", "algorithms.offline.pogo_multi_main",
        "--config_path", config_path,
        "--algorithm", algorithm,
        "--env", "halfcheetah-medium-v2",
        "--seed", "0",
        "--max_timesteps", str(steps),
        "--use_wandb", "false",
    ]
    cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            print(f"FAIL {algorithm}: returncode={r.returncode}\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}")
            return False
        print(f"OK {algorithm}: smoke training {steps} steps")
        return True
    except subprocess.TimeoutExpired:
        print(f"FAIL {algorithm}: timeout")
        return False
    except Exception as e:
        print(f"FAIL {algorithm}: {e}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("Policy interface smoke tests")
    print("=" * 50)
    test_action_and_log_prob_shape()
    test_sample_K_actions_variance()
    test_get_action_deterministic_eval()

    print("\n" + "=" * 50)
    print("End-to-end smoke: training 300 steps each")
    print("=" * 50)
    algorithms = ["td3_bc", "cql", "sac_n", "edac", "awac"]
    ok = 0
    for alg in algorithms:
        if run_smoke_training(alg, steps=300):
            ok += 1
    print(f"\n{ok}/{len(algorithms)} algorithms passed smoke test")
    sys.exit(0 if ok == len(algorithms) else 1)
