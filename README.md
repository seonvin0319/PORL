# PORL_unified — POGO Multi-Actor Offline RL

**기존 오프라인 RL 알고리즘의 Critic/Q 로직을 유지**하면서 **Actor만 multi-actor(JKO chain)로 확장**하는 통합 프레임워크입니다.

---

## 목차

- [개요](#개요)
- [아키텍처](#아키텍처)
- [학습 흐름](#학습-흐름)
- [코드 구조](#코드-구조)
- [구현 상세](#구현-상세)
- [설치 및 실행](#설치-및-실행)
- [문제 해결](#문제-해결)

---

## 개요

다양한 오프라인 RL 알고리즘(IQL, AWAC, TD3_BC, CQL, SAC-N, EDAC) 위에 **POGO(Policy Gradient with Optimal Transport)** multi-actor 구조를 올립니다.

- **Critic/V/Q**: 각 알고리즘 원본 업데이트 경로 유지
- **Actor만 확장**: JKO chain 형태로 W2/Sinkhorn penalty로 순차 학습
- **Config 기반**: `algorithm: iql` 등으로 알고리즘 선택

---

## 아키텍처

### 설계 원칙

| 구성요소 | 설계 의도 |
|---------|-----------|
| **Actor0** | 원래 알고리즘 actor loss만 사용 (W2 penalty 없음) |
| **Actor1+** | 기존 알고리즘 loss + w₂ᵢ · W2(πᵢ, πᵢ₋₁) |
| **Critic / V / Q** | 알고리즘 원본 그대로 유지 |

### Actor Loss

- **Actor0**: `L₀ = [기존 알고리즘 loss]`
- **Actor1+**: `Lᵢ = [기존 알고리즘 loss] + w₂ᵢ · W2(πᵢ, πᵢ₋₁)`

### W2 거리 계산 (`_compute_w2_distance`)

Policy 타입에 따라 자동 선택:

| 조건 | 방식 |
|------|------|
| Both Gaussian | Closed-form W2 (`||μ1-μ2||² + ||σ1-σ2||²`) |
| Both Stochastic (not Gaussian) | Sinkhorn (GeomLoss) |
| At least one Deterministic | L2 |

### Policy 타입 (`algorithms/networks/`)

| 타입 | W2 | 속성 |
|------|-----|------|
| GaussianMLP | Closed-form | `is_gaussian=True`, `is_stochastic=True` |
| TanhGaussianMLP | Sinkhorn | `is_gaussian=False`, `is_stochastic=True` |
| StochasticMLP | Sinkhorn | `is_gaussian=False`, `is_stochastic=True` |
| DeterministicMLP | L2 | `is_gaussian=False`, `is_stochastic=False` |

**참고**: 모든 Actor는 `ActorAPI(Protocol)` 인터페이스를 구현하며, `deterministic_actions`, `sample_actions`, `log_prob_actions` 메서드를 제공합니다.

### 알고리즘별 Actor Loss (기존)

| 알고리즘 | Actor Loss |
|---------|------------|
| IQL | `mean(exp(β·adv) · BC_loss)` |
| AWAC | `-mean(weights · log_prob)`, weights=exp(adv/λ) |
| TD3_BC | `-λ·Q + MSE(π, a_dataset)` |
| CQL | `α·log_π - Q` 또는 `log_prob(a_dataset)` |
| SAC-N / EDAC | `α·log_π - Q_min` |

**Actor1+ base loss**: IQL, AWAC는 `compute_actor_base_loss`에서 Actor0와 동일한 advantage-weighted BC 반환. TD3_BC는 `-λ·Q`만 사용(원래 의도).

---

## 학습 흐름

```
[Batch: s, a, r, s', d]
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  trainer.train(batch) 또는 trainer.update(batch)              │
│  → V, Q, Critic, Actor0 업데이트 (알고리즘 원본 그대로)        │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  policy_freq마다 (IQL: 매 step, TD3_BC: 2 step마다)           │
│  for i in [1, 2, ...]:                                       │
│    base_loss = trainer.compute_actor_base_loss(actor_i, ...) │
│    w2 = _compute_w2_distance(π_i, π_{i-1})                   │
│    loss = base_loss + w2_weight * w2                         │
│    actor_optimizers[i].step()                                 │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  actor_targets soft update (τ)                                │
│  Critic target: trainer 내부에서만 업데이트 (main 미호출)      │
└─────────────────────────────────────────────────────────────┘
```

---

## 코드 구조

```
algorithms/
├── networks/               # 통합 네트워크 패키지
│   ├── __init__.py        # 모든 클래스 re-export
│   ├── actors.py          # BaseActor, GaussianMLP, TanhGaussianMLP, DeterministicMLP 등
│   ├── critics.py         # FullyConnectedQFunction, EnsembleCritic, VectorizedLinear 등
│   ├── mlp.py             # build_mlp, init_module_weights, create_generator 등
│   └── adapters.py        # ActorAPI(Protocol), PolicyAdapter
├── offline/
│   ├── pogo_multi_main.py      # 통합 학습 진입점
│   │   ├── _create_actors()           # Actor0 + Actor1+ 생성
│   │   ├── _compute_w2_distance()     # W2 거리 (Gaussian/Sinkhorn/L2)
│   │   ├── _compute_actor_loss_with_w2()
│   │   ├── _train_multi_actor()       # trainer.train + Actor1+ 업데이트
│   │   └── train()                    # 메인 루프
│   ├── utils_pytorch.py        # 공통 유틸리티 (soft_update, ReplayBuffer, eval_actor 등)
│   ├── iql.py, awac.py, td3_bc.py, cql.py, sac_n.py, edac.py   # 각 알고리즘
│   │   └── compute_actor_base_loss()  # Actor1+용 base loss (알고리즘별 구현)
│   └── pogo_multi_jax.py       # JAX 버전 (ReBRAC)
└── utils/policy_call.py    # get_action, act_for_eval, sample_K_actions
```

### 핵심 파일 역할

| 파일 | 역할 |
|------|------|
| `pogo_multi_main.py` | 알고리즘 선택, Actor 생성, `_train_multi_actor`로 통합 학습 |
| `algorithms/networks/` | 통합 네트워크 패키지 (Actor, Critic, MLP 유틸리티) |
| `algorithms/networks/actors.py` | 모든 Actor 클래스 (GaussianMLP, TanhGaussianMLP 등) |
| `algorithms/networks/adapters.py` | `ActorAPI(Protocol)` 인터페이스 정의 |
| `utils_pytorch.py` | 공통 유틸리티 (soft_update, ReplayBuffer, eval_actor, normalize_states 등) |
| `policy_call.py` | 평가/학습 시 policy 호출 통일 |

### Actor 생성 (`_create_actors`)

- **Actor0**: `actor_creation.create_actor0()` — 알고리즘별 (예: IQL → GaussianMLP, TD3_BC → DeterministicMLP)
- **Actor1+**: `algorithms.networks`의 POGO policies. `actor_configs` 미지정 시 Actor0 구조 추론 후 동일 타입 사용, `state_dict` 복사로 초기화
- **인터페이스**: 모든 Actor는 `ActorAPI(Protocol)`을 구현하여 `deterministic_actions`, `sample_actions`, `log_prob_actions` 메서드를 제공

---

## 구현 상세

### 1. Critic / Actor target 업데이트 분리

- **Critic target**: trainer 내부 (`train()`/`update()` 내)에서만 soft_update. `pogo_multi_main`은 `call_trainer_update_target_network=False`로 **호출하지 않음**
- **Actor target**: `_train_multi_actor`에서만 업데이트 (모든 알고리즘 공통)

### 2. `compute_actor_base_loss`

각 trainer가 구현. Actor1+의 base loss를 Actor0와 동일하게 맞추기 위해 사용:

- **IQL, AWAC**: advantage-weighted BC (`exp(β·adv) · BC_loss` 또는 `weights · (-log_prob)`)
- **TD3_BC**: `-λ·Q`
- **CQL, SAC-N, EDAC**: 각 알고리즘의 policy loss

### 3. `action_for_loss`

W2/Loss 계산 시 **미분 가능한** action 필요. `deterministic_actions()`는 `@torch.no_grad()`로 gradient가 끊기므로, Gaussian은 `get_mean_std()[0]`, stochastic은 `sample_actions`, deterministic은 `forward` 사용.

### 4. 평가 모드

- Deterministic policy → deterministic 평가
- Stochastic policy → policy에서 샘플링 (`actor_is_stochastic` 기반 자동 선택)

### 5. 기존 알고리즘 파일

`iql.py`, `awac.py` 등은 **수정하지 않음**. `compute_actor_base_loss`만 POGO용으로 추가. `train()`/`update()` 내부 로직은 그대로 사용.

### 6. 코드 리팩토링 (2024)

- **네트워크 통합**: `algorithms/offline/networks.py` → `algorithms/networks/` 패키지로 분리
  - `actors.py`: 모든 Actor 클래스
  - `critics.py`: 모든 Critic 클래스
  - `mlp.py`: MLP 빌딩 및 초기화 유틸리티
  - `adapters.py`: `ActorAPI(Protocol)` 인터페이스
- **유틸리티 통합**: 중복 함수들을 `algorithms/offline/utils_pytorch.py`로 통합
  - `soft_update`, `ReplayBuffer`, `eval_actor`, `normalize_states`, `set_seed`, `wandb_init` 등
- **인터페이스 명확화**: `ActorAPI(Protocol)`로 모든 Actor의 공통 인터페이스 정의
  - `deterministic_actions(states) -> [B, A]`
  - `sample_actions(states, K=1, seed=None) -> [B, K, A]`
  - `log_prob_actions(states, actions) -> [B]`

---

## 설치 및 실행

```bash
pip install -r requirements/requirements.txt
pip install geomloss pyyaml
export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL=egl
```

```bash
python -m algorithms.offline.pogo_multi_main \
  --config_path configs/offline/iql_pogo_base.yaml \
  --env halfcheetah-medium-v2
```

**출력**: `results/{algorithm}/{env}/{task}/seed_{N}/checkpoints/`, `logs/`

---

## 문제 해결

- **Import 에러**: `python -m algorithms.offline.pogo_multi_main` 형식으로 실행
- **GeomLoss**: `pip install geomloss`
- **Headless**: `export MUJOCO_GL=egl`
- **Wandb**: config `use_wandb: true`(기본)면 활성화. 비활성화: `--no_wandb` 또는 config에 `use_wandb: false`
- **공통 env**: run 스크립트는 `env_common.sh`를 source하여 D4RL/MUJOCO/PYTHONUNBUFFERED 설정

---

## 라이선스

[LICENSE](LICENSE) 참조.
