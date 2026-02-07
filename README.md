# POGO Multi-Actor: CORL Integration

POGO_sv 프로젝트를 baseline으로 하여 CORL 프로젝트에 통합한 multi-actor offline reinforcement learning 알고리즘입니다.

## 빠른 시작

### 설치

```bash
pip install geomloss PyYAML
# JAX 버전 사용 시
pip install jax jaxlib flax optax
```

### 실행 예시

```bash
# PyTorch 버전 (IQL)
python -m algorithms.offline.pogo_multi_main \
    --config_path configs/offline/pogo_multi/halfcheetah/medium_v2_iql.yaml

# JAX 버전 (ReBRAC)
python -m algorithms.offline.pogo_multi_jax \
    --config_path configs/offline/pogo_multi/halfcheetah/medium_v2_rebrac.yaml
```

## 프로젝트 개요

이 프로젝트는 **각 기존 알고리즘의 구조를 그대로 유지**하면서, **Actor만 multi-actor로 교체**하여 학습하는 통합 프레임워크입니다.

### 핵심 원칙

- ✅ **각 알고리즘의 Critic, V, Q 구조는 변경 없음**
- ✅ **Critic 업데이트는 Actor0만 사용** (모든 알고리즘 공통)
- ✅ **Actor만 multi-actor로 교체** (개수와 loss만 변경)
- ✅ **Policy loss에서만 multi-actor 학습** (Critic은 기존 방식 그대로)
- ✅ **Config에서 algorithm 선택 가능** (`algorithm: iql`, `td3_bc`, `cql`, `awac`, `sac_n`, `edac`, `rebrac`, `fql`)
- ✅ **JAX/PyTorch 양쪽 구현**: 동일한 구조와 로직을 따르는 두 가지 구현 제공

### POGO 이론

POGO(**Policy Optimization as a Gradient Flow in Offline RL**)는 **JKO (Jordan-Kinderlehrer-Otto) chain**을 기반으로 한 gradient flow 방법입니다:

- **JKO Chain**: 연속적인 policy 업데이트를 이산화하여 `π₀ → π₁ → ... → πₙ` 형태로 학습
- **W2 Distance**: 각 단계에서 이전 policy와의 Wasserstein-2 거리를 penalty로 사용
- **Energy Function**: 각 알고리즘의 특성에 맞는 energy function을 정의하여 policy를 최적화

**참고**: 각 알고리즘의 Critic/Q 로직은 그대로 유지하면서, Actor만 multi-actor(JKO chain)로 확장합니다.

### Multi-Actor 구조

- **Actor0**: 각 알고리즘의 원래 actor loss만 사용 (W2 penalty 없음)
- **Actor1+**: Energy function + W2 distance to previous actor
  - Loss: `Lᵢ = E(πᵢ) + w₂ᵢ · W₂(πᵢ, πᵢ₋₁)`

### 지원 알고리즘

**PyTorch 버전**: IQL, TD3_BC, CQL, AWAC, SAC-N, EDAC  
**JAX 버전**: ReBRAC, FQL

### 알고리즘별 Energy Function (Actor1+용)

| 알고리즘 | Energy Function |
|---------|----------------|
| IQL | `-Q(state, π(state))` 또는 `-A` (`energy_function_type` config로 선택) |
| AWAC | `-Q(state, π(state))` 또는 `-A` (`energy_function_type` config로 선택) |
| SAC-N / EDAC | `α·log_π - Q_min` |
| CQL | `α·log_π - Q` (BC/CQL stage에 따라) |
| TD3_BC | `-Q(state, π(state))` |

**Config 설정**:
- `energy_function_type: "q"` (기본값): `-Q(state, π(state))` 사용
- `energy_function_type: "advantage"`: `-A` (advantage) 사용 (IQL/AWAC만 해당)

**참고**: Actor0는 각 알고리즘의 원본 `train()`/`update()` 로직으로 업데이트되며, Actor1+만 energy function + W2 distance를 사용합니다.

## 프로젝트 구조

```
PORL/
├── algorithms/
│   ├── networks/               # 통합 네트워크 패키지
│   │   ├── __init__.py        # 모든 클래스 re-export
│   │   ├── actors.py          # BaseActor, GaussianMLP, TanhGaussianMLP, DeterministicMLP 등
│   │   ├── critics.py         # FullyConnectedQFunction, EnsembleCritic, VectorizedLinear 등
│   │   ├── mlp.py             # build_mlp, init_module_weights, create_generator 등
│   │   ├── adapters.py        # ActorAPI(Protocol), PolicyAdapter
│   │   └── policy_call.py     # get_action, act_for_eval, sample_K_actions
│   └── offline/
│       ├── pogo_multi_main.py      # PyTorch 버전 메인 스크립트
│       ├── pogo_multi_jax.py       # JAX 버전 메인 스크립트
│       ├── pogo_policies_jax.py    # JAX Policy 구현
│       ├── utils_pytorch.py        # PyTorch 유틸리티 (soft_update, ReplayBuffer, eval_actor 등)
│       ├── utils_jax.py            # JAX 유틸리티 (AlgorithmInterface)
│       ├── iql.py, cql.py, ...      # 각 알고리즘 구현
│       │   └── compute_energy_function()  # Actor1+용 energy function (알고리즘별 구현)
│       ├── POGO_MULTI_README.md      # 상세 문서
│       └── CODE_EVALUATION_REPORT.md # 코드 평가 보고서
├── configs/
│   └── offline/
│       └── pogo_multi/              # 알고리즘별 설정 파일
└── README.md                         # 이 파일
```

### 아키텍처 개요

```
┌─────────────────────────────────────────────────────────┐
│                    Training Loop                          │
└─────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        │                                         │
┌───────▼────────┐                      ┌─────────▼────────┐
│  Critic Update │                      │  Actor Update    │
│  (Actor0만 사용)│                      │  (Multi-Actor)   │
└───────┬────────┘                      └─────────┬────────┘
        │                                         │
        │  ┌──────────────────────────────────┐  │
        │  │  AlgorithmInterface               │  │
        │  │  - update_critic()                │  │
        │  │  - compute_energy_function()     │  │
        └──┼──────────────────────────────────┼──┘
           │                                  │
    ┌──────▼──────┐                  ┌────────▼────────┐
    │  IQL/CQL/   │                  │  Actor0:        │
    │  TD3_BC/... │                  │  [기존 loss]     │
    │  (원래 구조) │                  │                 │
    └─────────────┘                  │  Actor1+:       │
                                      │  energy +       │
                                      │  w2_weight * W2 │
                                      └─────────────────┘
```

**핵심 설계 원칙**:
1. **Critic 업데이트**: 각 알고리즘의 원래 방식 그대로 (Actor0만 사용)
2. **Actor 업데이트**: Multi-actor 구조로 확장 (W2 regularization 추가)
3. **AlgorithmInterface**: 각 알고리즘을 통일된 인터페이스로 추상화

### 학습 흐름

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
│    energy = trainer.compute_energy_function(actor_i, ...)    │
│    w2 = _compute_w2_distance(π_i, π_{i-1})                   │
│    loss = energy + w2_weight * w2                            │
│    actor_optimizers[i].step()                                 │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  actor_targets soft update (τ)                                │
│  Critic target: trainer 내부에서만 업데이트 (main 미호출)      │
└─────────────────────────────────────────────────────────────┘
```

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

### 구현 상세

#### 1. Critic / Actor target 업데이트 분리

- **Critic target**: trainer 내부 (`train()`/`update()` 내)에서만 soft_update. `pogo_multi_main`은 `call_trainer_update_target_network=False`로 **호출하지 않음**
- **Actor target**: `_train_multi_actor`에서만 업데이트 (모든 알고리즘 공통)

#### 2. `compute_energy_function`

각 trainer가 구현. Actor1+의 energy function:

- **IQL, AWAC**: `-Q(state, π(state))` 또는 `-A` (config의 `energy_function_type`으로 선택)
- **SAC-N, EDAC**: `α·log_π - Q_min`
- **CQL**: `α·log_π - Q` (BC/CQL stage에 따라 다름)
- **TD3_BC**: `-Q(state, π(state))`

**참고**: Actor0는 각 알고리즘의 원본 `train()`/`update()` 로직으로 업데이트되며, `compute_energy_function`은 사용하지 않습니다.

#### 3. `action_for_loss`

W2/Loss 계산 시 **미분 가능한** action 필요. `deterministic_actions()`는 `@torch.no_grad()`로 gradient가 끊기므로, Gaussian은 `get_mean_std()[0]`, stochastic은 `sample_actions`, deterministic은 `forward` 사용.

#### 4. 평가 모드

- Deterministic policy → deterministic 평가
- Stochastic policy → policy에서 샘플링 (`actor_is_stochastic` 기반 자동 선택)

#### 5. 기존 알고리즘 파일

`iql.py`, `awac.py` 등은 **수정하지 않음**. `compute_energy_function`만 POGO용으로 추가. `train()`/`update()` 내부 로직은 그대로 사용하여 Actor0를 업데이트합니다.

#### 6. 코드 리팩토링 (2026)

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

## 설치 및 실행

### 설치

```bash
pip install -r requirements/requirements.txt
pip install geomloss PyYAML
# JAX 버전 사용 시
pip install jax jaxlib flax optax
export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL=egl
```

### 실행 예시

```bash
# PyTorch 버전 (IQL)
python -m algorithms.offline.pogo_multi_main \
    --config_path configs/offline/pogo_multi/halfcheetah/medium_v2_iql.yaml

# JAX 버전 (ReBRAC)
python -m algorithms.offline.pogo_multi_jax \
    --config_path configs/offline/pogo_multi/halfcheetah/medium_v2_rebrac.yaml
```

**출력**: `results/{algorithm}/{env}/{task}/seed_{N}/checkpoints/`, `logs/`

## 문제 해결

- **Import 에러**: `python -m algorithms.offline.pogo_multi_main` 형식으로 실행
- **GeomLoss**: `pip install geomloss`
- **Headless**: `export MUJOCO_GL=egl`
- **Wandb**: config `use_wandb: true`(기본)면 활성화. 비활성화: `--no_wandb` 또는 config에 `use_wandb: false`
- **공통 env**: run 스크립트는 `env_common.sh`를 source하여 D4RL/MUJOCO/PYTHONUNBUFFERED 설정

## 참고

- **CORL**: [Clean Offline Reinforcement Learning](https://github.com/corl-team/CORL)
- **POGO**: Policy Optimization as a Gradient Flow in Offline RL
- **D4RL**: [Datasets for Deep Data-Driven Reinforcement Learning](https://github.com/Farama-Foundation/D4RL)

## 라이선스

[LICENSE](LICENSE) 참조.
