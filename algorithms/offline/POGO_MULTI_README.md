# POGO Multi-Actor: POGO_sv Baseline + CORL Integration

POGO_sv 프로젝트를 baseline으로 하여 CORL 프로젝트에 통합한 multi-actor offline RL 알고리즘입니다.

## 이론적 배경

### JKO Chain과 Gradient Flow

POGO Multi-Actor는 **JKO (Jordan-Kinderlehrer-Otto) Chain**을 사용하여 여러 actor를 순차적으로 학습하는 gradient flow 기반 접근법입니다. 각 actor는 gradient flow의 한 단계를 나타내며, 전체 chain은 연속적인 정책 진화를 이산적으로 근사합니다.

### Multi-Actor 구조

- **Actor0 (π₀)**: 각 알고리즘의 원래 actor loss만 사용 (W2 regularization 없음)
  - Offline RL에서는 데이터셋의 action과의 L2 distance 또는 BC loss 사용
  - 이전 정책에 대한 제약 없이 자유롭게 학습
  
- **Actor1+ (π_i, i ≥ 1)**: 이전 actor (π_{i-1})에 대한 W2 거리로 학습
  - Loss: `base_loss + w_i · W₂(π_i, π_{i-1})`
  - 이전 actor라는 연속적인 분포를 reference로 하여 분포 간 거리를 측정하는 W2 거리 사용

### 학습 목표

각 actor는 다음 loss를 최소화합니다:

$$L_i = \begin{cases}
L_{\text{base}}(\pi_0) & \text{if } i = 0 \\
L_{\text{base}}(\pi_i) + w_i \cdot W_2(\pi_i, \pi_{i-1}) & \text{if } i \geq 1
\end{cases}$$

여기서:
- $L_{\text{base}}(\pi_i)$: 각 알고리즘의 원래 actor loss (예: IQL의 advantage-weighted BC, ReBRAC의 BC penalty - Q, FQL의 BC flow loss 등)
- $W_2(\pi_i, \pi_{i-1})$: W2 거리 (Wasserstein-2 distance)
- $w_i$: W2 거리의 가중치 (Actor1+부터 적용)

### W2 Distance 계산 방법

Policy 타입에 따라 적절한 거리 측정 방법을 자동으로 선택합니다:

- **Both Gaussian**: Closed-form W2 사용
  $$W_2^2(\pi_i, \pi_{i-1}) = ||\mu_i - \mu_{i-1}||^2 + ||\sigma_i - \sigma_{i-1}||^2$$
  
- **Both Stochastic (not Gaussian)**: Sinkhorn distance 사용
  - 샘플 기반 optimal transport 알고리즘
  
- **At least one Deterministic**: L2 distance 사용
  $$||\pi_i(s) - \pi_{i-1}(s)||^2$$

### Critic 학습

모든 알고리즘에서 Critic 업데이트는 **Actor0만 사용**합니다:
- Critic은 기존 알고리즘의 원래 방식 그대로 유지
- Multi-actor 학습은 Policy loss에서만 수행
- 이는 각 알고리즘의 구조를 그대로 유지하면서 Actor만 확장하는 핵심 원칙입니다

## 주요 특징

- **각 알고리즘의 구조를 그대로 사용**: Critic, V function, Q function 등은 각 알고리즘(IQL, CQL, TD3_BC 등)의 원래 구조 유지
- **Critic 업데이트는 원래 알고리즘의 actor만 사용**: 모든 알고리즘에서 Critic/V/Q 업데이트는 기존 방식 그대로
- **Actor만 multi-actor로 교체**: Actor 개수와 Actor loss만 변경
- **Policy loss에서만 multi-actor 학습**: Critic은 기존 방식 그대로, Actor만 multi-actor로 확장
- **Config에서 algorithm 선택**: `algorithm: iql` 또는 `algorithm: td3_bc` 등으로 선택
- **Multi-Actor 구조**: 여러 actor를 순차적으로 학습하는 JKO chain 방식
- **Actor0**: 각 알고리즘의 원래 actor loss만 사용 (W2 penalty 없음)
- **Actor1+**: 각 알고리즘의 actor loss + W2 distance to previous actor (`w2_weights` 리스트 사용)
- **JAX/PyTorch 양쪽 구현**: 동일한 구조와 로직을 따르는 두 가지 구현 제공

## Policy 타입

### GaussianMLP
- **Mean**: `tanh(...) * max_action` (bounded mean)
- **Sampling**: bounded mean과 std로부터 Gaussian 샘플링
- **W2 Distance**: Closed-form W2 사용 (`||μ1-μ2||² + ||σ1-σ2||²`)
- **속성**: `is_gaussian=True`, `is_stochastic=True`

### TanhGaussianMLP
- **Mean**: Unbounded
- **Sampling**: Unbounded Gaussian에서 샘플링 후 `tanh` 적용
- **W2 Distance**: Sinkhorn distance 사용 (closed-form W2 사용 불가)
- **속성**: `is_gaussian=False`, `is_stochastic=True`

### StochasticMLP
- **Sampling**: Latent `z` 벡터 사용
- **W2 Distance**: Sinkhorn distance 사용
- **속성**: `is_gaussian=False`, `is_stochastic=True`

### DeterministicMLP
- **Output**: 단일 deterministic action
- **W2 Distance**: L2 distance 사용
- **속성**: `is_gaussian=False`, `is_stochastic=False`

### FQLFlowPolicy (FQL 전용)
- **구조**: `actor_bc_flow` (multi-step flow matching) + `actor_onestep_flow` (one-step policy)
- **Actor0**: BC flow loss만 사용 (BC policy)
- **Actor1+**: Q loss만 사용 (`-Q`, 다른 policy 타입 사용 가능)
- **W2 Distance**: Sinkhorn distance 사용
- **속성**: `is_gaussian=False`, `is_stochastic=True`

## 코드 구조

### 핵심 컴포넌트

1. **`ActorConfig` dataclass**: Actor 관련 정보를 그룹화
   ```python
   @dataclass
   class ActorConfig:
       actor: nn.Module  # PyTorch
       # 또는 params: FrozenDict, module: nn.Module  # JAX
       is_stochastic: bool
       is_gaussian: bool
   ```

2. **공통 헬퍼 함수**:
   - `_compute_w2_distance()`: Policy 타입에 따라 자동으로 적절한 W2 계산 방법 선택
   - `_compute_actor_loss_with_w2()`: Actor0/Actor1+ loss 계산 통합

3. **Policy 타입 자동 감지**: Policy 클래스의 `is_gaussian`, `is_stochastic` 속성 자동 사용

## 설치

### 의존성

기본 CORL 의존성 외에 다음 패키지가 필요합니다:

```bash
pip install geomloss PyYAML
```

JAX 버전 사용 시:
```bash
pip install jax jaxlib flax optax ott-jax
```

**참고**: JAX 버전은 OTT-JAX 라이브러리를 사용하여 Sinkhorn distance를 계산합니다. 
OTT-JAX가 설치되어 있지 않으면 자동으로 fallback 구현을 사용합니다.

## 사용 방법

### PyTorch 버전

#### IQL 구조로 multi-actor 학습
```bash
cd /home/offrl/CORL
python -m algorithms.offline.pogo_multi_main \
    --config_path configs/offline/pogo_multi/halfcheetah/medium_v2_iql.yaml
```

#### TD3_BC 구조로 multi-actor 학습
```bash
python -m algorithms.offline.pogo_multi_main \
    --config_path configs/offline/pogo_multi/halfcheetah/medium_v2_td3_bc.yaml
```

### JAX 버전 (ReBRAC 기반)

```bash
python -m algorithms.offline.pogo_multi_jax \
    --config_path configs/offline/pogo_multi/halfcheetah/medium_v2_rebrac.yaml
```

### Command-line 인자로 override

```bash
python -m algorithms.offline.pogo_multi_main \
    --config_path configs/offline/pogo_multi/halfcheetah/medium_v2_iql.yaml \
    --algorithm iql \
    --max_timesteps 1000000 \
    --eval_freq 5000 \
    --w2_weights 10.0 10.0
```

## Config 파일 구조

### IQL 구조 예시

```yaml
# 필수: 사용할 알고리즘 선택
algorithm: iql

# 환경 설정
env: halfcheetah-medium-v2
seed: 0
eval_freq: 5000
n_episodes: 10
max_timesteps: 1000000

# POGO Multi-Actor 설정
# w2_weights: Actor1부터의 가중치 리스트 (Actor0는 W2 penalty 없음)
w2_weights: [10.0, 10.0]  # Actor1, Actor2용
num_actors: 3

# Actor 타입 설정 (선택사항)
actor_configs:
  - type: gaussian      # Actor0: Gaussian (closed-form W2)
  - type: tanh_gaussian # Actor1: TanhGaussian (Sinkhorn)
  - type: stochastic    # Actor2: Stochastic (Sinkhorn)

# Sinkhorn 설정 (Actor1+용, Gaussian이 아닌 경우에만 사용)
sinkhorn_K: 4
sinkhorn_blur: 0.05
sinkhorn_backend: "tensorized"

# IQL 파라미터 (IQL 구조 그대로 사용)
iql_tau: 0.7
beta: 3.0
vf_lr: 3e-4
qf_lr: 3e-4
actor_lr: 3e-4
iql_deterministic: false

# 공통 파라미터
batch_size: 256
discount: 0.99
tau: 0.005
buffer_size: 2000000
normalize: true
normalize_reward: false

project: CORL
group: pogo-multi-iql
name: pogo-multi-iql
```

### JAX 버전 (ReBRAC) 예시

```yaml
# ReBRAC 기본 설정
dataset_name: halfcheetah-medium-v2
actor_learning_rate: 1e-3
critic_learning_rate: 1e-3
hidden_dim: 256
gamma: 0.99
tau: 5e-3
beta: 1.0
policy_noise: 0.2
noise_clip: 0.5
normalize_q: true

# POGO Multi-Actor 설정
algorithm: rebrac  # "rebrac" or "fql"
w2_weights: [10.0, 10.0]
num_actors: 3
actor_configs:
  - type: deterministic
  - type: gaussian
  - type: tanh_gaussian

# Sinkhorn 설정
sinkhorn_K: 4
sinkhorn_blur: 0.05
```

### JAX 버전 (FQL) 예시

```yaml
# FQL 기본 설정
dataset_name: halfcheetah-medium-v2
actor_learning_rate: 1e-3
critic_learning_rate: 1e-3
hidden_dim: 256
gamma: 0.99
tau: 5e-3

# POGO Multi-Actor 설정
algorithm: fql
w2_weights: [10.0, 10.0]
num_actors: 3
actor_configs:
  - type: flow        # Actor0: FQLFlowPolicy (BC policy)
  - type: stochastic  # Actor1+: 다른 policy 타입 사용 가능
  - type: stochastic

# FQL 설정
fql_alpha: 10.0          # Distillation loss coefficient
fql_flow_steps: 10       # Number of flow steps for BC flow
fql_q_agg: "mean"        # Q aggregation: "mean" or "min"
fql_normalize_q_loss: false

# Sinkhorn 설정
sinkhorn_K: 4
sinkhorn_blur: 0.05
```

**FQL 구조:**
- **Actor0**: BC flow loss만 (BC policy)
- **Actor1+**: Q loss만 (`-Q`, W2는 자동으로 추가됨)

## 알고리즘 구조

### 핵심 원칙

1. **각 알고리즘의 구조를 그대로 사용**
   - IQL: V function + Q function (expectile regression)
   - TD3_BC: Twin Critic (TD3 style)
   - CQL: Conservative Q-learning (원래 Q loss 그대로 사용)
   - AWAC: Twin Critic + Advantage-weighted BC
   - SAC-N: Vectorized Critic (ensemble) + SAC-style actor
   - EDAC: Vectorized Critic (ensemble) + diversity loss + SAC-style actor
   - ReBRAC (JAX): Ensemble Critic + TD3-style actor
   - 각 알고리즘의 Critic, V, Q 구조는 변경 없음

2. **Critic 업데이트는 원래 알고리즘의 actor만 사용**
   - 모든 알고리즘에서 Critic/V/Q 업데이트는 기존 방식 그대로
   - IQL: V, Q 업데이트는 기존 IQL 방식 그대로 (Actor0는 호환성용으로만 연결)
   - CQL: Q loss 계산은 원래 TanhGaussianPolicy 사용 (CQL의 `_q_loss` 메서드 그대로)
   - TD3_BC, AWAC, SAC-N, EDAC: Critic 업데이트는 각 알고리즘의 원래 방식 그대로
   - ReBRAC: Critic 업데이트는 Actor0만 사용

3. **Actor만 multi-actor로 교체 (Policy loss에서만)**
   - 모든 알고리즘에서 POGO policies 사용
   - Actor 개수: config에서 지정 (예: 3개)
   - Actor loss만 변경: Multi-actor 학습은 Policy loss에서만 수행

### 알고리즘별 비교표

| 알고리즘 | 프레임워크 | Critic 구조 | Actor0 Loss | Actor1+ Loss | Critic 업데이트 |
|---------|-----------|------------|-------------|-------------|----------------|
| **IQL** | PyTorch | V + Q (expectile) | `exp(β·adv) · BC_loss` | `exp(β·adv) · BC_loss + w2·W2` | V, Q 업데이트 (Actor0 사용) |
| **TD3_BC** | PyTorch | Twin Critic | `-λ·Q + BC_loss` | `-λ·Q + BC_loss + w2·W2` | Twin Critic 업데이트 (Actor0 사용) |
| **CQL** | PyTorch | Twin Critic | `CQL_policy_loss` | `CQL_policy_loss + w2·W2` | Q loss (원래 TanhGaussianPolicy 사용) |
| **AWAC** | PyTorch | Twin Critic | `weights · BC_loss` | `weights · BC_loss + w2·W2` | Twin Critic 업데이트 (Actor0 사용) |
| **SAC-N** | PyTorch | Vectorized Critic | `α·log_π - Q_min` | `α·log_π - Q_min + w2·W2` | Ensemble Critic 업데이트 (Actor0 사용) |
| **EDAC** | PyTorch | Vectorized Critic | `α·log_π - Q_min` | `α·log_π - Q_min + w2·W2` | Ensemble Critic + diversity (Actor0 사용) |
| **ReBRAC** | JAX | Ensemble Critic | `β·BC_penalty - λ·Q` | `β·BC_penalty - λ·Q + w2·W2` | Ensemble Critic 업데이트 (Actor0 사용) |
| **FQL** | JAX | Q function | `BC_flow_loss` | `-Q + w2·W2` | Q 업데이트 (Actor0 사용) |

**주요 차이점**:
- **Critic 업데이트**: 모든 알고리즘에서 Actor0만 사용 (원래 알고리즘과 동일)
- **Actor0**: 각 알고리즘의 원래 actor loss만 사용 (W2 penalty 없음)
- **Actor1+**: 각 알고리즘의 actor loss + W2 distance to previous actor
- **FQL 특수**: Actor0는 BC flow loss, Actor1+는 Q loss만 사용

### Actor 학습 방식

#### Actor0 (W2 penalty 없음)
- 각 알고리즘의 원래 actor loss만 사용
- IQL: `mean(exp(β·adv) · ||π₀(s) - a_dataset||²)`
- TD3_BC: `-λ·Q(s, π₀(s)) + ||π₀(s) - a_dataset||²`
- CQL: `CQL_policy_loss`
- AWAC: `mean(weights · BC_loss)`
- SAC-N/EDAC: `(α·log_π - Q_min)`
- ReBRAC: `(β·BC_penalty - λ·Q)`
- FQL: `BC_flow_loss` (BC policy)

#### Actor1+ (W2 penalty 추가)
- 각 알고리즘의 actor loss + W2 distance to previous actor
- Loss: `base_loss + w2_weight_i * W₂(π_i, π_{i-1})`
- W2 distance 계산:
  - **Both Gaussian**: Closed-form W2 (`||μ1-μ2||² + ||σ1-σ2||²`)
  - **Both Stochastic (not Gaussian)**: Sinkhorn distance
  - **At least one Deterministic**: L2 distance
- **FQL의 경우**: Actor1+는 `-Q` loss만 사용 (W2는 자동으로 추가됨)

## Config 파라미터 설명

### 필수 파라미터

- `algorithm`: 사용할 알고리즘 선택 (필수)
  - **PyTorch 버전**: `"iql"`, `"td3_bc"`, `"cql"`, `"awac"`, `"sac_n"`, `"edac"`
  - **JAX 버전**: `"rebrac"`, `"fql"`

### POGO Multi-Actor 전용 파라미터

- `w2_weights`: Actor1부터의 W2/Sinkhorn loss 가중치 리스트 (Actor0는 W2 penalty 없음)
  - 예: `[10.0, 10.0]` → Actor1, Actor2용
  - `num_actors = len(w2_weights) + 1` (자동 계산)
- `num_actors`: Actor 개수 (None이면 `len(w2_weights) + 1` 사용)
- `actor_configs`: 각 actor의 설정 리스트
  - `type`: `"gaussian"`, `"tanh_gaussian"`, `"stochastic"`, `"deterministic"`, `"flow"` (FQL 전용)
  - 예: `[{"type": "gaussian"}, {"type": "tanh_gaussian"}, {"type": "stochastic"}]`
  - **FQL의 경우**: Actor0는 반드시 `"flow"` 타입이어야 함, Actor1+는 다른 타입 사용 가능
- `sinkhorn_K`: Sinkhorn 계산 시 각 state당 샘플 수
- `sinkhorn_blur`: Sinkhorn regularization parameter (epsilon)
- `sinkhorn_backend`: Sinkhorn backend (`"tensorized"`, `"online"`, `"auto"`)

### 알고리즘별 파라미터

각 알고리즘의 파라미터는 해당 알고리즘의 config와 동일하게 사용:

- **IQL**: `iql_tau`, `beta`, `vf_lr`, `qf_lr`, `actor_lr`, `iql_deterministic`
- **TD3_BC**: `alpha`, `policy_noise`, `noise_clip`, `policy_freq`
- **CQL**: `cql_alpha`, `cql_n_actions`, `target_entropy`, `qf_lr`, `actor_lr`
- **AWAC**: `awac_lambda`, `exp_adv_max`, `actor_lr`
- **SAC-N**: `num_critics`, `alpha_learning_rate`, `actor_lr`
- **EDAC**: `num_critics`, `eta`, `alpha_learning_rate`, `actor_lr`
- **ReBRAC (JAX)**: `actor_learning_rate`, `critic_learning_rate`, `beta`, `gamma`, `tau`, `policy_noise`, `noise_clip`, `normalize_q`
- **FQL (JAX)**: `fql_alpha`, `fql_flow_steps`, `fql_q_agg`, `fql_normalize_q_loss`

## 평가

각 actor마다 독립적으로 평가가 수행됩니다:
- `eval_actor_{i}`: Actor i의 normalized score
- 모든 actor의 평가 결과가 wandb에 로깅됩니다

## 파일 구조

```
algorithms/offline/
├── pogo_multi_main.py          # PyTorch 통합 학습 스크립트
├── pogo_multi_jax.py            # JAX 통합 학습 스크립트 (ReBRAC 기반)
├── pogo_policies.py             # POGO policies (GaussianMLP, TanhGaussianMLP, StochasticMLP, DeterministicMLP)
├── iql.py                      # IQL (그대로 사용)
├── td3_bc.py                   # TD3_BC (그대로 사용)
├── cql.py                      # CQL (그대로 사용)
├── awac.py                     # AWAC (그대로 사용)
├── sac_n.py                    # SAC-N (그대로 사용)
├── edac.py                     # EDAC (그대로 사용)
├── rebrac.py                   # ReBRAC (JAX, 그대로 사용)
├── fql.py                      # FQL Algorithm (JAX, POGO Multi-Actor 통합)
├── pogo_policies_jax.py        # JAX policies (GaussianMLP, TanhGaussianMLP, StochasticMLP, DeterministicMLP, FQLFlowPolicy)
├── POGO_MULTI_README.md        # 이 파일
└── POGO_MULTI_ARCHITECTURE.md  # 아키텍처 상세 설명

configs/offline/pogo_multi/
├── halfcheetah/
│   ├── medium_v2_iql.yaml      # algorithm: iql
│   ├── medium_v2_td3_bc.yaml   # algorithm: td3_bc
│   ├── medium_v2_cql.yaml      # algorithm: cql
│   ├── medium_v2_awac.yaml     # algorithm: awac
│   ├── medium_v2_sac_n.yaml    # algorithm: sac_n
│   ├── medium_v2_edac.yaml     # algorithm: edac
│   ├── medium_v2_rebrac.yaml   # JAX ReBRAC
│   └── ...
└── ...
```

## 코드 구조 개선 사항

### 1. Policy 타입 자동 관리
- 모든 Policy 클래스에 `is_gaussian`, `is_stochastic` 속성 자동 설정
- `_create_multi_actors`에서 자동으로 속성 읽기

### 2. ActorConfig dataclass
- Actor 관련 정보를 논리적으로 그룹화
- JAX와 PyTorch 버전 모두 동일한 구조 사용

### 3. 공통 헬퍼 함수
- `_compute_w2_distance()`: Policy 타입에 따라 자동으로 적절한 방법 선택
- `_compute_actor_loss_with_w2()`: Actor0/Actor1+ loss 계산 통합
- 코드 중복 제거 및 유지보수성 향상

### 4. 알고리즘 인터페이스 (JAX)
- `AlgorithmInterface`: 다른 알고리즘으로 확장 가능한 인터페이스 정의
- `ReBRACAlgorithm`: ReBRAC 구현체

## 예시 실행

### PyTorch 버전

```bash
# IQL 구조로 multi-actor 학습
python -m algorithms.offline.pogo_multi_main \
    --config_path configs/offline/pogo_multi/halfcheetah/medium_v2_iql.yaml

# TD3_BC 구조로 multi-actor 학습
python -m algorithms.offline.pogo_multi_main \
    --config_path configs/offline/pogo_multi/halfcheetah/medium_v2_td3_bc.yaml

# Custom 설정으로 실행
python -m algorithms.offline.pogo_multi_main \
    --config_path configs/offline/pogo_multi/halfcheetah/medium_v2_iql.yaml \
    --algorithm iql \
    --w2_weights 20.0 20.0 \
    --sinkhorn_K 8 \
    --sinkhorn_blur 0.1 \
    --max_timesteps 2000000
```

### JAX 버전

```bash
# ReBRAC 구조로 multi-actor 학습
python -m algorithms.offline.pogo_multi_jax \
    --config_path configs/offline/pogo_multi/halfcheetah/medium_v2_rebrac.yaml
```

## 문제 해결

### Import 에러

`python -m algorithms.offline.pogo_multi_main` 형식으로 실행해야 합니다 (상대 import 때문).

### GeomLoss 설치 에러

```bash
pip install geomloss
```

### JAX 관련 에러

```bash
pip install jax jaxlib flax optax ott-jax
```

### CUDA/Display 에러

Headless 환경에서는:
```bash
export MUJOCO_GL=egl
export D4RL_SUPPRESS_IMPORT_ERROR=1
```

## 참고

- **POGO_sv**: 원본 baseline 프로젝트
- **CORL**: 이 프로젝트가 통합된 프레임워크
- **GeomLoss**: Sinkhorn distance 계산을 위한 라이브러리 (PyTorch)
- **OTT-JAX**: Optimal Transport Tools for JAX (JAX 버전)

## 라이센스

CORL 프로젝트의 라이센스를 따릅니다.
