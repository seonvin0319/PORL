# POGO Multi-Actor: POGO_sv Baseline + CORL Integration

POGO_sv 프로젝트를 baseline으로 하여 CORL 프로젝트에 통합한 multi-actor offline RL 알고리즘입니다.

## 주요 특징

- **각 알고리즘의 구조를 그대로 사용**: Critic, V function, Q function 등은 각 알고리즘(IQL, CQL, TD3_BC 등)의 원래 구조 유지
- **Actor만 multi-actor로 교체**: Actor 개수와 Actor loss만 변경
- **Config에서 algorithm 선택**: `algorithm: iql` 또는 `algorithm: td3_bc` 등으로 선택
- **Multi-Actor 구조**: 여러 actor를 순차적으로 학습하는 JKO chain 방식
- **Actor0**: W2 (Wasserstein-2) distance를 사용하여 dataset action과의 L2 loss로 학습
- **Actor1+**: Sinkhorn distance를 사용하여 이전 actor와의 거리로 학습

## 설치

### 의존성

기본 CORL 의존성 외에 다음 패키지가 필요합니다:

```bash
pip install geomloss PyYAML
```

또는 `requirements.txt`를 업데이트했습니다:

```bash
pip install -r requirements/requirements.txt
```

## 사용 방법

### IQL 구조로 multi-actor 학습

```bash
cd /home/offrl/CORL
python -m algorithms.offline.pogo_multi_main \
    --config_path configs/offline/pogo_multi/halfcheetah/medium_v2_iql.yaml
```

### TD3_BC 구조로 multi-actor 학습

```bash
python -m algorithms.offline.pogo_multi_main \
    --config_path configs/offline/pogo_multi/halfcheetah/medium_v2_td3_bc.yaml
```

### Command-line 인자로 override

```bash
python -m algorithms.offline.pogo_multi_main \
    --config_path configs/offline/pogo_multi/halfcheetah/medium_v2_iql.yaml \
    --algorithm iql \
    --max_timesteps 1000000 \
    --eval_freq 5000 \
    --w2_weights 10.0 10.0 10.0
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
w2_weights: [10.0, 10.0, 10.0]
num_actors: 3

# Sinkhorn 설정 (Actor1+용)
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

### TD3_BC 구조 예시

```yaml
# 필수: 사용할 알고리즘 선택
algorithm: td3_bc

env: halfcheetah-medium-v2
w2_weights: [10.0, 10.0, 10.0]

# TD3_BC 파라미터 (TD3_BC 구조 그대로 사용)
alpha: 2.5
policy_noise: 0.2
noise_clip: 0.5
policy_freq: 2
qf_lr: 3e-4
actor_lr: 3e-4

# 공통 파라미터
batch_size: 256
discount: 0.99
tau: 0.005
...
```

## 알고리즘 구조

### 핵심 원칙

1. **각 알고리즘의 구조를 그대로 사용**
   - IQL: V function + Q function (expectile regression)
   - TD3_BC: Twin Critic (TD3 style)
   - CQL: Conservative Q-learning (원래 Q loss 그대로 사용)
   - AWAC: Twin Critic + Advantage-weighted BC
   - SAC-N: Vectorized Critic (ensemble) + SAC-style actor
   - EDAC: Vectorized Critic (ensemble) + diversity loss + SAC-style actor
   - 각 알고리즘의 Critic, V, Q 구조는 변경 없음

2. **Actor만 multi-actor로 교체**
   - 모든 알고리즘에서 POGO policies (StochasticMLP/DeterministicMLP) 사용
   - Actor 개수: config에서 지정 (예: 3개)
   - Actor loss만 변경

### Actor 학습 방식

#### IQL + Multi-Actor

1. **Base IQL의 V, Q 업데이트** (그대로):
   - V function: `V(s) = expectile(Q(s, a_dataset), τ)`
   - Q function: `Q(s, a) = r + γ·V(s')`

2. **Actor0**:
   - Loss: `mean(exp(β·adv) · ||π₀(s) - a_dataset||²) + w₂₀ · ||π₀(s) - a_dataset||²`
   - IQL의 advantage-weighted BC loss + W2 distance

3. **Actor1+**:
   - Loss: `-Q(s, πᵢ(s)) + w₂ᵢ · Sinkhorn(πᵢ(·|s), πᵢ₋₁(·|s))`
   - Q 최대화 + 이전 actor와의 Sinkhorn distance

#### CQL + Multi-Actor

1. **Base CQL의 Q loss 계산** (그대로):
   - CQL의 원래 `_q_loss` 메서드 사용
   - OOD action 샘플링, logsumexp, CQL penalty 모두 포함
   - TanhGaussianPolicy 사용 (Q loss 계산용)

2. **Actor0**:
   - Loss: `CQL_policy_loss + w₂₀ · ||π₀(s) - a_dataset||²`
   - CQL의 원래 policy loss + W2 distance

3. **Actor1+**:
   - Loss: `CQL_policy_loss + w₂ᵢ · Sinkhorn(πᵢ(·|s), πᵢ₋₁(·|s))`
   - CQL의 원래 policy loss + Sinkhorn distance

#### AWAC + Multi-Actor

1. **Base AWAC의 Critic 업데이트** (그대로):
   - Twin Q-network: `Q1, Q2 = AWAC_critic(s, a)`
   - Target: `r + γ·min(Q1(s', a'), Q2(s', a'))`

2. **Actor0**:
   - Loss: `mean(weights · BC_loss) + w₂₀ · ||π₀(s) - a_dataset||²`
   - AWAC의 advantage-weighted BC loss + W2 distance

3. **Actor1+**:
   - Loss: `-Q(s, πᵢ(s)) + w₂ᵢ · Sinkhorn(πᵢ(·|s), πᵢ₋₁(·|s))`
   - Q 최대화 + Sinkhorn distance

#### SAC-N + Multi-Actor

1. **Base SAC-N의 Alpha, Critic 업데이트** (그대로):
   - Alpha: adaptive entropy coefficient
   - Vectorized Critic: ensemble of Q-networks (여러 critic)
   - Critic loss: `MSE(Q_ensemble, target_Q)`

2. **Actor0**:
   - Loss: `(α·log_π - Q_min) + w₂₀ · ||π₀(s) - a_dataset||²`
   - SAC-N의 원래 actor loss + W2 distance

3. **Actor1+**:
   - Loss: `(α·log_π - Q_min) + w₂ᵢ · Sinkhorn(πᵢ(·|s), πᵢ₋₁(·|s))`
   - SAC-N의 원래 actor loss + Sinkhorn distance

#### EDAC + Multi-Actor

1. **Base EDAC의 Alpha, Critic 업데이트** (그대로):
   - Alpha: adaptive entropy coefficient
   - Vectorized Critic: ensemble of Q-networks
   - Critic loss: `MSE(Q_ensemble, target_Q) + η·diversity_loss`
   - Diversity loss: critic 간 gradient diversity 유지

2. **Actor0**:
   - Loss: `(α·log_π - Q_min) + w₂₀ · ||π₀(s) - a_dataset||²`
   - EDAC의 원래 actor loss + W2 distance

3. **Actor1+**:
   - Loss: `(α·log_π - Q_min) + w₂ᵢ · Sinkhorn(πᵢ(·|s), πᵢ₋₁(·|s))`
   - EDAC의 원래 actor loss + Sinkhorn distance

#### TD3_BC + Multi-Actor

1. **Base TD3_BC의 Critic 업데이트** (그대로):
   - Twin Q-network: `Q1, Q2 = TD3_critic(s, a)`
   - Target: `r + γ·min(Q1(s', a'), Q2(s', a'))`

2. **Actor0**:
   - Loss: `-λ·Q(s, π₀(s)) + ||π₀(s) - a_dataset||² + w₂₀ · ||π₀(s) - a_dataset||²`
   - TD3_BC loss + W2 distance

3. **Actor1+**:
   - Loss: `-λ·Q(s, πᵢ(s)) + w₂ᵢ · Sinkhorn(πᵢ(·|s), πᵢ₋₁(·|s))`
   - TD3_BC loss + Sinkhorn distance

## Config 파라미터 설명

### 필수 파라미터

- `algorithm`: 사용할 알고리즘 선택 (필수)
  - `"iql"`: IQL 구조 사용 (V function + Q function)
  - `"td3_bc"`: TD3_BC 구조 사용 (Twin Critic)
  - `"cql"`: CQL 구조 사용 (Conservative Q-learning, 원래 Q loss 그대로 사용)
  - `"awac"`: AWAC 구조 사용 (Twin Critic + Advantage-weighted BC)
  - `"sac_n"`: SAC-N 구조 사용 (Vectorized Critic ensemble)
  - `"edac"`: EDAC 구조 사용 (Vectorized Critic ensemble + diversity loss)

### POGO Multi-Actor 전용 파라미터

- `w2_weights`: 각 actor의 W2/Sinkhorn loss 가중치 리스트
- `num_actors`: Actor 개수 (None이면 `len(w2_weights)` 사용)
- `actor_configs`: 각 actor의 설정 리스트 (예: `[{"deterministic": false}, ...]`)
- `sinkhorn_K`: Sinkhorn 계산 시 각 state당 샘플 수
- `sinkhorn_blur`: Sinkhorn regularization parameter (epsilon)
- `sinkhorn_backend`: Sinkhorn backend ("tensorized", "online", "auto")

### 알고리즘별 파라미터

각 알고리즘의 파라미터는 해당 알고리즘의 config와 동일하게 사용:

- **IQL**: `iql_tau`, `beta`, `vf_lr`, `qf_lr`, `actor_lr`, `iql_deterministic`
- **TD3_BC**: `alpha`, `policy_noise`, `noise_clip`, `policy_freq`
- **CQL**: `cql_alpha`, `cql_n_actions`, `target_entropy`, `qf_lr`, `actor_lr`
- **AWAC**: `awac_lambda`, `exp_adv_max`, `actor_lr`
- **SAC-N**: `num_critics`, `alpha_learning_rate`, `actor_lr`
- **EDAC**: `num_critics`, `eta`, `alpha_learning_rate`, `actor_lr`

## 평가

각 actor마다 독립적으로 평가가 수행됩니다:
- `eval_actor_{i}`: Actor i의 normalized score
- 모든 actor의 평가 결과가 wandb에 로깅됩니다

## 파일 구조

```
algorithms/offline/
├── pogo_multi_main.py          # 통합 학습 스크립트 (각 알고리즘 선택 가능)
├── pogo_policies.py            # POGO policies (StochasticMLP, DeterministicMLP)
├── pogo_multi.py               # 이전 구현 (참조용)
├── iql.py                      # IQL (그대로 사용)
├── td3_bc.py                   # TD3_BC (그대로 사용)
├── cql.py                      # CQL (그대로 사용)
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
│   └── ...
└── ...
```

## 참고

- **POGO_sv**: `/home/offrl/POGO_sv` - 원본 baseline 프로젝트
- **CORL**: 이 프로젝트가 통합된 프레임워크
- **GeomLoss**: Sinkhorn distance 계산을 위한 라이브러리

## 예시 실행

### IQL 구조로 multi-actor 학습

```bash
python -m algorithms.offline.pogo_multi_main \
    --config_path configs/offline/pogo_multi/halfcheetah/medium_v2_iql.yaml
```

### TD3_BC 구조로 multi-actor 학습

```bash
python -m algorithms.offline.pogo_multi_main \
    --config_path configs/offline/pogo_multi/halfcheetah/medium_v2_td3_bc.yaml
```

### CQL 구조로 multi-actor 학습

```bash
python -m algorithms.offline.pogo_multi_main \
    --config_path configs/offline/pogo_multi/halfcheetah/medium_v2_cql.yaml
```

### AWAC 구조로 multi-actor 학습

```bash
python -m algorithms.offline.pogo_multi_main \
    --config_path configs/offline/pogo_multi/halfcheetah/medium_v2_awac.yaml
```

### SAC-N 구조로 multi-actor 학습

```bash
python -m algorithms.offline.pogo_multi_main \
    --config_path configs/offline/pogo_multi/halfcheetah/medium_v2_sac_n.yaml
```

### EDAC 구조로 multi-actor 학습

```bash
python -m algorithms.offline.pogo_multi_main \
    --config_path configs/offline/pogo_multi/halfcheetah/medium_v2_edac.yaml
```

### Custom 설정으로 실행

```bash
python -m algorithms.offline.pogo_multi_main \
    --config_path configs/offline/pogo_multi/halfcheetah/medium_v2_iql.yaml \
    --algorithm iql \
    --w2_weights 20.0 20.0 20.0 \
    --sinkhorn_K 8 \
    --sinkhorn_blur 0.1 \
    --max_timesteps 2000000
```

## 문제 해결

### Import 에러

`python -m algorithms.offline.pogo_multi_main` 형식으로 실행해야 합니다 (상대 import 때문).

### GeomLoss 설치 에러

```bash
pip install geomloss
```

### CUDA/Display 에러

Headless 환경에서는:
```bash
export MUJOCO_GL=egl
export D4RL_SUPPRESS_IMPORT_ERROR=1
```

## 라이센스

CORL 프로젝트의 라이센스를 따릅니다.
