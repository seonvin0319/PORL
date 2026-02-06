# POGO Multi-Actor: CORL Integration

POGO_sv 프로젝트를 baseline으로 하여 CORL 프로젝트에 통합한 multi-actor offline reinforcement learning 알고리즘입니다.

## 프로젝트 개요

이 프로젝트는 **각 기존 알고리즘의 구조를 그대로 유지**하면서, **Actor만 multi-actor로 교체**하여 학습하는 통합 프레임워크입니다.

### 핵심 원칙

- ✅ **각 알고리즘의 Critic, V, Q 구조는 변경 없음**
- ✅ **Critic 업데이트는 원래 알고리즘의 actor만 사용** (모든 알고리즘 공통)
- ✅ **Actor만 multi-actor로 교체** (개수와 loss만 변경)
- ✅ **Policy loss에서만 multi-actor 학습** (Critic은 기존 방식 그대로)
- ✅ **Config에서 algorithm 선택 가능** (`algorithm: iql`, `td3_bc`, `cql`, `awac`, `sac_n`, `edac`)
- ✅ **기존 알고리즘 파일은 전혀 수정하지 않음**
- ✅ **JAX/PyTorch 양쪽 구현**: 동일한 구조와 로직을 따르는 두 가지 구현 제공
- ✅ **Policy 타입 자동 관리**: `is_gaussian`, `is_stochastic` 속성 자동 감지
- ✅ **코드 구조 개선**: `ActorConfig` dataclass, 공통 헬퍼 함수로 중복 제거

## 주요 특징

### Multi-Actor 구조

**학습 방식:**
- **Critic 업데이트**: 모든 알고리즘에서 원래 알고리즘의 actor만 사용
  - IQL: V, Q 업데이트는 기존 IQL 방식 그대로
  - CQL: Q loss 계산은 원래 TanhGaussianPolicy 사용 (CQL의 `_q_loss` 메서드 그대로)
  - TD3_BC, AWAC, SAC-N, EDAC: Critic 업데이트는 각 알고리즘의 원래 방식 그대로
- **Actor 업데이트**: Multi-actor 학습은 Policy loss에서만 수행
  - **Actor0**: 각 알고리즘의 원래 actor loss만 사용 (W2 penalty 없음)
  - **Actor1+**: 각 알고리즘의 actor loss + W2 distance to previous actor (`w2_weights` 리스트 사용)
    - W2 distance 계산 방법:
      - **Both Gaussian**: Closed-form W2 (`||μ1-μ2||² + ||σ1-σ2||²`)
      - **Both Stochastic (not Gaussian)**: Sinkhorn distance
      - **At least one Deterministic**: L2 distance

### 지원 알고리즘

1. **IQL** (Implicit Q-Learning)
   - V function + Q function (expectile regression)
   - Advantage-weighted actor loss

2. **TD3_BC** (Twin Delayed DDPG + Behavior Cloning)
   - Twin Critic (TD3 style)
   - BC regularization

3. **CQL** (Conservative Q-Learning)
   - 원래 Q loss 그대로 사용 (`_q_loss` 메서드)
   - OOD action 샘플링, CQL penalty 포함

4. **AWAC** (Advantage Weighted Actor Critic)
   - Twin Critic
   - Advantage-weighted BC

5. **SAC-N** (Soft Actor-Critic with N critics)
   - Vectorized Critic (ensemble)
   - Adaptive entropy coefficient

6. **EDAC** (Ensemble-Diversified Actor-Critic)
   - Vectorized Critic (ensemble)
   - Critic diversity loss

## 설치

### 의존성

```bash
pip install geomloss PyYAML
```

또는:

```bash
pip install -r requirements/requirements.txt
```

### 환경 설정

```bash
cd /home/offrl/CORL
source $(conda info --base)/etc/profile.d/conda.sh
conda activate offrl
```

## 사용 방법

### 1. 단일 환경 학습

#### PyTorch 버전
```bash
# IQL 구조로 multi-actor 학습
python -m algorithms.offline.pogo_multi_main \
    --config_path configs/offline/pogo_multi/halfcheetah/medium_v2_iql.yaml

# 다른 알고리즘 예시
python -m algorithms.offline.pogo_multi_main \
    --config_path configs/offline/pogo_multi/halfcheetah/medium_v2_awac.yaml
```

#### JAX 버전
```bash
# ReBRAC 구조로 multi-actor 학습
python -m algorithms.offline.pogo_multi_jax \
    --config_path configs/offline/pogo_multi/halfcheetah/medium_v2_rebrac.yaml
```

### 2. 전체 환경 순차 학습 (IQL)

```bash
# 모든 환경을 순차적으로 학습 (seed 0)
bash run_iql_all_envs.sh
```

**실행 순서:**
1. halfcheetah-medium-v2
2. halfcheetah-medium-expert-v2 (weights: [0, 100, 100])
3. halfcheetah-medium-replay-v2
4. hopper-medium-v2
5. hopper-medium-expert-v2 (weights: [0, 100, 100])
6. hopper-medium-replay-v2
7. walker2d-medium-v2
8. walker2d-medium-expert-v2 (weights: [0, 100, 100])
9. walker2d-medium-replay-v2

**Weights 설정:**
- 일반 환경: `[10.0, 10.0]` (Actor1, Actor2용, Actor0는 W2 penalty 없음)
- Expert 환경: `[100.0, 100.0]` (Actor1, Actor2용, 일반 환경의 10배)

**참고**: `w2_weights`는 Actor1부터 시작하므로, `num_actors = len(w2_weights) + 1`입니다.

## Config 파일 구조

### 예시: IQL

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
```

## 결과 저장 구조

학습 결과는 다음 구조로 저장됩니다:

```
results/
├── {algorithm}/              # 예: iql, awac
│   ├── {env_name}/          # 예: halfcheetah_medium_v2
│   │   └── seed_{seed}/
│   │       ├── logs/
│   │       │   ├── {algorithm}_{timestamp}.log
│   │       │   └── {algorithm}_{env}_seed{seed}_{timestamp}.json
│   │       └── checkpoints/
│   │           └── model_{timestamp}.pt
```

**예시:**
```
results/iql/halfcheetah_medium_v2/seed_0/
├── logs/
│   ├── iql_20260206_124904.log
│   └── iql_halfcheetah_medium_v2_seed0_20260206_124904.json
└── checkpoints/
    └── model_20260206_124904.pt
```

## 주요 파일

- `algorithms/offline/pogo_multi_main.py`: PyTorch 메인 학습 스크립트
- `algorithms/offline/pogo_multi_jax.py`: JAX 메인 학습 스크립트 (ReBRAC 기반)
- `algorithms/offline/pogo_policies.py`: POGO policies (GaussianMLP, TanhGaussianMLP, StochasticMLP, DeterministicMLP)
- `configs/offline/pogo_multi/`: 각 알고리즘별 config 파일
- `run_iql_all_envs.sh`: 전체 환경 순차 학습 스크립트

## 상세 문서

- `algorithms/offline/POGO_MULTI_README.md`: 상세 사용법 및 알고리즘별 설명
- `algorithms/offline/POGO_MULTI_ARCHITECTURE.md`: 아키텍처 및 구현 세부사항

## 주의사항

- **기존 알고리즘 파일은 수정하지 않음**: `iql.py`, `td3_bc.py`, `cql.py` 등은 그대로 유지
- **Critic 업데이트는 원래 알고리즘의 actor만 사용**: 
  - 모든 알고리즘에서 Critic/V/Q 업데이트는 기존 방식 그대로
  - CQL의 경우 Q loss 계산에 원래 TanhGaussianPolicy 사용 (POGO multi-actors는 policy loss에만 사용)
- **Actor0는 W2 penalty 없음**: Actor0는 각 알고리즘의 원래 actor loss만 사용 (w2_weights 리스트에 포함되지 않음)
- **Multi-actor 학습은 Policy loss에서만**: Critic은 기존 방식 그대로, Actor만 multi-actor로 확장
- **Expert 환경은 weights * 10**: `run_iql_all_envs.sh`에서 자동 적용 (`[100.0, 100.0]`)

## 참고

- CORL: [Consistent Offline Reinforcement Learning](https://github.com/corl-team/CORL)
- POGO: Policy Optimization of Gradient flow for Offline Reinforcement Learning
- D4RL: [Datasets for Deep Data-Driven Reinforcement Learning](https://github.com/Farama-Foundation/D4RL)
