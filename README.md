# POGO Multi-Actor: CORL Integration

POGO_sv 프로젝트를 baseline으로 하여 CORL 프로젝트에 통합한 multi-actor offline reinforcement learning 알고리즘입니다.

## 프로젝트 개요

이 프로젝트는 **각 기존 알고리즘의 구조를 그대로 유지**하면서, **Actor만 multi-actor로 교체**하여 학습하는 통합 프레임워크입니다.

### 핵심 원칙

- ✅ **각 알고리즘의 Critic, V, Q 구조는 변경 없음**
- ✅ **Actor만 multi-actor로 교체** (개수와 loss만 변경)
- ✅ **Config에서 algorithm 선택 가능** (`algorithm: iql`, `td3_bc`, `cql`, `awac`, `sac_n`, `edac`)
- ✅ **기존 알고리즘 파일은 전혀 수정하지 않음**

## 주요 특징

### Multi-Actor 구조

- **Actor0**: W2 (Wasserstein-2) distance를 사용하여 dataset action과의 L2 loss로 학습
  - `w2_weight[0] = 0` (현재 설정: actor0는 기존 알고리즘)
- **Actor1+**: Sinkhorn distance를 사용하여 이전 actor와의 분포 거리로 학습
  - `w2_weight[1] = 10.0` (일반 환경), `100.0` (expert 환경)
  - `w2_weight[2] = 10.0` (일반 환경), `100.0` (expert 환경)

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

```bash
# IQL 구조로 multi-actor 학습
python -m algorithms.offline.pogo_multi_main \
    --config_path configs/offline/pogo_multi/halfcheetah/medium_v2_iql.yaml

# 다른 알고리즘 예시
python -m algorithms.offline.pogo_multi_main \
    --config_path configs/offline/pogo_multi/halfcheetah/medium_v2_awac.yaml
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
- 일반 환경: `[0.0, 10.0, 10.0]` (actor0=0, actor1/2만 적용)
- Expert 환경: `[0.0, 100.0, 100.0]` (actor0=0, actor1/2는 10배)

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
w2_weights: [0.0, 10.0, 10.0]  # actor0=0, actor1/2만 적용
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

- `algorithms/offline/pogo_multi_main.py`: 메인 학습 스크립트
- `algorithms/offline/pogo_policies.py`: POGO policies (StochasticMLP, DeterministicMLP)
- `configs/offline/pogo_multi/`: 각 알고리즘별 config 파일
- `run_iql_all_envs.sh`: 전체 환경 순차 학습 스크립트

## 상세 문서

- `algorithms/offline/POGO_MULTI_README.md`: 상세 사용법 및 알고리즘별 설명
- `algorithms/offline/POGO_MULTI_ARCHITECTURE.md`: 아키텍처 및 구현 세부사항
- `algorithms/offline/POGO_MULTI_FLOW.md`: 실행 흐름 및 동작 원리

## 주의사항

- **기존 알고리즘 파일은 수정하지 않음**: `iql.py`, `td3_bc.py`, `cql.py` 등은 그대로 유지
- **Actor0의 w2_weight는 0**: 현재 설정에서 actor0는 W2/Sinkhorn penalty 없이 학습
- **Expert 환경은 weights * 10**: `run_iql_all_envs.sh`에서 자동 적용

## 참고

- CORL: [Consistent Offline Reinforcement Learning](https://github.com/corl-team/CORL)
- POGO: Policy Optimization of Gradient flow for Offline Reinforcement Learning
- D4RL: [Datasets for Deep Data-Driven Reinforcement Learning](https://github.com/Farama-Foundation/D4RL)
