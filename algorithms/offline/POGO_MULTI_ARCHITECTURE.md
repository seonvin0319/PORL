# POGO Multi-Actor 아키텍처 설명

## 핵심 설계 원칙

1. **각 알고리즘의 구조를 그대로 사용**: Critic, V function, Q function 등은 각 알고리즘의 원래 구조 유지
2. **Actor만 multi-actor로 교체**: Actor 개수와 Actor loss만 변경
3. **Config에서 algorithm 선택**: `algorithm: iql` 또는 `algorithm: td3_bc` 등으로 선택

## 구조 비교

### 기존 알고리즘 (예: IQL)
```
IQL Trainer:
  - V function (ValueFunction)
  - Q function (TwinQ)
  - Actor (GaussianPolicy 또는 DeterministicPolicy) ← 단일 actor
  - train(): V, Q, Actor 순차 업데이트

CQL Trainer:
  - Critic (FullyConnectedQFunction) × 2
  - Actor (TanhGaussianPolicy) ← 단일 actor
  - train(): Q loss (CQL penalty 포함), Actor loss

AWAC Trainer:
  - Critic (Twin Critic)
  - Actor (GaussianPolicy) ← 단일 actor
  - train(): Critic loss, Advantage-weighted Actor loss

SAC-N Trainer:
  - Critic (VectorizedCritic - ensemble of Q-networks)
  - Actor (GaussianPolicy) ← 단일 actor
  - train(): Alpha loss, Critic loss, Actor loss

EDAC Trainer:
  - Critic (VectorizedCritic - ensemble + diversity loss)
  - Actor (GaussianPolicy) ← 단일 actor
  - train(): Alpha loss, Critic loss (with diversity), Actor loss
```

### POGO Multi-Actor (IQL 기반)
```
IQL Trainer (그대로 사용):
  - V function (ValueFunction) ← 그대로
  - Q function (TwinQ) ← 그대로
  - Actor (첫 번째 actor만 연결, 호환성용)

Multi-Actor Wrapper:
  - Actor0 (POGO StochasticMLP)
  - Actor1 (POGO StochasticMLP)
  - Actor2 (POGO StochasticMLP)
  - train(): 
    1. Base trainer의 V, Q 업데이트 (그대로)
    2. Multi-actor 업데이트 (새로 추가)
```

### POGO Multi-Actor (CQL 기반)
```
CQL Trainer (그대로 사용):
  - Critic (FullyConnectedQFunction) × 2 ← 그대로
  - Actor (TanhGaussianPolicy, Q loss 계산용) ← 그대로
  - Q loss: 원래 _q_loss 메서드 그대로 사용 (OOD 샘플링, CQL penalty 포함)

Multi-Actor Wrapper:
  - Actor0 (POGO StochasticMLP)
  - Actor1 (POGO StochasticMLP)
  - Actor2 (POGO StochasticMLP)
  - train():
    1. Base trainer의 Q loss 계산 (원래 _q_loss 그대로)
    2. Multi-actor 업데이트 (새로 추가)
```

### POGO Multi-Actor (SAC-N/EDAC 기반)
```
SAC-N/EDAC Trainer (그대로 사용):
  - Critic (VectorizedCritic - ensemble) ← 그대로
  - Actor (첫 번째 actor만 연결, 호환성용)
  - Alpha: adaptive entropy coefficient ← 그대로
  - Critic loss: ensemble MSE loss (+ diversity loss for EDAC) ← 그대로

Multi-Actor Wrapper:
  - Actor0 (POGO StochasticMLP)
  - Actor1 (POGO StochasticMLP)
  - Actor2 (POGO StochasticMLP)
  - train():
    1. Base trainer의 Alpha, Critic 업데이트 (그대로)
    2. Multi-actor 업데이트 (새로 추가)
```

## 학습 흐름

### IQL + Multi-Actor 예시

```python
# 1. Base trainer 생성 (IQL 구조 그대로)
trainer = ImplicitQLearning(
    v_network=v_network,  # IQL의 V function
    q_network=q_network,  # IQL의 TwinQ
    actor=actors[0],  # 첫 번째 actor만 연결 (호환성)
    ...
)

# 2. 학습 루프
for batch in replay_buffer:
    # Base trainer의 V, Q 업데이트 (IQL 방식 그대로)
    adv = trainer._update_v(observations, actions, log_dict)
    trainer._update_q(next_v, observations, actions, rewards, dones, log_dict)
    
    # Multi-actor 업데이트 (새로 추가)
    for i in range(num_actors):
        if i == 0:
            # Actor0: IQL loss + W2 to dataset
            actor_loss = IQL_style_loss(actor0, adv) + w2_weight * ||actor0 - dataset||²
        else:
            # Actor1+: Q 기반 + Sinkhorn to previous actor
            actor_loss = -Q(actor_i) + w2_weight * Sinkhorn(actor_i, actor_{i-1})
```

### TD3_BC + Multi-Actor 예시

```python
# 1. Base trainer 생성 (TD3_BC 구조 그대로)
trainer = TD3_BC(
    critic_1=critic_1,  # TD3의 twin critic
    critic_2=critic_2,
    actor=actors[0],  # 첫 번째 actor만 연결
    ...
)

# 2. 학습 루프
for batch in replay_buffer:
    # Base trainer의 Critic 업데이트 (TD3 방식 그대로)
    critic_loss = MSE(Q1, target_Q) + MSE(Q2, target_Q)
    critic_optimizer.step()
    
    # Multi-actor 업데이트 (policy_freq마다)
    if step % policy_freq == 0:
        for i in range(num_actors):
            if i == 0:
                # Actor0: TD3_BC loss + W2
                actor_loss = -lambda * Q(actor0) + ||actor0 - dataset||² + w2_weight * ||actor0 - dataset||²
            else:
                # Actor1+: TD3_BC loss + Sinkhorn
                actor_loss = -lambda * Q(actor_i) + w2_weight * Sinkhorn(actor_i, actor_{i-1})
```

## Actor Loss 공식

### 각 알고리즘별 Actor Loss (기존)

- **IQL**: `L = mean(exp(β·adv) · BC_loss)`
- **TD3_BC**: `L = -λ·Q + MSE(π, a_dataset)` (λ = α / |Q|)
- **CQL**: `L = α·log_π - Q` (BC steps 후) 또는 `L = α·log_π - log_prob(a_dataset)` (BC steps 중)
- **AWAC**: `L = -mean(weights · log_prob)` (weights = exp(adv/λ))
- **SAC-N**: `L = α·log_π - Q_min` (Q_min = min over ensemble)
- **EDAC**: `L = α·log_π - Q_min` (Q_min = min over ensemble, 동일)

### POGO Multi-Actor 적용 후

- **Actor0**: `L₀ = [기존 알고리즘 loss] + w₂₀ · ||π₀ - a_dataset||²`
- **Actor1+**: `Lᵢ = [기존 알고리즘 loss] + w₂ᵢ · Sinkhorn(πᵢ, πᵢ₋₁)`

## 파일 구조

```
algorithms/offline/
├── pogo_multi_main.py      # 통합 학습 스크립트 (각 알고리즘 선택 가능)
├── pogo_policies.py         # POGO policies (StochasticMLP, DeterministicMLP)
├── iql.py                   # IQL (그대로 사용)
├── td3_bc.py                # TD3_BC (그대로 사용)
├── cql.py                   # CQL (그대로 사용)
└── ...

configs/offline/pogo_multi/
├── halfcheetah/
│   ├── medium_v2_iql.yaml      # algorithm: iql
│   ├── medium_v2_td3_bc.yaml   # algorithm: td3_bc
│   └── ...
```

## 사용 방법

```bash
# IQL 구조로 multi-actor 학습
python -m algorithms.offline.pogo_multi_main \
    --config_path configs/offline/pogo_multi/halfcheetah/medium_v2_iql.yaml

# TD3_BC 구조로 multi-actor 학습
python -m algorithms.offline.pogo_multi_main \
    --config_path configs/offline/pogo_multi/halfcheetah/medium_v2_td3_bc.yaml

# CQL 구조로 multi-actor 학습
python -m algorithms.offline.pogo_multi_main \
    --config_path configs/offline/pogo_multi/halfcheetah/medium_v2_cql.yaml

# AWAC 구조로 multi-actor 학습
python -m algorithms.offline.pogo_multi_main \
    --config_path configs/offline/pogo_multi/halfcheetah/medium_v2_awac.yaml

# SAC-N 구조로 multi-actor 학습
python -m algorithms.offline.pogo_multi_main \
    --config_path configs/offline/pogo_multi/halfcheetah/medium_v2_sac_n.yaml

# EDAC 구조로 multi-actor 학습
python -m algorithms.offline.pogo_multi_main \
    --config_path configs/offline/pogo_multi/halfcheetah/medium_v2_edac.yaml
```

## 주요 차이점

| 항목 | 기존 알고리즘 | POGO Multi-Actor |
|------|--------------|------------------|
| Critic 구조 | 각 알고리즘 고유 | **그대로 유지** |
| V function (IQL) | IQL 고유 | **그대로 유지** |
| Q function | 각 알고리즘 고유 | **그대로 유지** |
| Actor 개수 | 1개 | **여러 개 (config에서 지정)** |
| Actor 구조 | 각 알고리즘 고유 | **POGO policies로 통일** |
| Actor Loss | 각 알고리즘 고유 | **기존 loss + W2/Sinkhorn** |

## 구현 세부사항

### 1. Actor 생성
- 모든 알고리즘에서 **POGO policies (StochasticMLP/DeterministicMLP) 사용**
- `sample_actions`, `deterministic_actions`, `act` 메서드 지원

### 2. Base Trainer 연결
- 각 알고리즘의 trainer를 그대로 생성
- 첫 번째 actor만 base trainer에 연결 (호환성)
- 나머지 actor들은 독립적으로 관리

### 3. 학습 함수
- 각 알고리즘별로 `_train_{algorithm}_multi_actor` 함수 구현
  - `_train_iql_multi_actor`: IQL의 V, Q 업데이트 그대로 호출
  - `_train_td3_bc_multi_actor`: TD3_BC의 Critic 업데이트 그대로 호출
  - `_train_cql_multi_actor`: CQL의 원래 `_q_loss` 메서드 그대로 호출
  - `_train_awac_multi_actor`: AWAC의 Critic 업데이트 그대로 호출
  - `_train_sac_n_multi_actor`: SAC-N의 Alpha, Critic 업데이트 그대로 호출
  - `_train_edac_multi_actor`: EDAC의 Alpha, Critic 업데이트 그대로 호출 (diversity loss 포함)
- Base trainer의 V, Q, Critic 업데이트는 그대로 호출
- Actor 업데이트만 multi-actor로 교체

### 4. 평가
- 각 actor마다 독립적으로 평가
- Base trainer의 eval 함수 사용하되 actor만 교체

## 지원 알고리즘

현재 지원하는 알고리즘:

1. **IQL** (Implicit Q-Learning)
   - V function + Q function (expectile regression)
   - Advantage-weighted actor loss

2. **TD3_BC** (Twin Delayed DDPG + Behavior Cloning)
   - Twin Critic (TD3 style)
   - BC regularization

3. **CQL** (Conservative Q-Learning)
   - 원래 Q loss 그대로 사용 (`_q_loss` 메서드)
   - OOD action 샘플링, logsumexp, CQL penalty 모두 포함
   - TanhGaussianPolicy 사용 (Q loss 계산용)

4. **AWAC** (Advantage Weighted Actor Critic)
   - Twin Critic
   - Advantage-weighted BC

5. **SAC-N** (Soft Actor-Critic with N critics)
   - Vectorized Critic (ensemble of Q-networks)
   - Adaptive entropy coefficient

6. **EDAC** (Ensemble-Diversified Actor-Critic)
   - Vectorized Critic (ensemble)
   - Critic diversity loss 추가
   - Adaptive entropy coefficient

## 핵심 보장: 기존 알고리즘 파일은 전혀 수정하지 않음

✅ **기존 알고리즘 파일들은 그대로 유지됩니다**:
- `iql.py`, `td3_bc.py`, `cql.py`, `awac.py`, `sac_n.py`, `edac.py` 등은 **전혀 수정하지 않음**
- 단지 `pogo_multi_main.py`에서 **import만** 사용
- 각 알고리즘의 trainer 클래스를 그대로 인스턴스화하고, 원래 메서드를 그대로 호출

✅ **각 알고리즘의 원래 로직 그대로 사용**:
- IQL: `trainer._update_v()`, `trainer._update_q()` 그대로 호출
- CQL: `trainer._q_loss()` 그대로 호출 (OOD 샘플링, CQL penalty 모두 포함)
- TD3_BC: Critic 업데이트 로직 그대로 사용
- EDAC: `trainer._critic_loss()` 그대로 호출 (diversity loss 포함)

✅ **Actor만 교체 - 하지만 구조적으로 비슷함**:
- 기존 알고리즘의 Actor는 사용하지 않음
- 대신 POGO policies (`StochasticMLP`/`DeterministicMLP`) 사용
- **하지만 POGO policies는 기존 Actor와 구조적으로 크게 다르지 않음**:
  
  **비교:**
  - **IQL의 `GaussianPolicy`**: MLP → Normal distribution → action
  - **POGO의 `StochasticMLP`**: MLP (state + z) → action (tanh)
  - **IQL의 `DeterministicPolicy`**: MLP → action (tanh)
  - **POGO의 `DeterministicMLP`**: MLP → action (tanh)
  
  **공통점:**
  - 둘 다 MLP 기반 네트워크 구조
  - 둘 다 `act(state, device)` 메서드 제공 (evaluation용)
  - 둘 다 state를 입력받아 action을 출력
  - 출력 범위: `[-max_action, max_action]` (tanh 사용)
  
  **차이점:**
  - IQL의 `GaussianPolicy`는 Normal distribution을 반환하지만, 실제 사용 시에는 `dist.sample()` 또는 `dist.mean` 사용
  - POGO는 `sample_actions()`, `deterministic_actions()` 메서드로 명시적으로 구분
  - 하지만 최종적으로는 비슷한 방식으로 action 생성
  
- **각 알고리즘의 Actor loss 계산 방식은 그대로 유지**:
  - IQL: `exp(β·adv) · BC_loss` → POGO에서도 동일하게 계산 (단지 actor만 교체)
  - TD3_BC: `-λ·Q + MSE(π, a_dataset)` → POGO에서도 동일하게 계산
  - CQL: `α·log_π - Q` → POGO에서도 동일하게 계산 (log_π는 근사)

## 장점

1. **각 알고리즘의 구조 보존**: Critic, V, Q 등은 원래대로 유지
2. **일관된 Actor 인터페이스**: 모든 알고리즘에서 POGO policies 사용
   - **문제**: 각 알고리즘은 원래 서로 다른 Actor 클래스를 사용
     - IQL: `GaussianPolicy` 또는 `DeterministicPolicy`
     - TD3_BC: `Actor` (deterministic)
     - CQL: `TanhGaussianPolicy`
     - AWAC: `Actor` (GaussianPolicy)
     - SAC-N/EDAC: `Actor` (GaussianPolicy)
   - **해결**: POGO Multi-Actor에서는 모든 알고리즘에서 동일한 POGO policies 사용
     - `StochasticMLP` 또는 `DeterministicMLP`
     - 모두 동일한 인터페이스: `sample_actions()`, `deterministic_actions()`, `act()`
   - **장점**:
     - Multi-actor 학습 로직을 일관되게 작성 가능
     - Sinkhorn distance 계산도 동일한 방식으로 처리 가능
     - 코드 중복 감소 및 유지보수 용이
3. **Config 기반 선택**: algorithm 파라미터로 쉽게 전환
4. **확장성**: 새로운 알고리즘 추가 시 `_train_{new}_multi_actor` 함수만 추가
5. **원래 알고리즘 로직 보존**: CQL의 Q loss, EDAC의 diversity loss 등 복잡한 로직도 그대로 사용
