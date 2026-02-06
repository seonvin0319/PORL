# POGO Multi-Actor 실행 흐름 (Flow)

## 전체 실행 흐름 개요

```
1. Config 로드 및 병합
2. 환경 및 데이터셋 준비
3. Agent 초기화 (Multi-Actor + Critic)
4. 학습 루프 (매 step마다)
   - Critic 업데이트
   - Actor 업데이트 (policy_freq마다)
5. 평가 (eval_freq마다)
```

---

## 상세 실행 흐름

### 1. 프로그램 시작: `train()` 함수 호출

```python
@pyrallis.wrap()  # pyrallis가 config 파일을 자동으로 파싱
def train(config: TrainConfig):
```

**진입점**: `python -m algorithms.offline.pogo_multi --config_path config.yaml`

---

### 2. Config 병합 (`_merge_base_config`)

```python
config = _merge_base_config(config)
```

**동작**:
- `base_algorithm` 또는 `base_config_path`가 설정되어 있으면
- 해당 알고리즘의 config 파일을 찾아서 로드
- 예: `base_algorithm: iql` → `configs/offline/iql/halfcheetah/medium_v2.yaml` 로드
- 다음 파라미터들을 병합: `batch_size`, `discount`, `tau`, `lr`, `buffer_size`, `normalize`, `normalize_reward`

**예시**:
```yaml
# pogo_multi config
base_algorithm: iql
w2_weights: [10.0, 10.0, 10.0]

# → IQL config의 batch_size=256, lr=3e-4 등이 자동으로 적용됨
```

---

### 3. 환경 및 데이터셋 준비

```python
env = gym.make(config.env)  # 예: "halfcheetah-medium-v2"
dataset = d4rl.qlearning_dataset(env)  # D4RL 데이터셋 로드
```

**단계**:
1. 환경 생성 및 state/action dimension 확인
2. D4RL 데이터셋 로드
3. Reward 정규화 (선택사항)
4. State 정규화 (선택사항)
   - Mean, std 계산
   - Dataset의 observations, next_observations 정규화
   - 환경에 wrapper 적용
5. ReplayBuffer에 데이터 로드

**결과**: `replay_buffer`에 정규화된 데이터가 저장됨

---

### 4. Agent 초기화 (`POGOMultiAgent.__init__`)

```python
agent = POGOMultiAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    max_action=max_action,
    w2_weights=config.w2_weights,  # 예: [10.0, 10.0, 10.0]
    ...
)
```

**초기화 내용**:

#### 4.1. Actor 생성 (여러 개)
```python
for i in range(self.num_actors):  # 예: 3개
    if deterministic:
        actor = DeterministicMLP(...)  # z 없이 state → action
    else:
        actor = StochasticMLP(...)     # state + z → action
    
    actor_target = copy.deepcopy(actor)  # Target network
    self.actors.append(actor)
    self.actor_targets.append(actor_target)
    self.actor_optimizers.append(Adam(...))
```

**결과**:
- `self.actors[0]`, `self.actors[1]`, `self.actors[2]` (3개 actor)
- 각각의 target network와 optimizer

#### 4.2. Critic 생성
```python
self.critic = Critic(state_dim, action_dim)  # Twin Q-network
self.critic_target = copy.deepcopy(self.critic)
```

**Critic 구조**:
- Q1, Q2 두 개의 Q-network (TD3 스타일)
- 각각: `state + action → Q value`

#### 4.3. Sinkhorn Loss 초기화
```python
self._sinkhorn_loss = SamplesLoss(
    loss="sinkhorn", p=2, blur=0.05, backend="tensorized"
)
```

---

### 5. 학습 루프 (매 step마다)

```python
for t in range(int(config.max_timesteps)):  # 예: 1,000,000 steps
    batch = replay_buffer.sample(config.batch_size)  # 256개 샘플
    log_dict = agent.train(batch)  # ← 여기가 핵심!
```

---

### 6. `agent.train(batch)` 상세 흐름

#### 6.1. Batch 준비
```python
state, action, reward, next_state, done = batch
# state: [256, state_dim]
# action: [256, action_dim]  ← dataset의 action
# reward: [256, 1]
# next_state: [256, state_dim]
# done: [256, 1]
```

#### 6.2. Critic 업데이트 (매 step마다)

```python
# Target Q 계산
with torch.no_grad():
    # Actor0의 target network로 next action 생성
    actor0_target = self.actor_targets[0]
    next_action = actor0_target.sample_actions(next_state, K=1)[:, 0, :]
    
    # TD3-style noise 추가
    noise = policy_noise * randn(...).clamp(-noise_clip, noise_clip)
    noisy_next_action = (next_action + noise).clamp(-max_action, max_action)
    
    # Target Q 계산
    target_Q1, target_Q2 = critic_target(next_state, noisy_next_action)
    target_Q = reward + (1 - done) * discount * min(target_Q1, target_Q2)

# Critic loss 및 업데이트
current_Q1, current_Q2 = critic(state, action)
critic_loss = MSE(current_Q1, target_Q) + MSE(current_Q2, target_Q)
critic_loss.backward()
critic_optimizer.step()
```

**핵심**: Critic은 Actor0의 target network를 사용하여 TD target 계산

---

#### 6.3. Actor 업데이트 (`policy_freq`마다, 예: 2 step마다)

```python
if self.total_it % self.policy_freq == 0:  # 예: 2 step마다
    for i in range(self.num_actors):  # Actor0, Actor1, Actor2 순차 업데이트
```

##### Actor0 업데이트 (W2 to dataset)

```python
i = 0
actor_i = self.actors[0]
pi_i = actor_i.sample_actions(state, K=1)[:, 0, :]  # Actor0의 action

# Q 값 계산
Q_i = critic.Q1(state, pi_i)

# W2 distance: Actor0 action과 dataset action의 L2
w2_i = ((pi_i - action) ** 2).mean()  # ← Dataset action과의 거리!

# Actor loss
actor_loss_i = -Q_i.mean() + w2_weight_i * w2_i
#              ↑Q 최대화    ↑Dataset에 가깝게
```

**의미**: Actor0는 Q를 최대화하면서도 dataset action에 가까워지도록 학습

---

##### Actor1, Actor2 업데이트 (Sinkhorn to previous actor)

```python
i = 1  # 또는 2
actor_i = self.actors[i]
ref_actor = self.actors[i - 1]  # 이전 actor 참조

pi_i = actor_i.sample_actions(state, K=1)[:, 0, :]
Q_i = critic.Q1(state, pi_i)

# Sinkhorn distance 계산
if is_stochastic and ref_is_stochastic:
    # 두 actor 모두 stochastic이면 Sinkhorn 사용
    w2_i = _per_state_sinkhorn(
        actor_i,      # 현재 actor
        ref_actor,    # 이전 actor (참조)
        state,
        K=4,          # 각 state당 4개 샘플
        blur=0.05,
        ...
    )
else:
    # Deterministic이면 L2 사용
    ref_action = ref_actor.deterministic_actions(state)
    w2_i = ((pi_i - ref_action) ** 2).mean()

# Actor loss
actor_loss_i = -Q_i.mean() + w2_weight_i * w2_i
```

**Sinkhorn 계산 과정** (`_per_state_sinkhorn`):
```python
# 현재 actor에서 K개 샘플
a = actor_i.sample_actions(states, K=4)  # [B, 4, action_dim]

# 이전 actor에서 K개 샘플 (detached)
with torch.no_grad():
    b = ref_actor.sample_actions(states, K=4).detach()  # [B, 4, action_dim]

# Sinkhorn distance 계산
loss = sinkhorn_loss(a, b)  # GeomLoss 사용
return loss.mean()
```

**의미**: Actor1은 Q를 최대화하면서도 Actor0에 가까워지도록 학습 (Sinkhorn으로 분포 거리 측정)

---

#### 6.4. Critic Gradient 제어

```python
if i == 0:
    # Actor0 업데이트 시: Critic gradient 활성화
    for p in critic.parameters():
        p.requires_grad_(True)
elif i == 1:
    # Actor1 업데이트 시: Critic gradient 비활성화
    for p in critic.parameters():
        p.requires_grad_(False)
```

**이유**: Actor1+ 업데이트 시 Critic이 변하지 않도록 고정

---

#### 6.5. Target Network 업데이트 (Soft Update)

```python
# 모든 actor 업데이트 후
for actor, actor_target in zip(self.actors, self.actor_targets):
    # Soft update: target = tau * actor + (1 - tau) * target
    for p, tp in zip(actor.parameters(), actor_target.parameters()):
        tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

# Critic도 동일하게
for p, tp in zip(critic.parameters(), critic_target.parameters()):
    tp.data.copy_(tau * p.data + (1 - tau) * tp.data)
```

**tau**: 예: 0.005 (매우 작은 값으로 천천히 업데이트)

---

### 7. 평가 (`eval_freq`마다, 예: 5000 step마다)

```python
if (t + 1) % config.eval_freq == 0:
    for i in range(agent.num_actors):
        scores = eval_actor(agent, env, device, n_episodes=10, actor_idx=i)
        norm_score = env.get_normalized_score(scores.mean()) * 100.0
        print(f"Actor {i} eval (norm): {norm_score:.1f}")
```

**평가 과정**:
1. 각 actor마다 독립적으로 평가
2. 10개 episode 실행
3. Normalized score 계산 (D4RL 스타일)
4. Wandb에 로깅

---

## 핵심 개념 정리

### 1. Multi-Actor Chain 구조

```
Dataset Actions
    ↓ (W2 distance)
Actor0 ← Critic (Q 최대화)
    ↓ (Sinkhorn distance)
Actor1 ← Critic (Q 최대화)
    ↓ (Sinkhorn distance)
Actor2 ← Critic (Q 최대화)
```

### 2. Actor Loss 공식

- **Actor0**: `L₀ = -Q(s, π₀(s)) + w₂₀ · ||π₀(s) - a_dataset||²`
- **Actor1+**: `Lᵢ = -Q(s, πᵢ(s)) + w₂ᵢ · Sinkhorn(πᵢ(·|s), πᵢ₋₁(·|s))`

### 3. 학습 순서

1. **Critic 업데이트** (매 step)
   - Actor0의 target network로 TD target 계산
   - MSE loss로 업데이트

2. **Actor 업데이트** (`policy_freq`마다)
   - Actor0 → Actor1 → Actor2 순차 업데이트
   - 각 actor는 Q 최대화 + 이전 actor/dataset과의 거리 최소화

### 4. Sinkhorn vs L2

- **Sinkhorn**: 두 분포 간의 거리 측정 (stochastic policy 간)
- **L2**: 두 점 간의 거리 측정 (deterministic 또는 dataset action과)

---

## 실행 예시 (타임라인)

```
Step 0:
  - Critic 업데이트 (Actor0 target 사용)
  
Step 1:
  - Critic 업데이트
  
Step 2 (policy_freq=2):
  - Critic 업데이트
  - Actor0 업데이트 (W2 to dataset)
  - Actor1 업데이트 (Sinkhorn to Actor0)
  - Actor2 업데이트 (Sinkhorn to Actor1)
  - Target networks soft update

Step 3:
  - Critic 업데이트
  
Step 4:
  - Critic 업데이트
  - Actor 업데이트 (반복)
  
...

Step 5000 (eval_freq=5000):
  - Critic 업데이트
  - Actor 업데이트
  - 평가: Actor0, Actor1, Actor2 각각 10 episode 실행
  - 결과 출력 및 Wandb 로깅
```

---

## 데이터 흐름 다이어그램

```
ReplayBuffer
    ↓ sample(batch_size=256)
Batch: (state, action, reward, next_state, done)
    ↓
┌─────────────────────────────────────┐
│ agent.train(batch)                  │
├─────────────────────────────────────┤
│ 1. Critic 업데이트                  │
│    - next_action = Actor0_target   │
│    - target_Q = r + γ·Q(s', a')    │
│    - critic_loss = MSE(Q, target_Q)│
│                                     │
│ 2. Actor 업데이트 (policy_freq마다) │
│    Actor0:                          │
│      - π₀ = Actor0(state)          │
│      - loss = -Q + w₂₀·||π₀-a||²   │
│                                     │
│    Actor1:                          │
│      - π₁ = Actor1(state)          │
│      - loss = -Q + w₂₁·Sinkhorn(π₁,π₀)│
│                                     │
│    Actor2:                          │
│      - π₂ = Actor2(state)          │
│      - loss = -Q + w₂₂·Sinkhorn(π₂,π₁)│
│                                     │
│ 3. Target update (soft)            │
└─────────────────────────────────────┘
    ↓
Metrics: {critic_loss, actor_0_loss, w2_0_distance, ...}
    ↓
Wandb 로깅
```

---

## 요약

1. **Config 병합**: 다른 알고리즘의 hyperparameter 가져오기
2. **Multi-Actor 초기화**: 여러 actor와 critic 생성
3. **학습 루프**:
   - Critic: Actor0 target으로 TD target 계산
   - Actor0: Dataset action과의 L2로 학습
   - Actor1+: 이전 actor와의 Sinkhorn으로 학습
4. **평가**: 각 actor마다 독립 평가

**핵심 아이디어**: Actor들을 chain으로 연결하여 순차적으로 개선시키는 JKO (Jordan-Kinderlehrer-Otto) 방식
