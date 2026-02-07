# IQL POGO Multi-Actor 훈련 Flow

IQL로 POGO Multi-Actor를 훈련할 때의 코드와 실행 흐름을 정리합니다.

---

## 1. 실행 명령

```bash
python -m algorithms.offline.pogo_multi_main \
  --config_path configs/offline/iql_pogo_base.yaml \
  --env halfcheetah-medium-v2
```

---

## 2. 초기화 (pogo_multi_main.py → train())

### 2.1 환경·데이터 로드

```python
env = gym.make(config.env)
dataset = d4rl.qlearning_dataset(env)
replay_buffer.load_d4rl_dataset(dataset)
```

### 2.2 Actor 생성 (_create_actors)

```python
# pogo_multi_main.py:659-675
def _iql_actor0():
    # config.iql_deterministic=False이면 GaussianPolicy
    a = GaussianPolicy(state_dim, action_dim, max_action, dropout=drop)
    return a, True, False  # is_stochastic, is_gaussian

actors, actor_targets, actor_optimizers, ... = _create_actors(
    state_dim, action_dim, max_action, config.num_actors,  # num_actors=3 (w2_weights [10,10] + 1)
    config.actor_configs,
    ActorCreationConfig(create_actor0=_iql_actor0, pogo_default_type="gaussian", pogo_default_tanh_mean=True),
    config.device, config.actor_lr,
)
```

**결과:**
- **Actor0**: `GaussianPolicy` (IQL 원본)
- **Actor1, Actor2**: `GaussianMLP` (POGO용, config 기본값)

### 2.3 IQL Trainer 생성

```python
# pogo_multi_main.py:676-696
q_network = TwinQ(state_dim, action_dim)
v_network = ValueFunction(state_dim)
trainer = ImplicitQLearning(
    actor=base_actor0,        # actors[0]만 trainer에 연결
    actor_optimizer=actor_optimizers[0],
    q_network=q_network,
    v_network=v_network,
    iql_tau=0.7, beta=3.0, discount=0.99, tau=0.005, ...
)
```

---

## 3. 메인 훈련 루프

```python
# pogo_multi_main.py:724-728
for t in range(config.max_timesteps):
    batch = replay_buffer.sample(config.batch_size)
    log_dict = train_fn(batch)
    wandb.log(log_dict, step=t+1)
```

---

## 4. 1 step 학습 흐름: train_fn → _train_multi_actor

### 4.1 Step 1: trainer.train(batch) — V, Q, Actor0 업데이트

```python
# pogo_multi_main.py:376-377
if hasattr(trainer, "train"):
    log_dict = trainer.train(batch)
```

이 호출이 IQL의 `train()`을 실행합니다.

```python
# iql.py:418-439
def train(self, batch):
    observations, actions, rewards, next_observations, dones = batch

    # 1) V 업데이트
    with torch.no_grad():
        next_v = self.vf(next_observations)
    adv = self._update_v(observations, actions, log_dict)
    #   target_q = q_target(obs, actions)
    #   adv = target_q - v
    #   v_loss = asymmetric_l2_loss(adv, tau=0.7)
    #   v_optimizer.step()

    # 2) Q 업데이트
    self._update_q(next_v, observations, actions, rewards, dones, log_dict)
    #   targets = r + γ * next_v
    #   q_loss = MSE(qf(obs,a), targets)
    #   q_optimizer.step()
    #   soft_update(q_target, qf, tau)

    # 3) Actor0 업데이트
    self._update_policy(adv, observations, actions, log_dict)
    #   exp_adv = exp(β * adv).clamp(max=100)
    #   policy_loss = mean(exp_adv * bc_losses)  # advantage-weighted BC
    #   actor_optimizer.step()
```

**정리:** V, Q, Actor0은 IQL 원래 방식 그대로 업데이트됩니다.

---

### 4.2 Step 2: Actor1, Actor2 업데이트 (policy_freq마다)

IQL은 `policy_freq`가 없어 기본값 1이 사용되므로, 매 step마다 Actor1, Actor2가 업데이트됩니다.

```python
# pogo_multi_main.py:389-414
state = batch[0]
actions = batch[1]
policy_freq = getattr(trainer, "policy_freq", 1)  # IQL→1

if trainer.total_it % policy_freq == 0:
    for i in range(1, len(actors)):  # i=1, 2 (Actor1, Actor2)
        # base_loss = IQL의 advantage-weighted BC (Actor0와 동일)
        base_loss_fn = lambda actor: trainer.compute_actor_base_loss(actor, state, actions, seed=...)

        actor_loss_i, w2_i = _compute_actor_loss_with_w2(
            base_loss_fn=base_loss_fn,
            actor_i_config=ActorConfig.from_actor(actors[i]),
            ref_actor_config=ActorConfig.from_actor(actors[i-1]),  # 이전 actor
            states=state,
            w2_weight=w2_weights[i-1],  # [10.0, 10.0]
            ...
        )
        # actor_loss_i = base_loss + 10.0 * W2(π_i, π_{i-1})

        actor_optimizers[i].zero_grad()
        actor_loss_i.backward()
        actor_optimizers[i].step()
```

---

### 4.3 Actor1+ base loss: compute_actor_base_loss

```python
# iql.py:442-465
def compute_actor_base_loss(self, actor, state, actions, seed=None):
    # adv 계산 (batch dataset action 기준)
    with torch.no_grad():
        target_q = self.q_target(state, actions)
    v = self.vf(state)
    adv = (target_q - v).detach()
    exp_adv = torch.exp(self.beta * adv).clamp(max=EXP_ADV_MAX)

    # BC loss: actor의 log_prob 또는 MSE
    if hasattr(actor, "log_prob"):
        bc_losses = -actor.log_prob(state, actions)
    else:
        pi = action_for_loss(actor, cfg, state, seed=seed)
        bc_losses = ((pi - actions) ** 2).sum(dim=1)

    return (exp_adv * bc_losses).mean()  # Actor0와 동일한 advantage-weighted BC
```

---

### 4.4 W2 distance (Actor1+)

```python
# pogo_multi_main.py:302-350 → _compute_actor_loss_with_w2
base_loss = base_loss_fn(actor_i)  # advantage-weighted BC
w2_distance = _compute_w2_distance(actor_i, actor_{i-1}, states, ...)

# Gaussian: closed-form W2
#   mean_i, std_i = actor_i.get_mean_std(states)
#   mean_ref, std_ref = actor_{i-1}.get_mean_std(states)
#   w2 = ||μ_i - μ_ref||² + ||σ_i - σ_ref||²

return base_loss + w2_weight * w2_distance, w2_distance
```

---

### 4.5 Target network soft update

```python
# pogo_multi_main.py:424-426
for actor, actor_target in zip(actors, actor_targets):
    for p, tp in zip(actor.parameters(), actor_target.parameters()):
        tp.data.copy_(tau * p.data + (1 - tau) * tp.data)
```

---

## 5. Flow 요약 (그림)

```
[Batch: s, a, r, s', d]
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  trainer.train(batch)  —  IQL 원본                           │
│  1. V 업데이트: adv = Q_target(s,a) - V(s)                   │
│  2. Q 업데이트: MSE(Q(s,a), r + γV(s'))                      │
│  3. Actor0 업데이트: mean(exp(β·adv) · BC_loss)               │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  policy_freq마다 (IQL에서는 매 step)                         │
│  for i in [1, 2]:                                            │
│    base_loss = compute_actor_base_loss(actor_i, s, a)        │
│             = mean(exp(β·adv) · BC_loss)  ← Actor0와 동일     │
│    w2 = W2(π_i, π_{i-1})                                     │
│    loss = base_loss + w2_weight * w2                          │
│    actor_optimizers[i].step()                                 │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  actor_targets soft update (τ=0.005)                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. 핵심 코드 경로

| 단계 | 파일 | 함수/영역 |
|------|------|-----------|
| 진입점 | `pogo_multi_main.py` | `train()` |
| 배치 샘플링 | `pogo_multi_main.py` | `replay_buffer.sample()` |
| 통합 학습 | `pogo_multi_main.py` | `_train_multi_actor()` |
| V, Q, Actor0 | `iql.py` | `ImplicitQLearning.train()` |
| Actor1+ base loss | `iql.py` | `compute_actor_base_loss()` |
| W2 penalty | `pogo_multi_main.py` | `_compute_actor_loss_with_w2()`, `_compute_w2_distance()` |
| 평가 | `pogo_multi_main.py` | `act_for_eval(actor_i, state)` |
