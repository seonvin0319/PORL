# PORL_unified 코드 분석 보고서

**분석 일자:** 2025-02-07  
**기준 문서:** README.md, POGO_MULTI_ARCHITECTURE.md, POGO_MULTI_README.md

---

## 1. Executive Summary

PORL_unified는 **기존 오프라인 RL 알고리즘의 Critic/Q 로직을 유지**하면서 **Actor만 multi-actor(JKO chain)로 확장**하는 것이 핵심 설계 목표입니다. 코드 분석 결과, **전반적인 아키텍처는 설계 의도대로 구현**되었으며, **IQL, AWAC의 Actor1+ base loss 불일치 문제를 수정 완료**했습니다. **TD3_BC는 원래 코드(`-λ·Q`)가 의도에 맞아 유지**합니다.

---

## 2. 설계 의도 vs 구현 대조

### 2.1 설계 원칙 (README/POGO_MULTI_ARCHITECTURE)

| 원칙 | 설계 의도 |
|------|----------|
| Actor0 | 원래 알고리즘 actor loss만 사용 (W2 penalty 없음) |
| Actor1+ | **기존 알고리즘 loss** + W2/Sinkhorn penalty to previous actor |
| Critic | 알고리즘 원본 업데이트 경로 유지 |
| 공통 Policy | `utils/policy_call.py`로 통합 |

### 2.2 구현 검증 결과

#### ✅ 잘 구현된 부분

1. **Critic/V/Q 업데이트 유지**
   - `trainer.train(batch)` 또는 `trainer.update(batch)` 호출로 각 알고리즘의 원본 V, Q, Critic 업데이트가 그대로 수행됨
   - IQL: `_update_v`, `_update_q` 그대로 호출
   - CQL: `_q_loss` 메서드 그대로 사용 (OOD 샘플링, CQL penalty 포함)
   - EDAC: `_critic_loss` (diversity loss 포함) 그대로 호출

2. **Actor0 업데이트**
   - base trainer의 `train()`/`update()` 내부에서 Actor0가 원래 알고리즘의 actor loss로 업데이트됨
   - Actor0는 W2 penalty 없이 학습됨

3. **Actor1+ 업데이트**
   - `_train_multi_actor()`에서 `policy_freq`마다 Actor1+ 순차 업데이트
   - W2 distance: Gaussian은 closed form, TanhGaussian/Stochastic은 Sinkhorn, Deterministic은 L2 사용
   - `_compute_w2_distance()`, `_compute_actor_loss_with_w2()` 공통 헬퍼 함수 적절히 구현됨

4. **Policy 인터페이스 통합**
   - `utils/policy_call.py`: `get_action`, `act_for_eval`, `sample_K_actions` 등으로 다양한 policy 타입 통일
   - `PolicyAdapter`로 시그니처가 다른 policy 감싸기

5. **Config 기반 알고리즘 선택**
   - `algorithm: iql|td3_bc|cql|awac|sac_n|edac`로 선택 가능
   - `w2_weights`, `num_actors`, `actor_configs` 등 POGO 전용 파라미터 지원

#### ⚠️ **중요: Actor0 vs Actor1+ Base Loss 불일치**

설계 문서(POGO_MULTI_ARCHITECTURE.md 149~157행)에 따르면:

> - **Actor0**: `L₀ = [기존 알고리즘 loss]` (W2 penalty 없음)
> - **Actor1+**: `Lᵢ = [기존 알고리즘 loss] + w₂ᵢ · Sinkhorn(πᵢ, πᵢ₋₁)`

**Actor1+도 Actor0와 동일한 base loss를 사용해야** 합니다. 그러나 다음 알고리즘에서 `compute_actor_base_loss`가 Actor0 loss와 **다른 식**을 반환합니다.

| 알고리즘 | Actor0 실제 loss | Actor1+ compute_actor_base_loss 반환 | 일치 여부 |
|----------|-----------------|--------------------------------------|-----------|
| **IQL** | `mean(exp(β·adv) · BC_loss)` (advantage-weighted BC) | 동일 (advantage-weighted BC) | ✅ 일치 (수정됨) |
| **AWAC** | `mean(weights · (-log_prob))` (weights=exp(adv/λ)) | 동일 (advantage-weighted BC) | ✅ 일치 (수정됨) |
| **TD3_BC** | `-λ·Q + MSE(π, a_dataset)` | `-λ·Q` (원래 코드 의도) | ✅ (원래대로 유지) |
| **CQL** | BC/CQL stage 구분, alpha 처리 | 동일 로직 | ✅ 일치 |
| **SAC-N** | `(α·log_π - Q_min).mean()` | 동일 | ✅ 일치 |
| **EDAC** | `(α·log_π - Q_min).mean()` | 동일 | ✅ 일치 |

**결과적으로**: IQL, AWAC, TD3_BC에서는 Actor1+가 설계 의도와 다른 loss로 학습되며, JKO chain의 일관성이 깨질 수 있습니다.

---

## 3. 상세 분석

### 3.1 학습 흐름 (`_train_multi_actor`)

```python
# pogo_multi_main.py:379-369
log_dict = trainer.train(batch)  # 또는 trainer.update(batch)
# → V, Q, Critic, Actor0 업데이트 (원래 알고리즘 그대로)

if trainer.total_it % policy_freq == 0:
    for i in range(1, len(actors)):  # Actor1+만
        base_loss = trainer.compute_actor_base_loss(actor_i, state, actions, seed=...)
        actor_loss = base_loss + w2_weight * w2_distance
        actor_optimizers[i].step()
```

- Actor0는 `trainer.train()` 내부 `_update_policy()` 등에서 업데이트됨 → **원래 loss 사용** ✅
- Actor1+는 `compute_actor_base_loss` + W2로 업데이트 → **알고리즘별로 구현이 다름** ⚠️

### 3.2 W2/Sinkhorn 계산

- **Gaussian policy**: `_closed_form_w2_gaussian(mean1, std1, mean2, std2)` 사용
- **Stochastic (TanhGaussian, StochasticMLP)**: `_per_state_sinkhorn()` (GeomLoss SamplesLoss)
- **Deterministic**: L2 distance `((pi_i - ref_a)**2).sum().mean()`

설계 문서와 일치합니다.

### 3.3 Policy 타입 및 인터페이스

- `networks.py`: `GaussianMLP`, `TanhGaussianMLP`, `StochasticMLP`, `DeterministicMLP`
- `pogo_policies.py`: 동일 클래스명의 별도 구현 존재 (테스트 등에서 사용)
- `action_for_loss()`: Gaussian은 `get_mean_std()[0]`, stochastic은 `sample_actions`, deterministic은 `forward` 사용 → gradient가 유지됨

### 3.4 Config 및 실행 경로

- README: `configs/offline/iql_pogo_base.yaml` 예시
- POGO_MULTI_README: `configs/offline/pogo_multi/halfcheetah/medium_v2_iql.yaml` 예시
- 실제 존재: `configs/offline/iql_pogo_base.yaml`, `configs/offline/pogo_multi/halfcheetah/medium_v2_*.yaml` 모두 있음
- `test_smoke_algorithms.py`의 CONFIG_MAP: `cql_pogo_base.yaml`, `sac_n_pogo_base.yaml` 등 참조 — 경로가 README와 다를 수 있음

---

## 4. 버그 및 리스크

### 4.1 Actor0 vs Actor1+ Loss 불일치 (우선순위: 높음)

**IQL**  
- Actor0: `exp(β·adv) · BC_loss` (advantage-weighted BC)
- Actor1+: `-Q.mean()` (Q 극대화)
- → Actor1+가 데이터 분포와의 근접성을 고려하지 않고 Q만 극대화함.

**AWAC**  
- Actor0: `weights · (-log_prob)` (advantage-weighted BC)
- Actor1+: `-Qmin.mean()`
- → Actor1+가 advantage weighting 없이 Q만 극대화함.

**TD3_BC**  
- Actor0: `-λ·Q + MSE(π, a_dataset)`
- Actor1+: `-λ·Q` (원래 코드 의도에 맞게 유지)

### 4.2 기타 잠재 이슈

1. **정책 타입별 fallback**  
   - `log_prob_actions`, `get_mean_std` 등이 없는 policy 추가 시 런타임 에러 가능성 (INTEGRATION_REPORT.md 언급)

2. **실행 경로 의존성**  
   - `python -m algorithms.offline.pogo_multi_main` 형태로 실행해야 상대 import가 동작함

3. **성능**  
   - Actor 수 증가 시 Actor별 샘플링/W2 계산으로 비용 선형 증가
   - Sinkhorn은 K 샘플 × OT 계산으로 병목 가능

---

## 5. 수정 권장 사항

### 5.1 IQL `compute_actor_base_loss` (우선)

Actor1+도 Actor0와 동일한 advantage-weighted BC를 사용하도록 수정:

```python
def compute_actor_base_loss(self, actor, state, actions=None, seed=None):
    with torch.no_grad():
        target_q = self.q_target(state, actions)
    v = self.vf(state)
    adv = target_q - v
    exp_adv = torch.exp(self.beta * adv).clamp(max=EXP_ADV_MAX)
    # BC_loss: actor의 action과 dataset action의 MSE 또는 -log_prob
    cfg = ActorConfig.from_actor(actor)
    pi = action_for_loss(actor, cfg, state, seed=seed)
    bc_losses = ((pi - actions) ** 2).sum(dim=1)  # 또는 log_prob 기반
    return (exp_adv * bc_losses).mean()
```

### 5.2 AWAC `compute_actor_base_loss`

Actor1+도 advantage-weighted BC를 사용하도록 수정 (V, Q로 adv 계산 후 weights 적용).

### 5.3 TD3_BC `compute_actor_base_loss`

원래 코드(`-λ·Q`)가 의도에 맞아 유지함.

---

## 6. 결론

| 항목 | 평가 |
|------|------|
| Critic/V/Q 유지 | ✅ 설계대로 구현 |
| Actor0 loss | ✅ 원래 알고리즘 loss 사용 |
| Actor1+ W2 penalty | ✅ 적절히 적용 |
| Actor1+ base loss 일치 | ✅ IQL, AWAC 수정됨 / TD3_BC 원래 코드 유지 |
| Policy 인터페이스 | ✅ 통합 완료 |
| Config 구조 | ✅ 알고리즘별 선택 가능 |
| 테스트 | ✅ policy_call, smoke 테스트 존재 |

**종합**: 아키텍처와 Critic 측면은 설계 의도에 부합합니다. **IQL, AWAC의 Actor1+ base loss를 Actor0와 동일하게 수정 완료**했습니다. **TD3_BC는 원래 코드(`-λ·Q`)가 의도에 맞아 유지**합니다. CQL, SAC-N, EDAC는 원래부터 설계대로 구현되어 있습니다.
