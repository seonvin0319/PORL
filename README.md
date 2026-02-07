# PORL_unified — POGO Multi-Actor Offline RL

`PORL_sh` + `PORL_sv` 개선 사항을 통합한 unified 버전입니다.  
핵심 목표는 **기존 알고리즘의 Critic/Q 로직을 유지**하면서, **Actor만 multi-actor(JKO chain)로 확장**하는 것입니다.

---

## 빠른 시작

```bash
cd PORL_unified

# 단위 테스트
pytest tests/test_policy_call.py -q
pytest tests/test_smoke_algorithms.py -q

# 학습 실행 예시 (IQL)
python -m algorithms.offline.pogo_multi_main \
  --config_path configs/offline/iql_pogo_base.yaml \
  --env halfcheetah-medium-v2
```

---

## 저장소 구조

```text
PORL_unified/
├── algorithms/offline/
│   ├── pogo_multi_main.py       # PyTorch multi-actor 메인
│   ├── pogo_multi_jax.py        # JAX(ReBRAC) 메인
│   ├── pogo_policies.py         # 통합 policy 구현
│   ├── pogo_policies_jax.py     # JAX policy 구현
│   ├── iql.py, td3_bc.py, cql.py, awac.py, sac_n.py, edac.py
│   ├── networks.py              # (sh 계열) 공통 네트워크
│   ├── utils_pytorch.py         # (sh 계열) 공통 유틸
│   └── utils_jax.py             # (sh 계열) JAX 유틸
├── utils/
│   └── policy_call.py           # policy 공통 호출 인터페이스
├── configs/
│   ├── offline/*
│   └── finetune/*
├── tests/
│   ├── test_policy_call.py
│   └── test_smoke_algorithms.py
└── INTEGRATION_REPORT.md        # 통합 분석 보고서
```

---

## 설계 원칙

- **Actor0**: 원래 알고리즘 actor loss 사용 (동치 유지)
- **Actor1+**: 기존 loss + W2/Sinkhorn penalty
- **Critic**: 알고리즘 원본 업데이트 경로 유지
- **공통 policy 인터페이스**: `utils/policy_call.py`로 통합

