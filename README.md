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

### Multi-Actor 구조

- **Actor0**: 각 알고리즘의 원래 actor loss만 사용 (W2 penalty 없음)
- **Actor1+**: 각 알고리즘의 actor loss + W2 distance to previous actor
  - Loss: `base_loss + w_i · W₂(π_i, π_{i-1})`

### 지원 알고리즘

**PyTorch 버전**: IQL, TD3_BC, CQL, AWAC, SAC-N, EDAC  
**JAX 버전**: ReBRAC, FQL

## 프로젝트 구조

```
PORL_sh/
├── algorithms/
│   └── offline/
│       ├── pogo_multi_main.py      # PyTorch 버전 메인 스크립트
│       ├── pogo_multi_jax.py        # JAX 버전 메인 스크립트
│       ├── pogo_policies_jax.py     # JAX Policy 구현
│       ├── networks.py              # PyTorch Policy 구현
│       ├── utils_pytorch.py         # PyTorch 유틸리티 (AlgorithmInterface)
│       ├── utils_jax.py             # JAX 유틸리티 (AlgorithmInterface)
│       ├── iql.py, cql.py, ...      # 각 알고리즘 구현
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
        │  │  - compute_actor_loss()          │  │
        └──┼──────────────────────────────────┼──┘
           │                                  │
    ┌──────▼──────┐                  ┌────────▼────────┐
    │  IQL/CQL/   │                  │  Actor0:        │
    │  TD3_BC/... │                  │  base_loss      │
    │  (원래 구조) │                  │                 │
    └─────────────┘                  │  Actor1+:       │
                                      │  base_loss +    │
                                      │  w2_weight * W2 │
                                      └─────────────────┘
```

**핵심 설계 원칙**:
1. **Critic 업데이트**: 각 알고리즘의 원래 방식 그대로 (Actor0만 사용)
2. **Actor 업데이트**: Multi-actor 구조로 확장 (W2 regularization 추가)
3. **AlgorithmInterface**: 각 알고리즘을 통일된 인터페이스로 추상화

## 상세 문서

더 자세한 내용은 다음 문서를 참고하세요:

- **[`algorithms/offline/POGO_MULTI_README.md`](algorithms/offline/POGO_MULTI_README.md)**: 
  - 이론적 배경 (JKO Chain, Gradient Flow)
  - 상세한 사용법 및 예시
  - 알고리즘별 비교표 및 설명
  - Config 파라미터 설명
  - Policy 타입별 설명
  - 코드 구조 및 구현 세부사항

- **[`algorithms/offline/CODE_EVALUATION_REPORT.md`](algorithms/offline/CODE_EVALUATION_REPORT.md)**: 
  - 코드 평가 보고서
  - Critic/Actor0 동치성 검증
  - 버그 가능성 분석
  - 로깅 및 성능 평가

## 참고

- CORL: [Consistent Offline Reinforcement Learning](https://github.com/corl-team/CORL)
- POGO: Policy Optimization of Gradient flow for Offline Reinforcement Learning
- D4RL: [Datasets for Deep Data-Driven Reinforcement Learning](https://github.com/Farama-Foundation/D4RL)
