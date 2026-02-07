# PORL_unified 통합 보고서

## 통합 방법 요약
- `PORL_sv`를 기준 베이스로 복사한 뒤, `PORL_sh`에만 존재하던 파일을 추가해 `PORL_unified`를 구성했다.
- 공통 파일에서 충돌하는 구현은 `PORL_sv` 버전을 우선 사용했다.

## 1) README가 전체 구조를 이해하기 쉽게 작성되었는지
- 기존 README는 실행 방법, 알고리즘 개요, policy 인터페이스, 디렉터리 구조를 포함하고 있어 **입문 가독성은 높은 편**이다.
- 다만 통합 저장소 관점에서는 다음 보완이 필요했다.
  - 저장소 이름/경로 표기 일관화 (`PORL` → `PORL_unified`)
  - `sh`/`sv` 계열에서 온 구성요소를 한 문서에서 설명
  - 테스트/실행 명령의 기준 작업 디렉터리 명시
- 위 보완 사항을 README에 반영했다.

## 2) critic / actor0 학습이 원래 알고리즘과 동치인지
- 코드 상 설계 의도는 “Critic은 원본 로직 유지, Actor0만 원본 actor loss 유지, Actor1+에 W2 penalty 추가”로 일관된다.
- IQL/AWAC/TD3+BC/SAC-N/EDAC/CQL 경로에서 Actor0는 분기 처리로 별도 loss를 사용하고, W2는 Actor1+에만 적용된다.
- 따라서 **구조적 동치성은 대체로 확보**되어 있으며, 실제 수치 동치는 시드/환경별 회귀 실험으로 최종 검증이 필요하다.

## 3) 버그 가능성
- 상대 import/실행 경로 의존성이 있어, 루트가 아닌 위치에서 테스트 실행 시 `ModuleNotFoundError`가 발생할 수 있다.
- 일부 알고리즘은 정책 타입별 fallback 분기가 많아, 새 policy 타입 추가 시 `log_prob_actions`/`get_mean_std` 인터페이스 누락으로 런타임 에러가 날 수 있다.

## 4) 로깅 품질
- 학습 loop에서 step 단위 train log, eval log, json summary 저장, wandb 전송이 모두 구현되어 있어 **기본 로깅은 충분**하다.
- 다만 예외 발생 시점에 대한 보호 로깅(예: 배치 shape 오류, NaN 탐지)과 per-actor gradient norm 로깅은 부족하다.

## 5) 성능 저하 가능 코드
- multi-actor 루프에서 actor별 샘플링/W2 계산을 반복하므로 actor 수 증가 시 비용이 선형 이상 증가한다.
- Sinkhorn 기반 W2는 `K` 샘플링과 OT 계산이 병목이 될 수 있어, 환경/배치에 따라 큰 오버헤드가 발생할 수 있다.
- eval 시 actor별로 별도 rollout을 수행하므로 actor 수가 많을수록 eval wall-clock이 증가한다.
