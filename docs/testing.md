# 커널 테스트

`crates/firepit-kernels/tests/common`에 f64 CPU oracle, 입력 전략, 오차 판정을 둔다.

## PBT

- 커널마다 기본 64건. `PROPTEST_CASES=256 scripts/dev.sh test ...`로 늘린다(컨테이너에 전달된다).
- MoE expert GEMV는 기본 16건에서 공통 입력과 행별 입력을 모두 검사한다. 기본 shape는 작은 타일 경계와 배수를 쓰고, `PROPTEST_CASES=256`에서는 expert 8개·64행·512열과 router 256개를 추가한다.
- proptest는 shape와 seed만 생성하고, 값은 seed에서 `prop_perturb`로 채운다. 분포는 원소마다 근사 Gaussian, 0, ±8 균등, ±32를 섞는다.
- GEMV의 런타임 N·K는 16~1024행, 128~2048열에서 생성한다(실제 K=2048, expert N=512 포함). RMSNorm 폭은 32/128/2048, SwiGLU 폭은 32/128/512이다. const generic(JIT 특수화)은 고정 집합에서만 고른다.
- 실패 시 축소된 반례는 테스트 파일 옆의 `.proptest-regressions`에 저장되고, 다음 실행에서 먼저 재생된다.
- f64 oracle이 debug 빌드에서 느려서 `[profile.test] opt-level = 2`를 쓴다(debug assertion은 유지).

## 판정 기준

내적(GEMV)은 출력값 대비 상대오차가 아니라 더한 항의 크기 대비로 판정한다. 출력이 상쇄로 작아진 원소에서 상대오차가 부풀려지는 거짓 실패가 없고, 기준이 반올림 이론에서 나온다.

```
|ŷ − y| ≤ γ̃_n(λ) · Σ|w_k x_k|,   γ̃_n(λ) = exp(λ√n·u + n·u²/(1−u)) − 1,   u = 2⁻²⁴, λ = 8
```

N. J. Higham and T. Mary, "A New Approach to Probabilistic Rounding Error Analysis", SIAM J. Sci. Comput. 41, A2815–A2835, 2019, doi:10.1137/18M1226312 (MIMS EPrint 2018.33). 정리 3.1, 식 (3.6). 반올림 오차가 평균 0인 독립 확률변수라는 가정(Model 2.1) 아래 덧셈 순서와 무관하게 확률 `1 − n(1 − P(λ))`, `P(λ) = 1 − 2exp(−λ²(1−u)²/2)` 이상으로 성립한다. λ = 8이면 실패 확률이 무시할 수준이고, 실측 오차(약 2u 이하)보다 50배 이상 여유가 있다(K=128에서 90u, K=2048에서 362u). 인덱스·순서 버그는 이 기준의 수만 배 오차를 낸다.

Q8 GEMV는 블록 합 뒤 scale 곱이 한 번 더 들어가므로 n = K + 2로 판정한다. RMSNorm과 SwiGLU는 상쇄가 없는 원소별 연산이라 상대오차로 판정한다.

## 층

GEMV는 oracle 비교와 선형성·배치 독립·가중치 행 치환을 검사한다. RMSNorm은 oracle 비교와 배치 독립·scale 성질을 검사한다. SwiGLU는 f64 oracle PBT와 SiLU 구간·극단값·NaN sweep로 확인한다. RoPE는 위치 0~262143의 f64 oracle PBT, 크기·상대 위치 보존, 위치 0 항등을 검사한다. Attention은 길이 1~256의 f64 oracle PBT, 순서·상수 값·점수 평행이동 불변, gate 극단값과 HF 5 step을 검사한다. MoE는 top-k·expert GEMV·결합의 f64 oracle PBT, expert 번호 치환·하위 점수 변경·가중치 합 성질, HF 4개 지점 × 5 step × 4 layer를 검사한다. 모델 규약은 ignored HF reference 테스트로, 테스트 민감도는 gate/up 순서 교환, RoPE의 인접 짝짓기·전체 차원 회전, attention의 scale 누락·modulo GQA, MoE의 재정규화 누락·top-(k−1) 대조 실험으로 확인한다.

기본 검증: `scripts/dev.sh test --workspace`. 실제 모델 정답 비교: `scripts/dev.sh test -p firepit --test reference -- --ignored`.
