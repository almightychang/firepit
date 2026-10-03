# 커널 테스트

`crates/firepit-kernels/tests/common`에 f64 CPU oracle, 입력 전략, 오차 판정을 둔다.

GEMV 검증은 `tests/gemv_dense.rs`와 `tests/gemv_q8.rs`에서 공개 `gemv(&Weight, ..)` 경로를 사용한다. dense PBT 가중치는 bf16로 반올림해 정확히 표현되는 f32 값으로 oracle을 계산한다. expert 변형은 `tests/moe.rs`에서 검증한다.

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

bf16 hi+lo GEMM은 잔차가 bf16 정상 범위인 유한 입력에서 `hi = RN_bf16(x)`, `lo = RN_bf16(x−hi)`를 사용한다. `|x−hi| ≤ u_b|x|`이고 Sterbenz 보조정리로 `x−hi`가 정확하므로 `|x−(hi+lo)| ≤ u_b²|x|` (`u_b=2⁻⁸`). 따라서 f64 oracle 판정은 `(γ̃_K(8)+u_b²)Σ|w_k x_k|`이다. GEMM 행과 GEMV의 차이는 두 경로의 상한 합으로 판정한다.

GDN chunk ①은 `A=I+L`, `X=A⁻¹`, `B_U=βv`, `B_W=βk exp(cum)`로 둔다. 오차 전파의 크기 기준은 각 성분의 `R=|X||B|`, `S=|X|(|L|+M_L)|Y|`이다. 여기서 `Y`는 정답 U 또는 W, `M_L[i,j]=β_i exp(cum_i−cum_j) Σ_d|k_id k_jd|` (`j<i`)이다. `M_L`은 L 생성 내적이 상쇄될 때도 입력 오차를 포착한다. 구현의 판정은 U에 `e_s R+(e_d+e_n)S+8u|U|`, W에 `(e_s+e_n)R+(e_d+e_n)S+8u|W|`를 쓴다. `e_d=3γ̃_D(8)+u_b²`, `e_m=3γ̃_C(8)+u_b²`, `e_n=γ̃_D(8)+γ̃_C(8)+16u`, `e_s=(2C/B+3)e_m`, `u=2⁻²⁴`, B는 대각 블록 크기다. `e_s`는 B 블록 안의 앞에서부터 대입과 C/B 블록 행의 세 번 분할 MMA 누적을 각 단계에 합한 보수적 계수다. 차이는 f32 최소 정상값을 분모 하한으로 둔다.

`intra`는 `QKᵀ` 내적 크기 `M_Q=exp(cum_i−cum_j)Σ_d|q_id k_jd|`에 `(e_d+e_n)M_Q`를 적용한다. 누적 gate의 오차가 지수로 전파되는 항은 `2γ̃_C(8)(Σ_{r≤i}|g_r|+Σ_{r≤j}|g_r|)M_Q`로 별도 더한다. `cum`은 `γ̃_C(8)|cum|+8u|cum|`로 판정한다. `tests/gdn_chunk.rs`의 f64 oracle이 성분별 크기를 계산하며, HF 기록 입력도 같은 기준을 쓴다.

Prefill attention은 Q·Kᵀ와 P·V 모두 f32 피연산자를 bf16 hi+lo로 나눠 hi·hi + hi·lo + lo·hi를 누적한다. `S_j = scale·Σ_i q_i k_{ji}`, `M_j = scale·Σ_i |q_i k_{ji}|`, `E_0 = max_j(γ̃_D M_j + 4u)`, `E_s = max_j((3γ̃_D+u_b²)M_j)`, `E = E_0+E_s`로 둔다. 출력 성분의 판정 상한은 `L·f32::MIN_POSITIVE·max|v| + [γ̃_(2L+2) + (e^(2E)−1) + 3γ̃_L + u_b² + 8u]·Σ_j a_j|v_j|`이다. `a_j`는 f64 softmax 가중치이며, `L`은 해당 쿼리가 볼 수 있는 cache 길이다. P·V 분할의 생략된 lo·lo와 세 MMA 누적 오차를 마지막 두 항에 더했다. Prefill과 decode의 행별 동치 판정은 각 경로 상한의 합을 쓴다.

## 층

GEMV는 oracle 비교와 선형성·배치 독립·가중치 행 치환을 검사한다. RMSNorm은 oracle 비교와 배치 독립·scale 성질을 검사한다. SwiGLU는 f64 oracle PBT와 SiLU 구간·극단값·NaN sweep로 확인한다. RoPE는 위치 0~262143의 f64 oracle PBT, 크기·상대 위치 보존, 위치 0 항등을 검사한다. Attention은 길이 1~256의 f64 oracle PBT, 순서·상수 값·점수 평행이동 불변, gate 극단값과 HF 5 step을 검사한다. Prefill attention은 T·p0·head/group·값 분포의 f64 oracle PBT, 같은 cache를 읽는 decode와 행별 동치, HF 150토큰 op 비교, causal mask·modulo GQA 대조를 검사한다. MoE는 top-k·expert GEMV·결합의 f64 oracle PBT, expert 번호 치환·하위 점수 변경·가중치 합 성질, HF 4개 지점 × 5 step × 4 layer를 검사한다. GDN conv는 채널 수와 커널 길이를 바꾸는 f64 oracle PBT, 토큰별 처리와 전체 causal conv 비교, 마지막 입력 state 확인, HF layer 0~2 × 5 step 비교, tap 순서 반전과 state 미갱신 대조를 검사한다. GDN delta rule은 다중 step f64 oracle PBT, β=0 감쇠, 직교 key 저장·회수, value 열 타일 불변, gate 극단 sweep, HF core·state·norm·Δx 비교, l2norm·head 매핑·norm weight 규약 대조를 검사한다. 모델 규약은 ignored HF reference 테스트로, 테스트 민감도는 gate/up 순서 교환, RoPE의 인접 짝짓기·전체 차원 회전, attention의 scale 누락·modulo GQA, MoE의 재정규화 누락, prefill의 미래 가리기·head 매핑, top-(k−1) 대조 실험으로 확인한다.

기본 검증: `scripts/dev.sh test --workspace`. 실제 모델 정답 비교: `scripts/dev.sh test -p firepit --test reference -- --ignored`.

## 판정 기준 현황

| 커널 | 판정 | 근거 |
|---|---|---|
| GEMV(dense, Q8), expert GEMV, MoE 결합 | Higham–Mary 내적 상한 | 정리 3.1 |
| GDN conv | 내적 상한 × max|SiLU'| + 활성화 반올림 | 정리 3.1 + 1차 전파 |
| attention | q·k 점수 오차 E → softmax 가중치 (e^{2E}−1)·a, 가중합 상한, f32 underflow 하한 | 정리 3.1 + 1차 전파 |
| prefill attention | attention 상한에 두 bf16×3 내적의 분할·누적 오차를 더함 | 정리 3.1 + 1차 전파 |
| rope | 2항 내적 상한 | 정리 3.1 |
| rms_norm, swiglu, top-k 가중치 | 상대·절대 오차 | 상쇄가 없는 연산(제곱합, 원소별 곱, [0,1] 확률)이라 출력 대비 오차가 원리적으로 맞다 |
| GDN delta rule | 상대 3e-4 (경험값) | **미해결.** 엄밀한 상한은 커널의 실제 연산 순서(FMA, l2norm 방식, softplus 급수, 하드웨어 exp 근사)를 따라가는 오차 모델이 필요하다. 연산 모델 없이 세운 추정 상한은 1.01~1.05배로 넘었다. 5000건 깊은 실행에서 경험값 판정은 통과 |

상쇄 때문에 출력 대비 상대오차가 거짓 실패를 낸 사례: SwiGLU 결합, MoE 결합, GDN conv, attention(정답이 f32 표현 범위 밖, 1e-61). 모두 커널 오류가 아니었다.
