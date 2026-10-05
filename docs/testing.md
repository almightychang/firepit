# 커널 테스트

`crates/firepit-kernels/tests/common`에 f64 CPU oracle, 입력 전략, 오차 판정을 둔다.

GEMV 검증은 `tests/gemv_dense.rs`와 `tests/gemv_q8.rs`에서 공개 `gemv(&Weight, ..)` 경로를 사용한다. dense PBT 가중치는 bf16로 반올림해 정확히 표현되는 f32 값으로 oracle을 계산한다. expert 변형은 `tests/moe.rs`에서 검증한다.

MoE prefill은 `tests/moe_prefill.rs`에서 host 안정 정렬·expert별 타일 패딩·역순열을 정확 비교하고, grouped gate/up·down GEMM 각각을 f64 내적 oracle과 비교한다. 각 GEMM의 성분별 허용 오차는 `(γ̃_K(8)+2⁻¹⁶)Σ|w_k x_k|`이며, 결합은 실제 routed 입력을 고정한 뒤 `γ̃_(k+2)(8)Σ|term|`로 판정한다. 토큰별 2.6 decode 경로와 전체 MoE 출력도 비교한다. 150토큰 HF prefill은 router, top-k, routed 합, shared 포함 Δx를 op별로 비교하고 expert 번호 순열·재정규화 누락 대조를 실행한다.

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

Q8 GEMV는 블록 합 뒤 scale 곱이 한 번 더 들어가므로 n = K + 2로 판정한다. FP8 GEMV도 scale을 타일 부분합 뒤에 곱하므로 n = K + 1로 판정한다. FP8 oracle PBT는 FP8 최대 유한값 ±448, 최소 subnormal, BF16 scale의 최소 양수·최대 유한값 및 큰 절댓값의 입력을 오버플로우 없는 조합으로 GEMV와 dense GEMM에서 검사한다. RMSNorm과 SwiGLU는 상쇄가 없는 원소별 연산이라 상대오차로 판정한다.

bf16 hi+lo GEMM은 잔차가 bf16 정상 범위인 유한 입력에서 `hi = RN_bf16(x)`, `lo = RN_bf16(x−hi)`를 사용한다. `|x−hi| ≤ u_b|x|`이고 Sterbenz 보조정리로 `x−hi`가 정확하므로 `|x−(hi+lo)| ≤ u_b²|x|` (`u_b=2⁻⁸`). 따라서 f64 oracle 판정은 `(γ̃_K(8)+u_b²)Σ|w_k x_k|`이다. GEMM 행과 GEMV의 차이는 두 경로의 상한 합으로 판정한다.

dense FP8 GEMM은 block의 N 범위가 scale 행 하나 안에 있어 K 단계마다 scale `s`가 스칼라다. 그래서 `x·(w·s) = (x·s)·w`로 계산한다. `x' = RN_f32(x·s)`를 hi+lo로 나누고, e4m3 가중치는 bf16으로 변환한다. e4m3의 가수 3비트와 지수 범위(2⁻⁹~448)가 bf16 정상 범위 안이라 이 변환은 정확하다. 곱 하나의 반올림이 더해지므로 n = K + 1이고, `x'`와 잔차가 f32·bf16 정상 범위인 입력에서 판정은 `(γ̃_(K+1)(8)+u_b²)Σ|w_k s x_k|`이다. grouped FP8 GEMM은 GB10에서 이 방식이 더 느려 `RN_bf16(w·s)`를 유지하고, 가중치마다 `u_b`가 더해진 `(γ̃_K(8)+u_b+u_b²)`로 판정한다.

NVFP4 가중치는 E2M1 값 `e`, 16원소마다 E4M3 블록 스케일 `s`, 행마다 f32 전역 스케일 `g`로 `w = e·s·g`다. `e·s`는 유효 비트가 2+4 ≤ 6이고 크기가 2⁻¹⁰~2688이라 f32와 bf16에서 모두 정확하다. f64 oracle은 `g·Σ(e·s)·x`다. GEMV는 그룹 부분합 뒤 블록 스케일 곱과 마지막 `g` 곱이 더해지므로 n = K + 2로 판정한다. GEMM은 정확한 bf16 가중치에 x의 hi+lo 분할을 쓰고 누적 뒤 `g`를 한 번 곱하므로 `(γ̃_(K+1)(8)+u_b²)Σ|w_k x_k|`로 판정한다. `tests/nvfp4_oracle.rs`는 dense GEMV·GEMM과 expert GEMV·grouped GEMM을 같은 입력으로 검사한다. 행렬은 `g`가 다른 두 부분을 쌓아 행별 스케일 위치도 확인한다. 극단값은 E2M1 ±6·±0.5·±0, 블록 스케일 448·2⁻⁹·0, 전역 스케일 2⁻¹⁰⁰·2¹⁰⁰을 결과가 유한한 조합으로 쓴다.

GDN chunk ①은 `A=I+L`, `X=A⁻¹`, `B_U=βv`, `B_W=βk exp(cum)`로 둔다. 오차 전파의 크기 기준은 각 성분의 `R=|X||B|`, `S=|X|(|L|+M_L)|Y|`이다. 여기서 `Y`는 정답 U 또는 W, `M_L[i,j]=β_i exp(cum_i−cum_j) Σ_d|k_id k_jd|` (`j<i`)이다. `M_L`은 L 생성 내적이 상쇄될 때도 입력 오차를 포착한다. 구현의 판정은 U에 `e_s R+(e_d+e_n)S+8u|U|`, W에 `(e_s+e_n)R+(e_d+e_n)S+8u|W|`를 쓴다. `e_d=3γ̃_D(8)+u_b²`, `e_m=3γ̃_C(8)+u_b²`, `e_n=γ̃_D(8)+γ̃_C(8)+16u`, `e_s=(2C/B+3)e_m`, `u=2⁻²⁴`, B는 대각 블록 크기다. `e_s`는 B 블록 안의 앞에서부터 대입과 C/B 블록 행의 세 번 분할 MMA 누적을 각 단계에 합한 보수적 계수다. 차이는 f32 최소 정상값을 분모 하한으로 둔다.

`intra`는 `QKᵀ` 내적 크기 `M_Q=exp(cum_i−cum_j)Σ_d|q_id k_jd|`에 `(e_d+e_n)M_Q`를 적용한다. 누적 gate의 오차가 지수로 전파되는 항은 `2γ̃_C(8)(Σ_{r≤i}|g_r|+Σ_{r≤j}|g_r|)M_Q`로 별도 더한다. `cum`은 `γ̃_C(8)|cum|+8u|cum|`로 판정한다. `tests/gdn_chunk.rs`의 f64 oracle이 성분별 크기를 계산하며, HF 기록 입력도 같은 기준을 쓴다.

GDN chunk ②는 3.4의 실제 f32 `U/W/intra/cum`을 고정 입력으로 삼고 f64로 `Δ=U−WS`, `o=Q̃S+intra·Δ`, `S'=exp(cum_last)S+K̃ᵀΔ`를 계산한다. 각 내적의 크기 `M=Σ|ab|`에 `e_n M`을 적용한다. bf16 세 곱의 `e_n=3γ̃_n(8)+3u_b²`는 두 피연산자의 hi+lo 분할 잔차와 생략한 lo·lo 항을 각각 포함한다. 정규화와 지수의 오차에는 `γ̃_D(8)+16u`, 출력 반올림에는 `8u|결과|`를 더한다. `Δ`의 상한은 `e_D M_WS+|W|E_S+8u|Δ|`, 출력의 상한은 `e_D M_QS+e_C M_AΔ+|Q̃|E_S+|intra|E_Δ+8u|o|`, 다음 state의 상한은 `exp(cum_last)E_S+8u exp(cum_last)|S|+e_C M_KΔ+|K̃|ᵀE_Δ+8u|S'|`다. 이 성분별 상한을 chunk마다 전파한다. `u=2⁻²⁴`, `u_b=2⁻⁸`이다. decode 동치에는 이 상한과 기존 delta_step의 경험값 `3e-4(1+|기준값|)`을 합한다. decode 쪽 엄밀한 상한은 아직 미해결이다.

Prefill attention은 Q·Kᵀ와 P·V 모두 f32 피연산자를 bf16 hi+lo로 나눠 hi·hi + hi·lo + lo·hi를 누적한다. `S_j = scale·Σ_i q_i k_{ji}`, `M_j = scale·Σ_i |q_i k_{ji}|`, `E_0 = max_j(γ̃_D M_j + 4u)`, `E_s = max_j((3γ̃_D+u_b²)M_j)`, `E = E_0+E_s`로 둔다. 출력 성분의 판정 상한은 `L·f32::MIN_POSITIVE·max|v| + [γ̃_(2L+2) + (e^(2E)−1) + 3γ̃_L + u_b² + 8u]·Σ_j a_j|v_j|`이다. `a_j`는 f64 softmax 가중치이며, `L`은 해당 쿼리가 볼 수 있는 cache 길이다. P·V 분할의 생략된 lo·lo와 세 MMA 누적 오차를 마지막 두 항에 더했다. Prefill과 decode의 행별 동치 판정은 각 경로 상한의 합을 쓴다.

## 층

GEMV는 oracle 비교와 선형성·배치 독립·가중치 행 치환을 검사한다. RMSNorm은 oracle 비교와 배치 독립·scale 성질을 검사한다. SwiGLU는 f64 oracle PBT와 SiLU 구간·극단값·NaN sweep로 확인한다. RoPE는 위치 0~262143의 f64 oracle PBT, 크기·상대 위치 보존, 위치 0 항등을 검사한다. Attention은 길이 1~256의 f64 oracle PBT, 순서·상수 값·점수 평행이동 불변, gate 극단값과 HF 5 step을 검사한다. Prefill attention은 T·p0·head/group·값 분포의 f64 oracle PBT, 같은 cache를 읽는 decode와 행별 동치, HF 150토큰 op 비교, causal mask·modulo GQA 대조를 검사한다. MoE는 top-k·expert GEMV·결합의 f64 oracle PBT, expert 번호 치환·하위 점수 변경·가중치 합 성질, HF 4개 지점 × 5 step × 4 layer를 검사한다. GDN conv는 채널 수와 커널 길이를 바꾸는 f64 oracle PBT, 토큰별 처리와 전체 causal conv 비교, 마지막 입력 state 확인, HF layer 0~2 × 5 step 비교, tap 순서 반전과 state 미갱신 대조를 검사한다. GDN delta rule은 다중 step f64 oracle PBT, β=0 감쇠, 직교 key 저장·회수, value 열 타일 불변, gate 극단 sweep, HF core·state·norm·Δx 비교, l2norm·head 매핑·norm weight 규약 대조를 검사한다. 모델 규약은 ignored HF reference 테스트로, 테스트 민감도는 gate/up 순서 교환, RoPE의 인접 짝짓기·전체 차원 회전, attention의 scale 누락·modulo GQA, MoE의 재정규화 누락, prefill의 미래 가리기·head 매핑, top-(k−1) 대조 실험으로 확인한다.

기본 검증: `scripts/dev.sh test --workspace`. 실제 모델 정답 비교: `scripts/dev.sh test -p firepit --test reference -- --ignored`.

### 3.7 엔진 전체 모델 검증

`tools/reference/trace_qwen36.py --engine`는 기존 prefill과 같은 150개 token ID로 HF 전체 층의 마지막 토큰 logits와 각 층 출력 residual의 마지막 행을 기록한다. GB10에서 `scripts/dev.sh test -p firepit --test engine_verify -- --ignored --nocapture`로 검증한다. 모델과 정답지 경로는 각각 `FIREPIT_VERIFY_MODEL`, `FIREPIT_VERIFY_REF`로 바꾸고, 작은 접두부 확인에는 `FIREPIT_VERIFY_LAYERS=2`를 사용한다.

- HF 비교: 마지막 토큰 logits의 top-1과 top-5 집합이 같아야 한다. logits의 최대 절대·상대 오차와 상대 L2, 층별 마지막 residual 행의 같은 세 수치는 기록한다.
- 경로 동치: `prefill(T)`와 `step_eager` T회, `prefill(A)` 뒤 `prefill(B)`와 `prefill(A+B)`의 마지막 logits top-1·top-5 집합이 같아야 한다. 두 경로의 logits 및 모든 층의 GDN conv·recurrent 또는 attention K·V state 차이를 기록한다. 현재 고정 입력은 T=150, A=100, B=50이다.
- 작은 random 모델 대조는 이어 붙이기에서 RoPE 위치를 0으로 재설정하거나 prefill lm_head가 첫 행을 읽으면 차이가 생기는지 검사한다.

GB10 결과(2026-10-04, [기록](research/measurements/engine-verify-gb10.md)): 세 비교 모두 통과. 두 경로의 상태 차이는 첫 MoE 라우팅 flip(근소 margin에서 top-k 집합이 갈림) 전까지 1e-5 수준이고, flip layer에서 계단식으로 뛴다. 따라서 첫 flip 뒤의 tensor 비교는 커널 정확도의 근거가 아니다. 커널 정확도를 전체 stack에서 보려면 라우팅을 고정한 비교가, 모델 품질은 많은 토큰에 대한 출력 분포 비교(KL, perplexity, top-k 일치율)가 맞다.

전체 layer stack의 수학적으로 엄밀한 오차 상한은 미해결이다. 따라서 위 top-k 일치만 통과 조건이고, 기록하는 오차 수치에는 근거 없는 허용치를 두지 않는다.

### 출력 품질 평가 (1차)

`tools/reference/quality_ref.py`는 [llama.cpp의 WikiText-2 test 출처](https://github.com/ggml-org/llama.cpp/blob/master/scripts/get-wikitext-2.sh)를 `--cache`에 저장하고 Qwen tokenizer로 전체 원문을 토큰화한다. 완전한 4096토큰 묶음마다 새 문맥을 만들고, 기본적으로 행 2048~4095의 다음 토큰 분포를 채점한다. 행 `i`의 logits는 토큰 `i`를 입력한 뒤 토큰 `i+1`을 예측한다. 마지막 행은 다음 묶음의 실제 토큰 유무와 관계없이 분포만 평가한다. 각 묶음의 HF bf16 모델 분포에서 상위 256개 ID·log 확률과 나머지 log 질량을 f32/I32 safetensors로 별도 저장한다. 완성된 묶음 파일은 검증 후 건너뛰므로 중단 후 같은 명령을 재실행할 수 있다.

```bash
python3 tools/reference/quality_ref.py --model "$HOME/Models/qwen3.6-35b-a3b" --cache /tmp/wiki.test.raw --out refs/quality-qwen36
scripts/dev.sh run --release -p firepit --example quality -- --model "$HOME/Models/qwen3.6-35b-a3b" --ref refs/quality-qwen36 --chunks 2 --mode prefill
scripts/dev.sh run --release -p firepit --example quality -- --model "$HOME/Models/qwen3.6-35b-a3b" --ref refs/quality-qwen36 --chunks 2 --mode decode
```

HF 생성은 CUDA PyTorch와 `transformers==5.17.0`, `safetensors==0.8.0`이 필요하다. `--chunks 1` 또는 `2`와 작은 `--chunk-size`, `--score-start`, `--k`, `--layers 2 --device cpu`로 접두부를 시험할 수 있다. Rust 예제에도 같은 `--layers 2`를 준다. 실제 품질 수치는 반드시 release 빌드로 잰다. `prefill`은 채점 범위 전체를 한 번에 계산하고, `decode`는 앞 절반을 prefill한 뒤 나머지를 teacher forcing으로 한 step씩 입력한다. 두 모드 모두 HF 기준의 같은 행을 읽는다. KL은 HF 상위 K 항과 나머지 질량을 합친 거친 분포의 KL이며, 실제 어휘 전체 KL의 하한이다. 출력은 묶음별·전체·채점 구간별 평균 KL과 top-1 일치율, KL 분위수·최댓값, 나머지 질량 최댓값, 묶음 평균 Student t 95% 신뢰구간이다. 묶음이 하나면 신뢰구간은 표시하지 않는다.

원인 진단에는 예를 들어 `--chunks 72 --only-chunks 3,17 --top-kl 10`을 덧붙인다. 선택한 묶음만 채점하고 KL 상위 10개 행의 상위 K 항·나머지 항, 양쪽 top-5 확률과 교차 top-1 확률을 출력한다. 묶음 마지막 행의 다음 토큰은 기준 파일에 없어 `n/a`로 표시한다.

정답 토큰 NLL/PPL 비교는 기준 파일에 `target_logp`를 추가한 뒤 같은 quality 예제를 실행한다. 기존 기준 파일에는 다음 명령으로 `target_logp`만 계산해 원자적으로 덧붙인다. 파일마다 메타데이터와 토큰·텐서 형상을 확인하며, 이미 추가된 파일은 건너뛴다.

```bash
python3 tools/reference/quality_ref.py --model "$HOME/Models/qwen3.6-35b-a3b" --cache /tmp/wiki.test.raw --out refs/quality-qwen36 --add-target-logp
scripts/dev.sh run --release -p firepit --example quality -- --model "$HOME/Models/qwen3.6-35b-a3b" --ref refs/quality-qwen36 --chunks 72 --mode prefill
```

행 `i`의 정답은 같은 묶음의 토큰 `i+1`이다. 마지막 행은 다음 토큰이 기준 파일에 없으므로 NLL/PPL에서 제외하고 기존 KL에는 포함한다. `target_logp`가 없는 기준 파일로 실행하면 firepit NLL/PPL만 표시한다. 기준이 있으면 HF PPL, firepit−HF 평균 ΔNLL과 묶음별 짝지은 t 95% 신뢰구간, ln(PPL 비), 채점 구간별 ΔNLL, 정답 확률 차의 분위수도 표시한다.

## 판정 기준 현황

| 커널 | 판정 | 근거 |
|---|---|---|
| GEMV(dense, Q8), expert GEMV, MoE 결합 | Higham–Mary 내적 상한 | 정리 3.1 |
| GDN conv | 내적 상한 × max|SiLU'| + 활성화 반올림 | 정리 3.1 + 1차 전파 |
| attention | q·k 점수 오차 E → softmax 가중치 (e^{2E}−1)·a, 가중합 상한, f32 underflow 하한 | 정리 3.1 + 1차 전파 |
| prefill attention | attention 상한에 두 bf16×3 내적의 분할·누적 오차를 더함 | 정리 3.1 + 1차 전파 |
| rope | 2항 내적 상한 | 정리 3.1 |
| rms_norm, swiglu, top-k 가중치 | 상대·절대 오차 | 상쇄가 없는 연산(제곱합, 원소별 곱, [0,1] 확률)이라 출력 대비 오차가 원리적으로 맞다 |
| GDN chunk ② | bf16×3 내적의 성분별 상한을 chunk마다 state로 전파 | Higham–Mary + 분할 잔차 세 항 |
| GDN delta rule | 상대 3e-4 (경험값) | **미해결.** 엄밀한 상한은 커널의 실제 연산 순서(FMA, l2norm 방식, softplus 급수, 하드웨어 exp 근사)를 따라가는 오차 모델이 필요하다. 연산 모델 없이 세운 추정 상한은 1.01~1.05배로 넘었다. 5000건 깊은 실행에서 경험값 판정은 통과 |

상쇄 때문에 출력 대비 상대오차가 거짓 실패를 낸 사례: SwiGLU 결합, MoE 결합, GDN conv, attention(정답이 f32 표현 범위 밖, 1e-61). 모두 커널 오류가 아니었다.
