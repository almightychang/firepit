# E4 양자화 설계

상태: 4.1 FP8 Sam 승인 2026-10-05. 사전 확인: [e4-prerequisites](research/e4-prerequisites.md). 품질 판정 도구: [design-quality](design-quality.md).

## 4.1 FP8 (공식 체크포인트, 가중치만)

- 체크포인트: `Qwen/Qwen3.6-35B-A3B-FP8`. E4M3 가중치 `[N, K]`와 128×128 블록 스케일 `weight_scale_inv` BF16 `[N/128, K/128]`. 변환되지 않은 텐서(lm_head, embedding, norm, router, shared expert gate, GDN in_proj_a/b·conv1d·A_log·dt_bias)는 BF16.
- `Weight::Fp8(Fp8Matrix)`를 추가한다(데이터 e4m3, 블록 스케일). GEMV/GEMM은 기존처럼 `Weight` 한 곳에서 dispatch한다.
- 로딩: config의 `quantization_config`를 parse 타입으로 읽고, 텐서마다 `weight_scale_inv` 유무로 Fp8/Dense를 정한다. config와 텐서 구성이 맞지 않으면 로드 시 에러. expert는 expert별 FP8 텐서를 기존 `[E·2I, H]`·`[E·H, I]` 배치로 쌓는다(행 수가 128의 배수라 스케일 블록이 그대로 맞는다).
- 커널: decode GEMV(dense, expert)는 FP8 타일을 f32로 바꿔 블록 스케일을 곱해 기존과 같이 누적한다. prefill GEMM(dense, grouped)은 FP8 타일을 bf16으로 풀어 기존 tensor core 경로를 쓴다. activation은 f32 그대로(가중치만 FP8).
- 검증: 커널은 dequantize한 가중치를 기준으로 기존 f64 판정. 로딩은 Python dequantize 값과 비트 비교. 품질은 기존 HF bf16 기준 72묶음으로 KL·ΔPPL·top-1을 잡음 바닥과 비교. 속도는 GB10 decode tok/s와 5000토큰 prefill.
- 나중: activation FP8 + `mmaf_scaled`(prefill 계산 이득, 정밀도 변화는 따로 측정), Q8_0·NVFP4 비교, grouped GEMM에 gather view 결합 검토.

## 4.2 NVIDIA NVFP4 혼합 체크포인트 (Sam 승인 2026-10-05)

- 체크포인트: `nvidia/Qwen3.6-35B-A3B-NVFP4` (ModelOpt 0.44, `hf_quant_config.json` `quant_algo: MIXED_PRECISION`, 23.4 GB). 구성: routed expert·shared expert·**lm_head**는 `W4A16_NVFP4`(group 16), GDN `in_proj_qkv`·`in_proj_z`·`out_proj`와 attention q/k/v/o는 `FP8`, router·norm·conv·in_proj_a/b·embedding은 BF16. `kv_cache_quant_algo: FP8`.
- NVFP4 텐서: `weight` U8 `[N, K/2]`(E2M1 두 개를 한 바이트에), `weight_scale` F8_E4M3 `[N, K/16]`, `weight_scale_2` F32 스칼라, `input_scale` F32 스칼라. dequant `w = e2m1 × weight_scale × weight_scale_2`(니블 순서와 식은 구현 첫 단계에서 ModelOpt/HF 코드로 확인).
- FP8 텐서: `weight` E4M3, `weight_scale` F32 스칼라(텐서 전체), `input_scale` F32 스칼라. 기존 FP8 경로의 스케일 블록을 "행렬 전체"로 일반화해 받는다.
- `input_scale`은 쓰지 않는다(activation f32 유지, 가중치 전용). KV cache FP8은 이번 범위에서 제외.
- `Weight::Nvfp4`를 추가한다. decode GEMV(dense·expert·lm_head)는 니블을 풀어 f32로 바꾸고 16개 블록 부분합에 블록 스케일, 마지막에 전역 스케일. prefill GEMM(dense·grouped)은 bf16으로 풀어 기존 mma. `mmaf_scaled` 하드웨어 경로는 나중.
- 로딩: `hf_quant_config.json`을 parse 타입으로 읽고 텐서 구성과 대조, 불일치 시 에러.
- 검증: dequant 가중치 기준 f64 판정, Python dequant 비트 비교, 품질 72묶음(FP8 +0.33%와 비교), GB10 decode·prefill.

## 결정: NVIDIA 방향 (Sam 2026-10-05)

- 형식 기준은 NVIDIA ModelOpt 체크포인트다. E6 모델에도 `nvidia/Qwen3.8-Flash-Next-NVFP4`가 있고(expert만 NVFP4/FP8 블록, lm_head 등은 bf16), Blackwell의 NVFP4/FP8 경로와 맞고, 로더가 이미 이 형식을 읽는다.
- 3.6 기본 구성: NVIDIA NVFP4 혼합 + lm_head bf16. 근거는 [분해 측정](research/measurements/nvfp4-gb10.md): lm_head 4비트가 손실의 약 절반, FP8 projection은 측정 가능한 손실 없음. 다른 체크포인트(RedHat·AWQ·Qwen FP8·unsloth)도 lm_head를 8비트 이상으로 둔다.
- ds4는 E6에서 같은 GB10의 성능·품질 비교 상대로 쓴다. unsloth의 뒤쪽 layer expert FP8 유지는 필요하면 실험 옵션으로 둔다.

## 4.3 decode GEMV 효율 (Sam 승인 2026-10-05)

1. 양자화 GEMV는 가중치를 `[행, K]` view로 읽는다(`0a29c40`). 근거와 결과는 `research/measurements/gemv-probe-gb10.md`.
2. shared expert는 항상 선택되는 마지막 expert다. 로드할 때 shared gate_up·down을 routed expert 묶음 끝(index E)에 이어 붙이고, 선택 목록은 토큰마다 k개 + shared 1칸(가중치 `sigmoid(shared_expert_gate)`)이다. decode와 prefill 모두 expert 경로 하나로 계산하고, shared 전용 op·slot·GEMM 경로와 combine의 shared 입력은 없앤다. shared와 routed의 intermediate 크기나 가중치 형식이 다르면 로드에서 실패한다. design-e2.9의 MoE op 목록이 이렇게 바뀌고, "커널 하나 = op 하나"는 그대로다.
3. FP8 scale 표현은 `Blocks(block_n × block_k)` 또는 `Rows(행마다 1개)`다. per-tensor 체크포인트(NVIDIA ModelOpt FP8)는 `Rows`(모두 같은 값)로 로드하고, GEMV는 K 한 루프로 합한 뒤 끝에서 행 scale을 곱한다. prefill GEMM은 출력 열마다 scale을 곱하는 epilogue로 처리한다. Qwen 128 블록 형식은 지금 커널 그대로다. 근거: `research/measurements/gemv-probe-gb10.md` 잔여분 분석, 9단계 prototype(step당 −0.28 ms).
4. 같은 입력을 받는 같은 형식의 GEMV는 로드할 때 행 방향으로 이어 붙인다: GDN `[qkv|z]`(FP8)와 `[b|a]`(bf16), attention `[q_gate|k|v]`(FP8), MoE `[router|shared_gate]`(bf16). 출력은 slot 하나에 쓰고 소비하는 op는 열 범위 slice view로 읽는다(복사 없음). 묶음 행 수는 묶음에서 가장 큰 행렬에 기존 타일 규칙이 고르는 타일 행 수의 배수로 0행 padding하고, padding 행의 출력은 쓰지 않는다. prefill도 같은 가중치로 묶음당 GEMM 1회를 돈다. 단 padding한 묶음은 prefill에서 padding 전 멤버별로 GEMM한다(가중치 행 slice와 출력 열 slice view, 복사 없음). padding은 decode GEMV 타일 규칙 때문에 생기고 prefill GEMM 타일을 작게 만들기 때문이다(router 묶음 층당 3263 → 781 µs). 형식이 다른 GEMV는 묶지 않는다. 근거: 9단계 prototype(A 포함 step당 −0.86 ms).
5. residual add 뒤의 rms_norm은 한 커널로 계산하고 출력 두 개(새 residual, normed)를 쓴다(`add_norm`). MoE 뒤에서는 combine까지 같은 커널에 넣는다(`combine_add_norm`). 다음 norm의 gamma는 그 자리의 norm(post-attention norm, 다음 층 input norm, 마지막 층이면 final norm)을 쓴다. decode와 prefill이 같은 op를 쓴다. 근거: `scratchpad/fusion-proto-report.md` prototype(combine+add+norm 층당 −192 µs/step, add+norm −46 µs/step, 결과 차이 상대 5.3e-7 이하). delta·norm·gate 묶기(타일 규칙 예외 필요)와 expert gate_up·swiglu(행 타일 64에서 tileiras 이상)는 하지 않는다.

## 결정: prefill 활성값 저정밀 보류 (Sam 2026-10-07)

prefill에 `mmaf_scaled`를 쓰려면 활성값도 FP8·FP4로 양자화해야 한다. 실험 결과(research/measurements/nvfp4-gb10.md 끝 절) NVFP4 활성값은 품질 손실이 크고 이득이 작으며, FP8 활성값은 projection GEMM이 약 6배 빨라지지만 ΔNLL +4.9e-3이고 prefill과 decode의 수치가 달라진다. E4는 가중치만 양자화를 유지하고 E5에서 다시 판단한다.
