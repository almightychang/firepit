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
- 3.6 기본 구성: NVIDIA NVFP4 혼합 + lm_head bf16. 근거는 [분해 측정](research/measurements/nvfp4-gb10.md): lm_head 4비트가 손실의 약 절반, FP8 투영은 측정 가능한 손실 없음. 다른 체크포인트(RedHat·AWQ·Qwen FP8·unsloth)도 lm_head를 8비트 이상으로 둔다.
- ds4는 E6에서 같은 GB10의 성능·품질 비교 상대로 쓴다. unsloth의 뒤쪽 layer expert FP8 유지는 필요하면 실험 옵션으로 둔다.
