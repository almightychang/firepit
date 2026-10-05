# E4 양자화 설계

상태: 4.1 FP8 Sam 승인 2026-10-05. 사전 확인: [e4-prerequisites](research/e4-prerequisites.md). 품질 판정 도구: [design-quality](design-quality.md).

## 4.1 FP8 (공식 체크포인트, 가중치만)

- 체크포인트: `Qwen/Qwen3.6-35B-A3B-FP8`. E4M3 가중치 `[N, K]`와 128×128 블록 스케일 `weight_scale_inv` BF16 `[N/128, K/128]`. 변환되지 않은 텐서(lm_head, embedding, norm, router, shared expert gate, GDN in_proj_a/b·conv1d·A_log·dt_bias)는 BF16.
- `Weight::Fp8(Fp8Matrix)`를 추가한다(데이터 e4m3, 블록 스케일). GEMV/GEMM은 기존처럼 `Weight` 한 곳에서 dispatch한다.
- 로딩: config의 `quantization_config`를 parse 타입으로 읽고, 텐서마다 `weight_scale_inv` 유무로 Fp8/Dense를 정한다. config와 텐서 구성이 맞지 않으면 로드 시 에러. expert는 expert별 FP8 텐서를 기존 `[E·2I, H]`·`[E·H, I]` 배치로 쌓는다(행 수가 128의 배수라 스케일 블록이 그대로 맞는다).
- 커널: decode GEMV(dense, expert)는 FP8 타일을 f32로 바꿔 블록 스케일을 곱해 기존과 같이 누적한다. prefill GEMM(dense, grouped)은 FP8 타일을 bf16으로 풀어 기존 tensor core 경로를 쓴다. activation은 f32 그대로(가중치만 FP8).
- 검증: 커널은 dequantize한 가중치를 기준으로 기존 f64 판정. 로딩은 Python dequantize 값과 비트 비교. 품질은 기존 HF bf16 기준 72묶음으로 KL·ΔPPL·top-1을 잡음 바닥과 비교. 속도는 GB10 decode tok/s와 5000토큰 prefill.
- 나중: activation FP8 + `mmaf_scaled`(prefill 계산 이득, 정밀도 변화는 따로 측정), Q8_0·NVFP4 비교, grouped GEMM에 gather view 결합 검토.
