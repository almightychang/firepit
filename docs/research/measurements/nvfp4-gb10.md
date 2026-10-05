# E4.2 NVIDIA NVFP4 혼합 체크포인트 GB10 측정 (`88c3465`, 2026-10-05)

`nvidia/Qwen3.6-35B-A3B-NVFP4`: routed·shared expert와 lm_head는 NVFP4(가중치 전용), GDN·attention 투영은 FP8(텐서 전체 스케일), 나머지 bf16. activation f32, KV cache f32.

| 지표 | bf16 | FP8 (Qwen) | NVFP4 (NVIDIA 혼합) |
|---|---:|---:|---:|
| 가중치 크기 | 67 GB | 35 GB | 23.4 GB |
| decode batch 1 (max-seq 256) | 27.7 ms (36.1 tok/s) | 24.3 ms (41.2 tok/s) | **20.8 ms (48.1 tok/s)** |
| prefill 5000 | 4.14~4.19 s | 3.96 s | 3.89 s |
| KL 평균 | 4.8e-3 (바닥) | 9.3e-3 | **3.61e-2** |
| top-1 일치 | 97.1% | 95.9% | 91.05% |
| ΔNLL (95% CI) | 0 | +3.3e-3 ± 1.1e-3 | **+2.04e-2 ± 2.3e-3** |
| PPL (HF bf16 5.744) | 5.744 | 5.763 | 5.862 |

- 테스트: nvfp4_oracle(극단값 포함), ModelOpt dequant 비트 비교, 2층 엔진 prefill/decode, 기존 FP8·GEMM·MoE 테스트 통과.
- 1차 실행에서 NVFP4 GEMV가 JIT 타입 오류로 실패했다(cuTile 결함 의심, issues/cutile-rs/007). reduce 결과를 reshape하는 우회로 해결.
- ModelOpt의 E2M1 표는 코드 0b1000을 +0으로 둔다.
- 품질 손실이 FP8의 약 6배다. lm_head까지 4비트인 구성의 영향이 섞여 있으며 기여를 분리하지 않았다.
