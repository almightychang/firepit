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

## 품질 손실 분해 (`abd177f`, `--override GROUP=DIR`, 72묶음 prefill)

| 변형 | 4비트 | FP8 | bf16 | KL 평균 | top-1 | ΔNLL (95% CI) | decode |
|---|---|---|---|---:|---:|---|---:|
| V1 NVIDIA 그대로 | expert·shared·lm_head | 투영(텐서 전체 스케일) | 나머지 | 3.61e-2 | 91.05% | +2.04e-2 ± 2.3e-3 | 20.8 ms (48.1 tok/s) |
| V2 lm_head bf16 | expert·shared | 투영(텐서 전체) | lm_head 포함 | 2.53e-2 | 93.33% | +1.08e-2 ± 2.2e-3 | 22.9 ms (43.7 tok/s) |
| V3 lm_head·투영 bf16 | expert·shared | 없음 | 나머지 전부 | 2.35e-2 | 93.62% | +1.14e-2 ± 2.3e-3 | — |
| V4 lm_head bf16, 투영 Qwen FP8 | expert·shared | 투영(128 블록) | lm_head 포함 | 2.55e-2 | 93.34% | +1.30e-2 ± 2.4e-3 | — |

- lm_head 4비트가 ΔNLL의 약 절반(+2.04e-2 → +1.08e-2)이다. decode에서는 lm_head가 약 2.1 ms를 아낀다.
- FP8 투영은 측정 가능한 손실을 더하지 않는다(V2 ≈ V3, CI 겹침). 텐서 전체 스케일과 128 블록 스케일의 차이도 CI 안이다(V2 vs V4).
- 남는 약 +1.1e-2(PPL +1.1%)는 expert·shared NVFP4에서 온다. Qwen FP8 전체(+3.3e-3)의 약 3배다.
