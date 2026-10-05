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

## `firepit pull qwen3.6` 조립본 확인 (`a95dd0a`, GB10)

- `firepit pull qwen3.6`: NVIDIA NVFP4 전체 + bf16 원본의 `lm_head.weight`만 범위 요청(1,017,118,720 B)으로 받아 조립, 23 GB. 두 번째 pull은 건너뜀.
- 72묶음 품질은 `--override lm_head=` V2와 **같은 값**: KL 2.534196e-2, top-1 93.33%, ΔNLL +1.076081e-2 ± 2.24e-3. 조립본과 실행 시 조합이 같은 가중치를 쓴다.
- decode batch 1 (max-seq 256): 23.20 ms (43.1 tok/s), V2 측정 22.90 ms와 같은 수준.
- `firepit run` 첫 실행은 JIT 포함이라 prefill 5토큰 24.7 s, decode 4.0 tok/s로 보인다(측정값과 다름).

## 기본 구성 decode 분해 (nsys 2025.3.2, `--cuda-graph-trace=node`, `f710059`, 24 step 평균)

`qwen3.6:nvfp4` 조립본, batch 1, max-seq 256. step 합 22.93 ms.

| 커널 | step당 ms | step당 호출 | 읽는 가중치 (계산) | 유효 GB/s |
|---|---:|---:|---:|---:|
| dense_fp8 (GDN·attention 투영) | 9.65 | 130 | 1.29 GB | 134 |
| expert_nvfp4 (routed gate+up, down) | 5.08 | 80 | 0.57 GB | 111 |
| dense_gemv bf16 (lm_head, router, in_proj_a/b, shared gate) | 4.84 | 141 | 1.07 GB | 221 |
| dense_nvfp4 (shared expert) | 1.96 | 80 | 0.07 GB | 36 |
| 나머지 (delta, norm, combine, top-k, conv 등) | 1.40 | | | |

토큰당 가중치 약 3.0 GB, bf16 GEMV가 내는 약 213~221 GB/s 기준 하한은 약 14 ms. 차이 약 8.8 ms 중 FP8 투영 약 3.6 ms, routed NVFP4 약 2.4 ms, shared NVFP4 약 1.6 ms(작은 행렬이라 대역폭이 아니라 지연에 묶인 것으로 추정).

## shared expert를 항상 선택되는 expert로 (`a045314` → `41bf08d`, GB10 같은 pod, design-e4 4.3)

- 테스트: kernels moe, moe_prefill, fp8_oracle, nvfp4_oracle, fp8_tile, gemm, lib; engine; nvfp4_prefix·reference moe_matches_hf·moe_prefill_matches_hf(ignored) 통과. `loads_prefix_four`는 GB10 `nvidia-smi`가 memory.used를 `[N/A]`로 내서 helper가 실패했다(모델 로드 전, 테스트 helper 문제로 수정).
- decode batch 1: `qwen3.6:nvfp4` 18.48 → **16.72 ms (54.1 → 59.8 tok/s)**, FP8 18.72 → 18.55 ms, bf16 27.77 → 27.42 ms.
- `bench --profile`(eager, nvfp4): gemv 350 → 270회, 13.49 → 10.63 ms. moe_expert_gemv 4.86 → 5.11 ms. swiglu 80 → 40회, 0.85 → 0.32 ms.
- prefill 5000토큰(1회): 4123 → 4258 ms. moe_grouped_gemm +119 ms, gemm +37 ms(호출 −80회). 변경과 무관한 gdn_chunk_second가 −46 ms 움직여 잡음과 구분하지 못했다. 반복 측정 필요. 가설(미확인): shared 칸이 expert 하나에 T행을 몰아 grouped GEMM 부하가 한 그룹에 집중된다.
- 품질(새 대 기준, 같은 묶음, 모두 CI 안): prefill 72묶음 FP8 ΔNLL 3.304e-3 대 3.537e-3(±1.2e-3), decode 8묶음 NVFP4 1.1448e-2 대 1.1426e-2(±5.9e-3). prefill 72묶음 NVFP4는 두 커밋 출력이 자리수까지 같다(이유 미확인: dense GEMM과 grouped GEMM의 합 순서가 같을 가능성).
