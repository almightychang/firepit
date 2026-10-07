# E4.2 NVIDIA NVFP4 혼합 체크포인트 GB10 측정 (`88c3465`, 2026-10-05)

`nvidia/Qwen3.6-35B-A3B-NVFP4`: routed·shared expert와 lm_head는 NVFP4(가중치 전용), GDN·attention projection은 FP8(텐서 전체 스케일), 나머지 bf16. activation f32, KV cache f32.

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
| V1 NVIDIA 그대로 | expert·shared·lm_head | projection(텐서 전체 스케일) | 나머지 | 3.61e-2 | 91.05% | +2.04e-2 ± 2.3e-3 | 20.8 ms (48.1 tok/s) |
| V2 lm_head bf16 | expert·shared | projection(텐서 전체) | lm_head 포함 | 2.53e-2 | 93.33% | +1.08e-2 ± 2.2e-3 | 22.9 ms (43.7 tok/s) |
| V3 lm_head·projection bf16 | expert·shared | 없음 | 나머지 전부 | 2.35e-2 | 93.62% | +1.14e-2 ± 2.3e-3 | — |
| V4 lm_head bf16, projection Qwen FP8 | expert·shared | projection(128 블록) | lm_head 포함 | 2.55e-2 | 93.34% | +1.30e-2 ± 2.4e-3 | — |

- lm_head 4비트가 ΔNLL의 약 절반(+2.04e-2 → +1.08e-2)이다. decode에서는 lm_head가 약 2.1 ms를 아낀다.
- FP8 projection은 측정 가능한 손실을 더하지 않는다(V2 ≈ V3, CI 겹침). 텐서 전체 스케일과 128 블록 스케일의 차이도 CI 안이다(V2 vs V4).
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
| dense_fp8 (GDN·attention projection) | 9.65 | 130 | 1.29 GB | 134 |
| expert_nvfp4 (routed gate+up, down) | 5.08 | 80 | 0.57 GB | 111 |
| dense_gemv bf16 (lm_head, router, in_proj_a/b, shared gate) | 4.84 | 141 | 1.07 GB | 221 |
| dense_nvfp4 (shared expert) | 1.96 | 80 | 0.07 GB | 36 |
| 나머지 (delta, norm, combine, top-k, conv 등) | 1.40 | | | |

토큰당 가중치 약 3.0 GB, bf16 GEMV가 내는 약 213~221 GB/s 기준 하한은 약 14 ms. 차이 약 8.8 ms 중 FP8 projection 약 3.6 ms, routed NVFP4 약 2.4 ms, shared NVFP4 약 1.6 ms(작은 행렬이라 대역폭이 아니라 지연에 묶인 것으로 추정).

## shared expert를 항상 선택되는 expert로 (`a045314` → `41bf08d`, GB10 같은 pod, design-e4 4.3)

- 테스트: kernels moe, moe_prefill, fp8_oracle, nvfp4_oracle, fp8_tile, gemm, lib; engine; nvfp4_prefix·reference moe_matches_hf·moe_prefill_matches_hf(ignored) 통과. `loads_prefix_four`는 GB10 `nvidia-smi`가 memory.used를 `[N/A]`로 내서 helper가 실패했다(모델 로드 전, 테스트 helper 문제로 수정).
- decode batch 1: `qwen3.6:nvfp4` 18.48 → **16.72 ms (54.1 → 59.8 tok/s)**, FP8 18.72 → 18.55 ms, bf16 27.77 → 27.42 ms.
- `bench --profile`(eager, nvfp4): gemv 350 → 270회, 13.49 → 10.63 ms. moe_expert_gemv 4.86 → 5.11 ms. swiglu 80 → 40회, 0.85 → 0.32 ms.
- prefill 5000토큰(1회): 4123 → 4258 ms. moe_grouped_gemm +119 ms, gemm +37 ms(호출 −80회). 변경과 무관한 gdn_chunk_second가 −46 ms 움직여 잡음과 구분하지 못했다. 반복 측정 필요. 가설(미확인): shared 칸이 expert 하나에 T행을 몰아 grouped GEMM 부하가 한 그룹에 집중된다.
- 품질(새 대 기준, 같은 묶음, 모두 CI 안): prefill 72묶음 FP8 ΔNLL 3.304e-3 대 3.537e-3(±1.2e-3), decode 8묶음 NVFP4 1.1448e-2 대 1.1426e-2(±5.9e-3). prefill 72묶음 NVFP4는 두 커밋 출력이 자리수까지 같다(이유 미확인: dense GEMM과 grouped GEMM의 합 순서가 같을 가능성).

### shared expert 통합의 prefill 영향 (6회 교대 측정)

- wall 중앙값(ms): T=1000 636 → 663, T=2000 1285 → 1324, T=5000 4102 → 4158(profile 없음). 처음 1회의 +135 ms는 잡음이었고 실제는 +1~4%다.
- "한 그룹에 부하 집중"은 아니다(nsys): grouped GEMM 행 tile이 ceil(T/32)만큼 늘었고 tile당 시간은 같다(1.56 → 1.54 µs). shared 몫이 dense NVFP4 GEMM(BM 64)에서 grouped GEMM(BM 32)으로 옮겨 가중치 tile을 두 배 읽는다(T=1000 13.0 → 22.7 ms, T=5000 103.0 → 112.3 ms, gather +2.8 / +16 ms). T=1000 차이 중 약 12.5 ms만 이것으로 설명되고 나머지는 확인하지 않았다.

## decode 분해 (`1d923cd`, nsys 2025.3.2 `--cuda-graph-trace=node`, 32 step 평균)

`qwen3.6:nvfp4` 조립본, batch 1, max-seq 256. nsys 없는 step 16.71 ms, 커널 합 16.49 ms.

| 커널 | step당 ms | step당 호출 | 읽는 가중치 (계산) | 유효 GB/s |
|---|---:|---:|---:|---:|
| dense_fp8 (GDN·attention projection) | 6.67 | 130 | 1.29 GB | 193 |
| dense_gemv bf16 (lm_head, router, in_proj_a/b, shared gate) | 4.87 | 141 | 1.07 GB | 220 |
| expert_nvfp4 (routed 8 + shared 1) | 3.69 | 80 | 0.64 GB | 174 |
| 원소별·세션 op 합 | 1.39 | | | |

원소별·세션 op: delta 0.39(30회), rms_norm 0.26(131), combine 0.17(40), top_k 0.14(40), conv1d_silu 0.11(30), add 0.10(80), attention 0.07(10), swiglu 0.05(40), 나머지 0.1 미만.

하한(가중치 3.0 GB / 220 GB/s ≈ 13.6 ms) 대비 약 3.1 ms 중 GEMV 초과 약 1.6 ms(FP8 +0.81, NVFP4 expert +0.78), 원소별·세션 op 1.4 ms, 커널 사이 공백 약 0.2 ms(추정).

## residual add + rms_norm fusion (`fdd5b5d` → `f399905`, GB10 같은 pod)

design-e4 4.3의 5번. 테스트 전부 통과, decode logits(nvfp4·FP8 33 step)와 품질 로그가 이전과 비트 동일(f32 연산 순서를 같게 유지).
- nsys(nvfp4): combine 166.8 + add 95.5 + norm 164.1 µs/step → combine_add_norm 235.8 + add_norm 87.1 µs/step, 원소별 커널 −104 µs/step. norm launch 131 → 51회(prefill).
- decode 3회 교대 중앙값: nvfp4 15.70 → 15.64 ms, FP8 −0.04, bf16 −0.08 ms(범위 겹침). 같은 실행에서 코드가 그대로인 lm_head가 3893 → 4040 µs로 느려져 커널 절감을 상쇄했다. 버퍼 배치에 따른 대역폭 차이로 추정(미확인). 사전 할당 여부로 lm_head가 236~259 GB/s로 달라진 관찰(gemv-probe-gb10.md)과 같은 종류다.
- prefill 5000: 차이 없음.

## firepit run JIT 캐시 (`f399905` → `318c2cf`, GB10, 같은 새 FIREPIT_HOME)

design-cli JIT 절. 테스트 통과, logits 비트 동일.

| 회차 | 프롬프트 / KV 구간 | 기준: tileiras, wall | 수정 후: tileiras / disk, wall |
|---|---|---|---|
| 1 | 8토큰 / 64 | 64~66, 45~48 s | 64 / 0, 45.2 s |
| 2 | 8토큰 / 64 | 64~66, 45~48 s | 0 / 64, **24.5 s** |
| 3 | 20토큰 / 64 | | 11 / 55, 28.3 s (prefill 커널만) |
| 4 | 37토큰 / 128 | | 9, 27.4 s (decode는 cache_append 하나) |

- stage1(cuTile 앞단) 4.3~4.8 s는 디스크 캐시가 있어도 매번 남는다(cutile-rs 이슈 초안 015).
- 출력 텐서의 동적 extent로 MAX_SEQ 재컴파일을 없애려던 시도는 launch 검사에서 거부돼 되돌렸다(이슈 초안 014, GB10 재현). KV 길이를 2의 거듭제곱 구간으로 올린다.

## hot path: prefill 정렬 counting sort(A1)와 decode 업로드 정리(D1) (`0cddab6` → `706ce45`, GB10)

측정 근거와 설계: scratchpad/hot-path-measure-report.md, hot-path-design.md. 테스트 전부 통과(engine_verify 8 포함), decode logits 비트 동일.
- decode graph 밖 구간(nsys, 45 step 중앙값): 227 → 151 µs/step. sync 9 → 3, HtoD 4 → 2, DtoD 5 → 3. 남은 몫은 graph launch 33 µs, logits DtoH 31 µs, host 처리. decode bench 15.65 → 15.59 ms.
- prefill 5000: host 정렬(로컬 CPU T=5000) 1010 → 166 µs. GPU 공백 59.5 → 32.3 ms, 그중 MoE 정렬·업로드 공백 44.9 → 18.4 ms. sync·복사 횟수는 그대로라 남은 공백은 왕복 자체다. wall 차이는 2회 측정 잡음 안(약 −11 ms).
- graph 안 업로드·logits 복사(D2)는 safe API가 없어 보류(cutile-rs 이슈 초안 016, 017). GPU 쪽 정렬(B)은 남은 18 ms를 보고 판단.

## prefill 활성값 저정밀 실험 (`mmaf_scaled`, 실험 브랜치 `scana-exp-actq` 692ce89, 제품 미반영)

`mmaf_scaled`는 두 피연산자가 같은 저정밀 형식이어야 해서 활성값도 양자화해야 한다. prefill GEMM 입력만 MXFP8(OCP MX v1.0) 또는 NVFP4(ModelOpt 규칙)로 흉내 양자화해 품질을 쟀다(prefill 72묶음, nvfp4 조립본).

| prefill 활성값 | ΔNLL | KL | top-1 | GEMM (T=5000, 호출당) |
|---|---:|---:|---:|---|
| 지금(가중치만) | 1.0925e-2 | 2.535e-2 | 93.35% | 기준 |
| FP8 projection GEMM만 MXFP8 | 1.5775e-2 | 2.992e-2 | 92.84% | `[qkv|z]` 18.38 → 2.54~3.20 ms, 양자화 상한 0.59 ms 포함 −14.7 ms |
| NVFP4 expert GEMM만 NVFP4 | 2.3508e-2 | 4.365e-2 | 90.95% | gate_up 15.92 → 5.90 ms, 양자화 상한 7.44 ms라 −2.6 ms. BK 128은 131.6 ms(tileiras 이상 후보) |
| 둘 다 | 2.6575e-2 | 4.730e-2 | 90.72% | |

결정(Sam, 2026-10-07): 채택하지 않는다. E4는 가중치만 양자화를 유지하고, FP8 activations(prefill 약 −25% 추정, ΔNLL +4.9e-3)는 E5에서 긴 prompt의 prefill 지연이 문제가 될 때 다시 판단한다.
