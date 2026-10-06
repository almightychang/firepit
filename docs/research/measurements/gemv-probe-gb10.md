# 양자화 GEMV 병목 원인 분리 GB10 측정 (`0b33638`, 2026-10-05)

`crates/firepit-kernels/examples/fp8_gemv_probe.rs`. 가중치를 읽기만 하는 변형으로 load 경로, 변환, CTA 수, 행렬 크기를 하나씩 바꿨다. do_bench 중앙값, L2 비움, 제품 타일(`fp8_tile`, NB=4). GB/s는 가중치 바이트 기준(NVFP4는 packed만).

## 결론

- **원인은 가중치 view의 모양이다.** 제품 커널은 가중치를 `[행, K/32, 32]`(NVFP4 `[행, K/32, 16]`)로 본다. TMA box의 가장 안쪽 줄이 FP8 32 B, NVFP4 16 B, bf16 64 B가 되고, 정상 상태 대역폭이 이 순서로 묶인다(NVFP4 약 90, FP8 약 145, bf16 약 230 GB/s).
- **같은 데이터를 `[행, K]`로 보면 FP8도 bf16과 같은 약 228 GB/s가 된다**(안쪽 줄 128 B). bf16은 64 B에서 이미 포화라 바뀌지 않는다.
- 변환(FP8→f32·bf16), 곱셈, 요청당 바이트, CTA 수(split-K)는 원인이 아니다.
- 작은 행렬에서 "시간이 원소 수에 비례"해 보인 것은 측정마다 약 75 µs의 고정 비용이 붙었기 때문이다. FP8과 bf16에 같은 크기로 붙는다. 이 고정 비용은 event 구간 안의 host 쪽 launch 비용으로 추정하며, 확인하지 않았다. 아래 정상 상태 대역폭은 크기별 시간의 기울기라 이 비용과 무관하다.

## 1. 변환과 요청 크기 (8192×2048, 128행 타일)

| 변형 | 하는 일 | ms | GB/s |
|---|---|---:|---:|
| P0 | FP8 제품 GEMV | 0.184 | 91 |
| P1 | FP8 load만 | 0.186 | 90 |
| P2 | FP8 load → f32 합 | 0.186 | 90 |
| P3 | FP8 load → bf16 → f32 합 | 0.186 | 90 |
| P6 | FP8 load, K 타일 2배 | 0.176 | 95 |
| P4 | bf16 load만 | 0.218 | 154 |
| N1 | NVFP4 packed load만 | 0.113 | 74 |

P1이 P0와 같으므로 변환과 곱셈은 시간을 더하지 않는다. 형상 4096×2048, 2048×4096, 512×2048과 16×4 타일에서도 경향이 같다.

## 2. SASS (GB10 JIT 캐시 cubin, `cuobjdump -sass`)

- 모든 변형에서 가중치는 `UTMALDG.3D`(TMA)로 shared memory에 오고, 소비는 `LDS.128`이다. 좁은 `LDG`는 없다.
- pipeline 구조는 원소 타입과 무관하게 같다: TMA 8개(2 stage × box 4개), mbarrier 33개. shared 버퍼는 FP8 32 KB, bf16 64 KB.
- `[행, K]` view 변형은 `UTMALDG.2D`로 바뀐다.

## 3. view 모양과 split-K (load만)

R은 `[행, K]` view, S는 그 view에서 K를 S등분해 CTA를 S배로 띄운 것이다. 단위 GB/s.

| 형상 | FP8 지금 (P1) | FP8 `[행, K]` (R1) | split 2 | split 4 | bf16 지금 (P4) | bf16 `[행, K]` (R4) | split 2 | split 4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8192×2048 | 91 | 118 | 120 | 122 | 154 | 150 | 151 | 154 |
| 2048×4096 | 80 | 100 | 102 | 104 | 121 | 117 | 118 | 121 |
| 512×2048 | 47 | 54 | 55 | 54 | 72 | 77 | 78 | 78 |

split-K로 CTA를 2배·4배 늘려도 바뀌지 않는다. 소비 쪽 병렬도는 원인이 아니다.

## 4. 행렬 크기 (8192행 이상, 128행 타일)

| 형상 | FP8 지금 (P1) ms | FP8 `[행, K]` (R1) ms | bf16 지금 (P4) ms | NVFP4 제품 (P0c) ms |
|---|---:|---:|---:|---:|
| 8192×2048 | 0.184 | 0.142 | 0.218 | 0.132 |
| 8192×4096 | 0.308 | 0.223 | 0.362 | 0.243 |
| 8192×8192 | 0.543 | 0.372 | 0.649 | 0.434 |
| 8192×16384 | 0.990 | 0.668 | 1.231 | 0.808 |
| 32768×8192 | 1.916 | 1.257 | 2.372 | 1.340 |

크기를 두 배로 할 때 늘어나는 시간으로 구한 정상 상태 대역폭:

| 변형 | 안쪽 줄 | 정상 상태 대역폭 | 고정 비용 |
|---|---:|---:|---:|
| FP8 지금 (P1) | 32 B | 약 145 GB/s | 약 70 µs |
| FP8 `[행, K]` (R1) | 128 B | 약 226 GB/s | 약 75 µs |
| bf16 지금 (P4) | 64 B | 약 232 GB/s | 약 74 µs |
| NVFP4 제품 (P0c) | 16 B | 약 90 GB/s | |

decode 분해의 dense_fp8 134 GB/s(nsys, graph 안 커널 시간)는 P1의 정상 상태 대역폭과 가깝다.

## 적용: GEMV를 `[행, K]` view로 (`7abcbd2` → `0a29c40`, GB10 같은 pod)

- 테스트: fp8_oracle, fp8_tile, nvfp4_oracle, moe, gemv_dense, kernels lib, nvfp4_prefix(실제 체크포인트) 모두 통과.
- decode batch 1: `qwen3.6:nvfp4` 23.11 → **18.34 ms (43.3 → 54.5 tok/s)**, FP8 24.50 → **18.94 ms (40.8 → 52.8 tok/s)**.
- `bench --profile`(eager, ms/step): gemv 16.16 → 13.56, moe_expert_gemv 6.36 → 4.86, lm_head 4.23 → 4.21.
- 품질(decode 모드 8묶음, 같은 묶음): NVFP4 ΔNLL 1.1441e-2 → 1.1426e-2(CI ±5.9e-3), FP8 2.2158e-3 → 2.2046e-3(CI ±3.7e-3). 평균 KL 차이 3e-7 이하, top-1 동일. 합 순서 변화 수준이라 통과.

probe 제품 경로(ms, 이전 커밋도 같은 pod에서 다시 잼):

| 형상 (타일 행) | FP8 P0 이전 → 새 | NVFP4 P0c 이전 → 새 |
|---|---|---|
| 8192×2048 (128) | 0.186 → 0.141 | 0.133 → 0.103 |
| 4096×2048 (64) | 0.103 → 0.083 | 0.078 → **0.155** |
| 2048×4096 (32) | 0.107 → 0.082 | 0.087 → **0.130** |
| 512×2048 (8) | 0.025 → 0.022 | 0.029 → 0.030 |

- FP8은 모든 형상에서 load만 하는 R1과 같은 시간이다.
- NVFP4는 타일 행 64·32에서 느려졌다. SASS: spill은 없다(LOCAL 0). 이 두 타일에서만 tileiras가 packed 가중치를 shared memory에서 `LDS.U8`로 1바이트씩 읽어 f32로 풀고 `F2FP.SATFINITE.E2M1.F32`로 다시 묶어 `STS`하는 왕복 경로를 고른다(4096×2048: 명령 4184 → 10176, `BAR.SYNC` 67 → 807). 타일 행 128과 16·8에는 이 경로가 없다. 어떤 reshape나 레이아웃 변환이 이 경로를 유발하는지는 확인하지 않았다.
- 모델 영향: routed expert는 타일 행 128이라 영향 밖이다(expert 커널 SASS는 직접 보지 않음). shared expert down(2048×512, 타일 행 32)은 0.021 → 0.028 ms, gate_up(1024×2048, 16)은 0.046 → 0.037 ms로 40층 순효과는 약 −0.08 ms(추정).

## decode GEMV 잔여분 원인 분석 (`1d61b54`~, GB10, nsys 커널 시간, 2026-10-06)

`qwen3.6:nvfp4` decode에서 GEMV는 step당 15.2 ms이고, 가중치 3.0 GB / 257 GB/s 하한 대비 약 3.4 ms 초과다(FP8 1.65, NVFP4 expert 1.12, bf16 소형 0.65). 단계별 보고서는 scratchpad의 gemv-residual-1~7, 예제는 `gemv_*_probe`.

측정 기준
- 커널 시간은 nsys로 잰다. eager do_bench event 시간에는 host launch 비용(약 70 µs)이 섞인다.
- L2는 읽기로 비운다(쓰기로 비우면 FP8 2048×4096이 77 µs로 decode 44.7 µs와 다르다).
- 데이터 값(0·상수·패턴·무작위·실제 가중치)은 시간에 영향이 없다(1% 안).
- 프로세스가 큰 메모리를 먼저 할당하지 않으면 같은 커널이 약 8% 느리다(lm_head 0 GiB 236, 22 GiB 이상 257 GB/s). decode는 모델 약 23 GB를 할당하므로 257이 기준이다. 메커니즘은 미확인.

확정 원인 (t = 절편 + 반복 수 × 반복당 시간, KT 128)
| 성분 | 원인 | 대책 | step당 (추정) |
|---|---|---|---:|
| launch마다 절편 2~4 µs, 작은 N의 반복당 지연 | 커널 수와 작은 행렬 | 같은 입력·같은 형식 GEMV를 로드 시 이어 붙이기 | 약 0.6 ms |
| FP8 per-tensor scale | BLOCK_K = cols라 바깥 루프 1회 안의 상수 반복 루프로 컴파일(STACK 최대 1 KB, z에서 +4.3 µs) | per-tensor를 행 scale로 표현하고 평탄한 K 루프 | 약 0.16 ms |
| FP8 작은 타일(BN 8)에서 반복마다 [1,1] scale 곱 | tileiras lowering: 루프 동기화 2배(BAR 14 → 26) | 평탄 루프(scale을 끝에서), 타일 선택은 E7 | 약 0.3 ms 중 일부 |
| NVFP4 그룹 scale 곱 | tileiras lowering: [BN, KT/16] 피연산자가 합류할 때 unpack된 FP4 값을 다시 변환(사전 할당 조건 반복당 +15.5%, F2FP 128 → 408). scale을 정수로 디코드해 convert를 없애도 F2FP 384 | 없음: 원소별 펼침·정수 디코드·bf16/f32 미리 변환 모두 같거나 느림 | 약 0.34 ms |
| 독립 GEMV 동시 실행 | safe graph API에 병렬 분기 없음 | cutile-rs 기능(이슈 010) | 상한 약 1 ms |

배제한 가설: pipeline 깊이(`load_pipelined`), 반복마다 끝내는 reduce, CTA 수·메모리 병렬도(8 CTA에서 이미 포화), 데이터 값, eager 대 graph, 지속 부하.
이슈 초안: 010(graph 병렬 분기), 011(FP8 반복당 scale), 012(FP4 그룹 scale), 008(FP4 unpack 후 reduce).

## 적용: FP8 행 scale(A)과 같은 입력 GEMV 이어 붙이기(B) (`fc8667a` → `cdee4a9`, GB10 같은 pod, 3회 교대)

design-e4 4.3의 3·4번. 테스트: firepit --lib 전체(음성 대조 포함), engine, reference(ignored), nvfp4_prefix, fp8_prefix, kernels 전부 통과.

| 항목 | 기준 | A+B | 변화 |
|---|---:|---:|---:|
| decode `qwen3.6:nvfp4` | 16.72 ms (59.8 tok/s) | **15.79 ms (63.3 tok/s)** | −0.93 ms (A −0.35, B −0.54) |
| decode FP8 | 18.39 ms | 17.68 ms | −0.71 ms |
| decode bf16 | 27.51 ms | 26.85 ms | −0.66 ms |
| prefill 5000 (nvfp4) | 4053 ms | 4127 ms | +74 ms (+1.8%, 범위 겹침) |
| 품질 prefill 72묶음 NVFP4 ΔNLL | 1.076e-2 | 1.093e-2 | CI ±2.2e-3 안 |
| 품질 decode 8묶음 NVFP4 ΔNLL | 1.145e-2 | 1.133e-2 | CI ±5.8e-3 안 |

- 처음 B는 router 묶음(padding 272행)을 prefill에서도 한 GEMM으로 돌려 GEMM 타일이 16으로 작아졌다(층당 3263 µs). padding한 묶음을 prefill에서 멤버별로 돌리게 고쳐(`714308f`) 층당 781 µs가 됐다.
- 남은 prefill +62 ms(gemm)는 행 scale GEMM의 형상별 차이다. 루프 안에 작은 load 하나가 있느냐에 따라 tileiras가 pipeline·unroll을 다르게 골라 형상마다 ±15~25%다(out·o 2048×4096에서 +22~28%, 묶음 `[qkv|z]`에서는 더 빠름). firepit 쪽 원칙적 우회는 없고 이슈 후보로 기록했다. 같은 측정이 프로세스 사이 최대 14% 흔들려 형상별 수치는 방향만 신뢰한다.
- 출력 80토큰 이상이면 decode 이득이 prefill 손실보다 크다.
- 음성 대조 `reset_rope_position_breaks_append`는 `41bf08d`(shared expert 통합)부터 무작위 소형 모델의 top-5 우연으로 실패하고 있었다. 판정을 정상 비교와 같은 허용 오차 함수로 바꿨다(`cdee4a9`).
