# ds4 Qwen3.8-Flash-Next decode profiling on DGX Spark (GB10)

> **정정 (2026-09-29):** 이 보고서의 `matvec_q8` 수치(3.12 GB/token, 133 GB/s, floor 대비 66%, 초과분 +7.9 ms)는 과소 추정이다. ncu 샘플에 output head가 빠져 있었다. 올바른 값은 3.58 GB/token, 152 GB/s(76%), +5.6 ms이고, 토큰당 전체 read는 6.36 GB, floor는 31.8 ms이다. 자세한 내용은 [candle-q8-baseline.md](candle-q8-baseline.md) 5절에 있다.

표기: **[확인]** = 실측 또는 코드/GGUF header로 확인. **[추정]** = 계산·가정에 기반, 직접 측정하지 않음. **[미확인]** = 원인 불명.

## 요약

- GB10에서 Q2 decode @8K는 토큰당 51.4 ms(19.5 t/s, n-gram warm)이다. [확인]
- GPU는 90% 시간 동안 커널을 실행 중이다. 커널 합 46.6 ms, idle 5.3 ms. [확인]
- 토큰당 DRAM read는 약 5.9 GB이고, 200 GB/s floor로 29.5 ms이다. 커널 시간의 63%가 floor로 설명된다. [확인: ncu bytes × nsys 시간]
- **GDN recurrence는 병목이 아니다.** GDN 전용 커널 합계 0.90 ms(커널 합의 1.9%)이고, `gdn_scan`의 read는 이미 floor 수준이다. [확인]
- floor 대비 초과 시간의 대부분은 Q8_0 dense GEMV(`matvec_q8`, +7.9 ms), MoE GEMV(`moe_mv`, +4.7 ms), latency-bound 소형 커널(+3.3 ms), GPU idle(+5.3 ms)이다. [확인]

## 측정 조건

| 항목 | 값 |
|---|---|
| 노드 | DGX Spark `spark-cb5e`, NVIDIA GB10 (compute capability 12.1), 20 CPU, RAM 125,481,412 kB, NVMe `/dev/nvme0n1p2` [확인] |
| 드라이버 | 580.178.04, `RmProfilingAdminOnly=1` [확인] |
| 실행 환경 | microk8s pod, `nvidia/cuda:13.0.0-devel-ubuntu24.04`, runtimeClass `nvidia`, `nvidia.com/gpu: 1`(GPU 단독 할당), `SYS_ADMIN` capability [확인] |
| 도구 | nvcc 13.0.48, nsys 2025.3.2 (`cuda-nsight-systems-13-0`), ncu 2025.3.0 [확인] |
| ds4 | main `0aaea5a` (2026-09-20), `make cuda-spark` (`sm_121a`). `make test-qwen4-cuda CUDA_ARCH=sm_121` 전부 통과 [확인] |
| 모델 | `Qwen3.8-Flash-Next-Q2.gguf`, HF `antirez/qwen3.8-flash-next-gguf` rev `d600fe1`, 147,207,127,040 B. resident 41.72 GiB, 계획 메모리 49.35 GiB (KV 0.26 + buffers 7.37) [확인] |
| 벤치 | `ds4-bench --cuda -m Qwen3.8-Flash-Next-Q2.gguf --prompt-file speed-bench/promessi_sposi.txt --ctx-start 8192 --ctx-max 8192 --gen-tokens 128 --teacher-forced-decode`, prefill chunk 기본값(8192), MTP 없음 [확인] |
| 동시 부하 | 측정 전마다 GPU compute process 없음 확인. 다른 GPU pod 없음 [확인] |

## 1. Decode 속도 (ds4-bench)

| 실행 | n-gram page cache | prefill t/s | gen t/s | gen steady t/s |
|---|---|---:|---:|---:|
| base r1 | 첫 실행 | 506.77 | 18.26 | 18.29 |
| base r2 | warm | 578.59 | 19.48 | 19.51 |
| cold1 | n-gram 범위 evict | 513.93 | 18.41 | 18.44 |
| warm1 | warm | 585.20 | 19.49 | 19.53 |
| cold2 | n-gram 범위 evict | 512.15 | 18.41 | 18.45 |
| warm2 | warm | 583.46 | 19.36 | 19.39 |

- cold/warm 차이는 토큰당 약 2.9 ms(54.3 vs 51.4 ms)이고 재현된다. [확인]
- cold 조건은 GGUF의 n-gram tensor 범위(offset 44,806,635,520, 길이 102,400,491,520 B)에 `posix_fadvise(DONTNEED)`를 걸어 만들었다. [확인]
- docs/QWEN38_FLASH_NEXT.md의 수치(Q2 22.6 t/s)보다 이 측정이 약 14% 느리다. 원인은 모른다(측정 commit, page cache 상태, 전력/온도 차이 가능). [미확인]

## 2. DRAM 대역폭 floor (saturating-read kernel)

`uint4` grid-stride read, 20회 반복의 median / max (GB/s). [확인]

| buffer | blocks/SM 4 | 8 | 16 | 32 | max |
|---|---:|---:|---:|---:|---:|
| 2 GiB | 208.6 | 213.3 | 215.3 | 217.4 | 236.1 |
| 8 GiB | 206.3 | 199.9 | 201.0 | 200.2 | 213.3 |
| 16 GiB | 202.6 | 190.8 | 201.1 | 197.6 | 205.6 |
| 8 GiB, 4-way unroll | 202.3 | 192.9 | 195.1 | 192.9 | 206.3 |

- 큰 buffer에서 지속 대역폭은 약 200 GB/s, peak는 약 235 GB/s이다. peak는 ds4 #773의 231–234 GB/s와 맞는다. [확인]
- 이 보고서의 floor는 200 GB/s를 쓴다. 모델 working set(41.7 GiB)이 큰 buffer 조건에 가깝기 때문이다. [추정]

## 3. nsys timeline (decode token 48–79, 32 tokens)

`DS4_BENCH_CUDA_PROFILE_RANGE=48:32`, `nsys profile --trace=cuda,osrt,nvtx --capture-range=cudaProfilerApi --sample=none --cpuctxsw=none`. n-gram warm 상태(같은 prompt 3번째 실행). [확인]

### 토큰당 요약 [확인]

| 항목 | 값 |
|---|---:|
| capture window | 51.99 ms/token (비계측 warm 51.4 ms와 일치) |
| 커널 수 | 1,067 /token |
| 커널 시간 합 | 46.61 ms |
| GPU busy (kernel ∪ memcpy) | 46.71 ms |
| GPU idle | 5.28 ms |
| memcpy | 52 /token, 3.0 MB, 0.10 ms |
| `cudaLaunchKernel` | 1,067 /token, 평균 3.3 µs, CPU 3.49 ms (GPU 실행과 겹침) |
| `cudaDeviceSynchronize` | 2 /token |

nsys 하에서 전체 run의 gen t/s는 14.47로 떨어졌다. capture window 안의 토큰 시간은 비계측과 같으므로, 감소분은 capture 시작/종료 구간 밖의 오버헤드로 본다. [추정]

### GPU idle 분해 (커널 사이 gap 크기별, ms/token) [확인]

| gap 크기 | ms/token |
|---|---:|
| < 2 µs | 0.52 |
| 2–5 µs | 1.72 |
| 5–10 µs | 0.29 |
| 10–100 µs | 0.02 |
| 100 µs–1 ms | 1.16 |
| > 1 ms | 1.66 |

- 가장 큰 gap 3개는 10.07, 9.96, 9.86 ms이다. 32토큰 중 3번이고 토큰당 평균 약 0.94 ms이다. 원인은 모른다. [확인: 크기 / 미확인: 원인]
- 그다음 gap들은 1.2–1.9 ms로, 토큰 경계의 host 작업(sync, logits 복사, n-gram 읽기 등)으로 본다. [추정]
- OSRT trace에 `pread`는 잡히지 않았다. warm 상태라 호출이 짧아 nsys 기본 threshold 아래였을 가능성이 있다. [추정]

### 커널별 시간 [확인]

| kernel | ms/token | 커널 합 대비 | launches/token | 평균 µs |
|---|---:|---:|---:|---:|
| `matvec_q8` | 23.495 | 50.4% | 183 | 128.4 |
| `moe_mv` | 9.558 | 20.5% | 96 | 99.6 |
| `matvec` | 5.607 | 12.0% | 217 | 25.8 |
| `hc_mix` | 3.466 | 7.4% | 97 | 35.7 |
| `router` | 0.946 | 2.0% | 48 | 19.7 |
| `attention` | 0.883 | 1.9% | 12 | 73.6 |
| `hc_norm` | 0.870 | 1.9% | 97 | 9.0 |
| `gdn_scan` | 0.560 | 1.2% | 36 | 15.6 |
| `idx_select` | 0.239 | 0.5% | 12 | 19.9 |
| `moe_reduce` | 0.194 | 0.4% | 48 | 4.0 |
| `hc_combine` | 0.161 | 0.3% | 48 | 3.4 |
| `conv` | 0.158 | 0.3% | 36 | 4.4 |
| `gdn_prep` | 0.124 | 0.3% | 36 | 3.4 |
| `attn_merge` | 0.084 | 0.2% | 12 | 7.0 |
| `idx_score` | 0.059 | 0.1% | 12 | 4.9 |
| `gdn_out` | 0.058 | 0.1% | 36 | 1.6 |
| `attn_prep` | 0.055 | 0.1% | 12 | 4.6 |
| `ngram_gate` | 0.031 | 0.1% | 1 | 30.8 |
| `idx_expand` | 0.028 | 0.1% | 12 | 2.4 |
| `tile_max` | 0.018 | 0.0% | 12 | 1.5 |
| `block_key` | 0.011 | 0.0% | 3 | 3.7 |
| `ngram_conv` | 0.005 | 0.0% | 1 | 5.0 |

## 4. ncu: 커널별 DRAM traffic

### 방법

- GB10(`gb10b`)은 `dram__*` metric을 제공하지 않는다. [확인]
- 대신 L2 miss read sector를 썼다: `lts__t_sectors_aperture_sysmem_op_read_lookup_miss.sum × 32 B` (device aperture는 0). GB10의 read는 모두 sysmem aperture로 잡힌다. [확인]
- 보정: 2 GiB saturating read에서 67,332,651 sector × 32 = 2.1546 GB, 실제 2.1475 GB와 비교해 +0.3%이다. 같은 커널의 ncu 시간은 9.81 ms(219 GB/s)이다. [확인]
- 범위는 `DS4_BENCH_CUDA_PROFILE_RANGE=48:1`로 한 토큰이다. 전체 1,064개 중 앞 245개 커널(대략 layer 0–10, GDN layer와 attention layer 2개 포함)만 측정하고 중단했다. [확인]
- 나머지 layer도 같은 layer 유형이면 커널당 bytes가 같다고 가정한다. [추정]
- ncu 기본 `--cache-control all`로 커널마다 L2를 flush하므로, 실제 실행보다 bytes가 약간 많게 잡힐 수 있다(특히 이전 커널이 L2에 남긴 activation, `gdn_scan` state). [추정]
- 커널 시간은 ncu 하에서 부풀려지므로(예: `router` 31.4 µs vs nsys 19.7 µs), 달성 대역폭은 **ncu bytes ÷ nsys 평균 시간**으로 계산했다. [확인: 계산 방식]

### 결과

| kernel | ncu 샘플 수 | read MB/launch | write MB/launch | 달성 GB/s (nsys 시간 기준) | read GB/token |
|---|---:|---:|---:|---:|---:|
| `matvec_q8` | 41 | 17.04 | 0.28 | 133 | 3.12 |
| `moe_mv` | 22 | 10.09 | 0.70 | 101 | 0.97 |
| `matvec` | 52 | 4.23 | 0.09 | 164 | 0.92 |
| `hc_mix` | 22 | 6.64 | 0.07 | 186 | 0.64 |
| `gdn_scan` | 9 | 3.21 | 4.08 | 206 (read만) | 0.12 |
| `attention` | 2 | 4.38 | 1.68 | 60 | 0.05 |
| `attn_merge` | 2 | 1.69 | 0.04 | 241 | 0.02 |
| `hc_norm` | 23 | 0.23 | 0.14 | 26 | 0.02 |
| 기타 소형 커널 | — | < 0.8 | < 1.4 | — | 약 0.04 |
| **합계** | | | | | **약 5.9** |

- `router` 커널의 read는 launch당 0.07 MB이다. router weight(F32, layer당 5 MiB)의 GEMV는 `matvec`에서 돌고 `router`는 top-k만 하는 것으로 본다. [확인: bytes / 추정: 역할 분담]
- `moe_mv`의 0.97 GB/token은 routed expert 10개(GGUF 기준 0.715 GB)에 shared expert(Q8_0, 0.25 GB)를 더한 값과 맞는다. 코드 주석상 shared expert는 MoE의 추가 slot으로 실행된다. [확인: bytes, 코드 주석 / 추정: 대응 관계]
- GDN state는 f32, layer당 6144 × 128 × 4 B = 3 MiB이다(`ds4.c:57997`). `gdn_scan`의 read 3.21 MB는 이와 맞다. [확인]

## 5. Floor 분석 (토큰당, floor = 200 GB/s)

| 구성 | 실측 ms | floor ms | 초과 ms | floor 대비 효율 |
|---|---:|---:|---:|---:|
| `matvec_q8` | 23.50 | 15.59 | +7.9 | 66% |
| `moe_mv` | 9.56 | 4.85 | +4.7 | 51% |
| `matvec` | 5.61 | 4.59 | +1.0 | 82% |
| `hc_mix` | 3.47 | 3.22 | +0.25 | 93% |
| `gdn_scan` | 0.56 | 0.58 | 0 | 약 100% (read 기준) |
| 나머지 커널 | 3.91 | 약 0.65 | +3.3 | 약 17% |
| **커널 합** | **46.61** | **29.5** | **+17.1** | **63%** |
| GPU idle | 5.28 | 0 | +5.3 | |
| **토큰당 합계** | **51.99** | **29.5** | **+22.4** | |
| n-gram cold read (cold일 때만) | +2.9 | | | |

bytes는 ncu, 시간은 nsys로 측정한 값이다. [확인] 커널별 floor는 두 값의 결합으로 계산했다. 235 GB/s peak를 쓰면 floor는 25.1 ms이다. [확인: 계산]

## 6. 사전 추정과 비교 (GGUF header 정적 분석)

- 토큰당 weight read를 GGUF header의 tensor 크기로 계산하면 5,815 MiB(6.10 GB)이다. [확인: header 기반 계산]
  - dense 5,133 MiB: GDN projection(qkv, gate, ssm_out, Q8_0) 2,090 / HC mixer(F16) 1,200 / output head(Q8_0) 644 / attention projection(Q8_0) 624 / router(F32) 240 / shared expert(Q8_0) 239 / 기타 약 96
  - routed expert 10/512: 682 MiB (IQ2_XXS gate/up, Q2_K down)
- ncu 실측 합계 약 5.9 GB와 3% 이내로 맞는다. [확인]
- "active 6B × Q2 ≈ 2 GB/token"이라는 초기 추정은 틀렸다. dense weight가 Q2가 아니라 Q8_0/F16이기 때문이다. [확인]
- `matvec_q8`의 3.12 GB/token 중 약 2.2 GB(2,090 MiB)는 GDN projection weight이다. [추정: 커널-tensor 대응은 측정하지 않았고 bytes 크기로만 대응시킴]

## 7. 가설별 판정

| 가설 | 판정 | 근거 |
|---|---|---|
| 1. GDN recurrent state read-modify-write | **기각** | GDN 전용 커널(`gdn_scan` 0.56 + `gdn_prep` 0.124 + `conv` 0.158 + `gdn_out` 0.058) = 0.90 ms, 커널 합의 1.9%. `gdn_scan` read는 floor 수준. [확인] |
| 2. n-gram(PLE) table SSD read | **부분 기여** | cold +2.9 ms/token(5.6%), warm이면 약 0. 행 수가 256 미만이면 한 thread에서 16번 `pread`를 순서대로 호출한다(`ds4.c:57436`). [확인] 실사용에서 cold/warm 비율은 모른다. [미확인] |
| 3. launch 수, host 오버헤드 | **부분 기여** | GPU idle 5.3 ms(10%). 커널 1,067개/token이고 Qwen CUDA decode 경로는 CUDA Graph capture를 쓰지 않는다(`ds4_gpu_decode_graph_begin` 미호출). [확인] 10 ms 급 stall 3회의 원인은 모른다. [미확인] |
| 4. 기타 | **주 병목** | quantized GEMV 효율: `matvec_q8` 66%, `moe_mv` 51%. 둘의 초과분 합계 +12.6 ms. latency-bound 소형 커널 +3.3 ms. attention(QSA)은 0.88 ms(1.9%). [확인] |

## 8. 결론

1. GB10에서 Qwen3.8-Flash-Next decode의 병목은 GDN이 아니다. GDN recurrence 커널은 이미 대역폭 floor 수준이고, 비중은 2% 미만이다. [확인]
2. 줄일 여지가 가장 큰 곳은 quantized GEMV 효율이다. Q8_0 dense GEMV(+7.9 ms)와 IQ2_XXS/Q2_K MoE GEMV(+4.7 ms)가 floor까지 가면 토큰당 약 12.6 ms가 줄어든다. [확인: 초과분 / 추정: 달성 가능성]
3. 그다음은 소형 커널과 launch gap(합 약 8.6 ms)이다. CUDA Graph와 커널 fusion이 대상이다. [확인: 시간 / 추정: 해결 수단]
4. cutile-rs의 첫 구현 대상은 GDN decode보다 Q8_0 GEMV나 quantized MoE GEMV가 decode 성능에 더 직접적이다. [추정]

## 9. 미해결 사항

- docs 수치(22.6 t/s)와의 14% 차이 원인. [미확인]
- nsys window 안의 10 ms 급 GPU stall 3회의 원인. [미확인]
- Q2(22.6)와 Q4(21.2)의 속도가 비슷한 이유. expert bytes 차이만으로 계산한 예측(+2.5 ms)은 문서 실측 차이(+2.9 ms)와 비슷하다. 그러나 `moe_mv`가 floor의 51%에 불과하므로 bytes만으로 설명된다고 볼 수 없다. Q4는 받지 않았고 Q4 header도 읽지 않았다. [미확인]
- 측정 시작 전 compute process가 없는데도 `nvidia-smi` GPU 사용률이 14%로 표시된 이유. 측정 중에는 GPU를 단독 할당했다. [미확인]
- ncu 전체 토큰(1,064 커널) 측정은 하지 않았다. 245개 샘플로 커널 종류별 bytes를 얻었다. [확인]

## 산출물 (Spark `/var/lib/sam-ds4`)

- `ds4/` (빌드 결과, commit `0aaea5a`)
- `base-r{1,2}.csv`, `cold{1,2}.csv`, `warm{1,2}.csv`: ds4-bench 결과
- `nsys-decode.nsys-rep`, `nsys-decode.sqlite`: 32토큰 timeline
- `partial.ncu-rep`, `partial.csv`: ncu 245커널
- `bwread.cu`: 대역폭 측정 커널
