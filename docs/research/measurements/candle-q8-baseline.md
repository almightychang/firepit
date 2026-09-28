# candle CUDA Q8_0 GEMV baseline on DGX Spark (GB10, sm_121)

표기: **[확인]** = 실측, 또는 코드·로그로 확인. **[추정]** = 계산이나 간접 근거. **[미확인]** = 원인이나 사실을 확인하지 못함.

## 요약

- candle `5ba5d5b`는 `CUDA_COMPUTE_CAP=121`로 `cuda` feature를 빌드할 수 있다. clean 빌드 70 s, `mmvq_gguf.cu`만 수정한 incremental 빌드 19 s. [확인]
- `cutile` feature(`cutile = "=0.3.1"`)도 컴파일된다. cuda 빌드에 이어 추가로 62 s. 단 빌드에 libclang이 필요하다. [확인]
- **candle 커널은 호스트 580 드라이버에서 로드되지 않는다** (`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`). CUDA 13.3으로 만든 PTX를 13.0 드라이버가 JIT하지 못하기 때문이다. 이미지의 forward-compat libcuda(13.3)로는 동작한다. [확인]
- `fast_mmvq`(bf16 입력)는 ds4 Q8_0 shape에서 **162–192 GB/s, floor 200 GB/s의 81–96%**를 낸다. 같은 shape에서 ds4 `matvec_q8`(평균 129–167 GB/s)보다 빠르다. [확인: 두 측정값 / 추정: 측정 방식이 달라 직접 비교에는 한계가 있음, 아래 참고]
- `FORCE_DMMV` 경로는 f32 입력만 받고, 119–148 GB/s(60–74%)이다. bf16 입력은 에러가 난다. [확인]
- **ds4 profiling 보고서 정정:** `matvec_q8`은 3.12 GB/token, 133 GB/s가 아니라 **3.58 GB/token, 152 GB/s(76%)**이다. 이전 값은 output head가 빠진 ncu 샘플 평균을 토큰 전체로 늘린 것이었다. [확인]

## 환경

| 항목 | 값 |
|---|---|
| 노드 | DGX Spark `spark-cb5e`, NVIDIA GB10 (compute capability 12.1) [확인] |
| 이미지 | `nvidia/cuda:13.3.1-devel-ubuntu24.04` (arm64), nvcc 13.3 [확인] |
| 실행 드라이버 | 이미지 내장 forward-compat libcuda `610.43.02` (CUDA 13.3). 호스트 드라이버는 580.178.04 [확인] |
| Rust | 1.98.0 (`/var/lib/sam-cutile/rustup` 재사용) [확인] |
| candle | `5ba5d5b468b5b1df40e82dd3d556987bedeea041` (2026-09-28) [확인] |
| cudaforge | 0.1.x (`CUDA_COMPUTE_CAP` 환경 변수로 아키텍처 고정. nvidia-smi가 없는 pod에서 필요) [확인: cudaforge 0.1.6 소스] |
| nsys | 2026.1.3 (`cuda-nsight-systems-13-3`) [확인] |

## 1. 빌드

| 빌드 | 결과 | 시간 | 비고 |
|---|---|---:|---|
| `cuda` feature, clean (의존성 + CUDA 커널 15개) | 성공 | 70 s | `CUDA_COMPUTE_CAP=121`, nvcc 10 thread [확인] |
| `mmvq_gguf.cu`만 touch 후 incremental | 성공 | 19 s | 커널 한 파일 수정 시 반복 속도 [확인] |
| `cutile` feature, 1차 | **실패** | 6 s | `cuda-bindings v0.3.1`의 bindgen이 libclang을 찾지 못함. 빌드 pod에 `clang libclang-dev`를 설치하지 않은 설정 실수 [확인] |
| `cutile` feature, libclang 설치 후 | 성공 | 62 s | 추가로 컴파일된 crate: `cuda-bindings`, `cutile-ir`, `cuda-core`, `cuda-async`, `cutile-compiler`, `cutile-macro`, `cutile` (모두 0.3.1) [확인] |

- candle의 `cutile` 의존성은 `cutile = "=0.3.1"`(workspace `Cargo.toml:60`)로 고정되어 있다. DGX Spark 지원이 CHANGELOG에 들어간 것은 0.4.0이다. 이번에는 컴파일만 확인했고, 0.3.1 커널이 sm_121에서 실행되는지는 확인하지 않았다. [확인: 버전 / 미확인: 실행 여부]
- `candle-kernels/build.rs`는 `mmvq_gguf.cu`, `mmq_*.cu`, `moe_*.cu`를 정적 라이브러리(`libmoe.a`)로 빌드한다. 나머지 커널(예: `quantize_q8_1`, dmmv)은 PTX로 빌드해 실행 시 드라이버 JIT로 로드한다. [확인: build.rs]

## 2. 드라이버 호환성

| 드라이버 | 결과 |
|---|---|
| 호스트 580 (CUDA 13.0), compat 비활성화 | `Error: DriverError(CUDA_ERROR_UNSUPPORTED_PTX_VERSION, "the provided PTX was compiled with an unsupported toolchain.")` [확인] |
| 이미지 forward-compat libcuda (CUDA 13.3) | 정상 동작 [확인] |

- 원인은 candle의 PTX 커널이다. nvcc 13.3이 만든 PTX를 13.0 드라이버가 JIT하지 못한다. minor-version compatibility는 PTX JIT를 보장하지 않는다. [확인: 에러 / 추정: 실패한 커널이 PTX로 빌드된 것 중 어느 것인지는 확인하지 않음]
- cutile은 tileiras가 네이티브 cubin을 내므로 호스트 드라이버에서도 동작했다(이전 smoke test). candle과 cutile을 한 프로세스에서 같이 쓰려면, 호스트 드라이버에서는 candle 쪽이 막힌다. [확인: 두 결과 / 추정: 같이 쓸 때의 결론]
- 해결책은 둘이다. compat libcuda를 쓰거나(이번 측정), CUDA 13.0 toolkit으로 candle을 빌드한다. 13.0 빌드는 시도하지 않았다.

## 3. Microbench 방법

bench 크레이트 `q8bench`를 candle 트리 밖에 두었다(`/var/lib/sam-candle/q8bench`). candle 소스는 수정하지 않았다. 기존 `candle-core/benches/benchmarks/qmatmul.rs`는 1024×1024 고정, f32 입력만 지원하고 정답 검증이 없어서 재사용하지 않았다. [확인]

- **가중치:** 난수 f32 (N×K)를 CPU에서 Q8_0으로 양자화한다(`QTensor::quantize`). 같은 bytes를 `QStorage::from_data`로 GPU에 올린다. 따라서 CPU와 GPU의 가중치 bytes가 동일하다. [확인]
- **L2 회피:** 같은 가중치를 여러 GPU 버퍼에 복사하고(합계가 512 MiB 이상이 되도록, 최대 32개), 호출마다 버퍼를 돌려 쓴다. 매 호출이 DRAM에서 읽도록 하기 위해서다. output head(644 MiB)는 복사본 1개다. [확인: 코드 / 추정: GB10 L2 크기는 확인하지 않음]
- **입력:** batch 1, `[1, K]`, bf16 또는 f32. CPU 기준값도 같은 bf16 반올림 값을 f32로 쓴다. [확인]
- **경로:** `mmvq` = 기본(`fast_mmvq::try_fwd`). `dmmv` = `quantized::cuda::set_force_dmmv(true)`. [확인]
- **시간:** warmup 20회 후 200회 연속 호출하고, `synchronize` 뒤 wall time을 호출 수로 나눈다(µs/call). launch가 비동기로 쌓이므로 GPU가 바쁘면 wall ≈ GPU 시간이다. 작은 shape에서는 host 오버헤드가 섞일 수 있다. [확인: 방법 / 추정: 해석]
- **대역폭:** 가중치 bytes(N·K/32·34) ÷ µs/call. 활성값과 출력 bytes는 무시했다(가중치의 0.1% 미만). [확인]
- **정답:** candle CPU `QMatMul::forward`와 비교했다. `rel_err = max|gpu − cpu| / max|cpu|`. [확인]

### Shape

ds4 Qwen3.8-Flash-Next Q2 GGUF의 Q8_0 tensor에서 골랐다. K = 입력 차원, N = 출력 차원이다. [확인: GGUF header]

| 이름 | GGUF tensor | K | N | 가중치 bytes |
|---|---|---:|---:|---:|
| `gdn_qkv` | `blk.*.attn_qkv` (GDN layer 36개) | 2560 | 10240 | 27,852,800 |
| `attn_q` | `blk.*.attn_q` (attention layer 12개, gate 포함) | 2560 | 12288 | 33,423,360 |
| `gdn_gate` | `blk.*.attn_gate` (GDN) | 2560 | 6144 | 16,711,680 |
| `gdn_out` | `blk.*.ssm_out`, `blk.*.attn_output` | 6144 | 2560 | 16,711,680 |
| `shexp_up` | `blk.*.ffn_up_shexp` (shared expert) | 2560 | 640 | 1,740,800 |
| `output_head` | `output.weight` | 2560 | 248320 | 675,430,400 |
| `qwen3_4b_down` | Qwen3-4B `mlp.down_proj` 규모 | 9728 | 2560 | 26,460,160 |

## 4. 결과

### µs/call과 대역폭 [확인]

| shape | mmvq bf16 µs | GB/s | floor 대비 | mmvq f32 µs | GB/s | floor 대비 | dmmv f32 µs | GB/s | floor 대비 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gdn_qkv | 153.60 | 181.3 | 91% | 142.58 | 195.3 | 98% | 188.97 | 147.4 | 74% |
| attn_q | 177.87 | 187.9 | 94% | 180.35 | 185.3 | 93% | 236.02 | 141.6 | 71% |
| gdn_gate | 87.06 | 192.0 | 96% | 93.82 | 178.1 | 89% | 119.82 | 139.5 | 70% |
| gdn_out | 94.86 | 176.2 | 88% | 92.10 | 181.5 | 91% | 125.26 | 133.4 | 67% |
| shexp_up | 10.72 | 162.4 | 81% | 10.82 | 160.9 | 80% | 14.60 | 119.2 | 60% |
| output_head | 3612.57 | 187.0 | 94% | 3613.96 | 186.9 | 93% | 4563.32 | 148.0 | 74% |
| qwen3_4b_down | 157.12 | 168.4 | 84% | 152.53 | 173.5 | 87% | 198.07 | 133.6 | 67% |

- floor는 ds4 profiling에서 잰 saturating-read 대역폭(8 GiB buffer median 약 200 GB/s)이다. [확인]
- dmmv bf16은 7개 shape 모두 `Error: unexpected dtype, expected: F32, got: BF16`로 실패했다. dmmv 경로는 f32 입력만 받는다. [확인]
- 각 설정은 한 번씩만 측정했다. 반복 측정으로 편차를 확인하지는 않았다. [확인]

### 정답 비교 (candle CPU 기준) [확인]

| shape | mmvq bf16 rel_err | mmvq f32 rel_err | dmmv f32 rel_err |
|---|---:|---:|---:|
| gdn_qkv | 2.42e-3 | 3.96e-7 | 4.23e-3 |
| attn_q | 2.04e-3 | 2.66e-7 | 3.35e-3 |
| gdn_gate | 3.30e-3 | 3.56e-7 | 3.90e-3 |
| gdn_out | 3.09e-3 | 7.56e-7 | 4.30e-3 |
| shexp_up | 2.94e-3 | 3.43e-7 | 5.21e-3 |
| output_head | 2.72e-3 | 3.36e-7 | 2.92e-3 |
| qwen3_4b_down | 2.01e-3 | 4.23e-7 | 3.75e-3 |

- mmvq f32는 CPU와 거의 같다(1e-6 미만). CPU 경로도 활성값을 8-bit block으로 양자화하므로, 같은 계산을 하는 것으로 본다. [확인: 수치 / 추정: 이유]
- mmvq bf16의 약 2–3e-3은 출력이 bf16으로 반올림되기 때문으로 본다. bf16의 상대 정밀도는 2^-8 ≈ 3.9e-3이다. [추정]
- dmmv f32의 약 3–5e-3은 dmmv가 활성값을 양자화하지 않는 반면 CPU 기준값은 양자화하기 때문으로 본다. [추정]
- 허용 오차는 따로 정하지 않았다. 모든 경로가 1e-2 미만이다. [확인]

## 5. ds4 `matvec_q8`과 비교

ds4 값은 이전 profiling의 nsys 데이터(`/var/lib/sam-ds4/nsys-decode.sqlite`, decode 32토큰)에서 커널 grid 크기로 shape을 구분해 뽑았다. `matvec_q8`은 block당 4행을 처리하므로 grid = N/4이다. ncu 샘플의 launch당 read bytes가 각 grid의 가중치 크기와 맞는 것으로 대응을 확인했다. [확인]

| shape | ds4 launches/token | ds4 평균 µs | ds4 중앙값 µs | ds4 GB/s (평균 / 중앙값) | candle mmvq bf16 µs | candle GB/s |
|---|---:|---:|---:|---:|---:|---:|
| gdn_qkv (N=10240) | 37 | 170.33 | 148.06 | 163.5 / 188.1 | 153.60 | 181.3 |
| attn_q (N=12288) | 12 | 208.76 | 175.55 | 160.1 / 190.4 | 177.87 | 187.9 |
| gdn_gate (N=6144) | 36 | 100.22 | 91.62 | 166.8 / 182.4 | 87.06 | 192.0 |
| gdn_out (N=2560, K=6144) | 49 | 129.05 | 102.14 | 129.5 / 163.6 | 94.86 | 176.2 |
| output_head (N=248320) | 1 | 4274.50 | 4122.85 | 158.0 / 163.8 | 3612.57 | 187.0 |
| (참고) N=512, K=2560 (attn_k/v) | 36 | 11.39 | 11.01 | 122 / 126 | — | — |
| (참고) shexp_up N=640 | — | — | — | — | 10.72 | 162.4 |

- ds4 launch 수에는 같은 grid를 쓰는 다른 tensor가 섞여 있을 수 있다. 예를 들어 N=10240 grid가 37회인데, GDN qkv는 36개다. [확인: 횟수 / 추정: 나머지 1회의 정체]
- **측정 방식이 다르다.** ds4는 실제 decode 중의 커널 단독 GPU 시간(nsys)이고, 사이사이에 다른 커널이 끼어 있다. candle은 같은 GEMV를 연속 호출한 wall time이다. candle 쪽에 host 오버헤드가 섞여 있다면, candle 커널 자체는 표보다 더 빠르다. [추정]
- ds4는 평균이 중앙값보다 15–25% 크다. 일부 launch가 느린 긴 꼬리가 있다는 뜻이다. 중앙값 기준으로는 ds4와 candle의 차이가 작다. 꼬리의 원인은 모른다. [확인: 분포 / 미확인: 원인]
- 드라이버와 toolkit이 다르다. ds4는 호스트 580 + CUDA 13.0 빌드, candle은 compat 13.3 + CUDA 13.3 빌드다. 이전 cutile smoke test에서는 compat과 호스트의 성능이 같았다. [확인]

### ds4 profiling 보고서 정정

`/tmp/claude-1000/ds4-profile-results.md`의 `matvec_q8` 값은 ncu 샘플(한 토큰의 앞쪽 245개 커널)의 launch당 평균 17.04 MB × 183 launch로 계산했다. 이 샘플에는 토큰 끝의 output head(675 MB)가 없어서 과소 추정이었다. [확인]

| 항목 | 이전 보고 | shape별 재계산 |
|---|---:|---:|
| `matvec_q8` read GB/token | 3.12 | **3.58** |
| `matvec_q8` 달성 대역폭 | 133 GB/s (66%) | **152 GB/s (76%)** |
| `matvec_q8` floor 대비 초과 시간 | +7.9 ms | **+5.6 ms** |
| 토큰당 전체 read | 약 5.9 GB | **약 6.36 GB** |
| 토큰당 floor (200 GB/s) | 29.5 ms | **31.8 ms** |

재계산에는 shape별 launch 수 × GGUF 가중치 bytes를 썼다. `matvec_q8` 시간(23.5 ms/token)은 두 계산에서 같다. [확인]

## 6. 해석

- candle `fast_mmvq`는 큰 Q8_0 shape에서 이미 floor의 88–96%를 낸다. 따라서 cutile Q8_0 GEMV가 넘어야 할 기준은 약 180–190 GB/s이고, 개선 여지는 작다. cutile 구현의 목표는 "더 빠르게"보다 "같은 성능을 safe 코드로"가 현실적이다. [추정]
- ds4 `matvec_q8`의 평균 대역폭이 낮은 것은 커널 자체보다는 꼬리 지연 때문으로 보인다. 원인은 확인하지 않았다. [추정]
- K가 큰 shape(`gdn_out` K=6144, `qwen3_4b_down` K=9728)와 작은 shape(`shexp_up`)의 효율이 상대적으로 낮다(81–88%). [확인]

## 7. 실패와 미해결 사항

| 항목 | 내용 |
|---|---|
| nsys 커널 시간 | 21개 설정 모두 nsys report에 CUDA kernel과 API 데이터가 없었다. 첫 시도에서는 `nsys stats`에 존재하지 않는 `-q` 옵션을 넘겨 usage만 출력됐다. 옵션을 고친 뒤 다시 보니 report 자체에 CUDA trace가 없었다. forward-compat libcuda와 nsys(CUPTI) 조합 문제로 추정하지만 확인하지 않았다. [확인: 데이터 없음 / 미확인: 원인] 그래서 mmvq 경로의 커널 구성(활성값 Q8_1 양자화와 GEMV 커널의 시간 분할)은 측정하지 못했다. |
| dmmv bf16 | 경로가 f32 입력만 지원한다. [확인] |
| CUDA 13.0 빌드 | 호스트 드라이버에서 돌리는 방법이지만 시도하지 않았다. |
| 반복 측정 | 설정당 1회만 측정했다. |
| GB10 L2 크기 | 확인하지 않았다. L2 회피 설계(512 MiB 회전)는 L2가 이보다 훨씬 작다는 가정에 기반한다. [추정] |

## 8. 재현 절차

### 빌드 pod (GPU 요청 없음)

- 이미지 `nvidia/cuda:13.3.1-devel-ubuntu24.04`
- hostPath `/var/lib/sam-candle` → `/work`, `/var/lib/sam-cutile` → `/cutile`
- 환경 변수: `CARGO_HOME=/work/cargo`, `RUSTUP_HOME=/cutile/rustup`, `RUSTUP_TOOLCHAIN=1.98.0`, `CARGO_TARGET_DIR=/work/target`, `CUDA_COMPUTE_CAP=121`, `CUDA_HOME=CUDA_TOOLKIT_PATH=/usr/local/cuda`

```bash
apt-get install -y git curl ca-certificates pkg-config libssl-dev build-essential cuda-tileiras-13-3 clang libclang-dev
git clone https://github.com/huggingface/candle.git /work/candle
git -C /work/candle checkout 5ba5d5b468b5b1df40e82dd3d556987bedeea041
cd /work/q8bench                          # Cargo.toml: candle-core = { path = "../candle/candle-core", features = ["cuda"] }
export PATH=/cutile/cargo/bin:/usr/local/cuda/bin:$PATH
cargo build --release                     # cuda feature
cargo build --release --features cutile   # cutile feature (libclang 필요)
```

### GPU pod

빌드 pod에 `runtimeClassName: nvidia`, `nvidia.com/gpu: 1`, `NVIDIA_DRIVER_CAPABILITIES=compute,utility`를 추가한다. compat libcuda를 그대로 둔다(이미지 기본값).

```bash
/work/q8bench-cuda SHAPE (mmvq|dmmv) (bf16|f32) [ITERS=200]
bash /work/run-gpu.sh      # 7 shape × 2 경로 × 2 dtype, 결과는 /work/results/wall.csv
```

## 9. 산출물 (Spark `/var/lib/sam-candle`)

| 경로 | 내용 |
|---|---|
| `candle/` | `5ba5d5b` checkout (수정 없음) |
| `q8bench/` | bench 크레이트 (`src/main.rs`, `Cargo.toml`) |
| `q8bench-cuda` | `cuda` feature로 빌드한 bench 바이너리 |
| `target/` | 빌드 결과 (`cutile` feature 빌드 포함) |
| `run-gpu.sh` | 전체 측정 스크립트 |
| `results/wall.csv` | 28개 설정의 µs/call, GB/s, 오차 |
| `results/*.err` | 설정별 stderr (dmmv bf16 에러 포함) |
| `results/nsys/` | nsys report (CUDA trace 없음) |
| `build-cuda.log`, `build-incr.log`, `build-cutile.log`, `build-cutile2.log` | 빌드 로그 |

작업 후 `candle-build-spark`, `candle-gpu-spark` pod는 모두 삭제했고, 노드의 GPU 할당은 0/1로 돌아갔다. [확인]
