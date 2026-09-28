# cutile-rs smoke test on DGX Spark (GB10, sm_121)

표기: **[확인]** = 실측, 또는 원문·코드로 확인. **[추정]** = 계산이나 간접 근거. **[미확인]** = 원인이나 사실을 확인하지 못함.

## 결론

- cutile-rs v0.4.0은 Spark의 **호스트 580 드라이버(CUDA 13.0)만으로** 빌드와 실행이 모두 된다. forward compatibility(`cuda-compat`)는 필요 없다. [확인]
- 예제 24개가 모두 rc=0으로 끝났다. 정답 검증 코드가 있는 예제는 모두 통과했다. `pdl`은 Tile IR 13.4가 필요해서 스스로 SKIP한다. [확인]
- DGX Spark tutorial(Grout, Qwen3-4B)을 재현했다. 출력이 정상이고 JIT 캐시 커널은 37개로 tutorial과 같다. decode 속도는 tutorial보다 약 20% 낮다. [확인: 수치 / 미확인: 차이 원인]

## 환경

| 항목 | 값 |
|---|---|
| 노드 | DGX Spark `spark-cb5e`, NVIDIA GB10, compute capability 12.1 [확인] |
| 호스트 드라이버 | 580.178.04, driver API 13000 [확인] |
| 이미지 | `nvidia/cuda:13.3.1-devel-ubuntu24.04` (arm64), nvcc 13.3.36 [확인] |
| tileiras | apt `cuda-tileiras-13-3` 13.3.36-1 [확인] |
| Rust | 1.98.0 (cutile-rs `rust-toolchain.toml`의 pin). Grout는 1.101.0-nightly (2026-09-27) [확인] |
| cutile-rs | `v0.4.0` = `cc720f1` (2026-09-25) [확인] |
| Grout | `elibol/grout` `deb7427` (tutorial이 고정한 commit) [확인] |

## 확인하려던 세 가지

### 1. 580 드라이버에서 CUDA 13.3과 Tile IR JIT가 동작하는가

동작한다. [확인]

- 호스트 libcuda만 로드되도록 만든 상태(driver API 13000, runtime 13030)에서 예제 24개와 Grout가 모두 정상 동작했다. [확인]
- tutorial은 "580 드라이버는 CUDA 13.0 드라이버이고, minor-version compatibility로 13.x toolkit이 동작하며, Tile IR assembler가 PTX가 아닌 네이티브 `sm_121` 코드를 내므로 새 드라이버가 필요 없다"고 설명한다. 실측이 이 설명과 맞는다. [원문 확인: `cutile-book/tutorials/12-dgx-spark-inference.md`]
- CHANGELOG에는 "DGX Spark support, verified with CUDA 13.3 and 13.4 on the 580 driver"라고 적혀 있다. [원문 확인]
- **주의:** `nvidia/cuda:13.3.1` 이미지는 forward-compat libcuda(`libcuda.so.610.43.02`)를 `/usr/local/cuda-13.3/compat`에 넣고, ldconfig에서 호스트 드라이버보다 먼저 등록한다. 이미지를 기본 상태로 쓰면 compat 드라이버 위에서 실행된다. 이때 `nvidia-smi`도 "CUDA Version: 13.3"을 표시한다. [확인]
- compat 드라이버와 호스트 드라이버 두 방식으로 모두 돌렸고, 결과와 JIT 시간은 오차 범위 안에서 같았다. [확인]

### 2. aarch64용 CUDA 13.3 devel 이미지를 구할 수 있는가

구할 수 있다. [확인]

- Docker Hub의 `nvidia/cuda:13.3.0-devel-ubuntu24.04`, `13.3.1-devel-ubuntu24.04`, `13.4.1-devel-ubuntu24.04`가 모두 arm64를 제공한다. [확인]
- 이미지에 tileiras는 없다. apt로 `cuda-tileiras-13-3`을 설치해야 한다. NVIDIA sbsa apt 저장소에 `cuda-tileiras-13-1`부터 `13-4`까지, 그리고 `cuda-compat-13-0`부터 `13-4`까지 있다. [확인]
- 이미지의 `NVIDIA_REQUIRE_CUDA`는 `cuda>=13.3`이다. 다만 brand 조건에 드라이버 535, 570, 580, 590, 595 범위가 들어 있어서, 580 드라이버에서도 pod가 정상적으로 시작됐다. [확인] GB10의 brand 값이 조건 목록 중 어느 것과 매칭됐는지는 확인하지 않았다. [미확인]

### 3. sm_121에서 예제 결과가 맞는가

맞다. 아래 "예제별 결과" 참고. [확인]

## 실행 절차 (재현용)

### 빌드 pod (GPU 요청 없음)

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: cutile-build-spark
  namespace: default
  labels: {app: sam-cutile}
spec:
  nodeSelector:
    kubernetes.io/hostname: spark-cb5e
  hostNetwork: true
  restartPolicy: Never
  containers:
    - name: work
      image: nvidia/cuda:13.3.1-devel-ubuntu24.04
      command: ["sleep", "infinity"]
      workingDir: /work
      env:
        - {name: CARGO_HOME, value: /work/cargo}
        - {name: RUSTUP_HOME, value: /work/rustup}
        - {name: CARGO_TARGET_DIR, value: /work/target}
      volumeMounts:
        - {name: work, mountPath: /work}
  volumes:
    - name: work
      hostPath: {path: /var/lib/sam-cutile, type: DirectoryOrCreate}
```

```bash
apt-get update
apt-get install -y cuda-tileiras-13-3 clang libclang-dev cmake build-essential pkg-config git curl ca-certificates

cd /work
git clone --branch v0.4.0 https://github.com/NVlabs/cutile-rs.git
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none --profile minimal
export PATH=/work/cargo/bin:/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda CUDA_TOOLKIT_PATH=/usr/local/cuda

cd /work/cutile-rs
cargo build --release -p cutile-examples --examples        # 1.98.0이 pin으로 자동 설치됨. 30 s, warning 0

cd /work
git clone https://github.com/elibol/grout.git
git -C grout checkout deb74274c731d4ca9e7f9b356a91892398498309
cd grout
rustup toolchain install nightly --profile minimal
rustup override set nightly
cargo build --release                                       # 31 s

apt-get install -y python3-venv python3-pip
python3 -m venv /work/venv && /work/venv/bin/pip install -U "huggingface_hub[cli]"
/work/venv/bin/hf download Qwen/Qwen3-4B --local-dir /work/hf_models/qwen3_4b   # 7.6 GB, 220 s
```

Grout의 `[patch.crates-io]`는 옆 디렉터리의 `../cutile-rs`를 참조한다. 그래서 두 저장소를 `/work` 아래에 나란히 둬야 한다. `.cargo/config.toml`의 `CUDA_TOOLKIT_PATH=/usr/local/cuda-13` 기본값은 환경 변수로 덮어쓴다. [확인]

### GPU pod

빌드 pod manifest에 다음을 추가한다. 이름은 `cutile-gpu-spark`로 바꾼다.

```yaml
spec:
  runtimeClassName: nvidia
  containers:
    - env:
        - {name: NVIDIA_DRIVER_CAPABILITIES, value: "compute,utility"}
      resources:
        limits: {nvidia.com/gpu: 1}
```

```bash
apt-get update && apt-get install -y cuda-tileiras-13-3 bc

# 호스트 580 드라이버만 쓰기 (compat 비활성화)
mv /usr/local/cuda-13.3/compat /usr/local/cuda-13.3/compat.off && ldconfig
ldconfig -p | grep libcuda.so.1      # => /lib/aarch64-linux-gnu/libcuda.so.1

# compat 드라이버로 되돌리기
mv /usr/local/cuda-13.3/compat.off /usr/local/cuda-13.3/compat && ldconfig
```

예제는 한 개씩 돌린다. 각 실행마다 새 `XDG_CACHE_HOME`을 주어 JIT disk cache를 비우고, `CUTILE_JIT_TIMING=1`로 JIT 단계별 시간을 기록한다.

```bash
export CUDA_HOME=/usr/local/cuda CUDA_TOOLKIT_PATH=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH CUTILE_JIT_TIMING=1
export XDG_CACHE_HOME=/tmp/jitcache-$MODE-$EX && rm -rf $XDG_CACHE_HOME
/work/target/release/examples/$EX
```

Grout:

```bash
export XDG_CACHE_HOME=/work/grout-jitcache   # first run 전에 비움, warm run은 재사용
cd /work/grout
/work/target/release/grout --model /work/hf_models/qwen3_4b \
  --prompt "Explain unified memory on DGX Spark in three sentences." \
  --max-new-tokens 200 --profile
```

## 예제별 결과

JIT 캐시 없이(cold) 한 번씩 실행했다. JIT 합계는 `CUTILE_JIT_TIMING`의 stage1~3 합계이고, tileiras는 그중 stage2다. [확인]

| 예제 | host rc | compat rc | 정답 확인 | JIT 커널 수 | host wall ms | host JIT 합계 ms | host tileiras ms | compat JIT 합계 ms |
|---|---|---|---|---:|---:|---:|---:|---:|
| async_add | 0 | 0 | 정상 종료 | 3 | 721 | 220 | 134 | 173 |
| async_and_then_example | 0 | 0 | 정상 종료 | 2 | 507 | 145 | 112 | 145 |
| async_mlp | 0 | 0 | 정상 종료 | 4 | 881 | 340 | 247 | 348 |
| async_saxpy | 0 | 0 | 정상 종료 | 4 | 742 | 259 | 193 | 288 |
| batch_matmul | 0 | 0 | assert 통과 (`|value - expected| <= 1e-3`) | 2 | 925 | 455 | 404 | 454 |
| cuda_graphs | 0 | 0 | "Validation: max abs diff = 0.00e0", 165.6 tok/s | 4 | 1145 | 389 | 292 | 389 |
| cuda_graphs_deviceop | 0 | 0 | 정상 종료 | 4 | 1194 | 393 | 298 | 433 |
| cudarc_interop | 0 | 0 | 정상 종료 | 1 | 548 | 67 | 45 | 68 |
| dropout | 0 | 0 | 정상 종료 | 2 | 616 | 122 | 91 | 149 |
| flash_attention_causal | 0 | 0 | non-causal, causal 모두 "check passed" (shape [2, 8, 64, 32]) | 3 | 1450 | 955 | 655 | 976 |
| gemm | 0 | 0 | 출력된 10개 모두 answer(8192)와 일치. assert 없음 | 2 | 7869 | 170 | 130 | 201 |
| gemm_static | 0 | 0 | 출력된 10개 모두 answer(8192)와 일치. assert 없음 | 2 | 7851 | 184 | 144 | 181 |
| hello_world | 0 | 0 | "Hello, I am program <0, 0, 0>…" 출력 | 1 | 517 | 65 | 51 | 81 |
| interop | 0 | 0 | 정상 종료 | 3 | 675 | 155 | 114 | 152 |
| jit_custom_store | 0 | 0 | 정상 종료 | 2 | 599 | 125 | 94 | 130 |
| jit_disk_cache | 0 | 0 | 정상 종료 | 2 | 602 | 137 | 106 | 144 |
| mxfp8 | 0 | 0 | "256 outputs matched", max_abs_err=0 | 2 | 964 | 497 | 417 | 492 |
| nvfp4 | 0 | 0 | "256 outputs matched", max_abs_err=0 | 2 | 1183 | 741 | 664 | 703 |
| pdl | 0 | 0 | **SKIP**: "requires Tile IR 13.4 or newer; selected Tile IR 13.3, target sm_121" | 0 | 149 | 0 | 0 | 0 |
| persistent_gemm | 0 | 0 | "4096 non-uniform outputs matched CPU reference", max_abs_err=1.9e-6 | 2 | 662 | 228 | 192 | 194 |
| rms_norm | 0 | 0 | **예제에 검증 코드 없음** (값만 출력) | 2 | 630 | 173 | 127 | 153 |
| saxpy | 0 | 0 | assert_eq 통과 | 2 | 620 | 142 | 108 | 138 |
| scale_ptr | 0 | 0 | 정상 종료 | 3 | 719 | 212 | 159 | 216 |
| softmax | 0 | 0 | 행 합 = 1, assert 통과 | 2 | 612 | 165 | 128 | 147 |

- "정상 종료"로 표시한 예제는 rc=0으로 끝났지만, 출력 내용을 개별적으로 검토하지는 않았다. [확인: rc / 미확인: 출력 정확성]
- 커널당 JIT 시간은 대략 45–660 ms이고, 대부분이 tileiras(stage2)에서 쓰인다. [확인]
- gemm과 gemm_static의 wall time은 약 7.9 s인데 JIT는 0.2 s 미만이다. 나머지 시간은 host 쪽 작업으로 보이지만 확인하지 않았다. [미확인]
- `flash_attention`, `async_gemm`, `autotune`은 필요한 feature(`reference-cpu`, `experimental-tune`)가 기본으로 꺼져 있어서 빌드 대상에서 빠졌다. `cutile-examples/Cargo.toml`에 따르면 `reference-cpu`는 candle의 `gemm-f16`이 aarch64 generic target에서 assemble되지 않아 기본으로 꺼져 있다. [확인: Cargo.toml 주석]

## Grout Qwen3-4B (tutorial 재현, 호스트 580 드라이버)

| 항목 | first run | warm run | tutorial 수치 |
|---|---|---|---|
| rc | 0 | 0 | — |
| 출력 | "Unified memory on DGX Spark is a technology that allows both the CPU and GPU to access the same pool of memory…" | first run과 같은 문장 | — |
| wall (model load 포함) | 26 s | 19 s | — |
| JIT | 약 7 s [추정: wall 차이], 캐시 파일 37개(2.2 MB) | 캐시 사용 | 11.7 s, 37 kernels |
| prompt | 511.75 t/s (24 tokens) | 500.55 t/s | — |
| decode phase | 23.73 t/s (평균 42.1 ms/step, 81 tokens에서 EOS) | 23.94 t/s (41.8 ms/step) | 29.7 / 29.5 t/s |
| end-to-end | 30.35 t/s | 30.60 t/s | — |

- 캐시 커널 수 37개는 tutorial과 같다. [확인]
- decode phase가 tutorial보다 약 20% 낮다. 원인은 확인하지 못했다. [미확인] 차이가 날 만한 요인은 두 가지다. tutorial은 CUDA 13.4를 썼고 이번에는 13.3이다. 그리고 tutorial의 "Decode 29.7 tokens/s"가 Grout 출력의 decode phase를 뜻하는지 end-to-end를 뜻하는지 문서에 명시되어 있지 않다. 이번 end-to-end 수치(30.35)는 tutorial 수치와 가깝다.
- tutorial이 쓴 Rust toolchain은 nightly다. 이번에도 Grout 디렉터리에 nightly override를 걸었다. Grout 소스에서 `feature(` 사용은 찾지 못했으므로, stable로도 빌드될 가능성이 있다. [추정]

## 막힌 지점과 해결

| 증상 | 원인 | 해결 |
|---|---|---|
| 호스트 드라이버 모드에서 `hello_world`가 SIGBUS(rc=135)로 종료 | cutile 문제가 아니라 테스트 방법이 잘못됐다. `LD_LIBRARY_PATH=/lib/aarch64-linux-gnu:…`로 호스트 libcuda를 강제했는데, nvcc로 만든 평범한 CUDA 13.3 프로그램도 같은 방식에서 SIGBUS가 났다. [확인] SIGBUS의 정확한 원인은 분석하지 않았다. [미확인] | compat 디렉터리를 옮기고 `ldconfig`를 다시 돌리는 방식으로 바꿨다. 이 상태에서는 같은 CUDA 프로그램이 driver=13000, runtime=13030으로 정상 동작했다. [확인] |
| `kubectl exec` 한 번이 "Internal error occurred: Timeout occurred"로 실패 | 명령이 실행되기 전에 실패했다. 대상 파일이 생성되지 않았다. pod와 노드는 정상이었다(uptime 10일, pod 재생성 없음). [확인] 원인은 모른다. [미확인] | 파일을 `kubectl cp`로 옮긴 뒤 다시 실행했다. |
| GB10에서 `dram__` 계열 ncu metric 없음 (ds4 profiling 때 발견, 이번 작업과는 무관) | — | — |

## 원래 brief와 달랐던 점

- brief에는 "stable Rust 1.89 이상"이라고 되어 있었다. 이는 README의 최소 요구사항이고, 저장소 자체는 `rust-toolchain.toml`로 **1.98.0**을 고정한다. [확인]
- DGX Spark tutorial은 CUDA **13.4**(`cuda-toolkit-13-4`, `cuda-tileiras-13-4`)와 nightly Rust를 쓴다. README가 권장하는 버전은 13.3이다. 이번에는 13.3으로 진행했다. [확인]

## 산출물 (Spark `/var/lib/sam-cutile`)

| 경로 | 내용 |
|---|---|
| `cutile-rs/` | v0.4.0 checkout |
| `grout/` | `deb7427` checkout (nightly override) |
| `target/release/examples/`, `target/release/grout` | 빌드 결과 |
| `cargo/`, `rustup/`, `venv/` | 도구 체인 |
| `hf_models/qwen3_4b/` | Qwen3-4B (7.6 GB) |
| `grout-jitcache/` | Grout JIT 캐시 (37개) |
| `runs/host.csv`, `runs/compat.csv` | 예제별 rc, wall, JIT 시간 |
| `runs/host/*.log`, `runs/compat/*.log` | 예제별 출력 |
| `runs/grout-first.log`, `runs/grout-warm.log` | Grout 출력 |
| `build-examples.log`, `prep-grout.log` | 빌드 로그 |
| `t.cu`, `t` | 드라이버 모드 확인용 CUDA 프로그램 |
| `run-all.sh` | 예제 일괄 실행 스크립트 |

두 pod(`cutile-build-spark`, `cutile-gpu-spark`)는 작업 직후 삭제했고, 노드의 GPU 할당은 0/1로 돌아갔다. [확인]
