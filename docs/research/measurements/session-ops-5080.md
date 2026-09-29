# 세션 op launch 방식 비교 on RTX 5080 (2026-09-29)

표기: **[확인]** = 실측. **[추정]** = 계산이나 간접 근거.

설계안 D([engine.md](../../engine.md) 2절)는 세션 state를 세션이 소유하고 세션 op을 세션마다 launch한다. cutile-rs에 기능(graph 병렬 가지, tensor 목록 인자)을 추가해야 하는지 판단하려고, 그 기능이 만들 GPU 동작을 지금 API로 흉내 내서 쟀다.

## 요약

- 세션 op이 느린 주원인은 launch 수가 아니라 **grid가 SM 수보다 작은 것**이었다. tile만 잘게 나눠도 2.9배 빨라졌다. [확인]
- tile을 나눈 D(세션별 launch)는 모든 세션을 한 tensor에 쌓아 1회 launch하는 방식보다 5~10% 느리다. [확인]
- 추가 launch 하나의 잔차는 2~4 µs다. [확인: 잔차 / 추정: 순수 launch 비용으로 해석]
- 결론: D를 채택하고, cutile-rs 변경은 지금 필요 없다.

## 조건

RTX 5080, `firepit-dev:cuda13.3`, cutile 0.4.0. 커널은 `S ← 0.5·S + k vᵀ`, state는 세션·layer마다 `[32, 128, 128] f32`(2 MiB), layer 30개. CUDA Graph replay를 8회 warmup 후 30회 CUDA event 중앙값으로 쟀다. 모든 변형에서 전 세션·전 layer state가 일치했다. [확인]

| 변형 | state 배치 | 행 tile | 세션당 grid | graph의 커널 수 |
|---|---|---:|---:|---:|
| V1 | 세션별 tensor, 일직선 graph | 128 | 32 | 30·B |
| V1′ | 같음 | 32 | 128 | 30·B |
| V1″ | 같음 | 16 | 256 | 30·B |
| V2 | 세션별 tensor, 세션마다 stream fork/join | 128 | 32 | 30·B |
| V3 | layer별로 쌓은 `[B·32, 128, 128]` | 128 | 32·B | 30 |
| V3′ | 같음 | 32 | 128·B | 30 |

## 결과 (µs)

| B | V1 | V1′ | V1″ | V2 | V3 | V3′ | 이론 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 209.5 | 88.9 | 98.7 | 265.4 | 262.8 | 88.5 | 131.1 |
| 2 | 917.8 | 409.2 | 322.0 | 525.9 | 522.6 | 292.4 | 262.1 |
| 4 | 1838.7 | 833.6 | 641.6 | 795.0 | 991.9 | 602.5 | 524.3 |
| 8 | 3674.5 | 1637.3 | 1272.0 | 1585.2 | 1898.8 | 1209.1 | 1048.6 |

- 이론값은 `4 MiB × B × 30 ÷ 960 GB/s`. B = 1이 이론보다 빠른 것은 state 60 MiB가 L2에 남기 때문으로 본다. [추정]
- V1″과 비교한 V3′은 행 32 기준이다. 행 16으로 쌓은 변형은 재지 않아서 D의 손해 5~10%는 약간 과소 추정일 수 있다.

### 추가 launch당 잔차 (V1′ − V3′, 같은 tile)

| B | 차이 µs | 추가 launch | µs / launch |
|---:|---:|---:|---:|
| 2 | 116.8 | 30 | 3.89 |
| 4 | 231.1 | 90 | 2.57 |
| 8 | 428.2 | 210 | 2.04 |

행 128 tile에서 같은 계산을 하면 8.5~13.2 µs다. 그 차이는 작은 grid의 낮은 병렬성에서 온다. [추정]

### graph 캡처 시간 (µs, JIT 제외)

| B | V1 | V1″ | V3′ |
|---:|---:|---:|---:|
| 1 | 136 | 96 | 82 |
| 8 | 632 | 604 | 81 |

세션 구성이 바뀔 때마다 재캡처해도 1 ms 미만이다. [확인]

## 선례 조사 (원문 확인분)

- graph 안 multi-stream: PyTorch가 공식 지원, SGLang(`alt_stream`, capture 중), llama.cpp(Q/K/V와 GDN 층, RTX 5090 decode +5~9%)가 사용한다. 선례는 모두 가지 2~3개다.
- tensor 목록 인자: PyTorch `multi_tensor_apply`, **cuTile Python의 list-of-arrays 인자**가 있다.
- 세션 state의 업계 표준은 pool + index(vLLM `block_table`, GDN `state_indices`)다. 간접 쓰기라 safe cuTile로는 표현되지 않는다.

## 재현

```bash
scripts/dev.sh session-bench
```

`crates/firepit-kernels/benches/session_ops.rs`. V2의 multi-stream 캡처는 `CudaGraph::scope`가 단일 stream이라 bench 안에서만 `cuda_core::sys`와 명시적 `ExecutionContext`로 구현했다(host 코드 unsafe). 커널은 모두 safe cuTile이다.
