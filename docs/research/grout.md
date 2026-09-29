# Grout

- 원본: https://github.com/huggingface/grout — 커밋 8개, 마지막 커밋 2026-07-01. 작성자는 Melih Elibol(cuTile-rs 핵심 기여자, 커밋 132개)과 Eric Buehler 둘뿐이다. [확인]
- DGX Spark tutorial이 쓰는 fork: `elibol/grout` `deb7427`. 이 커밋은 default branch가 아닌 branch에 있고 9/23에 push됐다. star 0, 외부 PR 0이다. [확인]
- 로컬 clone: `~/Repos/grout` (`23ae5a3`, huggingface 원본). 라이선스는 Apache-2.0이다. [확인]
- 성격: cuTile 논문의 end-to-end testbed다. 커뮤니티 프로젝트가 아니다. [확인]

## 규모와 범위

- `src/` 약 11.5k줄. 주요 파일: `model.rs`, `kernels.rs`, `flash_decode.rs`, `cublas.rs`(667줄), `loader.rs`. [확인]
- 모델: Qwen3 dense만(Qwen3-4B, Qwen3-32B). safetensors를 로드하고 fp16으로 계산한다. 양자화는 없다. [확인]
- **GEMM과 GEMV는 cuBLAS**다(`src/kernels.rs:9`). cuTile로 짠 것은 attention, norm, RoPE, embedding, argmax 등이다. [확인]
- cuTile entry 39개 중 25개가 `unsafe fn`이고 `unchecked_accesses = true`도 25곳이다. `pointer_to_tile` 등 raw pointer를 쓰는 entry가 8개다. 논문 버전 커밋(`5fb58a5`)에서 `unchecked_accesses`가 4곳 → 25곳으로 늘었다. [확인] (이전 기록 "unsafe 0개"는 틀렸다)
- cutile 0.2.0에 고정되어 있다. 0.2.0에서는 `partition_mut`/`PartitionMut::store`가 unsafe였고, 0.3.0부터 safe(bounds check 삽입)다. Grout의 `partition_mut` unsafe 블록 상당수는 버전 차이다. [확인]

## 구조 (`src/model.rs`)

| 구조체 | 역할 | 위치 |
|---|---|---|
| `StepGraph` | op 목록(`GraphOp`), value spec, use count, 최종 value, `TensorPoolPlan` | :825 |
| `GraphOp` | EmbeddingBatch, MatMul, MatVec, Add 등 | :624 |
| `TensorPool` / `TensorPoolPlan` | value 수명 분석을 바탕으로 중간 텐서 메모리를 재사용 | :818, :984 |
| `DecodeCudaGraphRunner` | decode step을 CUDA Graph로 캡처해서 실행 | :323 |
| `Layer { weights, state }` | **가중치와 KV cache를 한 구조체에** 묶음 | :1219 |
| `LayerState` | `k_cache`, `v_cache` | :1214 |
| `Qwen3Engine` | layer 목록, `step_graph_cache`(sequence 길이 구간별), `decode_runner` | :1276 |

- KV cache를 "빌려주고 돌려받는" 코드가 있다(`lend_kv_caches_to_layers`, `reclaim_kv_caches_from_layers`). CUDA Graph runner와 layer 사이에서 소유권을 옮기는 방식이다. [확인]
- `step_graph_cache`는 sequence 길이 구간을 key로 쓴다. step graph를 prefill과 decode 구간별로 한 번씩 만든다. [확인]

## firepit 출발점으로서의 평가

| 가진 것 | 없는 것 |
|---|---|
| safe cuTile로 host와 커널을 짜는 방식 (cuTile 저자의 코드) | 양자화 |
| `StepGraph`, `TensorPool`, CUDA Graph decode | GEMV/GEMM의 cuTile 구현 |
| GB10에서 동작 확인 (Qwen3-4B, decode 23.7 t/s) | GDN, MoE, hybrid layer |
| KV cache 소유권을 옮기는 선례 | 세션 여러 개 (엔진 하나 = 세션 하나) |
| | 서버, API |

- 설계상 가장 큰 변경은 `Layer`에서 state를 떼어 내는 것이다. 자세한 내용은 [design.md](../design.md) 1절에 있다.
## `StepGraph` 정독 결과 (2026-09-29)

- decode는 `StepGraph`를 쓰지 않는다. `build_decode_graph_scope`(`model.rs:2248-3550`)가 `CudaGraph::scope` 안에서 커널을 직접 나열하고, prime pass와 capture pass가 같은 layer 루프를 두 번 적는다. README의 "StepGraph를 CUDA Graph로 캡처"는 주석으로 남은 옛 방식이다. [확인]
- `GraphOp`과 `LayerWeightSlot`은 Qwen3 전용 closed enum이고, dtype은 f16 하나다. [확인]
- state op는 `layer_idx`만 들고 실행기가 `self.layers[i].state`를 꺼낸다. position은 스칼라 하나다. 세션 개념이 없다. [확인]
- `TensorPool`/pool plan은 모델과 무관해서 재사용할 만하다. [확인]
- 결론: fork하지 않고 참고한다. firepit은 연산 목록 하나에서 prefill, prime, CUDA Graph 캡처를 모두 만들고, state op이 세션 state 핸들을 명시적으로 받게 한다. [결정]

## Spark 실측 (tutorial 재현)

Qwen3-4B, 호스트 580 드라이버. first run wall 26 s(JIT 약 7 s, 커널 37개), decode 23.73 t/s(42.1 ms/step), end-to-end 30.35 t/s였다. tutorial 수치(29.7 t/s)는 end-to-end 기준으로 보인다. [확인: 수치 / 추정: tutorial 기준] 자세한 내용은 [measurements/cutile-smoke-spark.md](measurements/cutile-smoke-spark.md)에 있다.

> 앞선 대화에서 "Grout decode가 floor의 약 90%이니 cuTile GEMV도 floor 근처까지 간다"고 추정했는데, 이는 틀렸다. Grout의 GEMV는 cuBLAS다. cuTile로 짠 양자화 GEMV가 GB10에서 어디까지 가는지는 아직 측정되지 않았다.
