# Grout

- 원본: https://github.com/huggingface/grout — 커밋 8개, 마지막 커밋 2026-07-01. 작성자는 Melih Elibol(cuTile-rs 핵심 기여자, 커밋 132개)과 Eric Buehler 둘뿐이다. [확인]
- DGX Spark tutorial이 쓰는 fork: `elibol/grout` `deb7427`. 이 커밋은 default branch가 아닌 branch에 있고 9/23에 push됐다. star 0, 외부 PR 0이다. [확인]
- 로컬 clone: `~/Repos/grout` (`23ae5a3`, huggingface 원본). 라이선스는 Apache-2.0이다. [확인]
- 성격: cuTile 논문의 end-to-end testbed다. 커뮤니티 프로젝트가 아니다. [확인]

## 규모와 범위

- `src/` 약 11.5k줄. 주요 파일: `model.rs`, `kernels.rs`, `flash_decode.rs`, `cublas.rs`(667줄), `loader.rs`. [확인]
- 모델: Qwen3 dense만(Qwen3-4B, Qwen3-32B). safetensors를 로드하고 fp16으로 계산한다. 양자화는 없다. [확인]
- **GEMM과 GEMV는 cuBLAS**다(`src/kernels.rs:9`). cuTile로 짠 것은 attention, norm, RoPE, embedding, argmax 등이다. [확인]
- cuTile entry 39개 중 unsafe 0개다. `&mut Tensor` partition을 44곳에서 쓴다. [확인]

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
- 확인할 것: `StepGraph`가 layer 종류가 섞인 hybrid 모델과 batch 폭을 담을 수 있는가.

## Spark 실측 (tutorial 재현)

Qwen3-4B, 호스트 580 드라이버. first run wall 26 s(JIT 약 7 s, 커널 37개), decode 23.73 t/s(42.1 ms/step), end-to-end 30.35 t/s였다. tutorial 수치(29.7 t/s)는 end-to-end 기준으로 보인다. [확인: 수치 / 추정: tutorial 기준] 자세한 내용은 [measurements/cutile-smoke-spark.md](measurements/cutile-smoke-spark.md)에 있다.

> 앞선 대화에서 "Grout decode가 floor의 약 90%이니 cuTile GEMV도 floor 근처까지 간다"고 추정했는데, 이는 틀렸다. Grout의 GEMV는 cuBLAS다. cuTile로 짠 양자화 GEMV가 GB10에서 어디까지 가는지는 아직 측정되지 않았다.
