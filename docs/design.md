# 설계 (초안)

작성: 2026-09-29. 아직 코드를 쓰기 전의 초안이다. **[결정]**은 합의된 것, **[열림]**은 측정이나 프로토타입으로 정할 것이다.

## 1. 핵심 아이디어: 가중치와 세션 state를 분리한다

Grout는 `Layer { weights, state }`로 가중치와 KV cache를 한 구조체에 묶는다(`src/model.rs:1203-1222`). 그래서 엔진 하나가 곧 세션 하나다. firepit은 이것을 둘로 나눈다. [결정]

```
Model    가중치. 모든 세션이 읽기만 한다          → &Model (Arc)
Session  한 대화의 state. 그 세션만 쓴다          → owned
Batch    한 step 동안 여러 세션을 빌려 온다        → &mut Session × N, 수명 = 한 step
```

- 가중치는 공유 읽기(`&`)라서 여러 세션이 동시에 써도 안전하다.
- 세션 state는 독점 소유다. 한 step 동안 `Batch`가 `&mut`로 빌려 가므로, 같은 세션을 두 batch에 넣는 코드는 borrow checker가 막는다.
- 이 소유권 구조가 커널 launch까지 이어지게 하는 것이 safe cuTile의 역할이다. 세션별 state 조각을 겹치지 않는 partition으로 넘긴다.

## 2. 개념

| 개념 | 가진 것 | 하는 일 |
|---|---|---|
| `Model` | 가중치(양자화 포함), config, tokenizer | 로딩, 불변 |
| `Session` | layer별 GDN recurrent state, conv state, KV cache, n-gram history, position | 한 대화의 진행 상태 |
| `Batch` | `&mut Session` 목록 | 한 step의 입력 묶음 |
| `Scheduler` | 대기 중인 세션들 | 매 step의 batch 구성 결정, prefill과 decode 섞기 |
| `Engine` | `Model`, 커널 registry, step graph cache | batch 하나를 한 step 진행 |

세션 수는 적다(에이전트 2~4개). 대규모 서빙용 스케줄러는 만들지 않는다. [결정]

## 3. step 실행

Grout의 `StepGraph`는 fork하지 않고 참고한다. Grout decode는 `StepGraph`를 쓰지 않고 손으로 나열한 CUDA Graph scope이며, state op에 세션 개념이 없다([grout.md](research/grout.md)). `TensorPool`의 value 수명 분석은 가져온다. [결정, 2026-09-29 변경]

한 step의 op는 두 종류로 나뉜다.

| 종류 | 예 | batch 처리 |
|---|---|---|
| 가중치 op | GEMV, MoE expert GEMV, router, output head | 여러 세션의 행을 모아 가중치를 **한 번만** 읽는다. batching 이득의 원천 |
| 세션 op | GDN scan, conv, attention(자기 KV), MTP 검증 | 세션별 state 조각에 대해 독립적으로 실행 |

- step graph는 `(모델, phase, batch 폭)`마다 한 번 만들고 캐시한다. [결정]
- decode는 CUDA Graph로 캡처한다. Grout의 `DecodeCudaGraphRunner`가 선례다. ds4 Qwen CUDA 경로는 CUDA Graph를 쓰지 않아 GPU idle이 토큰당 5.3 ms였다. [결정]

## 4. 세션 state 저장: pool과 간접 매핑 문제

세션 state는 고정 크기 slot pool에 둔다. 에이전트가 들어오고 나가면서 매 step의 batch 구성이 바뀌므로, "batch의 i번째 행이 pool의 몇 번 slot인가"가 데이터로 정해진다. **이 간접 쓰기는 cuTile의 safe partition으로 표현되지 않는다.** [열림]

| 선택지 | 방법 | 대가 |
|---|---|---|
| A. 쓰기 간접을 읽기 간접으로 뒤집기 | grid를 요청이 아니라 pool slot 기준으로 잡는다. block이 자기 slot을 소유하고 역매핑만 읽는다 | 쉬는 slot도 다시 써서 대역폭이 든다. pool 압축 전략이 필요하다 |
| B. unsafe 섬 하나 | gather/scatter 커널 하나만 unsafe로 두고, 나머지 계산은 contiguous buffer에서 safe로 한다 | 복사 비용이 든다. unsafe 경계가 하나 생긴다 |
| C. batch 구성이 바뀔 때만 재배치 | batch가 유지되는 동안 slot = batch 행으로 고정한다 | 구성 변경 시 재배치 비용이 든다. 에이전트 수가 적어서 드물 수 있다 |

세션 수가 적다는 점을 생각하면 C가 가장 단순할 수 있다. 셋 다 프로토타입으로 재 보고 정한다. 어느 쪽이든 막힌 지점은 cuTile 공백 보고의 재료다.

**2026-09-29 결정:** A/B/C 대신 D(세션이 state를 소유하고 세션 op을 세션마다 launch)를 채택했다. tile 규칙을 지키면 쌓기 방식보다 5~10% 느릴 뿐이고 unsafe와 cutile-rs 변경이 필요 없다. [engine.md](engine.md) 2~3절, [측정](research/measurements/session-ops-5080.md).

## 5. 커널

### 1단계: Qwen3.6-35B-A3B

| 커널 | 비고 | 정답 기준 |
|---|---|---|
| 양자화 dense GEMV | ds4 Qwen3.8 decode 시간의 약 50%. Grout는 cuBLAS에 위임하므로 cuTile로 새로 짠다 | CPU reference |
| MoE expert GEMV | router top-k로 expert를 골라 읽는다(읽기 간접이라 safe로 가능할 것으로 봄) | CPU reference |
| GDN decode step, chunked prefill | mistral.rs의 cuTile GDN prefill이 있으나 raw pointer로 짜여 있다 | HF transformers |
| gated attention, RoPE, RMSNorm | Grout 커널을 가져오거나 고친다 | HF transformers |
| MTP | speculative decoding | vLLM·SGLang MTP 코드, ds4 CPU reference(HF transformers는 두 모델 모두 `mtp.*` 가중치를 무시한다) |

### 2단계: Qwen3.8-Flash-Next 추가분

HC mixer(F16), QSA(indexer와 block-sparse attention), n-gram PLE(SSD에서 행 읽기). trunk 정답 기준은 HF transformers(v5.16.0부터 `qwen4_exp` 지원)의 op 단위 출력이다. ds4는 가중치 양자화가 달라 분포 지표(KL, top-1, ΔNLL)로만 비교한다.

### 양자화 포맷 [열림]

- 속도를 올리는 가장 큰 수단은 dense 가중치 bytes 줄이기다. Qwen3.8 Q2 GGUF에서도 dense는 Q8_0/F16이고 토큰당 5 GB를 차지한다. ([근거](research/measurements/ds4-qwen38-decode-profile.md))
- 후보: Q8_0(GGUF 호환), FP8, NVFP4(cuTile block-scaled MMA가 sm_121에서 정답 확인됨, 속도는 미측정).
- 짜기 전에 확인할 것: 34-byte GGUF block을 safe tile load로 읽을 수 있는가, int8 내적을 cuTile로 표현할 수 있는가.

## 6. 검증

- **정답:** layer 단위 중간값을 정답 기준 구현(HF transformers, ds4)에서 덤프해 비교한다. 최종 logit은 허용 오차 안에서 일치해야 한다.
- **성능:** 커널마다 "읽은 bytes ÷ 시간"을 floor(GB10 지속 대역폭 약 200 GB/s)와 비교한 효율로 본다. 측정 방법은 [ds4 profiling](research/measurements/ds4-qwen38-decode-profile.md)을 따른다.
- **안전성:** 세션 state 경로의 unsafe 개수를 기록하고, unsafe마다 이유를 남긴다.

## 7. 출발점: Grout fork

~~Grout(`~/Repos/grout`, Apache-2.0)를 fork해서 시작한다.~~ fork하지 않는다. 참고와 부분 차용만 한다. [결정, 2026-09-29 변경]

| 가져올 것 | 바꿀 것 |
|---|---|
| `StepGraph`, `TensorPool` 계획 | `Layer`에서 state를 떼어 `Session`으로 |
| `DecodeCudaGraphRunner` | 단일 세션 → batch 폭별 graph |
| cuTile 커널(attention, norm, RoPE, argmax 등) | GEMV/GEMM을 cuBLAS 대신 cuTile 양자화 커널로 |
| safe cuTile host/커널 작성 방식 | safetensors 로더 → GGUF 로더 추가 |

확인할 것: `StepGraph`가 layer 종류가 섞인 hybrid 모델(GDN + attention)과 batch 폭을 무리 없이 담을 수 있는가.

## 8. 환경 제약

- Spark는 SSH가 안 되고 microk8s Job으로만 실행한다. GPU pod는 Sam의 승인을 받고 띄운다.
- cuTile은 네이티브 cubin을 만들어서 호스트 580 드라이버에서 그대로 동작한다. CUDA 13.3으로 빌드한 PTX(candle 커널 등)는 580 드라이버에서 로드되지 않는다. ([근거](research/measurements/candle-q8-baseline.md))
- 커널당 JIT 시간은 45~660 ms다. 디스크 캐시를 쓰고, 서버 시작 시 warmup한다.

## 9. 열린 질문

1. ~~세션 state 간접 매핑은 A, B, C 중 무엇으로 할까~~ D로 결정 ([engine.md](engine.md))
2. dense 가중치 양자화 포맷 (5절)
3. ~~Grout `StepGraph`가 hybrid 모델과 batch를 담을 수 있는가~~ 답: 담을 수 없다. 참고만 한다
4. cuTile로 짠 양자화 GEMV가 GB10 floor의 몇 %까지 가는가. RTX 5080에서는 safe v1이 B = 1에서 candle과 동급(peak 89%)이다([측정](research/measurements/q8-gemv-5080.md)). GB10은 미측정
5. MTP와 batching을 함께 쓸 때 세션별 accept 길이가 다르면 step을 어떻게 맞출까
