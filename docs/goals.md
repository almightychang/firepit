# 목표

작성: 2026-09-29. 근거가 되는 조사와 측정은 [research/](research/README.md)에 있다.

## 한 줄

개인 박스 한 대를 여러 에이전트가 동시에 나눠 쓰고, 서로의 상태를 침범할 수 없다는 것을 컴파일러가 보장하는 로컬 추론 엔진.

## 포지셔닝

| 프로젝트 | 언어 | 포지션 |
|---|---|---|
| llama.cpp | C++ | 모든 GGUF를 어디서나 돌리는 범용 runner |
| vLLM, SGLang | Python | 데이터센터 서빙, 처리량 극대화 |
| ds4 (DwarfStar) | C/CUDA/Metal | 소수 모델을 개인 하드웨어에서 가장 빠르게 |
| ds4-dfm-rs | Rust host + C/CUDA | ds4를 Rust host로 감싸 모델 계열을 넓힘 |
| mistral.rs | Rust (candle) | 모든 모델과 기능을 하나의 Rust 엔진에 |
| candle | Rust | 가벼운 ML 프레임워크 |
| Grout | Rust (cuTile) | cuTile 논문 testbed, Qwen3 dense, 단일 요청 |
| **firepit** | **Rust (cuTile)** | **개인 박스의 multi-agent 서빙, 세션 격리를 소유권으로 보장** |

## 목표

1. **세션 격리.** 세션의 state(GDN recurrent state, KV cache)는 그 세션만 소유한다. 한 세션의 state를 두 batch에 동시에 넘기는 코드는 컴파일되지 않는다.
2. **safe cuTile.** host부터 커널까지 소유권 보장이 끊기지 않게 짠다. unsafe는 명시적인 경계 안에만 두고, 경계마다 이유를 적는다.
3. **multi-agent batching.** 에이전트 여러 개가 가중치 읽기를 공유한다. ds4 CUDA 경로에는 없는 기능이다.
4. **cuTile 공백 기여.** 구현하다 safe로 표현되지 않는 지점을 만나면 기록하고 cuTile-rs upstream에 issue나 PR로 올린다. 이 프로젝트의 가장 중요한 부산물이다.
5. **실사용.** Sam이 Spark에서 에이전트 두 개 이상을 이 엔진으로 동시에 돌린다.

## 성공 기준

| 항목 | 기준 |
|---|---|
| 정확성 | Qwen3.6-35B-A3B는 HF transformers와, Qwen3.8-Flash-Next는 HF transformers와 op 단위로 허용 오차 안에서 일치하고 ds4와 분포 지표가 기준선 안에 든다 |
| 단일 세션 decode 속도 | 같은 모델과 양자화에서 ds4와 동급 |
| 다중 세션 처리량 | 세션 2개일 때 전체 처리량이 단일 세션보다 뚜렷하게 높음 |
| 안전성 | 세션 state에 접근하는 경로에 unsafe가 없음 |
| 기여 | cuTile-rs에 공백 보고나 PR 1건 이상 |

## 비목표

- **ds4보다 빠른 엔진.** GB10 decode는 메모리 대역폭에 묶여 있어서, 커널 언어로는 상한을 올릴 수 없다. 속도는 넘어야 할 기준선일 뿐 정체성이 아니다. ([근거](research/measurements/ds4-qwen38-decode-profile.md))
- **범용 GGUF runner.** 지원 모델은 좁게 유지한다.
- **NVIDIA 밖의 하드웨어.** cuTile은 NVIDIA 전용이다.
- **candle 통합.** candle은 리뷰 병목이 커서 큰 기여 대상이 아니다. cuTile 버전 bump 같은 작은 PR만 필요할 때 올린다. ([근거](research/candle.md))

## 대상

- 하드웨어: DGX Spark (GB10, sm_121, 128 GB unified memory)
- 모델 1단계: Qwen3.6-35B-A3B (`qwen3_5_moe`, GDN + MoE + MTP)
- 모델 2단계: Qwen3.8-Flash-Next (`qwen4_exp`, 1단계 구조에 HC, QSA, n-gram PLE 추가)
