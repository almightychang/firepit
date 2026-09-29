# 로드맵

갱신: 2026-09-30. ✔ 완료, ◐ 진행 중, ○ 미착수.

firepit은 엔진을 만드는 프로젝트이면서, LLM을 한 개념씩 이해하며 구현하는 공부 과정이기도 하다. 그래서 E2부터는 개념 하나를 한 덩이로 자른다.

## 전체

```
 0. 기반                                                          ✔
    ✔ 조사·설계 문서 (goals, design, engine)
    ✔ cuTile on Spark 동작 확인 · 개발 컨테이너
    ✔ safe Q8_0 GEMV v1 (RTX 5080에서 candle과 동급)
    ✔ cutile-rs fork: slice_mut (vendor 패치)

 E1. 엔진 골격 (합성 모델)                                        ✔
    ✔ Model / Session / Batch / Engine, 한 op 목록 → eager · CUDA Graph
    ✔ 세션 소유 state, tile 규칙, 출력 grid 일치
    ✔ parse 타입 (ModelDims, Engine::batch), poisoned, close_session, plan cache
    ✔ LayerDef 목록에서 가중치·state·plan 생성, unsafe 0 + 가드 테스트
    ✔ batch 1~4 (3은 4로 채움)

 E2. Qwen3.6-35B-A3B decode (실제 수학)                            ◐
    아래 "E2 공부 단위" 참고

 E3. prefill                                                      ○
    ○ GDN chunked (WY), attention prefill, prompt 처리 경로

 E4. 성능 (GB10)                                                  ○
    ○ Spark 측정 (GPU pod 승인 필요) · ds4 대비 decode t/s
    ○ dense 양자화 포맷 결정 (Q8_0 / FP8 / NVFP4)
    ○ GEMV 합치기 · quantize/RMSNorm fusion · hot path 할당 제거

 E5. 서비스                                                       ○
    ○ Server (대화 id → Session) · Scheduler (prefill/decode 섞기) · API
    ○ MTP (speculative)

 E6. Qwen3.8-Flash-Next                                           ○
    ○ HC mixer · QSA (indexer + sparse attention) · n-gram PLE (SSD)
    ○ ds4 layer 출력과 비교

 기여 트랙 (병행)
    ✔ cutile-rs #184 재현 · slice_mut 구현 (로컬 브랜치)
    ○ 이슈 001 게시 → PR (서명 커밋 설정 필요)
```

## E2 공부 단위

각 덩이는 같은 순서로 진행한다.

```
 ① 개념 설명 (TUI 다이어그램 1장, 수식 최소)   → 이해
 ② 작은 구현 (한 커널 또는 한 연결)
 ③ 정답 비교 (HF transformers의 같은 입력 중간값과 비교)   → 이해했다는 증거
```

| 덩이 | 배우는 것 | 끝났다는 기준 | 상태 |
|---|---|---|---|
| 2.0 정답 기준 만들기 | config 읽기, 가중치 이름과 모양, forward에서 흐르는 값 | HF로 layer별 중간값 덤프, Rust가 실제 가중치 일부 로드 | ○ |
| 2.1 입구와 출구 | token → embedding, 마지막 norm → LM head → logits | layer 0개 모델의 logits가 HF와 일치 | ○ |
| 2.2 RMSNorm | 정규화하는 이유, Qwen의 weight 형태 | 실제 가중치로 일치 | ◐ (scale 경로 있음) |
| 2.3 SwiGLU MLP | FFN의 gate · up · down | shared expert 하나가 일치 | ○ |
| 2.4 Attention ① | Q · K · V, RoPE, q/k norm | 한 token의 q, k가 일치 | ○ |
| 2.5 Attention ② | KV cache, GQA, softmax, output gate | attention layer 하나가 여러 step에서 일치 | ○ |
| 2.6 MoE | router, top-8 선택, expert 가중합 | MoE 블록 하나가 일치 | ○ |
| 2.7 GDN ① | causal conv1d, conv state | conv 출력 일치 | ○ |
| 2.8 GDN ② | gating(α, β), delta rule 상태 갱신, 출력 gate | GDN layer 하나가 여러 step에서 일치 | ○ |
| 2.9 전부 쌓기 | layer stack, tokenizer, greedy decode | 실제 문장 생성 | ○ |

순서는 쉬운 개념부터 쌓는다. 각 개념이 다음 개념의 전제가 된다. cuTile 표현력의 한계는 2.6(MoE, 간접 읽기)과 2.8(GDN)에서 만난다.

## 확인할 것

- 35B 모델은 RTX 5080(16 GB)에 통째로 올릴 수 없다. layer 한두 개의 가중치만 골라 로드해서 비교한다(safetensors는 tensor 단위로 읽을 수 있다).
- 같은 구조의 작은 공식 모델이 있으면 공부용으로 더 편하다. 아직 확인하지 않았다.
