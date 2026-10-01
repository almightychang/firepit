# 로드맵

갱신: 2026-10-01. ✔ 완료, ◐ 진행 중, ○ 미착수.

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

 E7. 하드웨어 aware 값 자동화                                       ○
    ○ GEMV 행 타일(BN): DEFAULT_GEMV_TILE 고정값 → row_tile 같은 SM 채우기 규칙
    ○ 행이 적은 행렬(MoE expert 등)이 SM을 못 채우면 split-K
    ○ GEMV 열 타일(NB): batch·레지스터 압박에 따라 autotune 또는 규칙
    ○ TARGET_BLOCKS_PER_SM, MIN_ROW_TILE, MAX_BATCH_SESSIONS 근거 재검토

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
| 2.0 정답 기준 만들기 | config 읽기, 가중치 이름과 모양, forward에서 흐르는 값 | HF로 layer별 중간값 덤프, Rust가 실제 가중치 일부 로드 | ✔ |
| 2.1 입구와 출구 | token → embedding, 마지막 norm → LM head → logits | layer 0개 모델의 logits가 HF와 일치 | ✔ |
| 2.2 RMSNorm | 정규화하는 이유, Qwen의 weight 형태 | 실제 가중치로 일치 | ✔ (2.1의 final norm으로 확인) |
| 2.3 SwiGLU MLP | FFN의 gate · up · down | shared expert 하나가 일치 | ✔ |
| 2.4 Attention ① | Q · K · V, RoPE, q/k norm | 한 token의 q, k가 일치 | ✔ |
| 2.5 Attention ② | KV cache, GQA, softmax, output gate | attention layer 하나가 여러 step에서 일치 | ✔ |
| 2.6 MoE | router, top-8 선택, expert 가중합 | MoE 블록 하나가 일치 | ✔ |
| 2.7 GDN ① | causal conv1d, conv state | conv 출력 일치 | ○ |
| 2.8 GDN ② | gating(α, β), delta rule 상태 갱신, 출력 gate | GDN layer 하나가 여러 step에서 일치 | ○ |
| 2.9 전부 쌓기 | layer stack, tokenizer, greedy decode | 실제 문장 생성 | ○ |

순서는 쉬운 개념부터 쌓는다. 각 개념이 다음 개념의 전제가 된다. cuTile 표현력의 한계는 2.6(MoE, 간접 읽기)과 2.8(GDN)에서 만난다.

## 정답지 만들기 (2.0)

```bash
# 앞쪽 4개 layer에 필요한 shard 4개 (약 10.9 GB)만 ~/Models/qwen3.6-35b-a3b 에 받는다
cd tools/reference
uv run python trace_qwen36.py --model ~/Models/qwen3.6-35b-a3b --out ../../refs/qwen36-prefix4.safetensors
cd ../..
scripts/dev.sh test -p firepit --test reference -- --ignored
```

- 정답지는 앞쪽 N개 layer만 가진 모델을 token 하나씩 decode하며, 각 layer로 들어가는 residual stream(`step{t}/residual{i}`, i = 0..N)을 기록한다. `residual0`은 embedding, `residualN`은 마지막 layer의 출력이다. `step{t}/exit_norm`과 `step{t}/logits`는 원래 HF norm과 LM head를 `residual0`에 적용한 0층 모델의 출구값이다.
- 중간값은 transformers 자체 기능(`output_hidden_states`)으로 얻는다. MoE router logits도 같은 방식으로 얻을 수 있다.
- Rust는 공식 `safetensors` crate의 header 타입으로 파일을 해석하고 필요한 바이트 범위만 읽는다(mmap을 쓰지 않아 unsafe가 없다).
- 2.1 비교(5 step): exit_norm 최대 절대 오차 `3.81469727e-6`, logits `5.00679016e-6`; top-1 5개 모두 일치. 잘못된 scale(`w`)을 쓰면 exit_norm 최대 절대 오차 `2.56955013e1`.
- 2.1b: 합성 모델의 bf16/Q8 가중치 dispatch를 같은 op 목록에 연결했다. batch 1~4, eager/graph, 세션 교체 비교 통과; 두 형식 출력의 최대 절대 차이는 `2.384186e-7`.
- 2.3: 앞쪽 4개 layer의 shared expert 입력/출력을 HF hook으로 기록해 5 step × 4 layer를 비교했다. 최대 상대오차 `5.36579723e-6`, 실제 bf16 가중치와 Gaussian 입력의 f64 CPU 비교 `3.36419940e-7`. gate/up 순서를 뒤집으면 상대오차 `4.83147961e0`.
- 2.4: layer 3 attention의 q/k/v/gate를 HF 5 step과 비교했다. 최대 상대오차는 각각 `4.68418046e-7`, `6.65132815e-7`, `3.92343450e-7`, `4.70510518e-7`. 텍스트의 세 위치 축이 같을 때 mRoPE 표는 1D 표와 일치했고, f32 host 표의 최대 절대오차는 `5.96046448e-8`이다. 인접 성분 회전과 전체 256차원 회전의 상대오차는 각각 `4.24366914e-1`, `5.18970913e-1`.
- 2.5: f32 KV cache와 online softmax, GQA, output gate, o_proj를 layer 3 HF 5 step과 비교했다. `o_proj` 입력과 attention Δx의 최대 상대오차는 각각 `1.50475377e-7`, `3.44580795e-7`. scale 누락과 modulo GQA 매핑의 상대오차는 각각 `6.86910900e-1`, `1.48213966e0`. E1 합성 `kv_append`/`attn_step`은 유지하며 2.9 엔진 통합에서 실제 커널로 교체한다.
- 2.6: router logits, top-k 번호·가중치, routed 합, MoE Δx를 layer 0~3의 5 step과 비교했다. 최대 상대오차는 logits `3.82721087e-7`, routed 합 `4.32246025e-7`, Δx `1.40249628e-6`; top-k 가중치 최대 절대오차 `1.49011612e-7`, 번호 불일치 0건. 재정규화 누락과 top-(k−1) 대조 상대오차는 `5.30466856e-1`, `7.22364042e-2`. `expert_gemv`는 기존 dense GEMV와 본문이 비슷하다. 이슈 002의 cuTile macro 문제 해결 후 중복 제거를 검토한다.

## 확인할 것

- 35B 모델은 RTX 5080(16 GB)에 통째로 올릴 수 없다. layer 한두 개의 가중치만 골라 로드해서 비교한다(safetensors는 tensor 단위로 읽을 수 있다).
- 같은 구조의 작은 공식 모델이 있으면 공부용으로 더 편하다. 아직 확인하지 않았다.
