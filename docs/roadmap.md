# 로드맵

갱신: 2026-10-03. ✔ 완료, ◐ 진행 중, ○ 미착수.

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

 E2. Qwen3.6-35B-A3B decode (실제 수학)                            ✔
    아래 "E2 공부 단위" 참고

 E3. prefill                                                      ✔
    아래 "E3 공부 단위" 참고. 동기: 5071토큰 prompt를 decode 경로로 215초 → GB10 5000토큰 4.97초

 E4. 성능 (GB10)                                                  ◐
    ✔ Spark 측정 → GDN conv 타일화로 decode 12 → 30.6 tok/s (대역폭 하한의 92%)
    ✔ 양자화 포맷 결정: NVIDIA ModelOpt 방향(2026-10-05). 3.6 기본 = NVIDIA NVFP4 혼합(expert NVFP4, 투영 FP8) + lm_head bf16
      → decode 43.7 tok/s, PPL +1.1% ([측정](research/measurements/nvfp4-gb10.md)). FP8 전체 41.2 tok/s +0.33%, NVFP4 그대로 48.1 tok/s +2.0%
    ✔ firepit CLI(models/pull/run) — 카탈로그 qwen3.6:nvfp4(기본)·fp8·bf16, pull이 lm_head bf16 조립까지 ([설계](design-cli.md))
    ○ run 첫 실행의 JIT 지연(prefill 5토큰 24.7 s) — 미리 컴파일하거나 진행 표시
    ○ FP8·NVFP4 GEMV 효율(대역폭 하한의 약 71%), NVFP4 prefill의 mmaf_scaled 경로
    ○ GEMV 합치기 · quantize/RMSNorm fusion · hot path 할당 제거

 E5. 서비스                                                       ○
    ○ Server (대화 id → Session) · Scheduler (prefill/decode 섞기) · API
    ○ MTP (speculative)

 E6. Qwen3.8-Flash-Next                                           ○
    ○ HC mixer · QSA (indexer + sparse attention) · n-gram PLE (SSD)
    ○ ds4 layer 출력과 비교

 E7. 하드웨어 aware 값 자동화                                       ○
    ○ GDN conv1d 채널 타일: 고정 타일 크기를 장치와 채널 수에 맞춰 선택
    ○ GDN delta rule value 열 타일: 현재 32열을 장치와 head 차원에 맞춰 선택
    ○ GEMV 행 타일(BN): DEFAULT_GEMV_TILE 고정값 → row_tile 같은 SM 채우기 규칙
    ○ 행이 적은 행렬(MoE expert 등)이 SM을 못 채우면 split-K
    ○ MoE grouped GEMM의 BM=32 기본값은 5080 T=5000 실험에서 BM=16보다 빠른 선택이다. expert별 행 수와 GPU에 맞춰 BM을 고르면 패딩과 CTA 수를 함께 줄일 수 있다.
    ○ MoE grouped GEMM의 X를 N block마다 다시 읽는 비용(게이트/업은 BN=128에서 8번)을 BN 확대나 한 block의 여러 N 타일로 줄이는 방안, cuTile gather load를 이용해 gather와 GEMM을 합치는 방안을 검토한다.
    ○ GEMV 열 타일(NB): batch·레지스터 압박에 따라 autotune 또는 규칙
    ○ TARGET_BLOCKS_PER_SM, MIN_ROW_TILE, MAX_BATCH_SESSIONS 근거 재검토
    ○ GDN prefill chunk 크기와 역행렬 대각 block 크기(현재 64/16)를 장치와 모델 형상에 따라 선택

FP8 GEMV의 임시 5080 타일 규칙은 batch 1에서 스케일 블록 안의 K 128개(`blocks=4`)를 처리하고, 선택된 expert 수를 포함한 출력 행을 약 64개 CTA에 나누는 2의 거듭제곱 행 타일(최대 128)을 쓴다. batch 2~4는 레지스터 사용량을 측정하지 않아 기존 행 타일을 유지한다. release `do_bench`에서 스케일 곱을 부분합 뒤로 옮긴 뒤 dense 8192×2048은 행 128에서 0.030592 ms(기존 16행 0.072992 ms), 4096×2048은 행 64에서 0.021664 ms(16행 0.045152 ms), 2048×4096은 행 32에서 0.029152 ms(16행 0.043808 ms), 512×2048은 행 8에서 0.016592 ms(16행 0.017344 ms)였다. 행 4/8/16/32에서는 K 32/64/128개를 비교했고 각 형상의 최선은 K 128개였다. 더 큰 행 64/128에서는 K 128개만 측정했다. 후속에 로컬 GPU가 다른 작업에도 사용 중임이 알려졌으므로 이 수치는 잠정값이다. expert와 GB10의 최적 타일은 아직 측정하지 않았으며 E7에서 장치별로 재선택한다.

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
| 2.7 GDN ① | causal conv1d, conv state | conv 출력 일치 | ✔ |
| 2.8 GDN ② | gating(α, β), delta rule 상태 갱신, 출력 gate | GDN layer 하나가 여러 step에서 일치 | ✔ |
| 2.9 전부 쌓기 | layer stack, tokenizer, greedy decode | 실제 문장 생성 | ✔ (GB10에서 전체 모델 생성, [기록](research/measurements/qwen36-first-generation-gb10.md)) |

순서는 쉬운 개념부터 쌓는다. 각 개념이 다음 개념의 전제가 된다. cuTile 표현력의 한계는 2.6(MoE, 간접 읽기)과 2.8(GDN)에서 만난다.

## E3 공부 단위

E2와 같은 순서(개념 → 구현 → 정답 비교)에 성능 기준을 더한다. prefill은 같은 수식을 다른 계산 순서로 하는 것이라 결과가 decode와 같아야 하고, 목적이 성능이라 얼마나 빨라졌는지도 끝났다는 기준에 들어간다. 성능 목표는 3.0에서 GB10의 실측 최대 처리량을 잰 뒤 "하한의 몇 %"로 정한다.

| 덩이 | 배우는 것 | 정확성 기준 | 성능 기준 | 상태 |
|---|---|---|---|---|
| 3.0 정답지와 입구 | prompt 전체를 한 번에 넣는 HF forward, prefill 입구, 계산 하한 | HF로 T토큰 중간값 기록 | GB10 GEMM 실측 최대 처리량(cuTile, PyTorch/cuBLAS) | ✔ |
| 3.1 GEMM | 행렬 × 행렬, tensor core, 정밀도 | projection이 HF와 일치 | 실측 최대 대비 % | ✔ (bf16 hi/lo 분할, GB10 7~19 TFLOPS: 메모리에 묶임, [측정](research/measurements/gemm-peak-gb10.md)) |
| 3.2 causal attention | flash attention, 미래 가리기, KV cache 채우기 | attention layer 일치 | 실측 최대 대비 % | ✔ |
| 3.3 GDN conv | 시퀀스 causal conv, 마지막 state | conv 출력과 state 일치 | 대역폭 대비 % | ✔ (GB10 5000토큰 layer당 2.35 ms, 대역폭의 약 70%) |
| 3.4 GDN chunk ① | chunk 안 계산, FLA 블록 삼각 역행렬 | f64 oracle·HF 입력·decode 합성 | 성분별 반올림 상한, GB10 5000토큰 layer당 6.1 ms | ✔ |
| 3.5 GDN chunk ② | chunk 사이 state 전달 | f64 oracle·decode 동치·HF 150토큰 GDN layer | RTX 5080 5000토큰 second phase 3.96 ms, 전체 chunk 5.74 ms | ✔ (HF GDN layer 끝-끝 일치, GB10 5000토큰 chunk 전체 layer당 21 ms) |
| 3.6 MoE prefill | expert별로 토큰 모으기 | MoE 블록 일치 | 실측 최대 대비 % | ✔ (host 안정 정렬·expert별 BM 패딩, f64 PBT·decode 동치·HF 150토큰 4 layer, RTX 5080 T=5000 gate_up 35.74 ms·down 13.33 ms) |
| 3.7 조립 | 엔진 prefill 경로 | prefill(T) = decode T번 (logits, state) | 5000토큰 prompt 215초 → 5초 이내 (계산 하한 약 0.4초, [기준선](research/measurements/gemm-peak-gb10.md)) | ✔ (GB10 4.97 s; 실제 모델 HF·prefill/step·이어 붙이기 top-1/top-5 일치, [검증](research/measurements/engine-verify-gb10.md)) |

E7 성능 단계에서 GEMM의 M·N·K와 GPU 특성에 따라 타일 크기를 자동 선택한다. E3.1은 호출자가 타일을 지정하며 기본값은 64×128×32이다.
E3.2 attention의 현재 기본 BM×BN은 16×32이다. RTX 5080의 head_dim 256 release cubin에서 255 registers/thread, 1008-byte stack/thread와 `STL` spill을 확인했다. 타일 자동 선택과 spill 저감은 E7에서 다룬다.
E3.3 conv prefill은 채널 타일별 block이 시간축을 순차 처리한다. f64 oracle, decode 반복, HF 150토큰 비교에서 출력과 마지막 state가 일치했다. 같은 GPU에서 다른 실험을 병행해 성능은 측정하지 않았다.
E3.5는 value head와 state 열 타일별 block이 chunk를 순서대로 처리한다. 현재 열 타일은 `row_tile`을 이용해 SM 수와 head 수에서 선택한다. RTX 5080 모델 형상에서 32열이며 release cubin은 255 registers/thread, 2368-byte stack/thread와 `STL` spill이 있다. 타일·spill 개선은 E7에서 한다.
E3.6의 행별 조립 커널은 RTX 5080 T=5000에서 255 registers/thread, 2240-byte stack/thread와 `STL`/`LDL` spill이 있었다. 분리형 실험에서 gather와 BM=32·BN=128·BK=32를 적용하자 같은 5080에서 gate/up 총시간이 36.38→6.27 ms, down이 13.56→2.72 ms로 줄었다. GB10의 [엔진 prefill 측정](research/measurements/gemm-peak-gb10.md)에서도 기존 grouped GEMM이 T=5000 시간의 57%를 차지했다. BM=32를 기본값으로 쓰되 GB10에서 패딩과 점유율을 재확인하고, 장치별 타일 선택은 E7에서 다룬다.

## 정답지 만들기 (2.0)

```bash
# 앞쪽 4개 layer에 필요한 shard 4개 (약 10.9 GB)만 ~/Models/qwen3.6-35b-a3b 에 받는다
cd tools/reference
uv run python trace_qwen36.py --model ~/Models/qwen3.6-35b-a3b --out ../../refs/qwen36-prefix4.safetensors
# E3.0: separate 150-token prefill reference and HF decode consistency check
uv run python trace_qwen36.py --model ~/Models/qwen3.6-35b-a3b --prefill --out ../../refs/qwen36-prefix4-prefill.safetensors
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
- 2.5: f32 KV cache와 online softmax, GQA, output gate, o_proj를 layer 3 HF 5 step과 비교했다. `o_proj` 입력과 attention Δx의 최대 상대오차는 각각 `1.50475377e-7`, `3.44580795e-7`. scale 누락과 modulo GQA 매핑의 상대오차는 각각 `6.86910900e-1`, `1.48213966e0`. E1 합성 `synthetic.rs`는 2.9a 엔진 통합에서 삭제했다.
- 2.6: router logits, top-k 번호·가중치, routed 합, MoE Δx를 layer 0~3의 5 step과 비교했다. 최대 상대오차는 logits `3.82721087e-7`, routed 합 `4.32246025e-7`, Δx `1.40249628e-6`; top-k 가중치 최대 절대오차 `1.49011612e-7`, 번호 불일치 0건. 재정규화 누락과 top-(k−1) 대조 상대오차는 `5.30466856e-1`, `7.22364042e-2`. `expert_gemv`는 기존 dense GEMV와 본문이 비슷하다. 이슈 002의 cuTile macro 문제 해결 후 중복 제거를 검토한다.
- 2.6 후속: `ExpertSelection`으로 device 번호와 가중치를 함께 보유한다. GPU top-k 또는 host 입력 검증으로만 생성하며 expert GEMV와 결합은 이 타입을 받는다.
- 2.7: GDN 입력 projection 4개는 공개 GEMV를 각각 호출한다. 채널별 causal conv+SiLU와 state 갱신은 한 커널에서 수행한다. state는 `state.load()`와 `state.store()`로 같은 block 안에서 제자리 갱신한다. HF cache는 `[1, conv_dim, kernel]`, 커널 state는 마지막 `kernel−1` 값인 `[conv_dim, kernel−1]`이다. layer 0~2 × 5 step에서 q/k/v와 state를 비교했다.
- 2.8: GDN delta rule은 `(value head, value 열 타일)`당 block 하나에서 q/k l2norm, β·g, state 감쇠·갱신·출력을 처리한다. 기존 RMSNorm 뒤에 SiLU gate를 적용하고 기존 GEMV로 `out_proj`를 계산한다. layer 0~2 × 5 step에서 core, recurrent state, gated norm, Δx를 HF와 비교했다.
- 2.9a: checkpoint 앞쪽 4개 layer(GDN 3, attention 1)를 실제 MoE와 함께 엔진에 연결했다. 5 step prefix logits의 HF 상대오차 최대 `5.68042530e-7`, top-1 5개 일치. graph 결과는 eager와 같았다. 5080에서 로드 전후 GPU 메모리는 `3459 → 13069 MiB`였다(`max_seq=5`, 계측 실행 기준).

## 확인할 것

- 35B 모델은 RTX 5080(16 GB)에 통째로 올릴 수 없다. layer 한두 개의 가중치만 골라 로드해서 비교한다(safetensors는 tensor 단위로 읽을 수 있다).
- 같은 구조의 작은 공식 모델이 있으면 공부용으로 더 편하다. 아직 확인하지 않았다.
