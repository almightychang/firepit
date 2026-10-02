# E2.9 설계 초안: 실제 layer를 엔진에 통합

상태: 확정 (2026-10-02, Sam 승인: 결정 5개 모두 추천안). 구현 중.

## 1. 목표와 범위

E1 엔진(합성 모델과 synthetic 커널)을 실제 Qwen3.6 수학으로 바꾼다. 커널은 E2.1–2.8에서 HF와 op 단위로 맞췄다. 2.9는 그 위의 엔진 층을 바꾼다.

- 2.9a: 모델 로드, layer 조립, 세션 state, plan, 엔진 테스트 이전, prefix 4 layer 끝 출력(logits) HF 비교 (5080)
- 2.9b: tokenizer와 greedy 생성 CLI
- 2.9c: Spark(GB10)에서 전체 모델로 문장 생성

E1의 원칙은 그대로 둔다: 설계 D(가중치 op는 batch 전체에 한 번, 세션 op는 세션마다 launch), tile 규칙, 출력 grid 일치, 세션 소유 state, unsafe 0.

## 2. 전체 구조

```
 config.json ──parse──► ModelDims { hidden, vocab, layers: [LayerKind], Attention/Rope/Moe/GdnDims }
                                   │
 checkpoint ──load──► Model { embed, layers: Vec<LayerWeights>, final_norm, lm_head }
                                   │
                     StepPlan::new(model, batch) ──► Vec<Op>   (커널 하나 = op 하나)
                                   │
                    ┌──────────────┴──────────────┐
                 Eager Sink                    Graph Sink      ← 같은 op 목록
                                   │
 Session { state: Vec<LayerState> }   ← 세션 op가 자기 state만 빌림
```

## 3. 결정할 것

| # | 결정 | 선택지 | 추천과 이유 |
|---|---|---|---|
| 1 | layer 표현 | (a) 평평한 op 목록 하나 (E1 방식 확장) (b) layer 종류별 객체가 각자 실행 | (a). op 목록이 데이터라 eager와 graph가 같은 목록을 쓴다. layer 종류는 목록을 만들 때만 쓰인다. |
| 2 | op 단위 | (a) 커널 하나 = op 하나 (b) 의미 단위 op(`GdnMixer` 등)가 내부에서 여러 커널 실행 | (a). 엔진 op가 E2에서 검증한 칸과 일대일로 대응한다. |
| 3 | 중간 버퍼 | (a) 이름 붙은 slot, 종류별 하나, layer끼리 재사용 (b) 수명 분석으로 공유 | (a). batch 4 기준 수 MB. slot 모양은 config에서 계산한다. |
| 4 | layer 가중치 | (a) 종류별 구조체 `GdnWeights`/`AttentionWeights`/`MoeWeights`, op는 `(layer, 이름)`으로 참조 (b) 번호 기반 `WeightRef` | (a). 번호로 순서를 맞추는 구조(이전 리뷰 지적)를 피한다. |
| 5 | MoE batch | (a) expert 단계는 토큰마다 launch (b) B×k 행을 한 번에 (입력 행 매핑 필요) | (a). 지금 커널을 그대로 쓴다. 묶기는 E4. |

## 4. layer별 op 목록 (B = batch, ● = 세션마다 launch)

```
GDN layer
  rms_norm(input_layernorm)            Residual → Normed            [B, H]
  gemv qkv / z / b / a                 Normed → Qkv, Z, B, A        [B, 8192] [B, 4096] [B, 32] [B, 32]
● conv1d_silu  (conv state)            Qkv 행 b → Conv 행 b         [1, 8192]
● gdn_step     (S)                     Conv·B·A 행 b → Mixer 행 b   [32, 128]
  rms_norm 행 + silu gate (gated norm) Mixer, Z → Gated             [B·32, 128]
  gemv out_proj                        Gated → Delta                [B, H]
  add                                  Residual + Delta → Residual'

Attention layer
  rms_norm(input_layernorm)            Residual → Normed
  gemv q(+gate) / k / v                Normed → QGate, K, V         [B·16, 512] [B·2, 256] [B·2, 256]
  rms_norm 행 (q_norm, k_norm)         QGate 의 q 열, K             [B·16, 256] [B·2, 256]
  rope (행별 위치)                      q, k
● cache_append (k, v cache)            K·V 행 b → 세션 cache
● online_attention                     q 행 b + 세션 cache → Mixer 행 b   [16, 256]
  output_gate                          Mixer, gate 열 → Gated
  gemv o_proj                          Gated → Delta
  add

MoE (모든 layer)
  rms_norm(post_attention_layernorm)   Residual → Normed
  gemv router                          Normed → Router              [B, 256]
  top_k                                Router → ExpertSelection     토큰별
  ◆ expert_gemv gate_up                Normed 행 b → ExpertUp       [8, 1024]   ◆ = 토큰마다
  ◆ swiglu                             ExpertUp → ExpertAct         [8, 512]
  ◆ expert_gemv down                   ExpertAct → ExpertOut        [8, H]
  gemv shared gate_up → swiglu → gemv shared down     Normed → Shared  [B, H]
  gemv shared_expert_gate (rows 1)     Normed → SharedGate          [B, 1]
  ◆ combine                            ExpertOut, Shared, SharedGate → Delta 행 b
  add

Final
  rms_norm(final) → gemv lm_head → Logits [B, vocab]
```

숫자는 Qwen3.6 값이고 모두 config에서 계산한다.

## 5. 세션 state

```
LayerState::Gdn       { conv: [8192, 3], recurrent: [32, 128, 128] }   f32
LayerState::Attention { k: [max_seq, 2, 256], v: [max_seq, 2, 256] }   f32
```

- `max_seq`는 모델 config가 아니라 `LoadOptions`의 런타임 선택이다. config의 262144를 쓰면 세션당 KV cache가 10 GB를 넘는다.
- 세션당 크기 (max_seq 4096): GDN 30층 × 2.1 MB + attention 10층 × 16.8 MB ≈ 230 MB.

## 6. 모델 로드

- `ModelConfig: TryFrom<config.json의 text_config>` → `ModelDims` 안에 AttentionDims, RopeDims, MoeDims, GdnDims를 품는다. layer 종류는 `layer_types`.
- `Model::from_checkpoint(dir, LoadOptions { max_seq, layers: Option<usize> })`. `layers`는 앞쪽 N layer만 로드하는 prefix (5080 검증용).
- 형식은 bf16만. Q8 로드(ISQ)는 E4.
- norm 규약: layer norm, final norm, q/k norm은 로드할 때 `1 + w`. GDN gated norm은 `w` 그대로.
- 가중치 이름 규칙과 로더는 reference 테스트와 공용으로 둔다.

## 7. graph 캡처와 매 step 업로드

- 매 step host → device: 토큰, 세션별 위치, RoPE cos/sin 행 [B, head_dim]. 모두 고정 버퍼에 써서 graph replay와 호환된다.
- expert 선택은 GPU 안에서 끝난다(top_k → ExpertSelection). host 동기화가 없다.
- batch 모양이 바뀌면(세션 수) plan을 다시 만든다. E1과 같다.

## 8. 테스트

- E1 synthetic 커널과 합성 수학은 삭제한다.
- 엔진 테스트(batch 동등성, 세션 격리, graph = eager, 재캡처, 세션 교체, trybuild)는 **실제 구조의 작은 무작위 모델**로 옮긴다. GDN과 attention layer를 모두 포함하고 hidden, head, expert 수는 작게 한다. 같은 parse 경로를 지난다.
- 끝 출력 비교(새, ignored): prefix 4 layer 실제 가중치, 정답지 토큰 5개 → 매 step logits를 HF prefix 모델의 진짜 final norm + lm_head 결과와 비교한다. top-1 5개 일치. 대조: layer 순서를 바꾸면 실패.
- 기존 op 단위 reference 테스트는 유지한다.

## 9. 열린 질문

- slot 재사용 범위: GDN의 Mixer [B, 4096]와 attention의 Mixer [B·16, 256]는 크기가 같지만 우연이다. slot 하나를 최대 크기로 잡고 view만 다르게 할지, 종류별로 둘지.
- MoE의 토큰별 launch는 batch 4에서 커널 수가 늘어난다(layer당 약 20개). 정확도와 무관하므로 E4에서 측정 후 결정.
- prefix 4 layer의 5080 메모리: expert 4 × 1.5 GiB + embed·lm_head 2 GiB + 기타 ≈ 9 GiB 추정 (미측정).
