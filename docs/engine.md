# 엔진 골격 설계

작성: 2026-09-29. [design.md](design.md) 1~4절을 구체화한다.

## 1. 목표

- 여러 세션이 한 step을 함께 진행하고, 가중치 op은 가중치를 한 번만 읽는다.
- 세션 state는 Rust 소유권으로 격리된다. 다른 세션의 state를 쓰는 코드는 컴파일되지 않는다.
- 한 step은 연산 목록 하나에서 eager 실행과 CUDA Graph 캡처를 모두 만든다. Grout처럼 같은 모델을 여러 곳에 손으로 적지 않는다.

## 2. 세션 state는 세션이 소유한다 [결정]

design.md 4절의 A/B/C는 state를 공용 pool에 두고 batch 행을 slot에 매핑하는 전제였다. 이를 D로 대체한다.

```
 D. state는 세션이 소유한 device tensor. 세션 op은 세션마다 launch한다.

   Session 0 ── owns ──► [GDN S, conv, KV] tensors (device)
   Session 1 ── owns ──► [GDN S, conv, KV] tensors (device)

   가중치 op : activation [B, H] 하나로 모아 한 번에 (가중치 1회 read)
   세션 op   : 세션 b마다 launch.  입력 = activation 행 b (&),  state = &mut session_b
```

근거 ([측정](research/measurements/session-ops-5080.md)):

- tile 규칙(3절)을 지키면 D는 모든 세션을 쌓아 1회 launch하는 방식보다 5~10% 느릴 뿐이다.
- unsafe가 없고 cutile-rs 변경이 필요 없다. 격리가 타입 수준에서 끝난다.
- 세션 구성이 바뀌면 CUDA Graph를 다시 캡처한다. B = 8에서도 1 ms 미만이다.

보류:

- 세션이 8개를 넘으면 launch 잔차(추가 launch당 2~4 µs)가 커진다. 그때 tensor 목록 인자를 cutile-rs에 제안한다(cuTile Python에 선례가 있다).
- graph 병렬 가지는 tile 규칙과 같은 효과라 필요 없다.

## 3. tile 규칙: launch 하나가 혼자서 GPU를 채운다 [결정]

cuTile에서 grid는 출력 partition 모양이 정한다. 컴파일러는 grid를 바꾸지 않는다. 세션 op은 세션 하나만 다루므로 grid가 작아지기 쉽다. 측정에서 grid 32(SM 84개)는 대역폭의 약 30%만 썼고, tile만 나눠도 2.9배 빨라졌다.

```
 목표 block 수 = SM 수 × SM당 동시 block 수        (device 조회 + 커널 occupancy)
 세션 op grid  = head 수 × (V / 행 조각)
 행 조각       = V를 나누는 2의 거듭제곱 중 grid ≥ 목표 block 수를 만족하는 가장 큰 값
                 (최소 크기 아래로는 나누지 않는다)
```

출력 grid 일치 원칙: 한 세션 커널의 출력들이 같은 grid를 만들도록 plan이 각 출력 view의 모양을 정한다. activation은 가중치·elementwise 연산에서는 `[B,H]`로 보고, 세션 연산에서는 같은 메모리를 `[B·heads,V]`로 본다(`H = heads·V`). 따라서 세션 b의 출력은 `[heads,V]` view이고 `[1,row_tile]`로 나누면 head와 행 조각마다 block 하나가 생긴다. KV append 출력도 `[max_seq,1,row_tile]`로 나눠 같은 head × 행 조각 grid를 만든다.

- 값은 step plan을 만들 때 계산해 const generic으로 넘긴다. JIT가 값마다 한 번 특수화해 캐시한다.
- 규칙으로 정하기 애매한 커널은 cutile-rs autotuner로 고르고 결과를 캐시한다.
- 하드코딩하지 않는다. 같은 규칙이 5080(84 SM)과 GB10에서 다른 값을 낸다.
- GDN state `S[V, K]`는 V축으로 나눈다. `S·k`, `S·q`의 합산 축이 K라서 행 조각끼리 통신이 없다. [추정, E2에서 확인]

## 4. 소유 관계

```
 에이전트 A ─┐  HTTP/API (대화 id)        ┌─ firepit 프로세스 ────────────────────────────┐
 에이전트 B ─┼───────────────────────────►│  Server                                      │
 에이전트 C ─┘                            │   sessions: HashMap<대화 id, Session>  ← 소유 │
                                          │        │ 이번 step에 진행할 세션 선택           │
                                          │        ▼                                     │
                                          │  Scheduler ──► Batch<'s> (&mut Session × B)  │
                                          │                     ▼                        │
                                          │                Engine::step                  │
                                          └──────────────────────────────────────────────┘
```

- 에이전트는 firepit에 요청을 보내는 바깥 클라이언트다. 세션은 대화 하나이고, 서버가 대화 id별로 소유한다.
- 격리 대상은 대화(세션) 사이다.
- E1에는 서버가 없다. 테스트 코드가 `Session`을 직접 소유한다.

## 5. 타입

```rust
pub struct Model {                        // 공유, 불변
    config: ModelConfig,
    weights: Weights,
}

pub struct Session {                      // 대화 하나
    id: SessionId,
    position: usize,
    state: Vec<LayerState>,               // layer 순서대로
}

pub enum LayerState {
    Gdn { recurrent: Tensor<f32>, conv: Tensor<f32> },
    Attention { k: Tensor<f16>, v: Tensor<f16> },
}

pub struct Batch<'s> {                    // 한 step 동안만 산다
    sessions: Vec<&'s mut Session>,
    tokens: Vec<u32>,                     // 세션마다 1개 (decode)
}

pub struct Engine {
    model: Arc<Model>,
    plans: HashMap<usize, CachedPlan>,    // 배치 크기별 한 세션 인스턴스 조합
}

impl Engine {
    pub fn step(&mut self, batch: Batch<'_>) -> Result<Logits>;
}
```

- `Batch`가 `&mut Session`을 들고 있으니 같은 세션을 두 번 넣거나 step 중에 다른 곳에서 세션을 만지는 코드는 컴파일되지 않는다.
- `Session`은 `Model`을 참조하지 않는다. state 모양은 `Model::new_session()`이 config로 만든다.
- `Engine::batch`는 세션 1~4개를 받는다. 3개는 내부에서 4행으로 채우며 빈 행의 세션 연산은 실행하지 않고 logits도 반환하지 않는다. 캐시는 채운 크기별 한 조합만 보관하고, 조합이 바뀌면 graph를 다시 캡처한다. `Engine::close_session`은 종료한 세션이 들어 있는 항목을 제거한다.

## 6. step plan: 연산 목록 하나

가중치 형식은 `firepit-kernels/src/weights`의 `Weight`가 나타낸다. 엔진은 가중치를 소유하고 `gemv(&Weight, x, y, tile)`을 호출한다. 형식별 GEMV 선택은 `firepit-kernels/src/gemv`에서 한다. `firepit-kernels/src/sink.rs`의 `Sink`가 eager 실행과 CUDA Graph 캡처를 공통으로 처리한다. E1의 embedding, RMSNorm, add는 각각 `embedding.rs`, `norm.rs`, `elementwise.rs`에 있고 합성 세션 커널은 `synthetic.rs`에 있다.

```
 StepPlan = [Op]                          (phase, B)마다 한 번 생성

 Op::Weight      { kind: Gemv | MoeGemv | ..., weight: WeightId, input: Act, output: Act }
                   activation [B, ...] 전체에 한 번
 Op::Session     { layer: usize, kind: GdnStep | AttnDecode | ..., input: Act, output: Act }
                   세션 b마다: input 행 b (&), output 행 b (partition [1, ...]), state = &mut sessions[b].state[layer]
 Op::Elementwise { kind: RmsNorm | Add | ..., ... }
                   activation 전체에 한 번

 실행기                              같은 목록으로
   eager(plan, batch)                  → 디버그, 정확성 비교
   capture(plan, batch) → CudaGraph    → decode 반복
```

- activation 버퍼 `Act`는 `[B, ...]` 연속 메모리다. 수명 분석과 재사용은 Grout `TensorPool`의 방식을 가져온다.
- 세션 op의 출력은 activation `[B, H]`를 행 단위 partition `[1, H]`로 나눠 세션 b에게 행 b를 준다.

## 7. 첫 마일스톤 E1: 합성 hybrid 모델

실제 Qwen 커널을 모두 짜기 전에 골격을 검증한다.

```
 embedding → [ RMSNorm → Q8 GEMV → GdnLike(state) → Q8 GEMV → Add ] × L
           → [ RMSNorm → Q8 GEMV → AttnLike(KV)   → Q8 GEMV → Add ] × L
           → RMSNorm → output head (Q8 GEMV) → logits
```

- 가중치는 난수 Q8_0. `GdnLike`/`AttnLike`는 state를 읽고 쓰는 가장 단순한 커널이다(`S ← αS + k vᵀ`, KV append + dot attention). 수학적 정확성보다 state 경로 검증이 목적이다. 두 커널 모두 3절 tile 규칙을 따른다.
- 검증:
  1. batch 동등성: 세션 B개를 한 batch로 N step 돌린 logits = 각 세션을 혼자 N step 돌린 logits.
  2. 격리: 세션 하나의 입력을 바꿔도 다른 세션의 출력이 안 바뀐다.
  3. graph 동등성: CUDA Graph replay 결과 = eager 결과.
  4. 컴파일 실패 테스트: 같은 세션을 한 batch에 두 번 넣는 코드가 컴파일되지 않는다(trybuild).
- 성능: B = 1, 2, 4에서 step 시간. 가중치 op이 B에 거의 무관한지, 세션 op 비중이 얼마인지.

E1이 끝나면 `GdnLike`/`AttnLike`를 실제 Qwen3.6 커널로 바꾸는 것이 E2다.

## 8. 범위 밖 (E1)

Server, Scheduler(prefill과 decode 섞기), prefill, MTP, GGUF 로더, tokenizer.
