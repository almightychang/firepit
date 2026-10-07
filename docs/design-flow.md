# decode step 흐름(flow) 설계

상태: 확정 (2026-10-07, Sam 승인).

## 목적

op마다 무엇을 읽고 쓰는지 데이터로 얻어, 같은 입력을 읽는 GEMV(수평 합치기)와 생산→소비 연쇄(원소별 fusion 후보)를 사람이 nsys와 config를 맞춰 보지 않고 찾는다. op 정의(`Op`)는 바꾸지 않는다. decode runner가 버퍼를 꺼내는 길을 접근자 하나로 모으고, 그 접근자가 기록한다.

근거는 CPU toy 실험이다(접근자 실행기와 직접 접근 실행기의 결과가 같았고, 우회는 E0616 컴파일 에러였으며, 커널 없이 지나가는 기록이 실행 기록과 같았다). cuTile 내부 기록과 CUDA graph 노드는 쓸 수 없다. cuTile의 접근 등록은 crate 내부이고, view의 주소·offset이 공개되지 않으며, graph 간선은 단일 stream 순서뿐이다.

## 범위

- decode runner(`execute.rs`)만. prefill(`prefill.rs`, `PrefillPlan`)은 자기 저장소를 그대로 쓴다.
- 기록 대상: decode 버퍼 slot(묶음 projection은 멤버 단위), 입력(tokens, positions, cos, sin), expert 선택, 세션 state, 가중치 행렬(`weight()`가 돌려주는 것).
- 가중치 벡터(norm gamma, conv, a_log, dt_bias, embedding)는 기록하지 않는다.
- 분석: ① 수평 합치기, ② 생산→소비·소비자 수·앞선 쓰기 없는 읽기. 바이트 대 시간은 나중이다.

## 구조

```
as-is
 engine.step_with ──► plan.tokens / positions / cos / sin  (StepPlan pub 필드)
 execute::run(model, plan, sessions, sink)
   ├─ plan.buffers[..], input_output(&mut plan.buffers, ..)
   ├─ Slot::projection(..) → ProjectionRows / NormInput heads / output_gate slice
   ├─ session.state[layer], plan.selections[l][b], plan.positions[b], plan.cos/sin
   └─ weight(model, ..), model.layers[l].moe.gate_up/down
         └─► firepit-kernels launcher ─► sink.run (Eager | Graph | Traced)

to-be
 engine.step_with ──► DecodeStore::upload_step(..)
 execute::run(ops, ctx: &mut Ctx, sink)
   op마다 sink.span(kind, |sink| { ctx.begin(i, kind, layer); …arm… })
   ctx.parts() ─┬─ buffers.io([Read; N], write) ─► ([View; N], &mut Tensor)
                ├─ inputs.tokens() / position(b) / cos() / sin()
                ├─ selections.get(_mut)(layer, b)
                ├─ states.gdn(b, layer) / attention(b, layer) / attention_ref(b, layer)
                └─ weights.matrix(layer, name) ─► &'m Weight  (기존 weight())
        ctx.model() ─► &'m Model (dims, 가중치 벡터; 기록 안 함)
                  │ log: Some일 때
                  ▼
          Vec<OpNode> ─► flow::report ─► firepit flow
                             ① 수평 합치기  ② 생산→소비·소비자 수·앞선 쓰기 없는 읽기
```

## 커밋 1: decode 접근자 (동작 동일)

### 저장소와 접근자

새 모듈 `crates/firepit/src/store.rs`:

```rust
pub(crate) struct DecodeStore {           // 필드 private
    buffers: Vec<Tensor<f32>>,            // Slot::all
    tokens: Tensor<i32>,
    positions: Vec<Tensor<i32>>,
    cos: Tensor<f32>,
    sin: Tensor<f32>,
    selections: Vec<Vec<ExpertSelection>>,
}

pub(crate) enum Read { Whole(Slot), Member { layer: usize, member: Projection } }

pub(crate) struct Ctx<'a, 'm> {           // 필드 private
    model: &'m Model,
    store: &'a mut DecodeStore,
    sessions: &'a mut [&'a mut Session],
}
impl<'a, 'm> Ctx<'a, 'm> {
    pub fn model(&self) -> &'m Model;
    pub fn parts(&mut self) -> Parts<'_, 'm>;
}
pub(crate) struct Parts<'c, 'm> {
    pub buffers: Buffers<'c, 'm>,
    pub inputs: Inputs<'c>,
    pub selections: Selections<'c>,
    pub states: States<'c>,
    pub weights: Weights<'m>,
}
```

| handle | 메서드 | 대체하는 지금 코드 |
|---|---|---|
| `Buffers` | `io::<N>(reads: [Read; N], write: Slot) -> ([View; N], &mut Tensor<f32>)`, `read::<N>(reads) -> [View; N]` | `input_output`, `two_inputs_output`, `three_inputs_output`, `plan.buffers[..]`, `Slot::projection(..)` |
| `View` | 원본 tensor와 멤버 열 범위. 기존 열 helper(`ProjectionRows`, `NormInput`, `output_gate`의 head slice)에 그대로 넘긴다 | — |
| `Inputs` | `tokens()`, `position(b)`, `cos()`, `sin()` | `plan.tokens`, `plan.positions[b]`, `plan.cos/sin` |
| `Selections` | `get(layer, b)`, `get_mut(layer, b)` | `plan.selections[layer][b]` |
| `States` | `gdn(b, layer) -> (&mut conv, &mut recurrent)`, `attention(b, layer) -> (&mut k, &mut v)`, `attention_ref(b, layer)` | `session.state[layer]` 패턴 매치 |
| `Weights<'m>` | `matrix(layer, WeightName) -> &'m Weight` (안에서 기존 `weight()`) | `weight(model, ..)`, `model.layers[l].moe.gate_up/down` |

`WeightName`에 `ExpertGateUp`, `ExpertDown`을 더해 `weight()`가 expert 행렬도 돌려주게 한다.

### borrow 제약을 푸는 방법

- 커널은 출력 `&mut Tensor`(partition)와 입력 `&Tensor`/view를 동시에 받는다. `Buffers::io`가 출력 인덱스로 `split_at_mut`를 하고 입력을 앞뒤 조각에서 꺼낸다(지금 helper 세 개를 입력 N개로 일반화). 입력과 출력이 같은 slot이면 에러다.
- `Weights`는 `&'m Model`만 쥐고 `&'m Weight`를 돌려준다. 그래서 `Buffers` 대여와 겹쳐 쓸 수 있다.
- `States`(sessions), `Selections`, `Inputs`는 `Ctx`의 다른 필드를 빌린다. `parts()`가 모듈 안에서 필드를 나눠 빌려 handle로 내주므로, `io` 결과와 동시에 산다. 세션 루프에서는 반복마다 `states.gdn(b, layer)`로 새로 빌린다.

예 (`GdnConv`):

```rust
let p = ctx.parts();
let ([qkv], target) = p.buffers.io([Read::Member { layer, member: Projection::GdnQkv }], Slot::Conv)?;
for b in 0..batch {
    let (conv, _) = p.states.gdn(b, layer)?;
    … qkv.row(b)? …
}
```

### 우회 금지

- `DecodeStore`의 필드는 `store` 모듈 private이다. 형제 모듈인 `execute.rs`에서 직접 접근하면 컴파일 에러(E0616)다.
- runner 시그니처는 `run(ops, ctx, sink)`다. `StepPlan`과 `Session`을 받지 않으므로 세션 state로 가는 다른 길이 없다.
- 컴파일러가 강제하는 것은 decode 저장소(버퍼, 입력, 선택)와 세션 state다. 가중치 행렬은 `ctx.model()`로 닿을 수 있다. 그래서 `Weights::matrix`를 쓰는 것은 규칙이다. `execute.rs`에서 `weight(`와 `.moe.gate_up`/`.moe.down` 직접 호출을 grep 테스트로 막는다.

### runner 밖

| 위치 | 커밋 1 이후 |
|---|---|
| `engine.step_with` 입력 업로드 4종 | `DecodeStore::upload_step(tokens, positions, cos, sin)` |
| `StepPlan::logits` | `DecodeStore::logits()` |
| `StepPlan::new` 할당 | `DecodeStore::new` |
| `run_routing_observed`(testing) | `#[cfg(feature = "testing")] Ctx::peek(slot)`, `peek_selection(layer, b)` |
| prefill이 쓰는 `execute.rs` helper(`input_output`, `two_inputs_output`, `gdn_gate`, `output_gate`, `rms_norm`, `weight`, …) | `input_output`·`two_inputs_output`·`three_inputs_output`은 내용 그대로 `prefill.rs`로 옮긴다. `gdn_gate`·`output_gate`는 `&mut [Tensor]` 대신 입력·게이트·출력 tensor를 받게 바꾸고, decode와 prefill이 함께 쓴다. `PrefillPlan`은 바뀌지 않는다 |

### 줄 수 (추정)

decode 실행 경로 접근 지점은 `execute.rs`에서 testing 관찰 코드를 빼고 51곳이다(`plan.buffers` 20, 묶음 열 조회 10, 가중치 10, `selections` 5, 세션 state 4, 입력 5, 일부 중복).

| 파일 | 내용 | 줄 |
|---|---|---|
| `store.rs` (새) | `DecodeStore`, `Read`, `View`, `Ctx`, `Parts`, handle 5개 | +230 |
| `execute.rs` | 접근 51곳을 handle로. helper 3개를 `prefill.rs`로 옮김. `gdn_gate`·`output_gate` 인자 | 수정 ~100, −50 |
| `prefill.rs` | helper 3개 받음. `gdn_gate`·`output_gate` 호출 2곳 | +50, 수정 ~6 |
| `plan.rs` | `StepPlan`의 저장소 필드를 `store: DecodeStore`로. `WeightName` 2개 | +12, −30 |
| `engine.rs` | 업로드·logits·runner 호출을 `DecodeStore`·`Ctx`로 | 수정 ~25 |
| `tests/no_bypass.rs` (새) | `execute.rs`의 가중치 직접 호출 grep | +25 |

합: 약 +320 / −80, 수정 약 130. op 정의 변경 없음, unsafe 없음.

## 커밋 2: 기록기와 분석

### 기록

```rust
pub struct OpNode { pub op: u32, pub kind: &'static str, pub layer: Option<u16>, pub uses: Vec<Use> }
pub struct Use { pub buffer: Buffer, pub write: bool }
pub enum Buffer {
    Slot(Slot),
    Member { layer: u16, member: Projection },
    Token, Position(u16), Cos, Sin,
    Selection { layer: u16, session: u16 },
    State { layer: u16, session: u16 },
    Matrix { layer: u16, name: WeightName },
}
```

- `Ctx`에 `log: Option<RefCell<Vec<OpNode>>>`를 둔다. 꺼져 있을 때 비용은 handle 호출마다 `Option` 검사 하나다. handle은 `&RefCell`을 나눠 가진다.
- runner 루프가 op마다 `ctx.begin(index, op.kind(), op.layer())`를 부른다. `Op::layer()`를 새로 두고, layer 필드가 없는 op는 `None`이다.
- 기록 규칙: `io`의 입력은 읽기, 출력은 쓰기다. `read`는 읽기다. `States::gdn`/`attention`은 쓰기, `attention_ref`는 읽기다. `Selections::get_mut`는 쓰기, `get`은 읽기다. `Inputs`와 `Weights`는 읽기다.
- 가중치 형식·열 수는 기록하지 않는다. 분석이 `Model`에서 찾는다(firepit-kernels `Weight::format()`·`cols()` 추가).

### 모드

| 모드 | sink | 쓰임 |
|---|---|---|
| 끔 (기본) | `Eager` / `Graph` | 평소 |
| 실행+기록 | `Eager` | 기록과 실제 실행을 함께 |
| Dry | 새 `Dry` sink(`run`이 op를 버림) | 커널 없이 기록만 |

- Dry도 모델과 plan 버퍼의 장치 할당은 필요하다(view를 실제 tensor 위에 만든다).
- decode runner는 graph capture가 되는 경로라 host 동기화가 없다. 그래서 Dry로 끝까지 지나갈 것으로 본다. 테스트 1이 확인한다.
- graph capture 중에는 기록하지 않는다.

### 분석 (`crates/firepit/src/flow.rs`)

값은 `(Buffer, 쓴 op)`다. `Member`는 해당 묶음 `Slot::Projection(i)`를 마지막에 쓴 op와 잇는다.

1. **수평 합치기.** `kind == "gemv"`인 op를 입력 값으로 묶는다.
   - 기존 묶음: 출력 `Projection(i)`의 layout 멤버가 둘 이상인 것. 멤버, 형식, padding 행, prefill 멤버별 분할(`gemm_rows`)을 보인다.
   - 남은 후보: 같은 입력·형식·열 수인 서로 다른 gemv.
   - 형식이 달라 못 합침: 같은 입력, 다른 형식.
2. **생산→소비.**
   - 값마다 소비자 op와 멤버, 소비자 수를 보인다. 소비자 0인 값도 보인다.
   - 앞선 쓰기 없는 읽기: 이번 step에서 앞에 쓴 op가 없는 읽기다. step 입력(`Token`, `Position`, `Cos`, `Sin`), `State`, `Matrix`는 제외한다.
   - layer 반복은 "패턴 × 출현 수"로 접어 출력한다.

리포트 주석으로 둘을 고정해 적는다. q/gate 멤버 단위 오차(아래 한계)와 layer 재사용 slot의 순서 검출 한계다.

### `firepit flow`

```
firepit flow <model> [--dry] [--batch B] [--layers N]
```

decode 한 step에서 op마다 어떤 버퍼를 만들고 쓰는지 기록해 텍스트 리포트를 출력한다. 시간 측정 도구(bench)와 섞지 않으려고 CLI subcommand로 둔다. CUDA graph와 헷갈리지 않게 이름은 graph가 아니라 flow다. 예상 형태(`cfdb01f` + NVFP4, 코드에서 추론, 미실행):

```
== 수평 합치기
입력 Normed (GDN ×30)    기존 [qkv|z] fp8, [b|a] bf16 · 후보 없음 · 형식이 달라 못 합침: fp8{[qkv|z]} / bf16{[b|a]}
입력 Normed (attn ×10)   기존 [q_gate|k|v] fp8
입력 Normed (MoE ×40)    기존 [router|shared_gate] bf16, padding 15행 (prefill 멤버별 2 GEMM)
== 생산 → 소비
Residual 2 · [qkv|z].qkv → gdn_conv 1 · [qkv|z].z → gdn_gate 1 · [q_gate|k|v].q_gate 2 (주석: q/gate 절반) …
== 앞선 쓰기 없는 읽기
없음
```

### 테스트

| 테스트 | 위치 | 확인 |
|---|---|---|
| Dry 기록 = 실행+기록 기록 | `tests/engine.rs`, 작은 무작위 모델 | 같은 `Vec<OpNode>` |
| 세션별 접근 | 같음, B = 2 | 세션 op마다 `State` 접근이 세션 수만큼 |
| 정상 plan | 같음 | 앞선 쓰기 없는 읽기 0. 작은 모델은 projection이 모두 bf16이라 기존 묶음 `[qkv|z|b|a]`, `[q_gate|k|v]`, `[router|shared_gate]` |
| 순서 뒤집기 검출 | `flow.rs` 단위 테스트. `StepPlan::ops`에서 layer 0의 MoE projection gemv와 `TopK`를 바꾼다 | `TopK`의 `Member{router}`, `Member{shared_gate}`가 앞선 쓰기 없는 읽기 |
| 재사용 slot 한계 | 같음, layer 1에서 바꾼다 | 앞선 쓰기 없는 읽기 0, layer 0 값으로 가는 간선. 한계를 테스트로 고정 |
| 기존 묶음 4개 (ignored) | `tests/nvfp4_prefix.rs` 옆, NVFP4 앞 4층, Dry | `[qkv|z]` fp8, `[b|a]` bf16, `[q_gate|k|v]` fp8, `[router|shared_gate]` bf16 padding 15. 남은 후보 0 |

수치는 비교하지 않는다. 판정은 구조만 본다.

### 줄 수 (추정)

| 파일 | 내용 | 줄 |
|---|---|---|
| `store.rs` | `log`, `OpNode`, `begin`, handle마다 기록 한 줄 | +80 |
| `plan.rs` | `Op::layer()` | +25 |
| firepit-kernels `sink.rs` | `Dry` | +12 |
| firepit-kernels `weights/mod.rs` | `format()`, `cols()` | +20 |
| `flow.rs` (새) | 값·간선, 소비자, 앞선 쓰기 없는 읽기, 수평 합치기, 텍스트 | +230 |
| `engine.rs` | `step_flow(batch, dry) -> Vec<OpNode>`, `flow_report` | +35 |
| firepit-cli `main.rs` | `flow` subcommand(`--batch`, `--layers`, `--dry`) | +35 |
| 테스트 | 위 표 | +120 |

합: 약 +560.

## 보장 / 방지 / 못 하는 것

**보장**
- decode runner가 꺼내는 slot, 입력, 선택, 세션 state는 모두 handle을 지나 기록된다. 우회는 컴파일 에러다.
- 가중치 행렬은 `Weights::matrix`를 지나고, 직접 호출은 grep 테스트가 막는다.
- Dry 기록은 실행 기록과 같다(테스트로 확인).

**방지**
- 처음 쓰이는 slot에서 읽기가 생산보다 앞서는 순서 오류. plan 생성 루프 전체에 걸친 실수는 layer 0에서 드러난다.
- 같은 입력·같은 형식 GEMV가 합쳐지지 않은 채 추가되는 것.

**못 하는 것**
- **layer 재사용 slot의 순서 오류.** `Normed`와 `Projection(i)`(같은 layout이면 layer끼리 공유)에는 이전 layer가 쓴 값이 남아 있다. layer k ≥ 1에서만 순서가 바뀌면 오래된 값을 읽는 간선만 생기고 "앞선 쓰기 없는 읽기"는 0이다.
- **q/gate 멤버 단위 오차.** `Norm Q`는 q_gate의 q 절반, `OutputGate`는 gate 절반을 읽는다. 둘 다 `Member{AttnQGate}`로 기록된다. 그래서 소비자 수가 2로 나온다(절반마다 1이 맞다). 리포트에 주석으로 적는다.
- prefill 경로, 가중치 벡터, firepit-kernels 내부 버퍼(FP8 scale, `ExpertSelection` 내부), expert GEMV가 실제로 읽는 expert 행, graph decode의 op 사이 GPU 공백.

## 위험과 검증

GPU 검증은 GB10에서 `scripts/spark-bench.sh`로 한다. 커밋 전(`B`)과 후(`A`)를 같은 pod에서 돌린다. 실행 전 Sam 승인을 받는다.

| 커밋 | 위험 | 검증 |
|---|---|---|
| 1 | 열 선택(묶음 멤버, q/gate 절반, 세션 행)을 잘못 옮겨 결과가 조용히 틀림 | 컨테이너 `scripts/dev.sh test --workspace` 통과. GB10에서 `spark-bench.sh B A -- 'cd $SRC && cargo test --release -p firepit -- --include-ignored'`(reference, `nvfp4_prefix`, `fp8_prefix` 포함). 같은 프롬프트의 greedy 출력(`$FIREPIT run`)이 B와 A에서 같은지. quality 예제 decode 모드 8묶음 값이 자리수까지 같은지 |
| 1 | handle 호출이 decode host 비용을 늘림 | `spark-bench.sh B A -- '$BIN/bench … --batches 1 --warmup 8 --steps 128 --max-seq 256'`를 3회 교대. graph decode는 캡처 후 재생이라 차이가 없을 것으로 본다. prefill 5000 wall도 같은 방식으로 잰다(prefill 코드는 helper 이동만). 잡음 범위 밖이면 보고 |
| 2 | Dry가 decode 경로에서 host 동기화를 만남 | 테스트 1(Dry = 실행+기록)이 실패로 드러냄 |
| 2 | 기록이 꺼져 있을 때 비용 | 커밋 1과 같은 decode bench 교대 측정 |
| 2 | 리포트가 기존 묶음을 잘못 읽음 | ignored 테스트(NVFP4 앞 4층)를 GB10에서 `spark-bench.sh A -- 'cd $SRC && cargo test --release -p firepit --test nvfp4_prefix -- --ignored'`로, `$FIREPIT flow <model> --dry`의 전체 모델 리포트를 함께 기록 |

pod 안의 모델 경로는 `docs/testing.md`의 GB10 예와 같은 `/work/…` 조립본을 쓴다. ignored 테스트가 읽는 모델·정답지 경로를 pod 경로로 넘기는 방법(환경 변수 등)은 구현 때 확인한다.
