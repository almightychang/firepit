# E3 prefill 입구 설계

상태: 3.0에서 입구 확정. 구현은 3.7.

## 호출과 의미

`engine.prefill(&mut session, &[u32]) -> Result<Vec<f32>>`는 세션 하나에 토큰 시퀀스를 한 번에 넣고 마지막 토큰의 전체 vocabulary logits를 반환한다. 입력은 비어 있지 않아야 하고 모든 토큰은 모델 vocabulary 안에 있어야 한다. `session.position + tokens.len()`은 런타임 `max_seq`를 넘지 않아야 한다.

prefill은 현재 세션 위치에서 시작한다. 성공 뒤 `session.position`은 입력 길이만큼 증가하고, 각 GDN의 conv 및 recurrent state와 attention의 K/V cache는 같은 입력을 `Engine::step`으로 T번 처리한 결과와 같다. 실패 뒤 세션은 기존 엔진 규칙대로 poisoned 상태가 되며 다시 쓰지 않는다.

3.7에서는 세션 하나의 시퀀스 축으로 plan을 만든다. layer별 residual은 `[T, hidden]`이며, 마지막 norm과 lm_head는 마지막 행에만 적용한다. 가중치 연산은 GEMM으로 묶고, causal attention은 같은 시퀀스의 앞 위치만 읽는다. GDN은 chunk 사이 recurrent state를 전달하고 마지막 conv state를 세션에 저장한다. 모델 차원과 state 모양은 `ModelDims`와 세션 설정에서 얻는다.

batch decode와 prefill을 같은 호출에 섞는 스케줄러는 E5에서 다룬다. E3의 호출은 세션 하나만 받는다.

## GDN chunk ① 출력 계약 (3.4 → 3.5)

입력은 0 패딩한 `[chunk_count, tile, head, dim]` 연속 f32이다. `chunk_size`는 논리적 길이이고 `tile=next_power_of_two(chunk_size)`다. key head와 value head는 각각 K/Hv이며, 매핑은 `value_head / (Hv/Hk)`다. eps와 query scale은 호출자가 모델 설정에서 넘긴다. 마지막 chunk의 실제 길이를 제외한 패딩 위치는 q/k/v/g/β를 0으로 채운다.

출력은 `flat=chunk_index*Hv+value_head` 순서의 U `[flat,tile,d_v]`, W `[flat,tile,d_k]`, intra `[flat,tile,tile]`, cum `[flat,tile]`이다. intra의 `j>i` 원소는 0이다. 3.5는 chunk마다 U, W, intra, cum을 같은 flat에서 읽고 chunk 사이 state를 순서대로 전파한다. 모든 출력은 f32이다.

내부 UT는 `[flat,tile,tile]`, 16행 기본 대각 역행렬은 `[flat*(tile/block),block,block]`, 전체 역행렬은 `[flat,tile,tile]`이다. 먼저 각 대각 블록을 독립적으로 앞에서부터 대입하고, 블록 행을 순서대로 병합한다. 병합은 이전 역행렬과 다음 역행렬을 분리해 같은 launch 안의 읽기/쓰기 경합을 피한다. 행렬 곱과 U/W 적용은 bf16 hi+lo 세 곱으로 f32 수준의 내적을 만든다. 블록 크기는 호출자의 generic 인자이며 현재 검증 및 성능 측정은 16이다.

## 3.7 조립 (Sam 승인 2026-10-04)

- 경로: `Engine::prefill`은 `StepPlan`과 별도인 `PrefillPlan`을 만든다. decode 설계(flat op 목록, 커널 하나 = op 하나, 이름 붙은 slot)를 따르되 op와 slot은 `PrefillOp`/`PrefillSlot`으로 분리한다. slot의 행 축은 T다.
- op 대응: projection은 GEMM, conv는 `conv1d_silu_prefill`, GDN 상태는 chunk ①②, attention은 `cache_append_prefill`과 `causal_attention_prefill`, MoE는 3.6 경로. norm, add, gated SiLU, output gate, embedding은 decode 커널을 그대로 쓴다. final norm과 lm_head는 마지막 행만 계산한다.
- 실행: eager. MoE host 정렬 때문에 CUDA graph를 쓰지 않는다. slot은 호출마다 T 크기로 할당하고 캐시하지 않는다.
- 범위: prompt 전체를 한 번에 넣는다(`position + T ≤ max_seq`). position > 0 이어 붙이기를 지원한다. 조각 나눠 넣기는 E5. GEMM은 dense bf16만 받으므로 Q8 가중치면 에러를 낸다.
- 중복 정리: 같은 알고리즘의 decode/prefill 복사본(top-k, combine)은 decode가 prefill 커널을 행 1개로 부르게 바꾸고 decode 복사본을 지운다. decode 성능이 유지되는지 측정으로 확인한다. 정리할 수 없는 경우(예: graph 재생 중 device 위치를 읽는 cache append)는 이유를 기록한다.

decode의 `cache_append`는 graph 재생마다 device의 position tensor를 읽는다. `cache_append_prefill`은 호출 시 정한 host position으로 cache view를 만든다. 둘을 합치면 graph에서 position을 host에 동기화하거나 prefill에 device position을 새로 전달해야 하므로 유지한다.
