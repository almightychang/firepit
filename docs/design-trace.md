# 실행 trace 설계

상태: Sam 승인 2026-10-04. E3.7 prefill 8.7 s(GB10, 5000토큰)를 op별로 나눠 보기 위한 계측.

## 용어

- span: 이름이 있고 시작과 끝이 있는 구간. op 하나 = span 하나.
- trace: 호출 한 번(prefill 또는 eager step)의 span 목록, 실행 순서대로.
- summary: trace를 op 종류별로 합산한 표.

## 구조

- `Sink`에 `span(kind, f)`를 추가한다. 기본 구현은 `f`를 그대로 실행한다. `Eager`와 `Graph`는 기본 구현을 쓴다.
- `Traced` Sink를 추가한다. `run`은 `Eager`처럼 실행 + 동기화하면서, launch 하나마다 CUDA event 쌍(cuda-core safe `Event::record`/`elapsed_time`)으로 GPU 시간을 잰다. `span`은 wall 시간으로 구간을 재고, 그 안의 `run` GPU 시간을 합산해 `Span { kind, gpu_ms, wall_ms }`를 쌓는다. wall − GPU가 host 비용(정렬, 할당, launch)이다. span 양끝 event 차이는 그 사이 host 대기까지 포함하므로 GPU 시간으로 쓰지 않는다(Sam 승인 2026-10-04 수정).
- runner(`execute.rs`, `prefill.rs`)는 op 루프 한 곳에서 각 op를 `sink.span(op.kind(), …)`으로 감싼다. `Op`/`PrefillOp`에 `kind()`를 둔다. 여러 launch와 host 작업으로 된 op(MoE)도 span 하나다.
- `Engine::prefill_traced`와 `Engine::step_eager_traced`가 `(logits, Trace)`를 반환한다. 기존 `prefill`/`step_eager`와 내부 경로를 공유하고 Sink만 다르다. 평소 경로에는 비용이 없다.
- graph decode는 op 경계가 없으므로 trace 대상이 아니다. nsys로 본다.
- `Sink`, `Traced`, `Trace`, `Span`, summary는 firepit-kernels `sink.rs`(또는 같은 crate의 모듈)에 둔다.
- `bench --profile`은 summary 표(kind, 횟수, GPU ms, wall ms, 비중)를 출력한다.
