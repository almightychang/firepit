# cuTile-rs와 cuda-oxide

## 개요

NVIDIA가 2026-09-08에 "CUDA Rust"로 공식 발표했다. 트랙은 두 개다. [확인: NVIDIA 블로그]

| | cuda-oxide | cuTile-rs |
|---|---|---|
| 모델 | SIMT (thread, block, shared memory를 직접 다룸) | Tile (tile 단위로 짜고 thread 배치는 컴파일러가 맡음) |
| 성숙도 | early alpha, pinned nightly 필요 | crates.io 배포됨(v0.4.0), stable Rust |
| 경로 | Rust MIR → LLVM → PTX | Tile IR → tileiras → 네이티브 cubin (JIT) |

- cuTile-rs 저장소는 2026-03-12에 만들어졌고, v0.1.0은 5/16, v0.4.0은 9/25에 나왔다. 9/8 발표는 출시가 아니라 공식화였다. [확인]
- 논문: "Fearless Concurrency on the GPU"(arxiv 2606.15991). 저자는 Elibol, Roesch, Gelado, **Buehler**, Garland다. mistral.rs를 만든 Eric Buehler가 공저자다. [확인]
- sm_121(DGX Spark) 지원은 v0.4.0 CHANGELOG에 "580 드라이버에서 13.3과 13.4로 검증"으로 적혀 있다. 전용 tutorial(`cutile-book/tutorials/12-dgx-spark-inference.md`)도 있다. [확인]
- Spark 실측 결과는 [measurements/cutile-smoke-spark.md](measurements/cutile-smoke-spark.md)에 있다.

## safe 모델

- 쓰기 대상 텐서(mutable output)는 host에서 겹치지 않는 partition으로 나누고, block마다 하나씩 준다. partition 모양이 launch grid를 정한다. 출력이 여러 개면 grid가 서로 맞아야 한다. [확인: cutile-book]
- `MappedPartitionMut`를 쓰면 한 block이 여러 tile을 돌며 쓸 수 있다(persistent 방식). [확인]
- 읽기 전용 입력(`&Tensor`)은 커널 안에서 어떤 tile이든 index로 load할 수 있다. [확인]
- 탈출구: `unchecked_accesses = true`(모든 검사를 끔, `unsafe fn` 필요), `pointer_to_tile`과 `load_ptr_tko`/`store_ptr_tko`(raw pointer, unsafe). 문서는 raw pointer를 "다른 방법이 없을 때만 쓰라"고 경고한다. [확인]
- bounds 검사는 가능하면 커널 밖, launch 시점의 전제 조건으로 옮긴다. [확인: 논문 초록]

## 기존 cuTile 사용처 분석

### mistral.rs (`mistralrs-quant/src/cutile/`, 약 14.4k줄)

- entry 약 25개 중 약 15개가 `unsafe fn`이고, 그중 12곳이 `unchecked_accesses = true`다. [확인]
- **출력이 규칙적인 커널은 safe다.** FP8 GEMM, W8A16, W8A8, NVFP4 GEMV, NVFP4 matmul은 `MappedPartitionMut`를 쓴다. [확인]
- **불규칙한 커널은 raw pointer다.** fused MoE, GDN prefill 3종, NVFP4 quantize, routed matmul, LoRA, split-k가 그렇다. [확인]
- candle 메모리를 cuTile 쪽으로 넘길 때 `unsafe borrow_raw_parts`를 쓰고, panic은 `catch_unwind`로 감싼다. [확인]
- 저장소에 `CLAUDE.md`와 `AGENTS.md`가 있다. 최근 6개월 커밋의 87%가 Eric이고, 최근 merge된 PR의 리뷰는 0건이다. [확인]

### Grout

entry 39개 중 unsafe는 0개다. GEMM과 GEMV는 cuBLAS에 위임한다. 자세한 내용은 [grout.md](grout.md)에 있다. [확인]

### candle

MoE 커널 하나가 `unsafe fn`, raw pointer, `unchecked_accesses = true`로 짜여 있다. [확인: [candle.md](candle.md)]

### cutile-rs 자체의 `cutile-kernels`

`publish = false`인 내부 커널 모음이다. attention, norm, RoPE, KV cache, argmax, 실험용 MoE가 있다. 주석에 "Gemm / Gemv run via cuBLAS rather than cuTile JIT"라고 적혀 있다. [확인]

### crates.io의 역의존 crate (7개)

axis, coeus-cuda, hanzo-ml(candle 계열), narsil-engine, onde-mistralrs-quant(mistral.rs 재배포), singe-kernel(일반 blockwise 양자화), ticit. **GGUF Q8_0 GEMV를 cuTile로 짠 crate는 없다.** [확인, 단 crates.io 기준]

## GDN 커널과 safe 표현 가능성 (mistral.rs `gdn_prefill.rs` 분석)

코드를 읽고 판단한 것이다. 컴파일로 확인하지는 않았다. [추정]

| 커널 | grid | 쓰는 대상 | safe 가능성 |
|---|---|---|---|
| `gdn_wy` | (bh, chunk) | w `[C,K]`, u `[C,V]` tile 하나씩 | 가능해 보임. 규칙적인 출력 2개이고 grid가 같다 |
| `gdn_out` | (bh, chunk) | token-major out `[C,V]` read-modify-write | 가능해 보임. `[B,S,H,V]` view를 `[1,C,1,V]`로 partition하면 in-place `&mut`가 된다 |
| `gdn_state` | bh | delta와 out은 chunk 수만큼 여러 tile, state는 `slots[batch]`로 간접 지정 | 어려움. 실행 시점에 정해지는 tile 개수와 간접 쓰기 두 가지가 걸린다 |

- head별 순차 state 전파는 block 하나 안의 루프라서, 순차성 자체는 safe 모델과 충돌하지 않는다. [추정]
- GDN decode step(토큰당 state 제자리 갱신)은 state가 contiguous하면 `[1,K,V]` partition과 in-place `&mut`로 표현될 것으로 본다. [추정]
- GDN state는 f32, layer당 6144 × 128 × 4 B = 3 MiB이다(ds4 기준). [확인]

## cuTile 공백 후보 (firepit에서 확인할 것)

1. 간접 쓰기(scatter): 세션 pool slot 매핑, GDN pooled state
2. 실행 시점에 정해지는 tile 개수(chunk 수)를 safe로 표현하기
3. 34-byte GGUF block(정렬이 어긋난 구조체 배열)을 tile load로 읽기
4. int8 내적(dp4a 계열)을 Tile IR로 표현하기
5. `pdl` 같은 기능이 Tile IR 13.4를 요구하는 문제(sm_121에서 13.3 기준 SKIP)

## 장단점 요약

- 장점: 쓰기 경합을 컴파일 시점에 막음(safe일 때만), NVIDIA 안에서의 이식성, 호스트 580 드라이버에서 동작(네이티브 cubin), block-scaled MMA가 sm_121에서 동작, Rust async와 CUDA Graph 통합. [확인]
- 단점: 손으로 짠 CUDA보다 조절할 수 있는 게 적어서 성능은 기껏해야 동급, 표현력 공백, JIT 지연(커널당 45~660 ms), v0.x API 변경, NVIDIA 전용, 가져다 쓸 커널이 적음. [확인]
