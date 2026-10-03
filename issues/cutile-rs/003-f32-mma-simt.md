<!--
상태: 초안 (미게시)
대상: https://github.com/NVlabs/cutile-rs/issues
확인 버전: vendor/cutile-rs a620b40 (main 9fe5756 기반), RTX 5080 (sm_120), CUDA 13.3
발견: firepit E3.2 (attention prefill), 실험 2026-10-03
-->

# `mma` with `f32` operands lowers to SIMT FMUL/FADD; flash-attention examples miss Tensor Cores

## Summary

`mma` is documented as a Tensor-Core operation, but with `f32` × `f32` operands it compiles to SIMT `FMUL`/`FADD` on sm_120 — no `HMMA`. Converting the operand tiles to `tf32` inside the kernel uses `HMMA.1688.F32.TF32` and is about 14× faster. The `flash_attention*.rs` examples use `f32` `mma`, so they do not exercise Tensor Cores.

This may well be intended (exact f32 semantics unless the user opts into `tf32`). The request is mainly to make it visible.

## Measurements (RTX 5080, release, GEMM M=512 N=8192 K=2048, tile 64×128×32, f32 accumulate)

| operands → `mma` | SASS (static count) | TFLOPS | max rel. error vs f64 |
|---|---|---:|---:|
| f32 × f32 | `FMUL` 2048, `FADD` 2112, no HMMA | 3.0 | 4.9e-7 |
| `convert_tile` to tf32 × tf32 | `HMMA.1688.F32.TF32` 128 | 42.9 | 2.8e-5 |
| bf16 × bf16 | `HMMA.16816.F32.BF16` 64 | 89.8 | 4.8e-5 |

`flash_attention_causal.rs` as shipped vs. the same kernel with q/k and p/v converted to `tf32` before each `mma` (both pass the example's own check; wall time per synchronized call, 100 calls after warm-up):

| | f32 `mma` (shipped) | tf32 `mma` |
|---|---:|---:|
| non-causal | 0.0124 ms | 0.0086 ms |
| causal | 0.1307 ms | 0.0205 ms |

## Where the docs point the other way

- `cutile/src/_core.rs:2015` — "Tensor-Core matrix multiply-accumulate: `result = lhs × rhs + acc`."
- `cutile-book/reference/dsl-api.md:677` — "Maps to hardware tensor cores when available."
- `cutile-book/guide/performance.md:68` — "The compiler lowers supported dtype and shape combinations to Tensor Core instructions:"

`cutile-book/guide/performance.md:150` does say "using `f32` when `f16`, `bf16`, FP8, or block-scaled formats are acceptable can leave Tensor Core throughput unused", but it does not mention that `f32` `mma` falls back entirely, nor the `tf32` opt-in (`tf32` is listed only as a type in `dsl-api.md:347`).

## Suggested changes

1. State in the `mma` doc comment and `dsl-api.md` that `f32` operands are computed exactly on SIMT units, and that `convert_tile` to `tf32` (or bf16/f16) selects Tensor Cores.
2. Mention the `tf32` opt-in in `performance.md` next to line 150.
3. Either convert the flash-attention examples to `tf32` (or bf16), or note in them that they run without Tensor Cores.

Happy to send a PR for 1–3 if that matches the intended semantics.
