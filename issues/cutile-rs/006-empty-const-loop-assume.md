<!--
상태: 초안 (미게시)
대상: https://github.com/NVlabs/cutile-rs/issues
확인 버전: vendor/cutile-rs a620b40 (main 9fe5756 기반), GB10 (sm_121, driver 580.178.04, CUDA 13.3 toolkit)
발견: firepit 커널 스캔 S1 (GDN conv `kernel_size`를 const generic으로 올린 뒤 kernel 크기 2 테스트가 JIT에서 실패)
재현: 006-empty-const-loop-probe.rs
-->

# Empty `for` loop with compile-time bounds fails JIT when the induction variable indexes a load or `extract`

## Summary

A unit-step `for` loop whose bounds are both compile-time constants and whose range is empty (for example `0..(N - 2)` with `N = 2`) fails at JIT time if the loop body uses the induction variable in an index:

```text
error: cuda_tile.assume: bounded endpoints must be ordered and fit the signed operand range
  --> crates/firepit-kernels/examples/empty_const_loop_probe.rs:20:8
```

The same kernel compiles and runs when the range is non-empty, when the upper bound is a runtime argument, or when the body does not index with the induction variable. In Rust an empty range is a valid loop that runs zero times, so we expect the kernel to compile and leave the accumulator unchanged.

## Reproduction

Full program: `006-empty-const-loop-probe.rs` (attached). The failing pattern:

```rust
#[cutile::entry()]
pub fn extract_in_const_loop<const N: i32>(
    out: &mut Tensor<f32, { [4, 1] }>,
    input: &Tensor<f32, { [4, 4] }>,
) {
    let tile: Tile<f32, { [4, 4] }> = input.partition(shape![4, 4]).load([0, 0]);
    let mut acc: Tile<f32, { [4, 1] }> = constant(0.0f32, shape![4, 1]);
    for slot in 0i32..(N - 2i32) {
        let column: Tile<f32, { [4, 1] }> =
            extract(tile, [scalar_to_tile(0i32), scalar_to_tile(slot + 1i32)]);
        acc = acc + column;
    }
    out.store(acc);
}
```

Results on GB10 (same for each variant at N = 3 and N = 4, which all produce the expected values):

| variant | N = 2 (empty range) |
|---|---|
| `extract` at `slot + 1`, const `N` | **error** (above) |
| `partition(..).load([row + 1, 0])`, const `N` | **error** (same message) |
| `extract` at `slot + 1`, runtime `n: i32` | ok, `[0, 0, 0, 0]` |
| const `N`, body does not use the induction variable | ok, `[0, 0, 0, 0]` |

## Where it seems to come from (reading the code, not verified with a patch)

`cutile-compiler/src/compiler/compile_expression.rs` (around line 2569) records the induction range for unit-step loops with exact constant bounds:

```rust
(Some(lo), Some(hi)) if lo.is_exact() && hi.is_exact() => {
    hi.start.checked_sub(1).map(|max| crate::bounds::Bounds {
        start: lo.start,
        end: max,
    })
}
```

For `0..0` this produces `[0, -1]`. `checks/placement.rs` (around line 180) then derives the static range of an affine index such as `slot + 1` from `induction_range`, and the resulting `Bounded { lb: 1, ub: 0 }` assume is rejected by `cutile-ir/src/verify_target.rs:337` ("bounded endpoints must be ordered").

## Suggested fix

Do not record an induction range when the constant range is empty, e.g. require `hi.start > lo.start` before building `Bounds` (returning `None` keeps the runtime path that already works). Alternatively skip lowering the loop body when the constant trip count is zero. `LoopFrame` already carries `known_non_empty`, so the emptiness information may already be available at that point.

We are happy to send a PR with the guard and a regression test based on the reproduction above.

## Workaround

Pass the upper bound as a runtime argument, or avoid an empty constant range (for example guard the loop with a const condition).
