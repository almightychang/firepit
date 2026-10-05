<!--
상태: 초안 (미게시)
대상: https://github.com/NVlabs/cutile-rs/issues
확인 버전: vendor/cutile-rs a620b40 (main 9fe5756 기반), GB10 (sm_121, driver 580.178.04, CUDA 13.3 toolkit)
발견: firepit E4.2 NVFP4 GEMV (`gemv/nvfp4.rs`, 우회 커밋 e795675)
재현: 007-reduce-type-probe.rs
-->

# Binary operator rejects a `reduce_sum` result next to a tile of the same declared shape when the shape uses a const generic

## Summary

Inside a kernel generic over `const N: i32`, multiplying the result of `reduce_sum` by another `Tile<f32, { [N] }>` fails at JIT time, although both operands are declared as `Tile<f32, { [N] }>`:

```text
error: binary `Mul` requires operands of the same type, but got `Tile < f32 , { [4] } >` and `Tile < f32 , { [N] } >`
  --> crates/firepit-kernels/examples/reduce_type_probe.rs:21:18
```

The same kernel compiles with a literal shape, or when the `reduce_sum` result is passed through `reshape` to its own shape first.

## Reproduction

Full program: `007-reduce-type-probe.rs` (attached).

```rust
#[cutile::entry()]
pub fn reduce_then_mul<const N: i32>(
    out: &mut Tensor<f32, { [N] }>,
    pairs: &Tensor<f32, { [-1, 2] }>,
    scale: &Tensor<f32, { [-1] }>,
) {
    let values: Tile<f32, { [N, 2] }> = pairs.partition(shape![N, 2]).load([0, 0]);
    let sums: Tile<f32, { [N] }> = reduce_sum(values, 1i32);
    let factor: Tile<f32, { [N] }> = scale.partition(shape![N]).load([0]);
    out.store(sums * factor); // JIT error with N = 4
}
```

Results on GB10 (launched with `.generics(vec!["4"])` where applicable):

| variant | result |
|---|---|
| `reduce_sum` then `*`, const generic `N` | **error** (above) |
| `reduce_sum`, `let sums: Tile<f32, { [N] }> = sums.reshape(shape![N]);`, then `*` | ok, `[1, 10, 27, 52]` |
| `reduce_sum` then `*`, literal shape `[4]` | ok, `[1, 10, 27, 52]` |

## Where it seems to come from (reading the code, not verified with a patch)

`cutile-compiler/src/compiler/compile_binary_op.rs` (`compile_binary_op_from_values`, around line 337) compares the operand Rust types by token:

```rust
if lhs.ty.rust_ty != rhs.ty.rust_ty {
    return self.jit_error_result(span, &format!(
        "binary `{:?}` requires operands of the same type, but got `{}` and `{}`", ...
```

For a call to a generic op such as `reduce_sum`, `cutile-compiler/src/generics.rs` (around lines 975-995) substitutes the instantiated const array into the result type, so the value carries `{ [4] }`. A value whose type comes from the `let` annotation or from `shape![N]` keeps `{ [N] }`. The two types are equal after instantiation, but their token streams differ.

## Suggested fix

Compare the instantiated types instead of the tokens, e.g. substitute `generic_vars` into both `rust_ty`s before the comparison, or compare the compiled tile types (element type and concrete shape). Alternatively, normalize the `let`-annotated type of a value to its instantiated form when it is bound.

We are happy to send a PR once the preferred place for the normalization is clear.

## Workaround

Reshape the `reduce_*` result to its own declared shape before using it in a binary op:

```rust
let sums: Tile<f32, { [N] }> = reduce_sum(values, 1i32);
let sums: Tile<f32, { [N] }> = sums.reshape(shape![N]);
```
