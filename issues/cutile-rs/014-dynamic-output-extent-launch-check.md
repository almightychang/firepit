<!--
상태: 초안 (미게시)
대상: https://github.com/NVlabs/cutile-rs/issues (불일치 보고 + 기능 요청)
확인 버전: vendor/cutile-rs a620b40 (main 9fe5756 기반), GB10 (sm_121, driver 580.178.04, CUDA 13.3 toolkit, rustc 1.98.0)
발견: firepit JIT 지연 작업. decode `cache_append`의 출력 slab을 `[MAX_SEQ, 1, D]`에서 `[-1, 1, D]`로 바꾼 커밋(e9cefef)이 driver-free Tile IR 테스트는 통과했지만 GB10 launch에서 모든 decode가 실패했다(`k_cache partition shape mismatch. Expected [-1, 1, 32], got [8, 1, 32]`). 되돌림 3e2c491
재현: 014-dynamic-output-extent-probe.rs (GB10에서 실행, 출력은 본문 Reproduction)
관련: NVlabs/cutile-rs#130 (닫힘, PR #131): 커널 안에서 동적 extent tensor에 `partition_mut`을 하는 경로의 JIT 오류를 고침
-->

# A `&mut` tensor parameter with a `-1` extent compiles to Tile IR but is always rejected at launch

## Summary

An entry point whose writable tensor parameter has a dynamic extent, for example `slab: &mut Tensor<f32, { [-1, D] }>`, behaves inconsistently:

- The compiler accepts it. `KernelCompiler::compile()` produces Tile IR, and the in-kernel `partition_mut(..).store(..)` on that slab is handled. Its extent appears as a host-resolved `ViewExtent`.
- The generated launcher rejects every launch. Its output-parameter check requires each declared extent to equal the given partition extent, so `-1` never matches:

```text
k_cache partition shape mismatch. Expected [-1, 1, 32], got [8, 1, 32]
```

The book states the rule: "Writable tensor parameters use static tile shapes". It does not say why. Read-only parameters accept `-1` and are matched at launch.

We expect one of two behaviors:
- (a) accept `-1` on writable parameters and bind it from the partition shape at launch, as read-only parameters already do, or
- (b) reject `-1` on a writable parameter when the kernel is compiled, with an error that points at the signature.

Today a kernel that passes a driver-free compile test fails only on a real launch.

## Motivation

A decode step appends one key and value row to a per-session KV cache at a position that lives on the device. The step is replayed from a CUDA graph, so the host cannot pick the row with `slice_mut`; that would bake one address into the graph. The kernel therefore takes the whole cache for a head as its output slab and stores one row at the device position:

```rust
#[cutile::entry()]
pub fn cache_append<const MAX_SEQ: i32, const KV_HEADS: i32, const D: i32>(
    k_cache: &mut Tensor<f32, { [MAX_SEQ, 1, D] }>,
    v_cache: &mut Tensor<f32, { [MAX_SEQ, 1, D] }>,
    key: &Tensor<f32, { [KV_HEADS, D] }>,
    value: &Tensor<f32, { [KV_HEADS, D] }>,
    position: &Tensor<i32, { [1] }>,
) {
    let pid: (i32, i32, i32) = get_tile_block_id();
    let index: i32 = tile_to_scalar(position.partition(shape![1]).load([0]).reshape(shape![]));
    let k: Tile<f32, { [1, 1, D] }> = key.partition(shape![1, D]).load([pid.1, 0]).reshape(shape![1, 1, D]);
    k_cache.partition_mut(shape![1, 1, D]).store(k, [index, 0, 0]);
    // v likewise
}
```

`MAX_SEQ` is not used for any tile. It exists only because the slab shape must be static.

Every cache length is therefore a separate JIT specialization. Our CLI sizes the cache as prompt length plus generation length, so each new prompt length recompiled this kernel. With the cuTile disk cache enabled, a third run with a different prompt still ran `tileiras` again. Our workaround rounds the cache length to a power of two. That leaves a logarithmic number of variants and costs up to twice the cache memory.

## Reproduction

Full program: `014-dynamic-output-extent-probe.rs`. Copy it to a crate's `examples/` and run it on a GPU. It has two kernels that differ only in the slab type:

```rust
#[cutile::entry()]
pub fn write_row_static<const ROWS: i32, const D: i32>(
    slab: &mut Tensor<f32, { [ROWS, D] }>,
    row: &Tensor<f32, { [1, D] }>,
    index: i32,
) {
    let tile: Tile<f32, { [1, D] }> = row.partition(shape![1, D]).load([0, 0]);
    slab.partition_mut(shape![1, D]).store(tile, [index, 0]);
}

#[cutile::entry()]
pub fn write_row_dynamic<const D: i32>(
    slab: &mut Tensor<f32, { [-1, D] }>,
    row: &Tensor<f32, { [1, D] }>,
    index: i32,
) {
    // same body
}
```

The program does three things:

1. Compiles `write_row_dynamic` to Tile IR with `KernelCompiler`.
2. Launches `write_row_static` on an `[8, 32]` slab partitioned `[8, 32]`.
3. Launches `write_row_dynamic` on the same slab and partition.

Output on GB10 (sm_121, CUDA 13.3, rustc 1.98.0):

```text
write_row_dynamic Tile IR: ok (3113 bytes)
write_row_static launch: ok, row 3 written: true
write_row_dynamic launch: ERROR kernel launch error: slab partition shape mismatch. Expected [-1, 32], got [8, 32]
```

The same kernel body compiles for the dynamic slab, launches and writes correctly with the static slab, and is refused at launch with the dynamic one. We first hit this on our real kernel: `cache_append` with a `[-1, 1, D]` slab compiled to identical Tile IR for cache lengths 48 and 4096, and every launch on GB10 failed with the message in the Summary.

## Where it seems to come from

Read in the code (vendor `a620b40`), not verified with a patch:

- `cutile-macro/src/kernel_launcher_generator.rs`, output-parameter validation (around line 1857):

  ```rust
  kernel_launch_assert_with(zip(valid_shape, given_shape).all(|(&expected, &given)| expected as i64 == given as i64), …)
  ```

  The input-parameter branch (around line 1885) instead uses `expected == -1 || expected == given`.
- `cutile-compiler/src/compiler/_function.rs`, `extent_atom` (around line 420): a `&mut` parameter's extent is `Atom::ViewExtent { param, axis }`, "resolved on the host against the partition shape". The compiler already models a slab extent known only at launch.
- `cutile/tests/optimization_hints.rs` and `checked_store_corpus.rs` compile kernels with `out: &mut Tensor<f32, { [-1, N] }>` to IR. We did not find one that launches such a kernel.
- `cutile-book/guide/host-vs-device.md`, entry point rules 2 and 3: writable parameters use static tile shapes, read-only ones may be dynamic. No reason is given.
- #130 / PR #131 fixed `partition_mut` on a dynamic outer extent inside a kernel. So dynamic writable extents are an intended path at least below the entry point.

We could not find a safety argument that needs a static extent here. The host partition already makes each block's slab disjoint, and the slab's runtime extent is passed to the kernel. If there is one, for example something that `partition_prefix` or mapped partitions rely on, the compile-time rejection in (b) would state it at the right place.

## Suggested fix

1. **Accept `-1` on writable parameters.** In the output branch, use the same predicate as the input branch: `expected == -1 || expected == given`. Bind the extent from the partition shape, which the launcher already passes for `ViewExtent` checks.
2. **Or reject it at compile time.** If static shapes are required, have the entry-point compiler reject `-1` on a `&mut` parameter with a message that cites the rule and its reason. A driver-free test then fails the same way a launch does.

We can test a patch on GB10, and we can send a PR for option 1 if that direction is right.

## Workaround

Keep the writable extent static and limit the number of distinct values. We round the cache length to a power of two, no smaller than the largest shape divisibility the JIT specializes on (`DivHint::default().max`).
