<!--
상태: 초안 (미게시)
대상: https://github.com/NVlabs/cutile-rs/issues
관련: #6 (zero-copy tensor views, closed), #242 / #272 (cuda-async ownership fixes)
구현: vendor/cutile-rs 브랜치 feat/tensor-view-mut (a620b40, 로컬, 미push)
-->

# Feature request: mutable tensor views (`slice_mut` / `split_mut`) usable as kernel outputs

## Summary

Host code can borrow part of a tensor immutably (`Tensor::slice`, `Tensor::view` → `TensorView`, #6), but there is no way to borrow part of a tensor **mutably** and pass it to a kernel's `&mut Tensor` parameter. Kernel outputs must be a whole `Tensor` or `&mut Tensor`.

This is the GPU-tensor counterpart of `slice::split_at_mut` / `chunks_mut`: handing out disjoint mutable pieces of one buffer, with disjointness proven on the host.

## Use case

A small inference engine where several sessions share one decode step. Weight ops run once over a contiguous activation `[B, H]`. Per-session ops (recurrent-state update, KV-cache attention) are launched once per session, each writing only its own row `b` of the activation and its own state tensor:

```rust
for (b, session) in batch.sessions.iter_mut().enumerate() {
    let row = activation.slice_mut(&[b..b + 1, 0..H])?;          // not available today
    gdn_step(&mut session.state[layer], (&mut row).partition([1, H_TILE]), ...).sync_on(&stream)?;
}
```

Row views let the borrow checker guarantee that a launch writes only its session's row, without an index argument the kernel must trust.

## Current behavior (0.4.0)

```rust
let mut activation = api::zeros::<f32>(&[2, 128]).sync()?;
let mut row = activation.slice(&[0..1, 0..128])?;   // TensorView (read-only)
let _output = (&mut row).partition([1, 128]);        // does not compile
```

```text
error[E0599]: `&mut cutile::prelude::TensorView<'_, f32>` is not an iterator
    --> examples/mutable_row_probe.rs:8:30
     |
   8 |     let _output = (&mut row).partition([1, 128]);
     |                              ^^^^^^^^^
```

`PartitionMut` / `KernelOutput` are implemented for `Tensor<T>` and `&mut Tensor<T>` only.

Workarounds today:

1. Pass the whole `&mut activation` to every per-session launch and have only the blocks for row `b` store. Safe, but the row isolation is carried by an index argument, not by types, and most blocks are empty.
2. Implement `KernelOutput` for a user-side row type. Possible because the generated launcher is generic over `KernelOutput<T>`, but `Storage::retain` is `pub(crate)` and pushing an offset pointer needs `unsafe AsyncKernelLaunch::push_device_ptr`, so user code has to carry an `unsafe` block that belongs inside cutile.

## Proposal

```rust
impl<T: DType> Tensor<T> {
    pub fn slice_mut(&mut self, ranges: &[Range<usize>]) -> Result<TensorViewMut<'_, T>, Error>;
    pub fn split_mut(&mut self, axis: usize, sizes: &[usize]) -> Result<Vec<TensorViewMut<'_, T>>, Error>;
}

// usable wherever a kernel takes `&mut Tensor<T, S>`
impl<'a, T: DType> PartitionMut<'a, T> for &'a mut TensorViewMut<'_, T> { ... }
impl<'a, T: DType> KernelOutput<T> for Partition<&'a mut TensorViewMut<'_, T>> { ... }
```

`TensorViewMut` mirrors `TensorView` (base, byte offset, shape, strides, specialization bits). The compiler side needs no change: the kernel sees an ordinary `&mut Tensor` parameter with the view's shape, strides and base pointer, and bounds checks apply to the view's shape.

## Open question: runtime access tracking

`KernelOutputStored::retain` acquires exclusive access per **storage** (`storage.retain(ctx, true)`). Two views from `split_mut` share one storage.

- (a) Track by storage, as today. Launches on disjoint views of the same storage are ordered like launches on the same tensor. Simple and conservative; sufficient for sequential per-session launches.
- (b) Track by byte range, so launches on disjoint views can overlap across streams or graph branches.

(a) seems enough for a first version. I have (a) implemented on a branch (`Tensor::slice_mut` → `TensorViewMut`, ~220 lines in `tensor.rs` mirroring the `&mut Tensor` output path, plus GPU tests for row isolation, unaligned column slices and range errors, and a `compile_fail` doctest for using the base while the view is alive). Would (a) fit the ownership model you have been hardening in cuda-async (#242, #272), or is there a reason to prefer a different shape of API here?

## Context

- Found while building firepit (local multi-session inference on DGX Spark with safe cuTile), where per-session state kernels are launched once per session and write one row of a shared activation buffer.
- Happy to send a PR for (a) if the direction looks right.
