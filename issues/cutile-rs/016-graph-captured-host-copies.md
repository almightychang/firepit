<!--
상태: 초안 (미게시)
대상: https://github.com/NVlabs/cutile-rs/issues (기능 요청)
확인 버전: vendor/cutile-rs a620b40 (main 9fe5756 기반), GB10 (sm_121, driver 580.178.04, CUDA 13.3 toolkit)
발견: firepit hot path 측정(88d3d91, nsys --cuda-graph-trace=graph). decode step 15.59 ms 중 graph 밖 235.7 µs. 입력 업로드와 logits 다운로드를 graph 안에 넣을 safe 방법이 없음
재현: 없음(기능 요청). 측정 원자료는 firepit scratchpad hot-path-measure-report.md
-->

# No safe way to record a pinned host ↔ device copy inside a CUDA graph capture

## Summary

`CudaGraph::scope` / `capture` records `GraphNode` operations, and the only copy that is a `GraphNode` is `api::memcpy` (device to device). Recording host-to-device or device-to-host copies needs two things that are not available today:
- an operation that copies between a pinned host buffer and a `Tensor`,
- a way for the graph to own (or borrow for its lifetime) that pinned buffer.

`cuda_core::PinnedHostBuffer` exists, but its async copies (`DeviceBuffer::from_pinned_host`, `copy_from_pinned_host_async`, …) are `unsafe fn` with a caller-held lifetime contract. They also work on `cuda_core::DeviceBuffer`, not on cutile `Tensor` storage. The `GraphExecHandle` comment in `cuda_graph.rs` says the capture context "keeps host-side owners the recorded work needs (pinned host buffers, ...) alive for as long as the graph can replay". That suggests this is anticipated, but there is no public API for it.

## Motivation (measured)

A decode step replays one captured graph for all layers (Qwen3.6-35B-A3B, batch 1, GB10). Every step has to upload the step inputs (token ids, positions, RoPE cos/sin) and read back the logits (1 MB). Measured with nsys over 45 steps (median):

| part of a step | µs |
|---|---:|
| step (graph launch to next graph launch) | 15585.3 |
| waiting for the graph | 15349.6 |
| **outside the graph** | **235.7** |
| ├ `cuGraphLaunch` call | 34.1 |
| ├ logits: alloc + DtoD (`dup`) + DtoH 1 MB + sync | about 37 |
| ├ input upload: (alloc, HtoD, sync, DtoD, sync) × 4 | about 58 |
| └ host work between API calls | about 117 |

The input values change every step, so they must come from host memory. Today they go through a freshly allocated device tensor (`copy_host_vec_to_device`) and a `memcpy` into the tensors the graph was captured with, outside the graph. If the graph's first node copied from a pinned staging buffer and its last node copied the logits into a pinned buffer, a step would be: write staging, launch, one synchronize, read pinned logits. We estimate this saves 30–40 µs per step on top of what we can already save outside the graph (estimate, not measured).

## Suggested API

Something in the spirit of `api::memcpy`, recordable as a `GraphNode`:

```rust
let staging: Arc<PinnedHostBuffer<i32>> = ...;          // or a cutile-owned pinned type
s.record(api::memcpy_from_pinned(&mut tokens, &staging))?;  // HtoD node
...
s.record(api::memcpy_to_pinned(&mut pinned_logits, &logits))?; // DtoH node
```

The capture context (or the graph) would retain the pinned buffers, as the `cuda_graph.rs` comment describes. Host writes into the staging buffer between replays then need a safe, synchronized access path, for example a guard that waits for the last replay that used the buffer.

## Workaround

Copy outside the graph: allocate a temporary device tensor, `memcpy` it into the captured tensor, and synchronize. We reduced the synchronizations by zipping the copies, but the allocation and the outside-the-graph copies remain.
