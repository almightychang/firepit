<!--
상태: 초안 (미게시)
대상: https://github.com/NVlabs/cutile-rs/issues (기능 요청)
확인 버전: vendor/cutile-rs a620b40 (main 9fe5756 기반), GB10 (sm_121, driver 580.178.04, CUDA 13.3 toolkit)
발견: firepit decode GEMV 고정 비용 분리 2단계(gemv-residual-2). 같은 입력을 받는 독립 GEMV를 두 stream으로 겹치면 짧아졌지만, graph decode에서 이 겹침을 unsafe 없이 기록할 방법이 없음
재현: 없음 (기능 요청). 측정 예제는 crates/firepit-kernels/examples/gemv_fixed_cost_probe.rs (b807539)
-->

# Feature request: record independent work as parallel branches inside `CudaGraph::scope`

## Summary

`CudaGraph::scope` records every `Scope::record` call on one capture stream, so a captured graph is always a single chain. We would like a safe way to record two independent pieces of work as parallel branches of the same graph, in the shape of `rayon::join`:

```rust
let graph = CudaGraph::scope(&stream, |s| {
    s.record(norm(&mut bufs.normed, &bufs.residual, &w.norm))?;
    s.join(
        |a| a.record(gemv(&mut bufs.qkv, &bufs.normed, &w.qkv)),   // FP8
        |b| b.record(gemv(&mut bufs.beta, &bufs.normed, &w.beta)), // bf16
    )?;
    s.record(conv(&mut bufs.conv, &bufs.qkv, &w.conv))?;
    Ok(())
})?;
```

The two closures are alive at the same time, so the borrow checker already guarantees that they do not share a buffer that either of them writes. That is the same argument that makes `rayon::join` safe, and the same disjointness would make the two branches race-free.

## Motivation

We run LLM decode (batch 1, Qwen3.6-35B-A3B) as one captured graph per step on GB10. A step is about 16.7 ms and is dominated by GEMVs. Several groups of GEMVs read the same input and are independent of each other:

| group | weights | formats |
|---|---|---|
| GDN input projections (30 layers) | `qkv`, `z`, `b`, `a` | FP8, FP8, bf16, bf16 |
| attention input projections (10 layers) | `q_gate`, `k`, `v` | FP8 |
| MoE gates (40 layers) | `router`, `shared_expert_gate` | bf16 |

Small GEMVs do not fill the GPU. On GB10 the per-launch start/finish cost is 2 to 4 µs, and GEMVs with few rows are bound by per-iteration latency, not by DRAM bandwidth. A 512 x 2048 FP8 GEMV runs at about 101 GB/s per K iteration, and a 2048 x 2048 one at about 198 GB/s. Running such a GEMV next to another one should hide part of its time.

We measured this outside a graph, with two streams in eager mode. The kernels are load-only kernels that read the same bytes as our FP8 GEMVs. Times come from nsys. L2 is scrubbed by a 64 MiB read before each launch, and each value is the median of 8 runs.

| pair (rows x K, FP8 bytes) | one stream, back to back | two streams | kernel times on two streams | launch skew between streams |
|---|---:|---:|---|---:|
| 4096 x 2048 + 512 x 2048 | 49.82 µs | **44.88 µs** | 42.26 / 15.97 µs | 28.9 µs |
| 32 x 2048 + 32 x 2048 | 10.27 µs | 9.76 µs | 4.32 / 4.21 µs | 5.5 µs |

In the larger pair, the small kernel ran inside the large one. Total time became the large kernel alone plus about 6 µs, and about 5 µs of the small kernel's 8.7 µs was hidden. In the small pair, the host launch skew (5.5 µs) was longer than either kernel (about 4.3 µs), so the kernels barely overlapped. In a graph, the launch skew comes from GPU-side scheduling instead of host launches. We expect it to be much smaller there, but have not measured it.

Some of these groups could instead be merged into one GEMV by concatenating the weights (for example `router` and `shared_expert_gate`, both bf16), and we plan to do that. Merging does not work when the weights have different formats (`qkv`/`z` are FP8 and `b`/`a` are bf16 in the NVIDIA NVFP4 checkpoint) or need different kernels. Graph branches would cover those cases.

The small GEMVs in these groups take about 1 ms of the 16.7 ms step (about 0.84 ms for the small bf16 GEMVs and 0.22 ms for `k`/`v`). Our rough upper bound for the gain is that 1 ms, and we expect half of it or less. These are estimates, not measurements.

## Proposed API

A method on `Scope` that forks the capture, runs two recording closures on two branches, and joins them before returning:

```rust
impl Scope {
    pub fn join<A, B, RA, RB>(&self, a: A, b: B) -> Result<(RA, RB), DeviceError>
    where
        A: FnOnce(&Scope) -> Result<RA, DeviceError>,
        B: FnOnce(&Scope) -> Result<RB, DeviceError>;
}
```

What it would do during capture:

1. Record an event `fork` on the scope's stream (the origin stream, or the stream of the enclosing branch).
2. Make a side stream wait on `fork`. CUDA then puts the side stream in capture mode for the same graph.
3. Run `a` with the current `Scope` (same stream), and run `b` with a branch `Scope` whose context points at the side stream. Both contexts must record into the same capture `Submission`, so `adopt_capture` sees the resources recorded by both branches.
4. Record an event `done` on the side stream and make the scope's stream wait on it. This step must also run when `a` or `b` fails, so that the capture is never left with an unjoined fork.

Side streams could be owned by the scope (created lazily and reused across `join` calls), so users do not need to manage them. Nested `join` calls would give more than two branches. A slice form (`join_all(Vec<impl FnOnce(&Scope)>)`) could come later. It is harder to type with distinct borrows.

The CUDA rules this relies on are in the CUDA Programming Guide, section 4.2.2.1.2.1 "Cross-stream Dependencies and Events" (checked in the 13.1 archive):

> Stream capture can handle cross-stream dependencies expressed with `cudaEventRecord()` and `cudaStreamWaitEvent()`, provided the event being waited upon was recorded into the same capture graph.

> When a captured event is waited on by a stream, it places the stream in capture mode if it is not already, and the next item in the stream will have additional dependencies on the nodes in the captured event.

> Any other streams which are being captured to the same capture graph, due to event-based dependencies, must also be joined back to the origin stream ... Failure to rejoin to the origin stream will result in failure of the overall capture operation.

## Safety argument

The current `Scope` docs prove `record` safe mainly by same-stream ordering: all nodes are recorded on one stream and replay in order. With branches, the argument would become:

1. **Before the fork.** Everything recorded before `join` is ordered before both branches, because both branches start after the `fork` event.
2. **Inside a branch.** Each branch records on one stream, so the existing same-stream argument applies unchanged.
3. **Between branches.** `a` and `b` are passed to `join` together, so their captured borrows coexist. If `a` captures `&mut x`, then `b` cannot capture `x` at all, and the reverse holds too. Two branches can therefore share only buffers that both of them read. Unordered reads are not a data race. This is the same reasoning as `rayon::join`.
4. **After the join.** Everything recorded after `join` is ordered after both branches, because the scope's stream waits on `done`.
5. **Nothing executes during capture.** `join` only records nodes. The thread-local execution lock is still held, and `GraphNode` still rules out allocation, so the rest of the existing proof is unchanged.

Points to confirm with the maintainers:

- Aliasing outside the borrow checker. Point 3 assumes that two distinct handles cannot write the same memory. We read that a mutable partition requires unique storage (`Tensor::assert_unique_storage` in `cutile/src/tensor.rs`, which rejects aliases made by `reshape_shared`, `reinterpret`, or `into_shared_alias`), so a write needs a unique `&mut`. Writes through `Arc<Tensor>` or other shared handles would break the argument if any such API exists.
- The runtime access tracker. `register_access` skips the cross-stream conflict check for recording accesses (`!recording && ...`). That is correct for branches, because the borrow checker provides the guarantee. On replay, `GraphLaunch` reacquires every recorded access on the launch stream only, so branches do not change replay bookkeeping.
- `then_unchecked` inside a branch keeps the same caveat it has today.

## Alternatives

1. **Concatenate weights and launch one kernel.** We will do this where formats match. It does not cover mixed formats or different kernels.
2. **Record on a second stream by hand.** `cuda_core` already has safe `Event::record` and `Stream::wait_event`, so the fork and join events can be written in safe code. But there is no safe way to *enqueue* a `DeviceOp` on the side stream during capture without blocking. `sync_on` calls `stream.synchronize()`, which is not allowed on a capturing stream. It also takes the execution lock, which the scope already holds. Driving a `DeviceFuture` polls with `cuStreamQuery`. The only path that works is `unsafe async_on`, and our engine crates do not allow `unsafe`.
3. **Build the graph with explicit node APIs** (`cuGraphAddKernelNode`, ...) instead of stream capture. This makes dependencies explicit, but it is a much larger API and gives up the `DeviceOp` composition that `Scope::record` reuses.
4. **Rely on multi-stream eager execution.** This works today (the measurement above), but decode depends on graph replay to avoid per-kernel host cost. In our runs, eager two-host-thread launches were also serialized by 177 to 191 µs. We have not found the cause, and that will be a separate report.

## Where in the code

Read in the code (vendor `a620b40`), not verified with a patch:

- `cuda-async/src/cuda_graph.rs`: `CudaGraph::scope` builds one `Scope` with `ExecutionContext::for_capture(stream)`. `Scope::record` calls `op.execute(&self.ctx)` on that context's stream. `capture_on` calls `begin_capture`/`end_capture` on that stream only. There is no API that creates a second capturing stream or an event dependency. The `Scope` safety proof ("Same-stream ordering is preserved") assumes one stream.
- `cuda-async/src/device_operation.rs`: `sync_on` takes the execution lock and calls `stream.synchronize()`. `async_on` is `unsafe` and calls `ctx.complete()` immediately. `ExecutionContext` has private fields and `pub(crate) for_capture`, so a branch context that shares the capture submission has to be built inside `cuda-async`.
- `cuda-async/src/device_operation.rs`, `Producer::join`: there is already an event-based cross-stream join (`consumer.wait_event(&self.event)`) for memoized results, so the event plumbing exists.
- `cutile/src/tensor.rs`, `register_access`: accesses made while recording are exempt from the cross-stream conflict check.
- `cuda-core/src/runtime.rs`: `Event::record`, `Stream::wait_event`, `Device::new_stream` are safe. `begin_capture`/`end_capture` are `unsafe`.

Verified: the eager two-stream overlap measurement above (GB10, nsys). Not verified: graph capture with a fork and a join from Rust, and the launch skew between branches during graph replay.

We are happy to test a patch on GB10, and we can send a PR if the API shape looks right.
