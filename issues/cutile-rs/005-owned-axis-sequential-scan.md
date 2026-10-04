<!--
상태: 초안 (미게시, 질문)
대상: https://github.com/NVlabs/cutile-rs/issues (Discussions가 있으면 그쪽)
확인 버전: vendor/cutile-rs a620b40 (main 9fe5756 기반)
발견: firepit E3.3 (causal conv prefill), E3.5 (GDN chunk state scan)
-->

# Is `MappedPartitionMut` with `OWNED` axes the intended way to write a sequential scan kernel?

## Use case

Prefill kernels where one tile block owns a whole sequence axis for its (head, column tile) and walks it in order, carrying state in registers:

- causal depthwise conv over T tokens with a per-channel state, and
- a chunked linear-attention scan: for each chunk, read and update a state tile `S [d_k, VT]`, and write that chunk's output rows.

Each block writes two mutable outputs: its slice of the state (in place) and the output rows for every chunk.

## What we hit

Launch grids of all mutable outputs must match (`cutile-book/reference/glossary.md`, "Launch grid inference"; error `launch grid … does not match the inferred partition grid …` from `cutile/src/tile_kernel.rs`). Partitioning the output per chunk gives a grid of `(chunks·heads, 1, d_v/VT)` while the state gives `(heads, 1, d_v/VT)`.

## What we did (works, passes our tests)

- Make each block's output tile span the whole time axis: tile `[next_power_of_two(T), 1, VT]` over output `[T_pad, H, d_v]`, so the grid becomes `(1, H, d_v/VT)`.
- View the state `[H, d_k, d_v]` as `[1, H·d_k, d_v]` with tile `[1, d_k, VT]` so its grid matches. (A rank-4 view with singleton axes was rejected: "Unable to partition tensors with rank > 3"; collapsing worked as the message suggests.)
- Inside the kernel, `partition_mut([C, 1, VT]).store(tile, [chunk, 0, 0])` writes each chunk at a dynamic index.

Costs: the time tile is a const generic, so every new power-of-two prompt length compiles a new kernel; and the shapes are hard to read.

## Question

`cutile/src/tensor.rs` documents `OWNED`: "An owned axis is not traversed by the mapped work-item stream: each stream item owns the axis's full extent, so in-kernel loops traverse it with plain (`Dim`-bounded) indices." That reads like exactly this case. Is `MappedPartitionMut` with an `OWNED` time axis the intended pattern? If so, could an example be added (the only `MappedPartitionMut` example, `persistent_gemm.rs`, does not use `OWNED`), including how a second mutable output (the in-place state) should be partitioned so the grids agree? We are happy to contribute the example once the intended shape is clear.

## Follow-up experiment (2026-10-04, RTX 5080, probe code in `005-owned-prefill-probe.rs`)

- A causal conv prefill with the time axis `OWNED` works: output `[T, C]`, tile `[1, CT]`, `map([OWNED, 1], C/CT)`, and the in-place state as an ordinary `&mut Tensor` `[C, K−1]` partitioned `[CT, KT]` share the physical grid `(C/CT, 1, 1)`. Results match our existing kernel bit for bit and an f64 oracle to 4.9e-9.
- The time tile leaves the const generics, but the in-process JIT count was 1 at T=3 and 1 at T=16, 0 for T=5, 9, 32 … 1024 (the existing kernel: 1 per T). Which shape properties trigger respecialization?
- Two mapped outputs with different map shapes (`[OWNED, 1]` and `[1, OWNED]`) launched with the same `num_tile_blocks`. Is the pairing of their `iter_indices()` guaranteed?
- For a chunked scan whose state is updated in place, `MappedPartitionMut` has no `load()`, so the state cannot simply be a mapped output too. What is the recommended pattern?
- Performance at CT=16: 1.08× the time of the non-mapped kernel (T = 256–1024).
