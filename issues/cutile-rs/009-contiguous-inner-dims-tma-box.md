<!--
상태: 초안 (미게시)
대상: https://github.com/NVlabs/cutile-rs/issues (성능 제안, 기능 버그 아님)
확인 버전: vendor/cutile-rs a620b40 (main 9fe5756 기반), GB10 (sm_121, driver 580.178.04, CUDA 13.3 toolkit, tileiras V13.3.36)
발견: firepit 양자화 GEMV 병목 probe(docs/research/measurements/gemv-probe-gb10.md). FP8 가중치를 [N, K/32, 32]로 읽을 때 [N, K]보다 약 35% 느림
재현: 009-tma-view-probe.rs
-->

# Performance: a 3-D view whose two inner dimensions are contiguous loads through 32-byte TMA box rows; passing the inner stride statically might let them merge

## Summary

This is a performance suggestion, not a correctness bug.

On GB10, loading the same bytes with the same `[BN, KT]` tile per step reaches different steady-state bandwidth depending only on the view:

- `[N, K/32, 32]` view, `[BN, NB, 32]` tile: the load becomes `UTMALDG.3D` with a 32-byte innermost box row, about 138 GB/s.
- `[N, K/128, 128]` view: `UTMALDG.3D` with 128-byte rows, about 212 GB/s.
- `[N, K]` view: `UTMALDG.2D` with 128-byte rows, about 214 GB/s.

In the first view, dimension 1 has stride 32, which equals the static extent of dimension 2. The two dimensions could therefore be treated as one 128-byte row. The Tile IR does not carry that stride (`strides=[?,?,1]`), so the backend cannot know that.

## Reproduction

Full program: `009-tma-view-probe.rs` (attached), run as `tma_view_probe N K 128 128`. It has three kernels that differ only in the view of one contiguous `u8` buffer:

```rust
#[cutile::entry()]
pub fn inner_32<const BN: i32, const NB: i32>(
    y: &mut Tensor<u32, { [1, BN] }>,
    w: &Tensor<u8, { [-1, -1, 32] }>,
    chunks: i32,
) {
    let pid: (i32, i32, i32) = get_tile_block_id();
    let wp = w.partition(shape![BN, NB, 32]);
    let mut acc: Tile<u32, { [BN] }> = constant(0u32, shape![BN]);
    for kc in 0i32..chunks {
        let bytes: Tile<u8, { [BN, NB, 32] }> = wp.load([pid.1, kc, 0]);
        let wide: Tile<u32, { [BN, NB, 32] }> = exti(bytes);
        let part: Tile<u32, { [BN, NB] }> = reduce_sum(wide, 2i32);
        let part: Tile<u32, { [BN] }> = reduce_sum(part, 1i32);
        acc = acc + part.reshape(shape![BN]);
    }
    y.store(acc.reshape(shape![1, BN]));
}
// inner_128: the same with { [-1, -1, 128] } and shape![BN, NB, 128]
// rows:      { [-1, -1] } and shape![BN, KT], loaded at [pid.1, kc]
```

Results on GB10, BN = 128, KT = 128 bytes per row per step, median of `cutile::bench::do_bench`. GB/s counts the bytes read:

| N x K (u8) | `[N, K/32, 32]` | `[N, K/128, 128]` | `[N, K]` |
|---|---:|---:|---:|
| 8192 x 2048 | 0.183 ms, 92 GB/s | 0.145 ms, 116 GB/s | 0.143 ms, 117 GB/s |
| 8192 x 16384 | 0.997 ms, 135 GB/s | 0.672 ms, 200 GB/s | 0.668 ms, 201 GB/s |
| 32768 x 8192 | 1.939 ms, 138 GB/s | 1.265 ms, 212 GB/s | 1.253 ms, 214 GB/s |

SASS and resource usage (8192 x 16384, JIT cache cubin):

| view | TMA loads in the kernel | shared memory |
|---|---|---:|
| `[N, K/32, 32]` | 8 x `UTMALDG.3D` | 34580 B |
| `[N, K/128, 128]` | 2 x `UTMALDG.3D` | 100052 B |
| `[N, K]` | 2 x `UTMALDG.2D` | 100052 B |

Rank does not matter: the 3-D view with a 128-byte inner dimension is as fast as the 2-D view. The innermost row length is what matters. We measured the same effect with `f8e4m3fn` weights in an FP8 GEMV (about 145 vs 226 GB/s steady state).

## Where it seems to come from

Verified (`CUTILE_DUMP=ir`): the tensor view type for the 32-wide view is

```text
partition_view<tile=(128x4x32), padding_value = zero, tensor_view<?x?x32xi8, strides=[?,?,1]>>
```

So the stride of dimension 1 is dynamic in the IR, although for this tensor it is always 32.

Read in the code, not verified with a patch: the launcher passes only "stride is 1" or "unknown" per dimension. `cutile-macro/src/kernel_launcher_generator.rs` (around line 1794) builds `stride_args` from `spec().stride_one` as `1` or `-1`. `kernel_entry_generator.rs` then turns these into the `strides` of `make_tensor_view`. A stride that equals the product of the static inner extents (here 32) becomes `?`.

We did not find code in cutile-rs that builds TMA descriptors or merges dimensions. TMA lowering appears to be done entirely by `tileiras` from the tensor/partition views. We have not verified whether `tileiras` would merge the two inner dimensions if the stride were static.

## Suggested fix

Two options, from least to most invasive:

1. Record exact static strides when they are known. If a tensor's stride equals the product of the static extents to its right (contiguous inner dimensions), pass that value instead of `-1`, so the view becomes `strides=[?,32,1]`. This only helps if `tileiras` merges contiguous dimensions with static strides, which we could not check.
2. Merge in cutile-rs. When a partition tile covers the full static extent of the inner dimensions and those dimensions are contiguous, emit the load through a lower-rank view (`[N, K]` with a `[BN, NB*32]` tile) and reshape the loaded tile to `[BN, NB, 32]`. This is what we do by hand today.

We are happy to test a patch on GB10. We can also send a PR for option 1 if that direction looks right.

## Workaround

Declare the kernel parameter with the merged inner dimensions (`[N, K]`, or `[N, K/128, 128]`), load `[BN, KT]` tiles, and `reshape` in registers if the kernel needs the 32-element blocks.
