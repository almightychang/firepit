<!--
상태: 초안 (미게시)
대상: https://github.com/NVlabs/cutile-rs/issues (주의: lowering은 tileiras에서 일어난다. cutile-rs 소스 차이는 루프 안 [1, 1] load와 곱 하나뿐이다. 게시처는 Sam 확인 필요)
확인 버전: vendor/cutile-rs a620b40 (main 9fe5756 기반), GB10 (sm_121, driver 580.178.04, CUDA 13.3 toolkit, tileiras V13.3.36)
발견: firepit decode GEMV 잔여분 분석 3~7단계. FP8 128 블록 scale GEMV가 작은 행 타일에서 반복당 BAR 26개짜리 무거운 루프가 됨. 7단계에서 [1, K] 입력·3D 입력·scale 유무로 나눠 유발 조건을 반복당 [1, 1] scale load로 좁힘
재현: 011-fp8-scale-tile-loop-probe.rs (crates/firepit-kernels/examples/fp8_gemv_sync_probe.rs, 13b9174)
-->

# A `[1, 1]` scale tile loaded inside a GEMV K loop doubles the loop's synchronization at small row tiles

## Summary

In a GEMV-shaped loop (load a `[BN, 128]` FP8 weight tile, convert, multiply by the input, reduce over K), loading one `f32` scale per iteration through a `[1, 1]` partition and multiplying it into the tile's row sums changes the generated loop at small `BN`.

At `BN = 8`:
- The loop grows from 681 to 1333 instructions.
- `BAR.SYNC` goes from 14 to 26 and `MEMBAR` from 13 to 25.
- The per-iteration time grows by 25% (4.66 → 5.80 µs on an 8192 × 2048 matrix).

At `BN = 128` the same change costs nothing measurable, because the per-iteration bytes hide it. For small row tiles the extra synchronization dominates.

## Reproduction

Full program: `011-fp8-scale-tile-loop-probe.rs` (attached). It is run as `fp8_gemv_sync_probe N K BN`, one row tile per process, and timed with nsys. Each launch follows a read of a 64 MiB buffer to evict L2 with clean lines.

The only difference between the two kernels compared below is the scale:

```rust
// fp8_dot: no scale
for kc in 0i32..chunks {
    let weights: Tile<f8e4m3fn, { [BN, KT] }> = wp.load([pid.1, kc]);
    let weights: Tile<f32, { [BN, KT] }> = convert_tile(weights);
    let xb: Tile<f32, { [1, KT] }> = xp.load([0, kc]);
    let part: Tile<f32, { [BN] }> = reduce_sum(weights * xb.broadcast(shape![BN, KT]), 1i32);
    acc = acc + part.reshape(shape![BN]);
}

// fp8_dot_scaled: plus one f32 per 128x128 block
for kc in 0i32..chunks {
    let scale: Tile<f32, { [1, 1] }> = sp.load([scale_row, kc]);   // sp = scales.partition(shape![1, 1])
    let weights: Tile<f8e4m3fn, { [BN, KT] }> = wp.load([pid.1, kc]);
    let weights: Tile<f32, { [BN, KT] }> = convert_tile(weights);
    let xb: Tile<f32, { [1, KT] }> = xp.load([0, kc]);
    let part: Tile<f32, { [BN] }> = reduce_sum(weights * xb.broadcast(shape![BN, KT]), 1i32);
    let part: Tile<f32, { [BN] }> = part.reshape(shape![BN]);
    acc = acc + part * scale.reshape(shape![1]).broadcast(shape![BN]);
}
```

Results on GB10, N = 8192, KT = 128. Kernel times are nsys medians of 8 launches. Per-iteration time and intercept come from a fit over K = 512, 2048, 8192 (iterations = K / 128).

| BN | byte-sum floor | `fp8_dot` | `fp8_dot_scaled` |
|---:|---|---|---|
| 128 | 1.5 + 4.63 µs/it | 2.2 + 4.66 µs/it | 2.3 + 4.66 µs/it |
| 64 | 1.8 + 4.62 | 3.3 + 4.67 | 3.2 + 4.68 |
| 32 | 1.5 + 4.53 | 2.6 + 4.73 | 2.9 + 4.73 |
| 16 | 3.7 + 4.46 | 7.3 + 4.61 | 9.5 + 4.63 |
| 8 | 13.8 + 4.24 | 25.2 + 4.66 | 24.8 + **5.80** |

At K = 2048 and BN = 8: floor 74.2 µs, `fp8_dot` 99.9 µs, `fp8_dot_scaled` **114.5 µs**. At K = 8192: 286.4, 323.0, **396.6** µs.

Main K loop in SASS (`cuobjdump -sass`, the largest backward-branch region; `-res-usage`):

| BN | kernel | loop instructions | `BAR.SYNC` | `LDS` | `SHFL` | `MEMBAR` (whole kernel) | REG / STACK |
|---:|---|---:|---:|---:|---:|---:|---|
| 8 | `fp8_dot` | 681 | 14 | 70 | 24 | 13 | 255 / 32 |
| 8 | `fp8_dot_scaled` | **1333** | **26** | 136 | 48 | **25** | 255 / 64 |
| 32 | `fp8_dot` | 1068 | 12 | 74 | 30 | 11 | 255 / 32 |
| 32 | `fp8_dot_scaled` | 1303 | 14 | 88 | 36 | 13 | 255 / 64 |
| 128 | `fp8_dot` | 2597 | 14 | 100 | 96 | 13 | 255 / 32 |
| 128 | `fp8_dot_scaled` | 2566 | 14 | 100 | 96 | 13 | 255 / 64 |

All kernels produce the expected values (every row sums to K).

## Where it seems to come from

Verified:
- The source difference is one `load` of a `[1, 1]` partition and one multiply per iteration.
- A third variant that reads the input as `[1, K/32, 32]` and reshapes it per tile has the same time and SASS as `fp8_dot`. So the input layout is not involved.

Not verified (reading SASS, no counters): at `BN = 8` the loop body looks like it gets a second consumer wait/fence sequence for the scale load. That would explain the doubled `BAR.SYNC`/`MEMBAR` and the duplicated `LDS`/`SHFL` work. With only `8 × 128` bytes of weights per CTA per iteration, this fixed synchronization is not hidden.

The Tile IR is produced by cutile-rs, but the loop structure is chosen in `tileiras`. So this report is mostly for the backend. We would appreciate routing.

## Suggested fix

- In the backend: avoid a full stage synchronization for a load that small (a plain cached load, or hoisting it across iterations).
- A note in the docs that tiny per-iteration loads are expensive at small tile sizes would also help.

## Workaround

- Avoid per-iteration scalar loads at small row tiles.
- When the scale is uniform along K (a per-tensor or per-row scale), apply it once after the loop. In our full FP8 GEMV on a 512 × 2048 matrix at BN 8, this "flat loop with a row-scale vector" took 10.8 µs, against 13.1 µs for the same GEMV with a per-iteration 128-block scale.
- For 128-block scales, loading the block row of scales once before the loop is another option we have not measured.

## Related

The CTA-count effect visible in the intercept column (13.8 µs for the byte-sum floor at BN = 8 with 1024 CTAs, 1.5 µs at BN = 128 with 64 CTAs) affects all kernels, including the floor, and is not part of this report.
