<!--
상태: 초안 (미게시)
대상: https://github.com/NVlabs/cutile-rs/issues (주의: lowering은 tileiras에서 일어난다. 008과 같은 FP4 unpack 경로라 008과 묶어 내는 것을 제안. 게시처는 Sam 확인 필요)
확인 버전: vendor/cutile-rs a620b40 (main 9fe5756 기반), GB10 (sm_121, driver 580.178.04, CUDA 13.3 toolkit, tileiras V13.3.36)
발견: firepit decode GEMV 잔여분 분석 4~7단계. NVFP4 expert GEMV(BN 128)가 같은 바이트의 load만 대비 K 2048에서 약 +10 µs. 정수 디코드·f16 경유·pipeline hint로 줄지 않음. 7단계에서 unpack·그룹 합·scale 곱으로 나눠 scale 곱 단계로 좁힘
재현: 012-fp4-group-scale-probe.rs (crates/firepit-kernels/examples/fp4_unpack_cost_probe.rs, 13b9174)
-->

# Multiplying unpacked FP4 group sums by a per-group scale tile triples the loop's conversions and shared-memory reads

## Summary

The NVFP4 matrix-vector body has these steps, with `BN = 128` rows and K tiles of 128 values:
1. Load packed bytes.
2. `unpack` to E2M1 and convert to `f32`.
3. Sum groups of 16.
4. Multiply by the E4M3 group scales.

The first three steps run at the memory floor's per-iteration time (within 4%). Adding the scale multiply (a `[BN, KT/16]` E4M3 tile loaded and converted per iteration) changes the loop:
- It grows from 634 to 1800 instructions.
- `F2FP` conversions go from 128 to 408 and `LDS` from 40 to 112.
- The per-iteration time grows by 21%.

The scale tile is 1/16 of the packed weight bytes, so the extra cost is not memory traffic.

## Reproduction

Full program: `012-fp4-group-scale-probe.rs` (attached). It is run as `fp4_unpack_cost_probe N K BN` and timed with nsys. Each launch follows a read of a 64 MiB buffer to evict L2 with clean lines.

```rust
// fp4_groups: unpack, convert, sum groups of 16
let packed: Tile<f4e2m1fnx2, { [BN, KT_PACKED] }> = wp.load([pid.1, kc]);
let values: Tile<f4e2m1fn, { [BN, KT] }> = packed.unpack(shape![BN, KT]);
let values: Tile<f32, { [BN, KT] }> = convert_tile(values);
let groups: Tile<f32, { [BN, KT_GROUPS] }> = reduce_sum(values.reshape(shape![BN, KT_GROUPS, 16]), 2i32);
acc = acc + groups.reshape(shape![BN, KT_GROUPS]);

// fp4_scaled: the same, then times the group scales and summed per row
let groups: Tile<f32, { [BN, KT_GROUPS] }> = groups.reshape(shape![BN, KT_GROUPS]);
let scale: Tile<f8e4m3fn, { [BN, KT_GROUPS] }> = sp.load([pid.1, kc]);
let scale: Tile<f32, { [BN, KT_GROUPS] }> = convert_tile(scale);
let part: Tile<f32, { [BN] }> = reduce_sum(groups * scale, 1i32);
acc = acc + part.reshape(shape![BN]);
```

Results on GB10, N = 9216, BN = 128, KT = 128. Kernel times are nsys medians of 8 launches. The fit is over K = 512, 1024, 2048, 4096 (iterations = K / 128).

| kernel | K = 512 | 1024 | 2048 | 4096 | fit | 18432 × 512 |
|---|---:|---:|---:|---:|---|---:|
| packed byte sum (floor) | 12.80 µs | 23.34 | 46.35 | 89.07 | 1.9 + 2.73 µs/it | 22.61 |
| `fp4_unpack` (row sum) | 16.94 | 28.35 | 50.58 | 96.72 | 5.4 + 2.85 µs/it | 26.69 |
| `fp4_groups` (no scale) | 15.49 | 25.94 | 46.24 | 92.35 | 3.8 + 2.75 µs/it | 25.36 |
| `fp4_scaled` | 16.90 | 30.32 | 53.30 | 110.02 | 2.9 + **3.32 µs/it** | 28.27 |

Main K loop in SASS (`cuobjdump -sass`, K = 2048 build; the generated code does not depend on K):

| kernel | loop instructions | `F2FP` | `LDS` | `BAR.SYNC` | REG / STACK / shared |
|---|---:|---:|---:|---:|---|
| packed byte sum | 390 | 0 | 16 | 5 | 255 / 16 / 75.6 KB |
| `fp4_unpack` | 800 | 128 | 40 | 3 | 255 / 16 / 50.8 KB |
| `fp4_groups` | 634 | 128 | 40 | 3 | 255 / 16 / 26.2 KB |
| `fp4_scaled` | **1800** | **408** | **112** | 5 | 255 / 32 / 26.2 KB |

All kernels produce the expected values (every row sums to K).

In a full NVFP4 expert GEMV (the above plus the input vector), the same pattern accounts for most of the gap to the floor. What we tried earlier:
- Decoding E2M1 with integer bit operations instead of `unpack`: 9% slower.
- Converting through `f16`: identical SASS.
- Multiplying in `f16`: same time.
- `load_pipelined` hints: no consistent change.

## Where it seems to come from

Verified: the source difference between `fp4_groups` and `fp4_scaled` is the scale load, its conversion, and one multiply and reduce over `[BN, KT/16]`.

Not verified: the tripled `F2FP` count suggests the unpacked values are converted again, or moved through shared memory (`LDS` ×2.8), to match a layout chosen for the `[BN, KT/16]` scale tile. This is related to our other report on `unpack` followed by a row reduction at fewer than 128 rows (008). In both cases the layout picked for the unpacked FP4 tile does not match what the following operations want, and the backend fixes it up every iteration.

As with 008, the Tile IR differs only in what the source asks for. The layout decision is made in `tileiras`, so we would appreciate routing.

## Suggested fix

- In the backend: choose a layout for the unpacked tile that the group reshape and the scale tile can share, or convert the scale tile into the unpacked tile's layout instead of the reverse.
- On the cutile-rs side, an `unpack` that keeps the tile rank (unpack along the last dimension, see 008) may give the backend more room.

## Workaround

None found in the kernel source (see above).
