<!--
상태: 초안 (미게시)
대상: https://github.com/NVlabs/cutile-rs/issues (주의: 원인은 tileiras 쪽으로 보인다. cutile-rs가 내는 Tile IR은 BN별로 shape 숫자만 다르다. 게시처는 Sam 확인 필요)
확인 버전: vendor/cutile-rs a620b40 (main 9fe5756 기반), GB10 (sm_121, driver 580.178.04, CUDA 13.3 toolkit, tileiras V13.3.36)
발견: firepit GEMV [행, K] view 변경(0a29c40) 검증에서 NVFP4 GEMV가 BN=64·32일 때만 느려짐. 사다리 probe(nvfp4_lowering_probe, d1f64ee)로 유발 단계를 좁힘
재현: 008-nvfp4-unpack-reduce-probe.rs
-->

# FP4 `unpack` followed by a row `reduce_sum` round-trips through shared memory when the tile has fewer than 128 rows

## Summary

Loading a packed FP4 tile `[BN, KT/2]`, unpacking it to `[BN, KT]`, converting to `f32` and summing each row compiles to a register-only loop when `BN >= 128`. For `BN = 64, 32, 16` the same kernel re-packs the unpacked values to E2M1 (`F2FP.SATFINITE.E2M1.F32`), exchanges them byte by byte through shared memory (`STS` / `LDS.U8`) with a named barrier between steps (`BAR.SYNC 0x1, 0x80`), and runs about 2x slower for the same bytes. Unpacking and storing the tile without the reduction stays in registers at every `BN`.

## Reproduction

Full program: `008-nvfp4-unpack-reduce-probe.rs` (attached), run as `nvfp4_unpack_reduce_probe 8192 2048 BN 128`.

```rust
#[cutile::entry()]
pub fn unpack_row_sum<const BN: i32, const KT: i32, const KT_PACKED: i32>(
    y: &mut Tensor<f32, { [1, BN] }>,
    w: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
    chunks: i32,
) {
    let pid: (i32, i32, i32) = get_tile_block_id();
    let wp = w.partition(shape![BN, KT_PACKED]);
    let mut acc: Tile<f32, { [BN] }> = constant(0.0f32, shape![BN]);
    for kc in 0i32..chunks {
        let packed: Tile<f4e2m1fnx2, { [BN, KT_PACKED] }> = wp.load([pid.1, kc]);
        let values: Tile<f4e2m1fn, { [BN, KT] }> = packed.unpack(shape![BN, KT]);
        let values: Tile<f32, { [BN, KT] }> = convert_tile(values);
        let part: Tile<f32, { [BN] }> = reduce_sum(values, 1i32);
        acc = acc + part.reshape(shape![BN]);
    }
    y.store(acc.reshape(shape![1, BN]));
}
```

Results on GB10, 8192 x 2048 FP4 (8 MiB packed), KT = 128, median of `cutile::bench::do_bench`. SASS from the JIT cache cubin (`cuobjdump -sass`, `-res-usage`). Both kernels produce the expected values at every `BN`.

| kernel | BN | ms | instructions | `LDS.U8` | `F2FP.SATFINITE.E2M1.F32` | `BAR.SYNC` | `STS` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `unpack_row_sum` | 128 | 0.093 | 1272 | 0 | 0 | 1 | 0 |
| `unpack_row_sum` | 64 | **0.199** | 5616 | 252 | 254 | 517 | 324 |
| `unpack_row_sum` | 32 | **0.147** | 2472 | 122 | 125 | 261 | 133 |
| `unpack_store` | 128 | 0.381 | 1456 | 0 | 0 | 6 | 32 |
| `unpack_store` | 64 | 0.382 | 760 | 0 | 0 | 1 | 16 |
| `unpack_store` | 32 | 0.380 | 328 | 0 | 0 | 1 | 8 |

(`unpack_store` writes 64 MiB of f32, so its time is the store.) Registers, stack and local memory do not explain it: `LOCAL:0` and no `LDL` in every variant.

The trigger is the row count, not the tile size. With the same probe at other tile shapes (`BN x KT`):

| BN x KT | elements | shared-memory round trip in `unpack_row_sum` |
|---|---:|---|
| 128 x 64, 128 x 128, 128 x 256, 256 x 64 | 8192 – 32768 | no |
| 64 x 64, 64 x 128, 64 x 256 | 4096 – 16384 | yes |
| 32 x 128, 32 x 512 | 4096 – 16384 | yes |
| 16 x 128, 16 x 1024 | 2048 – 16384 | yes |

The loop body at `BN = 64` (abridged):

```text
LDS.U8 R0, [R3]
F2FP.F16.E2M1.UNPACK_B R0, R0
STS [R2+0x2000], R5
BAR.SYNC.DEFER_BLOCKING 0x1, 0x80
LDS.U16 R4, [R0+0x2000]
F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C R4, R4, R4, RZ
F2FP.F16.E2M1.UNPACK_B R4, R4
FADD R79, RZ, R4
```

In a full NVFP4 GEMV (per-16 E4M3 scales and an input vector, KT = 128) over the same 8192 x 2048 weights, the time goes from 0.102 ms at BN = 128 to 0.227 ms at BN = 64 and 0.193 ms at BN = 32, with the same SASS pattern.

## Where it seems to come from

Verified (`CUTILE_DUMP=ir`): the Tile IR for `BN = 128` and `BN = 64` differs only in the shape constants. `unpack` is emitted as a flat 1-D op between two reshapes:

```text
%80 = reshape %74 : tile<64x64xi8> -> tile<4096xi8>
%81 = unpack %80 : tile<4096xi8> -> tile<8192xf4e2m1fn>
%82 = reshape %81 : tile<8192xf4e2m1fn> -> tile<64x128xf4e2m1fn>
%83 = ftof %82 : tile<64x128xf4e2m1fn> -> tile<64x128xf32>
%87 = reduce %83 dim=1 identities=[0] : tile<64x128xf32> -> tile<64xf32>
```

So the shared-memory exchange is chosen after cutile-rs, in `tileiras`. The flatten is required on the cutile-rs side: `cutile-ir/src/verify_target.rs` (around line 214) only accepts rank-1 source and result for `Pack` / `Unpack`. `compile_fp4_pack_unpack` in `cutile-compiler/src/compiler/compile_intrinsic.rs` builds the flat types accordingly.

Not verified: our guess is that the layout `tileiras` gives the flat `unpack` result matches the row-reduction layout only when every thread owns whole rows (128 threads, `BN >= 128`). Below that it converts the layout through shared memory, and the conversion is done on re-packed FP4 bytes.

## Suggested fix

This looks like a `tileiras` layout-assignment issue rather than a cutile-rs bug, so we would appreciate it if you could route it to the right team. On the cutile-rs side, two things could help if the Tile IR allows them:

- An `unpack` that keeps the tile rank (unpack along the last dimension), so the backend does not have to recover the 2-D layout from a flat value.
- A note in the FP4 docs on which tile shapes avoid the round trip, until the backend handles them.

## Workaround

Use at least 128 rows per tile for kernels that reduce an unpacked FP4 tile along a row. In our full GEMV body, a larger K tile at small BN (64 x 256, 32 x 512, 16 x 1024) also avoided the round trip. That did not hold for the minimal kernel above, so we do not rely on it.
