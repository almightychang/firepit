<!--
상태: 초안 (미게시)
대상: https://github.com/NVlabs/cutile-rs/issues
확인 버전: vendor/cutile-rs a620b40 (main 9fe5756 기반)
발견: firepit E3.4 (GDN chunkwise prefill의 단위 하삼각 풀이)
-->

# Safe way to replace one row (subtile) of a tile at a loop-variable index

## Summary

Updating one row of a register tile at a dynamic index needs `insert`, which is an `unsafe fn` (and requires Tile IR 13.4). In a codebase that keeps kernels free of `unsafe`, the only safe alternative we found is rebuilding the whole tile with an `iota`/`eq_tile` mask and `select`, which works but costs O(rows × cols) per row update.

## Use case

Forward substitution on a small unit lower-triangular matrix inside a tile, as in the chunkwise Gated DeltaNet algorithm (the same pattern as flash-linear-attention's `solve_tril` 16×16 kernel):

```text
for i in 2..16:
    a = row i of A           // extract: safe
    a = a + a · A_inv        // broadcast, multiply, reduce_sum: safe
    A_inv[i, :] = a          // insert: unsafe
```

## Current state

- `cutile/src/_core.rs` (`insert`): "Insert a subtile at slice indices, not element offsets. Tile IR 13.4+." with "# Safety — For each axis, `indices[axis]` must be in `0..(SOut[axis] / SIn[axis])` … Out-of-bounds indices are undefined behavior."
- `extract` is safe with the same kind of slice indices.

Our safe workaround (passes our f64-oracle tests):

```rust
let rows: Tile<i32, { [N, 1] }> = iota(shape![N]).reshape(shape![N, 1]);
let at_i = eq_tile(rows.broadcast(shape![N, N]), i.broadcast(shape![N, N]));
a_inv = select(at_i, new_row.broadcast(shape![N, N]), a_inv);
```

## Request

One of:

1. A safe `insert` variant that bounds-checks (or clamps and reports) dynamic indices, matching how `partition_mut().store()` became safe in 0.3.0 with bounds checks.
2. If that is not possible, document the `select`-mask pattern as the safe way to update a subtile, with its cost.
