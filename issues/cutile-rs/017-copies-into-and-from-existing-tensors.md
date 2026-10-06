<!--
상태: 초안 (미게시)
대상: https://github.com/NVlabs/cutile-rs/issues (기능 요청)
확인 버전: vendor/cutile-rs a620b40 (main 9fe5756 기반), GB10 (sm_121, driver 580.178.04, CUDA 13.3 toolkit)
발견: firepit hot path 측정(88d3d91). decode logits를 읽을 때마다 `dup`(1 MB 할당 + DtoD)이 필요하고, step 입력 업로드마다 새 device tensor를 할당한 뒤 DtoD로 옮긴다
재현: 없음(기능 요청)
-->

# No safe host-to-device copy into an existing `Tensor`, and no device-to-host copy from a borrowed one

## Summary

- **HtoD:** `api::copy_host_vec_to_device(&Arc<Vec<T>>)` always allocates a new tensor. There is no safe operation that writes host data into a tensor that already exists.
- **DtoH:** `ToHostVecOp::to_host_vec` requires `DeviceOp<Output = Tensor<T>>`, so reading a tensor you only borrow means `dup()` first. That is a device allocation plus a device-to-device copy of the whole tensor.

Both matter for buffers whose address must stay fixed, such as the tensors a CUDA graph was captured with.

## Motivation (measured)

Decode, Qwen3.6-35B-A3B, batch 1, GB10, nsys, per step:

- **Logits read:** `cuMemAllocAsync` + `cuMemcpyDtoDAsync` (4.0 µs) + `cuMemcpyDtoHAsync` (29.6 µs) + `cuStreamSynchronize` + `cuMemFreeAsync`, about 37 µs in all. The `dup` is only there to obtain an owned tensor for `to_host_vec`. GPU time was 18.2 µs for the 1 MB DtoH and 8.4 µs for all five DtoD copies of the step.
- **Input upload:** for each of tokens, position(s), cos and sin, `copy_host_vec_to_device` (allocation + HtoD) then `memcpy` into the captured tensor. That came to about 58 µs per step with a synchronization after each, and five allocations per step in total.

## Suggested API

Mirroring `api::memcpy(&mut dst, &src)` (DtoD):

```rust
api::memcpy_from_host(&mut tensor, &host_slice)   // HtoD into an existing tensor, sizes checked
api::memcpy_to_host(&mut host_vec_or_slice, &tensor) // DtoH from a borrowed tensor
```

Each would be a `DeviceOp` that keeps the borrows for its lifetime, so a later `sync_on` or `zip!` can batch them. The HtoD op could additionally be a `GraphNode` when the source is pinned memory (see the graph-capture request, 016).

## Workaround

- HtoD: allocate a temporary tensor, then `memcpy` it into the destination. We now zip the copies so a step pays two synchronizations instead of 2 + 2·batch.
- DtoH: `dup()` then `to_host_vec()`, paying an extra allocation and a full-size DtoD copy each time.
