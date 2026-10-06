<!--
상태: 초안 (미게시)
대상: https://github.com/NVlabs/cutile-rs/issues (성능 기능 요청)
확인 버전: vendor/cutile-rs a620b40 (main 9fe5756 기반), GB10 (sm_121, driver 580.178.04, CUDA 13.3 toolkit)
발견: firepit JIT 지연 작업(f60d2e9 GB10 검증, 2026-10-07). `firepit run`에 디스크 캐시를 켜자 두 번째 실행에서 tileiras가 0회가 됐지만, stage1 합계 약 4.6 s가 남았다(커널 63개)
재현: 없음(기능 요청). 측정은 `CUTILE_JIT_TIMING=1`로 같은 프로그램을 두 번 실행
-->

# The disk JIT cache stores only the cubin, so every process repeats the front-end (stage 1)

## Summary

With `jit_cache::enable(...)`, a second process serves every `tileiras` result from disk. It still runs stage 1 (kernel AST → Tile IR → bytecode, plus the launch validator) for every kernel. The reason is that the disk key is derived from the stage 1 output itself.

In our application that is about 4.6 s of the remaining start-up cost on GB10, for 63 kernels. We suggest caching stage 1 too, under a key that also covers the dependency modules.

## Motivation (measured)

`firepit run` (Qwen3.6-35B-A3B decode and prefill, 63 distinct kernels), `CUTILE_JIT_TIMING=1`, fresh cache directory, GB10. Sums over the `CUTILE_JIT_TIMING` lines:

| run | `tileiras` / disk hits | stage 1 sum | stage 2 sum |
|---|---|---:|---:|
| 1st (cold) | 63 / 0 | 4142 ms | 22174 ms |
| 2nd (same input) | 0 / 63 | 4586 ms | 51 ms |

The disk cache removes stage 2 as intended. Stage 1, about 70 ms per kernel on average, becomes the largest JIT cost left on every start.

## Where it comes from

Read in the code (vendor `a620b40`); the timings above are measured, but this explanation was not verified with a patch:

- `cutile/src/tile_kernel.rs`, `compile_from_context`: on an in-memory (L1) miss it always constructs `CUDATileFunctionCompiler` and calls `compiler.compile()` (stage 1, timed as `stage1_ms`). Only then does it serialize bytecode and call `compile_bytecode_cached`. That function derives the disk key from the bytecode (`jit_cache::l2_key`, SHA-256 over the bytecode, target, options, and `tileiras` fingerprint).
- The L1 key cannot serve as a disk key. The comment on `TileFunctionKey` says: "`source_hash` covers the kernel's own module, not the dependency modules the use-graph links in. … it is why the on-disk cache keys on the serialized bytecode rather than on this struct."
- Stage 1 also produces the `Validator`: the launch checks the generated launcher enforces, captured "AFTER compile" because the accumulator fills during compilation. So a cached stage 1 result has to include the validator, not only the bytecode.

## Suggested fix

Add a stage 1 disk entry in front of the existing cubin cache:

1. **Key:** the existing `TileFunctionKey` fields, plus a hash that covers every module the kernel links. That means the kernel's module and the dependency modules found by walking `use` against the linker registry, plus the compiler version.
   - The registry already holds each module's source for compilation, so hashing it is cheap compared with compiling.
   - Alternatively, use one hash over all registered modules. That is coarser but simple, and any source change still misses.
2. **Value:** the serialized bytecode and the serialized validator (launch checks), with the same integrity header as cubin entries.
3. **Lookup:** on an L1 miss, try the stage 1 entry. On a hit, feed its bytecode to the existing `compile_bytecode_cached` (a stage 2 disk hit in steady state) and install the validator.

On a warm start this would leave stage 3 (module load) and a few hash computations per kernel. We have not measured how much of our 4.6 s is AST construction versus lowering. A patch would show that.

`cutile::tune` already persists L2 keys outside the cache, with a schema version (`L2_KEY_SCHEMA`). A stage 1 key would need the same versioning discipline.

## Workaround

None on the user side. Disk caching cubins (stage 2) is what we do today. Stage 1 still runs once per kernel per process.
