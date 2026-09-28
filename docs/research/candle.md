# candle

- 저장소: https://github.com/huggingface/candle — 로컬 clone은 `~/Repos/candle` (`5ba5d5b`, 2026-09-28)
- README의 목표: "serverless inference를 가능하게". 공개 로드맵은 없다. [확인]

## 양자화 matmul dispatch (`candle-core/src/quantized/cuda.rs:846-860`)

```
QCudaStorage::fwd
 ├─ fast_mmvq::try_fwd   batch ≤ 8, Q4_0~Q6K (Q8_0 포함), 입력 bf16/f16/f32
 ├─ fast_mmq::try_fwd
 └─ 기존 dmmv / cuBLAS 경로 (FORCE_DMMV면 이 경로만)
```

- `try_fwd`가 `Ok(None)`을 반환하면 다음 단계로 넘어간다. 새 경로를 맨 앞에 끼우기 쉬운 구조다. [확인]
- `fast_mmvq`의 Q8_0 경로는 llama.cpp 방식이다. 활성값을 Q8_1로 양자화한 뒤 `dp4a`로 곱한다. batch 1이면 4 warp가 한 행을 맡는다(`candle-kernels/src/mmvq_gguf.cu`, 1,049줄). [확인]
- 가중치는 GGUF block을 그대로 GPU에 올리고, 512 원소 단위로 padding한다(`PaddedCudaSlice`). [확인]
- GB10 실측: floor의 88~96%. [확인: [measurements/candle-q8-baseline.md](measurements/candle-q8-baseline.md)]

## 지원 양자화 타입

`GgmlDType`: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2K~Q6K, Q8K. **IQ 계열(IQ2_XXS 등)과 MXFP4는 없다.** 그래서 ds4의 Qwen3.8 Q2 GGUF를 로드할 수 없다. IQ4_XS를 추가하는 PR(#3462)이 4월부터 open 상태다. [확인]

## 모델 지원

`candle-transformers`의 Qwen 계열은 qwen2, qwen3, qwen3_moe, qwen3_vl(과 각각의 quantized 버전)이다. GDN 구현은 없다. Qwen3.8-Flash-Next는 돌아가지 않는다. [확인]

## cuTile 통합 현황

- 8/31 PR #3934(Eric Buehler, "Initial integration of cutile and fused MoE kernel")로 opt-in feature `cutile`이 생겼다. 가이드는 `candle-book/src/guide/cutile.md`에 있다. [확인]
- cuTile 버전은 `=0.3.1`로 고정되어 있다. DGX Spark 지원은 0.4.0에 들어갔다. [확인]
- 연결 방식: `CutileContext`로 candle storage를 cuTile pointer로 빌려 온다. CustomOp에서 호출한다. [확인]
- 기존 cuTile 커널은 `candle-nn/src/moe/cutile.rs`의 BF16 routed grouped matmul 하나다. `unsafe fn`, raw pointer, `unchecked_accesses = true`로 짜여 있다. `moe.rs`의 dispatch에는 연결되어 있지 않고 공개 함수로만 노출된다. [확인]
- 양자화 경로에는 cuTile이 없다. [확인]
- Spark 실측: CUDA 13.3으로 빌드한 candle PTX 커널은 호스트 580 드라이버에서 `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`로 로드되지 않았다. forward-compat libcuda로는 동작했다. clean 빌드 70 s, 커널 파일 하나만 고친 incremental 빌드 19 s. [확인]

## 메인테이너와 리뷰 병목

- star 약 2.1만, open issue와 PR 합계 909개, open PR 300개. [확인]
- 최근 3개월 merge 60건을 누가 merge했는지 세면 ivarflakstad 46, EricLBuehler 11, 기타 3이다. PR에 리뷰 코멘트를 남긴 사람은 세지 않았다. [확인]
- **ivarflakstad는 cuTile-rs 기여자이기도 하다.** [확인]
- 최신 모델 지원 PR이 몇 달째 open이거나 closed됐다. Qwen3.5 (#3396, #3461 open, #3821과 #3869 closed), GDN CUDA 커널(#3890, 8/14 open), GDN Metal 커널(#3897, #3906), DeepSeek-V3와 Llama 4(closed). [확인]
- 해석: 수요와 기여자는 있지만 리뷰할 사람이 부족하다. [추정]

## firepit에서의 위치

큰 기여 대상으로 삼지 않는다. 필요할 때 작은 PR만 올린다(예: cuTile 0.4.0 bump). 가중치 로더, 토크나이저 연동처럼 가져다 쓸 수 있는 부분은 참고한다.
