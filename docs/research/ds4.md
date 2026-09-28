# ds4 (DwarfStar)

- 저장소: https://github.com/antirez/ds4 — Redis를 만든 antirez(Salvatore Sanfilippo)의 프로젝트
- 조사 시점 main: `0aaea5a` (2026-09-20)

## 정체성

- "소수의 뛰어난 LLM을 소비자 하드웨어에서 가장 잘 돌리는 방법"이 목표다. 범용 GGUF runner가 아니고, 프로젝트가 직접 만든 GGUF만 쓴다. [확인: README]
- Metal, CUDA, ROCm을 지원하고 CUDA는 "DGX Spark가 main goal"이다. [확인: README]
- 이름은 DwarfStar, 저장소와 실행 파일은 `ds4`다. 실행 파일은 `ds4`, `ds4-server`, `ds4-agent`, `ds4-bench`로 나뉜다. `ds4`가 DeepSeek V4와 겹치는 것은 추정이고, 이름의 유래는 설명되어 있지 않다. [확인: README, 블로그 / 추정: 유래]
- 철학: "소프트웨어는 모든 경우를 다루려 하지 말고, 가장 중요한 경우를 위한 작동하는 템플릿(rail)이어야 한다." AI 코딩 에이전트의 도움을 크게 받아 개발했다고 밝힌다. [확인: README]
- 커널은 별도 라이브러리로 떼지 않고 엔진 안에 둔다(`ds4_cuda.cu`, `ds4_qwen4_cuda.cuh`). [확인]

## Qwen3.8-Flash-Next 지원

- CUDA: antirez가 9/14~15에 직접 커밋했다(`df6eedd` 이후 최적화 커밋 여러 개). `ds4_qwen4_cuda.cuh`는 2,344줄이다. [확인]
- Metal: ivanfioravanti의 PR #991이 9/16 merge됐다. [확인]
- **CUDA는 세션을 순서대로 decode한다.** Metal만 `--batched-session N`으로 여러 세션을 함께 decode한다. Qwen batching 코드는 모두 `#ifdef DS4_HAS_QWEN4_METAL` 안에 있다. [확인: docs, 코드]
- GPU API: `ds4_gpu.h`에 Qwen 함수가 56개 선언돼 있고, Metal은 56개, CUDA는 39개를 구현한다. CUDA에 없는 17개는 모두 batched 변종이다(`gdn_scan_rows`, `batch_mm_q8`, `moe_*_grouped` 등). [확인]
- 모델 파일: HF `antirez/qwen3.8-flash-next-gguf`. Q2는 137.10 GiB(상주 41.73 GiB), Q4는 165.11 GiB(상주 69.74 GiB)이다. 95.37 GiB BF16 n-gram table은 SSD에 두고 필요한 행만 읽는다. [확인]
- Q2 expert는 IQ2_XXS gate/up과 Q2_K down, Q4 expert는 Q4_K gate/up과 MXFP4 down이다. dense는 Q8_0/F16이다. [확인]
- Spark 수치(docs): Q2 decode 22.6 t/s, Q4 21.2 t/s @8K, prefill 약 516~771 t/s. [확인]

## 기여 규칙 (CONTRIBUTING.md, AGENT.md)

- backend를 건드리는 PR은 correctness와 speed 회귀가 없음을 보여야 한다. `ds4-bench` before/after CSV가 필요하다. 속도 저하는 중요한 correctness 수정일 때만 허용한다. [확인]
- C만 쓴다(C++ 금지). "small, sharp, elegant, no slop"을 요구한다. [확인]
- 조사 시점 open issue는 274개였다. [확인]

## firepit과 관련된 issue

| issue | 내용 |
|---|---|
| #773 | GB10 DeepSeek decode가 대역폭 floor의 85~90%. 측정 방법론의 선례 |
| #1025 | GB10에서 Q4 sorted-pairs MoE 경로가 NaN을 더해서 42% 확률로 출력이 망가짐 |
| #1133 | Anthropic API + Qwen에서 system message가 누락되어 prefill을 처음부터 다시 함 (PR #1134) |
| #999 | Qwen과 GLM tool-call 문법에 decode 시점 보호가 없음 |
| #759, #847, #881, #734 | CUDA illegal memory access |

## fork

- **Entrpi/ds4-on-spark**: Blackwell CUDA 성능 fork. "prefill 약 3배, decode 약 1.5배, 연속 batching 지원"이라고 소개한다. 마지막 push는 8/27로, Qwen 지원 이전이다. [확인: repo 설명]
- **Baekpica/ds4-dfm-rs**: Rust host에 C/CUDA 엔진을 붙인 fork. 11개 모델 계열을 다룬다. Qwen3.8 Q5 기준 prefill 약 1,420~1,516 tok/s, decode 약 28~29.5 tok/s(MTP 포함)이고 2 bank를 지원한다. bank가 batched decode인지는 확인하지 않았다. README에 "not a rewrite of ds4 in Rust"라고 적혀 있다. [확인: README, docs / 미확인: bank의 의미]

## 사용법 메모

- 빌드: `make cuda-spark`. Qwen 커널 테스트: `make test-qwen4-cuda CUDA_ARCH=sm_121`. [확인]
- Claude Code 연결: `ds4-server`의 Anthropic 호환 endpoint를 쓴다(`docs/CLIENTS.md`). 모델 alias는 `qwen3.8-flash-next`이다. [확인: docs / 직접 연결해 보지는 않음]
