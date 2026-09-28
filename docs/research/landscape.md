# DGX Spark 로컬 LLM 생태계 (2026-09 기준)

## GB10 하드웨어 특성

- sm_121, 128 GB LPDDR5X unified memory, 대역폭 스펙 약 273 GB/s. [확인: 여러 자료]
- 실측 지속 read 대역폭은 약 200 GB/s(큰 buffer), peak 약 235 GB/s. [확인: [측정](measurements/ds4-qwen38-decode-profile.md)]
- datacenter Blackwell(sm_100)과 다르다. tcgen05, multicast, 2-SM MMA가 없고 shared memory는 약 101 KB다. [확인: NVIDIA forum 359960]
- 그래서 sm_100용 커널이 그대로 돌지 않고, sm_121 지원이 늦게 들어오는 경우가 많다. vLLM에는 sm_121 관련 open issue가 여러 개 있다(#45317 sparse MLA backend 없음, #43906 MXFP8 MoE가 Marlin으로 fallback). [확인]
- NVFP4가 FP8보다 느리다는 보고(ai-muninn: Qwen3.6-35B-A3B에서 FP8 53.8, NVFP4 40.8 tok/s)와 NVFP4 GEMM 356 TFLOPS 보고(forum)가 엇갈린다. [확인: 두 보고 / 원인은 미확인]

## 로컬 에이전트 사용 현황

- NVIDIA 공식 playbook이 Ollama + Qwen3.6 + Claude Code / OpenCode / Codex CLI 구성을 안내한다. [확인]
- 에이전트 workload는 prefill 비중이 크다. 한 실사용 사례에서 평균 prompt 78.8k tokens, prefix cache hit 95%였다(devashish.me, Qwen3.6-27B-FP8 + vLLM). [확인]
- 커뮤니티 흐름: 5월 ds4가 DeepSeek-V4-Flash를 Spark 1대에서 2-bit로 돌렸고, 8/26 Qwen3.8-Flash-Next가 나온 뒤로 Spark 1대의 기본 모델처럼 쓰이고 있다. [확인: 포럼, 블로그]

## Spark 1대에서 쓸 만한 모델 (9월 말)

| 모델 | 구성 | decode | 비고 |
|---|---|---|---|
| Qwen3.8-Flash-Next | vLLM NVFP4 (MiaAI 레시피) | ~48.7 tok/s, prefill ~2,314 tok/s @32K, 8 stream 162.9 tok/s | community patch 필요 [확인: repo] |
| Qwen3.8-Flash-Next | llama.cpp UD-Q3_K_XL + MTP | 78.9 tok/s (code), 2명 동시 | 실험 branch [확인: repo] |
| Qwen3.8-Flash-Next | ds4 Q2 | 22.6 t/s (docs), 19.5 t/s (우리 실측) | CUDA는 세션 순차 처리 [확인] |
| DeepSeek-V4-Flash | ds4 2-bit | 24~28 t/s | 단일 stream [확인: forum] |
| GLM-5.3-Flash (320B/18B) | llama.cpp 2-bit | 17.7 tok/s | EXL3 2-bit은 약 10K token 뒤 crash 보고 [확인] |
| Qwen3.6-35B-A3B | 공식 playbook | — | 공식 경로가 가장 잘 갖춰짐 [확인] |

## Spark에서 막혀 있는 모델 (vLLM upstream 기준)

- GLM-5.1, DeepSeek-V3.2 계열: sparse MLA backend 없음 (#45317). [확인]
- GLM-5.3-Flash: SM120 sparse MLA 커널이 NoPE MLA에서 crash. fork 패치 8겹, Spark 2대 필요. [확인]
- DeepSeek-V4.1-Flash: fork에서 Spark 4대로 동작. [확인]
- Kimi K3 (2.8T): 용량상 불가능. [확인]

## VLA 쪽

FlashRT(손으로 짠 CUDA 커널 + CUDA Graph)가 Thor에서 Pi0.5 23 Hz, 5090에서 57~76 Hz를 냈다. GB10은 LLM 벤치만 있고 VLA 수치는 없다. [확인: forum, repo]
