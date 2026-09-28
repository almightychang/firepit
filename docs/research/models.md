# 대상 모델 구조 비교

HF `config.json`에서 확인한 값이다(2026-09-29). [확인]

| 모델 | model_type | hidden | layer 구성 | MoE | MTP | 추가 구성 요소 |
|---|---|---:|---|---|---|---|
| Qwen3.6-35B-A3B | `qwen3_5_moe` | 2048 | 40 = GDN 30 + full attention 10 | 256개 중 8개 | 1 | gated attention (`attn_output_gate`) |
| Qwen3.8-27B | `qwen3_5` (dense) | 5120 | 64 = GDN 48 + full attention 16 | 없음 | 1 | — |
| Qwen3.8-Flash-Next | `qwen4_exp` | 2560 | 48 = GDN 36 + full attention 12 | 512개 중 10개 | 1 | HC(`hc_count`, `hc_lowrank`), n-gram PLE(`ngram_size`, `ple_*`), indexer/QSA(`indexer_*`) |

## Qwen3.8-Flash-Next 추가 정보

- 본체 125B + n-gram embedding table 51B, active 6B다. [확인: kubesimplify]
- ds4 Q2 GGUF 기준 토큰당 가중치 read는 약 6.36 GB다. dense(Q8_0/F16)가 약 5.0 GiB로 대부분이고, routed expert 10개는 0.67 GiB다. 그래서 Q2와 Q4의 decode 속도가 비슷하다. [확인: [측정](measurements/ds4-qwen38-decode-profile.md), [정정](measurements/candle-q8-baseline.md)]
- dense 구성(토큰당 MiB): GDN projection 2,090, HC mixer(F16) 1,200, output head 644, attention projection 624, router(F32) 240, shared expert 239. [확인: GGUF header]
- 대표 Q8_0 shape(K × N): GDN qkv 2560 × 10240, attention q 2560 × 12288, GDN gate 2560 × 6144, GDN out 6144 × 2560, output head 2560 × 248320. [확인]

## GDN (Gated DeltaNet)

- 원 논문: Yang et al., "Gated Delta Networks: Improving Mamba2 with Delta Rule"(ICLR 2025, NVIDIA). [확인: 논문 존재 / 세부 인용은 기억에 의존]
- head마다 고정 크기 state `S`(d_k × d_v)를 토큰마다 갱신한다. `S_t = α_t · S_{t-1} · (I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ`, `o_t = S_t q_t`. 전치나 순서 표기는 구현마다 다르다.
- decode: 토큰마다 state 전체를 읽고, 고치고, 다시 쓴다. 순차 의존이 있다.
- prefill: 64-token chunk 안에서는 WY representation으로 tensor core를 쓰고, chunk 사이에서는 state를 순차로 넘긴다.
- GB10 실측에서 ds4의 GDN recurrence 커널은 decode 시간의 약 2%이고 이미 floor 수준이다. [확인]

## 모델 순서 결정

1단계는 Qwen3.6-35B-A3B, 2단계는 Qwen3.8-Flash-Next다. 이유는 이렇다.
- Qwen3.6은 Flash-Next의 뼈대(GDN, MoE, gated attention, MTP)를 갖고 있고, 특이한 부품(HC, PLE, QSA)만 빠져 있다.
- HF transformers, vLLM, llama.cpp가 모두 지원해서 정답 기준이 많다. Flash-Next는 사실상 ds4 하나에 의존해야 한다.
- 버그를 격리할 수 있다. 뼈대를 먼저 검증하면, Flash-Next에서 틀렸을 때 원인이 새로 추가한 부품으로 좁혀진다.

두 트랙을 병렬로 진행할 수도 있다. 공통 부품은 Qwen3.6으로, Flash-Next 전용 부품은 ds4의 layer 출력으로 따로 검증한다. 이 경우 부품 인터페이스를 먼저 고정해야 한다.
