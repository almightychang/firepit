# 출력 품질 잡음 바닥 (GB10, `20b735c`)

2026-10-05. firepit(f32 activation, bf16 가중치) vs HF transformers 5.17.0 bf16(activation 포함). WikiText-2 test, Qwen tokenizer, 4096토큰 묶음, 뒤 절반 채점, HF 상위 256개 기준. `examples/quality`.

| 모드 | 묶음 | 채점 위치 | KL 평균 (95% CI) | 중앙값 | p99 | p99.9 | 최대 | top-1 일치 (95% CI) | HF 나머지 질량 최대 | 시간 |
|---|---:|---:|---|---:|---:|---:|---:|---|---:|---:|
| prefill | 72 | 147,456 | 4.82e-3 ± 0.70e-3 | 1.44e-3 | 4.25e-2 | 0.240 | 14.8 | 97.06% ± 0.14 | 0.53 | 842 s |
| decode (teacher forcing) | 4 | 8,192 | 4.78e-3 ± 2.55e-3 | 1.55e-3 | 4.71e-2 | 0.203 | 1.05 | 97.11% ± 1.56 | 0.40 | 475 s |

- 같은 4묶음에서 prefill과 decode의 묶음별 KL이 거의 같다(예: 묶음 0 6.141e-3 vs 6.146e-3).
- 채점 구간을 4등분한 위치별 KL 평균은 4.5e-3~5.2e-3으로 위치에 따라 늘지 않는다.
- HF가 bf16 activation이라, 이 바닥이 HF 쪽 오차인지 firepit 쪽 오차인지는 이 측정만으로 구분할 수 없다.
- prefill 최대 KL 14.8인 위치와, 상위 256개 밖 질량이 0.5를 넘는 위치가 있다. 원인은 확인하지 않았다.
- 참고(조건이 달라 직접 비교 불가): llama.cpp LLaMA 3 8B q8_0 KLD 1.36e-3, R3 Qwen3-30B-A3B SGLang vs Megatron 1.5e-3.

## KL 상위 위치 조사: firepit 버그 아님 (2026-10-05)

KL 상위 위치에서 firepit이 문맥과 무관한 토큰을 높은 확률로 예측했다(`--top-kl`). 예: 묶음 69 2861행 "…executed in Wandsworth Prison in late 1942 . By September 194" → HF bf16 '2' 0.97, firepit ' frog' 0.85.

| 확인 | 결과 |
|---|---|
| decode(teacher forcing) | prefill과 같은 토큰·확률 (' frog' 0.852) |
| 누적 layer 비교 (HF bf16) | L15·L19(full attention)에서 hidden 차이가 크게 뜀, 라우팅 flip 동반 |
| 격리 layer 비교 (HF 입력을 layer 하나에) | attention 10개 전부와 GDN 4개 모두 mixer·MoE·hidden 상대 L2 2.6e-3~1.2e-2 (bf16 수준). L39는 라우팅 flip 1개 |
| **HF activation f32** (`tools/reference/hf_fp32_logits.py`) | frog ' frog' 0.860 (firepit 0.852), Dutch 0.8616 (0.8615), Bishop 0.7326 (0.7325), Plot 0.6556 (0.6467), 정상 위치 '1' 0.9995 (0.9996) |

결론: 독립 구현 두 개(HF f32, firepit f32)가 같은 값을 낸다. 이 출력은 모델을 f32 activation으로 계산한 결과이고, HF bf16은 반올림 경로가 달라 다른(이 위치들에서는 더 그럴듯한) 답을 낸다. firepit 결함이 아니다. 각 layer는 HF와 bf16 수준으로 일치하고, 차이는 긴 문맥에서 누적과 라우팅 flip으로 커진다.

남은 질문: 이 모델은 bf16 activation에서 품질이 더 좋은가(학습 수치 환경과 같은 정밀도). 정답 토큰 NLL(perplexity)로 firepit f32와 HF bf16을 비교해야 판단할 수 있다.
