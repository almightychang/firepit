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
