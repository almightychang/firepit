# 3.7 실제 모델 검증 (GB10, `9bb977a`)

2026-10-04. 전체 Qwen3.6-35B-A3B, 150토큰 prompt. HF 정답지는 transformers 5.17.0 + torch 2.14.1+cu130, bf16 가중치·bf16 activation, GPU, `trace_qwen36.py --engine`. Rust는 `tests/engine_verify.rs` ignored 3개, release. 판정은 docs/testing.md 3.7 기준(top-1, top-5 집합). 원문 로그는 저장소 밖에 보관.

| 비교 | top-1 | top-5 집합 | logits rel L2 |
|---|---|---|---:|
| engine prefill vs HF | 760 = 760 | 같음 | 7.1e-2 |
| prefill(150) vs step 150번 | 760 = 760 | 같음 | 3.3e-2 |
| prefill(100)+prefill(50) vs prefill(150) | 760 = 760 | 같음 | 4.3e-2 |

- HF와의 layer별 마지막 행 hidden rel L2: L0 4.4e-3 → L18 1.0e-1 → L39 9.7e-2. HF가 bf16 activation이라 L0부터 bf16 반올림(u ≈ 3.9e-3) 수준 차이가 있다.
- firepit 두 경로(모두 f32) 사이 상태 rel L2는 앞 layer 1e-6~1e-5에서 시작해 특정 layer에서 계단식으로 뛴다: prefill vs step은 L6 recurrent 2e-4, L7 K/V 2e-3, L15 2e-2, L23 0.14~0.23. 이어 붙이기는 L23 K 2.3e-3, L33 conv 2.6e-2.
- 계단식 증가는 MoE top-k 선택이 근소 차이에서 갈리는 것(라우팅 flip)으로 추정한다. 확인하지 않았다.
