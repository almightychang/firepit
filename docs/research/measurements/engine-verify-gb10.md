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

## 라우팅 flip 확인 (`f4122eb`, `engine_verify routing_flip_`)

가설: 상태 차이의 계단은 MoE top-k 선택이 근소 차이(margin = k번째 − k+1번째 router logit)에서 두 경로 사이에 갈리는 것(flip) 때문이다. 확인 조건은 첫 flip layer와 첫 계단 layer의 일치, 그 앞 flip 0.

| 비교 | 첫 flip | 두 경로 margin | router logits 최대 차이 | 그 layer 출력(다음 입력) 차이 |
|---|---|---|---:|---|
| prefill vs step | L5, token 23 (expert 113 ↔ 131) | 1.5e-5 / 1.9e-6 | 2.5e-5 | 1.4e-5 → 1.9e-3 (L5) |
| 이어 붙이기 | L22, token 128 (expert 118 ↔ 29) | 2.5e-5 / 4.3e-6 | 1.0e-4 | 4.6e-5 → 1.4e-3 (L22) |

- 두 비교 모두 첫 flip 앞 layer는 flip 0이고 차이가 1e-5 수준에서 완만하게 자란다. 첫 flip layer의 출력 차이가 30~140배 뛰고, 다음 layer 상태에서 계단이 보인다(prefill vs step L6 recurrent 2e-4, L7 K/V 2.5e-3; 이어 붙이기 L23 K/V 2.3e-3). 가설 확인.
- 이후 flip이 layer마다 늘어난다(prefill vs step L15 10개, L17~18 21~22개). 차이가 커질수록 margin이 큰 토큰도 갈린다.
- 결론: 두 경로의 f32 반올림 차이(1e-5)는 top-k의 불연속 때문에 국소적으로 O(1) 차이로 바뀐다. 첫 flip 뒤의 상태·hidden 수치 비교는 커널 정확도의 근거가 되지 않으며, 실제 모델 판정은 마지막 logits top-1/top-5로 한다(docs/testing.md 3.7).

## 정답지 3종과 engine_verify 전체 (`88d3d91`, GB10, 2026-10-07)

정답지는 GB10 HF pod(/work/hfvenv, torch 2.14.1+cu130, transformers 5.17.0, gcc 필요)에서 생성했다(파일당 약 7분). 절차는 docs/testing.md "GB10 엔진 정답지".

| 정답지 | 입력 | sha256 |
|---|---|---|
| `/work/refs/qwen36-engine-prefill.safetensors` | prompt fixture 150토큰 | `ed708538521606594a18d4bfa394e1ff553925ad4839153cfc211288a3ce0f83` |
| `/work/refs/qwen36-engine-tokens.safetensors` | frog fixture 2862토큰 | `0c5dfaa926d7147952475a309b9fbe9a047b913312cdda6f9f99f3f12062e27d` |
| `/work/refs/qwen36-engine-isolated.safetensors` | frog fixture, 층 0·3 격리 | `fe47ab72c271c0286c96777f909377cd563ae4f84e4d7cebab572d5894295b55` |

engine_verify `--ignored` 7개 모두 통과(1306 s): hf_prefill top-5 일치(logits rel_l2 7.05e-2), append_vs_whole·prefill_vs_step top-5 일치, routing flip 첫 위치 layer 22 tok 128 / layer 5 tok 23, 층 0·3 격리 rel_l2 2.6e-3~6.0e-3과 expert 일치, locate_divergence에서 expert가 처음 다른 층은 4이고 hidden 차이는 층 15부터 커지며 frog 위치 재현(firepit 57859 대 HF 17).
