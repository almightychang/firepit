# LLM 추론 엔진·양자화의 출력 품질 동일성 지표 조사

2026-10-04. 원문은 모두 직접 열어 확인했다. 확인하지 못한 것은 **미확인**, 원문이 아니라 내 계산·판단인 것은 **(계산)**, **(제안)**으로 표시했다.

## 0. 결론

firepit의 핵심 지표는 세 개를 권한다.

1. **토큰별 KL(P_HF ‖ Q_firepit)**: 평균, 99%·99.9% 분위수, 최댓값. 평균은 전체 분포 차이를, 꼬리는 드물게 크게 갈리는 토큰(라우팅 flip이 만드는 것)을 잡는다.
2. **top-1 일치율(llama.cpp의 "Same top p")**: greedy decode가 언제 갈라지는지와 직접 연결된다. 지금 쓰는 top-5 집합 일치를 많은 토큰으로 넓힌 것이다.
3. **ln(PPL_Q / PPL_base)와 Δp 분포의 비대칭**: 차이가 무작위 잡음인지, 한쪽으로 나빠지는 열화인지 구분한다. 단독 지표로는 쓰지 않는다.

판정은 임의 허용치 대신 **잡음 바닥(noise floor)과 짝지은(paired) 가설검정**으로 한다. 수치적으로 같아야 할 최적화는 "변경이 만든 차이가 정당한 두 구현 사이 차이보다 작은가"로 판정하고, 손실을 감수하는 양자화는 pass/fail 대신 CI가 붙은 수치와 과제 단위 flip의 McNemar 검정으로 결정한다(5절).

MoE 라우팅 일치율은 품질 지표가 아니라 **진단 지표**로 쓴다. 엔진 사이 라우팅이 대부분 토큰에서 어딘가 갈려도 출력 KL은 1e-3 수준이라는 원문 측정이 있다(3절).

## 1. 지표별 정리

| 지표 | 잡는 것 | 놓치는 것 |
|---|---|---|
| 평균 KL(base ‖ Q) | 어휘 전체 분포의 차이. 정답 토큰이 없어도 계산된다 | 평균은 꼬리를 희석한다. 드물게 크게 갈리는 토큰은 분위수·최댓값을 따로 봐야 한다 |
| KL 분위수·최댓값 | 국소적 대형 차이(라우팅 flip, 특정 위치의 오류) | 분포가 길어서 최댓값은 표본 수에 따라 커진다. 비교는 같은 데이터에서만 |
| PPL, ΔPPL, ln PPL 비 | 정답 토큰 확률의 체계적 하락 | 오르는 토큰과 내리는 토큰이 상쇄된다(Dutta et al. 4.3절). 정답 외 토큰의 분포 변화는 보지 못한다 |
| Δp(정답 토큰 확률 변화) 분위수 | 분포가 대칭이면 잡음, 음수 쪽이 크면 열화(llama.cpp README) | 정답 토큰 하나만 본다 |
| RMS Δp | 정답 토큰 확률 잡음의 크기 | 방향 없음 |
| top-1 일치율 | greedy 출력이 바뀌는 빈도 | 1·2위가 근소한 토큰에서 흔들리는 것과 실제 열화를 구분하지 못한다 |
| top-k 집합 일치 (vLLM, firepit 3.7) | 상위 후보 구성의 변화 | 확률 값의 변화 |
| flips (Dutta et al.) | 과제 정답 → 오답, 오답 → 정답 전환. 정확도가 같아도 개별 답이 바뀌는 것 | 정답이 있는 과제가 필요하다. 오답 → 오답 변화는 의도적으로 제외 |
| 과제 정확도 (vLLM, SGLang, TRT-LLM CI) | 사용자에게 보이는 능력 | 개별 답의 변화가 상쇄된다(Dutta et al.의 핵심 주장) |
| 생성 텍스트 유사도 (ROUGE, BERTScore, STS) | 긴 생성에서 의미 보존 | 토큰 하나가 갈리면 이후 전체가 달라져 수치 해석이 어렵다 |
| JS divergence | 대칭 거리. 라우터 분포 비교에 쓰인 사례 있음 | 엔진 품질 비교에서 JS를 주 지표로 쓴 원문은 찾지 못했다(**미확인**) |

## 2. 실제 사용 사례 (원문 확인)

### 2.1 llama.cpp `llama-perplexity --kl-divergence`

원문: [tools/perplexity/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md), [perplexity.cpp](https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/perplexity.cpp)

- 사용법: 기준 모델(FP16)로 `--kl-divergence-base file.kld`를 돌려 logits를 저장하고, 양자화 모델로 `--kl-divergence-base file.kld --kl-divergence`를 돌린다.
- 출력 항목(코드에서 확인):
  - Perplexity: `Mean PPL(Q)`, `Mean PPL(base)`, `Cor(ln(PPL(Q)), ln(PPL(base)))`, `Mean ln(PPL(Q)/PPL(base))`, `Mean PPL(Q)/PPL(base)`, `Mean PPL(Q)-PPL(base)`. 비율과 차이의 불확실성은 두 모델 NLL의 공분산을 넣어 계산한다(짝지은 오차 전파).
  - KL: `Mean KLD ± 불확실성`, `Maximum`, `99.9%`, `99.0%`, `95.0%`, `90.0%`, `Median`, `10%`, `5%`, `1%`, `0.1%`, `Minimum`.
  - 토큰 확률: `Mean Δp ±`, Δp 최댓값·분위수·최솟값, `RMS Δp ±`, `Same top p ±`.
- 계산 세부(코드):
  - KL 방향은 `Σ p_base · (log p_base − log p_Q)`, 즉 **KL(base ‖ Q)**다. `log p_base ≤ −16`인 항은 건너뛴다.
  - 기본 `n_ctx = 512`. 각 chunk의 **뒤쪽 절반 토큰만** 채점한다(앞 절반은 문맥 확보용).
  - 불확실성은 토큰별 값이 Gaussian이라 가정한 표준오차다. Same top p는 이항분포의 Gaussian 근사 `sqrt(p(1−p)/(n−1))`. 토큰이 100개 미만이면 통계를 내지 않는다.
  - 기준 logits는 16비트 정수로 스케일해 저장한다. README는 "f16" 행이 이 downcast만으로 생긴 차이라고 밝힌다(LLaMA 3 8B, f16 KLD 0.000551).
- 데이터 관례: "contributors의 관례는 Wikitext-2 test set" (README). 기준 logits 파일은 LLaMA 2 11 GiB, LLaMA 3 37 GiB.
- README에 실린 값(LLaMA 3 8B, Wikitext-2, CUDA, `f364eb6f`): q8_0 KLD 0.001355 ± 0.000006, Same top p 97.674 ± 0.040%; q4_K_M KLD 0.031273, Same top p 91.901%. BF16 vs FP16(CPU, `83330d8c`)은 KLD 0.00002515, Maximum KLD 0.012206, Same top p 99.739%.
- README의 해석 지침: Δp 분위수가 대칭이면 양자화는 잡음을 더할 뿐이고, 음수 쪽이 크면 모델이 실제로 나빠진다. 또 llama.cpp 수치는 다른 프로젝트와 직접 비교할 수 없다고 명시한다.

### 2.2 "Accuracy is Not All You Need" (실제 존재 확인)

원문: [arXiv:2407.09141](https://arxiv.org/abs/2407.09141), Dutta, Krishnan, Kwatra, Ramjee (Microsoft Research), 2024-07-12. [HTML 본문](https://arxiv.org/html/2407.09141)

- flips: baseline과 압축 모델 사이 답이 정답 → 오답, 오답 → 정답으로 바뀐 비율. 오답 → 오답은 제외한다.
- KL: 객관식 데이터(MMLU 등)에서 각 보기의 각 토큰에 대한 어휘 전체 분포의 KL을 보기·토큰·표본으로 평균.
- 결과: 양자화 기법들의 정확도 차이는 0~2%인데 flips는 최대 13.6%. KL과 flips의 Spearman 상관은 MMLU에서 0.981. MT-Bench 자유 생성에서는 flips가 많은 압축 모델이 baseline보다 유의하게 나빴다.
- perplexity 비판(4.3절): 일부 토큰의 확률 하락이 다른 토큰의 상승으로 상쇄될 수 있다. 로그확률에 대칭 Gaussian 잡음을 더하면 perplexity는 유지되지만 출력은 나빠지는 예시를 든다. 권고: perplexity와 함께 KL을 보고하라.
- 모델: Llama2 chat 7B/13B/70B, Yi 6B/34B. 기법: BnB W8A8/W4A4, GPTQ W8A16/W4A16, AWQ W4A16, SmoothQuant W8A8, 그리고 layer dropping, WANDA, SliceGPT.

### 2.3 vLLM

- 모델 정합 테스트 [`tests/models/utils.py` `check_logprobs_close`](https://github.com/vllm-project/vllm/blob/main/tests/models/utils.py): 두 구현의 greedy 생성을 토큰마다 비교한다. 토큰이 같으면 넘어가고, **처음 갈린 위치에서 각자의 토큰이 상대의 top-N logprobs 안에 있으면 통과**한 뒤 거기서 비교를 멈춘다("sequences will now diverge"). prompt logprobs가 있으면 모든 prompt 위치에서 top-N 토큰 **집합**이 같아야 한다. 확률 값 허용치는 없다.
- 정확도 회귀 CI [`.buildkite/lm-eval-harness/test_lm_eval_correctness.py`](https://github.com/vllm-project/vllm/blob/main/.buildkite/lm-eval-harness/test_lm_eval_correctness.py): lm-eval-harness로 과제 점수를 내고 `measured ≥ ground_truth × (1 − rtol)`이면 통과. `DEFAULT_RTOL = 0.08`(한쪽 방향). 설정 예: Qwen1.5-MoE W4A16, GSM8K 5-shot, limit 1319, strict-match 0.30, flexible-extract 0.465. ground truth는 config에 박힌 값이다.

### 2.4 SGLang

- [`test/registered/e2e/models/test_generation_models.py`](https://github.com/sgl-project/sglang/blob/main/test/registered/e2e/models/test_generation_models.py), [`python/sglang/test/runners.py` `check_close_model_outputs`](https://github.com/sgl-project/sglang/blob/main/python/sglang/test/runners.py): HF와 SGLang에서 같은 prompt로 32토큰을 생성해 비교한다.
  - prefill: 각 위치의 top-5 logprob **값**(정렬된 값, 토큰 ID는 비교하지 않음)의 절대 차이가 `prefill_tolerance` 미만. 기본 5e-2.
  - decode: 같은 방식, 기본 6e-2. 모델별로 7e-2, 8e-2로 올린 사례가 있고 코드 주석에 "numerical error in issue #8614"로 올렸다고 적혀 있다.
  - 입력 길이가 100토큰 이하일 때만 logprob 검사를 한다.
  - 출력 문자열은 ROUGE-L ≥ `rouge_l_tolerance`(기본 1, 즉 완전 일치).
- 과제 정확도: [`eval_accuracy_kit.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/test/kits/eval_accuracy_kit.py)가 `score ≥ score_threshold`로 판정. MoE runner 수동 테스트는 MMLU 5문항, 0.48 이상.

### 2.5 TensorRT-LLM

원문: [`tests/integration/defs/accuracy/README.md`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tests/integration/defs/accuracy/README.md), [`accuracy_core.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tests/integration/defs/accuracy/accuracy_core.py)

- 임계값을 **가설검정**으로 정한다. README: 임계값이 기준에 너무 가까우면 정상 변동에 간헐 실패하고, 너무 멀면 회귀를 놓친다.
- 단측 검정. H0: 새 결과의 평균 ≥ 기준. 통계량 `t = (x̄' − x̄)/sqrt(2σ²/n)`.
  - 임계값 `γ = x̄ + Φ⁻¹(α)·sqrt(2σ²/n)`
  - 최소 검출 효과 `θ = −(Φ⁻¹(α) + Φ⁻¹(β))·sqrt(2σ²/n)`
- 기본값 α = 0.05, β = 0.2. 정답/오답 과제는 Bernoulli라 p = 0.5를 써서 σ = 50(0~100 척도). MMLU n = 4096 → θ ≈ 2.75점, 임계값은 기준 −1.82점(README 예시와 일치). GSM8K n = 1319(전체). CNN/DailyMail은 α = 0.002, σ = 11.06, n = 512.
- 데이터 타입·양자화는 정확도 차이를 정당화하므로 기준값을 따로 둔다. 병렬화 등은 정확도에 영향이 없어야 하므로 기준을 공유한다: "tensor parallelism을 켰을 때 통계적으로 유의한 손실이 있으면 정확도 버그를 의심해야 한다."
- 주의 (계산): 이 설계는 **두 독립 표본**을 가정한다(`2σ²`). 같은 문항을 두 번 푸는 상황이므로 짝지은 검정(문항별 차이, McNemar)을 쓰면 같은 n에서 θ가 훨씬 작아진다.

### 2.6 MLC-LLM

[`python/mlc_llm/bench/evaluation/mmlu.py`](https://github.com/mlc-ai/mlc-llm/blob/main/python/mlc_llm/bench/evaluation/mmlu.py)에 MLCEngine으로 MMLU를 평가하는 스크립트가 있다. CI 회귀 판정에 쓰는지, perplexity·KL 도구가 있는지는 **미확인**(저장소 코드 검색에서 "perplexity" 결과 없음).

### 2.7 양자화 논문

- GPTQ ([arXiv:2210.17323](https://arxiv.org/abs/2210.17323)): WikiText2, PTB, C4 perplexity와 lm-eval-harness zero-shot. "perplexity 과제는 양자화에 특히 민감하다"는 이유로 perplexity에 집중. 평가 방식(부록): validation set 전체를 두 줄바꿈으로 이어 tokenize하고, **2048 폭 겹치지 않는 segment**로 잘라 다음 토큰 로그확률의 평균을 지수화. calibration은 C4 학습 세트의 2048토큰 segment 128개라서 C4 perplexity는 완전한 zero-shot이 아니라고 밝힌다.
- "Give Me BF16 or Give Me Death" ([arXiv:2411.02355](https://arxiv.org/abs/2411.02355), Kurtic et al., Neural Magic/ISTA): Llama-3.1, FP8/INT8/INT4. Open LLM Leaderboard v1·v2, Arena-Hard, HumanEval(+), RULER(긴 문맥), 추론 모델은 AIME·MATH-500·GPQA. 지표는 baseline 대비 **recovery %**(예: W4A16-INT 99.36%)와 생성 텍스트 유사도(ROUGE-1, ROUGE-L, BERTScore, STS; 70B/405B 평균 ROUGE-L 0.56, STS 0.96). Arena-Hard에는 95% 신뢰구간을 붙였다.
- VSRAQ ([arXiv:2606.05688](https://arxiv.org/abs/2606.05688)): WikiText-2 perplexity와 라우팅 일치(3절).

## 3. MoE 특유의 고려

### 3.1 라우팅 일치율을 지표로 쓴 사례

- **VSRAQ** ([arXiv:2606.05688](https://arxiv.org/html/2606.05688v1), Park et al., Nota, 2026-06): FP 모델과 양자화 모델의 토큰별 top-k expert 집합의 **Jaccard 유사도**를 layer마다 측정한다. Solar-Open-100B, Nemotron-3-Nano-30B-A3B, W4A16·NVFP4. "작은 양자화 섭동이 top-k 선택을 바꿀 수 있다"고 쓰지만 flip률이나 섭동 크기의 임계는 정량화하지 않았다.
- **EAQuant** ([arXiv:2506.13329](https://arxiv.org/abs/2506.13329)): 라우터 logit MSE와 top 일부 expert에 한정한 KL("KL-Top")을 calibration 목적함수로 쓴다. 라우팅 일치율 자체를 결과로 보고하지는 않고 과제 정확도를 보고한다. OLMoE-7B, DeepSeek-MoE-16B, Mixtral-8x7B.
- **R3, Rollout Routing Replay** ([arXiv:2510.11370](https://arxiv.org/html/2510.11370), Ma et al.): **엔진 사이** 라우팅 불일치를 측정한 가장 가까운 사례다. Qwen3-30B-A3B에서 SGLang(추론)과 Megatron(학습)을 비교했다.
  - 라우터 단위로 약 10%가 다른 expert를 고른다.
  - 토큰의 **94%가 적어도 한 layer에서** 다른 expert를 고른다.
  - 그런데 토큰 확률의 KL은 MoE 1.535e-3, dense(Qwen3-8B) 6.4e-4. MoE가 약 2.4배.
  - 같은 Megatron에서 같은 입력을 두 번 forward해도 KL 8.4e-4.

### 3.2 firepit에 주는 함의 (판단)

- firepit의 측정(엔진 검증 문서)과 R3가 같은 현상을 보인다. 1e-5 반올림 차이가 margin이 작은 토큰의 top-k를 뒤집고, 그 뒤 hidden 차이는 계단식으로 커진다.
- 그래도 출력 분포 차이는 작게 남는다(R3: 94% 토큰에 flip이 있어도 KL ~1e-3). 즉 **라우팅 일치율은 출력 품질과 일대일 관계가 아니다.** margin이 작은 flip은 두 expert의 gate 가중치가 거의 같다는 뜻이라 출력 영향도 작다.
- 그래서 라우팅 일치율(layer별 flip 토큰 비율, Jaccard)은 판정 기준이 아니라 **진단**에 쓴다. KL 꼬리 토큰이 나왔을 때 그 토큰의 첫 flip layer와 margin을 보면, 근소 margin flip(정상)인지 큰 margin에서의 flip(커널 버그 가능성)인지 구분할 수 있다. 이미 있는 `routing_flip_` 진단이 그 도구다.
- 양자화에서는 VSRAQ·EAQuant처럼 라우터 가중치 처리가 품질에 직접 영향을 줄 수 있다. 라우터(gate) 가중치는 양자화하지 않는 선택지를 비교 대상에 넣을 만하다(**제안**, firepit에서 측정 안 함).

## 4. 평가 데이터와 통계

### 4.1 데이터와 규모

- WikiText-2 test가 llama.cpp와 GPTQ 계열의 관례다(2.1, 2.7).
- **Qwen3.6 tokenizer로 센 WikiText-2 test(`wiki.test.raw`, llama.cpp `get-wikitext-2.sh` 출처)는 297,193 토큰**이다(실측, 1,288,556자).
  - llama.cpp 방식(n_ctx 512, 뒤 절반 채점)이면 580 chunk × 255 ≈ 14.8만 토큰을 채점한다 (계산).
- 문맥 길이: llama.cpp 기본 512, GPTQ 2048. 긴 문맥은 Kurtic et al.이 RULER로 따로 본다. 짧은 문맥만 보는 관례는 firepit에 부족하다(5.3).

### 4.2 신뢰구간

- 쓰이는 방식은 모두 정규 근사다. llama.cpp는 토큰별 값의 표준오차와 짝지은 오차 전파, TRT-LLM은 중심극한정리 기반 단측 검정이다.
- 주의 (판단): 같은 문서 안의 토큰은 서로 독립이 아니다(같은 문맥, 같은 상태 오차를 공유). 토큰을 i.i.d.로 보는 llama.cpp의 ± 값은 신뢰구간을 좁게 잡을 가능성이 높다. **chunk(문서) 단위 bootstrap**이나 chunk 평균에 대한 t 구간이 더 정직하다. 원문에서 이 방식을 쓴 엔진 사례는 찾지 못했다(**미확인**).
- 필요한 표본 수 (계산): 짝지은 단측 검정에서 최소 검출 효과 θ를 원하면 `n ≥ ((z_{1−α} + z_{1−β}) · σ_d / θ)²`. σ_d는 chunk 단위 차이의 표준편차로, 파일럿 실행에서 잰다. α = 0.05, β = 0.2이면 (z합)² ≈ 6.18.
- top-1 일치율의 표준오차는 `sqrt(p(1−p)/n)`. p = 0.99, n = 10만이면 3.1e-4 (계산). 토큰 상관 때문에 실제는 더 크다.

## 5. firepit 추천

### 5.1 지표

| 우선 | 지표 | 이유 |
|---|---|---|
| 1 | 토큰별 KL(P_HF ‖ Q): 평균 + 99/99.9% 분위수 + 최댓값 | 정답 토큰 없이 분포 전체를 본다. 평균은 전반 오차, 꼬리는 flip성 국소 오차를 따로 보여준다. llama.cpp와 같은 방향으로 정의하면 공개 수치와 감각을 맞추기 쉽다 |
| 2 | top-1 일치율 (+ 기존 top-5 집합) | greedy 생성이 바뀌는 빈도와 직결. 지금의 150토큰 1점 검사를 수만~수십만 토큰의 비율로 확장한 것 |
| 3 | ln(PPL_Q/PPL_HF) (짝지은 CI)와 Δp 분포 비대칭 | KL은 방향이 없다. 차이가 잡음(대칭)인지 열화(음수 쪽 꼬리)인지 이것으로 가른다 |
| 진단 | layer별 라우팅 flip 토큰 비율, 꼬리 토큰의 첫 flip layer·margin | KL 꼬리의 원인 분석용. 판정에 쓰지 않는다(3.2) |
| 양자화 결정 시 | 과제 flips (정답 → 오답 / 오답 → 정답) | Dutta et al.이 KL과 강한 상관(0.981)을 보였고 사용자 체감에 가깝다. prefill만 쓰는 객관식 loglikelihood 과제면 비용이 낮다 |

### 5.2 기준 logits 저장 (제안)

- 어휘가 248,320이라 토큰당 f32 logits는 약 0.99 MB다. 채점 토큰 15만이면 f32 약 149 GB, 16비트 약 74 GB (계산).
- 16비트로 저장하면 안 된다(판단). llama.cpp README가 보이듯 저장 downcast 자체가 KL 바닥(LLaMA 3 8B에서 5.5e-4)을 만든다. 이 값은 llama.cpp q8_0의 KLD 1.36e-3과 같은 자릿수라 작은 차이를 가린다.
- 대안: 기준 쪽 top-K(예: 수백) 로그확률을 f32로, 나머지 질량은 합 하나로 저장한다. KL(base ‖ Q)는 base 확률이 큰 항이 지배하고 llama.cpp도 `log p_base ≤ −16` 항을 버린다. firepit 쪽 logits는 실행 중에 전체를 쓸 수 있으므로 KL이 정확히 계산된다. K는 잘린 질량이 KL 오차 상한이 되도록 정한다.
- HF는 bf16 lm_head 출력이라 기준 logits 자체가 bf16 반올림을 포함한다. 그래서 다음의 잡음 바닥 측정이 필요하다.

### 5.3 데이터와 규모 (제안)

- **기본**: WikiText-2 test 전체(Qwen 토큰 29.7만). 단 llama.cpp의 512가 아니라 **4096 토큰 chunk, 뒤 절반 채점**(72 chunk, 약 14.7만 채점 토큰) (계산). Gated DeltaNet recurrent state의 오차 누적을 보려면 512는 짧다.
- **긴 문맥**: 32K 이상 긴 문서 몇 개를 넣고 KL을 위치 구간(0–4K, 4–16K, 16–32K …)별로 보고한다. 위치에 따라 KL이 커지면 state 누적 오차다. 데이터셋은 아직 정하지 않았다(Kurtic et al.의 RULER는 과제 정확도라 KL 측정용은 아니다).
- **분포 대표성**: Qwen3.6이 chat 모델이라면 WikiText만으로는 실제 사용 분포와 다르다. HF가 chat template으로 생성한 응답을 teacher-forcing으로 넣어 같은 지표를 재는 세트를 추가하면 decode 상황에 가깝다.
- **비용**: E3 기록상 GB10 prefill은 5000토큰에 4.97초다. 29.7만 토큰이면 firepit 쪽 약 5분이다 (계산, 실행 안 함). HF 쪽 시간은 재지 않았다.
- **HF f32 기준**: 35B 파라미터 × 4바이트 ≈ 140 GB라 GB10 메모리(128 GB)에 올라가지 않는다 (계산). bf16이 실용 기준이고, f32는 offload나 layer 단위 실행이 필요하다.

### 5.4 판정 방식 (제안)

임의 허용치 대신 **잡음 바닥**을 기준으로 삼는다. 잡음 바닥은 "정당한 두 구현이 이만큼은 다르다"는 값이고 측정해서 얻는다.

1. **바닥 측정**: KL(HF bf16 ‖ firepit 기준 빌드). firepit은 f32라 차이 대부분은 HF bf16 반올림과 라우팅 flip이다. 참고로 공개 원문의 같은 성격 수치는 llama.cpp BF16 vs FP16(LLaMA 3 8B) 2.5e-5, R3의 같은 엔진 두 번 forward(Qwen3-30B-A3B) 8.4e-4다. 모델과 조건이 달라 그대로 쓰지 않는다.
2. **수치적으로 같아야 하는 변경**(fusion, 타일 변경, 커널 교체): 변경 전후 firepit 두 빌드를 같은 데이터로 돌리고 chunk별 KL(HF ‖ ·)의 **짝지은 차이**를 단측 검정한다(α = 0.05, β = 0.2, TRT-LLM과 같은 틀). 최소 검출 효과 θ는 1단계의 바닥 값에서 정한다: "이 변경이 HF와의 거리를 바닥만큼 늘렸다면 잡아낸다". 추가로 KL(기준 빌드 ‖ 새 빌드)이 바닥보다 작아야 한다. 꼬리 토큰이 새로 생기면 첫 flip layer의 margin을 확인한다. 근소 margin이면 정상이고, 큰 margin이면 버그다.
3. **양자화**(손실을 감수하는 변경): pass/fail이 아니라 결정이다.
   - 형식별로 KL 평균·꼬리, top-1 일치율, ln PPL 비, Δp 비대칭을 CI와 함께 표로 낸다. 바닥 대비 배수로 읽는다.
   - 최종 후보는 객관식 과제(예: MMLU 전체)에서 flips를 세고 **McNemar 검정**(정답 → 오답 수와 오답 → 정답 수의 비대칭)으로 순 열화가 통계적으로 검출되는지 본다. TRT-LLM의 독립 표본 검정보다 같은 문항 수에서 검출력이 높다.
   - 수용 기준(검출해야 할 최소 정확도 손실 θ)은 사람이 정하는 제품 결정이다. 다만 θ와 n, α, β를 함께 적어 "무엇을 놓칠 수 있는지"를 명시한다.
4. 기존 3.7의 150토큰 top-1/top-5 검사는 빠른 smoke test로 유지한다.

## 6. 미확인 목록

- JS divergence를 엔진·양자화 품질의 주 지표로 쓴 원문 사례.
- MLC-LLM이 CI에서 정확도 회귀를 판정하는지와 그 방식.
- 토큰 상관을 반영한(chunk bootstrap 등) 신뢰구간을 쓰는 엔진 사례.
- R3 논문의 측정 정밀도(bf16 여부)는 원문 요약에서 명시되지 않았다.
- Kurtic et al.의 생성 유사도 수치는 HTML 요약으로 확인했다. 표 단위 원문 대조는 하지 않았다.

## 출처

- llama.cpp perplexity README: https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md
- llama.cpp perplexity.cpp: https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/perplexity.cpp
- llama.cpp get-wikitext-2.sh: https://github.com/ggml-org/llama.cpp/blob/master/scripts/get-wikitext-2.sh
- Dutta et al., Accuracy is Not All You Need: https://arxiv.org/abs/2407.09141 , https://arxiv.org/html/2407.09141
- vLLM check_logprobs_close: https://github.com/vllm-project/vllm/blob/main/tests/models/utils.py
- vLLM lm-eval CI: https://github.com/vllm-project/vllm/blob/main/.buildkite/lm-eval-harness/test_lm_eval_correctness.py
- SGLang generation test: https://github.com/sgl-project/sglang/blob/main/test/registered/e2e/models/test_generation_models.py
- SGLang runners: https://github.com/sgl-project/sglang/blob/main/python/sglang/test/runners.py
- SGLang eval kit: https://github.com/sgl-project/sglang/blob/main/python/sglang/test/kits/eval_accuracy_kit.py
- TensorRT-LLM accuracy README: https://github.com/NVIDIA/TensorRT-LLM/blob/main/tests/integration/defs/accuracy/README.md
- TensorRT-LLM accuracy_core.py: https://github.com/NVIDIA/TensorRT-LLM/blob/main/tests/integration/defs/accuracy/accuracy_core.py
- MLC-LLM MMLU eval: https://github.com/mlc-ai/mlc-llm/blob/main/python/mlc_llm/bench/evaluation/mmlu.py
- GPTQ: https://arxiv.org/abs/2210.17323
- Kurtic et al., Give Me BF16 or Give Me Death: https://arxiv.org/abs/2411.02355
- VSRAQ: https://arxiv.org/abs/2606.05688
- EAQuant: https://arxiv.org/abs/2506.13329
- R3: https://arxiv.org/abs/2510.11370
