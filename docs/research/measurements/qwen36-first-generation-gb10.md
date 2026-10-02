# Qwen3.6-35B-A3B 첫 생성 (GB10)

2026-10-02, firepit `2f5cf3f`, DGX Spark(GB10) k8s pod, GPU 1개.

## 조건

- 모델: `Qwen/Qwen3.6-35B-A3B` revision `995ad96`, 전체 40 layer, bf16 가중치, f32 활성값·state
- 실행: `examples/generate --prompt "The capital of France is" --max-new-tokens 32 --max-seq 256`, CUDA Graph, batch 1, greedy
- prompt 처리는 decode 경로로 토큰마다 한 step (prefill은 E3)
- 구현은 safe and simple 단계: fusion, 양자화, split-KV, 타일 조정 없음

## 결과

```
The capital of France is Paris, a city renowned for its iconic landmarks such as the Eiffel Tower,
the Louvre Museum, and the Notre-Dame Cathedral. Paris is also famous
```

- 토큰: `[11751, 11, 264, 3177, 34756, 364, 1141, 25438, 57902, 1680, 430, 279, 242476, 300, 21262, 11, 279, 9440, 46890, 15849, 11, 321, 279, 42027, 9141, 368, 54832, 13, 11751, 369, 1048, 10919]`
- step 시간: 첫 step 4.36 s (JIT 컴파일 포함), 이후 prompt·decode step 83–87 ms (약 11.8 tok/s)
- 전체 101.7 s (모델 로드 포함, 로드가 대부분)

## 해석과 한계

- 끝 출력의 정확성 근거는 5080의 prefix 4 layer logits HF 비교(상대 L2 최대 5.7e-7, top-1 5/5)와 op 단위 비교다. 전체 40 layer logits를 HF와 직접 비교하지는 않았다 (HF 전체 모델을 f32로 돌릴 메모리가 없다).
- 85 ms/token은 최적화 전 기준선이다. 대역폭 대비 효율은 측정하지 않았다. 성능 측정과 ds4 비교는 E4에서 한다.
