# Qwen3.6 decode 프로파일 (GB10, E4 첫 측정)

2026-10-03, firepit `f65aa67`, Spark(GB10) pod, 전체 40 layer, bf16 가중치, CUDA Graph, `examples/bench`.

## 결론

- **GDN conv 커널 하나가 step 시간의 59%다** (82 ms 중 48.7 ms). 채널당 block 하나(grid 8192)로 layer당 1.65 ms를 쓴다. 읽는 데이터는 layer당 0.26 MB뿐이다. 2.7의 "채널 타일 1"이 원인이고, roadmap E7의 GDN 채널 타일 항목이다.
- GEMV는 대역폭 하한 근처다: GDN projection 177 GB/s, MoE expert 208 GB/s, lm_head 191 GB/s (대역폭 floor 약 200 GB/s).
- GPU는 step 내내 바쁘다(커널 사이 빈 시간 0%). launch overhead는 문제가 아니다.
- batch를 늘려도 이득이 거의 없는 이유도 conv다. conv는 세션 op이라 세션마다 49 ms가 더해진다.

## batch

| batch | step median | 합계 tok/s | 세션당 tok/s |
|---:|---:|---:|---:|
| 1 | 83.5 ms | 12.0 | 12.0 |
| 2 | 144.2 ms | 13.9 | 6.9 |
| 3 | 205.9 ms | 14.6 | 4.9 |
| 4 | 267.6 ms | 14.9 | 3.7 |

## batch 1 step 분해

step당 커널 943개 (계획과 일치: embedding 1 + GDN layer 23 op × 30 + attention layer 25 op × 10 + 2). 커널 이름이 모두 `full_entry`라서 실행 순서를 plan op 순서에 맞춰 분류했고, grid 크기로 대응을 확인했다.

| 분류 | ms | % | 읽는 GB | GB/s |
|---|---:|---:|---:|---:|
| GDN state (conv 48.7 + delta 0.8) | 49.5 | 60.2 | 0.13 | 2.7 |
| GDN projection | 11.4 | 13.9 | 2.02 | 177 |
| MoE routed expert | 9.7 | 11.8 | 2.01 | 208 |
| lm_head | 5.3 | 6.5 | 1.02 | 191 |
| Attention projection | 3.4 | 4.2 | 0.55 | 160 |
| MoE router + shared | 2.0 | 2.5 | 0.29 | 145 |
| 나머지 (norm, 원소별, attention, top-k 등) | 0.9 | 1.1 | – | – |
| 합계 | 82.3 | | 5.9 | |

토큰당 가중치 5.9 GB ÷ 200 GB/s ≈ 30 ms가 하한이다. conv를 고치면 약 34 ms(약 29 tok/s)로 하한 근처에 온다는 추정이다 (conv 외 나머지 33.6 ms + 고친 conv).

## 측정 방법과 주의

- nsys는 CUDA 13.0판(`cuda-nsight-systems-13-0`, 2025.3.2)을 써야 커널 기록이 남는다. 13.3판(2026.1.3)은 Spark 드라이버 580(CUDA 13.0)에서 커널 기록이 비어 있었다. `--cuda-graph-trace=node`로 graph 내부 커널을 본다.
- pod에 SYS_ADMIN을 주었지만, 원인은 권한이 아니라 nsys 판이었다.
- 바이트 수는 config 차원으로 계산했다 (가중치만, 활성값 제외).

## conv 채널 타일화 후 (`c723f65`)

GB10에서 C=128(grid 64). 같은 조건으로 다시 측정했다.

| batch | step median | 합계 tok/s | 세션당 tok/s | 이전 합계 tok/s |
|---:|---:|---:|---:|---:|
| 1 | 32.7 ms | 30.6 | 30.6 | 12.0 |
| 2 | 42.9 ms | 46.6 | 23.3 | 13.9 |
| 3 | 53.6 ms | 55.9 | 18.6 | 14.6 |
| 4 | 64.0 ms | 62.5 | 15.6 | 14.9 |

- batch 1은 2.6배 빨라져 대역폭 하한(약 30 ms)의 92% 수준이다.
- 세션 하나가 늘 때마다 약 10.5 ms가 더해진다. 토큰마다 다른 expert 8개를 읽는 비용(토큰당 2.0 GB ÷ 200 GB/s ≈ 10 ms)과 맞는다. MoE의 본질적인 비용이다.
- 생성 결과는 이전과 토큰 단위로 같다.
