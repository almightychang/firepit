# 조사 자료

조사 기간: 2026-09-27 ~ 2026-09-29. 표기는 모든 문서에서 같다. **[확인]**은 원문, 코드, 실측으로 확인한 것이고, **[추정]**은 계산이나 간접 근거에 기반한 것이다.

## 주제별 정리

| 문서 | 내용 |
|---|---|
| [landscape.md](landscape.md) | DGX Spark 로컬 LLM 생태계, Spark 1대에서 쓸 만한 모델 |
| [ds4.md](ds4.md) | antirez/ds4(DwarfStar): 구조, Qwen3.8 지원, 기여 규칙, fork들 |
| [cutile.md](cutile.md) | cuTile-rs와 cuda-oxide, safe 모델, mistral.rs의 cuTile 사용 분석, GDN 커널 분석 |
| [grout.md](grout.md) | Grout 구조와 firepit 출발점으로서의 평가 |
| [candle.md](candle.md) | candle의 양자화 dispatch, cuTile 통합 현황, 메인테이너 구조 |
| [models.md](models.md) | Qwen3.6 / 3.8-27B / 3.8-Flash-Next 구조 비교 |
| [e6-prerequisites.md](e6-prerequisites.md) | Flash-Next 착수 전 확인: MTP 구조, GB10 메모리·n-gram SSD, 정답지 기준 (2026-10-07) |

## 측정 보고서 (DGX Spark 실측)

| 문서 | 요약 |
|---|---|
| [measurements/ds4-qwen38-decode-profile.md](measurements/ds4-qwen38-decode-profile.md) | ds4 Qwen3.8 Q2 decode 51.4 ms/token. GDN recurrence는 2%, 병목은 양자화 GEMV와 GPU idle |
| [measurements/cutile-smoke-spark.md](measurements/cutile-smoke-spark.md) | cuTile-rs v0.4.0이 호스트 580 드라이버에서 동작. 예제 24개 통과, Grout Qwen3-4B 재현 |
| [measurements/candle-q8-baseline.md](measurements/candle-q8-baseline.md) | candle `fast_mmvq` Q8_0 GEMV가 GB10 floor의 88~96%. ds4 profiling 수치 정정 포함 |

측정 원본(csv, nsys, ncu)은 Spark의 `/var/lib/sam-ds4`, `/var/lib/sam-cutile`, `/var/lib/sam-candle`에 있다.
