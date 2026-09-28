# firepit

A warm fire for the token-poor.

개인 박스 한 대(DGX Spark)를 여러 에이전트가 나눠 쓰는 로컬 LLM 추론 엔진이다. 세션끼리 서로의 상태를 건드릴 수 없다는 것을 Rust 소유권과 safe cuTile로 보장하는 것이 목표다.

- [목표](docs/goals.md)
- [설계](docs/design.md)
- [조사 자료](docs/research/README.md)

상태: 설계 단계 (2026-09-29).
