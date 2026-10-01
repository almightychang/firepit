# 커널 테스트

`crates/firepit-kernels/tests/common`에 f64 CPU oracle, 입력 전략, 원소별 상대오차 검사를 둔다. GPU PBT는 GEMV/RMSNorm 각 8회, SwiGLU 12회이며 JIT 타일 특수화는 고정 집합에서만 고른다. GEMV의 런타임 N·K는 허용 타일 배수 범위에서 독립적으로 생성한다. 실패 시 축소된 반례는 테스트 파일 옆의 `.proptest-regressions`에 저장한다.

GEMV는 oracle 비교와 선형성·배치 독립·가중치 행 치환을 검사한다. RMSNorm은 oracle 비교와 배치 독립·scale 성질을 검사한다. SwiGLU는 f64 oracle PBT와 SiLU 구간·극단값·NaN sweep로 확인한다. 모델 규약은 ignored HF reference 테스트로, 테스트 민감도는 gate/up 순서를 뒤집은 대조 실험으로 확인한다.

기본 검증: `scripts/dev.sh test --workspace`. 실제 모델 정답 비교: `scripts/dev.sh test -p firepit --test reference -- --ignored`.
