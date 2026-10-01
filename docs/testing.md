# 커널 테스트

`crates/firepit-kernels/tests/common`에 f64 CPU oracle, 입력 전략, 원소별 상대오차 검사를 둔다. GPU PBT는 각 커널 8회이며 타일 특수화는 고정 집합에서만 고른다. 실패 시 축소된 반례는 테스트 파일 옆의 `.proptest-regressions`에 저장한다.

GEMV는 oracle 비교와 선형성·배치 독립·가중치 행 치환을 검사한다. RMSNorm은 oracle 비교와 배치 독립·scale 성질을 검사한다. 원소별 활성화 함수는 입력 범위 sweep와 극단값을 추가한다. 모델 규약은 ignored HF reference 테스트로, 테스트 민감도는 의도적으로 틀린 변형으로 확인한다.

기본 검증: `scripts/dev.sh test --workspace`. 실제 모델 정답 비교: `scripts/dev.sh test -p firepit --test reference -- --ignored`.
