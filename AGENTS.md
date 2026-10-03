# firepit 작업 규칙

코딩 에이전트가 이 저장소에서 일할 때 따르는 규칙이다. 프로젝트 소개는 [README](README.md), 진행 상황은 [로드맵](docs/roadmap.md), 테스트 체계는 [testing](docs/testing.md)에 있다.

## 설계

- 설계를 먼저 정하고 구현한다. 합의된 설계와 다르게 해야 할 이유가 생기면 구현 전에 멈추고 이유를 보고한다.
- 이미 있는 도구(crate, HF transformers 기능, cuTile API)를 먼저 찾는다. 직접 구현은 없을 때만 한다.
- 지금은 safe and simple 구현이 우선이다. 커널 fusion 같은 최적화는 로드맵의 성능 단계에서 한다.

## 하드코딩

- 편의로 박아 넣은 값은 버그다. 모델 config에서 와야 할 값(차원, head 수, eps, theta, 회전 비율 등)을 상수로 두지 않는다.
- 나중으로 미루는 것은 하드웨어 aware 최적화 값(타일 크기, occupancy 목표, split-K 등)뿐이다.
- 도구 제약(cuTile shape 안 const 산술 불가, tile 크기는 2의 거듭제곱 등) 때문에 박아야 할 것 같으면 먼저 우회 설계를 찾는다. 없으면 멈추고 보고한다.
- 포맷 규격(Q8_0 블록 32 등)과 테스트 fixture 값은 하드코딩이 아니다.

## 안전성

- `crates/*/src`에 `unsafe`를 쓰지 않는다. `crates/firepit/tests/no_unsafe.rs`가 검사한다.
- `vendor/cutile-rs`는 수정하지 않는다. cutile-rs 변경이 필요하면 멈추고 보고한다.

## 코드

- 긴 주석이나 docstring으로 설명해야 하는 코드는 잘못된 코드다. 코드 자체로 흐름이 읽혀야 한다.
- 주변 코드의 이름, 구조, 주석 밀도를 따른다.

## 테스트

- 커널은 docs/testing.md의 층(oracle PBT, metamorphic, sweep, HF 비교, 대조 실험)으로 검증한다.
- HF 비교는 op 단위로 한다. HF가 기록한 입력을 Rust에 그대로 넣고 출력만 비교한다. 정답지는 HF 모듈과 함수를 그대로 쓰고 수식을 재구현하지 않는다.
- 내적(GEMV) 오차는 Higham–Mary 상한(`tests/common/check.rs`)으로 판정한다.
- 테스트가 실패하면 자동으로 재시도하지 말고 원인을 분석해 보고한다.
- proptest가 저장한 반례(`*.proptest-regressions`)는 지우지 않는다. 실패를 재현하는 유일한 기록이다.

## 실행

- 빌드와 테스트는 컨테이너에서: `scripts/dev.sh test --workspace`. 실제 모델 비교: `scripts/dev.sh test -p firepit --test reference -- --ignored`.
- 포맷은 호스트에서 `cargo fmt --all` (컨테이너에서는 동작하지 않는다).
- 정답지 재생성: `cd tools/reference && uv run python trace_qwen36.py --model ~/Models/qwen3.6-35b-a3b --out ../../refs/qwen36-prefix4.safetensors`.
- 성능 측정은 반드시 release 빌드(`--release`)로 한다. debug 빌드는 host 오버헤드가 커서 커널 시간을 수십 배 부풀린다.
- 호스트 shell은 zsh noclobber다. 파일을 덮어쓸 때는 `>|`를 쓴다.

## 커밋

- author와 committer는 `almightychang <almightychang@icloud.com>`이다. 커밋 후 둘 다 확인한다.
- push하지 않는다. 검토 후 사람이 한다.

## 보고

- unsafe 현황은 diff가 아니라 저장소 전체(`crates/*/src`) 기준으로 적는다.
- 측정값은 실제 실행한 값만 적고, 실행하지 않은 것은 그렇다고 적는다.
- 설계와 달라진 점과 이유를 적는다.
