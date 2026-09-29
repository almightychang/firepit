# firepit

A warm fire for the token-poor.

개인 박스 한 대(DGX Spark)를 여러 에이전트가 나눠 쓰는 로컬 LLM 추론 엔진이다. 세션끼리 서로의 상태를 건드릴 수 없다는 것을 Rust 소유권과 safe cuTile로 보장하는 것이 목표다.

- [목표](docs/goals.md)
- [설계](docs/design.md)
- [조사 자료](docs/research/README.md)

상태: 설계 단계 (2026-09-29).

## 빌드 준비

firepit은 cutile-rs fork 브랜치(`feat/tensor-view-mut`, upstream 제안 중인 가변 tensor view)를 `vendor/cutile-rs`에서 `[patch.crates-io]`로 쓴다. `vendor/`는 git에 넣지 않는다.

```bash
git clone https://github.com/NVlabs/cutile-rs.git vendor/cutile-rs
git -C vendor/cutile-rs fetch https://github.com/almightychang/cutile-rs.git feat/tensor-view-mut
git -C vendor/cutile-rs checkout FETCH_HEAD
scripts/dev.sh test
```

브랜치를 push하기 전까지는 로컬 worktree(`git -C ~/Repos/cutile-rs worktree add vendor/cutile-rs feat/tensor-view-mut`)를 쓴다.
