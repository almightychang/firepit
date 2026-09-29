# upstream 이슈 초안

firepit을 만들다 부딪힌 upstream 공백을 올리기 전에 여기서 다듬는다. 올리기 전에 Sam 확인을 받는다. 올린 뒤에는 링크와 상태를 적는다.

| 파일 | 대상 | 요약 | 상태 |
|---|---|---|---|
| [cutile-rs/001-mutable-tensor-views.md](cutile-rs/001-mutable-tensor-views.md) | NVlabs/cutile-rs | host에서 tensor 일부를 가변으로 빌려 커널 출력으로 넘기는 API가 없다 | 초안 |

관련 기록(초안 아님):

- NVlabs/cutile-rs#184: `reduce` 결과의 generic shape 불일치. 이미 보고됨(P0, 2026-07-06 open, PR 없음). firepit Q8 GEMV에서 재현했다. 수정 PR을 낼지는 담당자에게 먼저 묻는다.
