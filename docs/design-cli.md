# firepit CLI 설계

상태: Sam 승인 2026-10-05. 양자화 결정: [design-e4](design-e4.md) "결정: NVIDIA 방향".

## 원칙

지원하는 모델×모드 조합은 측정하고 검증한 몇 개로 정해져 있다. 사용자는 모델과 모드만 고르고, 받기·조립·검증·실행은 firepit이 한다(Ollama의 `model:tag`와 같은 형태). 정밀도 구성은 산출물에 담고 실행은 저장된 디렉터리 하나를 읽는다(vLLM·llama.cpp·TensorRT-LLM·ModelOpt 컨벤션).

## 명령 (이번 범위: models, pull, run)

```
firepit models                     지원 모델과 모드, 크기, 측정 수치
firepit pull <MODEL>[:<MODE>]      받기 + 조립 + 검증 (모드 생략 시 모델의 기본 모드)
firepit run  <MODEL>[:<MODE>] [PROMPT]   없으면 pull 후 생성 (examples/generate 기능)
firepit serve <MODEL>[:<MODE>]     E5
```

clap derive. 인자는 파싱 단계에서 타입(`ModelRef { model: ModelId, mode: Mode }`)으로 바꾼다. 알 수 없는 모델·모드는 파싱 에러이고 지원 목록을 보여 준다.

## 카탈로그 (코드 안의 레시피)

| 모델:모드 | 구성 | 출처 (커밋 고정) |
|---|---|---|
| `qwen3.6:nvfp4` (기본) | NVIDIA NVFP4 혼합 + lm_head bf16 | `nvidia/Qwen3.6-35B-A3B-NVFP4@1355db6a` 전체 + `Qwen/Qwen3.6-35B-A3B@995ad96e`의 `lm_head.weight`만 |
| `qwen3.6:fp8` | Qwen 공식 FP8 | `Qwen/Qwen3.6-35B-A3B-FP8@95a723d0` |
| `qwen3.6:bf16` | 원본 | `Qwen/Qwen3.6-35B-A3B@995ad96e` |

레시피에는 정밀도 구성 설명, 예상 크기, 측정 수치(decode tok/s, PPL 변화, 측정 커밋)를 함께 둔다. 수치는 측정 문서와 같은 값이다.

## pull

1. 필요한 파일만 받는다. 텐서 하나만 필요한 경우(lm_head)는 safetensors 헤더를 범위 요청으로 읽어 그 텐서의 바이트 범위만 받는다. 받은 바이트는 헤더의 크기·dtype·shape와 대조한다.
2. 조립: 내부 derive. base 파일은 저장소 안에 두고(링크가 아니라 저장소 소유), 가져온 텐서는 `firepit-<group>.safetensors`로 쓰고, index와 `hf_quant_config.json`·`config.json`의 양자화 목록에서 그 그룹을 뺀다.
3. 검증: 기존 로더로 구성을 대조한다(GPU 불필요). 실패하면 저장소에 남기지 않는다.
4. manifest `firepit.json`: 모델·모드, 출처 repo와 커밋, 파일별 sha256, firepit 커밋.

저장소: `$FIREPIT_HOME/models/<model>/<mode>/`, 기본 `~/.cache/firepit`. 받는 중에는 임시 디렉터리에 쓰고 끝나면 원자적으로 이동한다. 이미 있으면 manifest를 대조하고 건너뛴다. 중단된 받기는 이어서 받는다.

## 범위 밖

serve(E5), 사용자 정의 조합(개발용 실험은 기존 `--override`), 인증이 필요한 repo.
