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

## JIT 캐시 (2026-10-07)

`firepit run` 첫 실행은 커널 JIT 때문에 첫 토큰까지 GB10에서 약 25 s가 걸렸다(`research/measurements/nvfp4-gb10.md`).

cuTile JIT는 세 단계다. stage1은 Rust 커널을 Tile IR bytecode로 만들고, stage2는 `tileiras`로 cubin을 만들고, stage3는 모듈을 로드한다. 프로세스 안의 메모리 캐시는 실행이 끝나면 사라진다. 디스크 캐시는 기본으로 꺼져 있다.

- **디스크 캐시를 켠다.** `run`은 `$FIREPIT_HOME/jit`에 cuTile 디스크 캐시를 설치한다.
  - 디스크 캐시는 stage2의 cubin을 저장한다. 키는 bytecode SHA-256, 대상 아키텍처, opt·flags, `tileiras` 지문이다. 소스·generics·툴체인이 바뀌면 자연히 miss다.
  - 드라이버가 캐시된 cubin을 거부하면 cuTile이 지우고 다시 컴파일한다.
  - 두 번째 실행부터 stage2가 빠지고 stage1·stage3는 프로세스마다 남는다.
- **캐시 디렉터리는 비공개여야 한다.** 캐시 hit는 저장된 바이트를 장치 코드로 실행한다. 그래서 디렉터리를 0700으로 만든다.
  - 이미 있는 디렉터리가 group·others에 열려 있으면 거부한다.
  - 0700 재설정은 소유자만 성공하므로 소유자 검사를 겸한다.
  - cuTile이 기본 위치(`~/.cache/cutile/kernels`)에 적용하는 정책과 같다. 위치를 `$FIREPIT_HOME` 아래에 둔 것은 두 가지 때문이다. 모델과 함께 지우고 옮길 수 있다. probe 예제들이 쓰는 기본 위치의 2 GiB 한도를 나눠 쓰지 않는다.
- **KV cache 길이를 2의 거듭제곱으로 올린다.** decode `cache_append`는 cache 길이를 컴파일 상수(`MAX_SEQ`, 출력 slab shape)로 받는다.
  - cuTile은 쓰기 tensor 인자에 정적 shape를 요구한다(cuTile book "Writable tensor parameters use static tile shapes"). 출력 slab을 `-1`로 두면 launch 검사가 거부한다(GB10에서 확인, 2026-10-07).
  - 한 행 view를 출력으로 넘길 수도 없다. graph 재생 중 위치를 장치에서 읽기 때문이다([design-e3](design-e3.md)).
  - 그래서 `run`은 `max_seq`(입력 길이 + 생성 길이)를 2의 거듭제곱으로 올린다. 하한은 cuTile이 특수화하는 shape 나눔의 최댓값(`DivHint::default().max`), 상한은 `max_position_embeddings`다.
  - 컴파일 변형은 길이의 로그 수만큼이다(16~4096 사이 9개). KV cache 메모리는 필요량의 최대 2배다. Qwen3.6은 토큰당 40 KB(attention 10층 × K·V × 2 head × 256 × 4 B)라, 4096 토큰 구간이면 최대 164 MB다.
- **사후 보고.** `run` 끝에 "컴파일한 커널 수, 그중 `tileiras`가 만든 수, 디스크 캐시에서 온 수"를 한 줄로 쓴다(cuTile 공개 카운터).
  - 진행 중 표시를 위한 hook은 없다.
  - JIT 누적 시간을 주는 공개 카운터도 없다. 단계별 시간은 `CUTILE_JIT_TIMING=1`로 본다.
- **pull 시점 미리 컴파일은 보류한다.**
  - cuTile의 compile-only 경로(`api::meta` + launcher의 `.compile()`)는 launcher마다 따로 생성된 메서드라, firepit의 `Sink::run` 경로로 부를 수 없다. runner와 같은 key(generics, view·slice의 특수화)를 재현하는 별도 코드가 필요하다.
  - 아니면 pull에서 모델을 올려 실제로 한 번 돌려야 한다. 그러면 GPU가 필요 없는 pull이 GPU를 요구하게 된다.
  - 디스크 캐시로 두 번째 실행부터 같은 효과를 얻는다.

## 범위 밖

serve(E5), 사용자 정의 조합(개발용 실험은 기존 `--override`), 인증이 필요한 repo.
