# E6 착수 전 확인: Qwen3.8-Flash-Next (2026-10-07)

코드 수정·커밋·GPU 사용 없음. 다운로드는 config/index/README와 safetensors·GGUF header(HTTP range)뿐이다. ds4는 `0aaea5a`(firepit 프로파일과 같은 commit)를 scratchpad에 clone해 읽었다.

## 결론

1. **MTP head**: 두 모델 다 "full attention 1층 + MoE + 공유 embedding·lm_head"이고 GDN이 없다. Flash-Next MTP에는 HC와 QSA indexer가 들어가고 PLE는 없다. speculative 제어 흐름(초안 → T=2/3 검증 → 거부 시 recurrent state 복원, KV는 pos만 되감기)은 공유할 수 있다. 입력 결합(fc), layer 몸통(HC vs RMSNorm residual), 출력(HC mixer vs norm), 복원할 state 목록(Flash-Next는 PLE conv history와 n-gram token history 추가)은 모델마다 다르다. ds4는 CUDA에서 MTP를 구현한다. 거부되면 커널이 검증 중에 떠 둔 snapshot과 live buffer를 포인터 swap하고, KV는 되돌리지 않는다.
2. **GB10 메모리**: 공식 BF16(`Qwen/Qwen3.8-Flash-Next`), 공식 FP8(Qwen), NVIDIA ModelOpt NVFP4가 모두 있다. NVIDIA NVFP4에서 vision과 n-gram을 빼면 상주 75.0 GiB, projection을 FP8로 직접 바꾸면 72.5 GiB다. MemTotal 119.67 GiB 안에 들어가고, KV·활성값·OS를 빼고 약 30 GiB가 남는다. n-gram table(BF16 95.37 GiB, 공식 FP8 47.68 GiB)은 FP8로도 메모리에 다 못 올린다. SSD에서 토큰당 16행을 읽으면 되고 대역폭은 문제가 아니다. 문제는 지연이다.
3. **정답지**: 더 이상 ds4 하나에 의존하지 않는다. HF transformers(v5.16.0~), vLLM, SGLang, llama.cpp가 `qwen4_exp`를 지원한다. 다만 HF는 두 모델 모두 `mtp.*`를 무시한다. trunk의 정답은 3.6처럼 HF op 단위 비교로 하고(prefix-4 shard 32.9 GB + n-gram 행은 range 요청), MTP는 vLLM/SGLang 코드나 ds4 CPU reference(`--first-token-test`)로 확인한다. ds4 양자화 GGUF는 firepit과 가중치가 달라서 원소 단위 허용 오차를 둘 수 없다. 분포 지표(KL, top-1, ΔNLL)로만 비교해야 한다.

---

## 1. MTP head 구조

### config [확인: HF config.json, `Qwen/Qwen3.8-Flash-Next` rev `de4b8e4`, `Qwen/Qwen3.6-35B-A3B` rev `995ad96`]

| 필드 | Qwen3.6 | Flash-Next |
|---|---|---|
| `mtp_num_hidden_layers` | 1 | 1 |
| `mtp_use_dedicated_embeddings` | false | false |
| `mtp` (dict) | 없음 | `{hybrid: true, layer_types: ["full_attention"], num_hidden_layers: 1, rope_theta: 1e7, mtp_use_hidden_state_from_layer: null}` |

`mtp.hybrid`의 의미는 모른다. [미확인]

### 텐서 (HF safetensors header) [확인]

| 역할 | Qwen3.6 | Flash-Next |
|---|---|---|
| 입력 norm | `pre_fc_norm_embedding [2048]`, `pre_fc_norm_hidden [2048]` | `pre_fc_norm_embedding [2560]`, `pre_fc_norm_hidden [10240]` (4 stream 전체) |
| 입력 결합 | `fc [2048, 4096]` (concat 하나) | `fc_embedding [2560,2560]`, `fc_hidden [2560,2560]` (분리) |
| layer 앞 norm | `input_layernorm`, `post_attention_layernorm` | `attn_hyper_connection.*`, `mlp_hyper_connection.*` (hc_norm [10240], down [320,10240], up [10240,320], block_inject [4,10240]) |
| attention | q [8192,2048], k/v [512,2048], o [2048,4096], q/k_norm [256] | q [12288,2560], k/v [512,2560], o [2560,6144], q/k_norm [256] + `indexer.index_qk_proj [640,2560]`, `indexer.q/k_layernorm [128]` |
| MoE | experts gate_up [256,1024,2048], down [256,2048,512], router, shared, shared_gate | experts gate_up [512,1280,2560], down [512,2560,640], router, shared, shared_gate |
| 출력 | `mtp.norm [2048]` → 공유 lm_head | `mtp.hyper_connection_mixer.*` (hc_norm, down, up) → 공유 lm_head. `mtp.norm` 없음 |

- PLE/n-gram 텐서는 MTP에 없다. trunk의 `layers.1.ple.*`에만 있다. [확인: header]
- ds4 GGUF(`antirez/qwen3.8-flash-next-gguf` rev `d600fe1`)에서는 MTP가 `blk.48.*`(1.405 GiB)다. `nextn.eh_proj [5120→2560]` Q8_0은 fc_embedding과 fc_hidden을 이어 붙인 것이다. `nextn.hnorm [10240]`, `nextn.hc_head_*`가 있고, `qwen4exp.attention.compress_ratios`의 마지막(48번) 값이 4라서 MTP 층도 QSA다. MTP expert는 Q2 파일에서도 Q4_K gate/up와 MXFP4 down이다. [확인: GGUF header]

### forward 차이 [확인: 코드]

- Qwen3.6(vLLM `qwen3_5_mtp.py:147-188`): `fc(cat(norm_e(e), norm_h(h)))` → MTP decoder layer(gated full attention + MoE) → `mtp.norm` → lm_head. vLLM에서 `h`는 `Qwen3NextModel.forward`가 final norm 뒤에 돌려주는 값이다(`qwen3_next.py:730`). `Qwen3_5Model`은 이 클래스를 상속한다. [확인: 코드 / 미확인: drafter로 넘어가는 hidden이 정확히 이 반환값인지 전체 경로]
- Flash-Next(ds4 CPU reference `ds4.c:67638 qwen4_ref_mtp`, vLLM `models/qwen4_exp/nvidia/mtp.py`):
  - `h`는 trunk final mixer **이전**의 4-stream R(10240)이다. vLLM 주석에도 "main model truly emits the pre-final-mixer multi stream"이라고 되어 있다.
  - hnorm은 10240 전체에 RMS를 한 번 건다.
  - stream s마다 `R_s = fc_hidden·hn_s + fc_embedding·en`을 계산한다(ds4 CUDA `mtp_stage`/`mtp_combine`, `ds4_qwen4_cuda.cuh:1385-1407`).
  - 그다음 HC mix → QSA attention(MTP 전용 KV와 indexer K cache) → HC combine → HC mix → MoE → combine → head HC mixer → lm_head 순서다.

### 같은 코드로 처리할 수 있나 [추정: 위 코드 근거로 판단]

| 부분 | 공유 가능 | 다른 점 |
|---|---|---|
| 제어 흐름: 초안, T=2/3 검증, accept/reject | ○ | |
| KV 되감기(pos만 되돌림) | ○ | Flash-Next는 indexer K cache도 같은 방식 |
| recurrent state snapshot/복원 | 구조는 같음 | Flash-Next는 GDN state·conv 외에 PLE conv history와 n-gram token history(`ple_prev`), mrope delta가 더 있음 |
| MTP 입력 결합 | × | concat→fc 1개 vs stream별 fc_hidden + 공통 fc_embedding |
| MTP layer 몸통 | 모델의 full-attention layer를 그대로 쓰면 ○ | RMSNorm residual vs HC, gated attention vs QSA |
| 출력 | × | norm vs HC mixer |

정리하면 "MTP 전용 코드"는 입력 결합과 출력 head만 남는다. 몸통은 각 모델의 full-attention layer를 재사용하고, 복원은 "세션 recurrent state 전체"로 일반화하면 된다. Flash-Next 검증 경로에는 초안 토큰의 n-gram 행 읽기(SSD)가 끼어든다. ds4 `tests/test_qwen4_ngram_state.c`가 MTP 중 읽기 실패와 복구를 따로 검사한다. [확인: docs]

### ds4 CUDA MTP와 reject 복원 [확인: 코드, docs]

- CUDA에서 동작한다. `--mtp`는 docs상 Metal/CUDA 공통이고, MTP 커널 `mtp_stage`/`mtp_combine`이 `ds4_qwen4_cuda.cuh`에 있다. 상위 로직은 `#ifdef DS4_HAS_QWEN4_GPU`(Metal·CUDA 공통)다. 기본은 depth 2이고, 첫 초안이 계속 맞으면 3으로 올리는 적응형이다(`DS4_QWEN4_MTP_DEPTH`).
- 검증 cycle: `ds4.c:74072 ds4_session_qwen4_spec_cycle`. 판정은 `74163`(greedy 일치)에서 하고, exact sampling 판정은 그 아래에 있다.
- snapshot: 검증 forward 중에 recurrent 커널이 row 0(과 row 1) 직후 state를 별도 buffer에 쓴다.
  - GDN: `gdn_scan` `ds4_qwen4_cuda.cuh:1672-1703`, 융합 경로 `gdn_front` `2173-2187`
  - GDN conv: `conv` `1630-1648`
  - PLE conv: `ngram_conv` `1747-1765`
  - host 쪽 `ple_prev`·pos·mrope: `ds4.c:58683-58698`
- 복원: 거부되면 live buffer와 snapshot buffer의 포인터를 swap한다(`ds4.c:58945 qwen4_graph_state_swap`). 3-row에서 둘째만 거부되면 `59015 qwen4_graph_state_swap2`다. 36층 state를 GPU로 다시 복사하지 않는다. snapshot buffer는 처음 쓸 때 할당한다(`58970`).
- KV와 indexer cache는 복원하지 않는다. `g->pos = snap_pos`로 되감고, 뒤쪽 행은 다음 forward가 덮어쓴다. 근거 주석: "Causal attention cannot read future entries, so existing KV rows are safe to overwrite" (`ds4.c` FT batch check).
- exact sampling에서 tool 경계를 다시 샘플링할 때는 검증 전 snapshot(`snap0`)을 따로 둔다.

## 2. GB10 메모리

### 구성 요소별 파라미터 (HF BF16 header 합산) [확인]

| 구성 | Qwen3.6 | Flash-Next | 원본 dtype |
|---|---:|---:|---|
| routed experts (trunk) | 32.21 B | 120.80 B | BF16 |
| n-gram table | — | 51.20 B | BF16 (`layers.1.ple…shard_0..127`, 각 [2,500,012, 160]) |
| GDN projection (qkv, z, out) | 1.007 B | 2.076 B | BF16 |
| HC (trunk 층별 + final mixer) | — | 0.641 B | BF16 |
| embedding | 0.509 B | 0.636 B | BF16 |
| lm_head | 0.509 B | 0.636 B | BF16 |
| attention (q, k, v, o, norm) | 0.273 B | 0.598 B | BF16 |
| shared expert | 0.126 B | 0.236 B | BF16 |
| router | 0.021 B | 0.063 B | BF16 |
| PLE 나머지 (key_proj, conv, norm) | — | 0.033 B | BF16 (+I64 multiplier) |
| indexer | — | 0.020 B | BF16 |
| GDN in_proj_a/b 등 | 0.005 B | 0.010 B | BF16 |
| MTP 전체 | 0.845 B | 2.607 B (expert 2.517) | BF16 |
| vision | 0.447 B | 0.449 B | BF16 |
| **합계** | **35.95 B** | **180.00 B** | |

NVIDIA README의 "125B total + 51B n-gram + 4B MTP"와 대략 맞는다. MTP는 header 합산으로 2.6 B다. [확인]

### 체크포인트 [확인: HF API, README, header]

| repo | 크기 | 내용 |
|---|---:|---|
| `Qwen/Qwen3.8-Flash-Next` (`de4b8e4`) | 360.0 GB | 공식 BF16. 공개(gated 아님) |
| `Qwen/Qwen3.8-Flash-Next-FP8` (`236dfdf`) | 172.76 GiB | routed expert(trunk·MTP)만 FP8 128×128 block, n-gram FP8. **GDN·attention projection, shared, HC, lm_head는 BF16** |
| `nvidia/Qwen3.8-Flash-Next-NVFP4` (`fc694b5`) | 123.53 GiB | trunk routed expert NVFP4 W4A4(group 16, MSE scale). MTP expert와 n-gram은 Qwen FP8과 byte 동일. 나머지 BF16. vLLM 전용, B200/B300 대상 |
| `antirez/qwen3.8-flash-next-gguf` (`d600fe1`) | Q2 137.10 / Q4 165.11 GiB | 상주 41.72 / 69.73 GiB + BF16 n-gram 95.37 GiB |
| `ggml-org`, `unsloth` 등 GGUF | 다수 | llama.cpp용. MTP는 별도 GGUF |

**3.6과 다른 점**: 3.6의 NVIDIA NVFP4는 projection을 FP8로 줬지만 Flash-Next판은 projection이 BF16이다. firepit의 "projection FP8" 방식을 쓰려면 BF16에서 FP8로 직접 양자화해야 한다. 공식 FP8도 projection은 BF16이라 가져다 쓸 곳이 없다. [확인: header]

### firepit 3.6 방식의 상주 크기 [확인: header 합산 / 추정: projection FP8 크기는 scale 무시]

NVIDIA NVFP4 checkpoint 기준, vision 제외, n-gram은 SSD:

| 항목 | GiB |
|---|---:|
| routed experts NVFP4 (scale 포함) | 63.28 |
| MTP routed experts FP8 block | 2.34 |
| projection (GDN + attention, trunk + MTP) BF16 | 5.07 |
| HC BF16 | 1.23 |
| lm_head BF16 | 1.18 |
| embedding BF16 | 1.18 |
| shared expert BF16 | 0.45 |
| 나머지 (router, norm, PLE key_proj 등) | 0.26 |
| **합계 (그대로)** | **75.0** |
| projection을 FP8로 직접 양자화 | **72.5** |

ds4 Q4(69.73 GiB)와 비슷한 크기다.

### 128 GB 안에서의 여유

| 항목 | GiB | 근거 |
|---|---:|---|
| MemTotal | 119.67 | [확인: 프로파일 문서의 125,481,412 kB] |
| 가중치 | 72.5~75.0 | 위 표 |
| KV + indexer cache, f32 (trunk 12층 + MTP 1층, 토큰당 58.5 KiB) | 32K 1.83 / 128K 7.31 / 262K 14.62 | [확인: 계산] KV를 bf16으로 두면 32K 1.02 GiB. firepit은 2의 거듭제곱 구간으로 잡아 최대 2배 |
| 세션당 recurrent state (GDN 36층 + conv + PLE history) | 0.11 (MTP depth 3이면 snapshot 2벌로 +0.22) | [확인: 계산] |
| prefill 활성값 | 2~7 | [추정] ds4는 8192 chunk에서 buffers 7.37 GiB였음 [확인: 프로파일]. firepit은 측정 안 함 |
| CUDA context + JIT | 약 1 | [추정, 미측정] |
| OS 등 | 약 4~8 | [추정, 미측정] |

32K context, 세션 1~2개 기준 남는 메모리는 약 25~35 GiB다. [추정] 이 몫이 n-gram page cache나 row cache로 쓸 수 있는 공간이다.

### n-gram table

- **크기**: 320,001,536행 × 160, BF16, 행당 320 B, 95.37 GiB. head 16개(n=2와 n=3 각 8개)가 약 2천만 행씩(소수 크기) 겹치지 않는 구간을 가진다. 공식 FP8판은 47.68 GiB(행당 160 B + 텐서 전체 scale)다. [확인: GGUF kv `qwen4exp.ple.*`, header]
- **위치**: PLE는 0-based layer 1에 있다. config의 `ple_layer_ids: [2]`는 1-based다(HF `modeling_qwen4_exp.py:1260 layer_idx + 1`). [확인]
- **토큰당 읽기**: 16행 × 320 B = 5,120 B. 행 위치는 현재 토큰과 앞 2토큰의 hash로 정해진다(`ds4.c:67273 qwen4_ple_step`). page cache miss면 4 KiB page 16개, 64 KiB를 읽는다. [확인: 코드 / 추정: page 단위 비용]
- **필요 대역폭** [추정: 계산]:
  - decode 20~60 tok/s: 320~960 IOPS, 1.3~3.8 MB/s. MTP depth 3이면 cycle당 48행. 세션 B개면 B배.
  - prefill: 750 t/s(ds4)에서 12K IOPS, 49 MB/s. 1,500 t/s면 24K IOPS, 98 MB/s.
  - NVMe 대역폭으로는 작다. 병목은 지연이다. ds4 실측 cold 페널티는 토큰당 +2.9 ms였다(16번을 순서대로 pread). [확인: 프로파일 / 미확인: GB10 NVMe 사양]
- **ds4 처리 방식** [확인: 코드]:
  - mmap하지 않는다. table 범위는 load할 때 munmap하고, 별도 fd에 `POSIX_FADV_RANDOM`과 `NOREUSE`를 건다(`ds4.c:2800-2848`).
  - 행마다 `pread`한다(`57347 qwen4_ngram_row`).
  - 256행 미만이면 한 thread가 순서대로 읽는다(decode는 16행이라 여기에 해당). 256행 이상이면 정렬한 뒤 16 thread가 나눠 읽는다(`57424 qwen4_ngram_read`).
  - 앱 수준 캐시는 없다. warm 효과는 OS page cache뿐이다.
  - GPU forward 전에 host에서 동기로 읽고 업로드한다(`58689-58708`). CUDA에서는 계산과 겹치지 않는다.
  - vLLM은 table을 pinned host memory에 두고 UVA로 읽으면서 side stream으로 prefetch한다(`docs/features/engram.md`). 단일 GB10에는 맞지 않는다.
- **메모리에 올리기**:
  - BF16 95.37 GiB: 불가.
  - 공식 FP8 47.68 GiB: 가중치 72.5 GiB와 합치면 120.2 GiB로 MemTotal보다 크다. 불가. [확인: 계산]
  - expert를 3비트 이하(ds4 Q2급, 상주 41.7 GiB)로 낮추면 들어가지만 firepit 방식에서 벗어난다.
  - 현실적인 안은 남는 25~35 GiB를 앱 수준 row cache로 쓰는 것이다. 320 B 행 기준 약 0.8~1.2억 행(전체의 25~37%), FP8 행으로 두면 그 두 배를 담는다. hit rate는 n-gram 빈도 분포에 달려 있다. [추정 / 미확인: 분포]
- **firepit 관점** [추정]:
  - 읽기는 std `FileExt::read_at`(safe)로 충분하고 mmap(unsafe)이 필요 없다.
  - prefill의 행 번호는 토큰 id만으로 미리 정해지므로 병렬 읽기를 layer 0 계산과 겹칠 수 있다.
  - decode는 다음 토큰이 샘플링된 뒤에야 hash를 계산할 수 있다. 읽기를 layer 0과 겹치려면 decode graph를 PLE layer(1) 앞에서 나누거나, 16행을 병렬로 읽어 지연을 한 번 값(약 0.1~0.2 ms)으로 줄여야 한다. 설계에서 정할 점이다.

## 3. 정답지

### ds4의 덤프 [확인: 코드]

- `./ds4 -m MODEL --first-token-test` + 환경변수(`ds4.c:67701-67900`):
  - `DS4_QWEN4_FT_TOKENS=id,id,…` (없으면 prompt)
  - `DS4_QWEN4_FT_OUT`: [n][vocab] f32 logits
  - `DS4_QWEN4_FT_HIDDEN`: [n][48][4×2560] f32. 모든 trunk layer 뒤의 4-stream
  - `DS4_QWEN4_MTP_OUT`: [n−1][vocab] teacher-forced MTP 초안
  - 이 값들은 **CPU 순차 reference**(`qwen4_ref_*`, double 누산 포함)가 만든다. 같은 GGUF 가중치를 dequant해서 쓴다.
  - `DS4_QWEN4_GPU=1`을 더하면 GPU graph도 돌려 위치마다 `max|gpu-cpu|`와 top-1을 출력한다. 이때 dump되는 logits는 GPU 값이다. GPU는 layer별 hidden을 dump하지 않는다.
  - `ds4_backend_uses_graph`가 CUDA를 포함하므로 CUDA에서도 동작한다.
- `DS4_QWEN4_FT_LIST=manifest`(`"<prompt ids>|<target ids>\t<out>"`): 실제 GPU 경로(chunk prefill + teacher-forced decode)의 all-row logits를 파일로 쓴다. `tests/test_qwen4_logit_dump.py`가 이것으로 chunk 경계를 비교한다.
- activation capture(dir-steering)는 Qwen에서 Metal 전용이다. [확인: README]
- `make test-qwen4-cuda`(`tests/test_qwen4_kernels.c`): 모델 가중치 없이 커널별로 무작위 입력을 넣고 double CPU reference와 비교한다. 대상은 HC, GDN, indexer score/select, sparse attention, PLE, router, MoE 형식별, MTP stage, decode fusion 등이고 판정은 `check_close`(scale 대비 상대오차)다. 모델 수준 비교는 하지 않는다.
- 외부 기준: `gguf-tools/quality-testing`가 Alibaba 공식 API의 continuation(짧은 prompt 100개, 2K~24K prompt 12개)과 비교해 점수를 낸다. [확인: docs]

### ds4 외 기준 구현 [확인: 파일 존재·코드 일부]

| 구현 | trunk | MTP | 비고 |
|---|---|---|---|
| HF transformers `models/qwen4_exp` | ○ (v5.16.0부터. v5.15.0에는 없음) | **×** (`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`) | `output_hidden_states`는 decoder layer마다 4-stream을 기록한다(`_can_record_outputs`). n-gram table은 device_map에서 빼서 CPU에 둔다(`_no_placement_params`) |
| vLLM `vllm/models/qwen4_exp/{nvidia,amd}` | ○ | ○ (`nvidia/mtp.py`) | NVIDIA README가 지정한 commit `d4d703c`. MTP 수정 PR #55513은 merge되지 않고 닫혔다(2026-09-08) |
| SGLang `srt/models/qwen4_exp.py`, `qwen4_exp_mtp.py`, `qwen4_exp_ple_table.py` | ○ | ○ | 내용은 읽지 않음 [미확인] |
| llama.cpp `src/models/qwen4exp.cpp` (#27742, 2026-08-27) | ○ | MTP 블록 load(`:290`), MTP 전용 GGUF 지원 | speculative 사용 여부는 미확인 |

**문서와 다른 점**: `docs/design.md:69`는 MTP 정답 기준을 "HF transformers"로 적었다. HF는 Qwen3.6(`qwen3_5_moe`, `modeling_qwen3_5_moe.py:1004`)과 Flash-Next 모두 MTP 가중치를 무시한다. MTP 정답은 vLLM/SGLang 코드나 ds4 CPU reference로 바꿔야 한다. `models.md`의 "Flash-Next는 사실상 ds4 하나에 의존"도 지금은 맞지 않는다.

### GB10에서 정답지를 만드는 절차 (제안) [추정]

**A. trunk op 단위 (주 정답)**: 3.6과 같은 방식으로 HF 모듈을 그대로 쓴다.
1. `Qwen/Qwen3.8-Flash-Next`에서 앞 4층과 embedding, lm_head, final mixer가 든 shard 14개(32.9 GB)만 받는다. 2층이면 8개, 18.9 GB다. [확인: index 계산] PLE가 layer 1이라 2층 이상이면 PLE가 들어간다.
2. n-gram table은 받지 않는다. 실행할 토큰의 행만 HF shard에서 range 요청으로 읽는다. 행 r은 `shard_{r // 2500012}` 안에 있고, 150토큰이면 2,400행이다. HF `Qwen4ExpTextNGramEmbedding.ngram_embedding`을 이 행들만 담은 lookup으로 바꾼다. 수식이 아니라 데이터 접근만 바꾸는 것이라 "HF 모듈 그대로" 원칙과 충돌하지 않는다고 본다.
3. 기록은 `output_hidden_states`와 hook으로 HC mix 입출력, QSA 선택 블록, PLE 출력을 남긴다. 판정은 기존 체계(Higham–Mary, op별 상대오차)를 그대로 쓴다.
4. 전체 모델 BF16 logits가 필요하면 HF disk offload(258 GB를 forward마다 SSD에서 읽음)로 짧은 prompt 한 번 정도는 만들 수 있다. 전체 360 GB를 받아야 하므로 지금 규칙(대용량 다운로드 금지)에서는 하지 않았다.

**B. MTP**: ds4 CPU reference의 `DS4_QWEN4_MTP_OUT` + `FT_HIDDEN`을 쓰거나, vLLM `nvidia/mtp.py`를 정답 코드로 삼아 HF BF16 MTP 텐서로 재현한다. 후자는 수식 재구현이라 AGENTS 원칙과 충돌한다. 3.6 MTP도 같은 문제가 있다.

**C. ds4 end-to-end (보조)**:
1. Spark `/var/lib/sam-ds4/ds4`(`0aaea5a`, `make cuda-spark`)를 쓴다.
2. 모델은 Q4 GGUF(165.11 GiB, 상주 69.73 GiB)를 권한다. 4비트 expert + Q8_0 dense라 firepit NVFP4 구성과 정밀도 수준이 가장 가깝다. 받으려면 165 GiB 다운로드가 필요하다. Q2는 프로파일 때 받은 파일이 Spark에 남아 있을 수 있다. [미확인]
3. `DS4_QWEN4_FT_LIST`로 GPU all-row logits를 받는다. layer별 4-stream이 필요하면 `DS4_QWEN4_FT_TOKENS` + `FT_HIDDEN`(CPU reference, 토큰당 1.97 MB)을 쓴다. CPU reference 속도는 모른다. [미확인]

### 양자화 GGUF를 기준으로 쓸 때의 허용 오차 (의견) [추정]

- firepit(NVFP4 expert, BF16/FP8 dense)과 ds4 Q4(Q4_K/MXFP4 expert, Q8_0 dense, F16 HC)는 가중치 자체가 다르다. logits 원소 단위 허용 오차는 의미가 없다.
- 원소 단위 비교를 하려면 가중치를 맞춰야 한다. firepit이 GGUF 가중치를 f32로 dequant해 prefix 몇 층만 load하는 비교 경로를 만들면, ds4 `FT_HIDDEN`(CPU, double 누산)과 layer 단위로 비교할 수 있다. 이때 판정은 기존 GEMV 상한을 쓴다. 비용 대비 A안(HF op 단위)이 더 깨끗해서 권하지 않는다.
- ds4는 분포 지표로만 비교하자. 3.6 실측(NVFP4 혼합 vs bf16: KL 2.5e-2, top-1 93.3%, ΔNLL +1.1e-2)을 근거로 두 양자화 모델 사이에서는 KL 약 5e-2, top-1 약 90%를 이상 감지선으로 시작해 보자는 것이 의견이다. 기준선 자체를 먼저 실측해야 한다. 구체적으로는 ds4 Q4와 ds4 Q2 사이 지표, 또는 둘 다를 공식 API continuation(ds4 quality-testing)과 비교한 top-1을 같은 묶음에서 재고, firepit이 그 범위 안에 드는지 본다.

---

## 확인하지 않은 것

- SGLang qwen4_exp 코드 내용, llama.cpp의 MTP speculative 사용 여부.
- `mtp.hybrid`의 의미.
- GB10 NVMe 사양, n-gram 행 접근 분포(cache hit rate), firepit의 prefill 활성값·JIT·OS 메모리 실측.
- ds4 CPU reference(`--first-token-test`)의 토큰당 실행 시간.
- Spark에 Q2 GGUF가 아직 남아 있는지.

