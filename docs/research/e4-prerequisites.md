# E4 사전 확인 (2026-10-05)

## cuTile 저정밀 지원 (vendor `a620b40`)

- 타입: `f8e4m3fn`, `f8e8m0fnu`(MX 스케일), packed `f4e2m1fnx2`(NVFP4), unpack으로 `f4e2m1fn` 타일.
- `mmaf_scaled`(블록 스케일 tensor core mma, f32 누적): Tile IR 13.3, `sm_100+`. README 요구사항 표와 `cutile-examples/src/requirements.rs`(sm_120, sm_121 지원).
- 공식 예제 `nvfp4`(FP4 + E4M3 스케일, 블록 16)와 `mxfp8`(E4M3 + E8M0 스케일, 블록 32)이 RTX 5080(sm_120)과 GB10(sm_121, 드라이버 580.178.04)에서 모두 통과(max_abs_err=0).
- 참고: README 표에 "Gather/scatter views — Tile IR 13.3"이 있다. MoE 융합형 gather load 후보(E7).

## Qwen 공식 FP8 체크포인트

`Qwen/Qwen3.6-35B-A3B-FP8` (HF API로 존재 확인, config.json 직접 확인):
- `quant_method: fp8`, `fmt: e4m3`, `weight_block_size: [128, 128]`(2차원 블록 스케일, `weight_scale_inv`), `activation_scheme: dynamic`.
- 변환하지 않은 모듈(`modules_to_not_convert`, 언어 모델 부분): `lm_head`, `embed_tokens`, 모든 norm, router(`mlp.gate`), `shared_expert_gate`, GDN `conv1d`·`in_proj_a`·`in_proj_b`·`A_log`·`dt_bias`. routed expert, shared expert, attention projection, GDN `in_proj_qkv`·`in_proj_z`·`out_proj`는 FP8.
