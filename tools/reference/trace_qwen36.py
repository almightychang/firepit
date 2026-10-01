import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextRotaryEmbedding, apply_rotary_pos_emb

MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
CHECKPOINT_PREFIX = "model.language_model."


def prefix_config(model_dir: Path, layers: int) -> Qwen3_5MoeTextConfig:
    text = json.loads((model_dir / "config.json").read_text())["text_config"]
    text |= {"num_hidden_layers": layers, "layer_types": text["layer_types"][:layers], "mtp_num_hidden_layers": 0}
    return Qwen3_5MoeTextConfig(**text)


def prefix_weights(model_dir: Path, layers: int) -> dict[str, torch.Tensor]:
    def local_name(name: str) -> str | None:
        if name == "lm_head.weight":
            return name
        if not name.startswith(CHECKPOINT_PREFIX):
            return None
        local = name.removeprefix(CHECKPOINT_PREFIX)
        if local.startswith("layers.") and int(local.split(".")[1]) >= layers:
            return None
        return "model." + local

    index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    weights = {}
    for name, shard in index.items():
        if (local := local_name(name)) is not None:
            with safe_open(model_dir / shard, framework="pt") as f:
                weights[local] = f.get_tensor(name).to(torch.float32)
    return weights


def load_prefix(model_dir: Path, layers: int) -> Qwen3_5MoeForCausalLM:
    with torch.device("meta"):
        model = Qwen3_5MoeForCausalLM(prefix_config(model_dir, layers))
    missing, unexpected = model.load_state_dict(prefix_weights(model_dir, layers), strict=False, assign=True)
    if unexpected or missing:
        raise SystemExit(f"missing={missing} unexpected={unexpected}")
    model.model.rotary_emb = Qwen3_5MoeTextRotaryEmbedding(model.config)
    norm = model.model.norm
    model.model.norm = torch.nn.Identity()
    return model.eval(), norm.eval()


def trace(model, norm, tokens: list[int]) -> dict[str, torch.Tensor]:
    records, cache = {}, None
    step = 0
    attention = {}
    def expert_hook(layer):
        def record(_module, args, output):
            records[f"step{step}/layer{layer}/shared_expert/in"] = args[0][0].clone()
            records[f"step{step}/layer{layer}/shared_expert/out"] = output[0].clone()
        return record

    def attn_input(_module, _args, kwargs):
        records[f"step{step}/layer3/attn/in"] = kwargs["hidden_states"][0, 0].clone()

    def projection(name, heads):
        def record(_module, _args, output):
            value = output[0, 0].reshape(heads, -1)
            if name == "q_proj":
                _, gate = value.chunk(2, dim=-1)
                records[f"step{step}/layer3/attn/gate"] = gate.reshape(-1).clone()
            else:
                records[f"step{step}/layer3/attn/v"] = value.reshape(-1).clone()
        return record

    def normalized(name):
        def record(_module, _args, output):
            attention[name] = output.clone()
        return record

    def rotary(_module, _args, output):
        attention["cos"], attention["sin"] = (x.clone() for x in output)

    def output_input(_module, args):
        records[f"step{step}/layer3/attn/o_proj_in"] = args[0][0, 0].clone()

    def attn_output(_module, _args, output):
        records[f"step{step}/layer3/attn/delta"] = output[0][0, 0].clone()

    hooks = [block.mlp.shared_expert.register_forward_hook(expert_hook(layer))
             for layer, block in enumerate(model.model.layers)]
    if len(model.model.layers) > 3:
        attn = model.model.layers[3].self_attn
        hooks += [
            attn.register_forward_pre_hook(attn_input, with_kwargs=True),
            attn.q_proj.register_forward_hook(projection("q_proj", 16)),
            attn.v_proj.register_forward_hook(projection("v_proj", 2)),
            attn.q_norm.register_forward_hook(normalized("q")),
            attn.k_norm.register_forward_hook(normalized("k")),
            model.model.rotary_emb.register_forward_hook(rotary),
            attn.o_proj.register_forward_pre_hook(output_input),
            attn.register_forward_hook(attn_output),
        ]
    with torch.no_grad():
        positions = torch.tensor([0, 1, 2, 3, 4, 100, 4095, 65535, 262143])
        position_ids = positions[None, None, :].expand(3, 1, -1)
        cos, sin = model.model.rotary_emb(torch.zeros(1, len(positions), 2048), position_ids)
        records["rope/positions"] = positions.to(torch.float32)
        records["rope/cos"] = cos[0].clone()
        records["rope/sin"] = sin[0].clone()
        for step, token in enumerate(tokens):
            attention.clear()
            out = model.model(input_ids=torch.tensor([[token]]), past_key_values=cache, use_cache=True, output_hidden_states=True)
            cache = out.past_key_values
            if len(model.model.layers) > 3:
                key = f"step{step}/layer3/attn"
                q, k = apply_rotary_pos_emb(attention["q"].transpose(1, 2), attention["k"].transpose(1, 2), attention["cos"], attention["sin"])
                records[f"{key}/q"] = q[0, :, 0].reshape(-1).clone()
                records[f"{key}/k"] = k[0, :, 0].reshape(-1).clone()
                records[f"{key}/q_norm"] = attention["q"][0, 0].reshape(-1).clone()
                records[f"{key}/k_norm"] = attention["k"][0, 0].reshape(-1).clone()
                records[f"{key}/cos"] = attention["cos"][0, 0].clone()
                records[f"{key}/sin"] = attention["sin"][0, 0].clone()
            for layer, residual in enumerate(out.hidden_states):
                records[f"step{step}/residual{layer}"] = residual[0, 0].clone()
            exit_norm = norm(out.hidden_states[0])
            records[f"step{step}/exit_norm"] = exit_norm[0, 0].clone()
            records[f"step{step}/logits"] = model.lm_head(exit_norm)[0, 0].clone()
    for hook in hooks:
        hook.remove()
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--prompt", default="The capital of France is")
    args = parser.parse_args()

    tokens = AutoTokenizer.from_pretrained(args.model)(args.prompt)["input_ids"]
    model, norm = load_prefix(args.model, args.layers)
    records = trace(model, norm, tokens)
    metadata = {"model": MODEL_ID, "layers": str(args.layers), "prompt": args.prompt, "tokens": json.dumps(tokens)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(records, args.out, metadata=metadata)
    print(f"tokens={tokens} records={len(records)} -> {args.out}")


if __name__ == "__main__":
    main()
