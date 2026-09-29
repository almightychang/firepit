import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM

MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
CHECKPOINT_PREFIX = "model.language_model."


def prefix_config(model_dir: Path, layers: int) -> Qwen3_5MoeTextConfig:
    text = json.loads((model_dir / "config.json").read_text())["text_config"]
    text |= {"num_hidden_layers": layers, "layer_types": text["layer_types"][:layers], "mtp_num_hidden_layers": 0}
    return Qwen3_5MoeTextConfig(**text)


def prefix_weights(model_dir: Path, layers: int) -> dict[str, torch.Tensor]:
    def local_name(name: str) -> str | None:
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
    if unexpected or set(missing) != {"lm_head.weight"}:
        raise SystemExit(f"missing={missing} unexpected={unexpected}")
    model.model.norm = torch.nn.Identity()
    return model.eval()


def trace(model, tokens: list[int]) -> dict[str, torch.Tensor]:
    records, cache = {}, None
    with torch.no_grad():
        for step, token in enumerate(tokens):
            out = model.model(input_ids=torch.tensor([[token]]), past_key_values=cache, use_cache=True, output_hidden_states=True)
            cache = out.past_key_values
            for layer, residual in enumerate(out.hidden_states):
                records[f"step{step}/residual{layer}"] = residual[0, 0].clone()
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--prompt", default="The capital of France is")
    args = parser.parse_args()

    tokens = AutoTokenizer.from_pretrained(args.model)(args.prompt)["input_ids"]
    records = trace(load_prefix(args.model, args.layers), tokens)
    metadata = {"model": MODEL_ID, "layers": str(args.layers), "prompt": args.prompt, "tokens": json.dumps(tokens)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(records, args.out, metadata=metadata)
    print(f"tokens={tokens} records={len(records)} -> {args.out}")


if __name__ == "__main__":
    main()
