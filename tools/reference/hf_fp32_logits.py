"""HF Qwen3.6 last-token logits with f32 activations (bf16 weights upcast per layer at call time)."""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import trace_qwen36 as t


def upcast(module, args, kwargs):
    module.float()
    lift = lambda v: v.float() if torch.is_tensor(v) and v.is_floating_point() else v
    return tuple(lift(a) for a in args), {k: lift(v) for k, v in kwargs.items()}


def downcast(module, _args, _kwargs, output):
    module.to(torch.bfloat16)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True, help='JSON {"name": [token ids], ...}')
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text())
    layers = json.loads((args.model / "config.json").read_text())["text_config"]["num_hidden_layers"]
    t.use_torch_reference_kernels()
    model, norm = t.load_engine(args.model, layers)
    model.model.embed_tokens.register_forward_hook(lambda _m, _i, output: output.float())
    for layer in model.model.layers:
        layer.register_forward_pre_hook(upcast, with_kwargs=True)
        layer.register_forward_hook(downcast, with_kwargs=True)
    norm.float()
    model.lm_head.float()
    for name, tokens in cases.items():
        with torch.inference_mode():
            hidden = model.model(input_ids=torch.tensor([tokens], device="cuda"), use_cache=False).last_hidden_state
            logits = model.lm_head(norm(hidden[:, -1:].float()))[0, 0]
        probs = torch.softmax(logits.double(), -1)
        top = torch.topk(probs, 5)
        print(name, [(int(i), round(float(p), 4)) for p, i in zip(top.values, top.indices)], flush=True)


if __name__ == "__main__":
    main()
