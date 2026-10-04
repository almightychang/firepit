import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as qwen_module
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextRotaryEmbedding, apply_rotary_pos_emb, causal_conv1d_fn, causal_conv1d_update

MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
CHECKPOINT_PREFIX = "model.language_model."


def prefix_config(model_dir: Path, layers: int) -> Qwen3_5MoeTextConfig:
    text = json.loads((model_dir / "config.json").read_text())["text_config"]
    text |= {"num_hidden_layers": layers, "layer_types": text["layer_types"][:layers], "mtp_num_hidden_layers": 0}
    return Qwen3_5MoeTextConfig(**text)


def prefix_weights(model_dir: Path, layers: int, dtype=torch.float32, device="cpu") -> dict[str, torch.Tensor]:
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
                weights[local] = f.get_tensor(name).to(device=device, dtype=dtype)
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


def load_engine(model_dir: Path, layers: int):
    with torch.device("meta"):
        model = Qwen3_5MoeForCausalLM(prefix_config(model_dir, layers))
    weights = prefix_weights(model_dir, layers, torch.bfloat16, "cuda")
    missing, unexpected = model.load_state_dict(weights, strict=False, assign=True)
    del weights
    if unexpected or missing:
        raise SystemExit(f"missing={missing} unexpected={unexpected}")
    model.model.rotary_emb = Qwen3_5MoeTextRotaryEmbedding(model.config)
    norm = model.model.norm
    model.model.norm = torch.nn.Identity()
    return model.to("cuda").eval(), norm.to("cuda").eval()


def trace_engine(model, norm, tokens: list[int]) -> dict[str, torch.Tensor]:
    with torch.inference_mode():
        out = model.model(
            input_ids=torch.tensor([tokens], device="cuda"),
            use_cache=False,
            output_hidden_states=True,
        )
        records = {
            f"layer{layer}/hidden": hidden[0, -1].float().cpu().contiguous()
            for layer, hidden in enumerate(out.hidden_states[1:])
        }
        records["logits"] = model.lm_head(norm(out.hidden_states[-1][:, -1:]))[0, 0].float().cpu().contiguous()
        records["tokens"] = torch.tensor(tokens, dtype=torch.float32)
    return records


def use_torch_reference_kernels():
    for name in (
        "torch_chunk_gated_delta_rule",
        "torch_recurrent_gated_delta_rule",
        "causal_conv1d_fn",
        "causal_conv1d_update",
    ):
        setattr(qwen_module, name, getattr(qwen_module, name).__wrapped__)


def trace(model, norm, tokens: list[int]) -> dict[str, torch.Tensor]:
    records, cache = {}, None
    step = 0
    attention = {}
    gdn_qkv = {}
    gdn_calls = [0]
    recurrent = qwen_module.torch_recurrent_gated_delta_rule
    chunk = qwen_module.torch_chunk_gated_delta_rule

    def delta_hook(fn):
        def record(query, key, value, **kwargs):
            layer = gdn_calls[0]
            gdn_calls[0] += 1
            prefix = f"step{step}/layer{layer}/gdn"
            prior = kwargs.get("initial_state")
            records[f"{prefix}/delta_q"] = query[0, 0].clone()
            records[f"{prefix}/delta_k"] = key[0, 0].clone()
            records[f"{prefix}/delta_v"] = value[0, 0].clone()
            records[f"{prefix}/g"] = kwargs["g"][0, 0].clone()
            records[f"{prefix}/beta"] = kwargs["beta"][0, 0].clone()
            records[f"{prefix}/recurrent_before"] = (torch.zeros(query.shape[2], query.shape[3], value.shape[3]) if prior is None else prior[0].clone())
            output, state = fn(query, key, value, **kwargs)
            records[f"{prefix}/core"] = output[0, 0].clone()
            records[f"{prefix}/recurrent_after"] = state[0].clone()
            direct_args = dict(kwargs)
            direct_args["initial_state"] = prior
            direct_args["output_final_state"] = True
            direct_output, direct_state = recurrent(query, key, value, **direct_args)
            records[f"{prefix}/direct_core"] = direct_output[0, 0].clone()
            records[f"{prefix}/direct_recurrent_after"] = direct_state[0].clone()
            return output, state
        return record

    qwen_module.torch_recurrent_gated_delta_rule = delta_hook(recurrent)
    qwen_module.torch_chunk_gated_delta_rule = delta_hook(chunk)

    def gdn_norm_input(layer):
        def record(_module, args):
            prefix = f"step{step}/layer{layer}/gdn"
            records[f"{prefix}/norm_in"] = args[0].clone()
            records[f"{prefix}/norm_z"] = args[1].clone()
        return record

    def gdn_norm_output(layer):
        def record(_module, _args, output):
            records[f"step{step}/layer{layer}/gdn/norm_out"] = output.clone()
        return record

    def gdn_output(layer):
        def record(_module, _args, output):
            records[f"step{step}/layer{layer}/gdn/delta"] = output[0, 0].clone()
        return record
    def gdn_input(layer):
        def record(_module, _args, kwargs):
            records[f"step{step}/layer{layer}/gdn/in"] = kwargs["hidden_states"][0, 0].clone()
        return record

    def gdn_projection(layer, name):
        def record(_module, _args, output):
            value = output[0, 0].clone()
            records[f"step{step}/layer{layer}/gdn/{name}"] = value
            if name == "qkv":
                gdn_qkv[layer] = value
        return record
    def expert_hook(layer):
        def record(_module, args, output):
            records[f"step{step}/layer{layer}/shared_expert/in"] = args[0][0].clone()
            records[f"step{step}/layer{layer}/shared_expert/out"] = output[0].clone()
        return record

    def moe_input(layer):
        def record(_module, args):
            records[f"step{step}/layer{layer}/moe/in"] = args[0][0, 0].clone()
        return record

    def router_hook(layer):
        def record(_module, _args, output):
            prefix = f"step{step}/layer{layer}/moe"
            records[f"{prefix}/logits"] = output[0][0].clone()
            records[f"{prefix}/weights"] = output[1][0].clone()
            records[f"{prefix}/indices"] = output[2][0].to(torch.float32).clone()
        return record

    def routed_hook(layer):
        def record(_module, _args, output):
            records[f"step{step}/layer{layer}/moe/routed"] = output[0].clone()
        return record

    def moe_output(layer):
        def record(_module, _args, output):
            records[f"step{step}/layer{layer}/moe/delta"] = output[0, 0].clone()
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
    for layer, block in enumerate(model.model.layers):
        hooks += [
            block.mlp.register_forward_pre_hook(moe_input(layer)),
            block.mlp.gate.register_forward_hook(router_hook(layer)),
            block.mlp.experts.register_forward_hook(routed_hook(layer)),
            block.mlp.register_forward_hook(moe_output(layer)),
        ]
        if block.block_type == "linear_attention":
            gdn = block.linear_attn
            hooks += [gdn.register_forward_pre_hook(gdn_input(layer), with_kwargs=True)]
            hooks += [gdn.norm.register_forward_pre_hook(gdn_norm_input(layer)),
                      gdn.norm.register_forward_hook(gdn_norm_output(layer)),
                      gdn.out_proj.register_forward_hook(gdn_output(layer))]
            for name in ("qkv", "z", "b", "a"):
                hooks.append(getattr(gdn, f"in_proj_{name}").register_forward_hook(gdn_projection(layer, name)))
    if len(model.model.layers) > 3:
        attn = model.model.layers[3].self_attn
        hooks += [
            attn.register_forward_pre_hook(attn_input, with_kwargs=True),
            attn.q_proj.register_forward_hook(projection("q_proj", model.config.num_attention_heads)),
            attn.v_proj.register_forward_hook(projection("v_proj", 2)),
            attn.q_norm.register_forward_hook(normalized("q")),
            attn.k_norm.register_forward_hook(normalized("k")),
            model.model.rotary_emb.register_forward_hook(rotary),
            attn.o_proj.register_forward_pre_hook(output_input),
            attn.register_forward_hook(attn_output),
        ]
    with torch.no_grad():
        positions = torch.tensor([0, 1, 2, 3, 4, 100, 4095, 65535, 262143])
        position_axes = len(model.config.rope_parameters["mrope_section"])
        position_ids = positions[None, None, :].expand(position_axes, 1, -1)
        cos, sin = model.model.rotary_emb(torch.zeros(1, len(positions), model.config.hidden_size), position_ids)
        records["rope/positions"] = positions.to(torch.float32)
        records["rope/cos"] = cos[0].clone()
        records["rope/sin"] = sin[0].clone()
        for step, token in enumerate(tokens):
            attention.clear()
            gdn_qkv.clear()
            gdn_calls[0] = 0
            previous = {}
            for layer, block in enumerate(model.model.layers):
                if block.block_type == "linear_attention":
                    gdn = block.linear_attn
                    prior = None if cache is None else cache.layers[layer].conv_states[0]
                    previous[layer] = (torch.zeros(1, gdn.conv_dim, gdn.conv_kernel_size) if prior is None else prior.clone())
                    records[f"step{step}/layer{layer}/gdn/state_before"] = previous[layer][0].clone()
            out = model.model(input_ids=torch.tensor([[token]]), past_key_values=cache, use_cache=True, output_hidden_states=True)
            cache = out.past_key_values
            for layer, prior in previous.items():
                gdn = model.model.layers[layer].linear_attn
                key = f"step{step}/layer{layer}/gdn"
                records[f"{key}/state_after"] = cache.layers[layer].conv_states[0][0].clone()
                state = prior[..., 1:].clone()
                mixed = causal_conv1d_update(gdn_qkv[layer].reshape(1, gdn.conv_dim, 1), state, gdn.conv1d.weight.squeeze(1), gdn.conv1d.bias, gdn.activation)
                records[f"{key}/conv"] = mixed[0, :, 0].clone()
                if step == 0:
                    first = causal_conv1d_fn(gdn_qkv[layer].reshape(1, gdn.conv_dim, 1), gdn.conv1d.weight.squeeze(1), gdn.conv1d.bias, activation=gdn.activation)
                    records[f"{key}/conv_first_path"] = first[0, :, -1].clone()
                records[f"{key}/state_update"] = state[0].clone()
                q, k, v = torch.split(mixed[0, :, 0], [gdn.key_dim, gdn.key_dim, gdn.value_dim])
                records[f"{key}/q"] = q.clone()
                records[f"{key}/k"] = k.clone()
                records[f"{key}/v"] = v.clone()
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
            prefix_norm = norm(out.hidden_states[-1])
            records[f"step{step}/prefix_logits"] = model.lm_head(prefix_norm)[0, 0].clone()
    for hook in hooks:
        hook.remove()
    qwen_module.torch_recurrent_gated_delta_rule = recurrent
    qwen_module.torch_chunk_gated_delta_rule = chunk
    return records


def cache_records(cache) -> dict[str, torch.Tensor]:
    records = {}
    for layer, state in enumerate(cache.layers):
        prefix = f"layer{layer}/cache"
        if hasattr(state, "conv_states"):
            records[f"{prefix}/conv"] = state.conv_states[0][0].clone()
            records[f"{prefix}/recurrent"] = state.recurrent_states[0][0].clone()
        else:
            records[f"{prefix}/k"] = state.keys[0].clone()
            records[f"{prefix}/v"] = state.values[0].clone()
    return records


def trace_prefill(model, norm, tokens: list[int]) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    records = {}
    hooks = []
    attention = {}
    gdn_qkv = {}
    chunk = qwen_module.torch_chunk_gated_delta_rule
    gdn_calls = [0]

    def record_chunk(query, key, value, **kwargs):
        layer = gdn_calls[0]
        gdn_calls[0] += 1
        if layer < 3:
            prefix = f"layer{layer}/gdn/chunk"
            for name, tensor in (("q", query), ("k", key), ("v", value),
                                 ("g", kwargs["g"]), ("beta", kwargs["beta"])):
                records[f"{prefix}/{name}"] = tensor[0].contiguous().clone()
        return chunk(query, key, value, **kwargs)

    qwen_module.torch_chunk_gated_delta_rule = record_chunk
    projections = (
        (0, "linear_attn", "in_proj_qkv"),
        (0, "linear_attn", "in_proj_z"),
        (3, "self_attn", "q_proj"),
    )
    for layer, block_name, name in projections:
        if layer >= len(model.model.layers):
            continue
        module = getattr(getattr(model.model.layers[layer], block_name), name)
        key = f"layer{layer}/{name}"
        def record(_module, args, output, key=key):
            records[f"{key}/input"] = args[0][0].clone()
            records[f"{key}/output"] = output[0].clone()
        hooks.append(module.register_forward_hook(record))
    for layer, block in enumerate(model.model.layers):
        def moe_input(_module, args, layer=layer):
            records[f"layer{layer}/moe/in"] = args[0][0].clone()
        def router(_module, _args, output, layer=layer):
            prefix = f"layer{layer}/moe"
            records[f"{prefix}/logits"] = output[0].clone()
            records[f"{prefix}/weights"] = output[1].clone()
            records[f"{prefix}/indices"] = output[2].to(torch.float32).clone()
        def routed(_module, _args, output, layer=layer):
            records[f"layer{layer}/moe/routed"] = output.clone()
        def moe_output(_module, _args, output, layer=layer):
            records[f"layer{layer}/moe/delta"] = output[0].clone()
        def shared(_module, _args, output, layer=layer):
            records[f"layer{layer}/moe/shared"] = output.clone()
        def shared_gate(_module, _args, output, layer=layer):
            records[f"layer{layer}/moe/shared_gate"] = output.clone()
        hooks.extend((
            block.mlp.register_forward_pre_hook(moe_input),
            block.mlp.gate.register_forward_hook(router),
            block.mlp.experts.register_forward_hook(routed),
            block.mlp.shared_expert.register_forward_hook(shared),
            block.mlp.shared_expert_gate.register_forward_hook(shared_gate),
            block.mlp.register_forward_hook(moe_output),
        ))
        if block.block_type != "linear_attention":
            continue
        gdn = block.linear_attn
        def record_qkv(_module, _args, output, layer=layer):
            gdn_qkv[layer] = output[0].clone()
            records[f"layer{layer}/gdn/qkv"] = output[0].clone()
        hooks.append(gdn.in_proj_qkv.register_forward_hook(record_qkv))
        if layer < 3:
            def record_z(_module, _args, output, layer=layer):
                records[f"layer{layer}/gdn/z"] = output[0].clone()
            def record_core(_module, args, layer=layer):
                records[f"layer{layer}/gdn/core"] = args[0].reshape(len(tokens), -1, args[0].shape[-1]).clone()
            def record_delta(_module, _args, output, layer=layer):
                records[f"layer{layer}/gdn/delta"] = output[0].clone()
            hooks.extend((
                gdn.in_proj_z.register_forward_hook(record_z),
                gdn.norm.register_forward_pre_hook(record_core),
                gdn.out_proj.register_forward_hook(record_delta),
            ))
    if len(model.model.layers) > 3:
        attn = model.model.layers[3].self_attn
        def normalized(name):
            def record(_module, _args, output):
                attention[name] = output.clone()
            return record
        def rotary(_module, _args, output):
            attention["cos"], attention["sin"] = (x.clone() for x in output)
        def value(_module, _args, output):
            heads = model.config.num_key_value_heads
            records["layer3/attn/v"] = output[0].reshape(len(tokens), heads, -1).clone()
        def gate(_module, _args, output):
            heads = model.config.num_attention_heads
            _, g = output[0].reshape(len(tokens), heads, -1).chunk(2, dim=-1)
            records["layer3/attn/gate"] = g.reshape(len(tokens), -1).clone()
        def output_input(_module, args):
            records["layer3/attn/o_proj_in"] = args[0][0].clone()
        def attn_output(_module, _args, output):
            records["layer3/attn/delta"] = output[0][0].clone()
        hooks += [
            attn.q_norm.register_forward_hook(normalized("q")),
            attn.k_norm.register_forward_hook(normalized("k")),
            model.model.rotary_emb.register_forward_hook(rotary),
            attn.v_proj.register_forward_hook(value),
            attn.q_proj.register_forward_hook(gate),
            attn.o_proj.register_forward_pre_hook(output_input),
            attn.register_forward_hook(attn_output),
        ]
    with torch.no_grad():
        out = model.model(
            input_ids=torch.tensor([tokens]), use_cache=True, output_hidden_states=True
        )
        for layer, qkv in gdn_qkv.items():
            gdn = model.model.layers[layer].linear_attn
            conv = causal_conv1d_fn(
                qkv.transpose(0, 1).unsqueeze(0),
                gdn.conv1d.weight.squeeze(1),
                gdn.conv1d.bias,
                activation=gdn.activation,
            )[0].transpose(0, 1).contiguous()
            q, k, v = torch.split(conv, [gdn.key_dim, gdn.key_dim, gdn.value_dim], dim=-1)
            prefix = f"layer{layer}/gdn"
            records[f"{prefix}/q"] = q.clone()
            records[f"{prefix}/k"] = k.clone()
            records[f"{prefix}/v"] = v.clone()
            records[f"{prefix}/conv_state"] = out.past_key_values.layers[layer].conv_states[0][0].clone()
        if attention:
            q, k = apply_rotary_pos_emb(
                attention["q"].transpose(1, 2), attention["k"].transpose(1, 2),
                attention["cos"], attention["sin"],
            )
            records["layer3/attn/q"] = q[0].transpose(0, 1).contiguous()
            records["layer3/attn/k"] = k[0].transpose(0, 1).contiguous()
        for layer, residual in enumerate(out.hidden_states):
            records[f"residual{layer}"] = residual[0].clone()
        records["logits"] = model.lm_head(norm(out.hidden_states[-1][:, -1:]))[0, 0].clone()
        records.update(cache_records(out.past_key_values))
        for hook in hooks:
            hook.remove()
        qwen_module.torch_chunk_gated_delta_rule = chunk

        decode_cache = None
        maximum = {"residual": 0.0, "logits": 0.0, "state": 0.0}
        state_maximum = {}
        for position, token in enumerate(tokens):
            step = model.model(
                input_ids=torch.tensor([[token]]), past_key_values=decode_cache,
                use_cache=True, output_hidden_states=True,
            )
            decode_cache = step.past_key_values
            for layer, residual in enumerate(step.hidden_states):
                key = f"residual{layer}"
                maximum["residual"] = max(
                    maximum["residual"],
                    (records[key][position] - residual[0, 0]).abs().max().item(),
                )
        logits = model.lm_head(norm(step.hidden_states[-1]))[0, 0]
        maximum["logits"] = (records["logits"] - logits).abs().max().item()
        for key, value in cache_records(decode_cache).items():
            state_maximum[key] = (records[key] - value).abs().max().item()
            maximum["state"] = max(maximum["state"], state_maximum[key])
        maximum["state_by_key"] = state_maximum
    return records, maximum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--prefill", action="store_true")
    parser.add_argument("--engine", action="store_true")
    args = parser.parse_args()

    if args.engine and args.prefill:
        parser.error("--engine and --prefill are separate modes")

    if args.engine:
        text = json.loads((args.model / "config.json").read_text())["text_config"]
        layers = text["num_hidden_layers"] if args.layers is None else args.layers
    else:
        layers = 4 if args.layers is None else args.layers

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.prefill or args.engine:
        source = args.prompt * 100
        tokens = tokenizer(source, add_special_tokens=False)["input_ids"][:150]
        if len(tokens) != 150:
            raise SystemExit(f"prompt produced only {len(tokens)} tokens")
    else:
        tokens = tokenizer(args.prompt)["input_ids"]
    if args.engine:
        if not torch.cuda.is_available():
            raise SystemExit("--engine requires a CUDA PyTorch build")
        use_torch_reference_kernels()
        model, norm = load_engine(args.model, layers)
        records = trace_engine(model, norm, tokens)
    else:
        model, norm = load_prefix(args.model, layers)
    if args.prefill:
        records, maximum = trace_prefill(model, norm, tokens)
        print(f"HF prefill vs decode max abs: {maximum}", flush=True)
    elif not args.engine:
        records = trace(model, norm, tokens)
    metadata = {"model": MODEL_ID, "layers": str(layers), "prompt": args.prompt, "tokens": json.dumps(tokens)}
    if args.engine:
        metadata["mode"] = "engine"
    if args.prefill:
        metadata["mode"] = "prefill"
        metadata["max_abs"] = json.dumps(maximum)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(records, args.out, metadata=metadata)
    print(f"tokens={tokens} records={len(records)} -> {args.out}")


if __name__ == "__main__":
    main()
