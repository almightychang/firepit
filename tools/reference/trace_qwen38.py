import argparse
import itertools
import json
import struct
import urllib.request
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers.core_model_loading import dot_natural_key
from transformers.models.qwen4_exp import modeling_qwen4_exp as qwen_module
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM, Qwen4ExpTextRotaryEmbedding

MODEL_ID = "Qwen/Qwen3.8-Flash-Next"
REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
CHECKPOINT_PREFIX = "model.language_model."
NGRAM_TABLE = "ple.ple_embedding.ngram_embedding."
SAFETENSORS_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
FIXTURE = Path(__file__).resolve().parents[2] / "crates/firepit/tests/fixtures/qwen38-prompt-tokens.json"


def prefix_config(model_dir: Path, layers: int) -> Qwen4ExpTextConfig:
    text = json.loads((model_dir / "config.json").read_text())["text_config"]
    text |= {"num_hidden_layers": layers, "layer_types": text["layer_types"][:layers], "mtp_num_hidden_layers": 0}
    return Qwen4ExpTextConfig(**text)


def checkpoint_index(model_dir: Path) -> dict[str, str]:
    return json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]


def prefix_weights(model_dir: Path, layers: int) -> dict[str, torch.Tensor]:
    def local_name(name: str) -> str | None:
        if name == "lm_head.weight":
            return name
        if not name.startswith(CHECKPOINT_PREFIX) or NGRAM_TABLE in name:
            return None
        local = name.removeprefix(CHECKPOINT_PREFIX)
        if local.startswith("layers.") and int(local.split(".")[1]) >= layers:
            return None
        return "model." + local

    weights = {}
    for name, shard in checkpoint_index(model_dir).items():
        if (local := local_name(name)) is not None:
            with safe_open(model_dir / shard, framework="pt") as f:
                tensor = f.get_tensor(name)
                weights[local] = tensor.float() if tensor.is_floating_point() else tensor
    return weights


class HubCheckpoint:
    def __init__(self, weight_map: dict[str, str], repo: str = MODEL_ID, revision: str = REVISION):
        self.weight_map = weight_map
        self.url = f"https://huggingface.co/{repo}/resolve/{revision}/"
        self.headers = {}

    def fetch(self, file: str, start: int, length: int) -> bytes:
        request = urllib.request.Request(self.url + file, headers={"Range": f"bytes={start}-{start + length - 1}"})
        with urllib.request.urlopen(request) as response:
            data = response.read()
            if response.status != 206 or len(data) != length:
                raise SystemExit(f"{file} bytes {start}+{length}: status {response.status}, got {len(data)} bytes")
        return data

    def header(self, file: str) -> tuple[int, dict]:
        if file not in self.headers:
            (size,) = struct.unpack("<Q", self.fetch(file, 0, 8))
            self.headers[file] = (8 + size, json.loads(self.fetch(file, 8, size)))
        return self.headers[file]

    def tensor(self, name: str) -> tuple[str, int, str, list[int]]:
        file = self.weight_map[name]
        data_start, header = self.header(file)
        info = header[name]
        return file, data_start + info["data_offsets"][0], info["dtype"], info["shape"]


class RangeTable:
    def __init__(self, hub: HubCheckpoint, prefix: str, rows: int, dim: int):
        self.hub = hub
        self.shards = sorted((name for name in hub.weight_map if name.startswith(prefix)), key=dot_natural_key)
        self.shard_rows, rest = divmod(rows, len(self.shards))
        if rest:
            raise SystemExit(f"{rows} rows do not split into {len(self.shards)} shards")
        self.dim = dim

    def locate(self, index: int) -> tuple[str, int]:
        shard, local = divmod(index, self.shard_rows)
        return self.shards[shard], local

    def row(self, index: int) -> torch.Tensor:
        name, local = self.locate(index)
        file, start, dtype, shape = self.hub.tensor(name)
        if shape != [self.shard_rows, self.dim]:
            raise SystemExit(f"{name} has shape {shape}, expected {[self.shard_rows, self.dim]}")
        dtype = SAFETENSORS_DTYPES[dtype]
        width = self.dim * dtype.itemsize
        return torch.frombuffer(bytearray(self.hub.fetch(file, start + local * width, width)), dtype=dtype)


class NGramRows(torch.nn.Module):
    def __init__(self, table: RangeTable, dtype: torch.dtype, offset: int = 0):
        super().__init__()
        self.table = table
        self.dtype = dtype
        self.weight = torch.empty(0, dtype=dtype)
        self.offset = offset
        self.rows = {}
        self.requested = []

    def row(self, index: int) -> torch.Tensor:
        if index not in self.rows:
            self.rows[index] = self.table.row(index).to(self.dtype)
        return self.rows[index]

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        self.requested.append(ids.clone())
        rows = [self.row(int(index) + self.offset) for index in ids.flatten()]
        return torch.stack(rows).reshape(*ids.shape, self.table.dim)


def ngram_embeddings(model) -> dict[int, torch.nn.Module]:
    return {layer: block.ple.ple_embedding for layer, block in enumerate(model.model.layers) if block.ple is not None}


def load_prefix(model_dir: Path, layers: int):
    weight_map = checkpoint_index(model_dir)
    hub = HubCheckpoint(weight_map)
    with torch.device("meta"):
        model = Qwen4ExpForCausalLM(prefix_config(model_dir, layers))
    tables = {}
    for layer, embedding in ngram_embeddings(model).items():
        table = embedding.ngram_embedding
        prefix = f"{CHECKPOINT_PREFIX}layers.{layer}.{NGRAM_TABLE}shard_"
        tables[layer] = NGramRows(RangeTable(hub, prefix, table.num_embeddings, table.embedding_dim), table.weight.dtype)
        embedding.ngram_embedding = tables[layer]
    missing, unexpected = model.load_state_dict(prefix_weights(model_dir, layers), strict=False, assign=True)
    if unexpected or missing:
        raise SystemExit(f"missing={missing} unexpected={unexpected}")
    model.model.rotary_emb = Qwen4ExpTextRotaryEmbedding(model.config)
    if meta := [name for name, t in itertools.chain(model.named_parameters(), model.named_buffers()) if t.is_meta]:
        raise SystemExit(f"tensors left on meta: {meta}")
    return model.eval(), tables


def residual_probe(model, record):
    def layer_input(_module, args):
        record("residual0", args[0][0, 0])

    def layer_output(layer):
        return lambda _module, _args, output: record(f"residual{layer + 1}", output[0, 0])

    layers = model.model.layers
    return [layers[0].register_forward_pre_hook(layer_input)] + [
        block.register_forward_hook(layer_output(layer)) for layer, block in enumerate(layers)
    ]


def hc_probe(model, record):
    def gated_residual(hc, prefix):
        def normed(_module, _args, output):
            record(f"{prefix}/normed", output[0, 0])

        def mix(_module, _args, output):
            record(f"{prefix}/mix", torch.sigmoid(output[0, 0]))

        def mixed(_module, _args, output):
            record(f"{prefix}/mixed", output[0][0, 0])
            record(f"{prefix}/inject", output[2][0, 0])

        return [
            hc.hc_norm.register_forward_hook(normed),
            hc.input_mix_weight_up.register_forward_hook(mix),
            hc.register_forward_hook(mixed),
        ]

    def block_out(prefix):
        return lambda _module, _args, output: record(
            f"{prefix}/block_out", (output[0] if isinstance(output, tuple) else output)[0, 0]
        )

    def attn_out(prefix):
        return lambda _module, args: record(f"{prefix}/out", args[0][0, 0])

    def mlp_out(prefix):
        return lambda _module, _args, output: record(f"{prefix}/out", output[0, 0])

    hooks = []
    for layer, block in enumerate(model.model.layers):
        attn, mlp = f"layer{layer}/hc/attn", f"layer{layer}/hc/mlp"
        mixer = block.linear_attn if block.layer_type == "linear_attention" else block.self_attn
        hooks += gated_residual(block.attn_hyper_connection, attn) + gated_residual(block.mlp_hyper_connection, mlp)
        hooks += [
            mixer.register_forward_hook(block_out(attn)),
            block.mlp.register_forward_hook(block_out(mlp)),
            block.mlp_hyper_connection.register_forward_pre_hook(attn_out(attn)),
            block.register_forward_hook(mlp_out(mlp)),
        ]
    return hooks


def trace(model, tokens: list[int], probes=(residual_probe, hc_probe)) -> dict[str, torch.Tensor]:
    records, cache = {}, None
    step = 0

    def record(name: str, tensor: torch.Tensor):
        records[f"step{step}/{name}"] = tensor.detach().clone()

    def exit_logits(residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mixed = model.model.hyper_connection_mixer(residual)
        return mixed, model.lm_head(mixed)

    hooks = [hook for probe in probes for hook in probe(model, record)]
    with torch.no_grad():
        for step, token in enumerate(tokens):
            out = model.model(input_ids=torch.tensor([[token]]), past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            exit_mix, logits = exit_logits(records[f"step{step}/residual0"])
            record("exit_mix", exit_mix)
            record("logits", logits)
            for layer in range(1, len(model.model.layers) + 1):
                record(f"logits_after{layer}", exit_logits(records[f"step{step}/residual{layer}"])[1])
    for hook in hooks:
        hook.remove()
    return records


def check_ngram(model, tables: dict[int, NGramRows], tokens: list[int], model_dir: Path) -> dict:
    results = {}
    weight_map = checkpoint_index(model_dir)
    for layer, embedding in ngram_embeddings(model).items():
        rows = tables[layer]
        table = rows.table
        buffers = {
            "layer_multipliers": qwen_module._build_layer_multipliers(
                embedding.unigram_vocab_size, embedding.ngram_size, embedding.ple_layer_index, embedding.seed
            ),
            "ngram_heads_vocab_sizes": torch.tensor(embedding.head_vocab_sizes),
            "ngram_heads_offsets": torch.tensor(embedding.head_offsets),
        }
        decode_ids = torch.cat(rows.requested, dim=1)
        rows.requested.clear()
        with torch.no_grad():
            embedding(torch.tensor([tokens]), None)
        whole_ids = rows.requested.pop()
        ids = decode_ids.unique()
        fresh = torch.stack([table.row(int(index)).to(rows.dtype) for index in ids])
        local = []
        for shard, name in enumerate(table.shards):
            if (model_dir / weight_map[name]).exists():
                with safe_open(model_dir / weight_map[name], framework="pt") as f:
                    for row in (0, table.shard_rows - 1):
                        expected = f.get_slice(name)[row : row + 1][0]
                        local.append(torch.equal(table.row(shard * table.shard_rows + row), expected))
        results[f"layer{layer}"] = {
            "buffers_match_hf_init": all(torch.equal(getattr(embedding, k), v) for k, v in buffers.items()),
            "decode_ids_match_whole_sequence": torch.equal(decode_ids, whole_ids),
            "rows": len(ids),
            "lookup_matches_range_bytes": torch.equal(rows(ids), fresh),
            "local_shard_rows_checked": len(local),
            "range_matches_local_shards": all(local) and bool(local),
        }
        rows.requested.clear()
    return results


def negative_control(model, tables: dict[int, NGramRows], tokens: list[int], records) -> dict:
    embeddings = ngram_embeddings(model)
    for layer, rows in tables.items():
        embeddings[layer].ngram_embedding = NGramRows(rows.table, rows.dtype, offset=1)
    shifted = trace(model, tokens, probes=(residual_probe,))
    for layer, rows in tables.items():
        embeddings[layer].ngram_embedding = rows
    results = {}
    for layer in tables:
        for name in (f"residual{layer}", f"residual{layer + 1}"):
            results[f"{name}_max_abs"] = max(
                (shifted[f"step{step}/{name}"] - records[f"step{step}/{name}"]).abs().max().item()
                for step in range(len(tokens))
            )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--tokens-json", type=Path, default=FIXTURE)
    parser.add_argument("--check", action="store_true", help="check the n-gram row lookup and run the off-by-one row control")
    args = parser.parse_args()

    fixture = json.loads(args.tokens_json.read_text())
    tokens = fixture["tokens"]
    model, tables = load_prefix(args.model, args.layers)
    records = trace(model, tokens)
    metadata = {
        "model": MODEL_ID,
        "revision": REVISION,
        "layers": str(args.layers),
        "tokens": json.dumps(tokens),
        "tokens_source": fixture["source"],
        "ngram_rows": json.dumps({layer: len(rows.rows) for layer, rows in tables.items()}),
    }
    if args.check:
        checks = {"ngram": check_ngram(model, tables, tokens, args.model)}
        checks["off_by_one_row"] = negative_control(model, tables, tokens, records)
        print(json.dumps(checks, indent=2), flush=True)
        metadata["checks"] = json.dumps(checks)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(records, args.out, metadata=metadata)
    print(f"tokens={tokens} records={len(records)} -> {args.out}")


if __name__ == "__main__":
    main()
