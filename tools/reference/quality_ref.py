import argparse
import json
import os
import urllib.request
import zipfile
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer

from trace_qwen36 import load_engine, use_torch_reference_kernels

WIKITEXT_URL = "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"
WIKITEXT_MEMBER = "wikitext-2-raw/wiki.test.raw"


def wikitext(cache: Path) -> str:
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        archive = cache.with_suffix(".zip")
        if not archive.exists():
            partial = archive.with_suffix(".zip.part")
            urllib.request.urlretrieve(WIKITEXT_URL, partial)
            os.replace(partial, archive)
        with zipfile.ZipFile(archive) as source:
            text = source.read(WIKITEXT_MEMBER)
        partial = cache.with_suffix(cache.suffix + ".part")
        partial.write_bytes(text)
        os.replace(partial, cache)
    return cache.read_text()


def chunk_file(out: Path, index: int) -> Path:
    return out / f"chunk-{index:06d}.safetensors"


def check_existing(path: Path, metadata: dict[str, str], tokens: list[int]) -> bool:
    with safe_open(path, framework="pt") as source:
        if source.metadata() != metadata or source.get_tensor("tokens").tolist() != tokens:
            raise SystemExit(f"existing chunk differs from requested run: {path}")
        scored = len(tokens) - int(metadata["score_start"])
        shapes = {
            "tokens": (len(tokens),),
            "top_ids": (scored, int(metadata["top_k"])),
            "top_logp": (scored, int(metadata["top_k"])),
            "tail_log_mass": (scored,),
        }
        dtypes = {"tokens": torch.int32, "top_ids": torch.int32, "top_logp": torch.float32, "tail_log_mass": torch.float32}
        for name, shape in shapes.items():
            value = source.get_tensor(name)
            if tuple(value.shape) != shape or value.dtype != dtypes[name]:
                raise SystemExit(f"existing {name} format differs from requested run: {path}")
        if "target_logp" in source.keys():
            values = source.get_tensor("target_logp")
            if values.shape != (scored - 1,) or values.dtype != torch.float32 or not torch.isfinite(values).all() or (values > 0).any():
                raise SystemExit(f"invalid target_logp in {path}")
            return True
        return False


def target_logp(logits: torch.Tensor, normalizer: torch.Tensor, tokens: list[int], start: int, score_start: int, device: str, out: torch.Tensor):
    end = min(start + logits.shape[0], len(tokens) - 1)
    if start < end:
        targets = torch.tensor(tokens[start + 1:end + 1], device=device)
        selected = logits[:end - start].gather(-1, targets[:, None]).squeeze(-1)
        out[start - score_start:end - score_start] = (selected - normalizer[:end - start]).float().cpu()


def record_target(model, norm, tokens: list[int], score_start: int, batch: int, device: str) -> torch.Tensor:
    target = torch.empty(len(tokens) - score_start - 1, dtype=torch.float32)
    with torch.inference_mode():
        hidden = model.model(input_ids=torch.tensor([tokens], device=device), use_cache=False).last_hidden_state
        for start in range(score_start, len(tokens) - 1, batch):
            end = min(start + batch, len(tokens) - 1)
            logits = model.lm_head(norm(hidden[:, start:end])).squeeze(0).double()
            target_logp(logits, torch.logsumexp(logits, dim=-1), tokens, start, score_start, device, target)
    return target


def record_chunk(model, norm, tokens: list[int], score_start: int, k: int, batch: int, device: str):
    scored = len(tokens) - score_start
    ids = torch.empty((scored, k), dtype=torch.int32)
    logp = torch.empty((scored, k), dtype=torch.float32)
    tail = torch.empty(scored, dtype=torch.float32)
    target_scores = torch.empty(scored - 1, dtype=torch.float32)
    with torch.inference_mode():
        hidden = model.model(input_ids=torch.tensor([tokens], device=device), use_cache=False).last_hidden_state
        for start in range(score_start, len(tokens), batch):
            end = min(start + batch, len(tokens))
            logits = model.lm_head(norm(hidden[:, start:end])).squeeze(0).double()
            normalizer = torch.logsumexp(logits, dim=-1)
            selected = torch.topk(logits, k, dim=-1)
            remainder = logits.scatter(-1, selected.indices, -torch.inf)
            rows = slice(start - score_start, end - score_start)
            ids[rows] = selected.indices.to(torch.int32).cpu()
            logp[rows] = (selected.values - normalizer[:, None]).float().cpu()
            tail[rows] = (torch.logsumexp(remainder, dim=-1) - normalizer).float().cpu()
            target_logp(logits, normalizer, tokens, start, score_start, device, target_scores)
    return {
        "tokens": torch.tensor(tokens, dtype=torch.int32),
        "top_ids": ids,
        "top_logp": logp,
        "tail_log_mass": tail,
        "target_logp": target_scores,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True, help="path to cached wiki.test.raw")
    parser.add_argument("--out", type=Path, required=True, help="directory for chunk safetensors")
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--score-start", type=int, default=2048)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--chunks", type=int)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--score-batch", type=int, default=32)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--add-target-logp", action="store_true", help="add target log probabilities to verified existing chunks")
    args = parser.parse_args()
    if args.chunk_size < 3 or not 0 < args.score_start < args.chunk_size - 1 or args.k <= 0 or args.score_batch <= 0:
        parser.error("invalid chunk, score, K, or score batch")
    if args.chunks is not None and args.chunks <= 0:
        parser.error("--chunks must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA PyTorch is required; use --device cpu for a small local run")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokens = tokenizer.backend_tokenizer.encode(wikitext(args.cache), add_special_tokens=False).ids
    available = len(tokens) // args.chunk_size
    count = available if args.chunks is None else args.chunks
    if count == 0:
        parser.error(f"only {len(tokens)} tokens; no complete chunk")
    if count > available:
        parser.error(f"requested {count} chunks but only {available} complete chunks exist")
    if args.k >= tokenizer.vocab_size:
        parser.error("K must be smaller than vocabulary")
    if args.layers is not None and args.layers <= 0:
        parser.error("--layers must be positive")
    layers = args.layers if args.layers is not None else json.loads((args.model / "config.json").read_text())["text_config"]["num_hidden_layers"]
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": "firepit-quality-v1",
        "model": str(args.model.resolve()),
        "layers": str(layers),
        "chunk_size": str(args.chunk_size),
        "score_start": str(args.score_start),
        "top_k": str(args.k),
        "source": WIKITEXT_URL,
    }
    use_torch_reference_kernels()
    model = norm = None
    for index in range(count):
        chunk_tokens = tokens[index * args.chunk_size:(index + 1) * args.chunk_size]
        path = chunk_file(args.out, index)
        if path.exists():
            has_target = check_existing(path, metadata, chunk_tokens)
            if not args.add_target_logp or has_target:
                print(f"verified existing {path}", flush=True)
                continue
        elif args.add_target_logp:
            raise SystemExit(f"missing chunk to augment: {path}")
        if model is None:
            model, norm = load_engine(args.model, layers, args.device)
        if args.add_target_logp:
            with safe_open(path, framework="pt") as source:
                data = {name: source.get_tensor(name) for name in source.keys()}
            data["target_logp"] = record_target(model, norm, chunk_tokens, args.score_start, args.score_batch, args.device)
        else:
            data = record_chunk(model, norm, chunk_tokens, args.score_start, args.k, args.score_batch, args.device)
        partial = path.with_suffix(".safetensors.part")
        save_file(data, partial, metadata=metadata)
        os.replace(partial, path)
        print(f"saved {path} target_rows={args.chunk_size - args.score_start - 1}", flush=True)


if __name__ == "__main__":
    main()
