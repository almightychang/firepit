import json
import struct
import sys
from pathlib import Path

import torch
from safetensors import safe_open


model_dir = Path(sys.argv[1])
index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]


def load(tensor_name):
    with safe_open(model_dir / index[tensor_name], framework="pt", device="cpu") as shard:
        return shard.get_tensor(tensor_name)


for name in sys.argv[2:]:
    weight = load(name)
    scale = load(name.replace(".weight", ".weight_scale_inv"))
    rows, cols = weight.shape
    block_n = rows // scale.shape[0]
    block_k = cols // scale.shape[1]
    locations = [(0, 0), (0, block_k - 1), (0, block_k),
                 (block_n - 1, block_k - 1), (block_n, 0),
                 (rows - 1, cols - 1)]
    for row, col in locations:
        value = weight[row, col].to(torch.float32)
        factor = scale[row // block_n, col // block_k].to(torch.float32)
        bits = struct.unpack("<I", struct.pack("<f", (value * factor).item()))[0]
        print(name, row, col, bits)
