import json
import struct
import sys
from pathlib import Path

import torch
from safetensors import safe_open


# ModelOpt NVFP4QTensor.dequantize (modelopt/torch/quantization/qtensor/nvfp4_tensor.py):
# even elements in the low nibble, values from e2m1_values, scale = weight_scale.float() * weight_scale_2.
E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])

model_dir = Path(sys.argv[1])
index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]


def load(tensor_name):
    with safe_open(model_dir / index[tensor_name], framework="pt", device="cpu") as shard:
        return shard.get_tensor(tensor_name)


def bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def nvfp4(name):
    packed = load(name)
    scale = load(name.replace(".weight", ".weight_scale")).to(torch.float32)
    global_scale = load(name.replace(".weight", ".weight_scale_2"))
    rows, cols = packed.shape[0], packed.shape[1] * 2
    for row, col in [(0, 0), (0, 1), (0, 15), (0, 16), (1, 17), (rows - 1, cols - 1)]:
        byte = int(packed[row, col // 2])
        code = byte & 0x0F if col % 2 == 0 else byte >> 4
        block_scale = scale[row, col // 16] * global_scale
        print(name, row, col, bits((E2M1[code] * block_scale).item()))


def fp8_tensor(name):
    weight = load(name)
    scale = load(name.replace(".weight", ".weight_scale"))
    rows, cols = weight.shape
    for row, col in [(0, 0), (0, 1), (rows - 1, cols - 1)]:
        print(name, row, col, bits((weight[row, col].to(torch.float32) * scale).item()))


for name in sys.argv[2:]:
    if load(name).dtype == torch.uint8:
        nvfp4(name)
    else:
        fp8_tensor(name)
