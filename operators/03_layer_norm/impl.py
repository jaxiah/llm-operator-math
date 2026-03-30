"""Layer Normalization — 纯 numpy 实现，并用 Qwen2-VL vision encoder 的真实权重验证。

用法:
    python -m operators.03_layer_norm.impl
"""

import glob
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# 核心算子
# ---------------------------------------------------------------------------


def layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Layer Normalization: y = (x - mean) / sqrt(var + eps) * weight + bias

    对最后一个维度做归一化。

    Args:
        x: 输入张量，任意形状，归一化沿最后一维进行。
        weight: 缩放参数 (gamma)，形状与最后一维相同。
        bias: 偏移参数 (beta)，形状与最后一维相同。
        eps: 防止除零的小常数。

    Returns:
        归一化后的张量，形状与输入相同。
    """
    # 沿最后一维计算均值和方差
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)

    # 归一化 + 仿射变换
    x_norm = (x - mean) / np.sqrt(var + eps)
    return x_norm * weight + bias


# ---------------------------------------------------------------------------
# 权重加载
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
SHARD_FILES = [
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
]


def load_weights(keys: list[str]) -> dict[str, np.ndarray]:
    """从 HuggingFace Hub 缓存中加载 safetensors 权重。

    使用 safetensors.torch 加载（因为 numpy 后端不支持 bfloat16），
    然后转换为 float32 numpy 数组。
    """
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    # 下载第一个 shard 以定位模型缓存目录
    first_shard = hf_hub_download(MODEL_ID, SHARD_FILES[0])
    model_dir = os.path.dirname(first_shard)

    result: dict[str, np.ndarray] = {}
    for sf_file in glob.glob(os.path.join(model_dir, "*.safetensors")):
        tensors = load_file(sf_file)
        for k in keys:
            if k in tensors:
                result[k] = tensors[k].float().numpy()
        if len(result) == len(keys):
            break
    return result


# ---------------------------------------------------------------------------
# 验证
# ---------------------------------------------------------------------------

DUMP_DIR = os.path.join("activations")

# Vision encoder block 0 的 norm1
WEIGHT_KEY = "visual.blocks.0.norm1.weight"
BIAS_KEY = "visual.blocks.0.norm1.bias"
INPUT_NAME = "model__visual__blocks__0__norm1_input"
OUTPUT_NAME = "model__visual__blocks__0__norm1_output"


def main() -> None:
    from e2e.validate import load_activation, validate

    print("=" * 60)
    print("Layer Normalization — Vision Encoder block 0, norm1")
    print("=" * 60)

    # 1) 加载激活值
    x = load_activation(DUMP_DIR, INPUT_NAME)
    expected = load_activation(DUMP_DIR, OUTPUT_NAME)
    print(f"输入形状: {x.shape}  输出形状: {expected.shape}")

    # 2) 加载权重
    weights = load_weights([WEIGHT_KEY, BIAS_KEY])
    w = weights[WEIGHT_KEY]
    b = weights[BIAS_KEY]
    print(f"weight 形状: {w.shape}  bias 形状: {b.shape}")

    # 3) 计算 LayerNorm
    actual = layer_norm(x, w, b, eps=1e-6)

    # 4) 验证
    print()
    ok = validate("vision_block_0_norm1", actual, expected)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
