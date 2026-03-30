"""Vision MLP（视觉编码器前馈网络）—— 纯 NumPy 实现与验证

MLP(x) = FC2(QuickGELU(FC1(x)))
       = (QuickGELU(x @ W1.T + b1)) @ W2.T + b2

用于 Qwen2-VL 视觉编码器的每个 Transformer block。
"""

import glob
import os
import numpy as np


# ---------------------------------------------------------------------------
# 核心实现
# ---------------------------------------------------------------------------


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid 函数: σ(x) = 1 / (1 + exp(-x))"""
    return 1.0 / (1.0 + np.exp(-x))


def quick_gelu(x: np.ndarray) -> np.ndarray:
    """QuickGELU 激活函数: x * σ(1.702x)"""
    return x * sigmoid(1.702 * x)


def vision_mlp(
    x: np.ndarray,
    fc1_weight: np.ndarray,
    fc1_bias: np.ndarray,
    fc2_weight: np.ndarray,
    fc2_bias: np.ndarray,
) -> np.ndarray:
    """Vision MLP: FC1 → QuickGELU → FC2

    Args:
        x: 输入张量 (n, d)
        fc1_weight: 第一层权重 (d_ff, d)
        fc1_bias: 第一层偏置 (d_ff,)
        fc2_weight: 第二层权重 (d, d_ff)
        fc2_bias: 第二层偏置 (d,)

    Returns:
        输出张量 (n, d)
    """
    h = x @ fc1_weight.T + fc1_bias   # (n, d_ff)
    h = quick_gelu(h)                  # (n, d_ff)
    y = h @ fc2_weight.T + fc2_bias    # (n, d)
    return y


# ---------------------------------------------------------------------------
# 权重加载
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"


def load_weights(keys: list[str]) -> dict[str, np.ndarray]:
    """从 HuggingFace 缓存加载指定的 safetensors 权重张量。"""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    first_shard = hf_hub_download(MODEL_ID, "model-00001-of-00002.safetensors")
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

if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from e2e.validate import validate, load_activation

    dump_dir = "activations"

    # 加载激活值
    x = load_activation(dump_dir, "model__visual__blocks__0__mlp_input")
    expected = load_activation(dump_dir, "model__visual__blocks__0__mlp_output")

    print(f"输入形状: {x.shape}")       # (14308, 1280)
    print(f"期望输出形状: {expected.shape}")  # (14308, 1280)
    print()

    # 加载权重
    weight_keys = [
        "visual.blocks.0.mlp.fc1.weight",
        "visual.blocks.0.mlp.fc1.bias",
        "visual.blocks.0.mlp.fc2.weight",
        "visual.blocks.0.mlp.fc2.bias",
    ]
    weights = load_weights(weight_keys)

    fc1_w = weights["visual.blocks.0.mlp.fc1.weight"]  # (5120, 1280)
    fc1_b = weights["visual.blocks.0.mlp.fc1.bias"]    # (5120,)
    fc2_w = weights["visual.blocks.0.mlp.fc2.weight"]  # (1280, 5120)
    fc2_b = weights["visual.blocks.0.mlp.fc2.bias"]    # (1280,)

    print(f"fc1_weight: {fc1_w.shape}, fc1_bias: {fc1_b.shape}")
    print(f"fc2_weight: {fc2_w.shape}, fc2_bias: {fc2_b.shape}")
    print()

    # 计算 Vision MLP
    actual = vision_mlp(x, fc1_w, fc1_b, fc2_w, fc2_b)

    # 验证
    ok = validate("Vision MLP (FC1 → QuickGELU → FC2)", actual, expected,
                  atol=1e-4, rtol=1e-4)
    sys.exit(0 if ok else 1)
