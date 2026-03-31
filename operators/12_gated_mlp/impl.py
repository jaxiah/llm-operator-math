"""Gated MLP (SwiGLU) —— 纯 NumPy 实现与验证

GatedMLP(x) = (SiLU(x @ Wg.T) ⊙ (x @ Wu.T)) @ Wd.T

用于 Qwen2-VL 文本解码器的每个 Transformer 层。
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


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU (Swish) 激活函数: x * σ(x)"""
    return x * sigmoid(x)


def gated_mlp(
    x: np.ndarray,
    gate_weight: np.ndarray,
    up_weight: np.ndarray,
    down_weight: np.ndarray,
) -> np.ndarray:
    """Gated MLP (SwiGLU): gate/up 并行投影 → 门控 → 下投影

    Args:
        x: 输入张量 (..., d)
        gate_weight: 门控投影权重 (d_ff, d)
        up_weight: 上投影权重 (d_ff, d)
        down_weight: 下投影权重 (d, d_ff)

    Returns:
        输出张量 (..., d)
    """
    gate = silu(x @ gate_weight.T)  # (..., d_ff)
    up = x @ up_weight.T  # (..., d_ff)
    hidden = gate * up  # (..., d_ff) 逐元素门控
    output = hidden @ down_weight.T  # (..., d)
    return output


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
    x = load_activation(dump_dir, "model__language_model__layers__0__mlp_input")
    expected = load_activation(dump_dir, "model__language_model__layers__0__mlp_output")

    print(f"输入形状: {x.shape}")  # (1, 3602, 1536)
    print(f"期望输出形状: {expected.shape}")  # (1, 3602, 1536)
    print()

    # 加载权重
    weight_keys = [
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    ]
    weights = load_weights(weight_keys)

    gate_w = weights["model.layers.0.mlp.gate_proj.weight"]  # (8960, 1536)
    up_w = weights["model.layers.0.mlp.up_proj.weight"]  # (8960, 1536)
    down_w = weights["model.layers.0.mlp.down_proj.weight"]  # (1536, 8960)

    print(f"gate_proj: {gate_w.shape}")
    print(f"up_proj:   {up_w.shape}")
    print(f"down_proj: {down_w.shape}")
    print()

    # 计算 Gated MLP
    actual = gated_mlp(x, gate_w, up_w, down_w)

    # 验证
    ok = validate("Gated MLP / SwiGLU (gate → silu → up → down)", actual, expected, atol=1e-4, rtol=1e-4)
    sys.exit(0 if ok else 1)
