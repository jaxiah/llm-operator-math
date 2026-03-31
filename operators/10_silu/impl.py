"""SiLU (Swish) 激活函数 —— 纯 NumPy 实现与验证

SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))

用于 Qwen2-VL 文本解码器的 Gated MLP 层。
"""

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


# ---------------------------------------------------------------------------
# 验证
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from e2e.validate import validate, load_activation

    dump_dir = "activations"

    # 加载文本解码器 MLP 激活函数的输入和输出
    x = load_activation(dump_dir, "model__language_model__layers__0__mlp__act_fn_input")
    expected = load_activation(dump_dir, "model__language_model__layers__0__mlp__act_fn_output")

    print(f"输入形状: {x.shape}")  # (1, 3602, 8960)
    print(f"期望输出形状: {expected.shape}")
    print()

    # 计算 SiLU
    actual = silu(x)

    # 验证
    ok = validate("SiLU (Text Decoder MLP act_fn)", actual, expected)
    sys.exit(0 if ok else 1)
