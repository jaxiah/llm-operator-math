"""QuickGELU 激活函数 —— 纯 NumPy 实现与验证

QuickGELU(x) = x * sigmoid(1.702 * x)

用于 Qwen2-VL 视觉编码器的 MLP 层。
"""

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


# ---------------------------------------------------------------------------
# 验证
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from e2e.validate import validate, load_activation

    dump_dir = "activations"

    # 加载视觉 MLP 激活函数的输入和输出
    x = load_activation(dump_dir, "model__visual__blocks__0__mlp__act_input")
    expected = load_activation(dump_dir, "model__visual__blocks__0__mlp__act_output")

    print(f"输入形状: {x.shape}")  # (14308, 5120)
    print(f"期望输出形状: {expected.shape}")
    print()

    # 计算 QuickGELU
    actual = quick_gelu(x)

    # 验证
    ok = validate("QuickGELU (Vision MLP act)", actual, expected)
    sys.exit(0 if ok else 1)
