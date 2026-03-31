"""GELU 激活函数 —— 纯 NumPy 实现与验证

精确形式: GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
tanh 近似: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))

用于 Qwen2-VL Patch Merger 的 MLP 层。
PyTorch 的 nn.GELU() 默认使用精确形式 (erf)。
"""

import math
import numpy as np

# ---------------------------------------------------------------------------
# 核心实现
# ---------------------------------------------------------------------------

_SQRT_2 = math.sqrt(2.0)
_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


def gelu_exact(x: np.ndarray) -> np.ndarray:
    """GELU 精确形式: x * Φ(x) = x * 0.5 * (1 + erf(x / √2))

    在 float64 下计算 erf 以获得高精度，将 Φ(x) 转回 float32 后再乘以 x，
    以匹配 PyTorch nn.GELU() 的行为。
    """
    x64 = x.astype(np.float64)
    phi = (0.5 * (1.0 + _erf(x64 / _SQRT_2))).astype(np.float32)
    return x * phi


def gelu_tanh(x: np.ndarray) -> np.ndarray:
    """GELU tanh 近似形式"""
    return (0.5 * x * (1.0 + np.tanh(_SQRT_2_OVER_PI * (x + 0.044715 * x**3)))).astype(np.float32)


def _erf(x: np.ndarray) -> np.ndarray:
    """误差函数实现。

    优先使用 scipy.special.erf（向量化 C 实现，最快）。
    回退到 math.erf + np.vectorize（精度相同，但较慢）。
    """
    try:
        from scipy.special import erf

        return erf(x)
    except ImportError:
        import math

        return np.vectorize(math.erf)(x)


# ---------------------------------------------------------------------------
# 验证
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from e2e.validate import validate, load_activation

    dump_dir = "activations"

    # 加载 Patch Merger MLP 的 GELU 输入和输出
    x = load_activation(dump_dir, "model__visual__merger__mlp__1_input")
    expected = load_activation(dump_dir, "model__visual__merger__mlp__1_output")

    print(f"输入形状: {x.shape}")  # (3577, 5120)
    print(f"期望输出形状: {expected.shape}")
    print()

    # 精确形式 (PyTorch nn.GELU 默认使用 erf 版本)
    actual_exact = gelu_exact(x)
    ok = validate("GELU exact (Patch Merger)", actual_exact, expected)

    if not ok:
        # 如果精确形式不匹配，尝试 tanh 近似
        print("\n精确形式未通过，尝试 tanh 近似...")
        actual_tanh = gelu_tanh(x)
        ok = validate("GELU tanh approx (Patch Merger)", actual_tanh, expected)

    sys.exit(0 if ok else 1)
