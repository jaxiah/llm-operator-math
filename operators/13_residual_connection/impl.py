"""13 — 残差连接 (Residual Connection): y = x + F(x)

用纯 NumPy 实现残差连接，并用 Qwen2-VL Vision Block 0 的真实激活值验证。
"""

import numpy as np

from e2e.validate import validate, load_activation

DUMP_DIR = "activations"


# ---------------------------------------------------------------------------
# 核心算子
# ---------------------------------------------------------------------------
def residual_add(x: np.ndarray, f_x: np.ndarray) -> np.ndarray:
    """残差连接: y = x + F(x)"""
    return x + f_x


# ---------------------------------------------------------------------------
# 验证
# ---------------------------------------------------------------------------
def validate_vision_block0_attn_residual() -> bool:
    """验证 Vision Block 0 的注意力残差连接。

    在 pre-norm Transformer 中：
        norm1_output = LayerNorm(block_input)
        attn_output  = Attention(norm1_output)
        residual_1   = block_input + attn_output   ← 这就是 norm2_input
    """
    print("\n=== Vision Block 0: 注意力残差连接 ===")
    print("    block_input + attn_output == norm2_input")

    block_input = load_activation(DUMP_DIR, "model__visual__blocks__0_input")
    attn_output = load_activation(DUMP_DIR, "model__visual__blocks__0__attn_output")
    expected = load_activation(DUMP_DIR, "model__visual__blocks__0__norm2_input")

    actual = residual_add(block_input, attn_output)
    return validate("vision_block0_attn_residual", actual, expected, atol=1e-5, rtol=1e-5)


def validate_synthetic() -> bool:
    """用合成数据验证残差连接的基本正确性。"""
    print("\n=== 合成数据测试 ===")
    rng = np.random.default_rng(42)
    x = rng.standard_normal((2, 16, 64)).astype(np.float32)
    f_x = rng.standard_normal((2, 16, 64)).astype(np.float32) * 0.1

    actual = residual_add(x, f_x)
    expected = x + f_x
    return validate("residual_synthetic", actual, expected, atol=0.0, rtol=0.0)


if __name__ == "__main__":
    results = [
        validate_vision_block0_attn_residual(),
        validate_synthetic(),
    ]
    print(f"\n{'='*60}")
    print(f"残差连接验证: {sum(results)}/{len(results)} 通过")
    if not all(results):
        raise SystemExit(1)
