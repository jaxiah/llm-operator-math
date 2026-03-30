"""02 — Softmax: softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))

用纯 NumPy 实现数值稳定的 softmax，并用合成数据验证正确性。
"""

import numpy as np

from e2e.validate import validate


# ---------------------------------------------------------------------------
# 核心算子
# ---------------------------------------------------------------------------
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """数值稳定的 softmax 实现。"""
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# 验证
# ---------------------------------------------------------------------------
def validate_basic_properties() -> bool:
    """验证 softmax 的基本性质：输出 > 0，总和 = 1。"""
    print("\n=== 基本性质测试 ===")
    rng = np.random.default_rng(42)
    x = rng.standard_normal((4, 8)).astype(np.float32)
    y = softmax(x, axis=-1)

    all_positive = np.all(y > 0)
    sums_to_one = np.allclose(np.sum(y, axis=-1), 1.0, atol=1e-6)

    passed = all_positive and sums_to_one
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] softmax_basic_properties")
    print(f"       all_positive={all_positive}  sums_to_one={sums_to_one}")
    return passed


def validate_against_manual() -> bool:
    """与手动逐行计算的 softmax 对比验证。"""
    print("\n=== 与手动实现对比 ===")
    rng = np.random.default_rng(123)
    x = rng.standard_normal((16, 64)).astype(np.float32)

    actual = softmax(x, axis=-1)

    # 手动逐行计算作为参考实现
    expected = np.empty_like(x)
    for i in range(x.shape[0]):
        row = x[i]
        e = np.exp(row - np.max(row))
        expected[i] = e / np.sum(e)

    return validate("softmax_vs_manual", actual, expected, atol=1e-6, rtol=1e-6)


def validate_numerical_stability() -> bool:
    """测试大数值输入下的数值稳定性。"""
    print("\n=== 数值稳定性测试 ===")
    x = np.array([[1000.0, 1001.0, 1002.0]], dtype=np.float32)
    y = softmax(x, axis=-1)

    has_no_nan = not np.any(np.isnan(y))
    has_no_inf = not np.any(np.isinf(y))
    sums_to_one = np.allclose(np.sum(y, axis=-1), 1.0, atol=1e-6)

    passed = has_no_nan and has_no_inf and sums_to_one
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] softmax_numerical_stability")
    print(f"       no_nan={has_no_nan}  no_inf={has_no_inf}  sums_to_one={sums_to_one}")
    print(f"       input_range=[1000, 1002]  output={y}")
    return passed


def validate_multidim() -> bool:
    """测试多维输入，沿不同轴的 softmax。"""
    print("\n=== 多维张量测试 (batch, seq, dim) ===")
    rng = np.random.default_rng(456)
    x = rng.standard_normal((2, 32, 64)).astype(np.float32)

    actual = softmax(x, axis=-1)

    # 展平后逐行计算参考值
    flat = x.reshape(-1, 64)
    ref = np.empty_like(flat)
    for i in range(flat.shape[0]):
        e = np.exp(flat[i] - np.max(flat[i]))
        ref[i] = e / np.sum(e)
    expected = ref.reshape(x.shape)

    return validate("softmax_3d_axis_last", actual, expected, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    results = [
        validate_basic_properties(),
        validate_against_manual(),
        validate_numerical_stability(),
        validate_multidim(),
    ]
    print(f"\n{'='*60}")
    print(f"Softmax 验证: {sum(results)}/{len(results)} 通过")
    if not all(results):
        raise SystemExit(1)
