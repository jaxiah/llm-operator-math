"""07 — 注意力机制 (Attention) — 纯 NumPy 实现

实现并验证：
  1. Scaled Dot-Product Attention (SDPA)
  2. Vision Multi-Head Attention 的 QKV / 输出投影
  3. Text Grouped Query Attention (GQA) 的 Q/K/V/O 投影
  4. repeat_kv 操作（教学用）

用法:
    python -m operators.07_attention.impl
"""

import glob
import os
import sys

import numpy as np

from e2e.validate import load_activation, validate

# ---------------------------------------------------------------------------
# 核心算子
# ---------------------------------------------------------------------------


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """数值稳定的 softmax: 先减最大值防止 exp 溢出。"""
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """缩放点积注意力。

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k) + mask) @ V

    Args:
        q: (..., seq_q, d_k)
        k: (..., seq_k, d_k)
        v: (..., seq_k, d_v)
        mask: 加法掩码，被屏蔽位置为 -inf。形状可广播到 (..., seq_q, seq_k)。

    Returns:
        (..., seq_q, d_v)
    """
    d_k = q.shape[-1]
    scores = q @ k.swapaxes(-2, -1) / np.sqrt(d_k)
    if mask is not None:
        scores = scores + mask
    weights = softmax(scores, axis=-1)
    return weights @ v


def linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
    """线性变换: y = x @ weight.T + bias"""
    y = x @ weight.T
    if bias is not None:
        y = y + bias
    return y


def repeat_kv(x: np.ndarray, n_rep: int) -> np.ndarray:
    """将 KV 头复制以匹配 Q 头数（用于 GQA）。

    (B, n_kv_heads, S, D) → (B, n_kv_heads * n_rep, S, D)

    例如: n_kv_heads=2, n_rep=6 → 12 个头
    第 0 个 KV 头 → Q 头 0,1,2,3,4,5
    第 1 个 KV 头 → Q 头 6,7,8,9,10,11
    """
    if n_rep == 1:
        return x
    B, n_kv_heads, S, D = x.shape
    # 插入新维度: (B, n_kv_heads, 1, S, D)
    x = x[:, :, np.newaxis, :, :]
    # 广播到: (B, n_kv_heads, n_rep, S, D)
    x = np.broadcast_to(x, (B, n_kv_heads, n_rep, S, D))
    # reshape: (B, n_kv_heads * n_rep, S, D)
    return x.reshape(B, n_kv_heads * n_rep, S, D)


# ---------------------------------------------------------------------------
# 权重加载
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
DUMP_DIR = "activations"


def load_weights(keys: list[str]) -> dict[str, np.ndarray]:
    """从 HuggingFace Hub 缓存加载 safetensors 权重。"""
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
# 验证 1: SDPA 解析测试（手算值对比）
# ---------------------------------------------------------------------------


def test_sdpa_analytical() -> bool:
    """用手算的小例子验证 SDPA 实现。"""
    print("\n=== 测试 1: SDPA 解析值验证 ===")

    q = np.array([[[1, 0], [0, 1], [1, 1]]], dtype=np.float32)  # (1, 3, 2)
    k = np.array([[[1, 0], [0, 1], [0.5, 0.5]]], dtype=np.float32)
    v = np.array([[[10, 0], [0, 10], [5, 5]]], dtype=np.float32)

    actual = scaled_dot_product_attention(q, k, v)

    # 手算 scores = QK^T / sqrt(2)
    d_k = 2.0
    scores = q @ k.swapaxes(-2, -1) / np.sqrt(d_k)
    weights = softmax(scores, axis=-1)
    expected = weights @ v

    return validate("sdpa_analytical", actual, expected, atol=1e-6, rtol=1e-6)


def test_sdpa_with_causal_mask() -> bool:
    """验证因果掩码：每个位置只能看到自己和之前的位置。"""
    print("\n=== 测试 2: SDPA 因果掩码 ===")

    np.random.seed(42)
    seq_len = 4
    d_k = 8
    q = np.random.randn(1, seq_len, d_k).astype(np.float32)
    k = np.random.randn(1, seq_len, d_k).astype(np.float32)
    v = np.random.randn(1, seq_len, d_k).astype(np.float32)

    # 因果掩码: 上三角为 -inf
    mask = np.triu(np.full((seq_len, seq_len), -np.inf, dtype=np.float32), k=1)

    output = scaled_dot_product_attention(q, k, v, mask=mask)

    # 第一个位置的输出应该只由第一个 V 决定（只能看到位置 0）
    expected_pos0 = v[0, 0:1, :]  # softmax([score, -inf, -inf, -inf]) = [1, 0, 0, 0]
    actual_pos0 = output[0, 0:1, :]

    return validate("sdpa_causal_mask_pos0", actual_pos0, expected_pos0, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# 验证 2: 视觉注意力 — QKV 和输出投影（线性层）
# ---------------------------------------------------------------------------


def test_vision_qkv_projection() -> bool:
    """验证视觉编码器 Block 0 的融合 QKV 线性投影。"""
    print("\n=== 测试 3: Vision QKV 线性投影 ===")

    x = load_activation(DUMP_DIR, "model__visual__blocks__0__attn__qkv_input")
    expected = load_activation(DUMP_DIR, "model__visual__blocks__0__attn__qkv_output")
    print(f"  输入: {x.shape}  期望输出: {expected.shape}")

    weights = load_weights(
        [
            "visual.blocks.0.attn.qkv.weight",
            "visual.blocks.0.attn.qkv.bias",
        ]
    )
    W = weights["visual.blocks.0.attn.qkv.weight"]
    b = weights["visual.blocks.0.attn.qkv.bias"]
    print(f"  权重: {W.shape}  偏置: {b.shape}")

    actual = linear(x, W, b)

    return validate("vision_qkv_projection", actual, expected, atol=1e-4, rtol=1e-4)


def test_vision_output_projection() -> bool:
    """验证视觉编码器 Block 0 的输出投影。"""
    print("\n=== 测试 4: Vision 输出投影 ===")

    x = load_activation(DUMP_DIR, "model__visual__blocks__0__attn__proj_input")
    expected = load_activation(DUMP_DIR, "model__visual__blocks__0__attn__proj_output")
    print(f"  输入: {x.shape}  期望输出: {expected.shape}")

    weights = load_weights(
        [
            "visual.blocks.0.attn.proj.weight",
            "visual.blocks.0.attn.proj.bias",
        ]
    )
    W = weights["visual.blocks.0.attn.proj.weight"]
    b = weights["visual.blocks.0.attn.proj.bias"]

    actual = linear(x, W, b)

    return validate("vision_output_projection", actual, expected, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# 验证 3: 文本 GQA — Q/K/V/O 各投影
# ---------------------------------------------------------------------------


def test_text_q_projection() -> bool:
    """验证文本解码器 Layer 0 的 Q 投影。"""
    print("\n=== 测试 5: Text Q 投影 ===")

    x = load_activation(DUMP_DIR, "model__language_model__layers__0__self_attn__q_proj_input")
    expected = load_activation(DUMP_DIR, "model__language_model__layers__0__self_attn__q_proj_output")
    print(f"  输入: {x.shape}  期望输出: {expected.shape}")

    weights = load_weights(
        [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.q_proj.bias",
        ]
    )
    W = weights["model.layers.0.self_attn.q_proj.weight"]
    b = weights["model.layers.0.self_attn.q_proj.bias"]

    actual = linear(x, W, b)

    return validate("text_q_projection", actual, expected, atol=1e-4, rtol=1e-4)


def test_text_k_projection() -> bool:
    """验证文本解码器 Layer 0 的 K 投影。"""
    print("\n=== 测试 6: Text K 投影 ===")

    x = load_activation(DUMP_DIR, "model__language_model__layers__0__self_attn__k_proj_input")
    expected = load_activation(DUMP_DIR, "model__language_model__layers__0__self_attn__k_proj_output")
    print(f"  输入: {x.shape}  期望输出: {expected.shape}")

    weights = load_weights(
        [
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.k_proj.bias",
        ]
    )
    W = weights["model.layers.0.self_attn.k_proj.weight"]
    b = weights["model.layers.0.self_attn.k_proj.bias"]

    actual = linear(x, W, b)

    return validate("text_k_projection", actual, expected, atol=1e-4, rtol=1e-4)


def test_text_v_projection() -> bool:
    """验证文本解码器 Layer 0 的 V 投影。"""
    print("\n=== 测试 7: Text V 投影 ===")

    x = load_activation(DUMP_DIR, "model__language_model__layers__0__self_attn__v_proj_input")
    expected = load_activation(DUMP_DIR, "model__language_model__layers__0__self_attn__v_proj_output")
    print(f"  输入: {x.shape}  期望输出: {expected.shape}")

    weights = load_weights(
        [
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.v_proj.bias",
        ]
    )
    W = weights["model.layers.0.self_attn.v_proj.weight"]
    b = weights["model.layers.0.self_attn.v_proj.bias"]

    actual = linear(x, W, b)

    return validate("text_v_projection", actual, expected, atol=1e-4, rtol=1e-4)


def test_text_o_projection() -> bool:
    """验证文本解码器 Layer 0 的 O（输出）投影。"""
    print("\n=== 测试 8: Text O 投影 ===")

    x = load_activation(DUMP_DIR, "model__language_model__layers__0__self_attn__o_proj_input")
    expected = load_activation(DUMP_DIR, "model__language_model__layers__0__self_attn__o_proj_output")
    print(f"  输入: {x.shape}  期望输出: {expected.shape}")

    weights = load_weights(
        [
            "model.layers.0.self_attn.o_proj.weight",
        ]
    )
    W = weights["model.layers.0.self_attn.o_proj.weight"]

    # O 投影无 bias
    actual = linear(x, W, bias=None)

    return validate("text_o_projection", actual, expected, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# 验证 4: repeat_kv 正确性
# ---------------------------------------------------------------------------


def test_repeat_kv() -> bool:
    """验证 repeat_kv 的输出形状和值正确。"""
    print("\n=== 测试 9: repeat_kv 正确性 ===")

    np.random.seed(7)
    B, n_kv_heads, S, D = 1, 2, 4, 128
    n_rep = 6  # 12 Q heads / 2 KV heads

    kv = np.random.randn(B, n_kv_heads, S, D).astype(np.float32)
    expanded = repeat_kv(kv, n_rep)

    # 检查形状
    expected_shape = np.array([B, n_kv_heads * n_rep, S, D], dtype=np.int64)
    actual_shape = np.array(list(expanded.shape), dtype=np.int64)
    ok_shape = validate("repeat_kv_shape", actual_shape, expected_shape)

    # 检查第 0 和第 5 个 Q 头应该等于第 0 个 KV 头
    ok_val0 = validate("repeat_kv_head0==head5", expanded[:, 5], kv[:, 0], atol=0, rtol=0)
    # 第 6 个 Q 头应该等于第 1 个 KV 头
    ok_val1 = validate("repeat_kv_head6==kv1", expanded[:, 6], kv[:, 1], atol=0, rtol=0)

    return ok_shape and ok_val0 and ok_val1


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("注意力机制 (Attention) — SDPA + Vision MHA + Text GQA 验证")
    print("=" * 60)

    results = [
        # SDPA 解析测试
        test_sdpa_analytical(),
        test_sdpa_with_causal_mask(),
        # 视觉注意力线性投影
        test_vision_qkv_projection(),
        test_vision_output_projection(),
        # 文本 GQA 线性投影
        test_text_q_projection(),
        test_text_k_projection(),
        test_text_v_projection(),
        test_text_o_projection(),
        # repeat_kv
        test_repeat_kv(),
    ]

    print(f"\n{'=' * 60}")
    passed = sum(results)
    total = len(results)
    print(f"注意力机制验证: {passed}/{total} 通过")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
