"""06 — 旋转位置编码 (Rotary Position Embedding, RoPE)

用纯 NumPy 实现 RoPE 的三种形式（1D / 2D 视觉 / 3D 多模态），
并通过解析性质（保范、相对位置依赖性）进行验证。

用法:
    python -m operators.06_rotary_pos_embed.impl
"""

import sys

import numpy as np

from e2e.validate import validate

# ---------------------------------------------------------------------------
# 核心算子
# ---------------------------------------------------------------------------


def rotate_half(x: np.ndarray) -> np.ndarray:
    """将向量前后两半交换并取负，等价于复数乘以 i。

    [x0, x1, ..., x_{d/2-1}, x_{d/2}, ..., x_{d-1}]
    → [-x_{d/2}, ..., -x_{d-1}, x0, ..., x_{d/2-1}]
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return np.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(
    q: np.ndarray,
    k: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    unsqueeze_dim: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """对 Q 和 K 应用旋转位置编码。

    公式: x_rotated = x * cos + rotate_half(x) * sin

    Args:
        q: Query 张量，形状 (..., seq_len, num_heads, head_dim) 或类似。
        k: Key 张量，形状同 q。
        cos: 余弦缓存，形状 (..., seq_len, head_dim)。
        sin: 正弦缓存，形状同 cos。
        unsqueeze_dim: 在 cos/sin 中插入的维度，用于广播到 num_heads。

    Returns:
        (q_rotated, k_rotated) 旋转后的 Q 和 K。
    """
    cos = np.expand_dims(cos, axis=unsqueeze_dim)
    sin = np.expand_dims(sin, axis=unsqueeze_dim)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# 频率与角度计算
# ---------------------------------------------------------------------------


def compute_rope_frequencies(head_dim: int, max_position: int, base: float = 10000.0) -> tuple[np.ndarray, np.ndarray]:
    """计算 1D RoPE 的 cos/sin 缓存。

    Args:
        head_dim: 每个注意力头的维度。
        max_position: 最大位置数。
        base: 频率基数（Qwen2-VL 语言模型用 1e6）。

    Returns:
        (cos, sin) 形状均为 (max_position, head_dim)。
    """
    # 频率: base^(-2i/d)，用 float64 提高精度
    inv_freq = base ** (-np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)
    # (head_dim // 2,)

    positions = np.arange(max_position, dtype=np.float64)  # (max_position,)
    # 外积得到角度矩阵 (max_position, head_dim // 2)
    angles = np.outer(positions, inv_freq)
    # 复制一次凑满 head_dim
    angles = np.concatenate([angles, angles], axis=-1)  # (max_position, head_dim)

    cos = np.cos(angles).astype(np.float32)
    sin = np.sin(angles).astype(np.float32)
    return cos, sin


def compute_vision_rope(grid_thw: np.ndarray, head_dim: int, base: float = 10000.0) -> tuple[np.ndarray, np.ndarray]:
    """计算 2D 视觉 RoPE 的 cos/sin。

    将 head_dim 对半分：前半用 height 坐标，后半用 width 坐标。

    Args:
        grid_thw: (num_videos, 3) 数组，每行为 [temporal, height, width]。
        head_dim: 视觉模型的 head_dim（如 80）。
        base: 频率基数。

    Returns:
        (cos, sin) 形状为 (total_patches, head_dim)。
    """
    half_dim = head_dim // 2
    inv_freq = base ** (-np.arange(0, half_dim, 2, dtype=np.float64) / half_dim)
    # (half_dim // 2,)

    all_cos, all_sin = [], []
    for thw in grid_thw:
        t, h, w = int(thw[0]), int(thw[1]), int(thw[2])
        # 为每个 patch 生成 (h_pos, w_pos)
        h_pos = np.repeat(np.arange(h, dtype=np.float64), w)  # (h*w,)
        w_pos = np.tile(np.arange(w, dtype=np.float64), h)  # (h*w,)
        # 每帧的 patch 重复 t 次
        h_pos = np.tile(h_pos, t)  # (t*h*w,)
        w_pos = np.tile(w_pos, t)  # (t*h*w,)

        # 角度
        angles_h = np.outer(h_pos, inv_freq)  # (N, half_dim//2)
        angles_w = np.outer(w_pos, inv_freq)  # (N, half_dim//2)

        # 拼接 [h_freqs, w_freqs] 再复制，使 rotate_half 的配对正确
        # rotate_half 将 dim i 与 dim i+head_dim/2 配对
        angles_hw = np.concatenate([angles_h, angles_w], axis=-1)  # (N, head_dim//2)
        angles = np.concatenate([angles_hw, angles_hw], axis=-1)  # (N, head_dim)

        all_cos.append(np.cos(angles).astype(np.float32))
        all_sin.append(np.sin(angles).astype(np.float32))

    cos = np.concatenate(all_cos, axis=0)
    sin = np.concatenate(all_sin, axis=0)
    return cos, sin


def compute_mrope(
    position_ids: np.ndarray,
    head_dim: int,
    mrope_section: list[int],
    base: float = 1_000_000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """计算 3D 多模态 RoPE (M-RoPE) 的 cos/sin。

    Args:
        position_ids: (3, seq_len) 三维位置 ID（temporal, height, width）。
        head_dim: 每个注意力头的维度（如 128）。
        mrope_section: 三个维度各自使用的频率数量，如 [16, 24, 24]。
        base: 频率基数（Qwen2-VL 用 1e6）。

    Returns:
        (cos, sin) 形状均为 (seq_len, head_dim)。
    """
    inv_freq = base ** (-np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)
    # (head_dim // 2,) = (64,) for head_dim=128

    # 按 mrope_section 拆分频率
    sec_cumsum = np.cumsum([0] + mrope_section)  # [0, 16, 40, 64]
    angle_parts = []
    for dim_idx in range(3):
        start, end = int(sec_cumsum[dim_idx]), int(sec_cumsum[dim_idx + 1])
        freq_slice = inv_freq[start:end]  # (section_size,)
        pos = position_ids[dim_idx].astype(np.float64)  # (seq_len,)
        angles = np.outer(pos, freq_slice)  # (seq_len, section_size)
        angle_parts.append(angles)

    # 拼接三个维度的角度 (seq_len, 64)，再复制凑成 head_dim
    angles = np.concatenate(angle_parts, axis=-1)  # (seq_len, head_dim//2)
    angles = np.concatenate([angles, angles], axis=-1)  # (seq_len, head_dim)

    cos = np.cos(angles).astype(np.float32)
    sin = np.sin(angles).astype(np.float32)
    return cos, sin


# ---------------------------------------------------------------------------
# 验证：解析性质测试
# ---------------------------------------------------------------------------


def test_rotation_preserves_norm() -> bool:
    """测试 1: 旋转不改变向量范数。"""
    print("\n=== 测试 1: 旋转保持向量范数 ===")
    np.random.seed(42)
    head_dim = 128
    seq_len = 32

    cos, sin = compute_rope_frequencies(head_dim, seq_len, base=1_000_000.0)

    q = np.random.randn(1, seq_len, 4, head_dim).astype(np.float32)
    k = np.random.randn(1, seq_len, 4, head_dim).astype(np.float32)

    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

    q_norms_before = np.linalg.norm(q, axis=-1)
    q_norms_after = np.linalg.norm(q_rot, axis=-1)

    return validate(
        "rope_preserves_q_norm",
        q_norms_after,
        q_norms_before,
        atol=1e-4,
        rtol=1e-4,
    )


def test_relative_position_property() -> bool:
    """测试 2: 注意力分数只依赖相对位置。

    对于相同的 q 和 k 向量，在不同绝对位置但相同相对位置下，
    点积应该相同。
    """
    print("\n=== 测试 2: 相对位置编码性质 ===")
    np.random.seed(123)
    head_dim = 64
    max_pos = 100

    cos, sin = compute_rope_frequencies(head_dim, max_pos, base=10000.0)

    q_vec = np.random.randn(head_dim).astype(np.float32)
    k_vec = np.random.randn(head_dim).astype(np.float32)

    # 场景 A: q 在位置 5, k 在位置 8 (相对距离 = 3)
    # 场景 B: q 在位置 20, k 在位置 23 (相对距离 = 3)
    dots = []
    for q_pos, k_pos in [(5, 8), (20, 23), (50, 53)]:
        q_r = q_vec * cos[q_pos] + rotate_half(q_vec) * sin[q_pos]
        k_r = k_vec * cos[k_pos] + rotate_half(k_vec) * sin[k_pos]
        dots.append(np.dot(q_r, k_r))

    # 三种情况的点积应该相同
    expected = np.array([dots[0], dots[0], dots[0]], dtype=np.float32)
    actual = np.array(dots, dtype=np.float32)

    return validate(
        "rope_relative_position_invariance",
        actual,
        expected,
        atol=1e-3,
        rtol=1e-3,
    )


def test_1d_rope_analytical() -> bool:
    """测试 3: 1D RoPE 与手算值对比。"""
    print("\n=== 测试 3: 1D RoPE 解析值验证 ===")
    head_dim = 4
    base = 10000.0
    position = 2

    x = np.array([[[[1.0, 2.0, 3.0, 4.0]]]], dtype=np.float32)
    # shape: (1, 1, 1, 4) = (batch, seq, heads, head_dim)

    cos, sin = compute_rope_frequencies(head_dim, max_position=3, base=base)
    # cos/sin shape: (3, 4), 取位置 2
    cos_p = cos[position : position + 1]  # (1, 4)
    sin_p = sin[position : position + 1]  # (1, 4)

    # unsqueeze_dim=1 (default) inserts dim at axis=1 for heads broadcasting
    q_rot, _ = apply_rotary_pos_emb(x, x, cos_p, sin_p, unsqueeze_dim=1)
    # q_rot shape: (1, 1, 1, 4)

    # 手算期望值
    inv_freq = base ** (-np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)
    # [1.0, 0.01]
    a0, a1 = position * inv_freq[0], position * inv_freq[1]
    # a0=2.0, a1=0.02

    expected = np.array(
        [
            [
                [
                    [
                        1.0 * np.cos(a0) - 3.0 * np.sin(a0),  # y0
                        2.0 * np.cos(a1) - 4.0 * np.sin(a1),  # y1
                        3.0 * np.cos(a0) + 1.0 * np.sin(a0),  # y2
                        4.0 * np.cos(a1) + 2.0 * np.sin(a1),  # y3
                    ]
                ]
            ]
        ],
        dtype=np.float32,
    )

    return validate(
        "rope_1d_analytical",
        q_rot,
        expected,
        atol=1e-5,
        rtol=1e-5,
    )


def test_mrope_text_only() -> bool:
    """测试 4: 纯文本 M-RoPE 等价于 1D RoPE（三个维度位置相同时）。"""
    print("\n=== 测试 4: 纯文本 M-RoPE 一致性 ===")
    head_dim = 128
    seq_len = 16
    mrope_section = [16, 24, 24]
    base = 1_000_000.0

    # 纯文本：三个维度位置相同
    text_positions = np.arange(seq_len)
    position_ids = np.stack([text_positions, text_positions, text_positions])
    # (3, seq_len)

    cos_mrope, sin_mrope = compute_mrope(position_ids, head_dim, mrope_section, base)

    # 等价 1D RoPE
    cos_1d, sin_1d = compute_rope_frequencies(head_dim, seq_len, base)

    return validate(
        "mrope_text_matches_1d",
        cos_mrope,
        cos_1d,
        atol=1e-6,
        rtol=1e-6,
    )


def test_vision_rope_shape() -> bool:
    """测试 5: 2D 视觉 RoPE 输出形状正确。"""
    print("\n=== 测试 5: 2D 视觉 RoPE 形状验证 ===")
    head_dim = 80
    grid_thw = np.array([[1, 24, 24]])  # 1 帧, 24x24 patch

    cos, sin = compute_vision_rope(grid_thw, head_dim, base=10000.0)

    expected_shape = np.array([1 * 24 * 24, head_dim], dtype=np.int64)
    actual_shape = np.array(list(cos.shape), dtype=np.int64)

    ok = validate("vision_rope_shape", actual_shape, expected_shape)
    if ok:
        # 额外检查范数保持
        np.random.seed(7)
        x = np.random.randn(576, head_dim).astype(np.float32)
        x_rot = x * cos + rotate_half(x) * sin
        ok2 = validate(
            "vision_rope_preserves_norm",
            np.linalg.norm(x_rot, axis=-1),
            np.linalg.norm(x, axis=-1),
            atol=1e-4,
            rtol=1e-4,
        )
        return ok and ok2
    return ok


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("Rotary Position Embedding (RoPE) — 解析性质验证")
    print("=" * 60)

    results = [
        test_rotation_preserves_norm(),
        test_relative_position_property(),
        test_1d_rope_analytical(),
        test_mrope_text_only(),
        test_vision_rope_shape(),
    ]

    print(f"\n{'=' * 60}")
    print(f"RoPE 验证: {sum(results)}/{len(results)} 通过")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
