"""Patch Merger（补丁合并器）—— 纯 NumPy 实现与验证

PatchMerger(x) = MLP(SpatialMerge(LayerNorm(x)))

空间合并将 2×2 相邻 patch 拼接，MLP 将维度从 5120 映射到 1536。
用于 Qwen2-VL 视觉编码器输出到语言模型的桥接。
"""

import glob
import importlib
import os
import sys
import numpy as np

# ---------------------------------------------------------------------------
# 子算子导入（目录名以数字开头，需要 importlib）
# ---------------------------------------------------------------------------

_layer_norm_mod = importlib.import_module("operators.03_layer_norm.impl")
_gelu_mod = importlib.import_module("operators.09_gelu.impl")

layer_norm = _layer_norm_mod.layer_norm
gelu_exact = _gelu_mod.gelu_exact


def spatial_merge(
    x_normed: np.ndarray,
    spatial_merge_size: int = 2,
) -> np.ndarray:
    """将相邻 patch 拼接合并。

    在 Qwen2-VL 中，视觉编码器的窗口注意力已经将 patch 重排为
    2×2 空间邻居连续排列的顺序，因此合并操作只需简单的 reshape。

    Args:
        x_normed: 归一化后的 patch 特征 (total_patches, d)
        spatial_merge_size: 合并窗口大小，默认 2

    Returns:
        合并后的特征 (total_patches / s^2, d * s^2)
    """
    d = x_normed.shape[-1]
    s = spatial_merge_size
    return x_normed.reshape(-1, s * s * d)


def patch_merger(
    x: np.ndarray,
    ln_weight: np.ndarray,
    ln_bias: np.ndarray,
    fc1_weight: np.ndarray,
    fc1_bias: np.ndarray,
    fc2_weight: np.ndarray,
    fc2_bias: np.ndarray,
    spatial_merge_size: int = 2,
) -> np.ndarray:
    """Patch Merger: LayerNorm → 空间合并 → Linear → GELU → Linear

    Args:
        x: 输入 patch 特征 (total_patches, d)
        ln_weight, ln_bias: LayerNorm 参数
        fc1_weight, fc1_bias: 第一层 Linear 参数
        fc2_weight, fc2_bias: 第二层 Linear 参数
        spatial_merge_size: 空间合并窗口大小

    Returns:
        合并后的特征 (total_merged_patches, d_out)
    """
    # 1. LayerNorm
    x_normed = layer_norm(x, ln_weight, ln_bias)

    # 2. 空间合并: (n, d) → (n/4, 4d)
    x_merged = spatial_merge(x_normed, spatial_merge_size)

    # 3. MLP: Linear → GELU → Linear
    h = x_merged @ fc1_weight.T + fc1_bias
    h = gelu_exact(h)
    y = h @ fc2_weight.T + fc2_bias

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

DUMP_DIR = "activations"

WEIGHT_KEYS = [
    "visual.merger.ln_q.weight",
    "visual.merger.ln_q.bias",
    "visual.merger.mlp.0.weight",
    "visual.merger.mlp.0.bias",
    "visual.merger.mlp.2.weight",
    "visual.merger.mlp.2.bias",
]


def validate_layer_norm() -> bool:
    """验证 Patch Merger 的 LayerNorm (ln_q)。"""
    from e2e.validate import load_activation, validate

    print("\n=== 验证 1: LayerNorm (ln_q) ===")
    x = load_activation(DUMP_DIR, "model__visual__merger__ln_q_input")
    expected = load_activation(DUMP_DIR, "model__visual__merger__ln_q_output")

    weights = load_weights(
        [
            "visual.merger.ln_q.weight",
            "visual.merger.ln_q.bias",
        ]
    )
    actual = layer_norm(
        x,
        weights["visual.merger.ln_q.weight"],
        weights["visual.merger.ln_q.bias"],
    )
    return validate("patch_merger_ln_q", actual, expected, atol=1e-5, rtol=1e-5)


def validate_mlp() -> bool:
    """验证 Patch Merger 的 MLP 部分（跳过空间合并）。

    使用 mlp.0 的输入转储直接验证 MLP 管道。
    """
    from e2e.validate import load_activation, validate

    print("\n=== 验证 2: MLP (Linear → GELU → Linear) ===")
    x = load_activation(DUMP_DIR, "model__visual__merger__mlp__0_input")
    expected = load_activation(DUMP_DIR, "model__visual__merger_output")

    print(f"MLP 输入形状: {x.shape}")  # (3577, 5120)
    print(f"期望输出形状: {expected.shape}")  # (3577, 1536)

    weights = load_weights(
        [
            "visual.merger.mlp.0.weight",
            "visual.merger.mlp.0.bias",
            "visual.merger.mlp.2.weight",
            "visual.merger.mlp.2.bias",
        ]
    )

    h = x @ weights["visual.merger.mlp.0.weight"].T + weights["visual.merger.mlp.0.bias"]
    h = gelu_exact(h)
    actual = h @ weights["visual.merger.mlp.2.weight"].T + weights["visual.merger.mlp.2.bias"]

    return validate("patch_merger_mlp", actual, expected, atol=1e-3, rtol=1e-3)


def validate_full_merger() -> bool:
    """验证完整的 Patch Merger 管道。"""
    from e2e.validate import load_activation, validate

    print("\n=== 验证 3: 完整 Patch Merger ===")
    x = load_activation(DUMP_DIR, "model__visual__merger_input")
    expected = load_activation(DUMP_DIR, "model__visual__merger_output")

    print(f"输入形状: {x.shape}")  # (14308, 1280)
    print(f"期望输出形状: {expected.shape}")  # (3577, 1536)

    weights = load_weights(WEIGHT_KEYS)

    actual = patch_merger(
        x,
        weights["visual.merger.ln_q.weight"],
        weights["visual.merger.ln_q.bias"],
        weights["visual.merger.mlp.0.weight"],
        weights["visual.merger.mlp.0.bias"],
        weights["visual.merger.mlp.2.weight"],
        weights["visual.merger.mlp.2.bias"],
    )

    return validate("patch_merger_full", actual, expected, atol=1e-3, rtol=1e-3)


def main() -> None:
    results = [
        validate_layer_norm(),
        validate_mlp(),
        validate_full_merger(),
    ]

    print(f"\n{'='*60}")
    print(f"Patch Merger 验证: {sum(results)}/{len(results)} 通过")

    if not all(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
