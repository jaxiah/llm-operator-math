"""05 — 3D 卷积图像块嵌入 (Conv3d Patch Embedding)

用纯 NumPy 实现 Qwen2-VL 视觉编码器的 patch embedding，并用真实激活值验证。
当 stride == kernel_size 时，3D 卷积退化为简单的线性投影。
"""

import os

import numpy as np

from e2e.validate import validate, load_activation

DUMP_DIR = "activations"
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"


# ---------------------------------------------------------------------------
# 核心算子
# ---------------------------------------------------------------------------
def conv3d_patch_embed(
    x_flat: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
) -> np.ndarray:
    """非重叠 3D 卷积 patch embedding（stride == kernel_size 特例）。

    当 stride 等于 kernel_size 时，每个 patch 只对应一个输出，等价于线性投影：
        y = x_flat @ w_flat.T + bias

    Args:
        x_flat: (N, C*T*H*W) 展平的 patch 像素值
        weight: (out_channels, C, kT, kH, kW) 卷积核权重
        bias: (out_channels,) 偏置（可选）

    Returns:
        (N, out_channels) 每个 patch 的嵌入向量
    """
    w_flat = weight.reshape(weight.shape[0], -1)  # (out_channels, C*kT*kH*kW)
    # Use float64 for the large dot product (1176 elements) to reduce accumulation error
    y = x_flat.astype(np.float64) @ w_flat.astype(np.float64).T
    if bias is not None:
        y = y + bias.astype(np.float64)
    return y.astype(np.float32)


# ---------------------------------------------------------------------------
# 权重加载工具
# ---------------------------------------------------------------------------
def load_weights(keys: list[str]) -> dict[str, np.ndarray]:
    """从 HuggingFace 缓存加载指定的 safetensors 权重张量。"""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from huggingface_hub import HfApi

    api = HfApi()
    siblings = api.model_info(MODEL_ID).siblings
    safetensor_files = [s.rfilename for s in siblings if s.rfilename.endswith(".safetensors")]

    result: dict[str, np.ndarray] = {}
    remaining = set(keys)
    for filename in safetensor_files:
        if not remaining:
            break
        path = hf_hub_download(repo_id=MODEL_ID, filename=filename)
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in list(remaining):
                if key in f.keys():
                    result[key] = f.get_tensor(key).float().numpy()
                    remaining.discard(key)
    if remaining:
        raise KeyError(f"Weights not found: {remaining}")
    return result


# ---------------------------------------------------------------------------
# 验证
# ---------------------------------------------------------------------------
def validate_patch_embed() -> bool:
    """验证 Conv3d patch embedding: pixel_values → patch embeddings。"""
    print("\n=== Conv3d Patch Embedding (14308, 1176) -> (14308, 1280) ===")

    # 加载激活值
    x = load_activation(DUMP_DIR, "model__visual__patch_embed_input")
    expected = load_activation(DUMP_DIR, "model__visual__patch_embed_output")

    # 加载权重（Qwen2-VL-2B 此层无偏置）
    weights = load_weights(["visual.patch_embed.proj.weight"])
    W = weights["visual.patch_embed.proj.weight"]  # (1280, 3, 2, 14, 14)

    # x 已经是展平形式 (N, 1176)，直接用线性投影
    actual = conv3d_patch_embed(x, W)

    # Relaxed tolerance: model uses bfloat16, and the large inner dimension (1176)
    # amplifies accumulation order differences between numpy and PyTorch conv3d.
    return validate("conv3d_patch_embed", actual, expected, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    results = [
        validate_patch_embed(),
    ]
    print(f"\n{'='*60}")
    print(f"Conv3d Patch Embedding 验证: {sum(results)}/{len(results)} 通过")
    if not all(results):
        raise SystemExit(1)
