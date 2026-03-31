"""RMS Normalization — 纯 numpy 实现，并用 Qwen2-VL text decoder 的真实权重验证。

用法:
    python -m operators.04_rms_norm.impl
"""

import glob
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# 核心算子
# ---------------------------------------------------------------------------


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """RMS Normalization: y = x / sqrt(mean(x^2) + eps) * weight

    对最后一个维度做均方根归一化。与 LayerNorm 相比，跳过了减均值的步骤，也没有 bias。

    Args:
        x: 输入张量，任意形状，归一化沿最后一维进行。
        weight: 缩放参数 (gamma)，形状与最后一维相同。
        eps: 防止除零的小常数。

    Returns:
        归一化后的张量，形状与输入相同。
    """
    # 计算 x^2 沿最后一维的均值
    mean_sq = np.mean(x**2, axis=-1, keepdims=True)

    # RMS = sqrt(mean(x^2) + eps)
    rms = np.sqrt(mean_sq + eps)

    # 归一化 + 缩放
    return (x / rms) * weight


# ---------------------------------------------------------------------------
# 权重加载
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
SHARD_FILES = [
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
]


def load_weights(keys: list[str]) -> dict[str, np.ndarray]:
    """从 HuggingFace Hub 缓存中加载 safetensors 权重。

    使用 safetensors.torch 加载（因为 numpy 后端不支持 bfloat16），
    然后转换为 float32 numpy 数组。
    """
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    # 下载第一个 shard 以定位模型缓存目录
    first_shard = hf_hub_download(MODEL_ID, SHARD_FILES[0])
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

DUMP_DIR = os.path.join("activations")

# Text decoder layer 0 的 input_layernorm
WEIGHT_KEY = "model.layers.0.input_layernorm.weight"
INPUT_NAME = "model__language_model__layers__0__input_layernorm_input"
OUTPUT_NAME = "model__language_model__layers__0__input_layernorm_output"


def main() -> None:
    from e2e.validate import load_activation, validate

    print("=" * 60)
    print("RMS Normalization — Text Decoder layer 0, input_layernorm")
    print("=" * 60)

    # 1) 加载激活值
    x = load_activation(DUMP_DIR, INPUT_NAME)
    expected = load_activation(DUMP_DIR, OUTPUT_NAME)
    print(f"输入形状: {x.shape}  输出形状: {expected.shape}")

    # 2) 加载权重
    weights = load_weights([WEIGHT_KEY])
    w = weights[WEIGHT_KEY]
    print(f"weight 形状: {w.shape}")

    # 3) 计算 RMSNorm
    actual = rms_norm(x, w, eps=1e-6)

    # 4) 验证
    print()
    ok = validate("text_decoder_layer_0_input_layernorm", actual, expected)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
