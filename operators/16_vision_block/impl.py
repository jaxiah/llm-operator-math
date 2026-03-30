"""Vision Transformer Block —— 纯 NumPy 实现与验证

Vision Block = Pre-Norm₁ → Attention → 残差 → Pre-Norm₂ → MLP → 残差

用于 Qwen2-VL 视觉编码器的每个 Transformer block。
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
_quickgelu_mod = importlib.import_module("operators.08_quickgelu.impl")
_residual_mod = importlib.import_module("operators.13_residual_connection.impl")

layer_norm = _layer_norm_mod.layer_norm
quick_gelu = _quickgelu_mod.quick_gelu
residual_add = _residual_mod.residual_add


def vision_mlp(
    x: np.ndarray,
    fc1_weight: np.ndarray,
    fc1_bias: np.ndarray,
    fc2_weight: np.ndarray,
    fc2_bias: np.ndarray,
) -> np.ndarray:
    """Vision MLP: FC1 → QuickGELU → FC2"""
    h = x @ fc1_weight.T + fc1_bias
    h = quick_gelu(h)
    return h @ fc2_weight.T + fc2_bias


def vision_block(
    x: np.ndarray,
    norm1_weight: np.ndarray,
    norm1_bias: np.ndarray,
    attn_fn,
    norm2_weight: np.ndarray,
    norm2_bias: np.ndarray,
    fc1_weight: np.ndarray,
    fc1_bias: np.ndarray,
    fc2_weight: np.ndarray,
    fc2_bias: np.ndarray,
) -> np.ndarray:
    """Vision Transformer Block (Pre-Norm).

    Args:
        x: 输入 (n, d)
        norm1_weight, norm1_bias: 第一个 LayerNorm 参数
        attn_fn: 注意力函数，接受归一化后的输入，返回注意力输出
        norm2_weight, norm2_bias: 第二个 LayerNorm 参数
        fc1_weight, fc1_bias: MLP 第一层参数
        fc2_weight, fc2_bias: MLP 第二层参数

    Returns:
        输出 (n, d)
    """
    # 子层 1: Attention
    x_normed = layer_norm(x, norm1_weight, norm1_bias)
    attn_out = attn_fn(x_normed)
    x = residual_add(x, attn_out)

    # 子层 2: MLP
    x_normed = layer_norm(x, norm2_weight, norm2_bias)
    mlp_out = vision_mlp(x_normed, fc1_weight, fc1_bias, fc2_weight, fc2_bias)
    x = residual_add(x, mlp_out)

    return x


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
    "visual.blocks.0.norm1.weight",
    "visual.blocks.0.norm1.bias",
    "visual.blocks.0.attn.qkv.weight",
    "visual.blocks.0.attn.qkv.bias",
    "visual.blocks.0.attn.proj.weight",
    "visual.blocks.0.attn.proj.bias",
    "visual.blocks.0.norm2.weight",
    "visual.blocks.0.norm2.bias",
    "visual.blocks.0.mlp.fc1.weight",
    "visual.blocks.0.mlp.fc1.bias",
    "visual.blocks.0.mlp.fc2.weight",
    "visual.blocks.0.mlp.fc2.bias",
]


def validate_norm1() -> bool:
    """验证 Block 0 的第一个 LayerNorm。"""
    from e2e.validate import load_activation, validate

    print("\n=== 验证 1: norm1 (LayerNorm₁) ===")
    x = load_activation(DUMP_DIR, "model__visual__blocks__0__norm1_input")
    expected = load_activation(DUMP_DIR, "model__visual__blocks__0__norm1_output")

    weights = load_weights([
        "visual.blocks.0.norm1.weight",
        "visual.blocks.0.norm1.bias",
    ])
    actual = layer_norm(
        x,
        weights["visual.blocks.0.norm1.weight"],
        weights["visual.blocks.0.norm1.bias"],
    )
    return validate("vision_block0_norm1", actual, expected, atol=1e-5, rtol=1e-5)


def validate_attn_residual() -> bool:
    """验证注意力子层的残差连接: x + attn_out == norm2_input。"""
    from e2e.validate import load_activation, validate

    print("\n=== 验证 2: 注意力残差连接 ===")
    print("    block_input + attn_output == norm2_input")
    block_input = load_activation(DUMP_DIR, "model__visual__blocks__0_input")
    attn_output = load_activation(DUMP_DIR, "model__visual__blocks__0__attn_output")
    expected = load_activation(DUMP_DIR, "model__visual__blocks__0__norm2_input")

    actual = residual_add(block_input, attn_output)
    return validate("vision_block0_attn_residual", actual, expected, atol=1e-5, rtol=1e-5)


def validate_norm2() -> bool:
    """验证 Block 0 的第二个 LayerNorm。"""
    from e2e.validate import load_activation, validate

    print("\n=== 验证 3: norm2 (LayerNorm₂) ===")
    x = load_activation(DUMP_DIR, "model__visual__blocks__0__norm2_input")
    expected = load_activation(DUMP_DIR, "model__visual__blocks__0__norm2_output")

    weights = load_weights([
        "visual.blocks.0.norm2.weight",
        "visual.blocks.0.norm2.bias",
    ])
    actual = layer_norm(
        x,
        weights["visual.blocks.0.norm2.weight"],
        weights["visual.blocks.0.norm2.bias"],
    )
    return validate("vision_block0_norm2", actual, expected, atol=1e-5, rtol=1e-5)


def validate_mlp() -> bool:
    """验证 Block 0 的 MLP 子层。"""
    from e2e.validate import load_activation, validate

    print("\n=== 验证 4: MLP (FC1 → QuickGELU → FC2) ===")
    x = load_activation(DUMP_DIR, "model__visual__blocks__0__mlp_input")
    expected = load_activation(DUMP_DIR, "model__visual__blocks__0__mlp_output")

    weights = load_weights([
        "visual.blocks.0.mlp.fc1.weight",
        "visual.blocks.0.mlp.fc1.bias",
        "visual.blocks.0.mlp.fc2.weight",
        "visual.blocks.0.mlp.fc2.bias",
    ])
    actual = vision_mlp(
        x,
        weights["visual.blocks.0.mlp.fc1.weight"],
        weights["visual.blocks.0.mlp.fc1.bias"],
        weights["visual.blocks.0.mlp.fc2.weight"],
        weights["visual.blocks.0.mlp.fc2.bias"],
    )
    return validate("vision_block0_mlp", actual, expected, atol=1e-4, rtol=1e-4)


def validate_mlp_residual() -> bool:
    """验证 MLP 子层的残差连接: norm2_input + mlp_out == block_output。

    注意：norm2_input 就是第一次残差连接后的 x'。
    """
    from e2e.validate import load_activation, validate

    print("\n=== 验证 5: MLP 残差连接 ===")
    print("    norm2_input + mlp_output == block_output")
    x_mid = load_activation(DUMP_DIR, "model__visual__blocks__0__norm2_input")
    mlp_output = load_activation(DUMP_DIR, "model__visual__blocks__0__mlp_output")
    expected = load_activation(DUMP_DIR, "model__visual__blocks__0_output")

    actual = residual_add(x_mid, mlp_output)
    return validate("vision_block0_mlp_residual", actual, expected, atol=1e-5, rtol=1e-5)


def validate_full_block() -> bool:
    """验证完整的 Vision Block（注意力使用转储值）。

    由于窗口注意力 + RoPE 的完整实现较复杂，这里使用模块化策略：
    将注意力子层的输出从转储文件加载，验证其他所有组件的组合正确性。
    """
    from e2e.validate import load_activation, validate

    print("\n=== 验证 6: 完整 Vision Block（注意力使用转储值）===")

    block_input = load_activation(DUMP_DIR, "model__visual__blocks__0_input")
    expected = load_activation(DUMP_DIR, "model__visual__blocks__0_output")

    # 加载注意力输出转储（跳过窗口注意力 + RoPE 的重新实现）
    attn_output_dump = load_activation(DUMP_DIR, "model__visual__blocks__0__attn_output")

    weights = load_weights(WEIGHT_KEYS)

    # 用转储的注意力输出构造 attn_fn
    def attn_fn(_x_normed: np.ndarray) -> np.ndarray:
        return attn_output_dump

    actual = vision_block(
        block_input,
        weights["visual.blocks.0.norm1.weight"],
        weights["visual.blocks.0.norm1.bias"],
        attn_fn,
        weights["visual.blocks.0.norm2.weight"],
        weights["visual.blocks.0.norm2.bias"],
        weights["visual.blocks.0.mlp.fc1.weight"],
        weights["visual.blocks.0.mlp.fc1.bias"],
        weights["visual.blocks.0.mlp.fc2.weight"],
        weights["visual.blocks.0.mlp.fc2.bias"],
    )

    print("注：注意力子层使用转储值，验证 norm + MLP + 残差连接的组合正确性")
    return validate("vision_block0_full", actual, expected, atol=1e-4, rtol=1e-4)


def main() -> None:
    results = [
        validate_norm1(),
        validate_attn_residual(),
        validate_norm2(),
        validate_mlp(),
        validate_mlp_residual(),
        validate_full_block(),
    ]

    print(f"\n{'='*60}")
    print(f"Vision Block 验证: {sum(results)}/{len(results)} 通过")

    if not all(results):
        print("\n注：视觉注意力使用了窗口注意力 + RoPE，完整实现见 operators/07_attention")
        sys.exit(1)


if __name__ == "__main__":
    main()
