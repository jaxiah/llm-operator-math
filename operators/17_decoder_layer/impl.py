"""17 — Decoder Layer（解码器层）—— 纯 NumPy 实现与验证

DecoderLayer(x) = x + MLP(RMSNorm(x + Attn(RMSNorm(x))))

组合 RMSNorm、Self-Attention、Gated MLP 和残差连接，
构成 Qwen2-VL 文本解码器的单个 Transformer 层。

用法:
    python -m operators.17_decoder_layer.impl
"""

import glob
import importlib
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# 子算子导入（目录名以数字开头，需要 importlib）
# ---------------------------------------------------------------------------

_rms_norm_mod = importlib.import_module("operators.04_rms_norm.impl")
_silu_mod = importlib.import_module("operators.10_silu.impl")
_residual_mod = importlib.import_module("operators.13_residual_connection.impl")

rms_norm = _rms_norm_mod.rms_norm
silu = _silu_mod.silu
residual_add = _residual_mod.residual_add


# ---------------------------------------------------------------------------
# 核心实现
# ---------------------------------------------------------------------------

def gated_mlp(
    x: np.ndarray,
    gate_weight: np.ndarray,
    up_weight: np.ndarray,
    down_weight: np.ndarray,
) -> np.ndarray:
    """Gated MLP (SwiGLU): gate/up 并行投影 → 门控 → 下投影"""
    gate = silu(x @ gate_weight.T)
    up = x @ up_weight.T
    return (gate * up) @ down_weight.T


def decoder_layer(
    x: np.ndarray,
    input_ln_w: np.ndarray,
    post_ln_w: np.ndarray,
    gate_w: np.ndarray,
    up_w: np.ndarray,
    down_w: np.ndarray,
    attn_output: np.ndarray,
) -> np.ndarray:
    """Decoder Layer: RMSNorm → Attn → Residual → RMSNorm → GatedMLP → Residual

    注意: self-attention 输出以参数形式传入（因为完整的 GQA + RoPE + 因果掩码
    实现复杂度高，这里先用 dump 值替代来验证层的组装逻辑）。

    Args:
        x:           层输入 (B, T, d)
        input_ln_w:  输入层归一化权重 (d,)
        post_ln_w:   注意力后层归一化权重 (d,)
        gate_w:      MLP 门控投影 (d_ff, d)
        up_w:        MLP 上投影 (d_ff, d)
        down_w:      MLP 下投影 (d, d_ff)
        attn_output: self-attention 的输出 (B, T, d)

    Returns:
        层输出 (B, T, d)
    """
    # Step 1: Input LayerNorm
    x_normed = rms_norm(x, input_ln_w)
    # Step 2: Self-Attention (使用预计算的输出)
    attn_out = attn_output
    # Step 3: 第一个残差连接
    hidden = residual_add(x, attn_out)
    # Step 4: Post-Attention LayerNorm
    hidden_normed = rms_norm(hidden, post_ln_w)
    # Step 5: Gated MLP (SwiGLU)
    mlp_out = gated_mlp(hidden_normed, gate_w, up_w, down_w)
    # Step 6: 第二个残差连接
    output = residual_add(hidden, mlp_out)
    return output


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
PREFIX = "model__language_model__layers__0"


def _load_validate():
    from e2e.validate import validate, load_activation
    return validate, load_activation


validate, load_activation = _load_validate()


def validate_input_layernorm() -> bool:
    """验证 1: Input LayerNorm"""
    print("\n=== Step 1: Input LayerNorm ===")
    x = load_activation(DUMP_DIR, f"{PREFIX}__input_layernorm_input")
    expected = load_activation(DUMP_DIR, f"{PREFIX}__input_layernorm_output")

    weights = load_weights(["model.layers.0.input_layernorm.weight"])
    w = weights["model.layers.0.input_layernorm.weight"]

    actual = rms_norm(x, w)
    return validate("input_layernorm", actual, expected, atol=1e-4, rtol=1e-4)


def validate_post_attn_residual() -> bool:
    """验证 2: 第一个残差连接 (layer_input + attn_output)"""
    print("\n=== Step 2: Post-Attention Residual ===")
    layer_input = load_activation(DUMP_DIR, f"{PREFIX}_input")
    attn_out = load_activation(DUMP_DIR, f"{PREFIX}__self_attn_output_idx0")
    expected = load_activation(DUMP_DIR, f"{PREFIX}__post_attention_layernorm_input")

    actual = residual_add(layer_input, attn_out)
    return validate("post_attn_residual", actual, expected, atol=1e-4, rtol=1e-4)


def validate_post_attn_layernorm() -> bool:
    """验证 3: Post-Attention LayerNorm"""
    print("\n=== Step 3: Post-Attention LayerNorm ===")
    x = load_activation(DUMP_DIR, f"{PREFIX}__post_attention_layernorm_input")
    expected = load_activation(DUMP_DIR, f"{PREFIX}__post_attention_layernorm_output")

    weights = load_weights(["model.layers.0.post_attention_layernorm.weight"])
    w = weights["model.layers.0.post_attention_layernorm.weight"]

    actual = rms_norm(x, w)
    return validate("post_attn_layernorm", actual, expected, atol=1e-4, rtol=1e-4)


def validate_gated_mlp() -> bool:
    """验证 4: Gated MLP (SwiGLU)"""
    print("\n=== Step 4: Gated MLP (SwiGLU) ===")
    x = load_activation(DUMP_DIR, f"{PREFIX}__mlp_input")
    expected = load_activation(DUMP_DIR, f"{PREFIX}__mlp_output")

    weight_keys = [
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    ]
    weights = load_weights(weight_keys)
    gate_w = weights["model.layers.0.mlp.gate_proj.weight"]
    up_w = weights["model.layers.0.mlp.up_proj.weight"]
    down_w = weights["model.layers.0.mlp.down_proj.weight"]

    actual = gated_mlp(x, gate_w, up_w, down_w)
    return validate("gated_mlp", actual, expected, atol=1e-4, rtol=1e-4)


def validate_final_residual() -> bool:
    """验证 5: 第二个残差连接 (post_attn_hidden + mlp_output)"""
    print("\n=== Step 5: Final Residual ===")
    hidden = load_activation(DUMP_DIR, f"{PREFIX}__post_attention_layernorm_input")
    mlp_out = load_activation(DUMP_DIR, f"{PREFIX}__mlp_output")
    expected = load_activation(DUMP_DIR, f"{PREFIX}_output")

    actual = residual_add(hidden, mlp_out)
    return validate("final_residual", actual, expected, atol=1e-4, rtol=1e-4)


def validate_full_layer() -> bool:
    """验证 6: 完整 Decoder Layer（attention 使用 dump 值）"""
    print("\n=== Step 6: Full Decoder Layer (attn from dump) ===")
    layer_input = load_activation(DUMP_DIR, f"{PREFIX}_input")
    attn_out = load_activation(DUMP_DIR, f"{PREFIX}__self_attn_output_idx0")
    expected = load_activation(DUMP_DIR, f"{PREFIX}_output")

    print(f"层输入形状: {layer_input.shape}")     # (1, 3602, 1536)
    print(f"注意力输出形状: {attn_out.shape}")     # (1, 3602, 1536)
    print(f"期望输出形状: {expected.shape}")        # (1, 3602, 1536)

    weight_keys = [
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    ]
    weights = load_weights(weight_keys)

    actual = decoder_layer(
        x=layer_input,
        input_ln_w=weights["model.layers.0.input_layernorm.weight"],
        post_ln_w=weights["model.layers.0.post_attention_layernorm.weight"],
        gate_w=weights["model.layers.0.mlp.gate_proj.weight"],
        up_w=weights["model.layers.0.mlp.up_proj.weight"],
        down_w=weights["model.layers.0.mlp.down_proj.weight"],
        attn_output=attn_out,
    )
    return validate("full_decoder_layer", actual, expected, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    results = [
        validate_input_layernorm(),
        validate_post_attn_residual(),
        validate_post_attn_layernorm(),
        validate_gated_mlp(),
        validate_final_residual(),
        validate_full_layer(),
    ]

    print(f"\n{'='*60}")
    passed = sum(results)
    total = len(results)
    print(f"Decoder Layer 验证: {passed}/{total} 通过")
    if not all(results):
        raise SystemExit(1)
