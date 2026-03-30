"""18 — LM Head（语言模型输出头）—— 纯 NumPy 实现与验证

LMHead(x) = RMSNorm(x) @ W_lm.T

最终的 RMSNorm 归一化加上线性投影，将隐藏状态映射到词表维度的 logits。

用法:
    python -m operators.18_lm_head.impl
"""

import glob
import importlib
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# 子算子导入
# ---------------------------------------------------------------------------

_rms_norm_mod = importlib.import_module("operators.04_rms_norm.impl")
rms_norm = _rms_norm_mod.rms_norm


# ---------------------------------------------------------------------------
# 核心实现
# ---------------------------------------------------------------------------

def lm_head(
    x: np.ndarray,
    norm_weight: np.ndarray,
    lm_weight: np.ndarray,
) -> np.ndarray:
    """LM Head: Final RMSNorm → Linear projection → logits

    Args:
        x:           隐藏状态 (B, T, d)
        norm_weight: 最终 RMSNorm 权重 (d,)
        lm_weight:   LM Head 线性层权重 (V, d)

    Returns:
        logits (B, T, V)
    """
    hidden = rms_norm(x, norm_weight)
    logits = hidden @ lm_weight.T
    return logits


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


def _load_validate():
    from e2e.validate import validate, load_activation
    return validate, load_activation


validate, load_activation = _load_validate()


def validate_final_norm() -> bool:
    """验证 1: Final RMSNorm"""
    print("\n=== Step 1: Final RMSNorm ===")
    x = load_activation(DUMP_DIR, "model__language_model__norm_input")
    expected = load_activation(DUMP_DIR, "model__language_model__norm_output")

    weights = load_weights(["model.norm.weight"])
    w = weights["model.norm.weight"]

    print(f"输入形状: {x.shape}")        # (1, 3602, 1536)
    print(f"权重形状: {w.shape}")         # (1536,)

    actual = rms_norm(x, w)
    return validate("final_rms_norm", actual, expected, atol=1e-4, rtol=1e-4)


def validate_lm_head_projection() -> bool:
    """验证 2: LM Head 线性投影（仅验证最后一个 token 位置以节省内存）"""
    print("\n=== Step 2: LM Head Linear (last token) ===")
    x = load_activation(DUMP_DIR, "lm_head_input")
    expected = load_activation(DUMP_DIR, "lm_head_output")

    # lm_head.weight 可能与 embed_tokens.weight 共享（weight tying）
    try:
        weights = load_weights(["lm_head.weight"])
        lm_w = weights["lm_head.weight"]
        print("使用 lm_head.weight")
    except (KeyError, Exception):
        print("lm_head.weight 不存在，尝试 model.embed_tokens.weight (weight tying)")
        weights = load_weights(["model.embed_tokens.weight"])
        lm_w = weights["model.embed_tokens.weight"]

    print(f"输入形状: {x.shape}")        # (1, 3602, 1536)
    print(f"权重形状: {lm_w.shape}")     # (151936, 1536)

    # 仅验证最后一个 token（完整矩阵乘法 ~2GB 结果）
    x_last = x[:, -1:, :]                # (1, 1, 1536)
    expected_last = expected[:, -1:, :]   # (1, 1, 151936)
    actual_last = x_last @ lm_w.T        # (1, 1, 151936)

    return validate("lm_head_last_token", actual_last, expected_last,
                     atol=1e-3, rtol=1e-3)


def validate_full_lm_head() -> bool:
    """验证 3: 完整 LM Head (RMSNorm → Linear)，仅最后 token"""
    print("\n=== Step 3: Full LM Head (norm → linear, last token) ===")
    norm_input = load_activation(DUMP_DIR, "model__language_model__norm_input")
    expected = load_activation(DUMP_DIR, "_final_logits")

    print(f"Norm 输入形状: {norm_input.shape}")   # (1, 3602, 1536)
    print(f"期望 logits 形状: {expected.shape}")   # (1, 3602, 151936)

    # 加载权重
    try:
        weight_keys = ["model.norm.weight", "lm_head.weight"]
        weights = load_weights(weight_keys)
        lm_w = weights["lm_head.weight"]
    except (KeyError, Exception):
        weight_keys = ["model.norm.weight", "model.embed_tokens.weight"]
        weights = load_weights(weight_keys)
        lm_w = weights["model.embed_tokens.weight"]

    norm_w = weights["model.norm.weight"]

    # 仅验证最后一个 token
    norm_input_last = norm_input[:, -1:, :]      # (1, 1, 1536)
    expected_last = expected[:, -1:, :]           # (1, 1, 151936)

    actual_last = lm_head(norm_input_last, norm_w, lm_w)

    ok = validate("full_lm_head_last_token", actual_last, expected_last,
                   atol=1e-3, rtol=1e-3)

    # 展示 argmax 预测结果
    pred_token_id = int(np.argmax(actual_last[0, -1, :]))
    expected_token_id = int(np.argmax(expected_last[0, -1, :]))
    print(f"\n预测 token ID (argmax): {pred_token_id}")
    print(f"期望 token ID (argmax): {expected_token_id}")
    print(f"argmax 一致: {'✓' if pred_token_id == expected_token_id else '✗'}")

    return ok


if __name__ == "__main__":
    results = [
        validate_final_norm(),
        validate_lm_head_projection(),
        validate_full_lm_head(),
    ]

    print(f"\n{'='*60}")
    passed = sum(results)
    total = len(results)
    print(f"LM Head 验证: {passed}/{total} 通过")
    if not all(results):
        raise SystemExit(1)
