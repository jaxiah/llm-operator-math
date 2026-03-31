"""E2E numpy validation runner for Qwen2-VL-2B-Instruct.

Runs all operator validations in inference execution order,
printing a summary report at the end.

Usage:
    python -m e2e.run_numpy_e2e [--dump-dir activations]
"""

import argparse
import glob
import importlib
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from e2e.validate import validate, load_activation  # noqa: E402

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
results: list[tuple[str, bool]] = []
DUMP_DIR = "activations"
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def record(name: str, passed: bool) -> bool:
    results.append((name, passed))
    return passed


def section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def load_weights(keys: list[str]) -> dict[str, np.ndarray]:
    """Load safetensors weights from HuggingFace cache."""
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
    if len(result) != len(keys):
        missing = set(keys) - set(result.keys())
        raise KeyError(f"Weights not found: {missing}")
    return result


# ---------------------------------------------------------------------------
# Import operator implementations
# ---------------------------------------------------------------------------
_conv3d_mod = importlib.import_module("operators.05_conv3d_patch_embed.impl")
_layer_norm_mod = importlib.import_module("operators.03_layer_norm.impl")
_rms_norm_mod = importlib.import_module("operators.04_rms_norm.impl")
_quickgelu_mod = importlib.import_module("operators.08_quickgelu.impl")
_gelu_mod = importlib.import_module("operators.09_gelu.impl")
_silu_mod = importlib.import_module("operators.10_silu.impl")
_residual_mod = importlib.import_module("operators.13_residual_connection.impl")
_token_embed_mod = importlib.import_module("operators.14_token_embedding.impl")
_patch_merger_mod = importlib.import_module("operators.15_patch_merger.impl")
_vision_block_mod = importlib.import_module("operators.16_vision_block.impl")
_decoder_layer_mod = importlib.import_module("operators.17_decoder_layer.impl")
_lm_head_mod = importlib.import_module("operators.18_lm_head.impl")

conv3d_patch_embed = _conv3d_mod.conv3d_patch_embed
layer_norm = _layer_norm_mod.layer_norm
rms_norm = _rms_norm_mod.rms_norm
quick_gelu = _quickgelu_mod.quick_gelu
gelu_exact = _gelu_mod.gelu_exact
silu = _silu_mod.silu
residual_add = _residual_mod.residual_add
token_embedding = _token_embed_mod.token_embedding
spatial_merge = _patch_merger_mod.spatial_merge
patch_merger = _patch_merger_mod.patch_merger
vision_mlp = _vision_block_mod.vision_mlp
vision_block = _vision_block_mod.vision_block
gated_mlp = _decoder_layer_mod.gated_mlp
decoder_layer = _decoder_layer_mod.decoder_layer
lm_head = _lm_head_mod.lm_head


# ============================================================
# Vision Encoder
# ============================================================


def run_conv3d_patch_embed() -> None:
    """05: Conv3d Patch Embedding — pixel_values → patch embeddings."""
    print("\n--- 05: Conv3d Patch Embedding ---")
    x = load_activation(DUMP_DIR, "model__visual__patch_embed_input")
    expected = load_activation(DUMP_DIR, "model__visual__patch_embed_output")

    weights = load_weights(["visual.patch_embed.proj.weight"])
    W = weights["visual.patch_embed.proj.weight"]

    actual = conv3d_patch_embed(x, W)
    record("conv3d_patch_embed", validate("conv3d_patch_embed", actual, expected, atol=1e-2, rtol=1e-2))


def run_vision_block() -> None:
    """16: Vision Block 0 — norm, attention residual, MLP, full block."""
    print("\n--- 16: Vision Block 0 ---")

    block_input = load_activation(DUMP_DIR, "model__visual__blocks__0_input")
    block_output = load_activation(DUMP_DIR, "model__visual__blocks__0_output")
    attn_output = load_activation(DUMP_DIR, "model__visual__blocks__0__attn_output")

    wk = [
        "visual.blocks.0.norm1.weight",
        "visual.blocks.0.norm1.bias",
        "visual.blocks.0.norm2.weight",
        "visual.blocks.0.norm2.bias",
        "visual.blocks.0.mlp.fc1.weight",
        "visual.blocks.0.mlp.fc1.bias",
        "visual.blocks.0.mlp.fc2.weight",
        "visual.blocks.0.mlp.fc2.bias",
    ]
    W = load_weights(wk)

    # Sub-step 1: norm1
    norm1_in = load_activation(DUMP_DIR, "model__visual__blocks__0__norm1_input")
    norm1_exp = load_activation(DUMP_DIR, "model__visual__blocks__0__norm1_output")
    actual = layer_norm(norm1_in, W["visual.blocks.0.norm1.weight"], W["visual.blocks.0.norm1.bias"])
    record("vision_block0_norm1", validate("vision_block0_norm1", actual, norm1_exp, atol=1e-5, rtol=1e-5))

    # Sub-step 2: attention residual
    norm2_exp = load_activation(DUMP_DIR, "model__visual__blocks__0__norm2_input")
    actual = residual_add(block_input, attn_output)
    record("vision_block0_attn_residual", validate("vision_block0_attn_residual", actual, norm2_exp, atol=1e-5, rtol=1e-5))

    # Sub-step 3: norm2
    norm2_out_exp = load_activation(DUMP_DIR, "model__visual__blocks__0__norm2_output")
    actual = layer_norm(norm2_exp, W["visual.blocks.0.norm2.weight"], W["visual.blocks.0.norm2.bias"])
    record("vision_block0_norm2", validate("vision_block0_norm2", actual, norm2_out_exp, atol=1e-5, rtol=1e-5))

    # Sub-step 4: MLP
    mlp_in = load_activation(DUMP_DIR, "model__visual__blocks__0__mlp_input")
    mlp_exp = load_activation(DUMP_DIR, "model__visual__blocks__0__mlp_output")
    actual = vision_mlp(
        mlp_in, W["visual.blocks.0.mlp.fc1.weight"], W["visual.blocks.0.mlp.fc1.bias"], W["visual.blocks.0.mlp.fc2.weight"], W["visual.blocks.0.mlp.fc2.bias"]
    )
    record("vision_block0_mlp", validate("vision_block0_mlp", actual, mlp_exp, atol=1e-4, rtol=1e-4))

    # Sub-step 5: MLP residual
    actual = residual_add(norm2_exp, mlp_exp)
    record("vision_block0_mlp_residual", validate("vision_block0_mlp_residual", actual, block_output, atol=1e-5, rtol=1e-5))

    # Sub-step 6: full block (attention from dump)
    def attn_fn(_x_normed):
        return attn_output

    actual = vision_block(
        block_input,
        W["visual.blocks.0.norm1.weight"],
        W["visual.blocks.0.norm1.bias"],
        attn_fn,
        W["visual.blocks.0.norm2.weight"],
        W["visual.blocks.0.norm2.bias"],
        W["visual.blocks.0.mlp.fc1.weight"],
        W["visual.blocks.0.mlp.fc1.bias"],
        W["visual.blocks.0.mlp.fc2.weight"],
        W["visual.blocks.0.mlp.fc2.bias"],
    )
    record("vision_block0_full", validate("vision_block0_full", actual, block_output, atol=1e-4, rtol=1e-4))


def run_patch_merger() -> None:
    """15: Patch Merger — LN → spatial merge → MLP."""
    print("\n--- 15: Patch Merger ---")

    wk = [
        "visual.merger.ln_q.weight",
        "visual.merger.ln_q.bias",
        "visual.merger.mlp.0.weight",
        "visual.merger.mlp.0.bias",
        "visual.merger.mlp.2.weight",
        "visual.merger.mlp.2.bias",
    ]
    W = load_weights(wk)

    # Sub-step 1: LayerNorm (ln_q)
    ln_in = load_activation(DUMP_DIR, "model__visual__merger__ln_q_input")
    ln_exp = load_activation(DUMP_DIR, "model__visual__merger__ln_q_output")
    actual = layer_norm(ln_in, W["visual.merger.ln_q.weight"], W["visual.merger.ln_q.bias"])
    record("patch_merger_ln_q", validate("patch_merger_ln_q", actual, ln_exp, atol=1e-5, rtol=1e-5))

    # Sub-step 2: MLP (from mlp.0 input → merger output)
    mlp_in = load_activation(DUMP_DIR, "model__visual__merger__mlp__0_input")
    merger_exp = load_activation(DUMP_DIR, "model__visual__merger_output")
    h = mlp_in @ W["visual.merger.mlp.0.weight"].T + W["visual.merger.mlp.0.bias"]
    h = gelu_exact(h)
    actual = h @ W["visual.merger.mlp.2.weight"].T + W["visual.merger.mlp.2.bias"]
    record("patch_merger_mlp", validate("patch_merger_mlp", actual, merger_exp, atol=1e-3, rtol=1e-3))

    # Sub-step 3: full pipeline
    merger_in = load_activation(DUMP_DIR, "model__visual__merger_input")
    actual = patch_merger(
        merger_in,
        W["visual.merger.ln_q.weight"],
        W["visual.merger.ln_q.bias"],
        W["visual.merger.mlp.0.weight"],
        W["visual.merger.mlp.0.bias"],
        W["visual.merger.mlp.2.weight"],
        W["visual.merger.mlp.2.bias"],
    )
    record("patch_merger_full", validate("patch_merger_full", actual, merger_exp, atol=1e-3, rtol=1e-3))


# ============================================================
# Text Decoder
# ============================================================


def run_token_embedding() -> None:
    """14: Token Embedding — token ids → embeddings."""
    print("\n--- 14: Token Embedding ---")

    token_ids = load_activation(DUMP_DIR, "model__language_model__embed_tokens_input").astype(np.int64)
    expected = load_activation(DUMP_DIR, "model__language_model__embed_tokens_output")

    weights = load_weights(["model.embed_tokens.weight"])
    E = weights["model.embed_tokens.weight"]

    actual = token_embedding(token_ids, E)
    record("token_embedding", validate("token_embedding", actual, expected, atol=1e-4, rtol=1e-4))


def run_decoder_layer() -> None:
    """17: Decoder Layer 0 — RMSNorm, attention residual, MLP, full layer."""
    print("\n--- 17: Decoder Layer 0 ---")

    PREFIX = "model__language_model__layers__0"

    layer_input = load_activation(DUMP_DIR, f"{PREFIX}_input")
    layer_output = load_activation(DUMP_DIR, f"{PREFIX}_output")
    attn_out = load_activation(DUMP_DIR, f"{PREFIX}__self_attn_output_idx0")

    wk = [
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    ]
    W = load_weights(wk)

    # Sub-step 1: Input RMSNorm
    ln_in = load_activation(DUMP_DIR, f"{PREFIX}__input_layernorm_input")
    ln_exp = load_activation(DUMP_DIR, f"{PREFIX}__input_layernorm_output")
    actual = rms_norm(ln_in, W["model.layers.0.input_layernorm.weight"])
    record("decoder0_input_rmsnorm", validate("decoder0_input_rmsnorm", actual, ln_exp, atol=1e-4, rtol=1e-4))

    # Sub-step 2: Post-attention residual
    post_attn_exp = load_activation(DUMP_DIR, f"{PREFIX}__post_attention_layernorm_input")
    actual = residual_add(layer_input, attn_out)
    record("decoder0_attn_residual", validate("decoder0_attn_residual", actual, post_attn_exp, atol=1e-4, rtol=1e-4))

    # Sub-step 3: Post-attention RMSNorm
    post_ln_exp = load_activation(DUMP_DIR, f"{PREFIX}__post_attention_layernorm_output")
    actual = rms_norm(post_attn_exp, W["model.layers.0.post_attention_layernorm.weight"])
    record("decoder0_post_attn_rmsnorm", validate("decoder0_post_attn_rmsnorm", actual, post_ln_exp, atol=1e-4, rtol=1e-4))

    # Sub-step 4: Gated MLP (SwiGLU)
    mlp_in = load_activation(DUMP_DIR, f"{PREFIX}__mlp_input")
    mlp_exp = load_activation(DUMP_DIR, f"{PREFIX}__mlp_output")
    actual = gated_mlp(mlp_in, W["model.layers.0.mlp.gate_proj.weight"], W["model.layers.0.mlp.up_proj.weight"], W["model.layers.0.mlp.down_proj.weight"])
    record("decoder0_gated_mlp", validate("decoder0_gated_mlp", actual, mlp_exp, atol=1e-4, rtol=1e-4))

    # Sub-step 5: Final residual
    actual = residual_add(post_attn_exp, mlp_exp)
    record("decoder0_final_residual", validate("decoder0_final_residual", actual, layer_output, atol=1e-4, rtol=1e-4))

    # Sub-step 6: Full decoder layer (attention from dump)
    actual = decoder_layer(
        x=layer_input,
        input_ln_w=W["model.layers.0.input_layernorm.weight"],
        post_ln_w=W["model.layers.0.post_attention_layernorm.weight"],
        gate_w=W["model.layers.0.mlp.gate_proj.weight"],
        up_w=W["model.layers.0.mlp.up_proj.weight"],
        down_w=W["model.layers.0.mlp.down_proj.weight"],
        attn_output=attn_out,
    )
    record("decoder0_full_layer", validate("decoder0_full_layer", actual, layer_output, atol=1e-4, rtol=1e-4))


def run_lm_head() -> None:
    """18: LM Head — Final RMSNorm → Linear → logits."""
    print("\n--- 18: LM Head ---")

    # Sub-step 1: Final RMSNorm
    norm_in = load_activation(DUMP_DIR, "model__language_model__norm_input")
    norm_exp = load_activation(DUMP_DIR, "model__language_model__norm_output")

    norm_w = load_weights(["model.norm.weight"])["model.norm.weight"]
    actual = rms_norm(norm_in, norm_w)
    record("final_rms_norm", validate("final_rms_norm", actual, norm_exp, atol=1e-4, rtol=1e-4))

    # Sub-step 2: LM Head projection (last token only to save memory)
    lm_in = load_activation(DUMP_DIR, "lm_head_input")
    lm_exp = load_activation(DUMP_DIR, "lm_head_output")

    try:
        lm_w = load_weights(["lm_head.weight"])["lm_head.weight"]
    except (KeyError, Exception):
        lm_w = load_weights(["model.embed_tokens.weight"])["model.embed_tokens.weight"]

    x_last = lm_in[:, -1:, :]
    expected_last = lm_exp[:, -1:, :]
    actual_last = x_last @ lm_w.T
    record("lm_head_last_token", validate("lm_head_last_token", actual_last, expected_last, atol=1e-3, rtol=1e-3))

    # Sub-step 3: Full LM Head (norm → linear, last token)
    final_logits = load_activation(DUMP_DIR, "_final_logits")
    norm_input_last = norm_in[:, -1:, :]
    expected_logits_last = final_logits[:, -1:, :]
    actual_logits_last = lm_head(norm_input_last, norm_w, lm_w)
    record("full_lm_head_last_token", validate("full_lm_head_last_token", actual_logits_last, expected_logits_last, atol=1e-3, rtol=1e-3))

    # Show argmax prediction
    pred_id = int(np.argmax(actual_logits_last[0, -1, :]))
    expected_id = int(np.argmax(expected_logits_last[0, -1, :]))
    print(f"\n  预测 token ID (argmax): {pred_id}")
    print(f"  期望 token ID (argmax): {expected_id}")
    print(f"  argmax 一致: {'✓' if pred_id == expected_id else '✗'}")


# ============================================================
# Main
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="E2E numpy validation for Qwen2-VL-2B-Instruct")
    parser.add_argument("--dump-dir", default="activations", help="Path to activation dumps (default: activations)")
    args = parser.parse_args()

    global DUMP_DIR
    DUMP_DIR = args.dump_dir

    if not os.path.isdir(DUMP_DIR):
        print(f"Error: dump directory '{DUMP_DIR}' not found.")
        print("Run `python -m e2e.dump_activations` first to generate activation dumps.")
        sys.exit(1)

    # ── Vision Encoder ──
    section("Vision Encoder")
    run_conv3d_patch_embed()
    run_vision_block()
    run_patch_merger()

    # ── Text Decoder ──
    section("Text Decoder")
    run_token_embedding()
    run_decoder_layer()
    run_lm_head()

    # ── Summary ──
    section("Summary")
    passed = sum(1 for _, p in results if p)
    total = len(results)
    for name, p in results:
        status = "✓" if p else "✗"
        print(f"  [{status}] {name}")
    print(f"\n  {passed}/{total} passed")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
