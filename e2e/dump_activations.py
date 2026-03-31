"""Dump intermediate activations from a Qwen2-VL-2B-Instruct inference pass.

Registers forward hooks on key modules, runs a single-image inference,
and saves each module's input/output tensors as .npy files (float32).

Repeated layers (vision blocks, decoder layers) only dump layer 0.

Usage:
    python -m e2e.dump_activations [--output-dir activations] [--image demo.jpeg]
"""

import argparse
import os
import re

import numpy as np
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ---------------------------------------------------------------------------
# Which modules to hook
# ---------------------------------------------------------------------------
# Patterns that match module names we want to dump.
# For repeated blocks/layers, we only dump index 0.
HOOK_PATTERNS = [
    # --- Vision Encoder ---
    # Patch embedding (Conv3d)
    r"^model\.visual\.patch_embed$",
    r"^model\.visual\.patch_embed\.proj$",
    # Vision block 0 — all sub-modules
    r"^model\.visual\.blocks\.0$",
    r"^model\.visual\.blocks\.0\.norm1$",
    r"^model\.visual\.blocks\.0\.norm2$",
    r"^model\.visual\.blocks\.0\.attn$",
    r"^model\.visual\.blocks\.0\.attn\.qkv$",
    r"^model\.visual\.blocks\.0\.attn\.proj$",
    r"^model\.visual\.blocks\.0\.mlp$",
    r"^model\.visual\.blocks\.0\.mlp\.fc1$",
    r"^model\.visual\.blocks\.0\.mlp\.act$",
    r"^model\.visual\.blocks\.0\.mlp\.fc2$",
    # Patch merger
    r"^model\.visual\.merger$",
    r"^model\.visual\.merger\.ln_q$",
    r"^model\.visual\.merger\.mlp\.0$",  # Linear
    r"^model\.visual\.merger\.mlp\.1$",  # GELU
    r"^model\.visual\.merger\.mlp\.2$",  # Linear
    # Vision encoder top-level (to capture final vision output)
    r"^model\.visual$",
    # --- Text Decoder ---
    # Token embedding
    r"^model\.language_model\.embed_tokens$",
    # Decoder layer 0 — all sub-modules
    r"^model\.language_model\.layers\.0$",
    r"^model\.language_model\.layers\.0\.input_layernorm$",
    r"^model\.language_model\.layers\.0\.self_attn$",
    r"^model\.language_model\.layers\.0\.self_attn\.q_proj$",
    r"^model\.language_model\.layers\.0\.self_attn\.k_proj$",
    r"^model\.language_model\.layers\.0\.self_attn\.v_proj$",
    r"^model\.language_model\.layers\.0\.self_attn\.o_proj$",
    r"^model\.language_model\.layers\.0\.post_attention_layernorm$",
    r"^model\.language_model\.layers\.0\.mlp$",
    r"^model\.language_model\.layers\.0\.mlp\.gate_proj$",
    r"^model\.language_model\.layers\.0\.mlp\.up_proj$",
    r"^model\.language_model\.layers\.0\.mlp\.down_proj$",
    r"^model\.language_model\.layers\.0\.mlp\.act_fn$",
    # Final norm
    r"^model\.language_model\.norm$",
    # LM head
    r"^lm_head$",
]


def _should_hook(name: str) -> bool:
    """Check if a module name matches any of our hook patterns."""
    return any(re.match(p, name) for p in HOOK_PATTERNS)


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    """Convert a tensor to float32 numpy array on CPU."""
    return t.detach().float().cpu().numpy()


def _save_tensor(output_dir: str, name: str, suffix: str, tensor: torch.Tensor):
    """Save a single tensor as .npy file."""
    # Replace dots with underscores for filesystem-friendly names
    safe_name = name.replace(".", "__")
    filename = f"{safe_name}_{suffix}.npy"
    filepath = os.path.join(output_dir, filename)
    np.save(filepath, _to_numpy(tensor))


def _extract_tensors(data):
    """Extract tensors from various output formats (tensor, tuple, BaseModelOutput, etc.)."""
    if isinstance(data, torch.Tensor):
        return [("", data)]

    if isinstance(data, (tuple, list)):
        results = []
        for i, item in enumerate(data):
            if isinstance(item, torch.Tensor):
                results.append((f"_idx{i}", item))
        return results

    # Handle transformers model outputs (BaseModelOutputWithPast, etc.)
    if hasattr(data, "last_hidden_state") and data.last_hidden_state is not None:
        return [("", data.last_hidden_state)]

    if hasattr(data, "logits") and data.logits is not None:
        return [("", data.logits)]

    return []


def register_hooks(model, output_dir: str) -> list:
    """Register forward hooks on target modules.

    Returns a list of hook handles for cleanup.
    """
    handles = []

    for name, module in model.named_modules():
        if not _should_hook(name):
            continue

        def make_hook(mod_name):
            def hook_fn(module, input, output):
                # Save inputs
                if isinstance(input, tuple):
                    for i, inp in enumerate(input):
                        if isinstance(inp, torch.Tensor):
                            suffix = "input" if len(input) == 1 or i == 0 else f"input_idx{i}"
                            _save_tensor(output_dir, mod_name, suffix, inp)
                elif isinstance(input, torch.Tensor):
                    _save_tensor(output_dir, mod_name, "input", input)

                # Save outputs
                tensors = _extract_tensors(output)
                if len(tensors) == 1:
                    _save_tensor(output_dir, mod_name, "output", tensors[0][1])
                else:
                    for suffix, tensor in tensors:
                        _save_tensor(output_dir, mod_name, f"output{suffix}", tensor)

            return hook_fn

        handle = module.register_forward_hook(make_hook(name))
        handles.append(handle)
        print(f"  Hooked: {name}")

    return handles


def run_inference(image_path: str, output_dir: str):
    """Load model, register hooks, run inference, save activations."""
    os.makedirs(output_dir, exist_ok=True)

    print("Loading model...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype=torch.float32,
        device_map="auto",
    )
    model.eval()

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

    print("\nRegistering hooks...")
    handles = register_hooks(model, output_dir)
    print(f"  Total hooks: {len(handles)}")

    # Prepare input
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    vision_info = process_vision_info(messages)
    if len(vision_info) == 3:
        image_inputs, video_inputs, _ = vision_info
    else:
        image_inputs, video_inputs = vision_info
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)

    # Also save the raw inputs for reference
    for key, val in inputs.items():
        if isinstance(val, torch.Tensor):
            safe_key = key.replace(".", "__")
            np.save(os.path.join(output_dir, f"_input_{safe_key}.npy"), _to_numpy(val))
            print(f"  Saved input: _input_{safe_key}  shape={val.shape}")

    # Run single forward pass (no generation loop, just one forward)
    print("\nRunning forward pass...")
    with torch.no_grad():
        outputs = model(**inputs)

    # Save final logits
    if hasattr(outputs, "logits") and outputs.logits is not None:
        np.save(os.path.join(output_dir, "_final_logits.npy"), _to_numpy(outputs.logits))
        print(f"  Saved _final_logits  shape={outputs.logits.shape}")

    # Cleanup hooks
    for h in handles:
        h.remove()

    print(f"\nDone! Activations saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Dump Qwen2-VL activations for numpy validation")
    parser.add_argument("--output-dir", default="activations", help="Directory to save .npy files")
    parser.add_argument("--image", default="./demo.jpeg", help="Path to input image")
    args = parser.parse_args()

    run_inference(args.image, args.output_dir)


if __name__ == "__main__":
    main()
