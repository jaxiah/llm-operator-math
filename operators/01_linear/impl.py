"""01 — 线性变换 (Linear Transform): y = xW^T + b

用纯 NumPy 实现线性变换，并用 Qwen2-VL 的真实权重和激活值验证。
"""

import numpy as np
from huggingface_hub import hf_hub_download

from e2e.validate import validate, load_activation

DUMP_DIR = "activations"
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"


# ---------------------------------------------------------------------------
# 核心算子
# ---------------------------------------------------------------------------
def linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
    """线性变换: y = x @ weight.T + bias"""
    y = x @ weight.T
    if bias is not None:
        y = y + bias
    return y


# ---------------------------------------------------------------------------
# 权重加载工具
# ---------------------------------------------------------------------------
def load_safetensors_weights(keys: list[str]) -> dict[str, np.ndarray]:
    """从 HuggingFace 缓存加载指定的 safetensors 权重张量。

    使用 safetensors.safe_open 逐个读取张量，通过 torch 处理 bfloat16 转换。
    """
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
                    tensor = f.get_tensor(key).float().numpy()
                    result[key] = tensor.astype(np.float32)
                    remaining.discard(key)

    if remaining:
        raise KeyError(f"未找到以下权重: {remaining}")
    return result


# ---------------------------------------------------------------------------
# 验证
# ---------------------------------------------------------------------------
def validate_vision_fc1() -> bool:
    """验证 Vision Encoder Block 0 的 MLP fc1 线性层。"""
    print("\n=== Vision MLP fc1 (14308, 1280) -> (14308, 5120) ===")

    x = load_activation(DUMP_DIR, "model__visual__blocks__0__mlp__fc1_input")
    expected = load_activation(DUMP_DIR, "model__visual__blocks__0__mlp__fc1_output")

    weights = load_safetensors_weights(["visual.blocks.0.mlp.fc1.weight", "visual.blocks.0.mlp.fc1.bias"])
    W = weights["visual.blocks.0.mlp.fc1.weight"]
    b = weights["visual.blocks.0.mlp.fc1.bias"]

    actual = linear(x, W, b)
    return validate("vision_fc1_linear", actual, expected, atol=1e-4, rtol=1e-4)


def validate_text_gate_proj() -> bool:
    """验证 Text Decoder Layer 0 的 gate_proj 线性层（无偏置）。"""
    print("\n=== Text gate_proj (1, 3602, 1536) -> (1, 3602, 8960) ===")

    x = load_activation(DUMP_DIR, "model__language_model__layers__0__mlp__gate_proj_input")
    expected = load_activation(DUMP_DIR, "model__language_model__layers__0__mlp__gate_proj_output")

    weights = load_safetensors_weights(["model.layers.0.mlp.gate_proj.weight"])
    W = weights["model.layers.0.mlp.gate_proj.weight"]

    actual = linear(x, W, bias=None)
    return validate("text_gate_proj_linear", actual, expected, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    results = [
        validate_vision_fc1(),
        validate_text_gate_proj(),
    ]
    print(f"\n{'='*60}")
    print(f"线性变换验证: {sum(results)}/{len(results)} 通过")
    if not all(results):
        raise SystemExit(1)
