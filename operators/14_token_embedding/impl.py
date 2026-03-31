"""14 — 词嵌入 (Token Embedding)

用纯 NumPy 实现 token embedding 查找，并用 Qwen2-VL 的真实权重和激活值验证。
"""

import os

import numpy as np

from e2e.validate import validate, load_activation

DUMP_DIR = "activations"
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"


# ---------------------------------------------------------------------------
# 核心算子
# ---------------------------------------------------------------------------
def token_embedding(token_ids: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """词嵌入：查找表索引。

    Args:
        token_ids: 整数 token ID，任意形状
        weight: (V, d) 嵌入矩阵

    Returns:
        (*token_ids.shape, d) 嵌入向量
    """
    return weight[token_ids]


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
def validate_token_embedding() -> bool:
    """验证 token embedding 查找。"""
    print("\n=== Token Embedding (1, 3602) -> (1, 3602, 1536) ===")

    # 加载激活值（token IDs 存储为 float32，需转为 int）
    token_ids = load_activation(DUMP_DIR, "model__language_model__embed_tokens_input").astype(np.int64)
    expected = load_activation(DUMP_DIR, "model__language_model__embed_tokens_output")

    # 加载嵌入权重
    weights = load_weights(["model.embed_tokens.weight"])
    E = weights["model.embed_tokens.weight"]  # (151936, 1536)

    # 嵌入查找
    actual = token_embedding(token_ids, E)

    return validate("token_embedding", actual, expected, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    results = [
        validate_token_embedding(),
    ]
    print(f"\n{'='*60}")
    print(f"Token Embedding 验证: {sum(results)}/{len(results)} 通过")
    if not all(results):
        raise SystemExit(1)
