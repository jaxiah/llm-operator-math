# Qwen2-VL 算子数学手册

## 项目简介

本项目旨在**逐算子理解 Qwen2-VL-2B-Instruct 的推理过程**。每个算子用纯 NumPy
从零实现，并与 PyTorch 真实激活值逐层比对验证，确保数学理解完全正确。

> 目标不是高性能推理框架，而是一份可执行的"数学笔记"——
> 读完每个算子的实现，你就能手算出模型的每一步输出。

---

## 架构概览

```
                        Qwen2-VL-2B-Instruct 架构
 ═══════════════════════════════════════════════════════════════

 ┌─────────────────────────────────────────────────────────────┐
 │                    Vision Encoder (ViT)                     │
 │                                                             │
 │  pixel_values ──► Conv3d Patch Embed (05) ──► patch embeds  │
 │                        (14308, 1176) → (14308, 1280)        │
 │                                                             │
 │       ┌──────────────────────────────────────┐              │
 │       │     Vision Block × 32 (16)           │              │
 │       │  ┌───────────────────────────────┐   │              │
 │       │  │ LayerNorm (03)                │   │              │
 │       │  │ Vision Attention (07)+2D RoPE │   │              │
 │       │  │ + Residual (13)               │   │              │
 │       │  │ LayerNorm (03)                │   │              │
 │       │  │ Vision MLP (11): FC1→QuickGELU│   │              │
 │       │  │   (08)→FC2                    │   │              │
 │       │  │ + Residual (13)               │   │              │
 │       │  └───────────────────────────────┘   │              │
 │       └──────────────────────────────────────┘              │
 │                                                             │
 │  patch embeds ──► Patch Merger (15) ──► vision tokens       │
 │      LN(03) → 空间合并 → Linear→GELU(09)→Linear             │
 │        (14308, 1280) → (3577, 1536)                         │
 └────────────────────────┬────────────────────────────────────┘
                          │ vision tokens
                          ▼
 ┌─────────────────────────────────────────────────────────────┐
 │                   Text Decoder (LLM)                        │
 │                                                             │
 │  input_ids ──► Token Embedding (14) ──► text embeddings     │
 │                  + vision token 替换                        │
 │                     (1, 3602, 1536)                         │
 │                                                             │
 │       ┌──────────────────────────────────────┐              │
 │       │     Decoder Layer × 28 (17)          │              │
 │       │  ┌───────────────────────────────┐   │              │
 │       │  │ RMSNorm (04)                  │   │              │
 │       │  │ GQA Attention (07)+3D M-RoPE  │   │              │
 │       │  │   (06) + Causal Mask          │   │              │
 │       │  │ + Residual (13)               │   │              │
 │       │  │ RMSNorm (04)                  │   │              │
 │       │  │ Gated MLP / SwiGLU (12):      │   │              │
 │       │  │   gate+up→SiLU(10)→down       │   │              │
 │       │  │ + Residual (13)               │   │              │
 │       │  └───────────────────────────────┘   │              │
 │       └──────────────────────────────────────┘              │
 │                                                             │
 │  hidden ──► Final RMSNorm (04) ──► LM Head (18) ──► logits  │
 │                          (1, 3602, 151936)                  │
 └─────────────────────────────────────────────────────────────┘
```

---

## 算子执行顺序

以单次前向推理为例，算子按以下顺序执行：

### Vision Encoder

1. **Conv3d Patch Embedding (05)** — 图像像素 → patch embeddings

   - 将 `pixel_values` 按 $(2 \times 14 \times 14)$ 窗口展平，投影到 $d=1280$
   - 当 stride = kernel_size 时退化为线性投影

2. **Vision Block × 32 (16)** — 代表层: block 0

   - **LayerNorm (03)** — 第一个归一化
   - **Vision Multi-Head Attention (07)** + **2D RoPE (06)** — 16 头自注意力 + 窗口注意力
   - **Residual Connection (13)** — $x' = x + \text{Attn}(\text{LN}(x))$
   - **LayerNorm (03)** — 第二个归一化
   - **Vision MLP (11)**: Linear (01) → QuickGELU (08) → Linear (01) — $1280 \to 5120 \to 1280$
   - **Residual Connection (13)** — $x'' = x' + \text{MLP}(\text{LN}(x'))$

3. **Patch Merger (15)** — 视觉 token 到文本空间的桥接
   - LayerNorm (03) → 空间合并 (2×2 patch 拼接) → Linear (01) → GELU (09) → Linear (01)
   - $(14308, 1280) \to (3577, 1536)$

### Text Decoder

4. **Token Embedding (14)** — token ids → embeddings + vision token 替换

   - 查表 $E[\text{token\_id}]$ 得到文本嵌入，vision token 位置替换为 Patch Merger 输出

5. **Decoder Layer × 28 (17)** — 代表层: layer 0

   - **RMSNorm (04)** — 输入归一化（无 bias，无中心化）
   - **Grouped Query Attention (07)** + **3D M-RoPE (06)** — 12 个 Q 头 / 2 个 KV 头 + 多模态旋转位置编码
   - **Residual Connection (13)** — $h = x + \text{Attn}(\text{RMSNorm}(x))$
   - **RMSNorm (04)** — 注意力后归一化
   - **Gated MLP / SwiGLU (12)**: gate_proj + up_proj → SiLU (10) → element-wise multiply → down_proj
     - $1536 \to 8960 \to 1536$
   - **Residual Connection (13)** — $y = h + \text{MLP}(\text{RMSNorm}(h))$

6. **Final RMSNorm (04)** — 最终归一化

7. **LM Head (18)** — Linear → logits
   - $\text{logits} = \text{RMSNorm}(x) \cdot W_{\text{lm}}^T$，输出维度 $V = 151936$
   - 权重与 Token Embedding 共享 (weight tying)

---

## 目录索引

| 目录                      | 算子                      | 说明                                                                   |
| ------------------------- | ------------------------- | ---------------------------------------------------------------------- |
| `01_linear/`              | Linear                    | 矩阵乘法 $y = xW^T + b$                                                |
| `02_softmax/`             | Softmax                   | 指数归一化 $\text{softmax}(x)_i = e^{x_i} / \sum e^{x_j}$              |
| `03_layer_norm/`          | Layer Normalization       | 层归一化 $(x - \mu) / \sqrt{\sigma^2 + \epsilon} \cdot \gamma + \beta$ |
| `04_rms_norm/`            | RMS Normalization         | 均方根归一化 $x / \sqrt{\text{mean}(x^2) + \epsilon} \cdot \gamma$     |
| `05_conv3d_patch_embed/`  | Conv3d Patch Embedding    | 3D 卷积图像块嵌入（stride=kernel → 线性投影）                          |
| `06_rotary_pos_embed/`    | Rotary Position Embedding | 旋转位置编码：1D RoPE / 2D Vision RoPE / 3D M-RoPE                     |
| `07_attention/`           | Attention                 | 缩放点积注意力 + GQA + KV repeat                                       |
| `08_quickgelu/`           | QuickGELU                 | 快速 GELU 近似 $x \cdot \sigma(1.702x)$                                |
| `09_gelu/`                | GELU                      | 高斯误差线性单元（精确版 + tanh 近似版）                               |
| `10_silu/`                | SiLU (Swish)              | $x \cdot \sigma(x)$，用于 SwiGLU 门控                                  |
| `11_vision_mlp/`          | Vision MLP                | 视觉前馈网络 FC1 → QuickGELU → FC2                                     |
| `12_gated_mlp/`           | Gated MLP (SwiGLU)        | 门控前馈网络 $(\text{SiLU}(xW_g) \odot xW_u) W_d$                      |
| `13_residual_connection/` | Residual Connection       | 残差连接 $y = x + F(x)$                                                |
| `14_token_embedding/`     | Token Embedding           | 词嵌入查找表 $E[\text{id}]$                                            |
| `15_patch_merger/`        | Patch Merger              | 补丁合并: LN → 空间合并 → MLP                                          |
| `16_vision_block/`        | Vision Block              | 视觉 Transformer 块: LN → Attn → Residual → LN → MLP → Residual        |
| `17_decoder_layer/`       | Decoder Layer             | 解码器层: RMSNorm → Attn → Residual → RMSNorm → GatedMLP → Residual    |
| `18_lm_head/`             | LM Head                   | 语言模型头: RMSNorm → Linear → logits                                  |

---

## 模型关键参数

> 来自 `Qwen/Qwen2-VL-2B-Instruct` 的 `config.json`

### Vision Encoder

| 参数                 | 值                |
| -------------------- | ----------------- |
| embed_dim            | 1280              |
| num_heads            | 16                |
| head_dim             | 80                |
| MLP intermediate dim | 5120 (= 1280 × 4) |
| num_blocks           | 32                |
| patch_size           | 14 × 14           |
| temporal_patch_size  | 2                 |
| spatial_merge_size   | 2                 |

### Text Decoder

| 参数                    | 值                |
| ----------------------- | ----------------- |
| hidden_size             | 1536              |
| num_attention_heads     | 12                |
| num_key_value_heads     | 2 (GQA, 6:1 比率) |
| head_dim                | 128               |
| MLP intermediate dim    | 8960              |
| num_hidden_layers       | 28                |
| vocab_size              | 151936            |
| max_position_embeddings | 32768             |
| rms_norm_eps            | 1e-6              |
| rope_theta              | 1000000.0         |

---

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
cd llm-operator-math

# 安装依赖（需要 numpy；验证模型权重还需要 huggingface_hub, safetensors）
pip install numpy huggingface_hub safetensors torch
```

### 2. 生成激活值转储

```bash
# 运行推理并保存中间激活值到 activations/ 目录
python -m e2e.dump_activations --output-dir activations
```

### 3. 运行单个算子验证

```bash
# 示例：验证 Layer Normalization
python -m operators.03_layer_norm.impl

# 示例：验证完整 Vision Block
python -m operators.16_vision_block.impl

# 示例：验证 LM Head
python -m operators.18_lm_head.impl
```

### 4. 运行端到端验证

```bash
# 按推理执行顺序验证所有算子
python -m e2e.run_numpy_e2e

# 指定自定义激活值目录
python -m e2e.run_numpy_e2e --dump-dir path/to/activations
```

端到端验证器会按推理流程顺序执行所有算子验证，最后输出汇总报告：

```
  [✓] conv3d_patch_embed
  [✓] vision_block0_norm1
  [✓] vision_block0_attn_residual
  ...
  [✓] full_lm_head_last_token

  18/18 passed
```
