# Decoder Layer（解码器层）

## 前置知识

在阅读本节之前，请确保理解：

- [RMS Normalization](../04_rms_norm/README.md)：$\text{RMSNorm}(x) = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \cdot \gamma$
- [Gated MLP / SwiGLU](../12_gated_mlp/README.md)：$(\text{SiLU}(xW_g^T) \odot xW_u^T) W_d^T$
- [残差连接](../13_residual_connection/README.md)：$y = x + F(x)$
- [注意力机制](../07_attention/README.md)：Scaled Dot-Product Attention

---

## 从单个组件到完整的层

前面的章节中，我们分别实现了 RMSNorm、Self-Attention、Gated MLP 和残差连接。现在，我们把它们 **组装** 成一个完整的 Transformer 解码器层。

这就像造汽车：之前我们分别制造了发动机、变速箱、底盘和车轮，现在要把它们组装在一起。

---

## 数学定义

### 完整的 Decoder Layer

给定输入 $x \in \mathbb{R}^{B \times T \times d}$，一个 Decoder Layer 的计算流程为：

$$
\begin{aligned}
\hat{x} &= \text{RMSNorm}(x, \gamma_1) & \text{(Pre-Norm)} \\
a &= \text{SelfAttn}(\hat{x}) & \text{(GQA + RoPE + 因果掩码)} \\
h &= x + a & \text{(第一个残差连接)} \\
\hat{h} &= \text{RMSNorm}(h, \gamma_2) & \text{(Pre-Norm)} \\
m &= \text{GatedMLP}(\hat{h}) & \text{(SwiGLU)} \\
y &= h + m & \text{(第二个残差连接)}
\end{aligned}
$$

其中：
- $\gamma_1$ 是 input_layernorm 的缩放权重
- $\gamma_2$ 是 post_attention_layernorm 的缩放权重
- $y$ 是层输出，形状与输入相同

### Pre-Norm vs Post-Norm

Qwen2-VL 使用 **Pre-Norm**（先归一化再做子层运算），而非原始 Transformer 的 Post-Norm：

| 方案 | 公式 | 特点 |
|------|------|------|
| Post-Norm | $y = \text{Norm}(x + F(x))$ | 原始 Transformer |
| **Pre-Norm** | $y = x + F(\text{Norm}(x))$ | **Qwen2-VL 使用** |

Pre-Norm 的优势：
- 残差路径上没有归一化操作，梯度可以 **无损地** 直接回传
- 训练更稳定，不需要 warm-up
- 大模型几乎都采用 Pre-Norm

---

## 数据流

在 Qwen2-VL 文本解码器中（以 layer 0 为例）：

```
输入 x: (1, 3602, 1536)
    │
    ▼
┌─────────────────────────────────────────┐
│ Step 1: RMSNorm (input_layernorm)       │
│   x̂ = RMSNorm(x, γ₁)                   │
│   (1, 3602, 1536) → (1, 3602, 1536)    │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Step 2: Self-Attention (GQA)            │
│   12 heads, 2 KV heads, head_dim=128   │
│   含 RoPE 位置编码 + 因果掩码            │
│   (1, 3602, 1536) → (1, 3602, 1536)    │
└─────────────────────────────────────────┘
    │                         │
    ▼                         │ (x 的残差路径)
┌─────────────────────────┐   │
│ Step 3: 残差连接         │◄──┘
│   h = x + attn_out       │
│   (1, 3602, 1536)        │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Step 4: RMSNorm (post_attention_ln)     │
│   ĥ = RMSNorm(h, γ₂)                   │
│   (1, 3602, 1536) → (1, 3602, 1536)    │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Step 5: Gated MLP (SwiGLU)              │
│   gate_proj: (1536 → 8960)              │
│   up_proj:   (1536 → 8960)              │
│   SiLU + 门控                            │
│   down_proj: (8960 → 1536)              │
│   (1, 3602, 1536) → (1, 3602, 1536)    │
└─────────────────────────────────────────┘
    │                         │
    ▼                         │ (h 的残差路径)
┌─────────────────────────┐   │
│ Step 6: 残差连接         │◄──┘
│   y = h + mlp_out        │
│   (1, 3602, 1536)        │
└─────────────────────────┘
    │
    ▼
输出 y: (1, 3602, 1536)
```

关键观察：**输入和输出形状完全相同**。28 个 Decoder Layer 可以串联堆叠。

---

## 逐步数值示例

假设 $d=4$，单 token 输入 $x = [1.0, -0.5, 0.3, 0.8]$。

### Step 1: Input LayerNorm

$$
\text{RMS}(x) = \sqrt{\frac{1.0^2 + 0.5^2 + 0.3^2 + 0.8^2}{4}} = \sqrt{\frac{1.98}{4}} = \sqrt{0.495} \approx 0.7036
$$

$$
\hat{x} = \frac{x}{0.7036} \cdot \gamma_1
$$

假设 $\gamma_1 = [1, 1, 1, 1]$，则 $\hat{x} \approx [1.421, -0.711, 0.426, 1.137]$。

### Step 2: Self-Attention

Self-Attention 将 $\hat{x}$ 投影到 Q、K、V 空间，计算注意力分数，输出 $a$。
（由于单 token 的因果注意力只有自身，相当于 $\hat{x}$ 经过 $W_V W_O$ 的线性变换。）

### Step 3: 第一个残差连接

$$
h = x + a = [1.0, -0.5, 0.3, 0.8] + a
$$

残差连接确保原始信息不丢失。

### Step 4–6: 对称地重复

Post-Attention LayerNorm → Gated MLP → 第二个残差连接，结构与 Step 1–3 对称。

---

## 为什么每一层形状不变？

Decoder Layer 的一个优雅设计是 **输入和输出维度完全相同** $(B, T, d)$。这带来了重要的好处：

1. **层可堆叠**：28 层可以直接串联，无需任何维度适配
2. **残差路径畅通**：梯度可以从最后一层直接流到第一层
3. **统一接口**：每一层的参数结构相同，只是权重值不同

$$
x_0 \xrightarrow{\text{Layer}_0} x_1 \xrightarrow{\text{Layer}_1} x_2 \xrightarrow{\text{Layer}_2} \cdots \xrightarrow{\text{Layer}_{27}} x_{28}
$$

---

## 参数统计

Qwen2-VL-2B 文本解码器每层参数：

| 组件 | 权重 | 形状 | 参数量 |
|------|------|------|--------|
| input_layernorm | $\gamma_1$ | $(1536,)$ | 1,536 |
| self_attn.q_proj | $W_Q, b_Q$ | $(1536, 1536) + (1536,)$ | 2,361,344 |
| self_attn.k_proj | $W_K, b_K$ | $(256, 1536) + (256,)$ | 393,472 |
| self_attn.v_proj | $W_V, b_V$ | $(256, 1536) + (256,)$ | 393,472 |
| self_attn.o_proj | $W_O$ | $(1536, 1536)$ | 2,359,296 |
| post_attention_layernorm | $\gamma_2$ | $(1536,)$ | 1,536 |
| mlp.gate_proj | $W_g$ | $(8960, 1536)$ | 13,762,560 |
| mlp.up_proj | $W_u$ | $(8960, 1536)$ | 13,762,560 |
| mlp.down_proj | $W_d$ | $(1536, 8960)$ | 13,762,560 |
| **合计** | | | **~46.8M** |

28 层合计约 **1.31B** 参数，占模型总参数的大部分。

---

## 验证策略

完整的 Self-Attention 包含 GQA + RoPE + 因果掩码，实现复杂度高。我们采用 **分步验证 + 中间值替代** 的策略：

1. ✅ 验证 Input LayerNorm：已在 [issue-004](../04_rms_norm/) 中证明
2. 🔄 Self-Attention：使用 dump 的 `attn_output` 替代
3. ✅ 验证第一个残差连接
4. ✅ 验证 Post-Attention LayerNorm
5. ✅ 验证 Gated MLP：已在 [issue-012](../12_gated_mlp/) 中证明
6. ✅ 验证第二个残差连接
7. ✅ 验证完整层（attention 步骤使用 dump 值）

---

## NumPy 实现

```python
import numpy as np

def decoder_layer(x, input_ln_w, post_ln_w, gate_w, up_w, down_w, attn_output):
    """Decoder Layer: RMSNorm → Attn → Residual → RMSNorm → GatedMLP → Residual"""
    # Step 1: Input LayerNorm
    x_normed = rms_norm(x, input_ln_w)
    # Step 2: Self-Attention
    attn_out = attn_output  # 使用预计算值
    # Step 3: 残差连接
    hidden = x + attn_out
    # Step 4: Post-Attention LayerNorm
    hidden_normed = rms_norm(hidden, post_ln_w)
    # Step 5: Gated MLP
    mlp_out = gated_mlp(hidden_normed, gate_w, up_w, down_w)
    # Step 6: 残差连接
    output = hidden + mlp_out
    return output
```

---

## 在 Qwen2-VL 中的位置

```
Qwen2-VL 文本解码器
├── Token Embedding
├── Decoder Layer × 28  ← 这里（每层结构相同，权重不同）
│   ├── RMSNorm (input_layernorm)
│   ├── Self-Attention (GQA + RoPE)
│   ├── 残差连接
│   ├── RMSNorm (post_attention_layernorm)
│   ├── Gated MLP (SwiGLU)
│   └── 残差连接
├── Final RMSNorm
└── LM Head
```
