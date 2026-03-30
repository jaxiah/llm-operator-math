# LM Head（语言模型输出头）

## 前置知识

在阅读本节之前，请确保理解：

- [RMS Normalization](../04_rms_norm/README.md)：$\text{RMSNorm}(x) = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \cdot \gamma$
- [线性变换](../01_linear/README.md)：矩阵乘法 $y = xW^T$
- [Decoder Layer](../17_decoder_layer/README.md)：Transformer 解码器层

---

## 从隐藏状态到词表概率

经过 28 层 Decoder Layer 的处理后，我们得到了丰富的隐藏表示 $x \in \mathbb{R}^{B \times T \times 1536}$。但模型最终需要回答一个问题：**下一个 token 应该是词表中的哪一个？**

LM Head 就是完成这最后一步的组件：将 1536 维的隐藏状态映射到 151936 维的词表空间。

---

## 数学定义

### 完整公式

$$
\text{logits} = \text{RMSNorm}(x, \gamma) \cdot W_{\text{lm}}^T
$$

其中：
- $x \in \mathbb{R}^{B \times T \times d}$ 是最后一个 Decoder Layer 的输出
- $\gamma \in \mathbb{R}^{d}$ 是最终 RMSNorm 的缩放权重
- $W_{\text{lm}} \in \mathbb{R}^{V \times d}$ 是 LM Head 的投影矩阵
- $V = 151936$ 是词表大小
- logits $\in \mathbb{R}^{B \times T \times V}$

### 分步展开

**Step 1: Final RMSNorm**

$$
\hat{x} = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \cdot \gamma
$$

这一步归一化确保不同位置的隐藏状态在相似的数值范围内，为线性投影提供稳定的输入。

**Step 2: 线性投影**

$$
\text{logits} = \hat{x} \cdot W_{\text{lm}}^T
$$

这是一个没有偏置的线性变换，将 $d=1536$ 维映射到 $V=151936$ 维。

---

## 数据流

```
最后一个 Decoder Layer 的输出
    │
    ▼
x: (1, 3602, 1536)
    │
    ▼
┌─────────────────────────────────────┐
│ Final RMSNorm                       │
│   x̂ = RMSNorm(x, γ)                │
│   (1, 3602, 1536) → (1, 3602, 1536)│
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Linear Projection (LM Head)         │
│   logits = x̂ @ W_lm.T              │
│   (1, 3602, 1536) × (1536, 151936) │
│   → (1, 3602, 151936)              │
└─────────────────────────────────────┘
    │
    ▼
logits: (1, 3602, 151936)
    │
    ▼
argmax → 预测的下一个 token ID
```

### 计算规模

最后的矩阵乘法是模型中 **最大的单次运算之一**：

$$
\text{FLOPs} = B \times T \times d \times V = 1 \times 3602 \times 1536 \times 151936 \approx 8.4 \times 10^{11}
$$

输出张量大小：$1 \times 3602 \times 151936 \times 4 \text{ bytes} \approx 2.05 \text{ GB}$（float32）

---

## 逐步数值示例

假设 $d=4$，$V=6$（一个微型词表），最后一个 token 的隐藏状态 $x = [0.8, -1.2, 0.5, 0.3]$。

### Step 1: RMSNorm

$$
\text{RMS}(x) = \sqrt{\frac{0.8^2 + 1.2^2 + 0.5^2 + 0.3^2}{4}} = \sqrt{\frac{2.42}{4}} = \sqrt{0.605} \approx 0.778
$$

假设 $\gamma = [1, 1, 1, 1]$：

$$
\hat{x} = \frac{[0.8, -1.2, 0.5, 0.3]}{0.778} \approx [1.029, -1.543, 0.643, 0.386]
$$

### Step 2: 线性投影

假设 $W_{\text{lm}}$ 的 6 行分别代表 6 个词的嵌入向量：

$$
\text{logits} = \hat{x} \cdot W_{\text{lm}}^T = [l_0, l_1, l_2, l_3, l_4, l_5]
$$

每个 $l_i$ 是 $\hat{x}$ 与第 $i$ 个词嵌入向量的内积，衡量"当前状态与该词的相似度"。

### Step 3: 解码

$$
\text{predicted\_token} = \arg\max_i \, \text{logits}[i]
$$

logits 最大值对应的词就是模型的预测。

---

## Weight Tying（权重绑定）

Qwen2-VL 使用了 **weight tying** 技术：

$$
W_{\text{lm}} = W_{\text{embed}}
$$

即 LM Head 的投影矩阵 **就是** Token Embedding 的嵌入矩阵。这意味着：

- **嵌入层**（Embedding）：token ID → 向量，用 $W_{\text{embed}}$ 做查表
- **LM Head**：向量 → logits，用 $W_{\text{embed}}^T$ 做投影

$$
\text{logits}_i = \hat{x} \cdot w_i
$$

其中 $w_i$ 是第 $i$ 个 token 的嵌入向量。**logits 就是归一化后的隐藏状态与每个词嵌入的内积（余弦相似度的未归一化版本）**。

### 为什么 weight tying 有效？

1. **节省参数**：$V \times d = 151936 \times 1536 \approx 233\text{M}$ 参数只存一份
2. **语义一致性**：输入和输出共享同一个语义空间
3. **正则化效果**：嵌入需要同时服务于输入和输出，学到更通用的表示

---

## 从 logits 到 token

LM Head 输出的 logits 还不是概率。要生成文本，还需要：

$$
P(\text{token}_i) = \frac{e^{l_i / \tau}}{\sum_{j=1}^{V} e^{l_j / \tau}}
$$

其中 $\tau$ 是温度参数。但这一步属于采样策略（greedy / top-k / top-p），不在 LM Head 的范围内。

LM Head 的职责很清晰：**将隐藏状态投影到词表空间，得到未归一化的分数**。

---

## NumPy 实现

```python
import numpy as np

def lm_head(x, norm_weight, lm_weight):
    """LM Head: Final RMSNorm → Linear → logits"""
    hidden = rms_norm(x, norm_weight)
    logits = hidden @ lm_weight.T
    return logits
```

仅两步操作，但涉及的矩阵乘法规模巨大。

---

## 在 Qwen2-VL 中的位置

```
Qwen2-VL 文本解码器
├── Token Embedding        (token ID → 1536 维向量)
├── Decoder Layer × 28     (1536 → 1536，逐层精炼)
├── Final RMSNorm          (1536 → 1536，归一化)  ← 这里
└── LM Head                (1536 → 151936，logits) ← 这里
```

权重参数：
- `model.norm.weight` — 形状 $(1536,)$，最终 RMSNorm
- `lm_head.weight` — 与 `model.embed_tokens.weight` 绑定，形状 $(151936, 1536)$

参数量（如不计 tying）：$151936 \times 1536 = 233{,}373{,}696 \approx 233\text{M}$
