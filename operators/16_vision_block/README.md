# Vision Transformer Block（视觉编码器块）

## 前置知识

在阅读本节之前，请确保理解：

- [Layer Normalization](../03_layer_norm/README.md)：$\text{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$
- [注意力机制](../07_attention/README.md)：Scaled Dot-Product Attention
- [QuickGELU](../08_quickgelu/README.md)：$\text{QuickGELU}(x) = x \cdot \sigma(1.702x)$
- [Vision MLP](../11_vision_mlp/README.md)：$\text{MLP}(x) = \text{FC2}(\text{QuickGELU}(\text{FC1}(x)))$
- [残差连接](../13_residual_connection/README.md)：$y = x + F(x)$

---

## 什么是 Vision Block？

Vision Block 是视觉编码器的**基本构建单元**。Qwen2-VL 的视觉编码器由多个 Vision Block 堆叠而成，每个 block 将输入的 patch 序列变换为更高级的特征表示。

一个 Vision Block 包含两个子层，每个子层都采用 **Pre-Norm + 残差连接** 的结构：

1. **Self-Attention 子层**：让 patch 之间交换信息
2. **MLP 子层**：对每个 patch 独立地进行非线性变换

---

## 数学定义

### Pre-Norm Transformer Block

给定输入 $x \in \mathbb{R}^{n \times d}$，Vision Block 的计算流程为：

$$
x' = x + \text{Attention}\big(\text{LN}_1(x)\big)
$$

$$
y = x' + \text{MLP}\big(\text{LN}_2(x')\big)
$$

其中：
- $\text{LN}_1, \text{LN}_2$ 是两个独立的 Layer Normalization
- $\text{Attention}$ 是多头自注意力（Multi-Head Attention）
- $\text{MLP}$ 是两层前馈网络（FC1 → QuickGELU → FC2）

### 展开每一步

**步骤 1：Pre-Norm（第一次归一化）**

$$
\hat{x} = \text{LN}_1(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma_1 + \beta_1
$$

**步骤 2：Self-Attention**

$$
\text{attn\_out} = \text{Attention}(\hat{x})
$$

在 Qwen2-VL 中，视觉注意力使用了**窗口注意力（Windowed Attention）**和**旋转位置编码（RoPE）**，其完整实现较为复杂。核心思想仍然是缩放点积注意力：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

**步骤 3：第一次残差连接**

$$
x' = x + \text{attn\_out}
$$

**步骤 4：Pre-Norm（第二次归一化）**

$$
\hat{x}' = \text{LN}_2(x')
$$

**步骤 5：MLP（前馈网络）**

$$
\text{mlp\_out} = \text{FC2}\big(\text{QuickGELU}(\text{FC1}(\hat{x}'))\big)
$$

**步骤 6：第二次残差连接**

$$
y = x' + \text{mlp\_out}
$$

---

## 数据流

在 Qwen2-VL 视觉编码器 Block 0 中：

```
输入 x: (14308, 1280)
    │
    ▼
LayerNorm₁: γ₁, β₁          ← norm1
    │
    ▼
x_normed: (14308, 1280)
    │
    ▼
Multi-Head Attention          ← qkv → 窗口注意力 + RoPE → proj
    │
    ▼
attn_out: (14308, 1280)
    │
    ▼
残差连接: x' = x + attn_out
    │
    ▼
x': (14308, 1280)
    │
    ▼
LayerNorm₂: γ₂, β₂          ← norm2
    │
    ▼
x'_normed: (14308, 1280)
    │
    ▼
MLP: FC1(5120,1280) → QuickGELU → FC2(1280,5120)
    │
    ▼
mlp_out: (14308, 1280)
    │
    ▼
残差连接: y = x' + mlp_out
    │
    ▼
输出 y: (14308, 1280)
```

---

## 为什么使用 Pre-Norm？

Transformer 有两种归一化方式：

| 方式 | 公式 | 特点 |
|------|------|------|
| **Post-Norm** | $y = \text{LN}(x + F(x))$ | 原始 Transformer 论文 |
| **Pre-Norm** | $y = x + F(\text{LN}(x))$ | 训练更稳定 |

Pre-Norm 的优势：
1. **梯度流更畅通**：残差连接直接传递梯度，不经过 LayerNorm
2. **训练更稳定**：归一化后的输入使得子层输出尺度可控
3. **深层网络更容易训练**：现代大模型普遍采用 Pre-Norm

---

## 残差连接的作用

残差连接解决深层网络的**梯度消失**问题：

$$
y = x + F(x)
$$

- 如果 $F(x)$ 学到的是"修正量"，则 $y$ 是在 $x$ 基础上的微调
- 梯度 $\frac{\partial y}{\partial x} = I + \frac{\partial F}{\partial x}$，恒等映射 $I$ 保证梯度不会消失
- 多个 block 堆叠时，每个 block 只需学习"增量变化"

---

## 参数量分析

Vision Block 0 的参数：

| 组件 | 权重形状 | 参数量 |
|------|---------|--------|
| norm1.weight | $(1280,)$ | 1,280 |
| norm1.bias | $(1280,)$ | 1,280 |
| attn.qkv.weight | $(3840, 1280)$ | 4,915,200 |
| attn.qkv.bias | $(3840,)$ | 3,840 |
| attn.proj.weight | $(1280, 1280)$ | 1,638,400 |
| attn.proj.bias | $(1280,)$ | 1,280 |
| norm2.weight | $(1280,)$ | 1,280 |
| norm2.bias | $(1280,)$ | 1,280 |
| mlp.fc1.weight | $(5120, 1280)$ | 6,553,600 |
| mlp.fc1.bias | $(5120,)$ | 5,120 |
| mlp.fc2.weight | $(1280, 5120)$ | 6,553,600 |
| mlp.fc2.bias | $(1280,)$ | 1,280 |
| **合计** | | **$\approx 19.7\text{M}$** |

注意 QKV 权重是 $3 \times 1280 = 3840$ 维，因为 Q、K、V 三个投影矩阵合并为一个。

---

## 关于窗口注意力和 RoPE

Qwen2-VL 的视觉注意力与标准注意力有两个重要区别：

1. **窗口注意力（Windowed Attention）**：不是全局注意力，而是在局部窗口内计算注意力，降低计算复杂度
2. **旋转位置编码（RoPE）**：将 2D 空间位置信息编码到注意力计算中

这些技术的完整实现较为复杂，详见 [注意力机制](../07_attention/README.md) 和 [旋转位置编码](../06_rotary_pos_embed/README.md)。

在本节的验证中，我们采用**模块化验证**策略：分别验证 block 中的各个已知组件（LayerNorm、MLP、残差连接），注意力子层使用已有的激活转储值。这体现了工程实践中的重要原则——**分而治之**。

---

## NumPy 实现

```python
import numpy as np

def layer_norm(x, weight, bias, eps=1e-6):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * weight + bias

def quick_gelu(x):
    return x * (1.0 / (1.0 + np.exp(-1.702 * x)))

def vision_mlp(x, fc1_w, fc1_b, fc2_w, fc2_b):
    h = x @ fc1_w.T + fc1_b
    h = quick_gelu(h)
    return h @ fc2_w.T + fc2_b

def vision_block(x, norm1_w, norm1_b, attn_fn,
                 norm2_w, norm2_b, fc1_w, fc1_b, fc2_w, fc2_b):
    # 子层 1：Attention
    x_normed = layer_norm(x, norm1_w, norm1_b)
    attn_out = attn_fn(x_normed)
    x = x + attn_out

    # 子层 2：MLP
    x_normed = layer_norm(x, norm2_w, norm2_b)
    mlp_out = vision_mlp(x_normed, fc1_w, fc1_b, fc2_w, fc2_b)
    x = x + mlp_out
    return x
```

---

## 在 Qwen2-VL 中的位置

```
Qwen2-VL
├── 视觉编码器 (ViT)
│   ├── Patch Embedding (Conv3D)
│   ├── Vision Block × 32            ← 本节（Block 0）
│   │   ├── LayerNorm₁ → Attention → 残差
│   │   └── LayerNorm₂ → MLP → 残差
│   └── Patch Merger
└── 语言模型 (LLM)
    └── Decoder Layer × 28
```

32 个 Vision Block 共享相同的结构，但拥有独立的参数。

---

## 总结

| 属性 | 值 |
|------|-----|
| 输入形状 | $(n, 1280)$ |
| 输出形状 | $(n, 1280)$ |
| 子层数 | 2（Attention + MLP） |
| 归一化方式 | Pre-Norm（先归一化再变换） |
| 残差连接 | 每个子层后都有 |
| 参数量 / block | $\approx 19.7\text{M}$ |
| 视觉编码器总 block 数 | 32 |
