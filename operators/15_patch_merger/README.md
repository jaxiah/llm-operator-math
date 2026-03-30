# Patch Merger（补丁合并器）

## 前置知识

在阅读本节之前，请确保理解：

- [Layer Normalization](../03_layer_norm/README.md)：$\text{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$
- [线性变换](../01_linear/README.md)：$y = xW^T + b$
- [GELU 激活函数](../09_gelu/README.md)：$\text{GELU}(x) = x \cdot \Phi(x)$

---

## 什么是 Patch Merger？

视觉编码器将图像分成很多小 patch（补丁），经过多层 Vision Block 处理后，每个 patch 都有了丰富的特征表示。但 patch 数量太多会给语言模型造成过大的计算负担。

**Patch Merger 的作用是减少 patch 数量**——将相邻的 $2 \times 2$ 个 patch 合并为一个，同时调整特征维度以适配语言模型。

直觉上：
- 合并前：14308 个 patch，每个 1280 维（视觉编码器维度）
- 合并后：3577 个 patch，每个 1536 维（语言模型维度）

这就像把高分辨率的"碎片"拼成低分辨率但信息更密集的"块"。

---

## 数学定义

Patch Merger 分为两个阶段：

### 阶段 1：空间合并（Spatial Merge）

给定 patch 序列 $X \in \mathbb{R}^{n \times d}$ 和空间网格尺寸 $(t, h, w)$：

1. **归一化**：

$$
\hat{X} = \text{LN}(X)
$$

2. **分组合并**：

在 Qwen2-VL 中，视觉编码器的窗口注意力机制已经将 patch 重新排列，使得每 $s^2 = 4$ 个相邻 patch 在序列中连续排列。因此空间合并只需一个简单的 reshape：

$$
\hat{X}_{\text{merged}} = \text{reshape}(\hat{X},\ [-1,\ s^2 \cdot d])
$$

每个合并后的 patch 是 4 个空间邻居特征的拼接，维度为 $4d$：

$$
\hat{X}_{\text{merged}} \in \mathbb{R}^{(n / s^2) \times 4d}
$$

### 阶段 2：MLP 投影

$$
y = W_2 \cdot \text{GELU}(W_1 \cdot \hat{X}_{\text{merged}} + b_1) + b_2
$$

其中：
- $W_1 \in \mathbb{R}^{d_{\text{ff}} \times 4d}$，$b_1 \in \mathbb{R}^{d_{\text{ff}}}$（本例中 $d_{\text{ff}} = 4d = 5120$）
- $W_2 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{ff}}}$，$b_2 \in \mathbb{R}^{d_{\text{out}}}$（$d_{\text{out}} = 1536$，匹配语言模型维度）

---

## 数据流

```
输入: (14308, 1280)       ← 所有 patch 的特征
    │
    ▼
LayerNorm: γ, β           ← ln_q 归一化
    │
    ▼
归一化后: (14308, 1280)
    │
    ▼
空间合并: reshape 分组     ← 窗口注意力已预排序 patch
    │
    ▼
合并后: (3577, 5120)      ← 14308/4 ≈ 3577, 1280×4 = 5120
    │
    ▼
Linear₁: (5120, 5120)     ← mlp.0
    │
    ▼
GELU 激活                  ← mlp.1
    │
    ▼
Linear₂: (1536, 5120)     ← mlp.2
    │
    ▼
输出: (3577, 1536)         ← 适配语言模型的维度
```

---

## 空间合并的细节

### 为什么只需 reshape？

一个关键的实现细节：Qwen2-VL 的视觉编码器在窗口注意力处理过程中，已经将 patch 重新排列为"$2 \times 2$ 空间邻居连续"的顺序。具体来说：

1. 视觉编码器在处理之前，使用 `_get_window_index` 函数将 patch 按窗口重排
2. 每个窗口内的 patch 按空间邻近性排列
3. 经过所有 Vision Block 后，patch 顺序保持不变

因此，Patch Merger 的"空间合并"操作极其简单：

```python
x_merged = x_normed.reshape(-1, 4 * d)  # 每 4 个连续 patch = 一个 2×2 块
```

这是一个优雅的设计——将复杂的空间重排前置到编码器的初始化阶段，使得后续的合并操作变为简单的内存重解释。

### 合并效果

```
合并前 (14308 个 patch):
[p0, p1, p2, p3, p4, p5, p6, p7, ...]   ← 每 4 个为一组 2×2 空间邻居
 └──────┘  └──────┘
  merged₀   merged₁

合并后 (3577 个 patch):
[concat(p0,p1,p2,p3), concat(p4,p5,p6,p7), ...]
  1280×4 = 5120 维      1280×4 = 5120 维
```

---

## 为什么需要 Patch Merger？

| 没有 Merger | 有 Merger |
|------------|-----------|
| 14308 个 token 进入 LLM | 3577 个 token 进入 LLM |
| 注意力复杂度 $O(n^2)$ 极高 | 注意力复杂度降低约 16 倍 |
| 视觉维度 1280 ≠ LLM 维度 1536 | 维度匹配，可直接拼接 |

Patch Merger 同时完成了两个目标：
1. **降低序列长度**：减少 4 倍的 token 数量
2. **维度适配**：将视觉特征从 1280 维映射到 1536 维（语言模型的隐藏维度）

---

## 逐步数值示例

假设 $d = 2$，网格为 $1 \times 2 \times 2$（4 个 patch），合并后为 1 个 patch：

4 个归一化后的 patch：
$$
\hat{x}_1 = [0.5, -0.3],\quad \hat{x}_2 = [1.2, 0.8],\quad \hat{x}_3 = [-0.1, 0.4],\quad \hat{x}_4 = [0.7, -0.6]
$$

拼接后：
$$
x_{\text{merged}} = [0.5, -0.3, 1.2, 0.8, -0.1, 0.4, 0.7, -0.6] \in \mathbb{R}^{8}
$$

然后经过 MLP：
$$
h = \text{GELU}(x_{\text{merged}} \cdot W_1^T + b_1)
$$
$$
y = h \cdot W_2^T + b_2
$$

---

## NumPy 实现

```python
import numpy as np

def layer_norm(x, weight, bias, eps=1e-6):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * weight + bias

def spatial_merge(x_normed, spatial_merge_size=2):
    """窗口注意力已将 patch 预排序，只需 reshape。"""
    d = x_normed.shape[-1]
    s = spatial_merge_size
    return x_normed.reshape(-1, s * s * d)

def patch_merger(x, ln_w, ln_b,
                 fc1_w, fc1_b, fc2_w, fc2_b,
                 spatial_merge_size=2):
    x_normed = layer_norm(x, ln_w, ln_b)
    x_merged = spatial_merge(x_normed, spatial_merge_size)
    h = x_merged @ fc1_w.T + fc1_b
    h = gelu_exact(h)
    return h @ fc2_w.T + fc2_b
```

---

## 参数量分析

| 组件 | 权重形状 | 参数量 |
|------|---------|--------|
| ln_q.weight | $(1280,)$ | 1,280 |
| ln_q.bias | $(1280,)$ | 1,280 |
| mlp.0.weight | $(5120, 5120)$ | 26,214,400 |
| mlp.0.bias | $(5120,)$ | 5,120 |
| mlp.2.weight | $(1536, 5120)$ | 7,864,320 |
| mlp.2.bias | $(1536,)$ | 1,536 |
| **合计** | | **$\approx 34.1\text{M}$** |

注意 mlp.0 的权重是 $(5120, 5120)$ 而非 $(5120, 1280)$，因为输入已经是合并后的 $4 \times 1280 = 5120$ 维。

---

## 在 Qwen2-VL 中的位置

```
Qwen2-VL
├── 视觉编码器 (ViT)
│   ├── Patch Embedding (Conv3D)
│   ├── Vision Block × 32
│   └── Patch Merger                ← 本节
│       ├── LayerNorm (ln_q)
│       ├── 空间合并 (2×2 → 1)
│       └── MLP: Linear → GELU → Linear
└── 语言模型 (LLM, d=1536)
    └── Decoder Layer × 28
```

Patch Merger 是视觉编码器与语言模型之间的**桥梁**，将视觉特征转换为语言模型可以理解的格式。

---

## 总结

| 属性 | 值 |
|------|-----|
| 输入形状 | $(14308, 1280)$ |
| 输出形状 | $(3577, 1536)$ |
| 空间合并比 | $2 \times 2 = 4$ 倍 |
| 维度变换 | $1280 \rightarrow 5120 \rightarrow 1536$ |
| 激活函数 | GELU（精确形式） |
| 参数量 | $\approx 34.1\text{M}$ |
