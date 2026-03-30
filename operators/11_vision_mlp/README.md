# Vision MLP（视觉编码器前馈网络）

## 前置知识

在阅读本节之前，请确保理解：

- [线性变换](../01_linear/README.md)：矩阵乘法 $y = xW^T + b$
- [QuickGELU 激活函数](../08_quickgelu/README.md)：$\text{QuickGELU}(x) = x \cdot \sigma(1.702x)$

---

## 什么是 MLP？

**MLP**（Multi-Layer Perceptron，多层感知机）是 Transformer 架构中的核心组件之一。每个 Transformer block 都包含两个子层：

1. **Self-Attention** — 负责 token 之间的信息交换
2. **MLP（前馈网络）** — 负责对每个 token 独立地进行非线性变换

MLP 的作用可以理解为：Attention 告诉每个 token "应该关注什么信息"，而 MLP 告诉每个 token "拿到信息后应该怎么处理"。

---

## 数学定义

Vision MLP 是最经典的两层前馈网络结构：

$$
\text{MLP}(x) = \text{FC2}\big(\text{QuickGELU}\big(\text{FC1}(x)\big)\big)
$$

展开为：

$$
\text{MLP}(x) = \Big(\text{QuickGELU}(xW_1^T + b_1)\Big) W_2^T + b_2
$$

其中：
- $x \in \mathbb{R}^{n \times d}$ 是输入，$n$ 是 token 数，$d$ 是特征维度
- $W_1 \in \mathbb{R}^{d_{\text{ff}} \times d}$ 是第一层权重（升维）
- $b_1 \in \mathbb{R}^{d_{\text{ff}}}$ 是第一层偏置
- $W_2 \in \mathbb{R}^{d \times d_{\text{ff}}}$ 是第二层权重（降维）
- $b_2 \in \mathbb{R}^{d}$ 是第二层偏置

---

## 数据流

在 Qwen2-VL 视觉编码器中，MLP 的数据流如下：

```
输入 x: (14308, 1280)
    │
    ▼
FC1: x @ W1.T + b1     ← 升维 1280 → 5120（扩展比 4x）
    │
    ▼
中间结果: (14308, 5120)
    │
    ▼
QuickGELU: h * σ(1.702h)  ← 非线性激活
    │
    ▼
激活后: (14308, 5120)
    │
    ▼
FC2: a @ W2.T + b2     ← 降维 5120 → 1280
    │
    ▼
输出: (14308, 1280)
```

维度先升后降，中间形成一个 **"瓶颈"结构的反转** — 升维到 4 倍宽度，激活后再压缩回来。

---

## 为什么要先升维再降维？

这是 MLP 的核心设计思想：

1. **升维（expansion）**：将 1280 维投影到 5120 维，相当于把每个 token 映射到一个更高维的空间
2. **非线性激活**：在高维空间中应用 QuickGELU，让网络能学到复杂的非线性关系
3. **降维（compression）**：压缩回 1280 维，保留最有用的信息

直觉上，这就像：
- 把一段话（1280 维）展开成更详细的描述（5120 维）
- 在展开的描述上做理解和筛选（QuickGELU）
- 再总结回精炼的版本（1280 维）

**扩展比（expansion ratio）** 通常是 4 倍，这里 $5120 / 1280 = 4$，是 Transformer 论文中的经典设计。

---

## 逐步数值示例

假设 $d = 3$，$d_{\text{ff}} = 6$，输入一个 token $x = [1.0, -0.5, 2.0]$：

### 步骤 1：FC1（升维）

$$
h = xW_1^T + b_1
$$

假设 $W_1$ 和 $b_1$ 使得 $h = [0.8, -1.2, 2.5, -0.3, 1.7, 0.1]$。

### 步骤 2：QuickGELU 激活

对 $h$ 的每个元素应用 $\text{QuickGELU}(z) = z \cdot \sigma(1.702z)$：

| $z$ | $1.702z$ | $\sigma(1.702z)$ | $\text{QuickGELU}(z)$ |
|-----|----------|-------------------|------------------------|
| $0.8$ | $1.362$ | $0.7962$ | $0.6370$ |
| $-1.2$ | $-2.042$ | $0.1149$ | $-0.1379$ |
| $2.5$ | $4.255$ | $0.9860$ | $2.4650$ |
| $-0.3$ | $-0.511$ | $0.3751$ | $-0.1125$ |
| $1.7$ | $2.893$ | $0.9476$ | $1.6109$ |
| $0.1$ | $0.170$ | $0.5424$ | $0.0542$ |

激活后：$a = [0.637, -0.138, 2.465, -0.113, 1.611, 0.054]$

### 步骤 3：FC2（降维）

$$
y = aW_2^T + b_2
$$

将 6 维压缩回 3 维，得到最终输出。

---

## NumPy 实现

```python
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def quick_gelu(x):
    return x * sigmoid(1.702 * x)

def vision_mlp(x, fc1_weight, fc1_bias, fc2_weight, fc2_bias):
    """Vision MLP: FC1 → QuickGELU → FC2"""
    h = x @ fc1_weight.T + fc1_bias        # 升维
    h = quick_gelu(h)                        # 激活
    y = h @ fc2_weight.T + fc2_bias          # 降维
    return y
```

---

## 在 Qwen2-VL 中的位置

视觉编码器的每个 block 中：

```
Vision Block
├── LayerNorm → Self-Attention → 残差连接
└── LayerNorm → MLP (FC1 → QuickGELU → FC2) → 残差连接  ← 这里
```

权重参数：
- `visual.blocks.0.mlp.fc1.weight` — 形状 $(5120, 1280)$
- `visual.blocks.0.mlp.fc1.bias` — 形状 $(5120,)$
- `visual.blocks.0.mlp.fc2.weight` — 形状 $(1280, 5120)$
- `visual.blocks.0.mlp.fc2.bias` — 形状 $(1280,)$

参数量：$5120 \times 1280 \times 2 + 5120 + 1280 = 13{,}113{,}600 \approx 13.1\text{M}$

---

## 与 Gated MLP 的对比

| 特性 | Vision MLP（本节） | Gated MLP（下一节） |
|------|-------------------|---------------------|
| 结构 | FC1 → 激活 → FC2 | gate/up/down 三路投影 |
| 激活函数 | QuickGELU | SiLU |
| 有偏置 | ✅ 有 $b_1, b_2$ | ❌ 无偏置 |
| 扩展比 | 4x (1280→5120) | ~5.8x (1536→8960) |
| 使用位置 | 视觉编码器 | 文本解码器 |
| 参数量 | ~13.1M / block | ~41.3M / layer |

Vision MLP 是经典的两层结构，简单高效；Gated MLP 通过门控机制获得更强的表达能力，但参数量更大。详见 [Gated MLP](../12_gated_mlp/README.md)。
