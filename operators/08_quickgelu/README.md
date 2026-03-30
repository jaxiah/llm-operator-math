# QuickGELU 激活函数

## 前置知识：Sigmoid 函数

在理解 QuickGELU 之前，我们需要先掌握 **sigmoid 函数**，它是深度学习中最基础的非线性函数之一。

### 定义

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

### 直觉理解

sigmoid 函数将任意实数映射到 $(0, 1)$ 区间：

- 当 $x \to +\infty$ 时，$\sigma(x) \to 1$
- 当 $x = 0$ 时，$\sigma(0) = 0.5$
- 当 $x \to -\infty$ 时，$\sigma(x) \to 0$

它的图像呈 **S 形曲线**（sigmoid 一词源自希腊字母 σ），中心对称于点 $(0, 0.5)$。

### 数值示例

| $x$ | $e^{-x}$ | $1 + e^{-x}$ | $\sigma(x)$ |
|-----|-----------|---------------|-------------|
| $-2$ | $7.389$ | $8.389$ | $0.1192$ |
| $-1$ | $2.718$ | $3.718$ | $0.2689$ |
| $0$ | $1.000$ | $2.000$ | $0.5000$ |
| $1$ | $0.368$ | $1.368$ | $0.7311$ |
| $2$ | $0.135$ | $1.135$ | $0.8808$ |

### NumPy 实现

```python
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
```

---

## QuickGELU 的动机

标准的 GELU（Gaussian Error Linear Unit）涉及高斯分布的累积分布函数（CDF），计算代价较高。研究人员发现，通过用 sigmoid 函数来近似 GELU，可以得到一个更简洁、计算更快的版本。

这个近似版本被称为 **QuickGELU**，最早出现在 OpenAI 的 CLIP 模型中，后来被 Qwen2-VL 的视觉编码器采用。

---

## 数学定义

$$
\text{QuickGELU}(x) = x \cdot \sigma(1.702x)
$$

其中 $\sigma$ 是 sigmoid 函数，$1.702$ 是一个经验常数，使得 QuickGELU 与标准 GELU 的曲线尽可能接近。

### 拆解理解

QuickGELU 可以分解为两个部分：

1. **门控信号**：$\sigma(1.702x)$ — 值域在 $(0, 1)$，决定"让多少信号通过"
2. **原始信号**：$x$ — 输入本身

两者相乘，就像一个"自门控"机制：输入值本身决定自己被保留多少。

### 为什么是 1.702？

常数 $1.702$ 来源于对 GELU 的最优 sigmoid 近似拟合。具体地，GELU 可以写成：

$$
\text{GELU}(x) = x \cdot \Phi(x)
$$

其中 $\Phi(x)$ 是标准正态分布的 CDF。当我们用 $\sigma(\alpha x)$ 来近似 $\Phi(x)$ 时，最优的 $\alpha \approx 1.702$。

---

## 函数图像

QuickGELU 的曲线形状：

- **正半轴**（$x > 0$）：接近恒等函数 $y = x$，因为 $\sigma(1.702x) \approx 1$
- **负半轴**（$x < 0$）：输出被压缩到接近 0，因为 $\sigma(1.702x) \approx 0$
- **原点附近**：平滑过渡，$\text{QuickGELU}(0) = 0$
- 与 ReLU 不同，QuickGELU 在 $x < 0$ 时**不完全为零**，而是有微小的负值区域

整体形状类似 GELU：一条在原点附近平滑弯曲的曲线，正半轴近似线性，负半轴快速衰减到零。

---

## 逐步数值示例

以 $x = 1.5$ 为例：

1. 计算 $1.702 \times 1.5 = 2.553$
2. 计算 $\sigma(2.553) = \frac{1}{1 + e^{-2.553}} = \frac{1}{1 + 0.0781} = \frac{1}{1.0781} = 0.9276$
3. 计算 $\text{QuickGELU}(1.5) = 1.5 \times 0.9276 = 1.3914$

以 $x = -1.0$ 为例：

1. 计算 $1.702 \times (-1.0) = -1.702$
2. 计算 $\sigma(-1.702) = \frac{1}{1 + e^{1.702}} = \frac{1}{1 + 5.485} = \frac{1}{6.485} = 0.1542$
3. 计算 $\text{QuickGELU}(-1.0) = (-1.0) \times 0.1542 = -0.1542$

### 更多数值

| $x$ | $1.702x$ | $\sigma(1.702x)$ | $\text{QuickGELU}(x)$ |
|-----|----------|-------------------|------------------------|
| $-2.0$ | $-3.404$ | $0.0322$ | $-0.0644$ |
| $-1.0$ | $-1.702$ | $0.1542$ | $-0.1542$ |
| $0.0$ | $0.0$ | $0.5000$ | $0.0000$ |
| $1.0$ | $1.702$ | $0.8458$ | $0.8458$ |
| $2.0$ | $3.404$ | $0.9678$ | $1.9355$ |

---

## NumPy 实现

```python
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def quick_gelu(x):
    return x * sigmoid(1.702 * x)
```

---

## 在 Qwen2-VL 中的使用位置

在 Qwen2-VL 模型中，**视觉编码器（Vision Encoder）** 的每个 Transformer block 都包含一个 MLP 层，其中使用 QuickGELU 作为激活函数：

```
Vision Block → LayerNorm → Attention → LayerNorm → MLP
                                                     ├── Linear (fc1)
                                                     ├── QuickGELU  ← 这里
                                                     └── Linear (fc2)
```

输入张量形状为 $(14308, 5120)$，其中 14308 是视觉 token 的数量，5120 是 MLP 的隐藏维度。

---

## 与其他激活函数的对比

| 激活函数 | 公式 | 使用位置 |
|---------|------|---------|
| QuickGELU | $x \cdot \sigma(1.702x)$ | Vision MLP |
| GELU | $x \cdot \Phi(x)$ 或 tanh 近似 | Patch Merger |
| SiLU (Swish) | $x \cdot \sigma(x)$ | Text Decoder MLP |

三者都是"自门控"激活函数，结构相似：$x$ 乘以一个 $(0, 1)$ 范围的门控值。区别在于门控函数的选择：
- SiLU 用 $\sigma(x)$
- QuickGELU 用 $\sigma(1.702x)$（放大后的 sigmoid）
- GELU 用 $\Phi(x)$（标准正态 CDF）
