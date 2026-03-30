# SiLU (Swish) 激活函数

## 前置知识：Sigmoid 回顾

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

Sigmoid 将任意实数映射到 $(0, 1)$ 区间，是一个平滑的"开关"函数。详细介绍见 [QuickGELU](../08_quickgelu/README.md)。

---

## SiLU 的动机

Google Brain 团队在 2017 年的论文 *"Searching for Activation Functions"* 中，使用**自动搜索**的方法从大量候选激活函数中找到了一个简单而有效的函数：

$$
f(x) = x \cdot \sigma(x)
$$

这个函数被称为 **Swish**。后来 PyTorch 将其命名为 **SiLU**（Sigmoid Linear Unit），两者完全相同。

### 为什么 SiLU 有效？

SiLU 的结构是 $x \cdot \sigma(x)$，和 QuickGELU、GELU 一样属于"自门控"家族：

- $\sigma(x)$ 作为门控信号，决定保留多少
- $x$ 是原始信号

与 QuickGELU（门控为 $\sigma(1.702x)$）和 GELU（门控为 $\Phi(x)$）相比，SiLU 是最简洁的形式——直接用 $\sigma(x)$ 门控，不需要额外的缩放系数或复杂函数。

---

## 数学定义

$$
\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$

### 关键性质

1. **$\text{SiLU}(0) = 0$**：因为 $0 \cdot \sigma(0) = 0 \cdot 0.5 = 0$
2. **无上界**：当 $x \to +\infty$ 时，$\sigma(x) \to 1$，所以 $\text{SiLU}(x) \to x$
3. **有下界**：最小值约在 $x \approx -1.28$ 处，$\text{SiLU}(-1.28) \approx -0.278$
4. **非单调**：在负半轴先下降后上升，这是 SiLU 区别于 ReLU 的重要特征
5. **处处可导**：导数 $\text{SiLU}'(x) = \sigma(x) + x \cdot \sigma(x) \cdot (1 - \sigma(x)) = \sigma(x)(1 + x(1 - \sigma(x)))$

---

## 函数图像

SiLU 的曲线形状：

- **正半轴**（$x > 0$）：接近恒等函数 $y = x$，但始终略低于它
- **原点**：$\text{SiLU}(0) = 0$，斜率为 $0.5$（不像 ReLU 那样突变）
- **负半轴**（$x < 0$）：先下降到约 $-0.278$（在 $x \approx -1.28$），然后回升趋近于 $0$
- 与 ReLU 相比，SiLU 更"柔软"——没有尖锐的拐点
- 与 GELU 和 QuickGELU 的形状非常相似，但负半轴的"谷底"稍微更深

---

## 逐步数值示例

以 $x = 2.0$ 为例：

1. 计算 $\sigma(2.0) = \frac{1}{1 + e^{-2.0}} = \frac{1}{1 + 0.1353} = \frac{1}{1.1353} = 0.8808$
2. 计算 $\text{SiLU}(2.0) = 2.0 \times 0.8808 = 1.7616$

以 $x = -1.0$ 为例：

1. 计算 $\sigma(-1.0) = \frac{1}{1 + e^{1.0}} = \frac{1}{1 + 2.718} = \frac{1}{3.718} = 0.2689$
2. 计算 $\text{SiLU}(-1.0) = (-1.0) \times 0.2689 = -0.2689$

### 更多数值

| $x$ | $\sigma(x)$ | $\text{SiLU}(x) = x \cdot \sigma(x)$ |
|-----|------------|--------------------------------------|
| $-3.0$ | $0.0474$ | $-0.1423$ |
| $-2.0$ | $0.1192$ | $-0.2384$ |
| $-1.0$ | $0.2689$ | $-0.2689$ |
| $0.0$ | $0.5000$ | $0.0000$ |
| $1.0$ | $0.7311$ | $0.7311$ |
| $2.0$ | $0.8808$ | $1.7616$ |
| $3.0$ | $0.9526$ | $2.8577$ |

注意 $x = -1.0$ 处 SiLU 值最负（$\approx -0.2689$），但最小值实际在 $x \approx -1.28$。

---

## NumPy 实现

```python
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def silu(x):
    return x * sigmoid(x)
```

---

## 在 Qwen2-VL 中的使用位置

在 Qwen2-VL 模型中，**文本解码器（Text Decoder）** 的每个 Transformer 层都包含一个 Gated MLP，其中使用 SiLU 作为门控分支的激活函数：

```
Decoder Layer → RMSNorm → Attention → RMSNorm → Gated MLP
                                                  ├── gate_proj (Linear) → SiLU  ← 这里
                                                  ├── up_proj   (Linear)
                                                  ├── gate * up（逐元素相乘）
                                                  └── down_proj (Linear)
```

输入张量形状为 $(1, 3602, 8960)$，其中：
- $1$ 是 batch size
- $3602$ 是序列长度（文本 + 视觉 token）
- $8960$ 是 MLP 的中间维度（gate_proj 的输出维度）

---

## 三种激活函数的对比总结

| 特性 | QuickGELU | GELU | SiLU (Swish) |
|------|-----------|------|-------------|
| 公式 | $x \cdot \sigma(1.702x)$ | $x \cdot \Phi(x)$ | $x \cdot \sigma(x)$ |
| 门控函数 | 缩放 sigmoid | 正态 CDF | 原始 sigmoid |
| 负半轴最小值 | $\approx -0.10$ | $\approx -0.17$ | $\approx -0.28$ |
| 计算复杂度 | 低 | 中（需要 erf） | 低 |
| 在 Qwen2-VL 中 | Vision MLP | Patch Merger | Text Decoder MLP |
| 提出时间 | 2021 (CLIP) | 2016 | 2017 |

### 共同特征

三者都属于 **"自门控"（self-gating）** 激活函数：

$$
f(x) = x \cdot g(x), \quad g(x) \in (0, 1)
$$

其中 $g(x)$ 是一个值域在 $(0, 1)$ 的门控函数。输入 $x$ 同时参与"被门控"和"产生门控信号"两个角色——这就是"自门控"的含义。

### 选择逻辑

- **GELU**：理论最优雅，基于概率论，是 BERT、GPT-2 等早期模型的标准选择
- **QuickGELU**：GELU 的 sigmoid 近似，计算更快，CLIP 和视觉模型常用
- **SiLU**：实验发现效果好，LLaMA 系列和现代 LLM 的标准选择
