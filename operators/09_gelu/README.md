# GELU 激活函数

## 前置知识：Sigmoid 与高斯分布

### Sigmoid 回顾

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

Sigmoid 将实数映射到 $(0, 1)$，是一个平滑的"开关"函数。我们在 [QuickGELU](../08_quickgelu/README.md) 中详细介绍了它。

### 标准正态分布的 CDF

GELU 的核心是 **标准正态分布的累积分布函数（CDF）**，记作 $\Phi(x)$：

$$
\Phi(x) = P(X \le x) = \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]
$$

其中 $\text{erf}$ 是**误差函数**（error function）：

$$
\text{erf}(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2} \, dt
$$

**直觉理解**：$\Phi(x)$ 表示"标准正态分布中，随机变量取值小于等于 $x$ 的概率"。

| $x$ | $\Phi(x)$ | 含义 |
|-----|-----------|------|
| $-\infty$ | $0$ | 不可能小于 $-\infty$ |
| $-2$ | $0.0228$ | 约 2.3% 的值小于 $-2$ |
| $-1$ | $0.1587$ | 约 15.9% |
| $0$ | $0.5$ | 正好一半 |
| $1$ | $0.8413$ | 约 84.1% |
| $2$ | $0.9772$ | 约 97.7% |

---

## GELU 的动机

传统的 ReLU 函数 $\text{ReLU}(x) = \max(0, x)$ 是一个硬阈值：

- $x > 0$：完全保留
- $x \le 0$：完全丢弃

这种"非此即彼"的策略有些粗暴。GELU 的创新想法是引入**概率视角**：

> 与其用确定性的阈值，不如根据输入值的"大小"**按概率**决定保留多少。

具体地，一个值 $x$ 被保留的概率等于"在标准正态分布中，比 $x$ 还小的概率"——即 $\Phi(x)$。这意味着：

- 较大的正值几乎 100% 被保留
- 接近 0 的值约 50% 被保留
- 较大的负值几乎被完全抑制

这就是 GELU 的核心思想，出自论文 *"Gaussian Error Linear Units (GELUs)"*（Hendrycks & Gimpel, 2016）。

---

## 数学定义

### 精确形式

$$
\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]
$$

### tanh 近似形式

由于 $\text{erf}$ 计算可能较慢，论文中还给出了一个 tanh 近似：

$$
\text{GELU}(x) \approx 0.5 \cdot x \cdot \left[1 + \tanh\left(\sqrt{\frac{2}{\pi}} \left(x + 0.044715 \cdot x^3\right)\right)\right]
$$

其中 $\sqrt{\frac{2}{\pi}} \approx 0.7979$，$0.044715$ 是拟合系数。

### 两种形式的关系

$$
\Phi(x) \approx \frac{1}{2}\left[1 + \tanh\left(\sqrt{\frac{2}{\pi}} \left(x + 0.044715 x^3\right)\right)\right]
$$

这个近似在整个实数域上都非常精确（最大误差约 $10^{-4}$）。

---

## 函数图像

GELU 的曲线形状：

- **正半轴**（$x > 0$）：接近恒等函数 $y = x$，因为 $\Phi(x) \approx 1$
- **负半轴**（$x < 0$）：输出快速趋近于 0，因为 $\Phi(x) \approx 0$
- **原点附近**：平滑过渡，$\text{GELU}(0) = 0$
- 在 $x \approx -0.17$ 处有一个浅浅的负值谷底（最小值约 $-0.17$）
- 整体曲线比 ReLU 更平滑，没有尖锐的拐点

与 QuickGELU 的曲线几乎重合，肉眼难以区分。

---

## 逐步数值示例

### 精确形式

以 $x = 1.0$ 为例：

1. 计算 $\frac{x}{\sqrt{2}} = \frac{1.0}{1.4142} = 0.7071$
2. 查表或计算 $\text{erf}(0.7071) = 0.6827$
3. 计算 $\Phi(1.0) = \frac{1}{2}(1 + 0.6827) = 0.8413$
4. 计算 $\text{GELU}(1.0) = 1.0 \times 0.8413 = 0.8413$

### tanh 近似

以 $x = 1.0$ 为例：

1. 计算 $x^3 = 1.0$
2. 计算 $0.044715 \times 1.0 = 0.044715$
3. 计算 $x + 0.044715 x^3 = 1.0 + 0.044715 = 1.044715$
4. 计算 $\sqrt{2/\pi} \times 1.044715 = 0.7979 \times 1.044715 = 0.8336$
5. 计算 $\tanh(0.8336) = 0.6836$
6. 计算 $1 + 0.6836 = 1.6836$
7. 计算 $0.5 \times 1.0 \times 1.6836 = 0.8418$

精确值 $0.8413$ vs 近似值 $0.8418$，差异仅 $0.0005$！

### 更多数值

| $x$ | $\Phi(x)$（精确） | $\text{GELU}(x)$（精确） | $\text{GELU}(x)$（tanh 近似） |
|-----|-------------------|--------------------------|-------------------------------|
| $-2.0$ | $0.0228$ | $-0.0455$ | $-0.0454$ |
| $-1.0$ | $0.1587$ | $-0.1587$ | $-0.1588$ |
| $0.0$ | $0.5000$ | $0.0000$ | $0.0000$ |
| $1.0$ | $0.8413$ | $0.8413$ | $0.8418$ |
| $2.0$ | $0.9772$ | $1.9545$ | $1.9546$ |

---

## NumPy 实现

### 精确形式（使用 erf）

```python
import numpy as np
from scipy.special import erf

def gelu_exact(x):
    return x * 0.5 * (1.0 + erf(x / np.sqrt(2.0)))
```

NumPy 本身没有 `erf`，但可以用 `scipy.special.erf`。不过我们也可以用纯 NumPy 实现（见下方）。

### tanh 近似形式

```python
import numpy as np

def gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
```

---

## 在 Qwen2-VL 中的使用位置

在 Qwen2-VL 模型中，**Patch Merger**（视觉 token 合并模块）的 MLP 使用 GELU 激活函数：

```
Patch Merger → LayerNorm → Linear → GELU ← 这里 → Linear
```

PyTorch 中 `nn.GELU()` 的默认实现使用**精确形式**（基于 erf），而非 tanh 近似。

输入张量形状为 $(3577, 5120)$，其中 3577 是合并后的视觉 token 数量，5120 是隐藏维度。

---

## 精确形式 vs tanh 近似

在验证时，我们优先使用精确形式（erf），因为 PyTorch 的 `nn.GELU()` 默认使用精确版本：

| 方面 | 精确形式 | tanh 近似 |
|------|---------|-----------|
| 公式 | $x \cdot \Phi(x)$ | $0.5x(1 + \tanh(\cdots))$ |
| 精度 | 数学精确 | 最大误差 $\sim 10^{-4}$ |
| PyTorch 默认 | ✅ `nn.GELU()` | `nn.GELU(approximate='tanh')` |
| 计算速度 | 稍慢 | 稍快 |

---

## 与其他激活函数的对比

| 激活函数 | 门控信号 | 公式 | 使用位置 |
|---------|---------|------|---------|
| QuickGELU | $\sigma(1.702x)$ | $x \cdot \sigma(1.702x)$ | Vision MLP |
| **GELU** | $\Phi(x)$ | $x \cdot \Phi(x)$ | Patch Merger |
| SiLU (Swish) | $\sigma(x)$ | $x \cdot \sigma(x)$ | Text Decoder MLP |

GELU 是三者中理论最优雅的——直接基于概率论。QuickGELU 是 GELU 的快速 sigmoid 近似，SiLU 则是最简洁的形式。
