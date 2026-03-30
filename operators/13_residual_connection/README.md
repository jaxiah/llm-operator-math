# 13 — 残差连接 (Residual Connection)

## 直觉：残差连接是什么？

残差连接是深度学习中最简单却最重要的技巧之一。它的核心思想惊人地简单：

> **不要让网络从零学习输出，让它学习"需要改变多少"。**

数学上就是一个加法：把层的输入直接"跳过"该层，加到层的输出上。

## 数学定义

$$
y = x + F(x)
$$

其中：
- $x$：残差块的输入（"跳跃连接"的那条线）
- $F(x)$：子层（如注意力层或 MLP）的输出
- $y$：残差块的输出

就是这么简单——**一个逐元素加法**。

## 为什么需要残差连接？

### 梯度消失问题

没有残差连接时，梯度需要依次穿过每一层。假设有 $L$ 层，每层梯度乘以一个系数 $\alpha < 1$：

$$
\frac{\partial \mathcal{L}}{\partial x_0} = \prod_{i=1}^{L} \alpha_i \approx \alpha^L \to 0
$$

层数越多，梯度越小，底层几乎学不到东西。

### 残差连接的解决方案

有了残差连接，梯度有一条"高速公路"可以直接回传：

$$
\frac{\partial y}{\partial x} = 1 + \frac{\partial F(x)}{\partial x}
$$

注意那个 **$+1$**！即使 $\frac{\partial F(x)}{\partial x}$ 很小，梯度至少还有 1，不会消失。

## 数值示例

```
输入 x     = [1.0, 2.0, 3.0]
子层输出 F(x) = [0.1, -0.2, 0.3]  （注意力层学到的"修正量"）

残差输出 y  = [1.0 + 0.1, 2.0 + (-0.2), 3.0 + 0.3]
            = [1.1, 1.8, 3.3]
```

## NumPy 实现

```python
import numpy as np

def residual_add(x, f_x):
    """残差连接: y = x + F(x)"""
    return x + f_x

# 示例
x   = np.array([1.0, 2.0, 3.0])
f_x = np.array([0.1, -0.2, 0.3])
y   = residual_add(x, f_x)
print(y)  # [1.1, 1.8, 3.3]
```

## 在 Qwen2-VL 中的应用

每个 Transformer 块都有 **两个** 残差连接：

```
x ──────────────────────┐
│                       │ (残差连接 1)
├─→ Norm → Attention ───┘──→ x₁ = x + Attn(Norm(x))
│                              │
x₁ ─────────────────────┐     │
│                       │ (残差连接 2)
├─→ Norm → MLP ─────────┘──→ x₂ = x₁ + MLP(Norm(x₁))
```

在 Qwen2-VL 的 Vision Encoder 中，每个 block 的结构为：
1. `residual_1 = block_input + attention(norm1(block_input))`
2. `residual_2 = residual_1 + mlp(norm2(residual_1))`

## 验证

运行 `python -m operators.13_residual_connection.impl` 验证 Vision Block 0 的第一个残差连接。
