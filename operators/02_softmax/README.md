# 02 — Softmax

## 直觉：Softmax 是什么？

Softmax 把一组**任意实数**转换成一个**概率分布**——所有输出值都在 $(0, 1)$ 之间，且总和为 1。

它是分类问题和注意力机制的核心：
- 分类时：把模型的原始输出（logits）变成"每个类别的概率"
- 注意力时：把相似度分数变成"注意力权重"

## 数学定义

给定向量 $x = [x_1, x_2, \ldots, x_n]$：

$$
\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}
$$

### 数值稳定版本

直接计算 $e^{x_i}$ 会遇到**数值溢出**问题——当 $x_i$ 很大时，$e^{x_i}$ 会变成 `inf`。

解决方法：先减去最大值 $m = \max(x)$：

$$
\text{softmax}(x)_i = \frac{e^{x_i - m}}{\sum_{j=1}^{n} e^{x_j - m}}
$$

这在数学上是等价的（分子分母同时除以 $e^m$），但数值上更稳定。

## 数值示例

输入 $x = [2.0, 1.0, 0.1]$：

1. 减去最大值：$x' = [0.0, -1.0, -1.9]$
2. 计算指数：$e^{x'} = [1.0, 0.368, 0.150]$
3. 求和：$S = 1.0 + 0.368 + 0.150 = 1.518$
4. 归一化：$\text{softmax}(x) = [0.659, 0.242, 0.099]$

验证：$0.659 + 0.242 + 0.099 = 1.0$ ✅

## NumPy 实现

```python
import numpy as np

def softmax(x, axis=-1):
    """数值稳定的 softmax 实现。"""
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

# 示例
x = np.array([2.0, 1.0, 0.1])
print(softmax(x))  # [0.659, 0.242, 0.099]
print(np.sum(softmax(x)))  # 1.0
```

## Softmax 的性质

| 性质 | 说明 |
|------|------|
| 输出范围 | 每个元素 $\in (0, 1)$ |
| 总和为 1 | $\sum_i \text{softmax}(x)_i = 1$ |
| 单调性 | 输入越大 → 输出越大 |
| 温度缩放 | $\text{softmax}(x / T)$：$T \to 0$ 趋近 one-hot，$T \to \infty$ 趋近均匀分布 |

## 在 Qwen2-VL 中的应用

- **注意力权重**：$\text{softmax}(QK^T / \sqrt{d_k})$ 计算每个 token 对其他 token 的注意力
- **最终分类**：语言模型头的输出经过 softmax 得到下一个 token 的概率分布

## 验证

运行 `python -m operators.02_softmax.impl` 进行合成数据测试，验证实现的正确性。
