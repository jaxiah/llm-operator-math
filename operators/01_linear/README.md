# 01 — 线性变换 (Linear Transform)

## 直觉：线性变换是什么？

想象你有一组数字（一个向量），线性变换就是把这组数字"旋转、拉伸、投影"到另一个空间。
在神经网络中，**几乎所有的"学习"都发生在线性变换里**——网络通过调整权重矩阵 $W$ 和偏置 $b$ 来学习数据的模式。

## 数学定义

$$
y = xW^T + b
$$

其中：
- $x \in \mathbb{R}^{n \times d_{\text{in}}}$：输入矩阵，$n$ 是样本数，$d_{\text{in}}$ 是输入维度
- $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$：权重矩阵
- $b \in \mathbb{R}^{d_{\text{out}}}$：偏置向量（可选）
- $y \in \mathbb{R}^{n \times d_{\text{out}}}$：输出矩阵

> **为什么是 $W^T$？** PyTorch 的 `nn.Linear` 存储权重的形状是 `(out_features, in_features)`，
> 所以计算时需要转置：$xW^T$，使得矩阵乘法的维度匹配 $(n \times d_{\text{in}}) \times (d_{\text{in}} \times d_{\text{out}}) = (n \times d_{\text{out}})$。

## 数值示例

假设输入 $x = [1, 2, 3]$，权重和偏置为：

$$
W = \begin{bmatrix} 0.1 & 0.2 & 0.3 \\ 0.4 & 0.5 & 0.6 \end{bmatrix}, \quad b = [0.01, 0.02]
$$

计算过程：

$$
xW^T = [1, 2, 3] \begin{bmatrix} 0.1 & 0.4 \\ 0.2 & 0.5 \\ 0.3 & 0.6 \end{bmatrix} = [1.4, 3.2]
$$

$$
y = [1.4, 3.2] + [0.01, 0.02] = [1.41, 3.22]
$$

## NumPy 实现

```python
import numpy as np

def linear(x, weight, bias=None):
    """线性变换: y = x @ weight.T + bias"""
    y = x @ weight.T
    if bias is not None:
        y = y + bias
    return y

# 示例
x = np.array([[1.0, 2.0, 3.0]])
W = np.array([[0.1, 0.2, 0.3],
              [0.4, 0.5, 0.6]])
b = np.array([0.01, 0.02])

y = linear(x, W, b)
print(y)  # [[1.41, 3.22]]
```

## 在 Qwen2-VL 中的应用

线性变换在 Transformer 模型中无处不在：

| 位置 | 输入维度 | 输出维度 | 有偏置？ |
|------|---------|---------|---------|
| Vision MLP fc1 | 1280 | 5120 | ✅ |
| Vision MLP fc2 | 5120 | 1280 | ✅ |
| Text gate_proj | 1536 | 8960 | ❌ |
| Text up_proj | 1536 | 8960 | ❌ |
| Text down_proj | 8960 | 1536 | ❌ |
| Attention Q/K/V/O proj | 各不相同 | 各不相同 | ✅/❌ |

## 验证

运行 `python -m operators.01_linear.impl` 可以验证我们的 NumPy 实现与模型实际输出的一致性。
