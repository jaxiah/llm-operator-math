# RMS Normalization（均方根归一化）

## 回顾：LayerNorm 做了什么？

在 [03_layer_norm](../03_layer_norm/README.md) 中我们学到，LayerNorm 的公式是：

$$
y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta
$$

它包含两个步骤：
1. **中心化**（减去均值 $\mu$）
2. **缩放**（除以标准差）

## RMSNorm：去掉中心化步骤

2019 年的论文 *"Root Mean Square Layer Normalization"* 提出了一个关键观察：**LayerNorm 的效果主要来自缩放操作，中心化（减均值）的贡献很小**。

因此 RMSNorm 直接跳过均值计算，只用 **均方根（Root Mean Square）** 来归一化：

$$
y = \frac{x}{\text{RMS}(x)} \cdot \gamma
$$

其中 RMS 的定义为：

$$
\text{RMS}(x) = \sqrt{\frac{1}{n} \sum_{i=1}^{n} x_i^2 + \epsilon}
$$

注意：
- **没有减均值**的步骤
- **没有 bias（偏置）** 参数 $\beta$
- 只有一个可学习参数 $\gamma$（weight）

## 公式展开

把两步合在一起写：

$$
y_i = \frac{x_i}{\sqrt{\frac{1}{n} \sum_{j=1}^{n} x_j^2 + \epsilon}} \cdot \gamma_i
$$

## 手算示例

假设输入向量 $x = [1.0, 2.0, 3.0, 4.0]$，权重 $\gamma = [1.0, 1.0, 1.0, 1.0]$，$\epsilon = 10^{-6}$。

**第一步：计算 $x^2$ 的均值**

$$
\text{mean}(x^2) = \frac{1^2 + 2^2 + 3^2 + 4^2}{4} = \frac{1 + 4 + 9 + 16}{4} = 7.5
$$

**第二步：计算 RMS**

$$
\text{RMS}(x) = \sqrt{7.5 + 10^{-6}} \approx 2.7386
$$

**第三步：归一化并缩放**

$$
y = \frac{x}{2.7386} \cdot \gamma = \left[\frac{1}{2.7386},\ \frac{2}{2.7386},\ \frac{3}{2.7386},\ \frac{4}{2.7386}\right] \approx [0.3651,\ 0.7303,\ 1.0954,\ 1.4606]
$$

### numpy 验证

```python
import numpy as np

x = np.array([1.0, 2.0, 3.0, 4.0])
eps = 1e-6

rms = np.sqrt(np.mean(x ** 2) + eps)  # 2.7386
y = x / rms

print(y)  # [0.3651  0.7303  1.0954  1.4606]
```

## LayerNorm vs RMSNorm 对比

| 特性 | LayerNorm | RMSNorm |
|------|-----------|---------|
| 中心化（减均值） | ✅ 有 | ❌ 无 |
| 缩放（除以标准差/RMS） | ✅ 有 | ✅ 有 |
| bias 参数 ($\beta$) | ✅ 有 | ❌ 无 |
| weight 参数 ($\gamma$) | ✅ 有 | ✅ 有 |
| 计算量 | 较大（需要算均值和方差） | 较小（只算平方的均值） |
| 公式 | $\frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$ | $\frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \cdot \gamma$ |

**为什么 RMSNorm 更快？**
- 少算一次均值（$\mu$）
- 少一次减法运算（$x - \mu$）
- 参数更少（没有 bias）
- 在大模型中，这些节省累积起来非常可观

## 在 Qwen2-VL 中的应用

Qwen2-VL 的 **Vision Encoder** 使用 LayerNorm，而 **Text Decoder**（语言模型部分）使用 RMSNorm。

这反映了大语言模型的趋势：LLaMA、Qwen 等现代大模型的 decoder 几乎都用 RMSNorm 替代了 LayerNorm。

例如，Text Decoder 第一层的 `input_layernorm`：

- 输入形状：`(1, 3602, 1536)` —— 1 个样本，3602 个 token，每个 token 有 1536 维特征
- RMSNorm 对每个 token 的 1536 维特征独立做归一化
- 可学习参数：`weight` 形状 `(1536,)`，无 bias

## 关键要点

| 特性 | 说明 |
|------|------|
| 归一化维度 | 最后一个维度（特征维度） |
| 可学习参数 | 仅 weight ($\gamma$)，无 bias |
| $\epsilon$ | 通常为 $10^{-6}$ |
| 核心公式 | $x / \sqrt{\text{mean}(x^2) + \epsilon} \cdot \gamma$ |
| 优势 | 比 LayerNorm 更快，效果相当 |
