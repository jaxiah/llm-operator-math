# Gated MLP（门控前馈网络 / SwiGLU）

## 前置知识

在阅读本节之前，请确保理解：

- [线性变换](../01_linear/README.md)：矩阵乘法 $y = xW^T$
- [SiLU 激活函数](../10_silu/README.md)：$\text{SiLU}(x) = x \cdot \sigma(x)$
- [Vision MLP](../11_vision_mlp/README.md)：经典两层前馈网络

---

## 从标准 MLP 到门控 MLP

### 标准 MLP 的局限

在 [Vision MLP](../11_vision_mlp/README.md) 中，我们学习了经典的两层 MLP：

$$
\text{MLP}(x) = W_2 \cdot \text{activation}(W_1 x + b_1) + b_2
$$

这种结构简单有效，但有一个隐含问题：**激活函数一视同仁地作用于所有隐藏维度**。网络无法"选择性地"决定哪些信息通过、哪些被抑制。

### 门控机制的思想

门控 MLP 引入了一个额外的投影来充当 **"门"（gate）**：

- **gate_proj（门）**：决定"让多少信号通过"
- **up_proj（值）**：产生"要通过的信号"
- 两者逐元素相乘，实现选择性过滤

这就像一个水闸：gate 控制闸门开合，up 控制水流量。

---

## 数学定义

### SwiGLU（Swish-Gated Linear Unit）

Qwen2-VL 文本解码器使用的 Gated MLP 基于 **SwiGLU** 机制：

$$
\text{GatedMLP}(x) = \Big(\text{SiLU}(xW_{\text{gate}}^T) \odot xW_{\text{up}}^T\Big) W_{\text{down}}^T
$$

其中：
- $x \in \mathbb{R}^{n \times d}$ 是输入
- $W_{\text{gate}} \in \mathbb{R}^{d_{\text{ff}} \times d}$ 是门控投影权重
- $W_{\text{up}} \in \mathbb{R}^{d_{\text{ff}} \times d}$ 是上投影权重
- $W_{\text{down}} \in \mathbb{R}^{d \times d_{\text{ff}}}$ 是下投影权重
- $\odot$ 表示逐元素乘法（Hadamard 积）
- $\text{SiLU}(z) = z \cdot \sigma(z)$ 是激活函数

注意：文本解码器的线性层 **没有偏置项**。

---

## 数据流

在 Qwen2-VL 文本解码器中：

```
输入 x: (1, 3602, 1536)
    │
    ├─────────────────────────┐
    ▼                         ▼
gate_proj: x @ Wg.T      up_proj: x @ Wu.T
    │                         │
    ▼                         │
(1, 3602, 8960)           (1, 3602, 8960)
    │                         │
    ▼                         │
SiLU 激活                     │
    │                         │
    ▼                         ▼
gate: (1, 3602, 8960)  ⊙  up: (1, 3602, 8960)
    │
    ▼
hidden: (1, 3602, 8960)
    │
    ▼
down_proj: hidden @ Wd.T
    │
    ▼
输出: (1, 3602, 1536)
```

关键特点：gate_proj 和 up_proj **并行计算**，然后逐元素相乘。

---

## 逐步数值示例

假设 $d = 2$，$d_{\text{ff}} = 3$，输入一个 token $x = [1.0, -0.5]$。

### 步骤 1：并行投影

假设权重使得：

$$
\text{gate\_raw} = xW_{\text{gate}}^T = [2.1, -0.8, 1.5]
$$
$$
\text{up} = xW_{\text{up}}^T = [0.6, 1.2, -0.4]
$$

### 步骤 2：门控值（对 gate_raw 应用 SiLU）

$\text{SiLU}(z) = z \cdot \sigma(z)$，其中 $\sigma(z) = \frac{1}{1 + e^{-z}}$：

| $z$ | $\sigma(z)$ | $\text{SiLU}(z)$ |
|-----|-------------|-------------------|
| $2.1$ | $0.8909$ | $1.8709$ |
| $-0.8$ | $0.3100$ | $-0.2480$ |
| $1.5$ | $0.8176$ | $1.2264$ |

$$
\text{gate} = [1.8709, -0.2480, 1.2264]
$$

### 步骤 3：逐元素相乘

$$
\text{hidden} = \text{gate} \odot \text{up} = [1.8709 \times 0.6,\; -0.2480 \times 1.2,\; 1.2264 \times (-0.4)]
$$
$$
= [1.1226, -0.2976, -0.4906]
$$

观察门控效果：
- 第 1 维：gate 值大（1.87），信号被放大
- 第 2 维：gate 值小且为负（-0.25），信号被反转并压缩
- 第 3 维：gate 值正（1.23），但 up 值为负，信号被反转

### 步骤 4：下投影

$$
\text{output} = \text{hidden} \cdot W_{\text{down}}^T
$$

将 3 维压缩回 2 维，得到最终输出。

---

## 为什么 Gated MLP 效果更好？

### 经验发现

Google 研究人员 Noam Shazeer 在 2020 年的论文 *"GLU Variants Improve Transformer"* 中系统比较了多种 MLP 变体，发现：

1. **SwiGLU 和 GeGLU 在多个基准测试上显著优于标准 MLP**
2. 门控机制提供了更好的梯度流动
3. 虽然参数量略增（多一个投影矩阵），但在固定参数预算下仍然更优

### 直觉解释

标准 MLP 中，激活函数对所有维度 "一视同仁"。门控 MLP 增加了一个学到的 "过滤器"：

- 标准 MLP：$\text{activation}(xW_1^T)$ — 所有维度同等处理
- 门控 MLP：$\text{activation}(xW_g^T) \odot (xW_u^T)$ — 门控选择性地放大或抑制每个维度

这让网络能学到更精细的特征组合模式。

---

## 扩展比的对比

| 模型部分 | 输入维度 $d$ | 隐藏维度 $d_{\text{ff}}$ | 扩展比 | 备注 |
|---------|------------|----------------------|--------|------|
| 视觉编码器 | 1280 | 5120 | $4.0\times$ | 经典 Transformer 设计 |
| 文本解码器 | 1536 | 8960 | $\approx 5.83\times$ | 补偿门控的参数效率 |

文本解码器扩展比更大的原因：门控 MLP 有 **三个** 投影矩阵（gate、up、down），而标准 MLP 只有两个（fc1、fc2）。为了保持与标准 MLP 相当的参数量，通常将 $d_{\text{ff}}$ 设为 $\frac{8}{3}d$ 而非 $4d$。Qwen2-VL 中 $8960 / 1536 \approx 5.83$，接近 $\frac{8}{3} \times 2 = 5.33$。

---

## NumPy 实现

```python
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def silu(x):
    return x * sigmoid(x)

def gated_mlp(x, gate_weight, up_weight, down_weight):
    """Gated MLP (SwiGLU): gate/up 并行投影 → 门控 → 下投影"""
    gate = silu(x @ gate_weight.T)   # 门控值
    up = x @ up_weight.T              # 上投影值
    hidden = gate * up                 # 逐元素门控
    output = hidden @ down_weight.T    # 下投影
    return output
```

---

## 在 Qwen2-VL 中的位置

文本解码器的每个层中：

```
Decoder Layer
├── RMSNorm → Self-Attention → 残差连接
└── RMSNorm → Gated MLP (SwiGLU) → 残差连接  ← 这里
```

权重参数：
- `model.layers.0.mlp.gate_proj.weight` — 形状 $(8960, 1536)$
- `model.layers.0.mlp.up_proj.weight` — 形状 $(8960, 1536)$
- `model.layers.0.mlp.down_proj.weight` — 形状 $(1536, 8960)$

参数量：$8960 \times 1536 \times 3 = 41{,}287{,}680 \approx 41.3\text{M}$

---

## Vision MLP vs Gated MLP 总结

| 特性 | Vision MLP | Gated MLP (SwiGLU) |
|------|-----------|---------------------|
| 结构 | $W_2(\text{act}(W_1 x + b_1)) + b_2$ | $(\text{SiLU}(xW_g^T) \odot xW_u^T) W_d^T$ |
| 投影数 | 2 个 (fc1, fc2) | 3 个 (gate, up, down) |
| 激活函数 | QuickGELU | SiLU |
| 偏置 | ✅ 有 | ❌ 无 |
| 门控 | ❌ 无 | ✅ 有（gate × up） |
| 扩展比 | 4x | ~5.8x |
| 使用位置 | 视觉编码器 | 文本解码器 |
| 每层参数量 | ~13.1M | ~41.3M |
