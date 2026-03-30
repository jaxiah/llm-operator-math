# 06 — 旋转位置编码 (Rotary Position Embedding, RoPE)

## 直觉：为什么需要位置编码？

Transformer 的核心操作——注意力机制——本质上是"看一堆向量，给它们打分"。但它有一个致命缺陷：
**它根本不知道谁在前、谁在后**。把句子 "猫追狗" 的 token 打乱成 "狗追猫"，注意力算出来的分数完全一样。

所以我们需要一种方式，**让模型知道每个 token 的位置**。这就是位置编码（Position Encoding）的作用。

早期方法是直接把位置信息加到输入上（如正弦位置编码）。但 RoPE 更聪明：**它把位置信息编码进注意力的 Q 和 K 向量中，让两个 token 的注意力分数自然地依赖于它们的相对距离**。

## 从复数到旋转

### 核心思想：用复数表示旋转

回忆复数乘法的几何意义：把一个复数乘以 $e^{i\theta}$，就是把它在复平面上旋转 $\theta$ 角度。

$$
(a + bi) \cdot e^{i\theta} = (a + bi)(\cos\theta + i\sin\theta)
$$

展开得：

$$
= (a\cos\theta - b\sin\theta) + (a\sin\theta + b\cos\theta)i
$$

写成矩阵形式：

$$
\begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} a \\ b \end{bmatrix} = \begin{bmatrix} a\cos\theta - b\sin\theta \\ a\sin\theta + b\cos\theta \end{bmatrix}
$$

**这就是 2D 旋转矩阵！**

RoPE 的做法：把向量的每两个相邻维度当作一个复数，然后根据 **位置** 给它旋转不同角度。

## 1D RoPE 详细推导

### 第一步：定义频率

对于 $d$ 维的 head（$d = $ `head_dim`），我们定义一组频率：

$$
\theta_i = \text{base}^{-2i/d}, \quad i = 0, 1, \ldots, d/2 - 1
$$

在 Qwen2-VL 中，$\text{base} = 1{,}000{,}000$（注意：不是常见的 10000）。

当 $d = 128$ 时，这产生 64 个频率值，从 $\theta_0 = 1.0$ 到 $\theta_{63} \approx 10^{-6}$。

- **低频**（$\theta$ 接近 1）：角度变化快，编码精细的位置差异
- **高频**（$\theta$ 接近 0）：角度变化慢，编码粗粒度的位置关系

### 第二步：计算角度

给定位置 $p$（整数 0, 1, 2, ...），每个频率对应一个旋转角度：

$$
\alpha_i(p) = p \cdot \theta_i
$$

### 第三步：生成 cos 和 sin

$$
\cos\_\text{cache}[p, i] = \cos(p \cdot \theta_i), \quad \sin\_\text{cache}[p, i] = \sin(p \cdot \theta_i)
$$

形状为 `(max_position, head_dim / 2)`，然后沿最后一维复制一次变成 `(max_position, head_dim)`。

### 第四步：应用旋转

这里是关键——HuggingFace 实现中，旋转不是逐对元素做 2×2 矩阵乘法，而是用一个巧妙的等价公式：

$$
\text{RoPE}(x) = x \cdot \cos + \text{rotate\_half}(x) \cdot \sin
$$

其中 `rotate_half` 把向量分成前半和后半，然后重排：

$$
\text{rotate\_half}([x_0, x_1, \ldots, x_{d/2-1}, x_{d/2}, \ldots, x_{d-1}]) = [-x_{d/2}, \ldots, -x_{d-1}, x_0, \ldots, x_{d/2-1}]
$$

**为什么这和旋转矩阵是等价的？** 展开验证一下：

对于第 $i$ 个位置对 $(x_i, x_{i+d/2})$：

$$
\begin{aligned}
y_i &= x_i \cdot \cos(\alpha_i) + (-x_{i+d/2}) \cdot \sin(\alpha_i) \\
    &= x_i \cos(\alpha_i) - x_{i+d/2} \sin(\alpha_i) \\[6pt]
y_{i+d/2} &= x_{i+d/2} \cdot \cos(\alpha_i) + x_i \cdot \sin(\alpha_i) \\
           &= x_i \sin(\alpha_i) + x_{i+d/2} \cos(\alpha_i)
\end{aligned}
$$

这正好是旋转矩阵的结果！✅

### 关键性质：相对位置编码

对 Q（位置 $m$）和 K（位置 $n$）应用 RoPE 后，它们的点积只依赖于 $m - n$：

$$
\langle \text{RoPE}(q, m), \text{RoPE}(k, n) \rangle = \langle \text{RoPE}(q, m-n), k \rangle
$$

**证明思路**：旋转矩阵 $R$ 满足 $R_m^T R_n = R_{m-n}$，因此 $q^T R_m^T R_n k = q^T R_{m-n} k$。

这意味着：无论两个 token 在序列中的绝对位置如何，只要它们的距离相同，注意力分数就相同。这就是"旋转位置编码"名字的来历——**用旋转来编码相对位置**。

## 手算示例

假设 `head_dim = 4`，`base = 10000`，位置 $p = 2$，输入向量 $x = [1.0, 2.0, 3.0, 4.0]$。

**第一步：计算频率**

$$
\theta_0 = 10000^{-0/4} = 1.0, \quad \theta_1 = 10000^{-2/4} = 10000^{-0.5} = 0.01
$$

**第二步：计算角度**

$$
\alpha_0 = 2 \times 1.0 = 2.0, \quad \alpha_1 = 2 \times 0.01 = 0.02
$$

**第三步：cos 和 sin（扩展到 head_dim）**

$$
\cos = [\cos(2.0), \cos(0.02), \cos(2.0), \cos(0.02)] \approx [-0.4161, 0.9998, -0.4161, 0.9998]
$$

$$
\sin = [\sin(2.0), \sin(0.02), \sin(2.0), \sin(0.02)] \approx [0.9093, 0.0200, 0.9093, 0.0200]
$$

**第四步：rotate_half**

$$
\text{rotate\_half}(x) = [-x_2, -x_3, x_0, x_1] = [-3.0, -4.0, 1.0, 2.0]
$$

**第五步：应用公式**

$$
y = x \cdot \cos + \text{rotate\_half}(x) \cdot \sin
$$

$$
\begin{aligned}
y_0 &= 1.0 \times (-0.4161) + (-3.0) \times 0.9093 = -0.4161 - 2.7279 = -3.1440 \\
y_1 &= 2.0 \times 0.9998 + (-4.0) \times 0.0200 = 1.9996 - 0.0800 = 1.9196 \\
y_2 &= 3.0 \times (-0.4161) + 1.0 \times 0.9093 = -1.2484 + 0.9093 = -0.3391 \\
y_3 &= 4.0 \times 0.9998 + 2.0 \times 0.0200 = 3.9992 + 0.0400 = 4.0392
\end{aligned}
$$

### numpy 验证

```python
import numpy as np

head_dim = 4
base = 10000
p = 2
x = np.array([1.0, 2.0, 3.0, 4.0])

# 频率
inv_freq = base ** (-np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)
# [1.0, 0.01]

# 角度
angles = p * inv_freq  # [2.0, 0.02]

# cos/sin (扩展到 head_dim)
cos_val = np.cos(np.concatenate([angles, angles])).astype(np.float32)
sin_val = np.sin(np.concatenate([angles, angles])).astype(np.float32)

# rotate_half
x_rot = np.concatenate([-x[head_dim//2:], x[:head_dim//2]])

# 应用 RoPE
y = x * cos_val + x_rot * sin_val
print(y)  # [-3.1440  1.9196 -0.3391  4.0392]
```

## 验证旋转保持向量长度

旋转矩阵是正交矩阵，所以旋转不改变向量的范数：

$$
\|Rx\| = \|x\|
$$

```python
np.linalg.norm(y)   # ≈ 5.4772
np.linalg.norm(x)   # = 5.4772  ✅
```

## 2D 视觉 RoPE

### 为什么视觉需要 2D？

文本是一维序列（token 0, 1, 2, ...），但图像 patch 有 **高度** 和 **宽度** 两个空间维度。
如果简单地把 2D patch 展平成 1D 序列，相邻行的 patch 在序列中距离很远，模型就丢失了空间邻接信息。

### 实现方式：拆分 head_dim

Qwen2-VL 视觉编码器的做法：

1. 把 `head_dim` 对半分成两部分
2. **前半部分**：使用 height 坐标做 1D RoPE
3. **后半部分**：使用 width 坐标做 1D RoPE

例如 `head_dim = 80`：
- 前 40 维和后 40 维配对（rotate_half 将 dim $i$ 与 dim $i+40$ 配对）
- 每对中：前 20 个频率用 height 坐标，后 20 个频率用 width 坐标

对于位置 $(h, w)$ 的 patch：

$$
\text{cos} = [\cos(h \cdot \theta_0), \ldots, \cos(h \cdot \theta_{19}),\; \cos(w \cdot \theta_0), \ldots, \cos(w \cdot \theta_{19})]
$$

### grid_thw 参数

视觉模型接收 `grid_thw` 参数（temporal, height, width），描述每段视频/图像的网格尺寸。
例如 `grid_thw = [[1, 24, 24]]` 表示 1 帧、高 24 个 patch、宽 24 个 patch。

## 3D 多模态 RoPE（M-RoPE）

### Qwen2-VL 的独特设计

Qwen2-VL 的语言模型部分使用了 **M-RoPE（Multimodal Rotary Position Embedding）**，
将位置编码扩展到三个维度：**时间 × 高度 × 宽度**。

### mrope_section 机制

配置文件中的 `rope_scaling.mrope_section = [16, 24, 24]` 定义了三个维度各自使用的频率数量：

| 维度 | mrope_section 值 | 频率数量 | 对应的 head_dim 分段大小 |
|------|-------------------|----------|--------------------------|
| 时间 (temporal) | 16 | 16 | 32（= 2 × 16） |
| 高度 (height) | 24 | 24 | 48（= 2 × 24） |
| 宽度 (width) | 24 | 24 | 48（= 2 × 24） |
| **合计** | **64** | **64** | **128**（= head_dim） |

每个频率对应 2 个 head_dim 维度（因为一个频率对应一对实数 = 一个复数旋转）。

### position_ids 的结构

`position_ids` 形状为 `(3, seq_len)`，三行分别是：
- `position_ids[0]`：temporal 维度的位置
- `position_ids[1]`：height 维度的位置
- `position_ids[2]`：width 维度的位置

**对于纯文本 token**：三个维度的位置相同（都是普通的 token 位置 0, 1, 2, ...）。

**对于视觉 token**：
- temporal = 帧索引
- height = patch 的行坐标
- width = patch 的列坐标

### 计算过程

1. 生成频率：$\theta_i = \text{base}^{-2i/d}$，$d = $ `head_dim`，$\text{base} = 1{,}000{,}000$
2. 按 `mrope_section` 拆分频率：前 16 个给时间，中间 24 个给高度，后 24 个给宽度
3. 每个维度用自己的 `position_ids` 和对应的频率做外积，得到角度
4. 拼接三个维度的 cos/sin，应用到 Q 和 K

```python
# 伪代码
inv_freq = base ** (-arange(0, head_dim, 2) / head_dim)  # (64,)

# 按 section 拆分频率
freq_t = inv_freq[0:16]         # 时间频率
freq_h = inv_freq[16:40]        # 高度频率
freq_w = inv_freq[40:64]        # 宽度频率

# 每个维度计算角度
angles_t = position_ids[0] @ freq_t  # (seq_len, 16)
angles_h = position_ids[1] @ freq_h  # (seq_len, 24)
angles_w = position_ids[2] @ freq_w  # (seq_len, 24)

# 拼接并复制
angles = concat([angles_t, angles_h, angles_w])  # (seq_len, 64)
cos = cos(concat([angles, angles]))               # (seq_len, 128)
sin = sin(concat([angles, angles]))               # (seq_len, 128)
```

## 完整旋转操作

无论是 1D、2D 还是 3D RoPE，最终应用到 Q/K 的方式相同：

$$
\boxed{q_{\text{rotated}} = q \cdot \cos + \text{rotate\_half}(q) \cdot \sin}
$$

$$
\boxed{k_{\text{rotated}} = k \cdot \cos + \text{rotate\_half}(k) \cdot \sin}
$$

其中 `cos` 和 `sin` 的形状会通过广播匹配 Q/K 的形状。

## 关键要点

| 特性 | 说明 |
|------|------|
| 作用 | 在 Q/K 中编码位置信息 |
| 核心操作 | 对每对维度做旋转（基于位置和频率） |
| 相对 vs 绝对 | 编码的是绝对位置，但注意力分数只依赖相对位置 |
| rotate_half | $[-x_{d/2:},\; x_{:d/2}]$——等价于复数旋转 |
| 1D RoPE | 文本序列：位置 0, 1, 2, ... |
| 2D RoPE | 视觉 patch：(height, width) 位置，head_dim 对半分 |
| 3D M-RoPE | 多模态：(temporal, height, width)，按 mrope_section 分段 |
| base | Qwen2-VL 使用 $10^6$（语言模型），视觉编码器可能不同 |
| head_dim | 语言模型 128，视觉编码器 80 |
| 保持范数 | 旋转不改变向量长度（正交变换） |
