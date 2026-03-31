# 13 — 残差连接 (Residual Connection)

> **一句话概括**: 残差连接是一根贯穿整个深度网络的"信息高速公路" — —
> 它让梯度和信号可以不经衰减地流过数十甚至上百层, 是深度学习从"深不下去"到"越深越好"的关键转折点.

---

## 1. 一个建筑师的类比

想象你在设计一栋 100 层的高楼. 每一层都有一个房间, 信息 (水流) 从底层流到顶层.

**没有残差连接**的设计: 水必须穿过每一层的管道系统. 每根管道有些许阻力和损耗.
经过 100 层后, 水几乎流不上去了 — — 即使你在楼顶等得到水, 也无法知道是底层的
哪个水龙头在供水 (因为反向追溯的能力也丧失了).

**有残差连接**的设计: 你在楼的外侧架了一根直通管道 (电梯井), 每一层都有一个
分支口. 水既可以走层内的处理管道 (学习"修正量"), 也可以直接通过电梯井上升.
即使某些楼层的管道堵塞了, 水依然可以顺畅地流到顶楼.

这根"电梯井"就是残差连接.

---

## 2. 历史背景: 深度学习的"深度危机"

### 2.1 更深就更好? 早期的朴素信念

2012 年 AlexNet (8 层) 赢得 ImageNet 竞赛后, 研究者们有了一个朴素的信念:
网络越深, 表达能力越强, 性能应该越好.

2014 年 VGGNet 把网络推到了 19 层, GoogLeNet (Inception) 到了 22 层.
趋势似乎验证了这个信念. 然而 — —

### 2.2 退化问题: 更深反而更差

当 Kaiming He 等人在 2015 年尝试训练 56 层的网络时, 发现了一个令人困惑的现象:

> **56 层网络的训练误差比 20 层网络更高. **

注意, 这不是过拟合 (overfitting)! 过拟合是训练误差低, 测试误差高.
这里连训练误差都变差了, 说明优化本身出了问题 — — 网络根本**学不到**好的参数.

这就是所谓的**退化问题** (degradation problem).

### 2.3 一个关键的思想实验

He 等人提出了一个精妙的反证论证:

> 如果我们取一个训练好的 20 层网络, 在上面堆 36 层**恒等映射**
> (identity mapping, 即什么都不做, 直接输出输入), 得到的 56 层网络
> 至少应该和原来的 20 层一样好 — — 因为额外的层什么都没做.

但实际上 56 层网络更差. 这说明问题不在于表达能力, 而在于**优化的难度**:
让一个卷积层学习恒等映射 $F(x) = x$ 比想象中困难得多.

### 2.4 残差学习的灵感

解决方案优雅得令人叹服: 与其让网络学习目标映射 $H(x)$,
不如让它学习**残差** $F(x) = H(x) - x$. 最终输出为:

$$
H(x) = F(x) + x
$$

如果最优映射接近恒等 ($H(x) \approx x$), 那么网络只需要学习
$F(x) \approx 0$ — — 把所有权重推向零, 比学习恒等映射容易得多.

这就是 2015 年 12 月发表的 **ResNet** (He et al., _"Deep Residual Learning for Image Recognition"_).
ResNet-152 赢得了 ImageNet 2015 冠军, 将错误率降到了 3.57% — — 首次超越人类水平 (约 5%).
这篇论文至今 (2024 年) 已被引用超过 **22 万次**, 是深度学习领域被引用最多的论文之一.

---

## 3. 数学定义

残差连接的数学形式极其简单:

$$
\boxed{y = x + F(x)}
$$

其中:

- $x \in \mathbb{R}^d$: 残差块的输入 ("跳跃连接"那条线传来的信号)
- $F(x) \in \mathbb{R}^d$: 子层 (注意力层, MLP 等) 的输出, 即学到的"修正量"
- $y \in \mathbb{R}^d$: 残差块的输出

**要求**: $x$ 和 $F(x)$ 必须形状相同 (维度匹配) 才能相加.
在 Transformer 中, 这通常是自动满足的, 因为注意力层和 MLP 的输出维度都与输入相同.

> **为什么叫"残差"? ** 在统计学中, residual (残差) 指的是观测值与拟合值之间的差.
> 这里 $F(x) = H(x) - x$ 就是目标输出与输入之间的"差", 即"残差".

---

## 4. 梯度流分析: 为什么残差连接解决了退化问题

这是本节最核心的数学分析.

### 4.1 没有残差连接的梯度流

考虑一个 $L$ 层的网络, 没有残差连接. 设第 $l$ 层的输出为 $x_{l+1} = f_l(x_l)$,
其中 $f_l$ 是第 $l$ 层的变换.

由链式法则, 损失 $\mathcal{L}$ 对第 $0$ 层输入 $x_0$ 的梯度为:

$$
\frac{\partial \mathcal{L}}{\partial x_0}
= \frac{\partial \mathcal{L}}{\partial x_L}
\cdot \frac{\partial x_L}{\partial x_{L-1}}
\cdot \frac{\partial x_{L-1}}{\partial x_{L-2}}
\cdots
\frac{\partial x_1}{\partial x_0}
= \frac{\partial \mathcal{L}}{\partial x_L}
\cdot \prod_{l=0}^{L-1} \frac{\partial f_l(x_l)}{\partial x_l}
$$

**问题**: 如果每一层的 Jacobian $\frac{\partial f_l}{\partial x_l}$ 的谱范数小于 1
(这在使用 sigmoid/tanh 激活函数时很常见), 这个连乘积将**指数衰减**:

$$
\left\| \prod_{l=0}^{L-1} \frac{\partial f_l}{\partial x_l} \right\| \leq \prod_{l=0}^{L-1} \left\| \frac{\partial f_l}{\partial x_l} \right\| \leq \alpha^L
$$

当 $\alpha = 0.9$, $L = 100$ 时, $0.9^{100} \approx 2.66 \times 10^{-5}$ — — 梯度几乎消失.

反过来, 如果谱范数大于 1, 梯度会**指数爆炸**.

### 4.2 有残差连接的梯度流

现在加入残差连接. 第 $l$ 层变为:

$$
x_{l+1} = x_l + F_l(x_l)
$$

对 $x_l$ 求导:

$$
\frac{\partial x_{l+1}}{\partial x_l} = I + \frac{\partial F_l(x_l)}{\partial x_l}
$$

其中 $I$ 是单位矩阵! 这个 $I$ 至关重要.

将 $L$ 层串联, 损失对 $x_0$ 的梯度为:

$$
\frac{\partial \mathcal{L}}{\partial x_0}
= \frac{\partial \mathcal{L}}{\partial x_L}
\cdot \prod_{l=0}^{L-1} \left(I + \frac{\partial F_l(x_l)}{\partial x_l} \right)
$$

展开这个连乘积 (类似多项式展开):

$$
\prod_{l=0}^{L-1} \left(I + J_l \right) = I + \sum_{l} J_l + \sum_{l_1 < l_2} J_{l_1} J_{l_2} + \cdots + \prod_l J_l
$$

其中 $J_l = \frac{\partial F_l}{\partial x_l}$.

**关键观察**: 展开式中包含了单位矩阵 $I$! 这意味着即使所有 $J_l$ 都很小或为零,
梯度至少包含一个 $\frac{\partial \mathcal{L}}{\partial x_L} \cdot I = \frac{\partial \mathcal{L}}{\partial x_L}$ 项
— — 梯度可以**不经任何衰减**地从顶层直达底层.

### 4.3 梯度高速公路

我们可以更直观地理解这一点. 考虑任意两层 $l$ 和 $L$ ($l < L$):

$$
x_L = x_l + \sum_{i=l}^{L-1} F_i(x_i)
$$

对 $x_l$ 求导:

$$
\frac{\partial \mathcal{L}}{\partial x_l}
= \frac{\partial \mathcal{L}}{\partial x_L} \cdot \frac{\partial x_L}{\partial x_l}
= \frac{\partial \mathcal{L}}{\partial x_L} \cdot \left(1 + \frac{\partial}{\partial x_l} \sum_{i=l}^{L-1} F_i(x_i) \right)
$$

第一项 $\frac{\partial \mathcal{L}}{\partial x_L} \cdot 1$ 就是梯度的"直通车" — —
无论中间经过多少层, 梯度都可以无损传播.
第二项 $\frac{\partial \mathcal{L}}{\partial x_L} \cdot \frac{\partial}{\partial x_l} \sum F_i$ 是通过各子层的
"正常"梯度传播路径.

这两条路径的存在使得梯度**不可能完全消失** (除非损失对最终输出的梯度本身为零).

### 4.4 一个定量分析

为了更具体地看效果, 假设每一层 $F_l$ 的 Jacobian 的谱范数为 $\epsilon$ (比较小).

**无残差连接**: 梯度范数 $\approx \epsilon^L$

**有残差连接**: 梯度至少包含 $I$ 项, 范数 $\geq 1 - O(\epsilon)$

| 层数 $L$ | $\epsilon = 0.5$ 无残差      | 有残差 (下界) |
| -------- | ---------------------------- | ------------- |
| 10       | $0.5^{10} \approx 10^{-3}$   | $\geq 0.5$    |
| 50       | $0.5^{50} \approx 10^{-15}$  | $\geq 0.5$    |
| 100      | $0.5^{100} \approx 10^{-30}$ | $\geq 0.5$    |

差异是天文数字般的!

---

## 5. 数值示例: 信号在残差网络中的传播

### 5.1 基本残差加法

```
输入 x       = [1.0,  2.0,  3.0]
子层输出 F(x) = [0.1, -0.2,  0.3]   ← 注意力层学到的"修正量"

残差输出 y   = [1.0 + 0.1,  2.0 + (-0.2),  3.0 + 0.3]
             = [1.1,  1.8,  3.3]
```

观察: 输出 $y$ 和输入 $x$ 非常接近 — — 这正是残差连接的设计意图.
子层只需要学习一个**小的修正量**.

### 5.2 多层传播对比: 有残差 vs 无残差

让我们用一个极简的 1 维例子, 追踪信号通过 5 层的变化.

假设每一层的变换为 $f(x) = 0.8x + 0.1$ (一个略微"缩小"信号的线性变换).
初始输入 $x_0 = 1.0$.

**无残差连接** ($x_{l+1} = f(x_l)$):

```
x_0 = 1.000
x_1 = 0.8 × 1.000 + 0.1 = 0.900
x_2 = 0.8 × 0.900 + 0.1 = 0.820
x_3 = 0.8 × 0.820 + 0.1 = 0.756
x_4 = 0.8 × 0.756 + 0.1 = 0.705
x_5 = 0.8 × 0.705 + 0.1 = 0.664
```

信号从 1.0 衰减到 0.664. 继续下去会收敛到不动点 $x^* = 0.8x^* + 0.1 \Rightarrow x^* = 0.5$.

**有残差连接** ($x_{l+1} = x_l + f(x_l) = x_l + 0.8x_l + 0.1 = 1.8x_l + 0.1$):

```
x_0 = 1.000
x_1 = 1.8 × 1.000 + 0.1 = 1.900
x_2 = 1.8 × 1.900 + 0.1 = 3.520
x_3 = 1.8 × 3.520 + 0.1 = 6.436
x_4 = 1.8 × 6.436 + 0.1 = 11.685
x_5 = 1.8 × 11.685 + 0.1 = 21.133
```

信号在增长! 这说明 $f(x) = 0.8x + 0.1$ 在有残差时不合适 — — 实际网络中 $F(x)$ 的输出
会被归一化层 (LayerNorm / RMSNorm) 控制在合理范围内.

### 5.3 更现实的例子: 带归一化的残差传播

现在假设 $F_l(x) = 0.05 \cdot \sin(x)$ (一个输出很小的非线性变换),
初始输入 $x_0 = 1.0$:

**有残差连接** ($x_{l+1} = x_l + F_l(x_l)$):

```
x_0 = 1.000
x_1 = 1.000 + 0.05 × sin(1.000) = 1.000 + 0.042 = 1.042
x_2 = 1.042 + 0.05 × sin(1.042) = 1.042 + 0.043 = 1.085
x_3 = 1.085 + 0.05 × sin(1.085) = 1.085 + 0.044 = 1.129
x_4 = 1.129 + 0.05 × sin(1.129) = 1.129 + 0.045 = 1.174
x_5 = 1.174 + 0.05 × sin(1.174) = 1.174 + 0.046 = 1.220
```

信号稳定, 缓慢地演变 — — 每一层都在做微小的修正. 这正是深度 Transformer 的工作方式:
每一层都在表示上做小幅度的精炼, 而残差连接保证了信号的主体不会丢失.

### 5.4 反向传播对比

继续上面"无残差"的例子, 计算反向传播的梯度 ($f(x) = 0.8x + 0.1$, $\frac{df}{dx} = 0.8$):

**无残差连接**: $\frac{\partial x_5}{\partial x_0} = 0.8^5 = 0.32768$

**有残差连接**: $\frac{\partial x_{l+1}}{\partial x_l} = 1 + 0.8 = 1.8$,
所以 $\frac{\partial x_5}{\partial x_0} = 1.8^5 = 18.90$

无残差时梯度衰减到 0.33, 有残差时梯度放大到 18.9.
实际网络通过 LayerNorm 和适当的学习率来平衡这个放大效应.

---

## 6. Pre-Norm vs Post-Norm: 归一化层的位置之争

残差连接需要与归一化层 (LayerNorm / RMSNorm) 配合使用.
归一化层放在哪里? 这个看似微小的选择对训练稳定性有巨大影响.

### 6.1 Post-Norm (原始 Transformer, Vaswani et al. 2017)

$$
x_{l+1} = \text{LayerNorm}(x_l + F_l(x_l))
$$

归一化放在残差加法**之后**. 这是原始 Transformer 论文的做法.

```
x → [+ 残差] → LayerNorm → 输出
     ↑
     F(x)
```

**问题**: 归一化层在残差路径上! 梯度流过 LayerNorm 时会被重新缩放,
打破了"梯度直通车"的优势. 训练深层 Post-Norm Transformer 需要非常小心的
学习率预热 (warmup) 策略.

### 6.2 Pre-Norm (现代主流, Xiong et al. 2020; He et al. 2016)

$$
x_{l+1} = x_l + F_l(\text{LayerNorm}(x_l))
$$

归一化放在子层**之前**.

```
x → [+ 残差] → 输出
     ↑
     F(Norm(x))
```

**优势**: 残差路径上没有任何变换 (纯粹的加法), 梯度可以完全无损地流过.
这使得训练深层网络变得容易得多, 不需要复杂的学习率调度.

### 6.3 Qwen2-VL 的选择

Qwen2-VL 在视觉编码器和文本解码器中都使用 **Pre-Norm** 结构:

**视觉编码器** (使用 LayerNorm):

```
x → x + Attention(LayerNorm(x)) → x₁ + MLP(LayerNorm(x₁)) → x₂
```

**文本解码器** (使用 RMSNorm):

```
x → x + Attention(RMSNorm(x)) → x₁ + MLP(RMSNorm(x₁)) → x₂
```

### 6.4 He et al. 2016 的后续研究

ResNet 的原始论文发表一年后, He 等人在 _"Identity Mappings in Deep Residual Networks"_
(2016) 中进一步分析了残差块的设计. 他们提出了 **pre-activation** 残差块:

- **原始 (post-activation) **: $x_{l+1} = \text{ReLU}(x_l + F_l(x_l))$
  — ReLU 在残差加法之后, 会截断负值, 破坏恒等映射
- **Pre-activation**: $x_{l+1} = x_l + F_l(\text{ReLU}(\text{BN}(x_l)))$
  — BN 和 ReLU 在子层内部, 残差路径完全干净

这与后来 Transformer 中的 Pre-Norm vs Post-Norm 之争是同一个原理!
**保持残差路径的"干净"**是关键.

---

## 7. "集成模型"视角: 残差网络的另一种理解

### 7.1 Veit et al. (2016) 的分析

Veit 等人在 _"Residual Networks Behave Like Ensembles of Relatively Shallow Networks"_
中提出了一个引人入胜的观点.

考虑一个 3 层残差网络:

$$
x_3 = x_0 + F_0(x_0) + F_1(x_0 + F_0(x_0)) + F_2(x_0 + F_0(x_0) + F_1(\cdots))
$$

展开后, 我们发现 $x_3$ 可以看作 $2^3 = 8$ 条不同长度路径的叠加:

| 路径 | 经过的层              | 长度 |
| ---- | --------------------- | ---- |
| 1    | 无 (直通)             | 0    |
| 2    | 只有 $F_0$            | 1    |
| 3    | 只有 $F_1$            | 1    |
| 4    | 只有 $F_2$            | 1    |
| 5    | $F_0 \to F_1$         | 2    |
| 6    | $F_0 \to F_2$         | 2    |
| 7    | $F_1 \to F_2$         | 2    |
| 8    | $F_0 \to F_1 \to F_2$ | 3    |

一个 $L$ 层的残差网络实际上是 $2^L$ 条不同路径的**隐式集成** (ensemble)!
而且实验发现, 贡献最大的路径是那些**中等长度**的路径 — — 太短的路径缺乏表达能力,
太长的路径梯度仍然会衰减.

### 7.2 对 Qwen2-VL 的启示

Qwen2-VL 的视觉编码器有 32 个 block, 每个 block 有 2 个残差连接.
这意味着从输入到输出有 $2^{64}$ 条可能的路径 — — 一个天文数字般的隐式集成!
这解释了为什么深度 Transformer 有如此强大的表达能力.

---

## 8. DenseNet 连接 (简述)

残差连接可以看作一种特殊的"密集连接".

在 **DenseNet** (Huang et al., 2017) 中, 每一层的输入不仅包括上一层的输出,
还包括**之前所有层**的输出 (通过拼接 concatenation):

$$
x_l = [x_0, x_1, \ldots, x_{l-1}]
$$

而 **ResNet** 只连接上一层的输入 (通过加法 addition):

$$
x_l = x_{l-1} + F_{l-1}(x_{l-1})
$$

残差连接是密集连接的一种极端简化: 只保留最近的一个跳跃连接, 用加法代替拼接.
加法的优势是不增加维度 (拼接会让特征维度线性增长), 这在 Transformer 中尤为重要.

---

## 9. 在 Qwen2-VL 中的具体应用

### 9.1 每个 Transformer Block 的双残差结构

无论是视觉编码器还是文本解码器, 每个 Transformer Block 都有 **两个** 残差连接:

```
                          ┌──────────────────────────────────┐
block_input ──────────────┤                                  │
      │                   │     残差连接 1（Attention）        │
      ▼                   │                                  │
  LayerNorm / RMSNorm     │                                  │
      │                   │                                  │
      ▼                   │                                  │
  Attention               │                                  │
      │                   │                                  │
      ▼                   │                                  │
   [+ block_input] ◄──────┘                                  │
      │                                                      │
      │ = x₁ (attn_residual)                                │
      │                                                      │
      ├────────────────────────────────────────────┐         │
      │                                            │         │
      ▼                                            │         │
  LayerNorm / RMSNorm                              │         │
      │                                            │         │
      ▼                   残差连接 2（MLP）          │         │
  MLP                                              │         │
      │                                            │         │
      ▼                                            │         │
   [+ x₁] ◄───────────────────────────────────────┘         │
      │                                                      │
      │ = x₂ (mlp_residual) = block_output                  │
      ▼                                                      │
```

用公式表示:

$$
x_1 = x + \text{Attention}(\text{Norm}(x)) \qquad \text{ (第一个残差连接) }
$$

$$
x_2 = x_1 + \text{MLP}(\text{Norm}(x_1)) \qquad \text{ (第二个残差连接) }
$$

### 9.2 信号穿过整个视觉编码器的路径

Qwen2-VL 的视觉编码器有 **32 个 block**. 输入 patch embedding
$x_0 \in \mathbb{R}^{N \times 1280}$ 经过的完整残差路径为:

$$
\begin{aligned}
x_1 &= x_0 + \text{Attn}_0(\text{LN}(x_0)) \\
x_2 &= x_1 + \text{MLP}_0(\text{LN}(x_1)) \\
x_3 &= x_2 + \text{Attn}_1(\text{LN}(x_2)) \\
x_4 &= x_3 + \text{MLP}_1(\text{LN}(x_3)) \\
&\;\;\vdots \\
x_{63} &= x_{62} + \text{Attn}_{31}(\text{LN}(x_{62})) \\
x_{64} &= x_{63} + \text{MLP}_{31}(\text{LN}(x_{63}))
\end{aligned}
$$

共 **64 个残差加法** (32 个 block × 2 个残差连接/block).

等价地写成求和形式:

$$
x_{64} = x_0 + \sum_{b=0}^{31} \Big[\text{Attn}_b(\text{LN}(\cdot)) + \text{MLP}_b(\text{LN}(\cdot)) \Big]
$$

**初始的 patch embedding $x_0$ 一直被保留到最终输出中! **
每一层只是在上面叠加微小的修正.

### 9.3 文本解码器的残差路径

文本解码器有 **28 层**, 同样每层两个残差连接, 使用 RMSNorm 代替 LayerNorm:

$$
\begin{aligned}
h_1 &= h_0 + \text{GQA}(\text{RMSNorm}(h_0)) \\
h_2 &= h_1 + \text{SwiGLU}(\text{RMSNorm}(h_1)) \\
&\;\;\vdots \\
h_{56} &= h_{55} + \text{SwiGLU}(\text{RMSNorm}(h_{55}))
\end{aligned}
$$

共 **56 个残差加法**. 加上视觉编码器的 64 个, 整个 Qwen2-VL 模型中有 **120 个残差连接**.

### 9.4 验证中的残差连接

在我们的实现中, 验证了 Vision Block 0 的第一个残差连接:

```python
# 预期关系：block_input + attn_output == norm2_input
block_input = load_activation("model__visual__blocks__0_input")
attn_output = load_activation("model__visual__blocks__0__attn_output")
expected    = load_activation("model__visual__blocks__0__norm2_input")

actual = block_input + attn_output  # 残差连接就这么简单
assert np.allclose(actual, expected, atol=1e-5)  # ✓ 通过！
```

---

## 10. 常见误解与陷阱

### 误解 1: "残差连接只是一个工程技巧"

**澄清**: 残差连接有深刻的数学基础 — — 它保证了梯度的下界 ($\geq 1$),
提供了集成模型的理论解释, 并且与微分方程 (Neural ODE) 有紧密联系.
事实上, 残差网络可以看作欧拉方法 (Euler method) 对常微分方程的离散化:

$$
x_{l+1} = x_l + F_l(x_l) \quad \longleftrightarrow \quad \frac{dx}{dt} = F(x, t)
$$

这一联系催生了 Neural ODE (Chen et al., 2018) 等后续工作.

### 误解 2: "残差连接解决了梯度消失, 所以可以无限堆叠层数"

**澄清**: 虽然残差连接大大缓解了梯度消失问题, 但无限增加深度仍然面临挑战:

- 表示坍缩 (representation collapse): 过深的网络中, 不同输入的表示可能趋于相似
- 计算成本线性增长
- 实践中需要与归一化, 适当的初始化等技术配合

### 误解 3: "残差连接需要复杂的实现"

**澄清**: 如你所见, 残差连接就是一个加法 `x + F(x)`. 它的深刻在于**设计哲学**,
而非实现复杂度.

### 误解 4: "F(x) 应该学习最终输出"

**澄清**: $F(x)$ 学习的是**残差 (修正量) **, 不是最终输出.
最终输出 $y = x + F(x)$ 是输入加上修正. 这个区别是残差学习的核心思想.

---

## 11. 与相关概念的比较

| 概念                   | 公式                                     | 特点                       |
| ---------------------- | ---------------------------------------- | -------------------------- |
| 残差连接 (ResNet)      | $y = x + F(x)$                           | 简单加法, 最常用           |
| 密集连接 (DenseNet)    | $y = [x_0, x_1, \ldots, x_{l-1}]$        | 拼接所有前层, 特征维度增长 |
| Highway Network        | $y = T(x) \cdot F(x) + (1-T(x)) \cdot x$ | 门控机制, $T(x) \in [0,1]$ |
| Squeeze-and-Excitation | $y = x + \text{SE}(F(x))$                | 通道注意力加权             |
| Skip Connection (一般) | 泛指任何"跳跃"连接                       | 残差连接是其最常见形式     |

**Highway Network** (Srivastava et al., 2015) 是残差连接的"前辈":

$$
y = T(x) \cdot H(x) + (1 - T(x)) \cdot x
$$

其中 $T(x)$ 是学习到的门控函数. 当 $T(x) = 0$ 时退化为恒等映射,
当 $T(x) = 1$ 时退化为普通网络. ResNet 可以看作 $T(x) = 1$ 的 Highway Network
(即把门控去掉, 强制让信息总是通过). 讽刺的是, 更简单的 ResNet 反而效果更好!

---

## 12. NumPy 实现与逐行解析

```python
import numpy as np

def residual_add(x: np.ndarray, f_x: np.ndarray) -> np.ndarray:
    """残差连接: y = x + F(x)

    这可能是整个项目中最简单的函数——只有一行。
    但正如本文所分析的，这一行加法背后蕴含着深刻的数学原理。

    Args:
        x:   残差块的输入（跳跃连接传来的信号）
             在视觉编码器中形状为 (N, 1280)
             在文本解码器中形状为 (1, L, 1536)
        f_x: 子层（Attention 或 MLP）的输出（学到的"修正量"）
             形状与 x 相同

    Returns:
        y = x + f_x，形状与输入相同
    """
    return x + f_x
```

**为什么代码这么简单? **

因为残差连接的全部智慧在于**网络架构设计**, 而非运算本身.
NumPy 的广播机制自动处理了批量维度. 对于 float32 的加法,
不存在精度问题 (不像矩阵乘法那样有大量累加).

**一个完整的 Transformer Block 的残差结构**:

```python
def transformer_block(x, attn_fn, mlp_fn, norm1_fn, norm2_fn):
    """展示残差连接在 Transformer block 中的完整用法。"""
    # 残差连接 1：Attention
    x = residual_add(x, attn_fn(norm1_fn(x)))

    # 残差连接 2：MLP
    x = residual_add(x, mlp_fn(norm2_fn(x)))

    return x
```

---

## 13. 扩展阅读

- **ResNet 原论文**: He et al., _"Deep Residual Learning for Image Recognition"_, CVPR 2016 (2015 年 12 月 arXiv)
- **Pre-activation ResNet**: He et al., _"Identity Mappings in Deep Residual Networks"_, ECCV 2016
- **集成视角**: Veit et al., _"Residual Networks Behave Like Ensembles of Relatively Shallow Networks"_, NeurIPS 2016
- **Pre-Norm Transformer**: Xiong et al., _"On Layer Normalization in the Transformer Architecture"_, ICML 2020
- **Highway Network**: Srivastava et al., _"Training Very Deep Networks"_, NeurIPS 2015
- **Neural ODE**: Chen et al., _"Neural Ordinary Differential Equations"_, NeurIPS 2018
- **DenseNet**: Huang et al., _"Densely Connected Convolutional Networks"_, CVPR 2017

---

## 验证

运行以下命令验证残差连接的正确性:

```bash
python -m operators.13_residual_connection.impl
```

该脚本验证了两件事:

1. **Vision Block 0 的注意力残差连接**: `block_input + attn_output == norm2_input`
2. **合成数据测试**: 随机生成的输入和子层输出, 验证加法的数值精确性
