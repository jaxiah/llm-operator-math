# Vision Transformer Block (视觉编码器块)

> **一句话概括**: Vision Block 是视觉编码器的最小"思考回合" — — 先让所有 patch 互相交流 (Attention), 再让每个 patch 独立深加工 (MLP), 如此循环 32 次, 原始像素便逐步演化为高级语义.

---

## 为什么需要 Vision Block?

试着回忆你看一幅陌生照片时的心理过程. 你的眼睛不会一次把所有像素都理解透 — — 你会经历多"轮"加工:

1. **第一轮**: 你注意到边缘, 颜色块, 纹理方向等最底层的视觉特征.
2. **第二轮**: 你开始把相邻的边缘组合成轮廓, 把颜色块组合成区域.
3. **第三轮**: 轮廓和区域进一步整合, 你认出了一只耳朵, 一个轮子.
4. **更多轮**: 耳朵和身体组合成"猫", 轮子和车身组合成"汽车".

每一"轮"都做了两件事:

- **环顾四周**: 把注意力投向画面的不同位置, 收集相关信息 (这对应 Attention).
- **内部消化**: 根据收集到的信息, 更新自己对当前位置的理解 (这对应 MLP).

Vision Block 就是对这"一轮加工"的精确数学建模. Qwen2-VL 的视觉编码器堆叠了 **32 个** 这样的 Block, 每个 Block 的结构完全相同, 只是权重不同 — — 就像同一个"思考模板"被反复使用, 但每次关注的层次不同.

---

## 前置知识

在阅读本节之前, 请确保理解以下五个基础模块. 它们是 Vision Block 的"零件", 本文会把它们组装起来:

| 模块                         | 核心公式                                                                         | 链接                                                          |
| ---------------------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| Layer Normalization          | $\text{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$ | [03_layer_norm](../03_layer_norm/README.md)                   |
| Scaled Dot-Product Attention | $\text{Attn}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$  | [07_attention](../07_attention/README.md)                     |
| QuickGELU 激活函数           | $\text{QuickGELU}(x) = x \cdot \sigma(1.702x)$                                   | [08_quickgelu](../08_quickgelu/README.md)                     |
| Vision MLP                   | $\text{MLP}(x) = W_2\,\text{QuickGELU}(W_1 x + b_1) + b_2$                       | [11_vision_mlp](../11_vision_mlp/README.md)                   |
| 残差连接                     | $y = x + F(x)$                                                                   | [13_residual_connection](../13_residual_connection/README.md) |

如果你对上面任何一项还不熟悉, 建议先阅读对应的章节再回来 — — 本文会假设你已经掌握了它们的数学细节.

---

## 历史背景: 从 Transformer 到 Vision Transformer

要理解 Vision Block, 我们需要先回溯它的思想源头.

### Transformer 的诞生

2017 年, Vaswani 等人发表了里程碑式的论文 _"Attention Is All You Need"_. 在此之前, 序列建模的主流工具是循环神经网络 (RNN) 和长短时记忆网络 (LSTM). 这些网络有一个根本性的限制: 它们必须**逐步**处理序列中的每个元素, 无法并行化.

Transformer 的核心洞察是: **完全抛弃循环结构, 仅使用注意力机制来捕捉序列中任意两个位置之间的关系**. 原始 Transformer 的编码器由 $N = 6$ 个相同结构的 Block 堆叠而成, 每个 Block 包含:

1. 一个 Multi-Head Self-Attention 子层
2. 一个 Position-wise Feed-Forward Network (即 MLP) 子层

每个子层后面跟着残差连接和 Layer Normalization. 这个"Attention + MLP"的双子层结构, 就是我们今天要深入剖析的 **Transformer Block**.

### 从文本到视觉: ViT 的跨越

Transformer 最初是为自然语言处理 (NLP) 设计的. 2020 年, Dosovitskiy 等人发表了 _"An Image is Worth 16×16 Words"_ (即 ViT 论文), 提出了一个大胆的想法: **把图像切成小方块 (patch), 当作"单词"输入 Transformer**.

具体做法是:

1. 将一张图像切成 $16 \times 16$ 像素的 patch
2. 将每个 patch 线性投影为一个向量 (类似 word embedding)
3. 把这些向量作为序列输入标准的 Transformer 编码器

这个简单的想法效果出奇地好 — — ViT 在大规模数据上训练后, 超越了当时所有的卷积神经网络 (CNN).

### Qwen2-VL 的视觉编码器

Qwen2-VL 继承了 ViT 的思路, 但做了一些现代化的改进:

- **深度从 6 到 32**: 原始 Transformer 只有 6 个 Block, 而 Qwen2-VL 的视觉编码器堆叠了 **32 个** Block. 更深的网络能学习更复杂的视觉表示.
- **Pre-Norm 替代 Post-Norm**: 改变了 LayerNorm 的位置, 显著提升训练稳定性 (后文详述).
- **窗口注意力**: 引入局部窗口机制, 大幅降低计算复杂度 (后文详述).
- **RoPE 位置编码**: 使用旋转位置编码来表达 2D 空间位置关系.

---

## 什么是 Vision Block?

如果说深度学习是一座大厦, 那么 **Block** 就是这座大厦的"砖块" — — 它是最小的, 可重复使用的建筑单元.

### 一个 Block = 一轮"思考"

每个 Vision Block 完成两件事, 对应两个子层:

**子层 1: Attention — "讨论会"**

想象一间会议室里坐着 $n$ 个人 ($n$ 个 patch token), 每个人手中有一份报告 (一个 $d$ 维向量). 讨论会上, 每个人都可以"看"到其他人的报告, 然后根据相关性来更新自己的理解. 这就是 Self-Attention 的本质 — — **让 token 之间互相交流信息**.

讨论会结束后, 每个人的报告都融合了来自其他人的信息. 一个原本只描述"左上角有一条线"的 token, 现在也知道了"右边那条线和我差不多方向" — — 这种跨位置的信息整合是 Attention 的核心价值.

**子层 2: MLP — "独立思考"**

讨论会结束后, 每个人回到自己的办公室, 独立地消化, 加工刚刚收集到的信息. 这就是 MLP 子层 — — 它**逐个 token 独立地**进行非线性变换, 不涉及 token 之间的交互.

MLP 的价值在于**深度加工**: 它先把信息投影到一个更高维的空间 ($1280 \to 5120$, 即 $4\times$ 扩展), 在高维空间中通过非线性激活 (QuickGELU) 进行复杂变换, 然后再投影回原始维度 ($5120 \to 1280$). 你可以把它想象成"用更大的白板来思考, 然后把结论写回原来的笔记本".

### 为什么两个子层缺一不可?

只有 Attention 没有 MLP: token 之间可以交流信息, 但无法对信息做非线性的深度加工. 网络的表达能力会受到严重限制.

只有 MLP 没有 Attention: 每个 token 只能看到自己, 对周围的 token 一无所知. 这样的网络无法捕捉空间关系.

**Attention 提供"广度" (跨位置的信息交流), MLP 提供"深度" (逐位置的非线性变换). 两者结合, 才构成一个完整的"思考回合". **

---

## Pre-Norm vs Post-Norm: 归一化放在哪里?

这是 Transformer 架构演化中最重要的一个细节改动, 值得深入理解.

### Post-Norm: 原始方案

原始 Transformer (Vaswani et al., 2017) 采用的是 **Post-Norm** — — 先做子层变换, 再做残差连接和归一化:

$$
y = \text{LayerNorm}(x + F(x))
$$

展开为两个子层就是:

$$
x' = \text{LN}_1(x + \text{Attention}(x))
$$

$$
y = \text{LN}_2(x' + \text{MLP}(x'))
$$

这里 $F$ 代表子层函数 (Attention 或 MLP), LayerNorm 在残差相加之后应用.

### Pre-Norm: 现代标准

现代大模型几乎全部采用 **Pre-Norm** — — 先归一化, 再做子层变换, 最后残差相加:

$$
y = x + F(\text{LayerNorm}(x))
$$

展开为两个子层就是:

$$
x' = x + \text{Attention}(\text{LN}_1(x))
$$

$$
y = x' + \text{MLP}(\text{LN}_2(x'))
$$

注意关键区别: 在 Pre-Norm 中, **残差连接直接把原始输入 $x$ 加到输出上**, LayerNorm 只作用在子层的输入上.

### 梯度分析: 为什么 Pre-Norm 更好?

让我们从反向传播的角度来分析.

**Post-Norm 的梯度路径**:

$$
\frac{\partial \text{LN}(x + F(x))}{\partial x} = \frac{\partial \text{LN}}{\partial (x + F(x))} \cdot \left(I + \frac{\partial F}{\partial x}\right)
$$

梯度必须**穿过 LayerNorm 层**才能到达 $x$. LayerNorm 对梯度有缩放和旋转作用, 当 Block 数量很多时 (比如 32 个), 梯度要连续穿过 32 个 LN 层, 其尺度可能变得不稳定.

**Pre-Norm 的梯度路径**:

$$
\frac{\partial [x + F(\text{LN}(x))]}{\partial x} = I + \frac{\partial F}{\partial \text{LN}(x)} \cdot \frac{\partial \text{LN}}{\partial x}
$$

注意这里有一个**恒等项 $I$**! 无论 $F$ 和 LN 的梯度如何变化, 这个 $I$ 始终存在. 它就像一条"梯度高速公路", 保证信息可以畅通无阻地从最后一个 Block 流回第一个 Block.

### 直觉解释

想象你在一栋 32 层的大楼里传递消息.

- **Post-Norm** 就像每层楼都有一个检查站, 消息必须经过检查才能传到下一层. 经过 32 个检查站后, 消息可能面目全非.
- **Pre-Norm** 就像大楼中间有一部直通电梯. 消息既可以走电梯 (恒等路径), 也可以走楼梯 (经过变换). 无论楼梯多拥挤, 电梯始终通畅.

### 训练稳定性

Xiong et al. (2020) 在论文 _"On Layer Normalization in the Transformer Architecture"_ 中严格证明了:

- **Post-Norm** 在训练初期梯度方差很大, 必须使用 **learning rate warmup** (先用很小的学习率, 慢慢增大) 才能稳定训练.
- **Pre-Norm** 从训练一开始就很稳定, **不需要 warmup**.

### 对比总结

| 特性         | Post-Norm             | Pre-Norm                    |
| ------------ | --------------------- | --------------------------- |
| 公式         | $\text{LN}(x + F(x))$ | $x + F(\text{LN}(x))$       |
| 论文         | Vaswani et al., 2017  | Xiong et al., 2020          |
| 梯度路径     | 必须穿过 LN           | 有恒等跳跃通道              |
| 训练稳定性   | 需要 warmup           | 天然稳定                    |
| 深层网络表现 | 容易不稳定            | 可稳定训练数百层            |
| 最终性能     | 精心调参后可略优      | 略低但非常接近              |
| 采用情况     | 早期模型              | GPT, LLaMA, Qwen 等现代模型 |

Qwen2-VL 的视觉编码器采用的正是 **Pre-Norm** 结构.

---

## 数学定义

现在让我们正式写出 Vision Block 的完整数学公式.

### 宏观公式

给定输入 $x \in \mathbb{R}^{n \times d}$ ($n$ 个 token, 每个 $d$ 维), Vision Block 的计算分为两个子层:

$$
x' = x + \text{Attention}(\text{LN}_1(x))
$$

$$
y = x' + \text{MLP}(\text{LN}_2(x'))
$$

在 Qwen2-VL 中, $d = 1280$.

### 六步展开

让我们把上面的宏观公式展开为 6 个具体的计算步骤:

**步骤 1: 第一次 Layer Normalization**

$$
\hat{x} = \text{LN}_1(x) = \frac{x - \mu_1}{\sqrt{\sigma_1^2 + \epsilon}} \odot \gamma_1 + \beta_1
$$

其中 $\gamma_1, \beta_1 \in \mathbb{R}^{1280}$ 是可学习参数, $\mu_1$ 和 $\sigma_1^2$ 是每个 token 在 1280 个维度上的均值和方差, $\epsilon = 10^{-6}$ 防止除零, $\odot$ 表示逐元素乘法.

输入形状: $(n, 1280)$. 输出形状: $(n, 1280)$.

**步骤 2: Multi-Head Self-Attention**

首先做 QKV 投影:

$$
[Q; K; V] = \hat{x} \, W_{qkv}^\top + b_{qkv}
$$

其中 $W_{qkv} \in \mathbb{R}^{3840 \times 1280}$, $b_{qkv} \in \mathbb{R}^{3840}$. 这里 $3840 = 3 \times 1280$, 一次矩阵乘法同时得到 $Q, K, V$.

然后将 $Q, K, V$ 各自拆分为 $h = 16$ 个头, 每个头的维度为 $d_k = 80$:

$$
Q_i, K_i, V_i \in \mathbb{R}^{n \times 80}, \quad i = 1, \ldots, 16
$$

对每个头计算缩放点积注意力:

$$
\text{head}_i = \text{softmax}\!\left(\frac{Q_i K_i^\top}{\sqrt{80}}\right) V_i
$$

将所有头拼接后通过输出投影:

$$
\text{attn\_out} = \text{Concat}(\text{head}_1, \ldots, \text{head}_{16}) \, W_{\text{proj}}^\top + b_{\text{proj}}
$$

其中 $W_{\text{proj}} \in \mathbb{R}^{1280 \times 1280}$, $b_{\text{proj}} \in \mathbb{R}^{1280}$.

输出形状: $(n, 1280)$.

**步骤 3: 第一次残差连接**

$$
x' = x + \text{attn\_out}
$$

注意: 这里加的是**原始的 $x$**, 不是归一化后的 $\hat{x}$. 这就是 Pre-Norm 的精髓 — — 归一化只影响子层的输入, 不影响残差路径.

输出形状: $(n, 1280)$.

**步骤 4: 第二次 Layer Normalization**

$$
\hat{x}' = \text{LN}_2(x') = \frac{x' - \mu_2}{\sqrt{\sigma_2^2 + \epsilon}} \odot \gamma_2 + \beta_2
$$

这里的 $\gamma_2, \beta_2$ 是**独立于** $\gamma_1, \beta_1$ 的另一组参数.

输出形状: $(n, 1280)$.

**步骤 5: MLP (前馈网络) **

$$
h = \hat{x}' \, W_1^\top + b_1 \quad \in \mathbb{R}^{n \times 5120}
$$

$$
h = \text{QuickGELU}(h) = h \odot \sigma(1.702 \, h) \quad \in \mathbb{R}^{n \times 5120}
$$

$$
\text{mlp\_out} = h \, W_2^\top + b_2 \quad \in \mathbb{R}^{n \times 1280}
$$

其中 $W_1 \in \mathbb{R}^{5120 \times 1280}$, $b_1 \in \mathbb{R}^{5120}$, $W_2 \in \mathbb{R}^{1280 \times 5120}$, $b_2 \in \mathbb{R}^{1280}$.

MLP 的隐藏维度 $5120 = 4 \times 1280$, 这个 $4\times$ 扩展比例是 Transformer 中的标准设计.

输出形状: $(n, 1280)$.

**步骤 6: 第二次残差连接**

$$
y = x' + \text{mlp\_out}
$$

输出形状: $(n, 1280)$.

**最终输出** $y \in \mathbb{R}^{n \times 1280}$ 与输入 $x$ 的形状完全相同 — — 这是 Block 可以堆叠的前提.

---

## 残差连接的深层意义

残差连接不只是一个技巧, 它是深度网络能够工作的根本保障. 让我们深入理解为什么**每个子层后面都需要独立的残差连接**.

### 梯度公式

对于残差结构 $y = x + F(x)$, 反向传播时:

$$
\frac{\partial y}{\partial x} = I + \frac{\partial F(x)}{\partial x}
$$

这个公式有两个关键含义:

1. **恒等项 $I$**: 即使 $\frac{\partial F}{\partial x}$ 趋近于零 (梯度消失), 梯度仍然至少为 $I$, 信息不会丢失.
2. **加法而非乘法**: 如果没有残差连接, 多层堆叠的梯度是连乘关系, 容易指数级衰减或爆炸. 有了残差连接, 梯度变成了类似求和的关系.

### 为什么每个子层都需要残差?

一个 Block 有两个子层, 为什么不能只在整个 Block 外面加一个残差连接呢?

假设我们只用一个残差: $y = x + \text{MLP}(\text{Attention}(x))$

那么梯度为:

$$
\frac{\partial y}{\partial x} = I + \frac{\partial \text{MLP}}{\partial \text{Attention}} \cdot \frac{\partial \text{Attention}}{\partial x}
$$

虽然有 $I$, 但 MLP 和 Attention 的梯度是**连乘**的. 如果其中一个接近零, 整个链条就断了.

而现在的双残差设计:

$$
x' = x + \text{Attention}(\text{LN}_1(x))
$$

$$
y = x' + \text{MLP}(\text{LN}_2(x'))
$$

展开 $y$ 关于 $x$ 的梯度:

$$
\frac{\partial y}{\partial x} = \frac{\partial x'}{\partial x} + \frac{\partial \text{MLP}}{\partial x'} \cdot \frac{\partial x'}{\partial x} = \left(I + \frac{\partial \text{Attn}}{\partial x}\right) + \frac{\partial \text{MLP}}{\partial x'} \cdot \left(I + \frac{\partial \text{Attn}}{\partial x}\right)
$$

可以看到, 其中包含了一个纯粹的 $I$ (来自两次残差连接的组合), 即使 Attention 和 MLP 的梯度都很小, 这个 $I$ 依然保证梯度畅通.

### "高速公路"比喻

你可以把 32 个 Block 想象成一条从起点到终点的公路系统.

- **没有残差连接**: 信息必须穿过每一个收费站 (子层变换), 每个收费站都可能堵车 (梯度消失) 或引发事故 (梯度爆炸).
- **有残差连接**: 每个收费站旁边都有一条**免费的高速公路**. 信息可以选择走收费站 (学习变换) 或走高速公路 (直接通过). 即使所有收费站都堵了, 信息仍然可以通过高速公路从终点传回起点.

这就是为什么我们可以训练 32 层甚至更深的网络 — — 残差连接给了梯度一条永远畅通的"高速公路".

---

## 信息流与张量形状

下面用 Block 0 的实际数据 ($n = 14308$ 个 patch token, $d = 1280$) 来展示完整的张量变化过程.

```
输入 x: (14308, 1280)
  │
  ├─── [恒等路径] ──────────────────────────────────────┐
  │                                                      │
  ▼                                                      │
LayerNorm₁                                               │
  γ₁: (1280,)  β₁: (1280,)                              │
  │                                                      │
  ▼                                                      │
x_normed: (14308, 1280)                                  │
  │                                                      │
  ▼                                                      │
QKV 投影: x_normed @ W_qkv^T + b_qkv                    │
  W_qkv: (3840, 1280)  b_qkv: (3840,)                   │
  │                                                      │
  ▼                                                      │
qkv: (14308, 3840)                                       │
  │                                                      │
  ├── Q: (14308, 1280) ──reshape──▶ (14308, 16, 80)      │
  ├── K: (14308, 1280) ──reshape──▶ (14308, 16, 80)      │
  └── V: (14308, 1280) ──reshape──▶ (14308, 16, 80)      │
        │                                                 │
        ▼                                                 │
  每个头: Q_i @ K_i^T / sqrt(80)                          │
  注意力分数: (16, n_win, w, w)   ← 窗口注意力             │
        │                                                 │
        ▼                                                 │
  softmax → 乘以 V_i                                      │
  每个头输出: (n_tokens, 80)                               │
        │                                                 │
        ▼                                                 │
  拼接 16 个头: (14308, 1280)                              │
        │                                                 │
        ▼                                                 │
  输出投影: concat @ W_proj^T + b_proj                     │
  W_proj: (1280, 1280)  b_proj: (1280,)                   │
        │                                                 │
        ▼                                                 │
  attn_out: (14308, 1280)                                 │
        │                                                 │
        ▼                                                 │
  残差连接: x' = x + attn_out  ◀──────────────────────────┘
  │
  ▼
x': (14308, 1280)
  │
  ├─── [恒等路径] ──────────────────────────────────────┐
  │                                                      │
  ▼                                                      │
LayerNorm₂                                               │
  γ₂: (1280,)  β₂: (1280,)                              │
  │                                                      │
  ▼                                                      │
x'_normed: (14308, 1280)                                 │
  │                                                      │
  ▼                                                      │
FC1: x'_normed @ W₁^T + b₁                              │
  W₁: (5120, 1280)  b₁: (5120,)                         │
  │                                                      │
  ▼                                                      │
h: (14308, 5120)    ← 维度扩展 4 倍                       │
  │                                                      │
  ▼                                                      │
QuickGELU(h): (14308, 5120)                              │
  │                                                      │
  ▼                                                      │
FC2: h @ W₂^T + b₂                                      │
  W₂: (1280, 5120)  b₂: (1280,)                         │
  │                                                      │
  ▼                                                      │
mlp_out: (14308, 1280)   ← 维度压缩回 1280               │
  │                                                      │
  ▼                                                      │
残差连接: y = x' + mlp_out  ◀──────────────────────────┘
  │
  ▼
输出 y: (14308, 1280)
```

请注意几个关键的形状变化:

- **QKV 投影**: $(14308, 1280) \to (14308, 3840)$, 一次投影同时产生 Q, K, V
- **多头拆分**: $(14308, 3840) \to 3 \times (14308, 16, 80)$
- **MLP 扩展**: $(14308, 1280) \to (14308, 5120) \to (14308, 1280)$, 先升维再降维
- **输入输出形状一致**: $(14308, 1280) \to (14308, 1280)$, 这保证了 Block 可以堆叠

---

## 窗口注意力: 驯服 $O(n^2)$ 的计算量

### 问题: 全局注意力太贵了

标准的 Self-Attention 需要计算所有 token 对之间的注意力分数. 对于 $n$ 个 token, 注意力矩阵的大小是 $n \times n$.

在 Qwen2-VL 中, 一张图像经过 Patch Embedding 后可能产生 $n = 14308$ 个 patch token. 全局注意力矩阵的大小为:

$$
14308^2 = 204{,}718{,}864 \approx 2.05 \times 10^8
$$

每个注意力头需要一个这样的矩阵, 16 个头总共需要约 $3.3 \times 10^9$ 个浮点数 — — **仅仅是注意力分数就需要约 12 GB 的显存** (FP32), 这还不算反向传播时需要保存的中间值.

更严重的是, 这个代价与序列长度的**平方**成正比:

$$
\text{计算量} = O(n^2 \cdot d_k)
$$

如果图像分辨率翻倍, patch 数量翻 4 倍, 注意力计算量就翻 **16 倍**. 这对于需要处理高分辨率图像和视频的模型来说是不可接受的.

### 解决方案: 窗口注意力

窗口注意力的思路很简单: **不让每个 token 看到所有其他 token, 只让它看到局部窗口内的邻居**.

具体来说, 将 $n$ 个 token 划分成若干个大小为 $w$ 的窗口 (window), 每个窗口内独立计算注意力. 这样:

$$
\text{计算量} = O\!\left(\frac{n}{w} \cdot w^2 \cdot d_k\right) = O(n \cdot w \cdot d_k)
$$

从 $O(n^2)$ 降到了 $O(n \cdot w)$! 当窗口大小 $w$ 远小于 $n$ 时, 这是巨大的节省.

### 局部注意力够用吗?

你可能会担心: 只看局部窗口, 不会丢失全局信息吗?

答案是: 单个 Block 只看局部, 但 **32 个 Block 堆叠后, 信息的感受野会逐层扩大**. 这类似于 CNN 中小卷积核堆叠后也能覆盖整个图像的道理:

- Block 0: 每个 token 只能看到窗口内的邻居
- Block 1: 每个 token 已经融合了邻居的信息, 而邻居在上一层也融合了它们的邻居 — — 间接感受野扩大了
- Block 31: 经过 32 轮传播, 信息已经可以在整个图像中流动

这就像"口口相传": 虽然每个人只跟邻居说话, 但消息最终会传遍整个村庄.

---

## 从 Block 0 到 Block 31: 深度的解读

32 个 Block 结构完全相同, 但参数各不相同. 随着深度增加, 它们学到的特征越来越抽象. 这就是深度学习中"深度"的意义所在.

### 抽象阶梯

计算机视觉研究 (特别是对 CNN 和 ViT 的可视化研究) 揭示了一个普遍规律:

| 层级 | Block 范围 (大致) | 学到的特征                     | 类比         |
| ---- | ----------------- | ------------------------------ | ------------ |
| 浅层 | Block 0–7         | 边缘, 角点, 颜色梯度, 纹理方向 | 笔画         |
| 中层 | Block 8–20        | 纹理组合, 局部形状, 部件轮廓   | 偏旁部首     |
| 深层 | Block 21–31       | 物体部件, 物体类别, 场景语义   | 完整的字, 词 |

**浅层 Block** 像是一个细心的素描画家, 关注线条的走向和颜色的变化. 它们的输出对人类来说几乎不可解读 — — 只是一些看似随机的数值模式.

**中层 Block** 开始把低级特征组合成有意义的结构. 一些 token 开始对"眼睛形状", "轮子圆弧"这类局部结构产生强响应.

**深层 Block** 的输出最接近人类的语义理解. 一个 token 可能代表"这里是一只猫的头"或"这里是建筑物的屋顶" — — 这些高级语义信息正是后续语言模型需要的.

### 参数共享?

一个常见的问题是: 既然 32 个 Block 结构相同, 为什么不共享参数?

答案是: **不同深度需要学习不同层次的特征**. 如果 Block 0 和 Block 31 共享参数, 它们就被迫用同一套"滤镜"来处理低级纹理和高级语义 — — 这两者的特征模式完全不同, 一套参数无法兼顾.

实践证明, **独立参数**的效果远优于参数共享. 代价是参数量增加 32 倍, 但换来的是更强的表达能力.

---

## 参数量分析

让我们精确计算 Vision Block 的每一个参数.

### 单个 Block 的参数

以 Block 0 为例, 权重键名前缀为 `visual.blocks.0`:

| 组件             | 权重键名                           | 形状           | 参数量    |
| ---------------- | ---------------------------------- | -------------- | --------- |
| norm1.weight     | `visual.blocks.0.norm1.weight`     | $(1280,)$      | 1,280     |
| norm1.bias       | `visual.blocks.0.norm1.bias`       | $(1280,)$      | 1,280     |
| attn.qkv.weight  | `visual.blocks.0.attn.qkv.weight`  | $(3840, 1280)$ | 4,915,200 |
| attn.qkv.bias    | `visual.blocks.0.attn.qkv.bias`    | $(3840,)$      | 3,840     |
| attn.proj.weight | `visual.blocks.0.attn.proj.weight` | $(1280, 1280)$ | 1,638,400 |
| attn.proj.bias   | `visual.blocks.0.attn.proj.bias`   | $(1280,)$      | 1,280     |
| norm2.weight     | `visual.blocks.0.norm2.weight`     | $(1280,)$      | 1,280     |
| norm2.bias       | `visual.blocks.0.norm2.bias`       | $(1280,)$      | 1,280     |
| mlp.fc1.weight   | `visual.blocks.0.mlp.fc1.weight`   | $(5120, 1280)$ | 6,553,600 |
| mlp.fc1.bias     | `visual.blocks.0.mlp.fc1.bias`     | $(5120,)$      | 5,120     |
| mlp.fc2.weight   | `visual.blocks.0.mlp.fc2.weight`   | $(1280, 5120)$ | 6,553,600 |
| mlp.fc2.bias     | `visual.blocks.0.mlp.fc2.bias`     | $(1280,)$      | 1,280     |

让我们分组汇总:

| 子模块                       | 参数量                                     |
| ---------------------------- | ------------------------------------------ |
| norm1 (权重 + 偏置)          | $1{,}280 + 1{,}280 = 2{,}560$              |
| Attention QKV (权重 + 偏置)  | $4{,}915{,}200 + 3{,}840 = 4{,}919{,}040$  |
| Attention Proj (权重 + 偏置) | $1{,}638{,}400 + 1{,}280 = 1{,}639{,}680$  |
| norm2 (权重 + 偏置)          | $1{,}280 + 1{,}280 = 2{,}560$              |
| MLP FC1 (权重 + 偏置)        | $6{,}553{,}600 + 5{,}120 = 6{,}558{,}720$  |
| MLP FC2 (权重 + 偏置)        | $6{,}553{,}600 + 1{,}280 = 6{,}554{,}880$  |
| **单个 Block 合计**          | **$19{,}677{,}440 \approx 19.68\text{M}$** |

### 整体比例分析

有趣的是, **MLP 占了每个 Block 参数量的大头**:

$$
\frac{6{,}558{,}720 + 6{,}554{,}880}{19{,}677{,}440} = \frac{13{,}113{,}600}{19{,}677{,}440} \approx 66.6\%
$$

而 Attention 部分 (QKV + Proj) 占约 33.3%, LayerNorm 几乎可以忽略不计 (0.03%).

### 32 个 Block 的总参数

$$
19{,}677{,}440 \times 32 = 629{,}678{,}080 \approx 629.7\text{M}
$$

32 个 Vision Block 共计约 **6.3 亿参数**. 这占了视觉编码器参数的绑大多数 (还有少量参数在 Patch Embedding 和 Patch Merger 中).

### 关于 QKV 合并

你可能注意到 `attn.qkv.weight` 的形状是 $(3840, 1280)$ 而不是三个独立的 $(1280, 1280)$ 矩阵. 这是一个常见的工程优化: 将 $Q$, $K$, $V$ 三个投影矩阵沿第一个维度拼接:

$$
W_{qkv} = \begin{bmatrix} W_Q \\ W_K \\ W_V \end{bmatrix} \in \mathbb{R}^{3840 \times 1280}
$$

其中 $3840 = 3 \times 1280$. 这样只需一次矩阵乘法就能同时得到 $Q$, $K$, $V$, 比三次独立的矩阵乘法更高效 (因为 GPU 更喜欢大矩阵运算).

---

## 数值示例

为了真正理解 Vision Block 的计算过程, 让我们用一个简化的 $d = 4$ 的例子, 手动走一遍全部 6 个步骤.

假设我们有 $n = 2$ 个 token, 维度 $d = 4$, $h = 2$ 个注意力头, 每个头维度 $d_k = 2$, MLP 隐藏维度 $= 8$.

### 输入

$$
x = \begin{bmatrix} 1.0 & -0.5 & 0.3 & 0.8 \\ -0.2 & 0.7 & -0.1 & 0.4 \end{bmatrix}
$$

### 步骤 1: LayerNorm₁

对第一个 token $[1.0, -0.5, 0.3, 0.8]$:

$$
\mu = \frac{1.0 + (-0.5) + 0.3 + 0.8}{4} = 0.4
$$

$$
\sigma^2 = \frac{(1.0 - 0.4)^2 + (-0.5 - 0.4)^2 + (0.3 - 0.4)^2 + (0.8 - 0.4)^2}{4} = \frac{0.36 + 0.81 + 0.01 + 0.16}{4} = 0.335
$$

$$
\hat{x}_1 = \frac{[1.0, -0.5, 0.3, 0.8] - 0.4}{\sqrt{0.335 + 10^{-6}}} = \frac{[0.6, -0.9, -0.1, 0.4]}{0.5788} \approx [1.037, -1.555, -0.173, 0.691]
$$

(假设 $\gamma_1 = [1,1,1,1]$, $\beta_1 = [0,0,0,0]$ 以简化.)

### 步骤 2 & 3: Attention + 第一次残差

假设经过 Attention 计算 (此处省略完整的多头注意力过程, 参见 [07_attention](../07_attention/README.md)), 得到:

$$
\text{attn\_out} = \begin{bmatrix} 0.12 & -0.08 & 0.05 & 0.03 \\ -0.04 & 0.15 & -0.02 & 0.07 \end{bmatrix}
$$

第一次残差连接:

$$
x' = x + \text{attn\_out} = \begin{bmatrix} 1.12 & -0.58 & 0.35 & 0.83 \\ -0.24 & 0.85 & -0.12 & 0.47 \end{bmatrix}
$$

### 步骤 4: LayerNorm₂

对 $x'$ 的每个 token 再做一次 LayerNorm (用独立的 $\gamma_2, \beta_2$), 得到 $\hat{x}'$ (过程同步骤 1, 此处省略).

### 步骤 5: MLP

假设 $\hat{x}'_1 = [0.98, -1.50, -0.10, 0.62]$ (LayerNorm 后的第一个 token).

FC1 投影 ($d=4 \to 8$): 假设 $W_1$ 的第一行是 $[0.1, 0.2, -0.1, 0.3]$:

$$
h_1[0] = 0.98 \times 0.1 + (-1.50) \times 0.2 + (-0.10) \times (-0.1) + 0.62 \times 0.3 + b_1[0]
$$

$$
= 0.098 - 0.300 + 0.010 + 0.186 + 0 = -0.006
$$

QuickGELU:

$$
\text{QuickGELU}(-0.006) = -0.006 \times \sigma(1.702 \times (-0.006)) = -0.006 \times \sigma(-0.010) \approx -0.006 \times 0.4975 \approx -0.003
$$

对所有 8 个维度重复此过程, 再通过 FC2 投影回 $d = 4$.

### 步骤 6: 第二次残差

$$
y = x' + \text{mlp\_out}
$$

最终 $y$ 的形状仍然是 $(2, 4)$ — — 与输入 $x$ 完全一致.

**核心观察**: 每一步都只是简单的矩阵乘法, 逐元素运算和加法. 没有任何"魔法" — — 深度学习的威力来自这些简单操作的**大规模组合**.

---

## 模块化验证策略

在 `impl.py` 的测试中, 我们采用了一种"分而治之"的验证策略. 这是工程实践中非常重要的原则.

### 为什么不直接测试整个 Block?

Vision Block 内部包含一个复杂的组件 — — 窗口注意力 (Windowed Attention with RoPE). 如果我们直接测试整个 Block 的输入输出, 一旦结果不对, 很难判断是哪个组件出了问题: 是 LayerNorm 的参数加载错误? 还是 Attention 的实现有误? 还是 MLP 的矩阵乘法方向反了?

### 分步验证方案

我们把 Block 的 6 个步骤拆开, 逐步验证:

1. **验证 norm1**: 用真实权重对输入做 LayerNorm, 与 PyTorch 参考值对比
2. **验证 Attention 残差**: 用预先转储的 Attention 输出 (从 PyTorch 运行中保存), 加上输入做残差, 验证 $x' = x + \text{attn\_out}$
3. **验证 norm2**: 对 $x'$ 做 LayerNorm, 与参考值对比
4. **验证 MLP**: 对 norm2 的输出做 MLP 变换, 与参考值对比
5. **验证 MLP 残差**: $y = x' + \text{mlp\_out}$, 与参考值对比
6. **验证完整 Block**: 将 Attention 封装为一个返回转储值的函数, 运行完整的 `vision_block()` 函数

这种策略的优势:

- **定位精准**: 如果第 3 步失败了, 你立即知道是 norm2 有问题
- **独立开发**: 可以先开发和验证简单的组件, 再组合成复杂的系统
- **复用已验证的组件**: 一旦确认 LayerNorm 和 MLP 是正确的, 就可以把它们当作"可信的零件"来构建 Block

---

## NumPy 实现

以下是 Vision Block 的完整 NumPy 实现, 包含逐行中文注释:

```python
import numpy as np


def layer_norm(x, weight, bias, eps=1e-6):
    """Layer Normalization：对每个 token 的特征维度做标准化"""
    mean = x.mean(axis=-1, keepdims=True)    # 沿最后一个维度求均值，形状 (n, 1)
    var = x.var(axis=-1, keepdims=True)      # 沿最后一个维度求方差，形状 (n, 1)
    x_norm = (x - mean) / np.sqrt(var + eps) # 标准化：减均值除标准差
    return x_norm * weight + bias            # 仿射变换：乘以 γ 加上 β


def quick_gelu(x):
    """QuickGELU 激活函数：GELU 的快速近似"""
    return x * (1.0 / (1.0 + np.exp(-1.702 * x)))
    # 等价于 x * sigmoid(1.702 * x)
    # 比标准 GELU 的 erf 实现快得多


def vision_mlp(x, fc1_weight, fc1_bias, fc2_weight, fc2_bias):
    """Vision MLP：两层前馈网络，中间用 QuickGELU 激活"""
    h = x @ fc1_weight.T + fc1_bias   # 第一层线性变换：(n, 1280) → (n, 5120)
    h = quick_gelu(h)                 # 非线性激活：逐元素，形状不变
    return h @ fc2_weight.T + fc2_bias # 第二层线性变换：(n, 5120) → (n, 1280)


def residual_add(x, sublayer_out):
    """残差连接：将子层输出加回原始输入"""
    return x + sublayer_out            # 就是简单的逐元素加法！


def vision_block(x, norm1_weight, norm1_bias,
                 attn_fn,
                 norm2_weight, norm2_bias,
                 fc1_weight, fc1_bias,
                 fc2_weight, fc2_bias):
    """
    Vision Transformer Block：视觉编码器的核心构建单元。

    参数：
        x:            输入张量，形状 (n, 1280)
        norm1_weight: 第一个 LayerNorm 的 γ 参数
        norm1_bias:   第一个 LayerNorm 的 β 参数
        attn_fn:      注意力函数（接受归一化后的输入，返回注意力输出）
        norm2_weight: 第二个 LayerNorm 的 γ 参数
        norm2_bias:   第二个 LayerNorm 的 β 参数
        fc1_weight:   MLP 第一层权重，形状 (5120, 1280)
        fc1_bias:     MLP 第一层偏置，形状 (5120,)
        fc2_weight:   MLP 第二层权重，形状 (1280, 5120)
        fc2_bias:     MLP 第二层偏置，形状 (1280,)

    返回：
        y: 输出张量，形状 (n, 1280)，与输入形状相同
    """

    # ===== 子层 1：Attention =====
    x_normed = layer_norm(x, norm1_weight, norm1_bias)  # Pre-Norm：先归一化
    attn_out = attn_fn(x_normed)                        # 多头自注意力
    x = residual_add(x, attn_out)                       # 残差连接：加回原始 x

    # ===== 子层 2：MLP =====
    x_normed = layer_norm(x, norm2_weight, norm2_bias)  # Pre-Norm：先归一化
    mlp_out = vision_mlp(                               # MLP 前馈网络
        x_normed, fc1_weight, fc1_bias,
        fc2_weight, fc2_bias
    )
    x = residual_add(x, mlp_out)                        # 残差连接：加回 x'

    return x
```

代码非常简洁 — — 整个 Vision Block 只有 6 行核心逻辑. 这体现了 Transformer 架构的优雅: **用简单的组件 (LN, 矩阵乘法, 激活函数, 加法) 组合出强大的功能**.

注意 `attn_fn` 是作为参数传入的函数. 这是因为注意力模块的实现 (涉及窗口划分, RoPE 等) 比较复杂, 我们将其作为一个"黑盒"接口传入, 保持 Block 的代码清晰.

---

## 在 Qwen2-VL 中的位置

Vision Block 在 Qwen2-VL 完整模型中的位置如下:

```
Qwen2-VL 完整模型
│
├── 视觉编码器 (Vision Encoder / ViT)
│   │
│   ├── Patch Embedding (Conv3D)
│   │   └── 将图像切成 patch 并投影为 (n, 1280) 的向量序列
│   │
│   ├── Vision Block × 32              ◀◀◀ 本节内容
│   │   │
│   │   ├── Block 0   ─── norm1 → Attention → 残差 → norm2 → MLP → 残差
│   │   ├── Block 1   ─── norm1 → Attention → 残差 → norm2 → MLP → 残差
│   │   ├── ...
│   │   └── Block 31  ─── norm1 → Attention → 残差 → norm2 → MLP → 残差
│   │
│   └── Patch Merger
│       └── 将视觉 token 压缩后送入语言模型
│
└── 语言模型 (LLM Decoder)
    │
    ├── Token Embedding (vocab_size=151936, hidden=1536)
    │
    ├── Decoder Layer × 28
    │   ├── RMSNorm → GQA Attention (12 heads, 2 KV heads, head_dim=128)→ 残差
    │   └── RMSNorm → SwiGLU MLP (hidden=8960) → 残差
    │
    └── LM Head → 输出 logits (151936,)
```

32 个 Vision Block 共享相同的架构, 但拥有完全独立的参数 (`visual.blocks.0.*` 到 `visual.blocks.31.*`).

注意视觉编码器和语言模型的 Block 结构有细微差别:

- 视觉编码器使用 **LayerNorm** (有偏置), 语言模型使用 **RMSNorm** (无偏置)
- 视觉编码器使用 **QuickGELU** 激活, 语言模型使用 **SwiGLU** 激活
- 视觉编码器使用 **标准 Multi-Head Attention** (16 个头), 语言模型使用 **GQA** (12 个 Q 头, 2 个 KV 头)

---

## 常见误解与陷阱

### 误解 1: Pre-Norm 和 Post-Norm 效果一样

**错误**: "归一化放在哪里不重要, 反正都是归一化. "

**事实**: 放置位置对梯度流, 训练稳定性, 以及最终性能都有显著影响. Pre-Norm 在深层网络 (如 32 层) 中训练更稳定, 不需要 learning rate warmup. Post-Norm 在精心调参后可能略优, 但对超参数非常敏感.

两者的数学结构本质不同:

- Post-Norm: $\text{LN}(x + F(x))$ — LayerNorm 作用在残差之后
- Pre-Norm: $x + F(\text{LN}(x))$ — LayerNorm 作用在子层变换之前

**在 Pre-Norm 中, 残差路径上没有任何非线性操作**, 梯度可以无损传播. 这不是一个可以忽略的细节.

### 误解 2: LayerNorm 和 BatchNorm 是一样的

**错误**: "归一化就是归一化, LayerNorm 和 BatchNorm 本质一样. "

**事实**: 它们在完全不同的维度上做归一化:

| 特性               | BatchNorm                 | LayerNorm             |
| ------------------ | ------------------------- | --------------------- |
| 归一化方向         | 跨样本 (batch 维度)       | 跨特征 (feature 维度) |
| 统计量依赖         | 整个 mini-batch           | 单个样本              |
| 推理时             | 需要维护 running mean/var | 直接在线计算          |
| 对 batch size 敏感 | 是 (小 batch 不稳定)      | 否                    |
| 适用场景           | CNN                       | Transformer           |

LayerNorm 对每个 token 独立归一化其 1280 个特征维度. BatchNorm 则对同一个特征维度跨整个 batch 做归一化. 在序列模型中, BatchNorm 效果很差, 因为不同位置的统计特性差异很大.

### 误解 3: MLP 在 token 之间有信息交互

**错误**: "MLP 层也是在做 token 之间的通信. "

**事实**: MLP 是 **position-wise** 的 — — 它对每个 token **完全独立地**施加同一个变换. 观察公式:

$$
\text{MLP}(x_i) = W_2 \, \text{QuickGELU}(W_1 x_i + b_1) + b_2
$$

这里 $x_i$ 是第 $i$ 个 token 的向量. MLP 的计算完全不涉及 $x_j$ ($j \neq i$).

**Token 之间的唯一信息交互发生在 Attention 子层中**. MLP 的角色是在每个位置上做独立的非线性特征变换 — — 就像给每个人发了一台相同的计算器, 让他们各自计算自己的结果.

### 误解 4: Block 的输出维度可以和输入不同

**错误**: "更深的 Block 应该输出更高维度的特征. "

**事实**: Vision Block 的输入和输出维度**必须相同** (都是 $(n, 1280)$), 这是 Block 能够堆叠的前提条件. 如果 Block 0 的输出维度变了, Block 1 就无法接收它 — — 所有的权重矩阵都是按 $d = 1280$ 设计的.

维度的变化发生在 Block **内部** — — MLP 子层先扩展到 5120 维再压缩回 1280 维. 但从 Block 的外部来看, 它就是一个 $\mathbb{R}^{n \times 1280} \to \mathbb{R}^{n \times 1280}$ 的映射.

---

## 总结

| 属性                    | 值                        |
| ----------------------- | ------------------------- |
| 输入形状                | $(n, 1280)$               |
| 输出形状                | $(n, 1280)$               |
| 子层数                  | 2 (Attention + MLP)       |
| 归一化方式              | Pre-Norm (先归一化再变换) |
| 归一化类型              | LayerNorm (有偏置项)      |
| 激活函数                | QuickGELU                 |
| 残差连接                | 每个子层后独立添加        |
| 注意力头数              | 16                        |
| 每头维度                | 80                        |
| MLP 隐藏维度            | 5120 ($4 \times 1280$)    |
| 参数量 / Block          | $\approx 19.68\text{M}$   |
| Block 数量              | 32                        |
| 视觉编码器 Block 总参数 | $\approx 629.7\text{M}$   |

**一句话总结**: Vision Block = Pre-Norm LayerNorm₁ → Multi-Head Attention → 残差 → Pre-Norm LayerNorm₂ → MLP (QuickGELU) → 残差. 简单的结构重复 32 次, 让原始像素逐步变成高级语义.

---

## 延伸阅读

1. **Vaswani, A. et al. (2017).** _"Attention Is All You Need."_ NeurIPS 2017.
   — Transformer 架构的奠基论文. 提出了 Multi-Head Attention, Position-wise FFN, 残差连接 + Post-Norm 的标准 Block 结构.

2. **Dosovitskiy, A. et al. (2020).** _"An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale."_ ICLR 2021.
   — Vision Transformer (ViT) 论文. 证明了纯 Transformer 架构可以在视觉任务上超越 CNN.

3. **Xiong, R. et al. (2020).** _"On Layer Normalization in the Transformer Architecture."_ ICML 2020.
   — 深入分析了 Pre-Norm 与 Post-Norm 的训练动态差异, 证明了 Pre-Norm 的梯度稳定性优势.

4. **He, K. et al. (2016).** _"Deep Residual Learning for Image Recognition."_ CVPR 2016.
   — ResNet 论文. 残差连接的思想最早来自这里 — — "学习残差比直接学习映射更容易. "

5. **Wang, P. et al. (2024).** _"Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution."_
   — Qwen2-VL 的技术报告. 详细描述了视觉编码器的架构设计, 包括窗口注意力和 RoPE 的具体实现.
