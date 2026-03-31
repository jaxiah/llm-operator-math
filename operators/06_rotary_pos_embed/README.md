# 06 — 旋转位置编码 (Rotary Position Embedding, RoPE)

> **一篇从零开始的数学博客**: 从"猫追狗"说起, 经过复数, 旋转矩阵,
> 最终理解 Qwen2-VL 如何用旋转给文字和图像标注位置.

---

## 1. 为什么词序很重要?

试着读两句话:

- "猫追狗"
- "狗追猫"

同样三个字, 谁追谁完全变了. 人类一眼就能分辨, 因为我们天然按从左到右的顺序阅读.
但 Transformer 的核心操作 — — **自注意力** — — 并不具备这种能力.

自注意力做的事情, 本质上是这样的: 给定一组向量 $\{x_1, x_2, \ldots, x_n\}$,
对每一对 $(x_i, x_j)$ 计算一个"分数"来衡量它们之间的关联程度.
这个分数只取决于 $x_i$ 和 $x_j$ 的数值 — — 跟它们在序列中的位置**毫无关系**.

用数学语言说: 自注意力是 **permutation-invariant** (置换不变的).
你把输入向量的顺序打乱, 得到的注意力权重也跟着打乱, 但每一对之间的分数不变.
这就像把一袋弹珠倒在桌上, 无论怎么排列, 每两颗弹珠之间的距离取决于弹珠本身,
而不是你倒出它们的顺序. 这个现象在 NLP 领域有一个经典的名字: **bag-of-words 问题**.

所以, 如果我们想让 Transformer 区分"猫追狗"和"狗追猫", 就必须**额外**注入位置信息.
这就是 **位置编码 (Position Encoding)** 的使命.

---

## 2. 位置编码的演化简史

### 2.1 正弦位置编码 (Sinusoidal Position Encoding)

2017 年, Vaswani 等人在里程碑式的论文 _"Attention Is All You Need"_ 中提出了最初的方案:
对每个位置 $p$, 生成一组固定的正弦和余弦值, 然后**加到**词嵌入向量上:

$$
\text{PE}(p, 2i) = \sin\!\left(\frac{p}{10000^{2i/d}}\right), \quad
\text{PE}(p, 2i+1) = \cos\!\left(\frac{p}{10000^{2i/d}}\right)
$$

它的优点是简洁, 无需学习参数, 且不同位置的编码向量之间存在线性关系.
但它的缺点也很明显: 位置信息被**加性地**混入词嵌入, 模型很难把"位置"和"语义"干净地分离.

### 2.2 可学习的位置嵌入 (Learned Positional Embeddings)

BERT 和 GPT-1 采用了另一种策略: 给每个位置分配一个可学习的向量.
例如位置 0 有一个 $d$ 维参数向量 $p_0$, 位置 1 有 $p_1$, 以此类推.
训练时, 这些向量和其他参数一起通过梯度下降来优化.

好处是灵活 — — 模型自己决定用什么编码.
坏处是**泛化能力差**: 如果训练时最长序列是 512 个 token,
那么位置 513 根本没有对应的嵌入向量, 推理时碰到更长序列就会失败.

### 2.3 相对位置编码 (Relative Position Encoding)

Shaw 等人 (2018) 以及后来的 Transformer-XL (Dai 等人, 2019) 提出了一个更根本的改进:
**不要编码绝对位置, 而是编码两个 token 之间的相对距离**.

直觉上, "猫"和"追"之间隔了 1 步, 这个信息比"猫在位置 0"更本质.
但相对位置编码通常需要修改注意力的计算公式, 引入额外的查找表或偏置项, 实现起来不够优雅.

### 2.4 旋转位置编码 (RoPE)

2021 年, Su 等人发表了 _"RoFormer: Enhanced Transformer with Rotary Position Embedding"_,
提出了一种**通过旋转矩阵**来编码位置的方法.
它的精髓在于: **编码的是绝对位置, 但注意力分数自然地只依赖于相对位置**.

RoPE 不改变注意力公式的结构, 只是在计算 attention score 之前,
对 Query 和 Key 向量做一次**旋转变换**. 这种方式简洁, 高效, 易于实现,
而且天然支持长度外推 — — 这也是为什么 LLaMA, Qwen, Mistral 等主流模型全部采用 RoPE.

在深入 RoPE 之前, 我们需要复习一些关键的数学工具. 别担心, 我们会从最基础的地方开始.

---

## 3. 复数速览

### 3.1 复数的定义

复数是形如 $z = a + bi$ 的数, 其中 $a$ 和 $b$ 是实数,
$i$ 是**虚数单位**, 满足 $i^2 = -1$.
$a$ 叫做 $z$ 的实部 ($\text{Re}(z)$), $b$ 叫做虚部 ($\text{Im}(z)$).

### 3.2 复平面 (Argand Diagram)

我们可以把复数 $z = a + bi$ 画在一个二维平面上:
横轴是实部, 纵轴是虚部. 这样, 每个复数对应平面上的一个点 $(a, b)$.

一个复数的两个重要属性:

- **模 (modulus)**: $|z| = \sqrt{a^2 + b^2}$, 就是点到原点的距离.
- **辐角 (argument)**: $\arg(z) = \text{atan2}(b, a)$, 就是从正实轴逆时针转到 $z$ 的角度.

### 3.3 Euler 公式的推导

Euler 公式是连接指数函数与三角函数的桥梁:

$$
e^{i\theta} = \cos\theta + i\sin\theta
$$

为什么这是对的? 让我们用 Taylor 级数来证明.

首先, 回顾三个函数的 Taylor 展开:

$$
e^x = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \frac{x^4}{4!} + \frac{x^5}{5!} + \cdots
$$

$$
\cos x = 1 - \frac{x^2}{2!} + \frac{x^4}{4!} - \frac{x^6}{6!} + \cdots
$$

$$
\sin x = x - \frac{x^3}{3!} + \frac{x^5}{5!} - \frac{x^7}{7!} + \cdots
$$

现在, 把 $x = i\theta$ 代入 $e^x$ 的展开式:

$$
e^{i\theta} = 1 + (i\theta) + \frac{(i\theta)^2}{2!} + \frac{(i\theta)^3}{3!} + \frac{(i\theta)^4}{4!} + \frac{(i\theta)^5}{5!} + \cdots
$$

利用 $i$ 的幂次循环规律 $i^0=1,\; i^1=i,\; i^2=-1,\; i^3=-i,\; i^4=1,\; \ldots$ 逐项展开:

$$
= 1 + i\theta - \frac{\theta^2}{2!} - \frac{i\theta^3}{3!} + \frac{\theta^4}{4!} + \frac{i\theta^5}{5!} - \cdots
$$

把实部和虚部分别收集:

$$
= \underbrace{\left(1 - \frac{\theta^2}{2!} + \frac{\theta^4}{4!} - \cdots\right)}_{\cos\theta}
+ \; i \underbrace{\left(\theta - \frac{\theta^3}{3!} + \frac{\theta^5}{5!} - \cdots\right)}_{\sin\theta}
$$

所以 $e^{i\theta} = \cos\theta + i\sin\theta$. $\blacksquare$

### 3.4 Euler 公式的几何意义

$e^{i\theta}$ 在复平面上是什么? 它的模为 $|e^{i\theta}| = \sqrt{\cos^2\theta + \sin^2\theta} = 1$,
辐角为 $\theta$. 所以 $e^{i\theta}$ 是**单位圆上辐角为 $\theta$ 的点**.

这马上告诉我们: **乘以 $e^{i\theta}$ 就是逆时针旋转 $\theta$ 角度**.

---

## 4. 复数乘法 = 旋转

### 4.1 代数证明

设 $z = a + bi$, 我们把它乘以 $e^{i\theta} = \cos\theta + i\sin\theta$:

$$
z \cdot e^{i\theta} = (a + bi)(\cos\theta + i\sin\theta)
$$

展开 (像多项式乘法一样展开, 记住 $i^2 = -1$):

$$
= a\cos\theta + a \cdot i\sin\theta + bi\cos\theta + bi \cdot i\sin\theta
$$

$$
= a\cos\theta + i \cdot a\sin\theta + i \cdot b\cos\theta + i^2 \cdot b\sin\theta
$$

$$
= a\cos\theta + i \cdot a\sin\theta + i \cdot b\cos\theta - b\sin\theta
$$

把实部和虚部分别整理:

$$
= (a\cos\theta - b\sin\theta) + i(a\sin\theta + b\cos\theta)
$$

也就是说, 如果原来的复数对应点 $(a, b)$, 旋转后对应点 $(a\cos\theta - b\sin\theta,\; a\sin\theta + b\cos\theta)$.

### 4.2 几何证明

用极坐标表示 $z = r \cdot e^{i\phi}$ (其中 $r = |z|$, $\phi = \arg(z)$), 那么:

$$
z \cdot e^{i\theta} = r \cdot e^{i\phi} \cdot e^{i\theta} = r \cdot e^{i(\phi + \theta)}
$$

结果的模是 $r$ (不变!), 辐角是 $\phi + \theta$ (增加了 $\theta$).
所以复数乘以 $e^{i\theta}$ 就是**保持模不变, 辐角增加 $\theta$** — — 这正是旋转的定义.

### 4.3 数值例子

把 $z = 3 + 4i$ 旋转 $\theta = \pi/6$ (即 30°):

先算 $\cos(\pi/6) = \frac{\sqrt{3}}{2} \approx 0.8660$, $\sin(\pi/6) = \frac{1}{2} = 0.5$.

$$
z' = (3 + 4i)(0.8660 + 0.5i)
$$

$$
= (3 \times 0.8660 - 4 \times 0.5) + (3 \times 0.5 + 4 \times 0.8660)i
$$

$$
= (2.5981 - 2.0) + (1.5 + 3.4641)i
$$

$$
= 0.5981 + 4.9641i
$$

验证模不变: $|z| = \sqrt{9 + 16} = 5$, $|z'| = \sqrt{0.5981^2 + 4.9641^2} = \sqrt{0.3577 + 24.6423} = \sqrt{25} = 5$. ✅

---

## 5. 2D 旋转矩阵

### 5.1 从复数乘法推导旋转矩阵

我们在第 4.1 节得到了结果:

$$
\begin{cases}
a' = a\cos\theta - b\sin\theta \\
b' = a\sin\theta + b\cos\theta
\end{cases}
$$

这是一个线性变换, 可以写成矩阵-向量乘法的形式:

$$
\begin{bmatrix} a' \\ b' \end{bmatrix}
= \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}
\begin{bmatrix} a \\ b \end{bmatrix}
$$

我们把这个 $2 \times 2$ 矩阵记作 $R(\theta)$:

$$
R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}
$$

### 5.2 旋转矩阵的性质

**性质 1: 行列式为 1**

$$
\det R(\theta) = \cos\theta \cdot \cos\theta - (-\sin\theta) \cdot \sin\theta = \cos^2\theta + \sin^2\theta = 1
$$

这意味着旋转不改变面积 (不拉伸, 不压缩).

**性质 2: 正交性 $R(\theta)^T R(\theta) = I$**

$$
R(\theta)^T = \begin{bmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{bmatrix}
$$

$$
R(\theta)^T R(\theta) = \begin{bmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}
$$

$$
= \begin{bmatrix} \cos^2\theta + \sin^2\theta & -\cos\theta\sin\theta + \sin\theta\cos\theta \\ -\sin\theta\cos\theta + \cos\theta\sin\theta & \sin^2\theta + \cos^2\theta \end{bmatrix}
= \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}
$$

正交矩阵保持向量的范数: $\|R(\theta) x\| = \|x\|$. 这就是为什么旋转不改变向量长度.

**性质 3: 旋转的叠加 $R(\alpha)R(\beta) = R(\alpha + \beta)$**

$$
R(\alpha)R(\beta) = \begin{bmatrix} \cos\alpha & -\sin\alpha \\ \sin\alpha & \cos\alpha \end{bmatrix} \begin{bmatrix} \cos\beta & -\sin\beta \\ \sin\beta & \cos\beta \end{bmatrix}
$$

$$
= \begin{bmatrix} \cos\alpha\cos\beta - \sin\alpha\sin\beta & -\cos\alpha\sin\beta - \sin\alpha\cos\beta \\ \sin\alpha\cos\beta + \cos\alpha\sin\beta & -\sin\alpha\sin\beta + \cos\alpha\cos\beta \end{bmatrix}
$$

由三角恒等式 $\cos(\alpha+\beta) = \cos\alpha\cos\beta - \sin\alpha\sin\beta$ 以及
$\sin(\alpha+\beta) = \sin\alpha\cos\beta + \cos\alpha\sin\beta$, 得到:

$$
= \begin{bmatrix} \cos(\alpha+\beta) & -\sin(\alpha+\beta) \\ \sin(\alpha+\beta) & \cos(\alpha+\beta) \end{bmatrix} = R(\alpha + \beta)
$$

这条性质对 RoPE 至关重要 — — 它意味着 $R_m^T R_n = R(- m) R(n) = R(n - m)$.

---

## 6. RoPE 的核心思想

现在我们有了所有的数学工具, 可以理解 RoPE 在做什么了.

### 6.1 把向量维度两两配对

假设 Query 向量 $q$ 的维度是 $d$ (例如 Qwen2-VL 文本模型中 $d = 128$).
RoPE 把这 $d$ 个维度分成 $d/2$ 对: $(q_0, q_{d/2})$, $(q_1, q_{d/2+1})$,..., $(q_{d/2-1}, q_{d-1})$.

每一对被当作一个 2D 向量 (或者说一个复数), 然后用**不同的角度**做旋转.

### 6.2 位置决定旋转角度

对于位置 $p$ 的 token, 第 $i$ 对维度的旋转角度为:

$$
\alpha_i(p) = p \cdot \theta_i
$$

其中 $\theta_i$ 是第 $i$ 个**基础频率** (下一节详细讨论).
注意: 不同的维度对用不同的频率, 同一个维度对在不同位置用不同的角度.

### 6.3 分块对角旋转矩阵

把所有 $d/2$ 个 2D 旋转拼在一起, 我们得到一个 $d \times d$ 的**分块对角矩阵**:

$$
R_p = \begin{bmatrix}
R(\alpha_0(p)) & & & \\
& R(\alpha_1(p)) & & \\
& & \ddots & \\
& & & R(\alpha_{d/2-1}(p))
\end{bmatrix}
$$

其中每个 $R(\alpha_i(p))$ 是一个 $2 \times 2$ 旋转矩阵.
空白处是零 — — 不同维度对之间的旋转是**完全独立**的.

RoPE 的操作就是:

$$
q_{\text{rotated}} = R_p \, q, \qquad k_{\text{rotated}} = R_p \, k
$$

注意: 在 HuggingFace 的实际代码中, 维度对的组织方式略有不同 — —
第 $i$ 对是 $(q_i, q_{i+d/2})$ 而非 $(q_{2i}, q_{2i+1})$.
这不影响数学本质, 只是实现上的排列选择. 我们会在第 9 节详细解释.

---

## 7. 频率设计: 为什么是几何级数?

### 7.1 频率公式

RoPE 的频率定义为:

$$
\theta_i = \text{base}^{-2i/d}, \quad i = 0, 1, \ldots, d/2 - 1
$$

这是一个**几何级数 (geometric sequence)**:
相邻两个频率的比值是常数 $\text{base}^{-2/d}$.

以 Qwen2-VL 文本模型 ($d = 128$, $\text{base} = 1{,}000{,}000$) 为例:

- $\theta_0 = 1{,}000{,}000^{0} = 1.0$ (最高频率)
- $\theta_1 = 1{,}000{,}000^{-2/128} = 1{,}000{,}000^{-1/64} \approx 0.7943$
- $\theta_{63} = 1{,}000{,}000^{-126/128} \approx 1.22 \times 10^{-6}$ (最低频率)

### 7.2 频率与波长的关系

每个频率 $\theta_i$ 对应一个**波长** (旋转角度转满一圈 $2\pi$ 所需的位置数):

$$
\lambda_i = \frac{2\pi}{\theta_i}
$$

- **高频率** ($\theta_i$ 接近 1): $\lambda_i \approx 2\pi \approx 6.28$.
  几个 token 的位移就转了一大圈. 这种维度对**位置变化敏感**,
  编码的是**局部/近距离**的位置差异.

- **低频率** ($\theta_i$ 接近 0): $\lambda_i$ 可能达到数百万.
  要经过很多很多 token 才转完一圈. 这种维度对变化缓慢,
  编码的是**全局/远距离**的位置关系.

### 7.3 为什么用几何级数? — — 多尺度表示

这就像音乐中的音阶: 频率呈几何级数分布可以**均匀覆盖从局部到全局的所有尺度**.

- 如果所有频率都很高, 模型只能看到近邻关系, 无法捕捉远程依赖.
- 如果所有频率都很低, 模型只能看到粗粒度的位置, 无法区分相邻 token.
- 几何级数同时提供高频和低频, 让模型在**每个尺度上**都有位置感知能力.

类比: 地图的比例尺. 你既需要 1:1000 的街道地图来找巷子里的餐馆,
也需要 1:1000000 的省级地图来规划跨城市的路线. RoPE 的频率设计让模型同时拥有两种"地图".

### 7.4 为什么 Qwen2-VL 使用 base=1,000,000?

原始 RoPE 论文用的 base 是 10,000. Qwen2-VL 把它提升到了 1,000,000. 为什么?

base 越大, 所有频率都会变小, 波长都会变长.
最低频率的波长从约 $2\pi \cdot 10000^{1} \approx 6.28 \times 10^4$ 跃升到
$2\pi \cdot 10^6 \approx 6.28 \times 10^6$.
这意味着在更长的序列中, 旋转角度的变化仍然是平缓的, 模型不容易因为角度"缠绕"而混淆远距离位置.

这与 **NTK-aware scaling** (Neural Tangent Kernel) 的思想一致:
通过拉伸频率空间, 让模型在**不经过微调**的情况下就能较好地外推到训练时未见过的长度.
base 越大, 最低频率段的"刻度"就越细, 远距离位置的区分度越高, 长上下文能力越强.

当然, 这不是"越大越好" — — 过大的 base 会让高频段也被拉伸, 导致相邻 token 的区分度下降.
$10^6$ 是 Qwen2-VL 团队根据实验选定的平衡点.

---

## 8. 相对位置性质的详细推导

这是 RoPE 最优美的性质: **注意力分数只依赖于 token 之间的相对位置**.

### 8.1 陈述

设 $q$ 在位置 $m$, $k$ 在位置 $n$. 应用 RoPE 后的注意力分数 (在 softmax 之前) 是:

$$
\text{score}(m, n) = \langle R_m q,\; R_n k \rangle
$$

我们要证明: $\text{score}(m, n) = \text{score}(m', n')$, 只要 $m - n = m' - n'$.

等价地, 我们要证明 $\langle R_m q,\; R_n k \rangle$ 只是 $(m - n)$ 的函数.

### 8.2 逐步证明

**第一步**: 利用内积的性质 $\langle Ax, y \rangle = \langle x, A^T y \rangle$ (对任意矩阵 $A$):

$$
\langle R_m q,\; R_n k \rangle = \langle q,\; R_m^T R_n k \rangle
$$

**第二步**: 计算 $R_m^T R_n$. 由于 $R_m^T = R(-m)$ (旋转的转置就是反向旋转), 以及旋转叠加性质:

$$
R_m^T R_n = R(-m) \cdot R(n) = R(n - m)
$$

这里"旋转叠加"是对**每个** $2 \times 2$ 块分别成立的 (它们是独立的旋转).

**第三步**: 代回去:

$$
\langle R_m q,\; R_n k \rangle = \langle q,\; R_{n-m} k \rangle
$$

**第四步**: 改写为等价形式. 利用 $\langle x, Ay \rangle = \langle A^T x, y \rangle$:

$$
= \langle R_{n-m}^T q,\; k \rangle = \langle R_{m-n} q,\; k \rangle
$$

至此我们证明了:

$$
\boxed{\langle R_m q,\; R_n k \rangle = \langle R_{m-n} q,\; k \rangle}
$$

右边只依赖于 $m - n$, 与绝对位置无关. $\blacksquare$

### 8.3 为什么这很重要?

想象一句话: "我 今天 很 开心".
无论这句话出现在文档的第 1 个位置还是第 10000 个位置,
"今天"和"开心"之间隔了 2 步, 它们的注意力分数应该是一样的.
RoPE 的相对位置性质保证了这一点.

这比显式的相对位置编码更优雅: 我们只在 Q 和 K 上做一次旋转 (绝对位置编码),
相对位置信息就**自动浮现**在了注意力分数中.

---

## 9. rotate_half 技巧: 从矩阵到向量化

### 9.1 标准的分块矩阵公式

按照第 6.3 节的分块对角矩阵, 对每一对 $(x_{2i}, x_{2i+1})$ 做 $2 \times 2$ 旋转需要逐对循环.
但在 GPU 上, 我们希望一次性操作整个向量.

HuggingFace (及 Qwen2-VL) 采用的 `rotate_half` 技巧巧妙地绕开了显式矩阵乘法.

### 9.2 rotate_half 的定义

```
rotate_half([x_0, x_1, ..., x_{d/2-1}, x_{d/2}, ..., x_{d-1}])
    = [-x_{d/2}, -x_{d/2+1}, ..., -x_{d-1}, x_0, x_1, ..., x_{d/2-1}]
```

注意维度的配对方式: 它把第 $i$ 个元素和第 $i + d/2$ 个元素视为一对.
这与"相邻配对" $(x_0, x_1), (x_2, x_3), \ldots$ 不同!

### 9.3 等价性证明

对于维度对 $(x_i, x_{i+d/2})$ ($0 \le i < d/2$),
它们在 cos/sin 缓存中对应的角度都是 $\alpha_i$
(因为 cos/sin 被 concatenate 了两份: $[\alpha_0, \ldots, \alpha_{d/2-1}, \alpha_0, \ldots, \alpha_{d/2-1}]$).

RoPE 公式 $y = x \cdot \cos + \text{rotate\_half}(x) \cdot \sin$ 逐元素展开后:

**对于第 $i$ 个元素** ($0 \le i < d/2$):

$$
y_i = x_i \cdot \cos(\alpha_i) + \bigl(\text{rotate\_half}(x)\bigr)_i \cdot \sin(\alpha_i)
$$

其中 $\bigl(\text{rotate\_half}(x)\bigr)_i = -x_{i+d/2}$, 所以:

$$
y_i = x_i \cos(\alpha_i) - x_{i+d/2} \sin(\alpha_i) \quad \cdots\;(*)
$$

**对于第 $i + d/2$ 个元素**:

$$
y_{i+d/2} = x_{i+d/2} \cdot \cos(\alpha_i) + \bigl(\text{rotate\_half}(x)\bigr)_{i+d/2} \cdot \sin(\alpha_i)
$$

其中 $\bigl(\text{rotate\_half}(x)\bigr)_{i+d/2} = x_i$, 所以:

$$
y_{i+d/2} = x_{i+d/2} \cos(\alpha_i) + x_i \sin(\alpha_i) \quad \cdots\;(**)
$$

把 $(*)$ 和 $(**)$ 写成矩阵形式:

$$
\begin{bmatrix} y_i \\ y_{i+d/2} \end{bmatrix}
= \begin{bmatrix} \cos\alpha_i & -\sin\alpha_i \\ \sin\alpha_i & \cos\alpha_i \end{bmatrix}
\begin{bmatrix} x_i \\ x_{i+d/2} \end{bmatrix}
$$

这**正好是** $2 \times 2$ 旋转矩阵 $R(\alpha_i)$ 作用在 $(x_i, x_{i+d/2})$ 上的结果! $\blacksquare$

所以 `rotate_half` 技巧等价于分块对角旋转矩阵, 但只用了**逐元素乘法和加法** — —
不需要任何矩阵乘法, 非常适合 GPU 并行.

---

## 10. 完整数值例子

让我们用 `head_dim = 8`, `base = 10000`, 位置 $p = 3$ 来走一遍完整流程.

### 10.1 计算频率

$d = 8$, 共 $d/2 = 4$ 个频率:

$$
\theta_0 = 10000^{-0/8} = 10000^{0} = 1.0
$$

$$
\theta_1 = 10000^{-2/8} = 10000^{-0.25} = \frac{1}{10000^{0.25}} = \frac{1}{10} = 0.1
$$

$$
\theta_2 = 10000^{-4/8} = 10000^{-0.5} = \frac{1}{100} = 0.01
$$

$$
\theta_3 = 10000^{-6/8} = 10000^{-0.75} = \frac{1}{1000} = 0.001
$$

### 10.2 计算角度

位置 $p = 3$ 时:

$$
\alpha_0 = 3 \times 1.0 = 3.0
$$

$$
\alpha_1 = 3 \times 0.1 = 0.3
$$

$$
\alpha_2 = 3 \times 0.01 = 0.03
$$

$$
\alpha_3 = 3 \times 0.001 = 0.003
$$

### 10.3 cos 和 sin (扩展到 dim 8)

cos/sin 缓存将角度 concatenate 一次: $[\alpha_0, \alpha_1, \alpha_2, \alpha_3, \alpha_0, \alpha_1, \alpha_2, \alpha_3]$

$$
\cos = [\cos(3.0),\; \cos(0.3),\; \cos(0.03),\; \cos(0.003),\; \cos(3.0),\; \cos(0.3),\; \cos(0.03),\; \cos(0.003)]
$$

$$
\approx [-0.9900,\; 0.9553,\; 0.9996,\; 1.0000,\; -0.9900,\; 0.9553,\; 0.9996,\; 1.0000]
$$

$$
\sin = [\sin(3.0),\; \sin(0.3),\; \sin(0.03),\; \sin(0.003),\; \sin(3.0),\; \sin(0.3),\; \sin(0.03),\; \sin(0.003)]
$$

$$
\approx [0.1411,\; 0.2955,\; 0.0300,\; 0.0030,\; 0.1411,\; 0.2955,\; 0.0300,\; 0.0030]
$$

### 10.4 应用 RoPE

设 $x = [1, 2, 3, 4, 5, 6, 7, 8]$.

**rotate_half(x)**: 前半 = $[1,2,3,4]$, 后半 = $[5,6,7,8]$

$$
\text{rotate\_half}(x) = [-5, -6, -7, -8, 1, 2, 3, 4]
$$

**逐元素计算 $y = x \cdot \cos + \text{rotate\_half}(x) \cdot \sin$**:

$$
y_0 = 1 \times (-0.9900) + (-5) \times 0.1411 = -0.9900 - 0.7055 = -1.6955
$$

$$
y_1 = 2 \times 0.9553 + (-6) \times 0.2955 = 1.9106 - 1.7730 = 0.1376
$$

$$
y_2 = 3 \times 0.9996 + (-7) \times 0.0300 = 2.9988 - 0.2100 = 2.7888
$$

$$
y_3 = 4 \times 1.0000 + (-8) \times 0.0030 = 4.0000 - 0.0240 = 3.9760
$$

$$
y_4 = 5 \times (-0.9900) + 1 \times 0.1411 = -4.9500 + 0.1411 = -4.8089
$$

$$
y_5 = 6 \times 0.9553 + 2 \times 0.2955 = 5.7318 + 0.5910 = 6.3228
$$

$$
y_6 = 7 \times 0.9996 + 3 \times 0.0300 = 6.9972 + 0.0900 = 7.0872
$$

$$
y_7 = 8 \times 1.0000 + 4 \times 0.0030 = 8.0000 + 0.0120 = 8.0120
$$

所以 RoPE 的输出是:

$$
y \approx [-1.6955,\; 0.1376,\; 2.7888,\; 3.9760,\; -4.8089,\; 6.3228,\; 7.0872,\; 8.0120]
$$

观察: 高频维度 ($y_0, y_4$) 变化最大, 低频维度 ($y_3, y_7$) 几乎没变. 这正是多尺度编码的体现.

---

## 11. 数值验证: 相对位置性质

取 $q = [1,0,1,0,1,0,1,0]$, $k = [0,1,0,1,0,1,0,1]$.
用 `head_dim = 8`, `base = 10000`.

**场景 A**: $q$ 在位置 5, $k$ 在位置 8 (相对距离 = 3)

$$
q_{\text{rot}} = q \cdot \cos[5] + \text{rotate\_half}(q) \cdot \sin[5]
$$

$$
k_{\text{rot}} = k \cdot \cos[8] + \text{rotate\_half}(k) \cdot \sin[8]
$$

$$
\text{score}_A = \langle q_{\text{rot}},\; k_{\text{rot}} \rangle
$$

**场景 B**: $q$ 在位置 20, $k$ 在位置 23 (相对距离 = 3)

$$
\text{score}_B = \langle q_{\text{rot}}',\; k_{\text{rot}}' \rangle
$$

由于我们在第 8 节证明了 $\langle R_m q, R_n k \rangle = \langle R_{m-n} q, k \rangle$,
而两个场景的相对距离都是 $n - m = 3$, 所以 $\text{score}_A = \text{score}_B$.

在 `impl.py` 的 `test_relative_position_property()` 函数中,
实际用三对位置 $(5,8)$, $(20,23)$, $(50,53)$ 做了验证,
三个点积的数值差异在 $10^{-3}$ 量级内, 确认了相对位置不变性.

---

## 12. 数值验证: 范数保持

旋转矩阵是正交矩阵, 所以 $\|R_p x\| = \|x\|$.

用第 10 节的例子验证:

$$
\|x\| = \sqrt{1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2 + 7^2 + 8^2} = \sqrt{204} \approx 14.2829
$$

$$
\|y\| = \sqrt{(-1.6955)^2 + 0.1376^2 + 2.7888^2 + 3.9760^2 + (-4.8089)^2 + 6.3228^2 + 7.0872^2 + 8.0120^2}
$$

$$
= \sqrt{2.8747 + 0.0189 + 7.7774 + 15.8086 + 23.1255 + 39.9778 + 50.2284 + 64.1921}
$$

$$
= \sqrt{204.0034} \approx 14.2830
$$

差异小于 $10^{-3}$ (由浮点精度造成), 范数几乎完全保持. ✅

在 `impl.py` 的 `test_rotation_preserves_norm()` 中,
对随机的 Q 和 K 张量 (`head_dim=128`, `seq_len=32`) 进行了系统验证,
旋转前后的范数误差均在 $10^{-4}$ 以内.

---

## 13. 2D 视觉 RoPE

### 13.1 为什么图像需要 2D 位置编码?

文本是天然的一维序列: token 0, token 1, token 2......
但图像被切成 patch 后, patch 之间有**二维的空间邻接关系**.

考虑一个 $24 \times 24$ 的 patch 网格. 如果把它简单展平成长度为 576 的一维序列,
patch $(0, 23)$ (第 0 行最右边) 和 patch $(1, 0)$ (第 1 行最左边) 在一维中是相邻的 (位置 23 和 24),
但在二维空间中它们隔了整整一行的宽度!

反过来, patch $(0, 0)$ 和 patch $(1, 0)$ 在空间中紧紧相邻 (上下),
但在一维展平后它们的距离是 24.

用 1D RoPE 处理图像会把空间结构彻底打乱. 所以我们需要 **2D RoPE**.

### 13.2 Qwen2-VL 视觉编码器的做法

Qwen2-VL 的视觉编码器参数:

- `embed_dim = 1280`, `num_heads = 16`, `head_dim = 80`
- MLP 隐藏层 = 5120, 共 32 个 block

对于 `head_dim = 80` 的 2D RoPE:

1. 把 80 维**对半分**成两组: 前 40 维 + 后 40 维
2. `rotate_half` 将 dim $i$ 与 dim $i + 40$ 配对
3. 每组内部有 20 个频率 ($40 / 2 = 20$)
4. **前 20 个频率**: 使用 **height** 坐标
5. **后 20 个频率**: 使用 **width** 坐标

角度的拼接结构如下:

$$
\text{angles} = \underbrace{[h \cdot \theta_0, \ldots, h \cdot \theta_{19}]}_{\text{height 频率}} \;\|\;
\underbrace{[w \cdot \theta_0, \ldots, w \cdot \theta_{19}]}_{\text{width 频率}}
$$

然后 concatenate 一次变成 80 维: $[\text{angles}, \text{angles}]$.

### 13.3 grid_thw 参数

`grid_thw` 的形状是 `(num_videos, 3)`, 每行为 $[t, h, w]$:

- $t$: 帧数 (对于静态图片, $t = 1$)
- $h$: 高度方向上的 patch 数
- $w$: 宽度方向上的 patch 数

例如 `grid_thw = [[1, 24, 24]]` 表示:
1 帧, $24 \times 24 = 576$ 个 patch.

### 13.4 例子: patch (3, 5) 的位置编码

对于位置 $(h, w) = (3, 5)$ 的 patch, 频率 $\theta_i = \text{base}^{-2i/40}$:

- 前 20 个频率用高度坐标 $h = 3$: 角度为 $[3\theta_0, 3\theta_1, \ldots, 3\theta_{19}]$
- 后 20 个频率用宽度坐标 $w = 5$: 角度为 $[5\theta_0, 5\theta_1, \ldots, 5\theta_{19}]$

这样, 两个在同一行的 patch (如 $(3,5)$ 和 $(3,7)$) 的 height 分量完全相同,
只有 width 分量不同 — — 模型能清楚地知道它们是"同行不同列".

---

## 14. 3D M-RoPE (Multimodal RoPE)

### 14.1 为什么需要三个维度?

Qwen2-VL 不仅处理文本和图像, 还处理**视频**.
视频有三个维度: **时间** (帧), **高度**, **宽度**.
即使是纯文本, Qwen2-VL 也用三维位置来表示 (只不过三个维度的值相同).

为此, Qwen2-VL 的文本 decoder 采用了 **M-RoPE (Multimodal Rotary Position Embedding)**.

### 14.2 模型参数

Qwen2-VL-2B-Instruct 的文本 decoder:

- `hidden_size = 1536`, `num_heads = 12`, `num_kv_heads = 2`, `head_dim = 128`
- MLP 隐藏层 = 8960, 28 层, 词表大小 = 151936
- RoPE `base = 1,000,000`, `mrope_section = [16, 24, 24]`

### 14.3 mrope_section 的含义

`mrope_section = [16, 24, 24]` 表示 `head_dim / 2 = 64` 个频率按如下方式分配:

| 维度            | 分配的频率数 | 频率索引范围 | 对应 head_dim 维数   |
| --------------- | ------------ | ------------ | -------------------- |
| temporal (时间) | 16           | $[0, 16)$    | 32 (= $2 \times 16$) |
| height (高度)   | 24           | $[16, 40)$   | 48 (= $2 \times 24$) |
| width (宽度)    | 24           | $[40, 64)$   | 48 (= $2 \times 24$) |
| **总计**        | **64**       | $[0, 64)$    | **128**              |

为什么时间维度分配的频率最少? 因为视频的帧数通常远少于空间 patch 数.
一个视频可能只有几帧到几十帧, 但空间上可能有 $24 \times 24 = 576$ 个 patch.
时间维度的"分辨率"需求较低, 16 个频率已经足够.
而空间维度需要更高的分辨率来区分密集排列的 patch, 所以分配更多频率.

### 14.4 position_ids 的结构

`position_ids` 的形状是 `(3, seq_len)`:

- 第 0 行: 每个 token 的 **temporal** 位置
- 第 1 行: 每个 token 的 **height** 位置
- 第 2 行: 每个 token 的 **width** 位置

**纯文本 token**: 三行完全相同.
例如 token 在位置 $p$, 则 `position_ids[:, p] = [p, p, p]`.
这时 M-RoPE 退化为标准的 1D RoPE (因为三个维度用同一个位置值,
拼接后等价于一个位置值作用于所有频率).

**视觉 token**: 三行不同.
例如视频第 2 帧, 行 3, 列 5 的 patch: `position_ids[:, p] = [2, 3, 5]`.

### 14.5 计算流程伪代码

```python
# 1. 生成所有 64 个频率
inv_freq = base ** (-arange(0, head_dim, 2) / head_dim)   # (64,)

# 2. 按 mrope_section 累积索引: [0, 16, 40, 64]
boundaries = cumsum([0, 16, 24, 24])

# 3. 对每个维度，用对应的 position_ids 和频率计算角度
angles_t = outer(position_ids[0], inv_freq[0:16])   # (seq_len, 16) - temporal
angles_h = outer(position_ids[1], inv_freq[16:40])   # (seq_len, 24) - height
angles_w = outer(position_ids[2], inv_freq[40:64])   # (seq_len, 24) - width

# 4. 拼接三个维度的角度
angles = concat([angles_t, angles_h, angles_w], axis=-1)  # (seq_len, 64)

# 5. 复制一次凑满 head_dim
angles = concat([angles, angles], axis=-1)                 # (seq_len, 128)

# 6. 计算 cos/sin
cos, sin = cos(angles), sin(angles)                        # 各 (seq_len, 128)
```

第 5 步的复制是关键: `rotate_half` 将 dim $i$ 与 dim $i + 64$ 配对,
所以两个位置的角度必须相同, 这正是 concatenate 两份角度所保证的.

---

## 15. NumPy 实现代码详解

以下是 `impl.py` 中核心函数的逐行解读.

### 15.1 rotate_half

```python
def rotate_half(x: np.ndarray) -> np.ndarray:
    """将向量前后两半交换并取负，等价于复数乘以 i。"""
    half = x.shape[-1] // 2          # 取最后一维的一半长度，如 head_dim=8 → half=4
    x1 = x[..., :half]               # 前半部分: x[0..half-1]
    x2 = x[..., half:]               # 后半部分: x[half..d-1]
    return np.concatenate([-x2, x1], axis=-1)  # 拼接: [-后半, 前半]
```

关键点: `...` 语法让函数能处理任意前导维度 (batch, seq_len, num_heads 等).
`-x2` 是对后半部分取负. 最终结果的形状与输入完全相同.

### 15.2 apply_rotary_pos_emb

```python
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """对 Q 和 K 应用旋转位置编码。"""
    cos = np.expand_dims(cos, axis=unsqueeze_dim)  # 在 num_heads 维度插入 1，以便广播
    sin = np.expand_dims(sin, axis=unsqueeze_dim)  # 同上
    q_embed = q * cos + rotate_half(q) * sin       # Q 的旋转: x·cos + rot(x)·sin
    k_embed = k * cos + rotate_half(k) * sin       # K 的旋转: 同样的公式
    return q_embed, k_embed
```

`unsqueeze_dim=1` 的作用: cos/sin 的形状通常是 `(seq_len, head_dim)`,
而 Q/K 的形状是 `(batch, seq_len, num_heads, head_dim)`.
在 axis=1 (即 num_heads 的位置) 插入一个维度后变成 `(seq_len, 1, head_dim)`,
NumPy 的广播机制会自动把它复制到所有 head.

### 15.3 compute_rope_frequencies

```python
def compute_rope_frequencies(head_dim, max_position, base=10000.0):
    """计算 1D RoPE 的 cos/sin 缓存。"""
    # 频率: base^(-2i/d)，i=0,1,...,d/2-1
    inv_freq = base ** (-np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)
    # inv_freq 形状: (head_dim // 2,)

    positions = np.arange(max_position, dtype=np.float64)  # [0, 1, ..., max_position-1]

    # 外积: 每个位置 × 每个频率 = 旋转角度
    angles = np.outer(positions, inv_freq)  # (max_position, head_dim // 2)

    # 复制一次，让 rotate_half 的配对关系正确
    angles = np.concatenate([angles, angles], axis=-1)  # (max_position, head_dim)

    cos = np.cos(angles).astype(np.float32)  # 转回 float32 节省内存
    sin = np.sin(angles).astype(np.float32)
    return cos, sin
```

`np.outer(positions, inv_freq)` 是整个计算的核心:
一个 `(max_position,)` 向量和一个 `(head_dim//2,)` 向量的外积,
得到 `(max_position, head_dim//2)` 的角度矩阵, 其中第 $[p, i]$ 个元素就是 $p \cdot \theta_i$.

### 15.4 compute_mrope

```python
def compute_mrope(position_ids, head_dim, mrope_section, base=1_000_000.0):
    """计算 3D 多模态 RoPE 的 cos/sin。"""
    # 生成所有 head_dim/2 个频率
    inv_freq = base ** (-np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)

    # 按 mrope_section 拆分频率
    sec_cumsum = np.cumsum([0] + mrope_section)  # 如 [0, 16, 40, 64]
    angle_parts = []
    for dim_idx in range(3):  # temporal=0, height=1, width=2
        start, end = int(sec_cumsum[dim_idx]), int(sec_cumsum[dim_idx + 1])
        freq_slice = inv_freq[start:end]              # 该维度对应的频率子集
        pos = position_ids[dim_idx].astype(np.float64) # 该维度的位置 (seq_len,)
        angles = np.outer(pos, freq_slice)             # (seq_len, section_size)
        angle_parts.append(angles)

    # 拼接三个维度的角度，再复制凑成 head_dim
    angles = np.concatenate(angle_parts, axis=-1)      # (seq_len, head_dim//2)
    angles = np.concatenate([angles, angles], axis=-1)  # (seq_len, head_dim)

    cos = np.cos(angles).astype(np.float32)
    sin = np.sin(angles).astype(np.float32)
    return cos, sin
```

与 1D 版本的关键区别在于 `for dim_idx in range(3)` 循环:
每个空间维度用**自己的** position_ids 和**自己的**频率子集来计算角度,
然后拼接在一起. 当三个维度的 position_ids 相同时 (纯文本),
结果与 1D RoPE 完全一致 (`impl.py` 中的 `test_mrope_text_only()` 验证了这一点).

---

## 16. 与 Qwen2-VL 架构的联系

### 16.1 视觉编码器 (ViT)

- 32 个 Transformer block
- 每个 block 的自注意力层: 先计算 Q 和 K, 然后应用 **2D RoPE** (`compute_vision_rope`)
- `head_dim = 80`, 分成 $40 + 40$: height 20 频率 + width 20 频率
- cos/sin 根据 `grid_thw` 参数**预计算并缓存** — — 对于同一张图片的所有 block 和所有 head,
  位置编码完全相同, 只计算一次

### 16.2 文本 Decoder

- 28 层 Transformer
- 每层的自注意力: 先计算 Q 和 K (含 GQA, `num_heads=12`, `num_kv_heads=2`),
  然后应用 **M-RoPE** (`compute_mrope`)
- `head_dim = 128`, 按 `[16, 24, 24]` 分配给 temporal/height/width
- cos/sin 同样**预计算并缓存** — — 不需要在每次 forward 时重新计算

### 16.3 预计算的好处

位置编码的 cos 和 sin 只取决于位置和模型超参数, 不依赖于输入内容.
所以可以在模型初始化时 (或第一次 forward 时) 一次性算好,
存在一个缓存中 (通常注册为 PyTorch 的 `buffer`), 之后直接查表取用.
这避免了在每一层, 每一步都重复计算三角函数, 显著提升推理速度.

---

## 17. 常见误解与陷阱

### ❌ 误解 1: "RoPE 是加到输入上的"

不对. 正弦位置编码 (Vaswani 2017) 是**加到**词嵌入向量上的: $x' = x + \text{PE}$.
RoPE 完全不同 — — 它是一种**乘性的旋转变换**, 作用在 Q 和 K 上, 而不是输入嵌入上.

$$
\text{正弦编码}:\; x' = x + \text{PE}(p) \qquad \text{ (加法) }
$$

$$
\text{RoPE}:\; q' = R_p \cdot q, \quad k' = R_p \cdot k \qquad \text{ (乘法/旋转) }
$$

### ❌ 误解 2: "RoPE 的频率是可学习的"

不对. 频率 $\theta_i = \text{base}^{-2i/d}$ 完全由超参数 `base` 和 `head_dim` 决定,
**没有**任何可学习参数. 这是 RoPE 的设计哲学之一:
用确定性的数学结构来编码位置, 而不是交给梯度下降去学.

### ❌ 误解 3: "rotate_half 配对的是相邻维度"

不对. 很多初学者以为 rotate*half 是把 $(x_0, x_1)$, $(x_2, x_3)$ 等**相邻**维度配对.
实际上, 它把 $(x_0, x*{d/2})$, $(x*1, x*{d/2+1})$ 等**间隔 $d/2$** 的维度配对.

这两种配对方式在数学上是等价的 (只是维度排列不同), 但如果你在实现时搞混了,
cos/sin 缓存的排列就会对不上, 结果就会完全错误.

### ❌ 误解 4: "base 越大越好"

不对. 更大的 base 意味着所有频率都更低, 波长都更长.
好处是长上下文外推能力更强 (远距离位置更容易区分).
坏处是最高频率也被拉低了, **相邻 token 之间的旋转角度差变小**,
模型区分近距离位置的能力下降.

`base = 10000` (LLaMA) 和 `base = 1000000` (Qwen2-VL) 代表了不同的设计取舍.
前者在短上下文中可能有更好的局部位置感知, 后者在超长上下文中更有优势.

---

## 18. 关键性质总结

| 性质              | 说明                                                |
| ----------------- | --------------------------------------------------- |
| **作用对象**      | Q 和 K 向量 (不是输入嵌入)                          |
| **作用方式**      | 乘性旋转 (不是加性叠加)                             |
| **核心操作**      | 把每对维度 $(x_i, x_{i+d/2})$ 做 2D 旋转            |
| **频率设计**      | $\theta_i = \text{base}^{-2i/d}$, 几何级数          |
| **相对位置**      | $\langle R_m q, R_n k \rangle$ 只依赖 $m - n$       |
| **保范性**        | $\|R_p x\| = \|x\|$ (旋转矩阵是正交矩阵)            |
| **rotate_half**   | $[-x_{d/2:}, x_{:d/2}]$, 等价于分块旋转矩阵         |
| **1D RoPE**       | 文本序列: 位置 $p = 0, 1, 2, \ldots$                |
| **2D 视觉 RoPE**  | 图像 patch: head_dim 对半分给 height/width          |
| **3D M-RoPE**     | 多模态: 按 mrope_section 分给 temporal/height/width |
| **Qwen2-VL 视觉** | `head_dim=80`, 32 blocks, base=10000                |
| **Qwen2-VL 文本** | `head_dim=128`, 28 layers, base=1,000,000           |
| **参数量**        | 零 — — 所有频率都是确定性的, 无需学习               |
| **缓存策略**      | cos/sin 预计算并缓存, 不随 forward 重复计算         |
