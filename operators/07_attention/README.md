# 注意力机制 — — 从直觉到 Qwen2-VL 的工程实现

> 本文面向具有基础线性代数知识, 但没有深度学习背景的工科研究生.
> 数学公式使用 MathJax 渲染 (`$...$` 行内, `$$...$$` 块级).
> 中文行文为主, 英文技术术语保留原文.

---

## 1. 从人类注意力说起

读下面这句中文:

> "小猫坐在垫子上, **它**很舒服. "

当你的目光扫到"它"这个字时, 大脑几乎不假思索地把它与前面的"小猫"联系了起来, 而不是"垫子". 这个过程如此自然, 以至于你可能从未意识到自己正在做一件极其精妙的事 — — **选择性关注**.

这就是"注意力 (Attention) "的本质: 面对一段输入, 模型 (或大脑) 需要动态决定**关注哪些部分, 忽略哪些部分**. 对于人来说, 它是上亿年进化的产物; 对于机器来说, 它是一个可学习的函数.

一个更宏观的问题随之而来: 如果我们希望一台机器也能"阅读"一段文字, "观看"一张图片, 它应该如何决定把"目光"投向哪里? 这正是本文要回答的问题.

---

## 2. 神经注意力简史

在注意力机制出现之前, 序列到序列 (seq2seq) 模型依赖 **固定长度的向量** 来压缩整个输入序列:

### 2.1 固定长度编码的瓶颈

Sutskever et al. (2014) 提出的 seq2seq 框架由 Encoder 和 Decoder 两部分组成. Encoder 把输入序列逐词读入一个 RNN, 最终时刻的 hidden state $\mathbf{h}_T$ 被当作整个句子的"摘要"传给 Decoder.

问题很明显: 一个固定维度的向量 $\mathbf{h}_T$ 需要承载所有信息. 当输入序列变长, 信息势必丢失 — — 这被称为**信息瓶颈 (information bottleneck) **.

### 2.2 Bahdanau Attention (2014)

Bahdanau et al. 提出了一个关键的突破: **让 Decoder 在每一步都能"回头看" Encoder 的所有 hidden states**.

具体做法是定义一个 **alignment model** (对齐模型) $a(s_{t-1}, h_j)$, 它用一个小型前馈网络计算当前 Decoder 状态 $s_{t-1}$ 与第 $j$ 个 Encoder 状态 $h_j$ 的"相关程度":

$$
e_{tj} = a(s_{t-1}, h_j) = v^T \tanh(W_1 s_{t-1} + W_2 h_j)
$$

然后做 softmax 归一化得到注意力权重 $\alpha_{tj}$, 最后加权求和得到 context 向量:

$$
\alpha_{tj} = \frac{\exp(e_{tj})}{\sum_{k=1}^{T_x} \exp(e_{tk})}, \quad c_t = \sum_{j=1}^{T_x} \alpha_{tj} h_j
$$

这种方式被称为 **additive attention** (加性注意力), 因为它在隐藏层中用加法组合了两个输入.

### 2.3 Luong Attention (2015)

Luong et al. 简化了 Bahdanau 的做法, 提出了三种计算 alignment score 的方式:

| 名称        | 公式                              | 说明                       |
| ----------- | --------------------------------- | -------------------------- |
| **dot**     | $e_{tj} = s_t^T h_j$              | 直接点积, 最简洁           |
| **general** | $e_{tj} = s_t^T W h_j$            | 中间加一个可学习矩阵       |
| **concat**  | $e_{tj} = v^T \tanh(W[s_t; h_j])$ | 拼接后过网络 (同 Bahdanau) |

其中 **dot-product attention** 因其计算效率而成为后续工作的主流选择.

### 2.4 Self-Attention 与 Transformer (Vaswani et al., 2017)

2017 年, Google 的论文 "Attention Is All You Need" 提出了 **Transformer** 架构, 做出了一个划时代的决定: **完全移除 RNN, 仅用 attention 构建模型**.

核心创新:

1. **Self-Attention**: 不再是 Decoder 看 Encoder; 而是序列中的每个位置都去看序列中的所有位置 (包括自己).
2. **Multi-Head Attention**: 把注意力拆分成多个"头", 每个头学习不同的注意力模式.
3. **Positional Encoding**: 因为没有了 RNN 的顺序性, 需要额外编码位置信息.

### 2.5 发展时间线

| 年份 | 里程碑                        | 关键思想                            |
| ---- | ----------------------------- | ----------------------------------- |
| 2014 | Sutskever et al. seq2seq      | 固定长度编码                        |
| 2014 | Bahdanau Attention            | "回头看"encoder, additive alignment |
| 2015 | Luong Attention               | dot-product 变体, 更高效            |
| 2017 | Transformer                   | Self-Attention, 移除 RNN            |
| 2019 | Multi-Query Attention (MQA)   | 所有 Q 头共享一套 KV                |
| 2023 | Grouped Query Attention (GQA) | 折中方案, 分组共享 KV               |

---

## 3. Q, K, V — — 从第一性原理出发

### 3.1 数据库查询的类比

在理解 Q, K, V 之前, 先想象一个简单的场景 — — **数据库查询**:

- **Query (查询) **: 你在搜索框里输入的关键词 — — "我想找什么? "
- **Key (键) **: 数据库中每条记录的索引/标签 — — "这条记录描述的是什么? "
- **Value (值) **: 数据库中每条记录的实际内容 — — "这条记录的内容是什么? "

搜索的过程就是: 用 Query 去和每个 Key 做匹配, 匹配度越高, 就越关注对应的 Value.

### 3.2 形式化定义

给定输入矩阵 $X \in \mathbb{R}^{n \times d_{\text{model}}}$ ($n$ 为序列长度, $d_{\text{model}}$ 为模型维度), 我们学习三个投影矩阵:

$$
W^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}, \quad W^K \in \mathbb{R}^{d_{\text{model}} \times d_k}, \quad W^V \in \mathbb{R}^{d_{\text{model}} \times d_v}
$$

然后:

$$
Q = X W^Q, \quad K = X W^K, \quad V = X W^V
$$

其中 $Q \in \mathbb{R}^{n \times d_k}$, $K \in \mathbb{R}^{n \times d_k}$, $V \in \mathbb{R}^{n \times d_v}$.

### 3.3 为什么需要三个不同的投影?

你可能会问: 为什么不直接用 $X$ 本身做 Query, Key 和 Value? 答案是: **三者扮演不同的角色**.

- **Q (Query) **: 编码"我在寻找什么样的信息".
- **K (Key) **: 编码"我能提供什么样的信息" — — 相当于每个位置对外展示的"标签".
- **V (Value) **: 编码"我实际携带的内容".

用不同的线性变换把同一个输入投影到三个不同的空间, 让模型能够把"匹配"和"内容提取"解耦. 打个比方: 你去图书馆查书, 你心中的搜索意图 (Q), 书的封面标题 (K), 书的正文内容 (V) 是三件不同的事.

---

## 4. 点积作为相似度度量

### 4.1 回顾: 向量点积的几何含义

两个向量 $\mathbf{a}, \mathbf{b} \in \mathbb{R}^d$ 的点积定义为:

$$
\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{d} a_i b_i = \|\mathbf{a}\| \|\mathbf{b}\| \cos\theta
$$

其中 $\theta$ 是两个向量之间的夹角.

这告诉我们: **点积同时编码了向量的"大小"和"方向一致性"**. 方向越一致 ($\cos\theta$ 越大), 点积越大.

### 4.2 与 cosine similarity 的关系

Cosine similarity 定义为:

$$
\text{cos\_sim}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}
$$

它只保留方向信息, 去掉了长度的影响. 点积可以看作"未归一化的 cosine similarity".

### 4.3 为什么用点积而不是 L2 距离?

| 度量        | 优点                              | 缺点                       |
| ----------- | --------------------------------- | -------------------------- |
| **点积**    | 计算快 (矩阵乘法), 可微, GPU 友好 | 受向量长度影响             |
| **L2 距离** | 直觉上更"像"距离                  | 需要平方和开根号, 计算量大 |
| **cosine**  | 只看方向                          | 需要归一化, 额外计算       |

在 Transformer 中, $Q$ 和 $K$ 的点积可以用高效的矩阵乘法 $QK^T$ 一次性计算所有 pair — — 这是选择点积的核心原因.

### 4.4 小例子

取 $\mathbf{a} = [1, 0, 1]$, $\mathbf{b} = [1, 1, 0]$:

$$
\mathbf{a} \cdot \mathbf{b} = 1 \times 1 + 0 \times 1 + 1 \times 0 = 1
$$

$$
\|\mathbf{a}\| = \sqrt{1^2 + 0^2 + 1^2} = \sqrt{2}, \quad \|\mathbf{b}\| = \sqrt{1^2 + 1^2 + 0^2} = \sqrt{2}
$$

$$
\cos\theta = \frac{1}{\sqrt{2} \cdot \sqrt{2}} = \frac{1}{2}, \quad \theta = 60°
$$

直觉上, 这两个向量有一半的分量方向一致 (第一个维度), 另外两个维度"正交", 所以夹角 60° 是合理的.

---

## 5. 缩放因子 $1/\sqrt{d_k}$ — — 完整推导

这是 Transformer 论文中一个容易被忽略但极其重要的细节. 让我们从头推导它存在的原因.

### 5.1 假设与推导

假设 $\mathbf{q}$ 和 $\mathbf{k}$ 都是 $d_k$ 维向量, 每个分量独立地从标准正态分布 $\mathcal{N}(0, 1)$ 中采样:

$$
q_i \sim \mathcal{N}(0, 1), \quad k_i \sim \mathcal{N}(0, 1), \quad i = 1, \ldots, d_k
$$

且 $q_i$ 和 $k_j$ 互相独立.

**Step 1**: 单个乘积项的期望与方差.

$$
\mathbb{E}[q_i k_i] = \mathbb{E}[q_i] \cdot \mathbb{E}[k_i] = 0 \times 0 = 0
$$

$$
\text{Var}[q_i k_i] = \mathbb{E}[q_i^2 k_i^2] - (\mathbb{E}[q_i k_i])^2 = \mathbb{E}[q_i^2] \cdot \mathbb{E}[k_i^2] - 0 = 1 \times 1 = 1
$$

(这里用了独立性: $\mathbb{E}[q_i^2 k_i^2] = \mathbb{E}[q_i^2] \mathbb{E}[k_i^2]$.)

**Step 2**: 点积 $\mathbf{q} \cdot \mathbf{k} = \sum_{i=1}^{d_k} q_i k_i$ 是 $d_k$ 个独立随机变量的和.

$$
\mathbb{E}[\mathbf{q} \cdot \mathbf{k}] = \sum_{i=1}^{d_k} \mathbb{E}[q_i k_i] = 0
$$

$$
\text{Var}[\mathbf{q} \cdot \mathbf{k}] = \sum_{i=1}^{d_k} \text{Var}[q_i k_i] = d_k
$$

**Step 3**: 标准差.

$$
\text{Std}[\mathbf{q} \cdot \mathbf{k}] = \sqrt{d_k}
$$

### 5.2 为什么方差大是个问题?

考虑 softmax 函数 $\text{softmax}(z_i) = e^{z_i} / \sum_j e^{z_j}$. 当输入 $z$ 的分量之间差异很大时, $\exp$ 函数会让最大值指数级地压过其他值, 导致输出接近 one-hot 向量.

这就是所谓的 **softmax 饱和 (saturation) **. 在饱和区域, 梯度几乎为零, 模型无法有效学习 — — 这就是**梯度消失**问题.

### 5.3 解决方案

除以 $\sqrt{d_k}$:

$$
\frac{\mathbf{q} \cdot \mathbf{k}}{\sqrt{d_k}}
$$

缩放后的方差:

$$
\text{Var}\!\left[\frac{\mathbf{q} \cdot \mathbf{k}}{\sqrt{d_k}}\right] = \frac{\text{Var}[\mathbf{q} \cdot \mathbf{k}]}{d_k} = \frac{d_k}{d_k} = 1
$$

方差被归一化回 1, softmax 的输入保持在合理范围内.

### 5.4 数值直觉

在 Qwen2-VL 的文本解码器中, $d_k = 128$ (head_dim).

- 不缩放时: 点积的标准差 $\approx \sqrt{128} \approx 11.3$. 一个典型的点积值可能在 $[-20, +20]$ 范围内.
- 缩放后: 标准差 $\approx 1$, 点积值在 $[-3, +3]$ 范围内 — — softmax 的"舒适区".

---

## 6. Softmax 的性质

### 6.1 定义

给定向量 $\mathbf{z} = [z_1, z_2, \ldots, z_n]$, softmax 定义为:

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}, \quad i = 1, \ldots, n
$$

### 6.2 关键性质

1. **输出范围**: 每个分量 $\text{softmax}(z_i) \in (0, 1)$, 严格正.
2. **归一化**: $\sum_{i} \text{softmax}(z_i) = 1$, 可以解释为概率分布.
3. **可微**: 处处连续可导, 适合梯度下降.
4. **保序性 (Monotone-preserving) **: 如果 $z_i > z_j$, 则 $\text{softmax}(z_i) > \text{softmax}(z_j)$.

### 6.3 数值稳定性

直接计算 $e^{z_i}$ 可能导致数值溢出. 例如 $e^{1000}$ 已经超出 float64 的表示范围.

**解决方法**: 利用 softmax 的平移不变性 — — 先减去最大值:

$$
\text{softmax}(z_i) = \frac{e^{z_i - \max(\mathbf{z})}}{\sum_j e^{z_j - \max(\mathbf{z})}}
$$

这在数学上等价 (分子分母同除以 $e^{\max(\mathbf{z})}$), 但保证了 $\exp$ 的输入 $\leq 0$, 避免溢出.

### 6.4 Temperature scaling

在 softmax 中引入温度参数 $T > 0$:

$$
\text{softmax}(z_i / T) = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}
$$

- **高温 ($T \to \infty$) **: 所有 $z_i/T \to 0$, 输出趋近均匀分布 — — 模型"不太确定".
- **低温 ($T \to 0^+$) **: 输出趋近 one-hot 向量, 最大值独占 — — 模型"非常确定".
- **$T = 1$**: 标准 softmax.

这与统计力学中的 **Boltzmann 分布** $p_i \propto e^{-E_i / k_B T}$ 有着深刻的联系: 注意力权重就像粒子在不同能量状态上的分布. 温度越低, 粒子越倾向于集中在最低能量状态.

---

## 7. 缩放点积注意力 (SDPA) — — 3 token 手算示例

现在把所有零件组装起来. 我们用一个具体的 3 token, $d_k = 2$ 的例子走完整个 SDPA 流程.

### 7.1 定义输入

$$
Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}, \quad
K = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0.5 & 0.5 \end{bmatrix}, \quad
V = \begin{bmatrix} 10 & 0 \\ 0 & 10 \\ 5 & 5 \end{bmatrix}
$$

三个 token, 每个 token 有 2 维.

### 7.2 Step 1: 计算 $QK^T$

$$
QK^T = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}
\begin{bmatrix} 1 & 0 & 0.5 \\ 0 & 1 & 0.5 \end{bmatrix}
= \begin{bmatrix}
1 \cdot 1 + 0 \cdot 0 & 1 \cdot 0 + 0 \cdot 1 & 1 \cdot 0.5 + 0 \cdot 0.5 \\
0 \cdot 1 + 1 \cdot 0 & 0 \cdot 0 + 1 \cdot 1 & 0 \cdot 0.5 + 1 \cdot 0.5 \\
1 \cdot 1 + 1 \cdot 0 & 1 \cdot 0 + 1 \cdot 1 & 1 \cdot 0.5 + 1 \cdot 0.5
\end{bmatrix}
= \begin{bmatrix} 1 & 0 & 0.5 \\ 0 & 1 & 0.5 \\ 1 & 1 & 1 \end{bmatrix}
$$

矩阵的 $(i, j)$ 元素就是第 $i$ 个 query 与第 $j$ 个 key 的点积 — — **原始相似度得分**.

### 7.3 Step 2: 缩放

$$
\frac{QK^T}{\sqrt{d_k}} = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & 0 & 0.5 \\ 0 & 1 & 0.5 \\ 1 & 1 & 1 \end{bmatrix}
\approx \begin{bmatrix} 0.707 & 0 & 0.354 \\ 0 & 0.707 & 0.354 \\ 0.707 & 0.707 & 0.707 \end{bmatrix}
$$

### 7.4 Step 3: Softmax (逐行)

对第一行 $[0.707, 0, 0.354]$:

$$
e^{0.707} \approx 2.028, \quad e^{0} = 1.000, \quad e^{0.354} \approx 1.425
$$

$$
\text{sum} = 2.028 + 1.000 + 1.425 = 4.453
$$

$$
\alpha_1 = [0.456, 0.225, 0.320]
$$

对第二行 $[0, 0.707, 0.354]$:

$$
\alpha_2 = [0.225, 0.456, 0.320]
$$

对第三行 $[0.707, 0.707, 0.707]$:

$$
e^{0.707} \approx 2.028 \text{ (三个相同) }, \quad \text{sum} = 6.084
$$

$$
\alpha_3 = [0.333, 0.333, 0.333]
$$

(第三个 token 的 query 与所有 key 同样相似, 所以均匀关注所有位置.)

### 7.5 Step 4: 加权求和

$$
\text{Output} = \alpha \cdot V
$$

第一行: $0.456 \times [10, 0] + 0.225 \times [0, 10] + 0.320 \times [5, 5] = [6.16, 3.85]$

第二行: $0.225 \times [10, 0] + 0.456 \times [0, 10] + 0.320 \times [5, 5] = [3.85, 6.16]$

第三行: $0.333 \times [10, 0] + 0.333 \times [0, 10] + 0.333 \times [5, 5] = [5.00, 5.00]$

### 7.6 解读

- **token 0** (query = $[1, 0]$): 主要关注 key 0 ($[1, 0]$, 方向一致), 输出偏向 value 0 ($[10, 0]$).
- **token 1** (query = $[0, 1]$): 主要关注 key 1 ($[0, 1]$), 输出偏向 value 1 ($[0, 10]$).
- **token 2** (query = $[1, 1]$): 与所有 key 等距, 均匀关注, 输出是所有 value 的平均.

---

## 8. 因果掩码 (Causal Mask)

### 8.1 为什么需要?

在**自回归语言模型** (如 GPT, Qwen 的文本解码器) 中, 模型按从左到右的顺序逐词生成. 在预测第 $t$ 个 token 时, 它只能使用位置 $\leq t$ 的信息 — — **不能偷看未来**.

### 8.2 实现方式

构造一个上三角矩阵, 对角线以上的位置填充 $-\infty$:

$$
M = \begin{bmatrix}
0 & -\infty & -\infty & -\infty \\
0 & 0 & -\infty & -\infty \\
0 & 0 & 0 & -\infty \\
0 & 0 & 0 & 0
\end{bmatrix}
$$

这个掩码在 softmax **之前**加到 score 矩阵上:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}} + M\right) V
$$

### 8.3 为什么有效?

因为 $e^{-\infty} = 0$. 被掩码遮住的位置在 softmax 之后的权重恰好为零, 从而在加权求和中完全不贡献信息.

### 8.4 手算示例: 3 token 加因果掩码

沿用第 7 节的例子, 掩码矩阵 (3×3):

$$
M = \begin{bmatrix}
0 & -\infty & -\infty \\
0 & 0 & -\infty \\
0 & 0 & 0
\end{bmatrix}
$$

缩放后的 score 加掩码:

$$
\frac{QK^T}{\sqrt{2}} + M \approx \begin{bmatrix}
0.707 & -\infty & -\infty \\
0 & 0.707 & -\infty \\
0.707 & 0.707 & 0.707
\end{bmatrix}
$$

Softmax 逐行:

- 第一行 $[0.707, -\infty, -\infty]$: $\alpha_1 = [1.0, 0.0, 0.0]$ (只能看到自己)
- 第二行 $[0, 0.707, -\infty]$: $e^{0} = 1.0$, $e^{0.707} \approx 2.028$, sum $= 3.028$, $\alpha_2 = [0.330, 0.670, 0.0]$
- 第三行 $[0.707, 0.707, 0.707]$: 同前, $\alpha_3 = [0.333, 0.333, 0.333]$ (因为第三行没有被遮挡)

加权输出:

- 第一行: $1.0 \times [10, 0] = [10.0, 0.0]$
- 第二行: $0.330 \times [10, 0] + 0.670 \times [0, 10] = [3.30, 6.70]$
- 第三行: 同前 $= [5.00, 5.00]$

注意 token 0 的输出变成了完全等于 value 0 — — 因为它只能看到自己.

---

## 9. Multi-Head Attention (MHA)

### 9.1 单头的局限

一个 attention head 只能学习**一种**注意力模式. 但语言中的关系是多元的:

- 语法关系: 主语→谓语
- 指代关系: 代词→先行词 ("它"→"小猫")
- 位置关系: 相邻词的局部依赖

单头 attention 就像只有一个人在做决策 — — 能力有限.

### 9.2 多头公式

**Multi-Head Attention** 的做法是设置 $h$ 个独立的 head, 每个 head 在自己的低维子空间中做 attention:

$$
\text{head}_i = \text{Attention}(X W_i^Q,\; X W_i^K,\; X W_i^V)
$$

$$
\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \; W^O
$$

其中:

- $W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$
- $d_k = d_v = d_{\text{model}} / h$ (通常)
- $W^O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$

### 9.3 实际实现中的 reshape 技巧

在工程实践中, 我们不会真的创建 $h$ 个小矩阵. 而是用一个大矩阵投影, 然后 reshape 拆分成多个 head:

**Step 1**: 投影. 用一个大的 $W^Q \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$ 做投影:

$$
Q_{\text{full}} = X W^Q \quad \in \mathbb{R}^{n \times d_{\text{model}}}
$$

**Step 2**: Reshape. 把 $d_{\text{model}}$ 拆成 $h \times d_k$:

$$
Q_{\text{full}} \in \mathbb{R}^{n \times d_{\text{model}}} \;\to\; \mathbb{R}^{n \times h \times d_k}
$$

**Step 3**: Transpose. 把 head 维度提前, 方便并行计算:

$$
\mathbb{R}^{n \times h \times d_k} \;\to\; \mathbb{R}^{h \times n \times d_k}
$$

(加上 batch 维度就是 $(B, h, n, d_k)$.)

**Step 4**: 每个 head 独立做 SDPA.

**Step 5**: Transpose 回来, reshape 合并.

$$
\mathbb{R}^{h \times n \times d_k} \;\to\; \mathbb{R}^{n \times h \times d_k} \;\to\; \mathbb{R}^{n \times d_{\text{model}}}
$$

**Step 6**: 输出投影 $W^O$, 把多头的信息融合.

### 9.4 为什么需要 $W^O$?

$W^O$ 的作用是把 $h$ 个 head 各自提取的信息**混合**回统一的 $d_{\text{model}}$ 维空间. 没有它, 各 head 的输出只是简单拼接, 无法交互.

---

## 10. 为什么拆分成多头有效?

### 10.1 参数量不变

假设 $d_{\text{model}} = 512$, $h = 8$, 则 $d_k = 64$.

- **单头大矩阵**: $W^Q$ 为 $512 \times 512 = 262{,}144$ 个参数.
- **8 个小矩阵**: 每个 $W_i^Q$ 为 $512 \times 64 = 32{,}768$, 共 $8 \times 32{,}768 = 262{,}144$ 个参数.

参数总量相同! 但多头的好处是: 每个 head 在 64 维子空间中学习**不同的注意力模式**.

### 10.2 "专家委员会"类比

把多头 attention 想象成一个由 $h$ 位专家组成的委员会:

- 专家 1 可能负责"语法结构": 主语在哪? 动词在哪?
- 专家 2 可能负责"指代消解": 这个代词指的是谁?
- 专家 3 可能负责"局部上下文": 前后相邻的词是什么?

每位专家各看各的, 最后由 $W^O$ 综合所有意见.

### 10.3 实证观察

研究者可视化 Transformer 的注意力权重后发现, 不同 head 确实学到了截然不同的模式: 有些 head 关注相邻位置, 有些关注句法依赖, 有些关注特定距离的位置. 这验证了多头设计的有效性.

---

## 11. Grouped Query Attention (GQA)

### 11.1 动机: KV Cache 的内存压力

在自回归推理 (inference) 阶段, 模型逐词生成. 每生成一个新 token, 之前所有 token 的 $K$ 和 $V$ 向量需要被缓存以避免重复计算 — — 这就是 **KV Cache**.

对于标准 MHA, KV Cache 的大小为:

$$
\text{KV Cache} = 2 \times n_{\text{layers}} \times n_{\text{heads}} \times \text{seq\_len} \times d_k \times \text{sizeof(dtype)}
$$

其中因子 2 是因为 K 和 V 各需要一份.

以 Qwen2-VL 文本解码器为例 (假设 MHA):

$$
2 \times 28 \times 12 \times \text{seq\_len} \times 128 \times 2 \;\text{bytes (fp16)}
$$

当 seq_len = 4096 时:

$$
2 \times 28 \times 12 \times 4096 \times 128 \times 2 = 704{,}643{,}072 \;\text{bytes} \approx 672 \;\text{MB}
$$

**每增加一层, 一个 head, 或序列长度翻倍, 显存就翻倍**. 这在长序列推理中是巨大的负担.

### 11.2 GQA 的思路

GQA 的核心思想很简单: **让多个 Q head 共享同一组 K, V head**.

在 Qwen2-VL 文本解码器中:

- $n_{\text{q\_heads}} = 12$
- $n_{\text{kv\_heads}} = 2$
- 每组 Q head 数量 $= 12 / 2 = 6$

也就是说, Q head 0, 1, 2, 3, 4, 5 共享 KV head 0; Q head 6, 7, 8, 9, 10, 11 共享 KV head 1.

### 11.3 `repeat_kv` 操作

为了让共享的 KV head 与所有 Q head 对齐, 需要把 KV head 复制 (repeat):

$$
(B, \;n_{\text{kv\_heads}}, \;S, \;D) \;\xrightarrow{\text{repeat\_kv}}\; (B, \;n_{\text{kv\_heads}} \times n_{\text{rep}}, \;S, \;D)
$$

其中 $n_{\text{rep}} = n_{\text{q\_heads}} / n_{\text{kv\_heads}} = 6$.

实现步骤:

1. 在 head 维度之后插入一个新轴: $(B, 2, 1, S, 128)$
2. 利用 broadcasting 扩展到: $(B, 2, 6, S, 128)$
3. Reshape 合并: $(B, 12, S, 128)$

### 11.4 内存节省

KV Cache 大小从 $n_{\text{q\_heads}}$ 降到 $n_{\text{kv\_heads}}$:

$$
\text{节省比例} = \frac{n_{\text{q\_heads}}}{n_{\text{kv\_heads}}} = \frac{12}{2} = 6\times
$$

---

## 12. MHA vs GQA vs MQA 对比

| 方案    | Q head 数 | KV head 数        | 每组 Q head | 特点                   |
| ------- | --------- | ----------------- | ----------- | ---------------------- |
| **MHA** | $H$       | $H$               | 1           | 最大表达力, 最大显存   |
| **GQA** | $H$       | $G$ ($1 < G < H$) | $H/G$       | 性能与显存的平衡       |
| **MQA** | $H$       | 1                 | $H$         | 最小显存, 可能损失质量 |

### Qwen2-VL 文本解码器的 KV Cache 大小 (seq_len = 4096, fp16)

公式: $\text{KV Cache} = 2 \times 28 \times n_{\text{kv\_heads}} \times 4096 \times 128 \times 2$ bytes.

| 方案                | $n_{\text{kv\_heads}}$ | KV Cache (MB) | 相对 MHA     |
| ------------------- | ---------------------- | ------------- | ------------ |
| MHA                 | 12                     | $\approx 672$ | $1\times$    |
| **GQA (Qwen2-VL) ** | **2**                  | $\approx 112$ | $1/6\times$  |
| MQA                 | 1                      | $\approx 56$  | $1/12\times$ |

GQA 用 2 个 KV head 就实现了 6 倍的内存节省, 同时实验表明质量损失极小.

---

## 13. 完整数值示例 — — 4 token, 2 head, head_dim = 3

为了看清 Multi-Head Attention + GQA 的完整流程, 我们用一个足够小的例子手算每一步.

### 13.1 设定

- 序列长度 $n = 4$, $d_{\text{model}} = 6$
- $h = 2$ 个 Q head, $d_k = 3$ ($6 / 2 = 3$)
- GQA 设定: $n_{\text{kv\_heads}} = 1$ (两个 Q head 共享一套 KV)

输入 (4 个 token, 每个 6 维):

$$
X = \begin{bmatrix}
1 & 0 & 1 & 0 & 1 & 0 \\
0 & 1 & 0 & 1 & 0 & 1 \\
1 & 1 & 0 & 0 & 1 & 1 \\
0 & 0 & 1 & 1 & 0 & 0
\end{bmatrix}
$$

### 13.2 投影矩阵

为简化, 定义 $W^Q \in \mathbb{R}^{6 \times 6}$ 和 $W^K, W^V \in \mathbb{R}^{6 \times 3}$ (因为只有 1 个 KV head):

$$
W^Q = I_6 \quad (\text{单位矩阵, 简化演示})
$$

$$
W^K = \begin{bmatrix}
1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \\ 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1
\end{bmatrix}, \quad
W^V = \begin{bmatrix}
0 & 1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \\ 0 & 1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1
\end{bmatrix}
$$

### 13.3 计算 Q, K, V

$$
Q = X W^Q = X = \begin{bmatrix}
1 & 0 & 1 & 0 & 1 & 0 \\
0 & 1 & 0 & 1 & 0 & 1 \\
1 & 1 & 0 & 0 & 1 & 1 \\
0 & 0 & 1 & 1 & 0 & 0
\end{bmatrix}
$$

$$
K = X W^K = \begin{bmatrix}
1 & 1 & 1 \\
1 & 1 & 1 \\
1 & 1 & 1 \\
1 & 1 & 1
\end{bmatrix}
\quad (\text{巧合: 每行恰好都是 } [1,1,1])
$$

(验算第一行: $1 \cdot 1 + 0 \cdot 0 + 1 \cdot 0 + 0 \cdot 1 + 1 \cdot 0 + 0 \cdot 0 = 1$; $1 \cdot 0 + 0 \cdot 1 + 1 \cdot 0 + 0 \cdot 0 + 1 \cdot 1 + 0 \cdot 0 = 1$; $1 \cdot 0 + 0 \cdot 0 + 1 \cdot 1 + 0 \cdot 0 + 1 \cdot 0 + 0 \cdot 1 = 1$. 正确.)

$$
V = X W^V = \begin{bmatrix}
1 & 1 & 1 \\
1 & 1 & 1 \\
2 & 1 & 1 \\
0 & 1 & 1
\end{bmatrix}
$$

(验算第三行: $1 \cdot 0 + 1 \cdot 1 + 0 \cdot 0 + 0 \cdot 0 + 1 \cdot 1 + 1 \cdot 0 = 2$; $1 \cdot 1 + 1 \cdot 0 + 0 \cdot 0 + 0 \cdot 1 + 1 \cdot 0 + 1 \cdot 0 = 1$; $1 \cdot 0 + 1 \cdot 0 + 0 \cdot 1 + 0 \cdot 0 + 1 \cdot 0 + 1 \cdot 1 = 1$. 正确.)

### 13.4 Reshape Q 为 2 个 head

$Q$ 的形状 $(4, 6)$ → reshape 为 $(4, 2, 3)$ → transpose 为 $(2, 4, 3)$:

$$
Q_{\text{head 0}} = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 0 \\ 1 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}, \quad
Q_{\text{head 1}} = \begin{bmatrix} 0 & 1 & 0 \\ 1 & 0 & 1 \\ 0 & 1 & 1 \\ 1 & 0 & 0 \end{bmatrix}
$$

### 13.5 repeat_kv: 1 个 KV head → 2 份

$K$ 和 $V$ 的形状 $(4, 3)$ → 视为 $(1, 1, 4, 3)$ → repeat → $(1, 2, 4, 3)$. 两个 Q head 看到的 $K$ 和 $V$ 完全相同.

### 13.6 Head 0 的 SDPA (带因果掩码)

$$
Q_0 K^T = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 0 \\ 1 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}
\begin{bmatrix} 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 \end{bmatrix}
= \begin{bmatrix} 2 & 2 & 2 & 2 \\ 1 & 1 & 1 & 1 \\ 2 & 2 & 2 & 2 \\ 1 & 1 & 1 & 1 \end{bmatrix}
$$

缩放: 除以 $\sqrt{3} \approx 1.732$:

$$
\text{Scaled} = \begin{bmatrix} 1.155 & 1.155 & 1.155 & 1.155 \\ 0.577 & 0.577 & 0.577 & 0.577 \\ 1.155 & 1.155 & 1.155 & 1.155 \\ 0.577 & 0.577 & 0.577 & 0.577 \end{bmatrix}
$$

加因果掩码后:

$$
\begin{bmatrix}
1.155 & -\infty & -\infty & -\infty \\
0.577 & 0.577 & -\infty & -\infty \\
1.155 & 1.155 & 1.155 & -\infty \\
0.577 & 0.577 & 0.577 & 0.577
\end{bmatrix}
$$

Softmax 逐行:

- 第 0 行: $[1.0, 0, 0, 0]$
- 第 1 行: $[0.5, 0.5, 0, 0]$
- 第 2 行: $[0.333, 0.333, 0.333, 0]$
- 第 3 行: $[0.25, 0.25, 0.25, 0.25]$

乘以 $V$:

$$
\text{Out}_0 = \begin{bmatrix}
1.0 & 1.0 & 1.0 \\
1.0 & 1.0 & 1.0 \\
1.333 & 1.0 & 1.0 \\
1.0 & 1.0 & 1.0
\end{bmatrix}
$$

(第 2 行: $0.333 \times 1 + 0.333 \times 1 + 0.333 \times 2 = 1.333$; 其余维度 $= 1.0$.)

### 13.7 Head 1 的 SDPA

$Q_1 K^T$ 同理, 每行的点积值等于 $Q_1$ 各行元素之和:

- 行 0: $0 + 1 + 0 = 1 \to$ 缩放后 $0.577$
- 行 1: $1 + 0 + 1 = 2 \to$ 缩放后 $1.155$
- 行 2: $0 + 1 + 1 = 2 \to$ 缩放后 $1.155$
- 行 3: $1 + 0 + 0 = 1 \to$ 缩放后 $0.577$

加因果掩码 + softmax 后, 注意力权重与 Head 0 完全相同 (因为 $K$ 的每行都是 $[1,1,1]$, 使得 score 矩阵每行内所有未遮挡元素相等). 因此 $\text{Out}_1 = \text{Out}_0$.

### 13.8 Concat + Output Projection

$$
\text{Concat}(\text{Out}_0, \text{Out}_1) \in \mathbb{R}^{4 \times 6}
$$

最后乘以 $W^O \in \mathbb{R}^{6 \times 6}$ 得到最终输出 (此处省略具体 $W^O$ 的数值, 关键是理解流程).

---

## 14. Qwen2-VL 中的视觉注意力

Qwen2-VL 的视觉编码器 (ViT 部分) 使用**标准 Multi-Head Attention (MHA) **, 没有 GQA.

### 14.1 核心参数

| 参数                           | 值   |
| ------------------------------ | ---- |
| embed*dim ($d*{\text{model}}$) | 1280 |
| num_heads ($h$)                | 16   |
| head_dim ($d_k$)               | 80   |
| MLP 中间维度                   | 5120 |
| Block 数量                     | 32   |

### 14.2 融合 QKV 投影

视觉编码器把 Q, K, V 的投影合并成一个大矩阵:

$$
W_{\text{qkv}} \in \mathbb{R}^{3840 \times 1280}, \quad b_{\text{qkv}} \in \mathbb{R}^{3840}
$$

其中 $3840 = 3 \times 1280$ (Q, K, V 各占 1280 维).

计算流程:

$$
\text{QKV} = X W_{\text{qkv}}^T + b_{\text{qkv}} \quad \in \mathbb{R}^{n \times 3840}
$$

然后沿最后一维拆分:

$$
Q, K, V = \text{split}(\text{QKV}, 3) \quad \text{每个} \in \mathbb{R}^{n \times 1280}
$$

### 14.3 完整数据流 (含形状变化)

```
输入 x: (n_patches, 1280)
        ↓ 融合 QKV 线性投影
QKV:    (n_patches, 3840)
        ↓ split
Q, K, V: 各 (n_patches, 1280)
        ↓ reshape
Q, K, V: 各 (n_patches, 16, 80)
        ↓ transpose
Q, K, V: 各 (16, n_patches, 80)
        ↓ 2D RoPE 应用于 Q 和 K
Q', K': (16, n_patches, 80)
        ↓ Windowed SDPA（无因果掩码）
Attn:   (16, n_patches, 80)
        ↓ transpose + reshape
Out:    (n_patches, 1280)
        ↓ 输出投影 W_proj
Final:  (n_patches, 1280)
```

### 14.4 关键差异

- **无因果掩码**: 图像 patch 之间不存在"先后顺序", 每个 patch 可以看到所有其他 patch.
- **Windowed attention**: 为了处理大量 patch (如 14308 个), 采用窗口注意力 — — 将 patch 分成若干窗口, 每个窗口内部做 attention, 减少计算量从 $O(n^2)$ 到 $O(n \cdot w)$.
- **2D RoPE**: 位置编码是二维的 (行和列), 与文本的一维位置编码不同.

---

## 15. Qwen2-VL 中的文本注意力

文本解码器使用 **Grouped Query Attention (GQA) **.

### 15.1 核心参数

| 参数                             | 值           |
| -------------------------------- | ------------ |
| hidden*size ($d*{\text{model}}$) | 1536         |
| num_heads ($h_Q$)                | 12           |
| num*kv_heads ($h*{KV}$)          | 2            |
| head_dim ($d_k$)                 | 128          |
| MLP 中间维度                     | 8960         |
| 层数                             | 28           |
| 词汇表大小                       | 151936       |
| RoPE base                        | 1,000,000    |
| mrope_section                    | [16, 24, 24] |

### 15.2 投影矩阵

| 投影 | 权重形状       | 偏置 | 输出维度               |
| ---- | -------------- | ---- | ---------------------- |
| Q    | $(1536, 1536)$ | 有   | $12 \times 128 = 1536$ |
| K    | $(256, 1536)$  | 有   | $2 \times 128 = 256$   |
| V    | $(256, 1536)$  | 有   | $2 \times 128 = 256$   |
| O    | $(1536, 1536)$ | 无   | $1536$                 |

注意 K 和 V 的投影矩阵只有 $(256, 1536)$ — — 因为只有 2 个 KV head.

### 15.3 完整数据流

```
输入 x: (B, S, 1536)
        ↓ Q 投影
Q:      (B, S, 1536) → reshape → (B, S, 12, 128) → transpose → (B, 12, S, 128)
        ↓ K 投影
K:      (B, S, 256)  → reshape → (B, S, 2, 128)  → transpose → (B, 2, S, 128)
        ↓ V 投影
V:      (B, S, 256)  → reshape → (B, S, 2, 128)  → transpose → (B, 2, S, 128)
        ↓ M-RoPE 应用于 Q 和 K
Q', K': (B, 12, S, 128), (B, 2, S, 128)
        ↓ repeat_kv (n_rep=6)
K_exp:  (B, 12, S, 128)
V_exp:  (B, 12, S, 128)
        ↓ SDPA（带因果掩码）
Attn:   (B, 12, S, 128)
        ↓ transpose + reshape
Out:    (B, S, 1536)
        ↓ O 投影（无 bias）
Final:  (B, S, 1536)
```

### 15.4 M-RoPE (Multimodal RoPE)

Qwen2-VL 使用三维的旋转位置编码 — — **M-RoPE** (Multimodal Rotary Position Embedding):

- 对于文本 token: 使用一维位置 $[t, t, t]$ (时间维度)
- 对于图像 token: 使用三维位置 $[t, h, w]$ (时间, 高度, 宽度)

`mrope_section = [16, 24, 24]` 意味着 head_dim = 128 维被分成三段:

- 前 32 维 ($16 \times 2$): 编码时间位置
- 中间 48 维 ($24 \times 2$): 编码高度位置
- 后 48 维 ($24 \times 2$): 编码宽度位置

这使得模型能在同一个注意力机制中处理文本和视觉的位置信息.

---

## 16. NumPy 实现详解

下面是 `impl.py` 中核心函数的代码与逐行解释.

### 16.1 `softmax` 函数

```python
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """数值稳定的 softmax: 先减最大值防止 exp 溢出。"""
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)
```

| 行                                            | 说明                                                                                   |
| --------------------------------------------- | -------------------------------------------------------------------------------------- |
| `x_max = np.max(x, axis=axis, keepdims=True)` | 沿指定轴取最大值, `keepdims=True` 保持维度以便广播. 这是第 6.3 节数值稳定性技巧的实现. |
| `e_x = np.exp(x - x_max)`                     | 先减去最大值再取 exp, 保证所有指数 $\leq 0$, 避免溢出. 最大值对应的 exp 恰好为 1.      |
| `return e_x / np.sum(e_x, ...)`               | 归一化, 使输出和为 1.                                                                  |

### 16.2 `scaled_dot_product_attention` 函数

```python
def scaled_dot_product_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    d_k = q.shape[-1]
    scores = q @ k.swapaxes(-2, -1) / np.sqrt(d_k)
    if mask is not None:
        scores = scores + mask
    weights = softmax(scores, axis=-1)
    return weights @ v
```

| 行                                   | 说明                                                                                                                                  |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| `d_k = q.shape[-1]`                  | 取 query 的最后一维作为 $d_k$, 用于缩放因子.                                                                                          |
| `q @ k.swapaxes(-2, -1)`             | 计算 $QK^T$. `swapaxes(-2, -1)` 交换最后两维 — — 等价于转置最后两维, 得到 $(\ldots, \text{seq\_q}, \text{seq\_k})$ 形状的 score 矩阵. |
| `/ np.sqrt(d_k)`                     | 除以 $\sqrt{d_k}$, 对应第 5 节的推导.                                                                                                 |
| `scores = scores + mask`             | 加法掩码. 被遮挡的位置为 $-\infty$, 加上后在 softmax 中变为 0 权重.                                                                   |
| `weights = softmax(scores, axis=-1)` | 对每个 query (每一行) 做 softmax, 得到注意力权重.                                                                                     |
| `return weights @ v`                 | 加权求和: $\alpha V$, 输出形状为 $(\ldots, \text{seq\_q}, d_v)$.                                                                      |

### 16.3 `repeat_kv` 函数

```python
def repeat_kv(x: np.ndarray, n_rep: int) -> np.ndarray:
    if n_rep == 1:
        return x
    B, n_kv_heads, S, D = x.shape
    x = x[:, :, np.newaxis, :, :]          # (B, n_kv_heads, 1, S, D)
    x = np.broadcast_to(x, (B, n_kv_heads, n_rep, S, D))
    return x.reshape(B, n_kv_heads * n_rep, S, D)
```

| 行                              | 说明                                                                                                         |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `if n_rep == 1: return x`       | 如果重复次数为 1 (MHA 场景), 直接返回.                                                                       |
| `B, n_kv_heads, S, D = x.shape` | 拆解形状: batch, KV head 数, 序列长度, head 维度.                                                            |
| `x[:, :, np.newaxis, :, :]`     | 在 head 维度后插入新轴, 形状变为 $(B, n_{\text{kv}}, 1, S, D)$.                                              |
| `np.broadcast_to(...)`          | 利用广播机制将新轴扩展 $n_{\text{rep}}$ 次, 形状变为 $(B, n_{\text{kv}}, n_{\text{rep}}, S, D)$. 不复制内存. |
| `.reshape(...)`                 | 合并 head 维度: $n_{\text{kv}} \times n_{\text{rep}}$ → 总 Q head 数.                                        |

**具体到 Qwen2-VL**: $(B, 2, S, 128) \to (B, 2, 6, S, 128) \to (B, 12, S, 128)$.

---

## 17. KV Cache 内存分析

以 Qwen2-VL 文本解码器为例, 在 fp16 (每个参数 2 字节) 下, sequence length $= 4096$.

### 17.1 通用公式

$$
\text{KV Cache (bytes)} = 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times \text{seq\_len} \times d_k \times \text{sizeof(dtype)}
$$

### 17.2 MHA ($n_{\text{kv\_heads}} = 12$)

$$
2 \times 28 \times 12 \times 4096 \times 128 \times 2 = 2 \times 28 \times 12 \times 4096 \times 128 \times 2
$$

分步:

- $28 \times 12 = 336$
- $336 \times 4096 = 1{,}376{,}256$
- $1{,}376{,}256 \times 128 = 176{,}160{,}768$
- $176{,}160{,}768 \times 2 = 352{,}321{,}536$ (K 的部分)
- $\times 2 = 704{,}643{,}072$ bytes $\approx 672$ MB

### 17.3 GQA ($n_{\text{kv\_heads}} = 2$, Qwen2-VL 实际配置)

$$
2 \times 28 \times 2 \times 4096 \times 128 \times 2
$$

- $28 \times 2 = 56$
- $56 \times 4096 = 229{,}376$
- $229{,}376 \times 128 = 29{,}360{,}128$
- $29{,}360{,}128 \times 2 = 58{,}720{,}256$ (K 的部分)
- $\times 2 = 117{,}440{,}512$ bytes $\approx 112$ MB

### 17.4 MQA ($n_{\text{kv\_heads}} = 1$)

$$
2 \times 28 \times 1 \times 4096 \times 128 \times 2
$$

- $28 \times 1 = 28$
- $28 \times 4096 = 114{,}688$
- $114{,}688 \times 128 = 14{,}680{,}064$
- $14{,}680{,}064 \times 2 = 29{,}360{,}128$ (K 的部分)
- $\times 2 = 58{,}720{,}256$ bytes $\approx 56$ MB

### 17.5 汇总

| 方案    | $n_{\text{kv\_heads}}$ | KV Cache   | 相对 MHA        |
| ------- | ---------------------- | ---------- | --------------- |
| MHA     | 12                     | 672 MB     | $1\times$       |
| **GQA** | **2**                  | **112 MB** | **$1/6\times$** |
| MQA     | 1                      | 56 MB      | $1/12\times$    |

**结论**: GQA 选择 2 个 KV head, 在内存开销仅为 MHA 的 1/6 的条件下, 实现了接近 MHA 的模型质量. 这是一个非常实用的工程权衡.

---

## 18. 常见误解与陷阱

### 误解 1: 注意力权重 = "重要性"

**错误**: 认为注意力权重大的 token "更重要".

**正确**: 注意力权重反映的是**与当前 query 的相关性**, 而非绝对重要性. 同一个 token 在不同 query 眼中的权重是不同的.

### 误解 2: Softmax 使注意力成为"概率模型"

**错误**: 因为 softmax 的输出像概率分布, 就认为 attention 在做概率推断.

**正确**: Softmax 只是一种归一化手段, 让权重非负且和为 1. 这不等于在做贝叶斯推断或概率建模.

### 误解 3: 缩放因子可以省略

**错误**: $1/\sqrt{d_k}$ 只是锦上添花.

**正确**: 当 $d_k$ 较大时 (如 128), 不缩放会导致 softmax 严重饱和, 梯度消失, 模型几乎无法训练. 这是一个**必需的**数值稳定措施.

### 误解 4: GQA 意味着 KV head "质量更低"

**错误**: KV head 变少了, 所以每个 head 承载的信息变少了.

**正确**: GQA 中 KV head 被**共享**, 不是被"缩减". 每个 KV head 仍然拥有完整的 $d_k$ 维表达能力, 只是多个 Q head 复用它.

### 误解 5: 因果掩码是在 softmax 之后应用的

**错误**: 先做 softmax 再屏蔽.

**正确**: 因果掩码必须在 softmax **之前**加上 (以 $-\infty$ 的形式). 如果在 softmax 之后把某些位置置零, 剩余权重的和就不再为 1, 注意力分布被破坏.

---

## 19. 与相关概念的对比

### 19.1 Attention vs Convolution

| 特性       | Attention                     | Convolution                   |
| ---------- | ----------------------------- | ----------------------------- |
| 感受野     | **全局** (每个位置看所有位置) | **局部** (固定大小的滑动窗口) |
| 参数共享   | 权重矩阵在所有位置共享        | 卷积核在所有位置共享          |
| 位置感知   | 需要额外的位置编码            | 天然感知局部位置              |
| 计算复杂度 | $O(n^2 d)$                    | $O(n k d)$ ($k$ 为核大小)     |

Attention 的优势在于**长距离依赖**的建模能力 — — 不管两个 token 距离多远, 都可以直接交互. Convolution 必须通过多层堆叠才能传播信息.

### 19.2 Attention vs RNN Hidden State

| 特性       | Attention                     | RNN                            |
| ---------- | ----------------------------- | ------------------------------ |
| 并行性     | **可并行** (所有位置同时计算) | **串行** (必须按顺序处理)      |
| 记忆方式   | **显式** (直接计算权重矩阵)   | **隐式** (压缩到 hidden state) |
| 长距离记忆 | 路径长度 $O(1)$               | 路径长度 $O(n)$, 容易遗忘      |

### 19.3 Self-Attention vs Cross-Attention

| 特性      | Self-Attention     | Cross-Attention           |
| --------- | ------------------ | ------------------------- |
| Q 来源    | 当前序列 $X$       | 当前序列 $X$              |
| K, V 来源 | **同一个序列 $X$** | **另一个序列 $Y$**        |
| 用途      | 建模序列内部关系   | 建模两个序列之间的关系    |
| 例子      | Encoder 自注意力   | Decoder 看 Encoder 的输出 |

在 Qwen2-VL 中, 视觉编码器内部使用 **self-attention** (patch 之间互相看), 而文本解码器在处理图像信息时, 通过 cross-attention 或融合 token 的方式引入视觉特征.

---

## 20. 总结

### 20.1 核心公式速查表

| 公式           | 表达式                                                                                  |
| -------------- | --------------------------------------------------------------------------------------- |
| Q, K, V 投影   | $Q = XW^Q,\; K = XW^K,\; V = XW^V$                                                      |
| 缩放点积注意力 | $\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}} + M\right) V$  |
| 缩放因子推导   | $\text{Var}[\mathbf{q} \cdot \mathbf{k}] = d_k \;\Rightarrow\; \text{除以 } \sqrt{d_k}$ |
| Softmax        | $\text{softmax}(z_i) = e^{z_i} / \sum_j e^{z_j}$                                        |
| 多头注意力     | $\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$         |
| repeat_kv      | $(B, G, S, D) \to (B, G \cdot r, S, D)$ 其中 $r = H / G$                                |
| KV Cache 大小  | $2 \times L \times G \times S \times d_k \times \text{dtype\_size}$                     |

### 20.2 Qwen2-VL 参数速查

| 模块 | 参数          | 值           |
| ---- | ------------- | ------------ |
| 视觉 | embed_dim     | 1280         |
| 视觉 | num_heads     | 16           |
| 视觉 | head_dim      | 80           |
| 视觉 | MLP 中间维度  | 5120         |
| 视觉 | Block 数      | 32           |
| 文本 | hidden_size   | 1536         |
| 文本 | num_heads (Q) | 12           |
| 文本 | num_kv_heads  | 2            |
| 文本 | head_dim      | 128          |
| 文本 | MLP 中间维度  | 8960         |
| 文本 | 层数          | 28           |
| 文本 | 词汇表        | 151936       |
| 文本 | RoPE base     | 1,000,000    |
| 文本 | mrope_section | [16, 24, 24] |

---

> **后记**: 注意力机制的发展史是深度学习中最令人振奋的篇章之一. 从 2014 年 Bahdanau 的"回头看"到 2017 年 Transformer 的"完全抛弃循环", 再到今天 GQA 在工程上的精妙权衡 — — 每一步都是在回答同一个问题: 机器如何更好地"关注"信息? 希望这篇文章能帮助你建立起对这一核心概念的完整直觉.
