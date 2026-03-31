# Decoder Layer (解码器层)

> **一句话总结**: Decoder Layer 是大语言模型的"思考单元" — — 每一层都在反复打磨模型对"下一个词应该是什么"的判断, 就像写文章时一遍遍修改草稿, 直到最终定稿.

---

## 为什么需要 Decoder Layer?

想象你正在写一篇文章. 你的大脑不会一次性蹦出完美的句子, 而是经历一个渐进的过程:

1. **第一遍**: 脑海中浮现一个大致的想法 — — "我想说点关于天气的事"
2. **第二遍**: 想法变得更具体 — — "今天天气不错, 适合出去走走"
3. **第三遍**: 措辞被打磨 — — "午后阳光正好, 不妨去公园散步"
4. **第 N 遍**: 最终定稿

大语言模型 (LLM) 生成文本的过程惊人地相似. 每一个 Decoder Layer 就像大脑的一次"修改迭代" — — 它接收上一层传来的"草稿" (一个向量序列), 通过 **注意力机制** 理解上下文, 通过 **前馈网络** 融入知识, 然后输出一份更精炼的"草稿". 在 Qwen2-VL 中, 这样的层有 **28 层**, 意味着每个 token 的表示要被精心打磨 28 次, 才最终用于预测下一个词.

这就是 Decoder Layer 的核心使命: **逐层精炼表示, 使模型对"下一个词"的预测越来越准确. **

---

## 前置知识

在阅读本节之前, 请确保理解以下算子 (它们是 Decoder Layer 的零部件):

| 算子               | 核心公式                                                                          | 链接                                                          |
| ------------------ | --------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| RMS Normalization  | $\text{RMSNorm}(x) = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \cdot \gamma$   | [04_rms_norm](../04_rms_norm/README.md)                       |
| Gated MLP / SwiGLU | $(\text{SiLU}(xW_g^T) \odot xW_u^T) W_d^T$                                        | [12_gated_mlp](../12_gated_mlp/README.md)                     |
| 残差连接           | $y = x + F(x)$                                                                    | [13_residual_connection](../13_residual_connection/README.md) |
| 注意力机制         | $\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$ | [07_attention](../07_attention/README.md)                     |

如果你已经理解了这四个组件, 那么 Decoder Layer 只是把它们按照特定的顺序"组装"起来. 就像你分别学会了切菜, 炒菜, 调味, 摆盘, 现在要把它们串成一道完整的菜谱.

---

## Autoregressive 生成: 一次一个 token

在理解 Decoder Layer 之前, 我们需要先明白它服务的核心任务 — — **自回归文本生成** (autoregressive generation).

### 基本原理

大语言模型生成文本的方式非常简单: **每次只预测一个 token (词或子词), 然后把预测的结果追加到输入中, 再预测下一个 token**. 这个过程不断循环, 直到生成结束标记或达到最大长度.

用一个具体的中文例子来说明. 假设我们给模型一个 prompt: "今天天气", 模型的生成过程如下:

```
第 1 步：输入 = [今天, 天气]         → 模型预测下一个 token → "真"
第 2 步：输入 = [今天, 天气, 真]      → 模型预测下一个 token → "不"
第 3 步：输入 = [今天, 天气, 真, 不]   → 模型预测下一个 token → "错"
第 4 步：输入 = [今天, 天气, 真, 不, 错] → 模型预测下一个 token → "，"
第 5 步：输入 = [今天, 天气, 真, 不, 错, ，] → 模型预测下一个 token → "适合"
...
```

注意两个关键特征:

1. **输入序列不断增长**: 每预测一个 token, 序列长度 $T$ 就加 1
2. **每一步只需要预测最后一个位置的 token**: 虽然模型处理整个序列, 但我们只关心序列末尾位置的输出

### 用伪代码表达

```python
tokens = tokenize("今天天气")           # 初始 prompt
for step in range(max_length):
    logits = model(tokens)              # 所有 28 个 Decoder Layer 依次处理
    next_token = sample(logits[-1])     # 只取最后一个位置的预测
    tokens.append(next_token)           # 追加到输入序列
    if next_token == EOS:               # 遇到结束标记就停止
        break
```

这里的 `model(tokens)` 就是 token 依次通过 28 个 Decoder Layer 的过程. 每一个 Decoder Layer 都在精炼序列中每个位置的表示, 使得最终层的输出足够"聪明", 能够准确预测下一个词.

---

## Decoder-Only 架构: 为什么不需要 Encoder?

### 原始 Transformer 的 Encoder-Decoder 结构

2017 年 Vaswani 等人提出的原始 Transformer 包含两个部分:

- **Encoder**: 读入源序列 (如英文句子), 生成上下文表示
- **Decoder**: 基于 Encoder 的输出, 逐词生成目标序列 (如中文翻译)

这种结构天然适合"输入 → 输出"的任务 (如翻译, 摘要), 因为 Encoder 和 Decoder 各有明确的分工.

### Decoder-Only 的崛起

2018 年, OpenAI 发布了 **GPT-1** (Radford et al., "Improving Language Understanding by Generative Pre-Training"), 做了一个大胆的简化: **扔掉 Encoder, 只保留 Decoder**.

这个看似"偷懒"的决定, 背后有深刻的道理:

1. **语言建模的本质是预测下一个词**. 给定一段文本的前缀, 预测后面是什么 — — 这个任务本身不需要一个单独的"源序列"
2. **统一的框架**: 无论是问答, 翻译还是写作, 都可以统一为"给定 prompt, 续写下去"
3. **更简洁的架构**: 减少了 Encoder-Decoder 之间的 Cross-Attention 层, 整体结构更干净

随后的发展印证了这一选择:

| 年份 | 模型     | 架构         | 关键突破                         |
| ---- | -------- | ------------ | -------------------------------- |
| 2018 | GPT-1    | Decoder-Only | 证明了 Decoder-Only 可行         |
| 2019 | GPT-2    | Decoder-Only | 证明了 scaling 有效 (1.5B 参数)  |
| 2020 | GPT-3    | Decoder-Only | 证明了 few-shot 能力 (175B 参数) |
| 2023 | LLaMA    | Decoder-Only | 开源社区的基石                   |
| 2024 | Qwen2-VL | Decoder-Only | 多模态扩展                       |

如今, 几乎所有主流 LLM — — GPT-4, LLaMA, Qwen, Mistral — — 都采用 Decoder-Only 架构. 我们讨论的 Decoder Layer 正是这种架构的核心构建块.

### Decoder-Only 与 Encoder-Decoder 的关键区别

| 特征            | Encoder-Decoder                                | Decoder-Only                    |
| --------------- | ---------------------------------------------- | ------------------------------- |
| 注意力类型      | Encoder: 双向; Decoder: 因果 + Cross-Attention | 纯因果 (causal) Self-Attention  |
| 输入 / 输出     | 明确的源序列和目标序列                         | prompt 和续写合并在同一个序列中 |
| Cross-Attention | ✅ 有                                          | ❌ 无                           |
| 代表模型        | T5, BART                                       | GPT, LLaMA, Qwen                |

---

## Causal Masking (因果掩码): 不能偷看未来

### 为什么需要因果掩码?

回忆 autoregressive 生成的过程: 当模型在预测位置 $t$ 的下一个 token 时, 它**只应该看到位置 $1, 2, \ldots, t$ 的信息**, 而不能"偷看"位置 $t+1, t+2, \ldots$ 的内容 — — 因为那些内容在实际生成时还不存在.

但在训练阶段, 为了效率, 我们会把整个序列一次性输入模型 (而不是像推理时那样一个一个 token 地喂). 这就产生了一个矛盾: 模型一次看到了所有 token, 但我们希望位置 $t$ 的计算只能使用 $\leq t$ 的信息.

**因果掩码** (causal mask) 就是解决这个矛盾的工具.

### 三角掩码矩阵

对于一个长度为 $T=4$ 的序列, 因果掩码是一个下三角矩阵:

$$
M = \begin{pmatrix}
1 & 0 & 0 & 0 \\
1 & 1 & 0 & 0 \\
1 & 1 & 1 & 0 \\
1 & 1 & 1 & 1
\end{pmatrix}
$$

其中 $M_{ij} = 1$ 表示位置 $i$ **可以看到** 位置 $j$, $M_{ij} = 0$ 表示**不可以看到**.

看第 3 行 (位置 3): $[1, 1, 1, 0]$, 意思是位置 3 可以看到位置 1, 2, 3, 但看不到位置 4.

在实际计算中, 掩码的工作方式是:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T + M'}{\sqrt{d_k}}\right) V
$$

其中 $M'$ 是一个加性掩码: $M'_{ij} = 0$ 表示允许注意, $M'_{ij} = -\infty$ 表示禁止注意. softmax 后 $e^{-\infty} = 0$, 被掩码的位置注意力权重自动变为零.

### 直观理解

可以这样类比: 你在考试中写一篇作文, 每写一个字的时候, 你只能参考自己已经写下的内容 (前面的字), 不能提前翻到后面看答案. 因果掩码就是这个"不许翻页"的规则.

---

## KV Cache: 推理加速的关键

### 朴素推理的问题

在 autoregressive 生成中, 每预测一个新 token 都需要处理 **整个序列**. 如果不做任何优化, 生成第 $t$ 个 token 时, Self-Attention 需要计算:

$$
Q_t K_{1:t}^T \quad \text{ (query 与所有 key 的点积) }
$$

这意味着每一步都要重新计算所有位置的 $K$ 和 $V$, 即使位置 $1$ 到 $t-1$ 的 $K$, $V$ 在上一步已经算过了.

### KV Cache 的思路

**KV Cache** 的想法很简单: **把已经计算过的 $K$ 和 $V$ 缓存起来, 每一步只计算新 token 的 $K$, $V$, 然后追加到缓存中**.

```
步骤 1: 输入 token 1     → 计算 K₁, V₁ → 缓存 [K₁], [V₁]
步骤 2: 输入 token 2     → 计算 K₂, V₂ → 缓存 [K₁,K₂], [V₁,V₂]
步骤 3: 输入 token 3     → 计算 K₃, V₃ → 缓存 [K₁,K₂,K₃], [V₁,V₂,V₃]
...
步骤 t: 输入 token t     → 计算 Kₜ, Vₜ → 缓存 [K₁,...,Kₜ], [V₁,...,Vₜ]
```

每一步的注意力计算只需要用新 token 的 $Q_t$ 与缓存的全部 $K$, $V$ 做运算.

### 计算量对比

假设我们要生成 100 个 token (每个 token 的序列长度依次为 1, 2,..., 100):

**不用 KV Cache** (每步重新计算全部 Q, K, V):

Self-Attention 的核心操作是 $QK^T$, 复杂度与序列长度的平方成正比. 总计算量正比于:

$$
\sum_{t=1}^{100} t^2 = \frac{100 \times 101 \times 201}{6} = 338{,}350
$$

**使用 KV Cache** (每步只计算新 token 的 Q, 复用缓存的 K, V):

每一步只需要 $Q_t$ (1 个 token) 与 $K_{1:t}$ ($t$ 个 token) 做点积, 计算量正比于 $t$. 总计算量正比于:

$$
\sum_{t=1}^{100} t = \frac{100 \times 101}{2} = 5{,}050
$$

**加速比**: $338{,}350 / 5{,}050 \approx 67\times$. 这就是为什么 KV Cache 对推理效率至关重要.

> **注意**: KV Cache 只在推理 (生成) 时使用. 训练时整个序列是一次性并行处理的, 不需要 cache.

---

## 数学定义

理解了背景知识后, 让我们回到 Decoder Layer 本身. 给定输入 $x \in \mathbb{R}^{B \times T \times d}$, 一个 Decoder Layer 执行以下 6 步运算:

$$
\hat{x} = \text{RMSNorm}(x, \gamma_1)
$$

$$
a = \text{SelfAttn}(\hat{x})
$$

$$
h = x + a
$$

$$
\hat{h} = \text{RMSNorm}(h, \gamma_2)
$$

$$
m = \text{GatedMLP}(\hat{h})
$$

$$
y = h + m
$$

其中:

- $\gamma_1$ 是 `input_layernorm` 的可学习缩放权重, 形状 $(d,)$
- $\gamma_2$ 是 `post_attention_layernorm` 的可学习缩放权重, 形状 $(d,)$
- $\text{SelfAttn}$ 是带因果掩码和 RoPE 位置编码的 Grouped Query Attention
- $\text{GatedMLP}$ 是 SwiGLU 门控前馈网络
- $y$ 是层输出, 形状与输入 $x$ 完全相同 $(B, T, d)$

**六步可以分为两个对称的"三步曲"**:

| 阶段     | 步骤      | 操作                       | 目的           |
| -------- | --------- | -------------------------- | -------------- |
| 上半部分 | 1 → 2 → 3 | RMSNorm → Attention → 残差 | 融合上下文信息 |
| 下半部分 | 4 → 5 → 6 | RMSNorm → MLP → 残差       | 融入参数化知识 |

直觉上, Attention 负责让 token 之间"交流" ("这个'他'指的是前文的'小明'"), 而 MLP 负责引入知识性变换 ("'巴黎'是法国的首都").

---

## Pre-Norm with RMSNorm (而非 LayerNorm!)

### Pre-Norm vs Post-Norm

原始 Transformer (Vaswani et al. 2017) 使用 **Post-Norm**:

$$
y = \text{LayerNorm}(x + F(x)) \quad \text{(Post-Norm)}
$$

而现代 LLM 几乎都改用 **Pre-Norm**:

$$
y = x + F(\text{Norm}(x)) \quad \text{(Pre-Norm)}
$$

两者的核心区别在于: Pre-Norm 中, 残差路径上**没有**归一化操作, 梯度可以沿残差路径"无损"回传到任意早期层. 这使得训练深层网络 (28 层, 甚至上百层) 变得更加稳定.

| 方案         | 公式                        | 训练稳定性   | 现状                    |
| ------------ | --------------------------- | ------------ | ----------------------- |
| Post-Norm    | $y = \text{Norm}(x + F(x))$ | 需要 warm-up | 原始 Transformer        |
| **Pre-Norm** | $y = x + F(\text{Norm}(x))$ | **更稳定**   | **GPT, LLaMA, Qwen 等** |

### 为什么用 RMSNorm 而不是 LayerNorm?

LayerNorm 包含两个步骤: **中心化** (减去均值) 和**缩放** (除以标准差):

$$
\text{LayerNorm}(x) = \frac{x - \mu}{\sigma} \cdot \gamma + \beta
$$

RMSNorm (Zhang & Sennrich, 2019, "Root Mean Square Layer Normalization") 去掉了中心化步骤, 只保留缩放:

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \cdot \gamma
$$

对比如下:

| 特征                  | LayerNorm           | RMSNorm          |
| --------------------- | ------------------- | ---------------- |
| 中心化 (减均值)       | ✅ 有               | ❌ 无            |
| 缩放 (除以标准差/RMS) | ✅ 有               | ✅ 有            |
| 偏置参数 $\beta$      | ✅ 有               | ❌ 无            |
| 计算量                | 需要计算均值 + 方差 | 只需计算平方均值 |
| 效果                  | 基准                | 相似甚至更好     |

直觉上, 归一化最重要的作用是控制数值范围 (缩放), 中心化对最终效果的贡献很小. 去掉它可以减少计算量, 在大模型中积少成多.

---

## GQA: 用更少的 KV 头服务更多的 Query 头

### 从 MHA 到 GQA

标准的 Multi-Head Attention (MHA) 中, $Q$, $K$, $V$ 各有相同数量的头. Qwen2-VL 的 Vision Encoder 就使用 MHA: 16 个 query 头, 16 个 KV 头.

但在文本 Decoder 中, Qwen2-VL 使用 **Grouped Query Attention** (GQA, Ainslie et al. 2023):

- **12 个 query 头**, 每个 head_dim = 128
- **2 个 KV 头**, 每个 head_dim = 128
- **分组比例**: $12 / 2 = 6$, 即每 6 个 query 头共享 1 组 KV 头

### 为什么要这样做?

原因在于 **KV Cache 的内存开销**. 回忆前面讨论的 KV Cache: 推理时, 每一层都要缓存所有 token 的 $K$ 和 $V$. 如果使用 MHA (12 个 KV 头), 缓存大小为:

$$
\text{KV Cache (MHA)} = 2 \times 28 \times 12 \times T \times 128 = 86{,}016 \cdot T \text{ 个浮点数}
$$

使用 GQA (2 个 KV 头):

$$
\text{KV Cache (GQA)} = 2 \times 28 \times 2 \times T \times 128 = 14{,}336 \cdot T \text{ 个浮点数}
$$

**节省比例**: $86{,}016 / 14{,}336 = 6\times$. 对于长序列 ($T$ 很大), 这意味着巨大的内存节省.

### 张量形状

在 Qwen2-VL 文本 Decoder 中, 注意力权重的形状反映了 GQA 的设计:

```
Q: (1, T, 12, 128) → 12 个 query 头
K: (1, T,  2, 128) →  2 个 key 头
V: (1, T,  2, 128) →  2 个 value 头
```

权重矩阵:

```
q_proj.weight: (1536, 1536) → 12 heads × 128 = 1536
k_proj.weight: ( 256, 1536) →  2 heads × 128 =  256
v_proj.weight: ( 256, 1536) →  2 heads × 128 =  256
```

在计算注意力时, 每个 KV 头的 $K$ 和 $V$ 被"广播"给与它对应的 6 个 query 头. 这样, 所有 12 个 query 头都可以正常计算注意力, 只是有些头共享了相同的 key-value 信息.

---

## SwiGLU MLP: 带门控的前馈网络

### 标准 MLP vs SwiGLU

原始 Transformer 的前馈网络很简单:

$$
\text{FFN}(x) = \text{ReLU}(xW_1^T + b_1) W_2^T + b_2
$$

两层线性变换, 中间夹一个 ReLU 激活. 而 Qwen2-VL 的 Decoder 使用 **SwiGLU** (Shazeer, 2020, "GLU Variants Improve Transformer"), 它引入了门控机制:

$$
\text{SwiGLU}(x) = \left(\text{SiLU}(xW_g^T) \odot xW_u^T\right) W_d^T
$$

其中 $\odot$ 是逐元素乘法. 这里有三个投影矩阵 (而不是两个), 我们来逐一理解:

1. **门控路径** (gate): $xW_g^T$, 经过 SiLU 激活, 产生"门控信号"
2. **值路径** (up): $xW_u^T$, 产生"候选信息"
3. **门控乘积**: $\text{SiLU}(xW_g^T) \odot xW_u^T$, 门控信号筛选候选信息
4. **下投影** (down): $(...)W_d^T$, 将高维空间映射回原始维度

### SiLU 激活函数

$\text{SiLU}(x) = x \cdot \sigma(x)$, 其中 $\sigma(x) = \frac{1}{1 + e^{-x}}$ 是 sigmoid 函数.

SiLU (也叫 Swish) 是一个平滑的, 非单调的激活函数. 它允许小的负值通过 (不像 ReLU 直接截断为零), 这有助于梯度流动.

### 门控的直觉

想象一个水闸控制系统:

- **up 路径** 是水流本身 — — 原始信号经过线性变换
- **gate 路径** 是闸门 — — 决定每条"水道"打开多少
- **逐元素乘积** 是闸门控制水流 — — gate 值接近 0 的维度被关闭, gate 值接近 1 的维度被放行

这种机制让网络可以 **动态地, 逐维度地** 选择哪些信息需要保留, 哪些需要丢弃.

### Qwen2-VL 的 SwiGLU 维度

```
输入:  (1, T, 1536) ─── gate_proj ──→ (1, T, 8960)  ─── SiLU ──→ (1, T, 8960) ─┐
                     └── up_proj ────→ (1, T, 8960)  ──────────────────────────────┤
                                                                逐元素乘法         ▼
                                                              (1, T, 8960) ─ down_proj ──→ (1, T, 1536)
```

扩展比: $8960 / 1536 \approx 5.83\times$. 这比原始 Transformer 的 $4\times$ 扩展比更大, 因为 SwiGLU 用三个矩阵代替两个矩阵, 为了保持参数量大致平衡, 中间维度做了调整.

---

## 信息流与张量形状

让我们追踪一个具体的张量在 Decoder Layer 0 中的完整旅程. 假设输入序列长度 $T=3602$ (来自实际的 Qwen2-VL 推理场景):

```
输入 x: (1, 3602, 1536)
    │
    ▼
┌──────────────────────────────────────────────────────┐
│ Step 1: RMSNorm (input_layernorm)                    │
│                                                      │
│   γ₁: (1536,)                                        │
│   x̂ = RMSNorm(x, γ₁)                                │
│                                                      │
│   输入: (1, 3602, 1536) → 输出: (1, 3602, 1536)      │
│   对每个 token 的 1536 维向量独立做归一化              │
└──────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────┐
│ Step 2: Self-Attention (GQA + RoPE + 因果掩码)       │
│                                                      │
│   Q = x̂ @ Wq.T + bq   (1,3602,1536)→(1,3602,1536) │
│   K = x̂ @ Wk.T + bk   (1,3602,1536)→(1,3602, 256) │
│   V = x̂ @ Wv.T + bv   (1,3602,1536)→(1,3602, 256) │
│                                                      │
│   reshape → Q:(1,12,3602,128) K:(1,2,3602,128)      │
│   GQA: 每 6 个 Q 头共享 1 个 KV 头                    │
│   apply RoPE, causal mask, softmax, weighted sum     │
│                                                      │
│   O = concat(heads) @ Wo   → (1, 3602, 1536)        │
└──────────────────────────────────────────────────────┘
    │                              │
    ▼                              │ x 的残差路径
┌──────────────────────────┐       │
│ Step 3: 残差连接          │◄──────┘
│                          │
│   h = x + attn_out       │
│   (1, 3602, 1536)        │
└──────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────┐
│ Step 4: RMSNorm (post_attention_layernorm)           │
│                                                      │
│   γ₂: (1536,)                                        │
│   ĥ = RMSNorm(h, γ₂)                                │
│                                                      │
│   输入: (1, 3602, 1536) → 输出: (1, 3602, 1536)      │
└──────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────┐
│ Step 5: Gated MLP (SwiGLU)                           │
│                                                      │
│   gate = SiLU(ĥ @ Wg.T)    (1,3602,1536)→(1,3602,8960) │
│   up   = ĥ @ Wu.T          (1,3602,1536)→(1,3602,8960) │
│   mlp  = (gate ⊙ up) @ Wd.T (1,3602,8960)→(1,3602,1536)│
└──────────────────────────────────────────────────────┘
    │                              │
    ▼                              │ h 的残差路径
┌──────────────────────────┐       │
│ Step 6: 残差连接          │◄──────┘
│                          │
│   y = h + mlp_out        │
│   (1, 3602, 1536)        │
└──────────────────────────┘
    │
    ▼
输出 y: (1, 3602, 1536)    → 送入 Decoder Layer 1
```

**关键观察**: 输入和输出形状完全相同 $(1, 3602, 1536)$. 这使得 28 个 Decoder Layer 可以像积木一样串联堆叠.

---

## Residual Stream (残差流) 的解释

Anthropic 的研究者 (Elhage et al., 2021, "A Mathematical Framework for Transformer Circuits") 提出了一个优雅的理解方式: **把残差连接看作一条"共享通信总线" (residual stream) **.

### 传统视角 vs 残差流视角

**传统视角**: 数据"流经"一系列处理模块 — — Attention, MLP, Attention, MLP......

**残差流视角**: 有一条贯穿所有层的"主干道" (residual stream), 每个子层 (Attention 或 MLP) 从主干道上 **读取** 信息, 经过处理后 **写回** 主干道.

$$
\underbrace{x_0}_{\text{初始}} \xrightarrow{+a_0} \underbrace{x_0 + a_0}_{\text{经过 Attn\_0}} \xrightarrow{+m_0} \underbrace{x_0 + a_0 + m_0}_{\text{经过 MLP\_0}} \xrightarrow{+a_1} \cdots \xrightarrow{+m_{27}} \underbrace{x_0 + \sum_{l=0}^{27}(a_l + m_l)}_{\text{最终输出}}
$$

换句话说, 最终输出是 **初始 embedding 加上所有 56 个子层 (28 个 Attention + 28 个 MLP) 的贡献之和**.

### 这意味着什么?

1. **任意两个子层可以直接"通信"**: 第 5 层的 Attention 写入的信息, 第 20 层的 MLP 可以直接读取, 中间的层不会抹去它
2. **信息是叠加的**: 每个子层是在已有信息上"添加"新信息, 而不是"替换"
3. **早期层的信息永远存在**: 由于残差连接, 初始 embedding $x_0$ 的信息一直保留到最终输出

这也解释了为什么 Pre-Norm 配合残差连接如此有效 — — 残差路径上没有任何归一化或非线性操作, 信息和梯度都能畅通无阻地流动.

---

## 逐步数值示例 ($d=4$)

让我们用 $d=4$ 的简化版本, 手动追踪一个 token 通过 Decoder Layer 的全部 6 步.

### 初始设定

$$
x = [1.0, \; -0.5, \; 0.3, \; 0.8]
$$

$$
\gamma_1 = [1.0, \; 1.0, \; 1.0, \; 1.0] \quad \text{(input\_layernorm 权重)}
$$

$$
\gamma_2 = [0.5, \; 1.5, \; 1.0, \; 0.8] \quad \text{(post\_attention\_layernorm 权重)}
$$

### Step 1: Input RMSNorm

计算 RMS 值:

$$
\text{RMS}(x) = \sqrt{\frac{1.0^2 + (-0.5)^2 + 0.3^2 + 0.8^2}{4}} = \sqrt{\frac{1.0 + 0.25 + 0.09 + 0.64}{4}} = \sqrt{\frac{1.98}{4}} = \sqrt{0.495} \approx 0.7036
$$

归一化后乘以权重:

$$
\hat{x} = \frac{x}{0.7036} \cdot \gamma_1 = \frac{[1.0, -0.5, 0.3, 0.8]}{0.7036} \cdot [1, 1, 1, 1] \approx [1.421, \; -0.711, \; 0.426, \; 1.137]
$$

### Step 2: Self-Attention (简化)

对于单个 token 的因果 Self-Attention, 只有自身这一个位置可以参与注意力. softmax 作用在长度为 1 的序列上, 权重必然是 1.0. 因此注意力输出等价于 $\hat{x}$ 经过 $W_V$ 和 $W_O$ 的线性变换.

为了简化, 假设注意力输出为:

$$
a = [0.3, \; 0.1, \; -0.2, \; 0.4]
$$

### Step 3: 第一个残差连接

$$
h = x + a = [1.0 + 0.3, \; -0.5 + 0.1, \; 0.3 + (-0.2), \; 0.8 + 0.4] = [1.3, \; -0.4, \; 0.1, \; 1.2]
$$

### Step 4: Post-Attention RMSNorm

计算 RMS 值:

$$
\text{RMS}(h) = \sqrt{\frac{1.3^2 + (-0.4)^2 + 0.1^2 + 1.2^2}{4}} = \sqrt{\frac{1.69 + 0.16 + 0.01 + 1.44}{4}} = \sqrt{\frac{3.30}{4}} = \sqrt{0.825} \approx 0.9083
$$

归一化后乘以权重 $\gamma_2$:

$$
\hat{h} = \frac{h}{0.9083} \cdot \gamma_2 = \frac{[1.3, -0.4, 0.1, 1.2]}{0.9083} \cdot [0.5, 1.5, 1.0, 0.8]
$$

逐元素计算:

$$
\hat{h} \approx [1.431 \times 0.5, \; -0.440 \times 1.5, \; 0.110 \times 1.0, \; 1.321 \times 0.8] = [0.716, \; -0.661, \; 0.110, \; 1.057]
$$

### Step 5: SwiGLU MLP

假设 $d_{ff} = 6$ (简化的中间维度), 并设定以下权重矩阵 (仅取一小部分做演示):

为简化, 我们用 $\hat{h} = [0.716, -0.661, 0.110, 1.057]$ 与小矩阵做计算.

假设 gate 投影和 up 投影的结果分别为:

$$
xW_g^T = [0.5, \; -0.3, \; 1.2, \; 0.1, \; -0.8, \; 0.6]
$$

$$
xW_u^T = [0.4, \; 0.7, \; -0.5, \; 0.9, \; 0.2, \; -0.3]
$$

**Gate 路径** (经过 SiLU 激活):

$\text{SiLU}(z) = z \cdot \sigma(z)$, 其中 $\sigma(z) = 1/(1+e^{-z})$:

| $z$  | $\sigma(z)$ | $\text{SiLU}(z) = z\sigma(z)$ |
| ---- | ----------- | ----------------------------- |
| 0.5  | 0.622       | 0.311                         |
| -0.3 | 0.426       | -0.128                        |
| 1.2  | 0.769       | 0.923                         |
| 0.1  | 0.525       | 0.052                         |
| -0.8 | 0.310       | -0.248                        |
| 0.6  | 0.646       | 0.387                         |

**门控乘积** (逐元素相乘):

$$
\text{gate} \odot \text{up} = [0.311 \times 0.4, \; (-0.128) \times 0.7, \; 0.923 \times (-0.5), \; 0.052 \times 0.9, \; (-0.248) \times 0.2, \; 0.387 \times (-0.3)]
$$

$$
= [0.124, \; -0.089, \; -0.461, \; 0.047, \; -0.050, \; -0.116]
$$

**下投影** (假设 $W_d$ 将 6 维映射回 4 维):

假设下投影后结果为:

$$
m = [0.15, \; -0.08, \; 0.22, \; -0.12]
$$

### Step 6: 第二个残差连接

$$
y = h + m = [1.3 + 0.15, \; -0.4 + (-0.08), \; 0.1 + 0.22, \; 1.2 + (-0.12)] = [1.45, \; -0.48, \; 0.32, \; 1.08]
$$

### 回顾全过程

```
x    = [1.000, -0.500,  0.300,  0.800]   原始输入
 ↓ RMSNorm
x̂    = [1.421, -0.711,  0.426,  1.137]   归一化后
 ↓ Attention
a    = [0.300,  0.100, -0.200,  0.400]   注意力输出
 ↓ + x（残差）
h    = [1.300, -0.400,  0.100,  1.200]   融合上下文后
 ↓ RMSNorm
ĥ    = [0.716, -0.661,  0.110,  1.057]   再次归一化
 ↓ SwiGLU
m    = [0.150, -0.080,  0.220, -0.120]   MLP 输出
 ↓ + h（残差）
y    = [1.450, -0.480,  0.320,  1.080]   最终输出
```

可以看到, 输出 $y$ 与输入 $x$ 相近但有所不同 — — 每一层都是在原有信息的基础上做"微调", 而不是彻底改写. 这正是残差连接 + Pre-Norm 的效果.

---

## 参数量分析

Qwen2-VL-2B 文本解码器每层 (layer 0 为例) 的参数明细:

| 组件                     | 权重键      | 形状                     | 参数量     | 说明                   |
| ------------------------ | ----------- | ------------------------ | ---------- | ---------------------- |
| input_layernorm          | $\gamma_1$  | $(1536,)$                | 1,536      | 每维一个缩放系数       |
| self_attn.q_proj         | $W_Q + b_Q$ | $(1536, 1536) + (1536,)$ | 2,361,344  | $12 \times 128 = 1536$ |
| self_attn.k_proj         | $W_K + b_K$ | $(256, 1536) + (256,)$   | 393,472    | $2 \times 128 = 256$   |
| self_attn.v_proj         | $W_V + b_V$ | $(256, 1536) + (256,)$   | 393,472    | $2 \times 128 = 256$   |
| self_attn.o_proj         | $W_O$       | $(1536, 1536)$           | 2,359,296  | 无 bias                |
| post_attention_layernorm | $\gamma_2$  | $(1536,)$                | 1,536      | 每维一个缩放系数       |
| mlp.gate_proj            | $W_g$       | $(8960, 1536)$           | 13,762,560 | SwiGLU 门控            |
| mlp.up_proj              | $W_u$       | $(8960, 1536)$           | 13,762,560 | SwiGLU 上投影          |
| mlp.down_proj            | $W_d$       | $(1536, 8960)$           | 13,762,560 | SwiGLU 下投影          |
| **单层合计**             |             |                          | **~46.8M** |                        |

### 参数量计算验证

让我们逐项核实:

- **q_proj**: $1536 \times 1536 + 1536 = 2{,}359{,}296 + 1{,}536 = 2{,}361{,}344$ ✓ (注意 $12 \text{ heads} \times 128 \text{ head\_dim} = 1536$)
- **k_proj**: $256 \times 1536 + 256 = 393{,}216 + 256 = 393{,}472$ ✓ (注意 $2 \text{ heads} \times 128 \text{ head\_dim} = 256$)
- **MLP 三矩阵**: $8960 \times 1536 \times 3 = 41{,}287{,}680$ ✓

**28 层总计**:

$$
46{,}798{,}336 \times 28 \approx 1.31\text{B}
$$

这 1.31B 参数占了 Qwen2-VL-2B 模型总参数的绝大部分. 可以说, Decoder Layer 才是真正的"大头".

### 参数分布

从比例上看, 每层参数的分布非常不均匀:

- **MLP 占比**: $41{,}287{,}680 / 46{,}798{,}336 \approx 88\%$
- **Attention 占比**: $\approx 12\%$
- **LayerNorm 占比**: $< 0.01\%$

MLP 的参数量远远超过 Attention, 这是因为 SwiGLU 使用了三个大矩阵. 这也是为什么模型中大部分"知识"被认为存储在 MLP 权重中.

---

## Vision Block vs Decoder Layer 对比

Qwen2-VL 中有两种"层": Vision Encoder 的 Vision Block 和 Text Decoder 的 Decoder Layer. 虽然它们都源自 Transformer, 但设计上有显著差异:

| 方面           | Vision Block         | Decoder Layer           |
| -------------- | -------------------- | ----------------------- |
| **归一化**     | LayerNorm            | RMSNorm                 |
| **注意力类型** | MHA (16 个头)        | GQA (12 Q 头 / 2 KV 头) |
| **MLP 类型**   | 标准 MLP (QuickGELU) | SwiGLU (门控 MLP)       |
| **因果掩码**   | ❌ 无 (双向注意力)   | ✅ 有 (只看过去)        |
| **位置编码**   | RoPE                 | RoPE                    |
| **embed_dim**  | 1280                 | 1536                    |
| **层数**       | 32                   | 28                      |
| **每层参数**   | ~19.7M               | ~46.8M                  |
| **KV Cache**   | 不需要               | 需要                    |

### 为什么设计不同?

这些差异反映了 **视觉和语言任务的本质区别**:

1. **双向 vs 因果**: 图像中每个 patch 都应该能看到所有其他 patch (全局语义), 但文本生成必须遵循时间顺序 (不能偷看未来)

2. **MHA vs GQA**: Vision Encoder 只做一次前向传播 (不需要自回归生成), 所以没有 KV Cache 的内存压力; 而文本 Decoder 需要逐 token 生成, GQA 大幅减少 KV Cache 大小

3. **标准 MLP vs SwiGLU**: 文本理解需要更强的非线性表达能力, SwiGLU 的门控机制在大量实验中被证明优于标准 MLP

4. **LayerNorm vs RMSNorm**: 这主要是时代和效率的选择 — — 较新的文本模型普遍采用 RMSNorm

---

## NumPy 实现

以下是 Decoder Layer 及其子模块的完整 NumPy 实现, 对应 `impl.py` 中的代码:

```python
def gated_mlp(x, gate_weight, up_weight, down_weight):
    """SwiGLU 门控前馈网络

    Args:
        x:            输入张量 (B, T, d)
        gate_weight:  门控投影权重 (d_ff, d)
        up_weight:    上投影权重 (d_ff, d)
        down_weight:  下投影权重 (d, d_ff)

    Returns:
        输出张量 (B, T, d)
    """
    # 门控路径：线性投影 + SiLU 激活
    # x @ gate_weight.T: (B,T,d) @ (d,d_ff) → (B,T,d_ff)
    gate = silu(x @ gate_weight.T)

    # 值路径：纯线性投影，不加激活函数
    # x @ up_weight.T: (B,T,d) @ (d,d_ff) → (B,T,d_ff)
    up = x @ up_weight.T

    # 门控乘积 + 下投影
    # (gate * up): (B,T,d_ff) 逐元素相乘 → (B,T,d_ff)
    # ... @ down_weight.T: (B,T,d_ff) @ (d_ff,d) → (B,T,d)
    return (gate * up) @ down_weight.T


def decoder_layer(x, input_ln_w, post_ln_w, gate_w, up_w, down_w, attn_output):
    """Decoder Layer：一个完整的 Transformer 解码器层

    注意：Self-Attention 输出以参数形式传入（完整的 GQA + RoPE + 因果掩码
    实现复杂度高，这里用 dump 值替代来验证层的组装逻辑）。

    Args:
        x:            层输入 (B, T, d)
        input_ln_w:   输入层归一化权重 (d,)
        post_ln_w:    注意力后层归一化权重 (d,)
        gate_w:       MLP 门控投影 (d_ff, d)
        up_w:         MLP 上投影 (d_ff, d)
        down_w:       MLP 下投影 (d, d_ff)
        attn_output:  self-attention 的输出 (B, T, d)

    Returns:
        层输出 (B, T, d)，形状与输入完全相同
    """
    # Step 1: Pre-Norm（在 attention 之前做 RMSNorm）
    x_normed = rms_norm(x, input_ln_w)

    # Step 2: Self-Attention（这里使用预计算值）
    attn_out = attn_output

    # Step 3: 第一个残差连接——注意是 x（而非 x_normed）加上 attn_out
    # 这就是 Pre-Norm 的核心：归一化只作用于子层内部，残差路径保持干净
    hidden = residual_add(x, attn_out)

    # Step 4: Pre-Norm（在 MLP 之前做 RMSNorm）
    hidden_normed = rms_norm(hidden, post_ln_w)

    # Step 5: SwiGLU MLP
    mlp_out = gated_mlp(hidden_normed, gate_w, up_w, down_w)

    # Step 6: 第二个残差连接——同样是 hidden（而非 hidden_normed）加上 mlp_out
    output = residual_add(hidden, mlp_out)

    return output
```

请注意 Step 3 和 Step 6 的关键细节: **残差连接加的是归一化之前的值**. 这是 Pre-Norm 架构的标志 — — 归一化只影响子层 (Attention / MLP) 的输入, 而残差路径始终保持"原始"状态.

---

## 在 Qwen2-VL 中的位置

```
Qwen2-VL 完整架构
│
├── Vision Encoder (ViT)
│   ├── Patch Embedding
│   ├── Vision Block × 32
│   └── ...
│
├── Visual-Language Merger
│   └── 将视觉 token 映射到文本空间
│
└── Text Decoder (Qwen2)
    ├── Token Embedding         ← 将 token ID 映射为 1536 维向量
    │
    ├── Decoder Layer × 28      ← 我们在这里 ★
    │   ├── RMSNorm (input_layernorm)
    │   ├── Self-Attention (GQA + RoPE + Causal Mask)
    │   ├── 残差连接
    │   ├── RMSNorm (post_attention_layernorm)
    │   ├── Gated MLP (SwiGLU)
    │   └── 残差连接
    │
    ├── Final RMSNorm           ← 最后一层归一化
    └── LM Head                 ← 线性层，输出 vocab_size=151936 的 logits
```

从 Token Embedding 到 LM Head, 数据维度的变化为:

$$
\text{token IDs} \; (B, T) \xrightarrow{\text{Embedding}} (B, T, 1536) \xrightarrow{\text{Layer} \times 28} (B, T, 1536) \xrightarrow{\text{LM Head}} (B, T, 151936)
$$

28 个 Decoder Layer 保持维度不变, 最终由 LM Head 将 1536 维向量投影到 151,936 维的词汇表空间, 产生每个 token 位置上"下一个词是什么"的概率分布.

---

## 常见误解与陷阱

### ❌ 误解 1: "Decoder 需要配合 Encoder 才能工作"

这是对原始 Encoder-Decoder Transformer 的过度泛化. 在 Decoder-Only 架构中, Decoder 自给自足 — — 它通过 Self-Attention 自行处理上下文, 不需要 Cross-Attention 连接 Encoder 的输出. GPT, LLaMA, Qwen 的文本部分都是纯 Decoder-Only.

> Qwen2-VL 确实有一个 Vision Encoder, 但视觉信息是通过 Merger 层转换为 token 后直接拼接到文本序列中的, 而不是通过 Cross-Attention.

### ❌ 误解 2: "因果掩码和 padding 掩码是一回事"

- **因果掩码** (causal mask): 防止看到未来的 token, 是一个下三角矩阵, 所有序列都一样
- **Padding 掩码** (padding mask): 防止注意到 padding token, 因 batch 中不同序列长度不同而不同

两者的目的完全不同, 虽然在实现中可能被合并到同一个掩码矩阵中.

### ❌ 误解 3: "GQA 减少了模型容量"

GQA 只减少了 **K 和 V 的头数**, Q 的头数保持不变. 由于 Query 仍然有 12 个独立的头, 模型可以从不同角度"提问". 实验表明, GQA 在推理效率和模型质量之间取得了很好的平衡 — — KV cache 缩小 $6\times$, 但性能几乎不降.

### ❌ 误解 4: "Pre-Norm 中, RMSNorm 作用于残差路径"

不对! Pre-Norm 的关键是: **RMSNorm 只作用于子层的输入, 而残差路径是干净的**. 看代码中的 `hidden = residual_add(x, attn_out)` — — 加的是 `x` (归一化前), 不是 `x_normed` (归一化后).

### ❌ 误解 5: "每层都完全重写 token 的表示"

由于残差连接, 每层只是在原有表示上**叠加**一个增量 $\Delta x$. 输出 $y = x + \Delta x$ 中, $x$ 的信息被完整保留. 28 层下来, 最终表示 = 初始 embedding + 56 个增量的累加.

---

## 总结

| 要素          | 细节                                                 |
| ------------- | ---------------------------------------------------- |
| **核心任务**  | 逐层精炼 token 表示, 使下一词预测越来越准确          |
| **架构类型**  | Decoder-Only (无 Encoder, 无 Cross-Attention)        |
| **计算流程**  | RMSNorm → Attention → 残差 → RMSNorm → SwiGLU → 残差 |
| **归一化**    | Pre-Norm + RMSNorm (非 LayerNorm)                    |
| **注意力**    | GQA (12 Q 头, 2 KV 头, head_dim=128)                 |
| **掩码**      | 因果掩码 (下三角矩阵, 禁止看未来)                    |
| **MLP**       | SwiGLU (gate + up + down 三矩阵)                     |
| **残差连接**  | 两处: Attention 后和 MLP 后                          |
| **输入/输出** | $(B, T, 1536) \to (B, T, 1536)$, 形状不变            |
| **单层参数**  | ~46.8M (MLP 占 ~88%, Attention 占 ~12%)              |
| **层数**      | 28 层, 共 ~1.31B 参数                                |
| **推理优化**  | KV Cache ($\sim\!67\times$ 加速)                     |
| **残差流**    | 贯穿所有层的共享通信总线                             |

---

## 延伸阅读

1. **Vaswani et al. (2017)** — _"Attention Is All You Need"_
   原始 Transformer 论文, 定义了 Encoder-Decoder 架构和 Self-Attention 机制.

2. **Radford et al. (2018)** — _"Improving Language Understanding by Generative Pre-Training"_ (GPT-1)
   首次证明 Decoder-Only 架构在语言理解任务上的有效性.

3. **Zhang & Sennrich (2019)** — _"Root Mean Square Layer Normalization"_
   提出 RMSNorm, 去掉均值中心化, 计算更高效.

4. **Shazeer (2020)** — _"GLU Variants Improve Transformer"_
   系统比较了各种 GLU 变体 (包括 SwiGLU), 证明门控机制优于标准 MLP.

5. **Ainslie et al. (2023)** — _"GQA: Training Generalized Multi-Query Attention from Multi-Head Checkpoints"_
   提出 Grouped Query Attention, 在 MHA 和 MQA 之间找到平衡.

6. **Elhage et al. (2021)** — _"A Mathematical Framework for Transformer Circuits"_ (Anthropic)
   提出残差流 (residual stream) 视角, 为理解 Transformer 内部机制奠定了数学基础.

7. **Touvron et al. (2023)** — _"LLaMA: Open and Efficient Foundation Language Models"_
   开源 Decoder-Only LLM 的代表作, 使用了 Pre-Norm + RMSNorm + SwiGLU + GQA 的组合.

8. **Qwen Team (2024)** — _"Qwen2-VL Technical Report"_
   Qwen2-VL 模型的技术报告, 本系列教程的分析对象. ```
