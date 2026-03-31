# 14 — 词嵌入 (Token Embedding)

> **一句话概括**: 词嵌入是连接人类语言和机器数字世界的桥梁 — —
> 它把离散的 token ID 翻译成稠密的高维向量, 让"国王"和"女王"在数学空间中成为邻居.

---

## 1. 从一个根本问题开始

计算机只懂数字. 它不认识"猫", "love", "月亮"这些符号.

当你对 ChatGPT 说"帮我写一首诗"时, 第一步不是生成优美的文字 — —
而是把你的话**翻译成数字**. 这个翻译过程就是**词嵌入** (Token Embedding).

但"翻译成数字"有很多方式. 编号? 独热编码? 随机向量?
选择哪种方式, 决定了模型能否理解语义.

让我们从最朴素的方案开始, 一步步推导出现代嵌入的设计.

---

## 2. 历史演进: 从独热编码到学习型嵌入

### 2.1 最朴素的方案: 编号

假设词表有 5 个词: $\{\text{猫}, \text{狗}, \text{鱼}, \text{鸟}, \text{虫}\}$.

最简单的编码: 猫=0, 狗=1, 鱼=2, 鸟=3, 虫=4.

**问题**: 这暗示了一个不存在的顺序关系 — — "鱼"(2) 是"猫"(0) 和"鸟"(4) 的"中间"吗?
数字的大小关系会误导模型.

### 2.2 独热编码 (One-Hot Encoding)

为了消除数字大小的干扰, 我们给每个词分配一个**正交**的向量:

$$
\text{猫} = [1, 0, 0, 0, 0], \quad
\text{狗} = [0, 1, 0, 0, 0], \quad
\text{鱼} = [0, 0, 1, 0, 0], \quad \ldots
$$

用数学记号, token ID 为 $t$ 的独热向量为 $\mathbf{e}_t \in \mathbb{R}^V$,
其中 $V$ 是词表大小, 只有第 $t$ 个位置为 1, 其余为 0.

**优点**: 每个词与其他词完全正交, 没有虚假的相似性.

**三大缺点**:

1. **维度爆炸**: Qwen2-VL 的词表有 151936 个 token. 每个独热向量就是 151936 维的,
   一个长度为 $L$ 的序列需要 $L \times 151936$ 个数字 — — 几乎全是 0.

2. **没有语义信息**: $\text{猫}$ 和 $\text{狗}$ 的独热向量点积为 0,
   $\text{猫}$ 和 $\text{石头}$ 的点积也是 0. 在独热空间里, 所有词都等距 — —
   "猫"和"狗"之间的距离等于"猫"和"微积分"之间的距离.

3. **计算浪费**: 后续的矩阵乘法中, 与 0 相乘的运算全是浪费.

### 2.3 分布式表示: Word2Vec 的革命 (Mikolov et al., 2013)

2013 年, Google 的 Tomas Mikolov 提出了 Word2Vec, 核心思想来自语言学家 J.R. Firth
(1957) 的名言:

> _"You shall know a word by the company it keeps."_
> (从一个词的上下文就能了解这个词.)

Word2Vec 用低维 (通常 100-300 维) 的稠密向量来表示词语, 这些向量通过**预测上下文**
来训练. 训练完成后, 语义相近的词在向量空间中距离也相近.

最著名的例子:

$$
\vec{\text{king}} - \vec{\text{man}} + \vec{\text{woman}} \approx \vec{\text{queen}}
$$

这个"king-queen"类比说明, 嵌入空间捕捉到了"性别"这个语义维度!

### 2.4 从 Word2Vec 到 Transformer 嵌入

Word2Vec 的嵌入是**静态的** — — "bank"在"river bank"和"bank account"中
有相同的向量.

现代 Transformer (包括 Qwen2-VL) 使用的嵌入层本质上和 Word2Vec 类似 — —
都是一个查找表 — — 但有两个关键区别:

1. **嵌入是端到端学习的**: 不是单独预训练, 而是作为整个模型的一部分一起训练
2. **上下文由后续层提供**: 嵌入层本身还是静态的 (同一个 token 总是查到同一个向量),
   但经过多层 Attention 后, 每个 token 的表示会融合上下文信息变成动态的

---

## 3. 数学定义

### 3.1 嵌入查找

设嵌入矩阵为 $\mathbf{E} \in \mathbb{R}^{V \times d}$, 其中:

- $V$ = 词表大小 (Qwen2-VL: 151936)
- $d$ = 嵌入维度 (Qwen2-VL: 1536)

矩阵的第 $t$ 行 $\mathbf{E}[t] \in \mathbb{R}^d$ 就是 token $t$ 的嵌入向量.

对于一个 token 序列 $[t_0, t_1, \ldots, t_{L-1}]$, 嵌入操作为:

$$
\mathbf{X} = \begin{bmatrix} \mathbf{E}[t_0] \\ \mathbf{E}[t_1] \\ \vdots \\ \mathbf{E}[t_{L-1}] \end{bmatrix} \in \mathbb{R}^{L \times d}
$$

**就是按行索引取出对应的向量. **

### 3.2 与独热编码 + 矩阵乘法的等价性

这是一个重要的数学观察. 设 token $t$ 的独热向量为
$\mathbf{o}_t = [0, \ldots, 0, 1, 0, \ldots, 0]^T \in \mathbb{R}^V$
(第 $t$ 位为 1).

那么:

$$
\mathbf{o}_t^T \cdot \mathbf{E} = \mathbf{E}[t]
$$

**证明** (逐元素展开):

$$
(\mathbf{o}_t^T \cdot \mathbf{E})_j = \sum_{i=0}^{V-1} o_{t,i} \cdot E_{i,j}
$$

由于 $o_{t,i} = 0$ 对所有 $i \neq t$, $o_{t,t} = 1$, 所以:

$$
(\mathbf{o}_t^T \cdot \mathbf{E})_j = 1 \cdot E_{t,j} = E_{t,j}
$$

即结果就是 $\mathbf{E}$ 的第 $t$ 行. $\blacksquare$

### 3.3 用一个小例子验证

```
嵌入矩阵 E (V=4, d=3):
    E = [[0.1, 0.2, 0.3],   ← token 0
         [0.4, 0.5, 0.6],   ← token 1
         [0.7, 0.8, 0.9],   ← token 2
         [1.0, 1.1, 1.2]]   ← token 3

查找 token 2:

方法 1：直接索引
    E[2] = [0.7, 0.8, 0.9]  ✓

方法 2：独热编码 × 矩阵
    one_hot(2) = [0, 0, 1, 0]

    [0, 0, 1, 0] × [[0.1, 0.2, 0.3],
                      [0.4, 0.5, 0.6],
                      [0.7, 0.8, 0.9],
                      [1.0, 1.1, 1.2]]

    = 0×[0.1,0.2,0.3] + 0×[0.4,0.5,0.6] + 1×[0.7,0.8,0.9] + 0×[1.0,1.1,1.2]
    = [0.7, 0.8, 0.9]  ✓

两种方法结果完全相同！
```

**为什么实际使用索引而非矩阵乘法? **

索引操作的时间复杂度是 $O(d)$ — — 只需要复制一行.
矩阵乘法的时间复杂度是 $O(V \times d)$ — — 需要与 151936 行都做乘法 (其中 151935 次
乘以 0, 纯属浪费).

---

## 4. 嵌入空间的几何直觉

### 4.1 什么是"空间中的距离"?

1536 维的空间无法可视化, 但我们可以借助低维类比来建立直觉.

想象一个 2D 平面. 每个词是平面上的一个点. 经过良好训练后:

- "猫"和"狗"这样语义相近的词, 在空间中距离很近
- "猫"和"微积分"这样无关的词, 距离很远
- "跑"和"走"的距离 < "跑"和"飞"的距离 < "跑"和"睡觉"的距离

### 4.2 余弦相似度

在高维空间中, 通常用**余弦相似度**而非欧氏距离来衡量相似性:

$$
\text{sim}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{|\mathbf{a}| \cdot |\mathbf{b}|}
= \cos(\theta)
$$

余弦相似度衡量的是两个向量的**方向**是否接近 (忽略长度/模长).
值域为 $[-1, 1]$, 1 表示方向完全相同, 0 表示正交, -1 表示方向相反.

### 4.3 嵌入维度的选择: 为什么是 1536?

嵌入维度 $d$ 是表达能力和计算成本之间的权衡:

| 模型         | 嵌入维度 | 参数规模  |
| ------------ | -------- | --------- |
| Word2Vec     | 100-300  | 数千万    |
| GPT-2 Small  | 768      | 1.24 亿   |
| GPT-2 Large  | 1280     | 7.74 亿   |
| Qwen2-VL-2B  | 1536     | 15 亿     |
| Llama-3-8B   | 4096     | 80 亿     |
| GPT-4 (推测) | 12288+   | 1.8 万亿? |

维度越高:

- **优点**: 能编码更多语义维度 (情感, 时态, 领域, 风格......), 表达能力更强
- **缺点**: 嵌入矩阵更大 (内存), 后续计算更多 (FLOPs)

Qwen2-VL-2B 选择 1536 是在 ~15 亿参数的预算下做出的平衡.

### 4.4 嵌入空间中的"方向"有意义

训练好的嵌入空间中, 某些方向对应特定的语义概念. 经典的例子:

$$
\vec{\text{king}} - \vec{\text{queen}} \approx \vec{\text{man}} - \vec{\text{woman}}
$$

这说明"性别"在嵌入空间中被编码为一个大致固定的方向向量.

类似地:

$$
\vec{\text{Paris}} - \vec{\text{France}} \approx \vec{\text{Tokyo}} - \vec{\text{Japan}}
$$

"首都"关系也被编码为一个方向.

虽然现代 Transformer 的嵌入可能不像 Word2Vec 那样有如此清晰的线性类比关系
(因为上下文化由后续 Attention 层完成), 但嵌入层仍然学到了丰富的词汇级语义信息.

---

## 5. 子词分词 (Subword Tokenization)

### 5.1 为什么不直接用词?

如果每个英文单词是一个 token, 词表会非常大 (英文有数十万词),
而且无法处理新词 (OOV, out-of-vocabulary 问题).

如果每个字符是一个 token, 词表很小 (ASCII 只有 128 个),
但序列会非常长 ("understanding" 变成 13 个 token), 丢失了词级语义.

**子词分词**是折中方案: 把常见词保留为整词, 把罕见词拆成子词片段.

### 5.2 BPE (Byte Pair Encoding)

Qwen2-VL 使用 BPE 变体. BPE 的核心算法:

1. 从字符级别开始
2. 统计所有相邻 token 对的频率
3. 合并频率最高的 token 对为一个新 token
4. 重复步骤 2-3 直到达到目标词表大小

例如:

```
初始词表：a, b, c, d, e, ...
语料中 "ab" 出现 100 次 → 合并为 "ab"
语料中 "abc" 出现 80 次 → 合并为 "abc"
...
```

### 5.3 Qwen2-VL 的词表: 151936 个 token

这个数字包括:

- 基本的 Unicode 字符和字节
- 常见的英文单词和子词
- 常见的中文词语和短语
- 特殊 token (如 `<|im_start|>`, `<|im_end|>`, `<|vision_start|>`, `<|vision_end|>` 等)

151936 这个数字看起来有点奇怪 — — 为什么不是整数千或 2 的幂?
这是因为 BPE 训练过程中的合并次数和初始词表大小共同决定的, 不追求"好看"的数字.

---

## 6. 权重绑定 (Weight Tying / Shared Embedding)

### 6.1 LM Head 是什么?

Transformer 语言模型的最后一步是**语言模型头** (LM Head), 它把隐藏状态映射回词表上的概率分布:

$$
\text{logits} = h \cdot \mathbf{E}^T \in \mathbb{R}^V
$$

其中 $h \in \mathbb{R}^d$ 是最后一层的隐藏状态, $\mathbf{E} \in \mathbb{R}^{V \times d}$ 是嵌入矩阵.

等等 — — 这里用的矩阵和嵌入层的矩阵**是同一个** $\mathbf{E}$!

### 6.2 为什么可以共享权重?

这个设计叫做**权重绑定** (weight tying), 最早由 Press & Wolf (2017) 提出.
直觉如下:

- **嵌入层**: token ID → 向量. $\mathbf{E}[t]$ 是 token $t$ 的"语义编码".
- **LM Head**: 向量 → token 概率. $h \cdot \mathbf{E}[t]^T$ 是隐藏状态 $h$
  与 token $t$ 的"语义编码"的**余弦相似度** (未归一化).

两者使用同一个"语义编码本"是合理的: 如果一个 token 的嵌入向量与某个隐藏状态
方向一致, 那么这个 token 就应该是模型想要输出的 token.

### 6.3 内存节省

权重绑定的实际好处是**节省一半的嵌入参数**:

$$
\text{嵌入矩阵大小} = V \times d = 151936 \times 1536 \approx 2.33 \times 10^8 \text{ 个参数}
$$

如果不共享, 嵌入层和 LM Head 各需要一份, 总计 $4.66 \times 10^8$ 个参数.
共享后只需要 $2.33 \times 10^8$ 个.

在 Qwen2-VL-2B 的约 15 亿总参数中, 嵌入矩阵占了约 **15.5%** — — 这是一个
显著的比例!

### 6.4 在 Qwen2-VL 中的体现

```python
# 嵌入层权重
embed_weight = model_weights["model.embed_tokens.weight"]  # (151936, 1536)

# LM Head 权重
lm_head_weight = model_weights["lm_head.weight"]  # (151936, 1536)

# 在 Qwen2-VL-2B 中，这两个是同一个张量！
assert np.array_equal(embed_weight, lm_head_weight)  # True
```

---

## 7. 多模态嵌入: 视觉 Token 的替换

### 7.1 Qwen2-VL 的多模态输入

Qwen2-VL 是一个**多模态模型** — — 它可以同时处理文本和图像.
但 Transformer 只接受向量序列作为输入. 如何统一?

答案是: **在嵌入序列中"替换"特定位置的向量**.

### 7.2 处理流程

```
Step 1: 文本分词
    "请描述这张图片 <image>" → [t₀, t₁, t₂, ..., t_text, v₀, v₁, ..., v_k]
    其中 v₀...v_k 是图像占位符 token

Step 2: 嵌入查找（对所有 token，包括占位符）
    embeddings = E[token_ids]    ← 我们验证的就是这一步
    形状: (1, L, 1536)

Step 3: 视觉编码器处理图像
    visual_tokens = VisionEncoder(image)
    形状: (num_visual_tokens, 1536)

Step 4: 替换占位符位置
    embeddings[:, visual_positions, :] = visual_tokens
```

### 7.3 `_input_mm_token_type_ids`: 多模态位置标记

Qwen2-VL 使用一个名为 `_input_mm_token_type_ids` 的张量来标记哪些位置是视觉 token:

- 值为 **0** 的位置: 纯文本 token
- 值为 **> 0** 的位置: 视觉 token (图像或视频)

```
token_ids:           [108386, 104,   151655, 151655, ..., 151655, 198  ]
mm_token_type_ids:   [0,      0,     1,      1,      ..., 1,      0    ]
                      文本     文本    图像    图像          图像    文本
```

> **重要提示**: 我们在算子 14 中验证的是 **Step 2 的输出** — — 即替换发生之前的
> 原始嵌入结果. 这使得验证很简单: 只需要比较 `E[token_ids]` 与保存的激活值即可.

---

## 8. 详细数值示例

### 8.1 最小嵌入表示例

```python
import numpy as np

# 词表大小 V=5, 嵌入维度 d=3
E = np.array([
    [0.1, 0.2, 0.3],   # token 0: "猫"
    [0.4, 0.5, 0.6],   # token 1: "狗"
    [0.7, 0.8, 0.9],   # token 2: "鱼"
    [1.0, 1.1, 1.2],   # token 3: "鸟"
    [1.3, 1.4, 1.5],   # token 4: "虫"
])

# 输入序列: "鱼 猫 虫 狗"
token_ids = np.array([2, 0, 4, 1])

# 嵌入查找
output = E[token_ids]
```

手工计算:

```
token_ids[0] = 2 → E[2] = [0.7, 0.8, 0.9]    # "鱼"
token_ids[1] = 0 → E[0] = [0.1, 0.2, 0.3]    # "猫"
token_ids[2] = 4 → E[4] = [1.3, 1.4, 1.5]    # "虫"
token_ids[3] = 1 → E[1] = [0.4, 0.5, 0.6]    # "狗"

output = [[0.7, 0.8, 0.9],
          [0.1, 0.2, 0.3],
          [1.3, 1.4, 1.5],
          [0.4, 0.5, 0.6]]
```

### 8.2 独热编码等价性验证

继续上面的例子, 验证 token 2 的查找:

$$
\mathbf{o}_2 = [0, 0, 1, 0, 0]
$$

$$
\mathbf{o}_2^T \cdot E = 0 \times [0.1, 0.2, 0.3] + 0 \times [0.4, 0.5, 0.6]
+ 1 \times [0.7, 0.8, 0.9] + 0 \times [1.0, 1.1, 1.2] + 0 \times [1.3, 1.4, 1.5]
$$

$$
= [0.7, 0.8, 0.9] = E[2] \; ✓
$$

### 8.3 余弦相似度计算

用上面的嵌入向量计算"猫"和"狗", "猫"和"虫"的相似度:

$$
\text{sim}(\text{猫}, \text{狗}) = \frac{[0.1, 0.2, 0.3] \cdot [0.4, 0.5, 0.6]}{|[0.1, 0.2, 0.3]| \cdot |[0.4, 0.5, 0.6]|}
$$

分子: $0.1 \times 0.4 + 0.2 \times 0.5 + 0.3 \times 0.6 = 0.04 + 0.10 + 0.18 = 0.32$

分母: $\sqrt{0.01 + 0.04 + 0.09} \times \sqrt{0.16 + 0.25 + 0.36} = \sqrt{0.14} \times \sqrt{0.77} = 0.374 \times 0.877 = 0.328$

$$
\text{sim}(\text{猫}, \text{狗}) = 0.32 / 0.328 \approx 0.976
$$

$$
\text{sim}(\text{猫}, \text{虫}) = \frac{0.1 \times 1.3 + 0.2 \times 1.4 + 0.3 \times 1.5}{0.374 \times \sqrt{1.69+1.96+2.25}}
= \frac{0.13 + 0.28 + 0.45}{0.374 \times \sqrt{5.90}}
= \frac{0.86}{0.374 \times 2.429}
= \frac{0.86}{0.909} \approx 0.946
$$

> 注意: 这个例子中所有向量都在同一象限, 所以余弦相似度都很高.
> 在真实的嵌入空间中, 1536 维向量可以指向各种方向, 差异会更明显.

---

## 9. 内存分析

### 9.1 嵌入矩阵的内存占用

$$
\text{内存} = V \times d \times \text{bytes\_per\_param}
$$

| 精度               | 每参数字节数 | 嵌入矩阵大小                                      |
| ------------------ | ------------ | ------------------------------------------------- |
| float32            | 4            | $151936 \times 1536 \times 4 = 933.5\text{ MB}$   |
| float16 / bfloat16 | 2            | $151936 \times 1536 \times 2 = 466.8\text{ MB}$   |
| int8               | 1            | $151936 \times 1536 \times 1 = 233.4\text{ MB}$   |
| int4               | 0.5          | $151936 \times 1536 \times 0.5 = 116.7\text{ MB}$ |

在 float32 下, **仅嵌入矩阵就占了近 1 GB 的内存**!
这就是为什么权重绑定 (与 LM Head 共享) 如此重要 — — 省了一整个 $V \times d$ 的矩阵.

### 9.2 参数占比

Qwen2-VL-2B 约有 15 亿参数. 嵌入矩阵有 $151936 \times 1536 \approx 2.33$ 亿参数.

$$
\text{嵌入参数占比} = \frac{233 \times 10^6}{1.5 \times 10^9} \approx 15.5\%
$$

也就是说, 这个模型约 **1/6 的参数**只是一张查找表!
这在小模型中尤为显著 (大模型中这个比例更低, 因为 Transformer 层的参数占主导).

### 9.3 嵌入矩阵的稀疏访问

虽然嵌入矩阵很大, 但每次前向传播只访问其中很小的一部分.
对于长度 $L = 3602$ 的序列:

$$
\text{实际访问的参数} = L \times d = 3602 \times 1536 \approx 5.5 \times 10^6
$$

$$
\text{访问比例} = \frac{5.5 \times 10^6}{2.33 \times 10^8} \approx 2.4\%
$$

每次前向传播只访问了嵌入矩阵的 **2.4%**! 这就是为什么嵌入层虽然参数多,
但计算量却很小的原因.

---

## 10. 与相关概念的比较

| 概念            | 输入           | 输出           | 特点           |
| --------------- | -------------- | -------------- | -------------- |
| 独热编码        | token ID       | $V$ 维稀疏向量 | 无语义, 高维   |
| Word2Vec        | token ID       | $d$ 维稠密向量 | 预训练, 静态   |
| Token Embedding | token ID       | $d$ 维稠密向量 | 端到端训练     |
| 位置编码        | 位置索引       | $d$ 维向量     | 编码位置信息   |
| LM Head         | $d$ 维隐藏状态 | $V$ 维 logits  | 嵌入的"逆操作" |

### Token Embedding vs 位置编码

Token Embedding 编码的是"**这是什么词**", 位置编码编码的是"**这个词在哪里**".
在 Qwen2-VL 中, 这两个信息分别处理:

- Token Embedding 通过查找表获得词级语义
- 位置信息通过 Rotary Position Embedding (RoPE, 算子 06) 编码在注意力计算中

### Token Embedding vs Visual Embedding

文本和视觉走不同的路:

```
文本: token_ids → E[token_ids] → (L_text, 1536)
视觉: pixel_values → Conv3d → 32×VisionBlock → PatchMerger → (L_visual, 1536)
```

两条路的输出维度都是 1536, 所以可以在同一个序列中拼接!

---

## 11. 常见误解与陷阱

### 误解 1: "嵌入向量是人工设计的"

**澄清**: 嵌入矩阵中的值完全是**通过训练学到的**. 在训练开始时,
通常用小的随机数初始化 (如高斯分布 $\mathcal{N}(0, 0.02)$),
然后通过反向传播逐渐调整到有意义的值.

### 误解 2: "嵌入维度应该越大越好"

**澄清**: 增加维度有收益递减效应. 从 128 到 256 维的提升远大于从 4096 到 8192.
而且更大的维度意味着更多参数, 可能导致过拟合和更慢的推理速度.
选择合适的维度是工程上的权衡.

### 误解 3: "相同的词在不同模型中有相同的嵌入向量"

**澄清**: 每个模型有自己的嵌入空间. GPT-4 的"猫"向量和 Qwen2-VL 的"猫"向量
完全不同 — — 它们生活在不同的空间中, 维度甚至都不同.

### 误解 4: "嵌入层需要很多计算"

**澄清**: 嵌入层是整个模型中**计算最少**的部分之一.
它只是数组索引 ($O(L \times d)$), 不涉及矩阵乘法.
真正的计算瓶颈在 Attention 和 MLP 层.

### 误解 5: "Token Embedding 输出就是 token 的最终表示"

**澄清**: Token Embedding 只提供**初始表示**. 经过多层 Transformer 后,
每个位置的向量会融合大量上下文信息, 变得完全不同于初始嵌入.
"bank" 在 "river bank" 和 "bank account" 中经过 Attention 后会有不同的表示,
但在 Embedding 层输出时是相同的.

---

## 12. NumPy 实现与逐行解析

```python
import numpy as np

def token_embedding(token_ids: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """词嵌入：查找表索引。

    这个函数的实现只有一行——但它承载了从独热编码到分布式表示的
    整个演进历史，以及 NLP 从规则系统到神经网络的范式转变。

    Args:
        token_ids: 整数 token ID，任意形状 (*)
                   在 Qwen2-VL 中通常为 (1, L)，L 是序列长度
                   值域: [0, V-1]，即 [0, 151935]
        weight:    (V, d) 嵌入矩阵
                   在 Qwen2-VL 中为 (151936, 1536)
                   权重键名: "model.embed_tokens.weight"

    Returns:
        (*, d) 嵌入向量
        在 Qwen2-VL 中为 (1, L, 1536)

    数学等价于:
        one_hot(token_ids) @ weight
    但直接索引比矩阵乘法快 V 倍（V = 151936）
    """
    return weight[token_ids]
```

**为什么 NumPy 的索引操作就够了? **

`weight[token_ids]` 利用了 NumPy 的 **花式索引** (fancy indexing).
当 `token_ids` 是一个整数数组时, NumPy 会沿第 0 轴取出对应的行,
自动处理批量维度和广播.

对于 `token_ids` 形状为 `(1, 3602)` 的情况:

- NumPy 取出 `weight` 中第 `token_ids[0,0]`, `token_ids[0,1]`,..., `token_ids[0,3601]` 行
- 输出形状自动变为 `(1, 3602, 1536)`

**精度考虑**:
嵌入查找是精确的 — — 没有浮点运算, 只有内存拷贝.
不存在累加误差, 舍入误差等问题.
因此在验证中, 我们使用较宽松的容差 `atol=1e-4` (主要是因为存储的激活值
可能经历了 float32 ↔ bfloat16 的转换).

---

## 13. 连接到更大的图景

Token Embedding 是 Qwen2-VL 文本管线的**第一步**:

```
用户输入文本
    │
    ▼
分词器 (BPE Tokenizer)
    │  "你好世界" → [57668, 30709, ...]
    ▼
token_ids (1, L)
    │
    ▼
┌─────────────────────────────────┐
│  Token Embedding (算子 14)      │  ← 你在这里
│  (1, L) → (1, L, 1536)         │
└─────────────────────────────────┘
    │
    ▼
视觉 Token 替换
    │  image_positions 处的向量被替换为视觉编码器输出
    ▼
28 × Decoder Layer (算子 17)
  ├── RMSNorm (算子 04)
  ├── GQA Attention (算子 07)
  ├── Residual Connection (算子 13)
  ├── RMSNorm (算子 04)
  ├── Gated MLP / SwiGLU (算子 12)
  └── Residual Connection (算子 13)
    │
    ▼
Final RMSNorm (算子 04)
    │
    ▼
LM Head (算子 18)
  (1, L, 1536) → (1, L, 151936)
  使用的是 E^T（与嵌入矩阵共享！）
    │
    ▼
Softmax → 下一个 token 的概率分布
```

嵌入层和 LM Head 就像一对**编码器-解码器**:

- 嵌入层: 离散 token → 连续向量 (进入 Transformer 的"入口")
- LM Head: 连续向量 → 离散 token 概率 (Transformer 的"出口")

它们共享同一个矩阵 $\mathbf{E}$, 形成了一个优雅的对称结构.

---

## 14. 扩展阅读

- **Word2Vec**: Mikolov et al., _"Efficient Estimation of Word Representations in Vector Space"_, 2013
- **GloVe**: Pennington et al., _"GloVe: Global Vectors for Word Representation"_, EMNLP 2014
- **Weight Tying**: Press & Wolf, _"Using the Output Embedding to Improve Language Models"_, EACL 2017
- **BPE 分词**: Sennrich et al., _"Neural Machine Translation of Rare Words with Subword Units"_, ACL 2016
- **SentencePiece**: Kudo & Richardson, _"SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing"_, EMNLP 2018
- **Qwen2-VL 技术报告**: Wang et al., _"Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution"_, 2024

---

## 验证

运行以下命令验证我们的 NumPy 实现与模型实际输出的一致性:

```bash
python -m operators.14_token_embedding.impl
```

该脚本加载 Qwen2-VL-2B-Instruct 的嵌入权重 (151936 × 1536),
对 3602 个 token 进行嵌入查找, 并与 PyTorch 的 `nn.Embedding` 输出逐元素比较.
