## Parent PRD

[PRD-001-qwen2vl-operator-math.md](PRD-001-qwen2vl-operator-math.md)

## What to build

实现注意力机制的数学讲解和 numpy 实现：

- **Scaled Dot-Product Attention** (`07_attention/`)：通用注意力计算 $\text{softmax}(QK^T / \sqrt{d_k})V$
- **Vision Multi-Head Attention**：无 causal mask，所有 patch 互相 attend
- **Text Grouped Query Attention (GQA)**：带 causal mask，KV heads 少于 Q heads（num_key_value_heads < num_attention_heads）

先讲通用 SDPA，再分别讲 vision MHA（含 2D RoPE 应用）和 text GQA（含 3D RoPE 应用、KV head 重复扩展）。

## Acceptance criteria

- [ ] `07_attention/README.md` 从 Q/K/V 的直觉讲起，推导 scaled dot-product attention
- [ ] 包含 MHA 到 GQA 的渐进讲解，解释 KV head grouping
- [ ] `07_attention/impl.py` 分别实现 vision MHA 和 text GQA
- [ ] Vision attention 用 vision block 0 的 dump 验证通过
- [ ] Text GQA 用 decoder layer 0 的 dump 验证通过
- [ ] impl.py 可独立运行

## Blocked by

- Blocked by [ISSUE-003-basic-ops-linear-softmax-residual.md](ISSUE-003-basic-ops-linear-softmax-residual.md)
- Blocked by [ISSUE-004-normalization-layernorm-rmsnorm.md](ISSUE-004-normalization-layernorm-rmsnorm.md)
- Blocked by [ISSUE-007-rotary-position-embedding.md](ISSUE-007-rotary-position-embedding.md)

## User stories addressed

- User story 2
- User story 3
- User story 4
- User story 5
- User story 6
- User story 10
- User story 13
- User story 14
