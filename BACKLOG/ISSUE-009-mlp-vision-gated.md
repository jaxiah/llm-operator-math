## Parent PRD

[PRD-001-qwen2vl-operator-math.md](PRD-001-qwen2vl-operator-math.md)

## What to build

实现两种 MLP 组合算子的数学讲解和 numpy 实现：

- **Vision MLP** (`11_vision_mlp/`)：FC1 → QuickGELU → FC2
- **Gated MLP / SwiGLU** (`12_gated_mlp/`)：gate_proj 和 up_proj 并行，SiLU 门控，down_proj 投影回原维度

先讲通用前馈网络概念，再分别讲解 vision path 的标准 MLP 和 text path 的 gated MLP (SwiGLU) 变体。

## Acceptance criteria

- [ ] `11_vision_mlp/README.md` 讲解两层 MLP 的数学，嵌入 numpy 代码片段
- [ ] `11_vision_mlp/impl.py` 组合 linear + quickgelu + linear，用 vision block 0 的 dump 验证通过
- [ ] `12_gated_mlp/README.md` 讲解 SwiGLU 门控机制的数学动机
- [ ] `12_gated_mlp/impl.py` 组合 gate/up/down projection + silu，用 decoder layer 0 的 dump 验证通过
- [ ] 所有 impl.py 可独立运行

## Blocked by

- Blocked by [ISSUE-003-basic-ops-linear-softmax-residual.md](ISSUE-003-basic-ops-linear-softmax-residual.md)
- Blocked by [ISSUE-005-activations-quickgelu-gelu-silu.md](ISSUE-005-activations-quickgelu-gelu-silu.md)

## User stories addressed

- User story 2
- User story 3
- User story 4
- User story 5
- User story 6
- User story 10
- User story 11
- User story 12
