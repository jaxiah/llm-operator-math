## Parent PRD

[PRD-001-qwen2vl-operator-math.md](PRD-001-qwen2vl-operator-math.md)

## What to build

实现旋转位置编码的数学讲解和 numpy 实现：

- **RoPE 通用原理** (`06_rotary_pos_embed/`)：从复数旋转讲起，推导旋转位置编码的数学基础
- **Vision 2D RoPE**：基于图像 patch 的 height/width 两个维度生成旋转频率
- **Text 3D Multimodal RoPE**：temporal/height/width 三个维度的 mrope_section 通道分割

先讲 1D RoPE 的通用原理，再扩展到 2D（vision）和 3D（text multimodal）。

## Acceptance criteria

- [ ] `06_rotary_pos_embed/README.md` 从复数乘法和旋转矩阵讲起，逐步推导 RoPE
- [ ] 包含 1D → 2D → 3D 的渐进讲解
- [ ] `06_rotary_pos_embed/impl.py` 分别实现 vision 2D RoPE 和 text 3D multimodal RoPE
- [ ] 两种 RoPE 变体分别与 dump 数据验证通过
- [ ] impl.py 可独立运行

## Blocked by

- Blocked by [ISSUE-002-e2e-validation-infra.md](ISSUE-002-e2e-validation-infra.md)
- Blocked by [ISSUE-003-basic-ops-linear-softmax-residual.md](ISSUE-003-basic-ops-linear-softmax-residual.md)

## User stories addressed

- User story 2
- User story 3
- User story 4
- User story 5
- User story 6
- User story 10
