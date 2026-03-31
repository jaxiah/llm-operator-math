## Parent PRD

[PRD-001-qwen2vl-operator-math.md](PRD-001-qwen2vl-operator-math.md)

## What to build

实现两种归一化算子的数学讲解和 numpy 实现：

- **LayerNorm** (`03_layer_norm/`)：标准 Layer Normalization（vision path 使用）
- **RMSNorm** (`04_rms_norm/`)：Root Mean Square Normalization（text path 使用），附带与 LayerNorm 的对比

先讲通用归一化概念，再分别说明两者的差异和各自在 vision/text path 中的用法。

## Acceptance criteria

- [ ] `03_layer_norm/README.md` 从均值/方差基础讲起，推导 LayerNorm 公式，含手算示例
- [ ] `03_layer_norm/impl.py` 加载 vision encoder 中的 LayerNorm 权重，验证通过
- [ ] `04_rms_norm/README.md` 讲解 RMSNorm 与 LayerNorm 的异同
- [ ] `04_rms_norm/impl.py` 加载 text decoder 中的 RMSNorm 权重，验证通过
- [ ] 所有 impl.py 可独立运行

## Blocked by

- Blocked by [ISSUE-002-e2e-validation-infra.md](ISSUE-002-e2e-validation-infra.md)

## User stories addressed

- User story 2
- User story 3
- User story 4
- User story 5
- User story 6
- User story 10
- User story 11
