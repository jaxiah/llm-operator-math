## Parent PRD

[PRD-001-qwen2vl-operator-math.md](PRD-001-qwen2vl-operator-math.md)

## What to build

实现三种激活函数的数学讲解和 numpy 实现：

- **QuickGELU** (`08_quickgelu/`)：Vision MLP 使用，$\text{QuickGELU}(x) = x \cdot \sigma(1.702x)$
- **GELU** (`09_gelu/`)：Patch Merger 使用
- **SiLU** (`10_silu/`)：Text Gated MLP 使用，$\text{SiLU}(x) = x \cdot \sigma(x)$

先讲通用 sigmoid 基础，再分别推导三种变体，对比它们的区别和各自的使用场景。

## Acceptance criteria

- [ ] `08_quickgelu/README.md` 从 sigmoid 讲起，推导 QuickGELU，含函数图像描述和数值示例
- [ ] `08_quickgelu/impl.py` 验证通过
- [ ] `09_gelu/README.md` 讲解 GELU 的概率论动机和近似公式
- [ ] `09_gelu/impl.py` 验证通过
- [ ] `10_silu/README.md` 讲解 SiLU (Swish) 及其与 sigmoid 的关系
- [ ] `10_silu/impl.py` 验证通过
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
