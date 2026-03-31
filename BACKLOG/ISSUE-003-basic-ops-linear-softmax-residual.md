## Parent PRD

[PRD-001-qwen2vl-operator-math.md](PRD-001-qwen2vl-operator-math.md)

## What to build

实现三个最基础的算子的数学讲解和 numpy 实现：

- **Linear** (`01_linear/`)：矩阵乘法 + bias，即 $y = xW^T + b$
- **Softmax** (`02_softmax/`)：指数归一化，含数值稳定性技巧
- **Residual Connection** (`13_residual_connection/`)：跳跃连接，即 $y = x + F(x)$

每个算子包含 README.md（数学原理 + MathJax + 数值示例 + numpy 代码片段）和独立可运行的 impl.py（加载真实权重/激活，与 dump 对比验证）。

## Acceptance criteria

- [ ] `01_linear/README.md` 从矩阵乘法基础讲起，包含 MathJax 公式和手算示例
- [ ] `01_linear/impl.py` 加载真实权重，输入 dump 数据，输出与 PyTorch 一致
- [ ] `02_softmax/README.md` 讲解 softmax 公式、数值稳定性（减最大值技巧）
- [ ] `02_softmax/impl.py` 验证通过
- [ ] `13_residual_connection/README.md` 讲解残差连接的动机和数学
- [ ] `13_residual_connection/impl.py` 验证通过
- [ ] 所有 impl.py 的 `__main__` 块可独立运行，打印 pass/fail

## Blocked by

- Blocked by [ISSUE-002-e2e-validation-infra.md](ISSUE-002-e2e-validation-infra.md)

## User stories addressed

- User story 2
- User story 3
- User story 4
- User story 5
- User story 6
- User story 11
