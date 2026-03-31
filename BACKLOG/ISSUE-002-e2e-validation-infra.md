## Parent PRD

[PRD-001-qwen2vl-operator-math.md](PRD-001-qwen2vl-operator-math.md)

## What to build

搭建端到端验证基础设施：用 PyTorch hook 对 Qwen2-VL-2B-Instruct 跑一次单图 inference，dump 每一层/算子的输入输出 tensor 为 `.npy` 文件；编写通用的 `np.allclose` 验证工具供后续所有算子 `impl.py` 调用。

具体包括：

- `e2e/dump_activations.py`：注册 forward hook，dump 中间激活值（bfloat16 先转 float32），重复层只 dump 第 0 层
- `e2e/validate.py`：封装 `np.allclose` 对比逻辑，打印 pass/fail、max abs error、mean abs error

## Acceptance criteria

- [ ] `dump_activations.py` 能成功运行，对 demo.jpeg 执行一次 inference 并生成 `.npy` 文件
- [ ] dump 文件按 `{module_path}_{input|output}.npy` 命名，覆盖 vision encoder 和 text decoder 的关键层
- [ ] 重复层（vision block、decoder layer）只 dump 第 0 层
- [ ] `validate.py` 提供 `validate(name, actual, expected, atol, rtol)` 函数，打印对比结果
- [ ] 所有 dump 的 tensor 为 float32 精度

## Blocked by

None - can start immediately

## User stories addressed

- User story 7
- User story 8
- User story 12
