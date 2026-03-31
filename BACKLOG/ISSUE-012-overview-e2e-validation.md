## Parent PRD

[PRD-001-qwen2vl-operator-math.md](PRD-001-qwen2vl-operator-math.md)

## What to build

编写全局 overview 文档和 E2E numpy 全链路验证：

- **Overview** (`00_overview/README.md`)：完整的算子列表、执行顺序、架构示意图、各算子目录的导航索引
- **E2E Runner** (`e2e/run_numpy_e2e.py`)：按执行顺序串联所有 numpy 算子实现，从图像像素和 token ids 出发，逐层运行并与 dump 对比，输出完整的验证报告

## Acceptance criteria

- [ ] `00_overview/README.md` 包含完整的 Qwen2-VL 架构图（文字版）和算子执行顺序
- [ ] 包含每个算子目录的链接索引
- [ ] `e2e/run_numpy_e2e.py` 按顺序调用所有算子的 numpy 实现
- [ ] E2E runner 对每一步打印验证结果，最终输出整体 pass/fail 汇总
- [ ] Vision path（Conv3d → Vision Block 0 → Patch Merger）全链路验证通过
- [ ] Text path（Token Embedding → Decoder Layer 0 → Final Norm → LM Head）全链路验证通过

## Blocked by

- Blocked by [ISSUE-010-vision-path-assembly.md](ISSUE-010-vision-path-assembly.md)
- Blocked by [ISSUE-011-text-path-assembly.md](ISSUE-011-text-path-assembly.md)

## User stories addressed

- User story 1
- User story 9
- User story 15
