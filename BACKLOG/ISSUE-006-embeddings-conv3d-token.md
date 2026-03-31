## Parent PRD

[PRD-001-qwen2vl-operator-math.md](PRD-001-qwen2vl-operator-math.md)

## What to build

实现两种嵌入算子的数学讲解和 numpy 实现：

- **Conv3d Patch Embedding** (`05_conv3d_patch_embed/`)：3D 卷积将图像像素切成 patch 并投影到 embedding 空间
- **Token Embedding** (`14_token_embedding/`)：查表操作将 token id 映射为向量，以及 vision token 替换机制

## Acceptance criteria

- [ ] `05_conv3d_patch_embed/README.md` 从 2D 卷积基础讲起，扩展到 3D，讲解 stride=kernel_size 的非重叠 patch 切分
- [ ] `05_conv3d_patch_embed/impl.py` 加载 vision encoder 的 patch_embed 权重，输入真实图像 tensor，验证通过
- [ ] `14_token_embedding/README.md` 讲解 embedding 查表原理，以及 vision/text token 的合并方式
- [ ] `14_token_embedding/impl.py` 加载 embedding 权重，验证通过
- [ ] 所有 impl.py 可独立运行

## Blocked by

- Blocked by [ISSUE-002-e2e-validation-infra.md](ISSUE-002-e2e-validation-infra.md)
- Blocked by [ISSUE-003-basic-ops-linear-softmax-residual.md](ISSUE-003-basic-ops-linear-softmax-residual.md)

## User stories addressed

- User story 2
- User story 3
- User story 4
- User story 5
- User story 6
- User story 13
- User story 14
