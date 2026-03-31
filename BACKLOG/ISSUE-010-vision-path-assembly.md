## Parent PRD

[PRD-001-qwen2vl-operator-math.md](PRD-001-qwen2vl-operator-math.md)

## What to build

将 vision path 的底层算子组装为完整的 vision encoder pipeline：

- **Vision Block** (`16_vision_block/`)：LayerNorm → Vision MHA → Residual → LayerNorm → Vision MLP → Residual
- **Patch Merger** (`15_patch_merger/`)：spatial reshape → LayerNorm → Linear → GELU → Linear

讲解各算子如何串联，数据如何从图像像素经过 Conv3d → N 个 Vision Block → Patch Merger 变成可以喂给 text decoder 的 vision token embeddings。

## Acceptance criteria

- [ ] `16_vision_block/README.md` 讲解 vision block 的组装逻辑和数据流
- [ ] `16_vision_block/impl.py` 组装底层算子，用 vision block 0 的输入/输出 dump 验证通过
- [ ] `15_patch_merger/README.md` 讲解 spatial merge 的 reshape 逻辑和 MLP
- [ ] `15_patch_merger/impl.py` 用 patch merger 的 dump 验证通过
- [ ] 所有 impl.py 可独立运行

## Blocked by

- Blocked by [ISSUE-004-normalization-layernorm-rmsnorm.md](ISSUE-004-normalization-layernorm-rmsnorm.md)
- Blocked by [ISSUE-006-embeddings-conv3d-token.md](ISSUE-006-embeddings-conv3d-token.md)
- Blocked by [ISSUE-008-attention-sdpa-mha-gqa.md](ISSUE-008-attention-sdpa-mha-gqa.md)
- Blocked by [ISSUE-009-mlp-vision-gated.md](ISSUE-009-mlp-vision-gated.md)

## User stories addressed

- User story 1
- User story 9
- User story 13
- User story 15
