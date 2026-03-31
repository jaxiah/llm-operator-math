## Parent PRD

[PRD-001-qwen2vl-operator-math.md](PRD-001-qwen2vl-operator-math.md)

## What to build

将 text path 的底层算子组装为完整的 text decoder pipeline：

- **Decoder Layer** (`17_decoder_layer/`)：RMSNorm → GQA → Residual → RMSNorm → Gated MLP → Residual
- **LM Head** (`18_lm_head/`)：Final RMSNorm → Linear 投影到 vocab_size → logits

讲解各算子如何串联，数据如何从 token embeddings（含 vision token 替换）经过 N 个 Decoder Layer → Final RMSNorm → LM Head 变成 logits。

## Acceptance criteria

- [ ] `17_decoder_layer/README.md` 讲解 decoder layer 的组装逻辑和数据流
- [ ] `17_decoder_layer/impl.py` 组装底层算子，用 decoder layer 0 的输入/输出 dump 验证通过
- [ ] `18_lm_head/README.md` 讲解最终投影到词表空间的原理
- [ ] `18_lm_head/impl.py` 用 final norm + lm_head 的 dump 验证通过
- [ ] 所有 impl.py 可独立运行

## Blocked by

- Blocked by [ISSUE-004-normalization-layernorm-rmsnorm.md](ISSUE-004-normalization-layernorm-rmsnorm.md)
- Blocked by [ISSUE-006-embeddings-conv3d-token.md](ISSUE-006-embeddings-conv3d-token.md)
- Blocked by [ISSUE-008-attention-sdpa-mha-gqa.md](ISSUE-008-attention-sdpa-mha-gqa.md)
- Blocked by [ISSUE-009-mlp-vision-gated.md](ISSUE-009-mlp-vision-gated.md)

## User stories addressed

- User story 1
- User story 9
- User story 14
- User story 15
