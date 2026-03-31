## Problem Statement

学习和理解 Qwen2-VL 多模态大模型的推理过程，需要将其拆解到每一个细粒度算子，逐一理解数学原理并用最朴素的方式实现验证。目前缺少一份从头到尾、面向初学者的资料，能够把 Qwen2-VL 的完整 inference pipeline 中每个算子的数学细节讲清楚，并且用 numpy 实现来证明理解的正确性。

## Solution

对 Qwen2-VL-2B-Instruct 模型执行一次完整的单图推理，记录所有经过的 layer 和算子。按执行顺序，为每个细粒度算子：

1. 编写 Markdown 笔记，从最基础的数学知识讲起，使用 MathJax 公式和具体数值示例，嵌入 numpy 关键代码片段辅助理解（PyTorch 仅用于 dump ground truth，不出现在教学内容中）
2. 编写独立可运行的 numpy naive 实现（带详细注释），能加载真实模型权重、接受真实中间激活值作为输入，并与 PyTorch 输出做 `np.allclose` 验证

先通用后特化：相同概念的算子（如 attention、norm、linear）先讲通用版本，再分别说明 Vision path 和 Text path 的差异。

E2E 验证策略：先用 PyTorch 模型跑一次完整 inference，dump 每一层的输入/输出 tensor 作为 ground truth；然后 numpy 逐层/逐算子加载 dump 数据进行对比验证。重复性的层（如 28 层 decoder layer）只跑 1 层验证正确性即可。

## User Stories

1. As a learner, I want to see the complete list of operators in Qwen2-VL inference order, so that I know the full picture before diving into details
2. As a learner, I want each operator explained from first principles with MathJax formulas, so that I can understand the math without prior knowledge
3. As a learner, I want concrete numerical examples in each explanation, so that I can trace through the computation by hand
4. As a learner, I want key numpy code snippets embedded in the markdown notes alongside the math, so that I can see exactly how each formula translates to transparent, readable code (no opaque PyTorch ops in teaching material)
5. As a learner, I want a standalone runnable numpy implementation for each operator, so that I can execute and experiment with it independently
6. As a learner, I want the numpy implementation to load real model weights, so that I can verify it produces the same results as PyTorch
7. As a learner, I want an activation dumper that captures every intermediate tensor during a real PyTorch inference pass, so that I have ground truth to compare against
8. As a learner, I want each numpy operator to validate its output against the dumped PyTorch activations, so that I can be confident my understanding is correct
9. As a learner, I want operators organized by execution order with clear numbering, so that I can follow the inference flow sequentially
10. As a learner, I want common concepts (attention, norm, linear) explained generically first with vision/text differences noted separately, so that I can see the shared foundation before the variants
11. As a learner, I want the numpy implementations to be modular and decoupled, so that I can study and run each operator independently without loading the entire pipeline
12. As a learner, I want only one representative layer validated for repeated structures (e.g., decoder layer 0 only), so that validation runs in reasonable time
13. As a learner, I want to understand Vision Encoder operators (Conv3d patch embed, vision attention, QuickGELU, patch merger), so that I know how images become token embeddings
14. As a learner, I want to understand Text Decoder operators (GQA, RMSNorm, SwiGLU MLP, multimodal RoPE), so that I know how token generation works
15. As a learner, I want to understand how vision and text paths merge (vision token replacement in the embedding sequence), so that I see the full multimodal picture

## Implementation Decisions

### Directory Structure

```
operators/
  00_overview/
    README.md              # 完整算子列表、执行顺序、架构图
  01_linear/
    README.md              # 矩阵乘法 + bias 的数学原理
    impl.py                # numpy 实现
  02_softmax/
    README.md
    impl.py
  03_layer_norm/
    README.md              # 通用 LayerNorm，附带 vision path 用法说明
    impl.py
  04_rms_norm/
    README.md              # RMSNorm，附带与 LayerNorm 的对比
    impl.py
  05_conv3d_patch_embed/
    README.md              # 3D 卷积如何把图像切成 patch
    impl.py
  06_rotary_pos_embed/
    README.md              # 通用 RoPE 原理，再分别讲 vision 2D RoPE 和 text 3D multimodal RoPE
    impl.py
  07_attention/
    README.md              # 通用 scaled dot-product attention，再分别讲 vision MHA 和 text GQA
    impl.py
  08_quickgelu/
    README.md              # QuickGELU 激活函数（vision path）
    impl.py
  09_gelu/
    README.md              # GELU 激活函数（patch merger）
    impl.py
  10_silu/
    README.md              # SiLU 激活函数（text path）
    impl.py
  11_vision_mlp/
    README.md              # Vision MLP = Linear + QuickGELU + Linear
    impl.py
  12_gated_mlp/
    README.md              # Gated MLP (SwiGLU) = gate_proj * silu(up_proj) -> down_proj
    impl.py
  13_residual_connection/
    README.md
    impl.py
  14_token_embedding/
    README.md              # 查表嵌入的原理
    impl.py
  15_patch_merger/
    README.md              # spatial merge: reshape + LayerNorm + MLP
    impl.py
  16_vision_block/
    README.md              # 组装: norm -> attention -> residual -> norm -> mlp -> residual
    impl.py
  17_decoder_layer/
    README.md              # 组装: norm -> GQA -> residual -> norm -> gated_mlp -> residual
    impl.py
  18_lm_head/
    README.md              # 最终 linear + argmax/sampling
    impl.py
  e2e/
    dump_activations.py    # 用 PyTorch hook 跑一次 inference，dump 所有中间 tensor 到 .npy
    run_numpy_e2e.py       # 加载 dump，逐算子跑 numpy 实现并对比验证
    validate.py            # np.allclose 封装 + 误差报告工具
```

### 算子执行顺序（单次 inference）

**Vision Encoder:**

1. Conv3d Patch Embedding — 图像像素 → patch embeddings
2. Vision Block × 1 层（代表所有层）:
   - LayerNorm
   - Vision Multi-Head Attention（含 2D RoPE）
   - Residual Connection
   - LayerNorm
   - Vision MLP（Linear + QuickGELU + Linear）
   - Residual Connection
3. Patch Merger（reshape + LayerNorm + Linear + GELU + Linear）

**Text Decoder:**

4. Token Embedding — token ids → embeddings，vision token 替换

5. Decoder Layer × 1 层（代表所有层）:

- RMSNorm
- Grouped Query Attention（含 3D Multimodal RoPE）
- Residual Connection
- RMSNorm
- Gated MLP / SwiGLU（gate_proj, up_proj, SiLU, down_proj）
- Residual Connection

6. Final RMSNorm
7. LM Head（Linear → logits）

### 关键技术决策

- **权重加载**：每个 `impl.py` 通过 `safetensors` 或 `torch.load` 从 HuggingFace 缓存中提取对应层的权重，转为 numpy array
- **激活 dump 格式**：使用 `.npy` 文件，按 `{layer_name}_{input|output}.npy` 命名
- **验证容差**：float32 精度下使用 `np.allclose(atol=1e-5, rtol=1e-5)`；如果模型以 bfloat16 加载，dump 时先转 float32
- **模块解耦**：每个 `impl.py` 只依赖 numpy 和 `e2e/validate.py` 中的工具函数，不依赖其他算子的 `impl.py`。组合算子（vision_block、decoder_layer）通过 import 低层算子实现
- **Python 环境**：统一使用项目根目录的 `.venv`

## Testing Decisions

- 测试即验证：每个 `impl.py` 的 `if __name__ == "__main__"` 块加载对应的 dump 数据，运行 numpy 实现，调用 `e2e/validate.py` 做 `np.allclose` 对比，打印 pass/fail 和最大误差
- `e2e/run_numpy_e2e.py` 按执行顺序串联所有算子，逐步对比，作为端到端集成验证
- 好的测试只验证外部行为（输入 → 输出的数值一致性），不验证实现细节
- 重复层只验证第 0 层（vision block 0, decoder layer 0）

## Out of Scope

- 性能优化（GPU、SIMD、量化等）——本项目只关心数学正确性
- 训练过程（backward pass、梯度计算、优化器）
- Sampling 策略（top-k、top-p、temperature）的详细实现——只到 logits 为止
- KV Cache 机制的实现——只关注单次前向传播
- Flash Attention / SDPA 的优化实现——只实现 naive attention
- Qwen3.5 模型——本 PRD 只覆盖 Qwen2-VL-2B-Instruct
- 视频输入——只覆盖单图输入场景

## Further Notes

- 所有 markdown 笔记应面向完全初学者，不假设读者有深度学习背景
- 数学公式使用 MathJax 语法，确保在 VS Code markdown preview 和 GitHub 上可渲染
- 建议实现顺序：先完成 `e2e/dump_activations.py` 和 `e2e/validate.py`，再从最底层算子（linear, softmax, norm）开始逐步向上构建
- 2B 模型的具体参数需在实现时从 `config.json` 中读取，不要硬编码
