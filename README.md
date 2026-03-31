# LLM Operator Math

逐算子拆解 **Qwen2-VL-2B-Instruct** 的完整推理流程：每个算子配有从零讲起的数学原理长文（中文）和纯 NumPy 实现，并用真实模型权重与激活值做 `np.allclose` 验证。

## 算子列表

| #   | 算子                                                     | 说明                         |
| --- | -------------------------------------------------------- | ---------------------------- |
| 00  | [Overview](operators/00_overview/)                       | 完整算子列表与架构总览       |
| 01  | [Linear](operators/01_linear/)                           | 线性变换 $y = xW^T + b$      |
| 02  | [Softmax](operators/02_softmax/)                         | Softmax 归一化               |
| 03  | [Layer Norm](operators/03_layer_norm/)                   | 层归一化 (Vision path)       |
| 04  | [RMS Norm](operators/04_rms_norm/)                       | 均方根归一化 (Text path)     |
| 05  | [Conv3d Patch Embed](operators/05_conv3d_patch_embed/)   | 3D 卷积图像分块嵌入          |
| 06  | [Rotary Pos Embed](operators/06_rotary_pos_embed/)       | 旋转位置编码 (RoPE)          |
| 07  | [Attention](operators/07_attention/)                     | 多头注意力 / GQA             |
| 08  | [QuickGELU](operators/08_quickgelu/)                     | QuickGELU 激活函数 (Vision)  |
| 09  | [GELU](operators/09_gelu/)                               | GELU 激活函数 (Patch Merger) |
| 10  | [SiLU](operators/10_silu/)                               | SiLU/Swish 激活函数 (Text)   |
| 11  | [Vision MLP](operators/11_vision_mlp/)                   | 视觉编码器前馈网络           |
| 12  | [Gated MLP](operators/12_gated_mlp/)                     | 门控 MLP / SwiGLU (Text)     |
| 13  | [Residual Connection](operators/13_residual_connection/) | 残差连接                     |
| 14  | [Token Embedding](operators/14_token_embedding/)         | 词嵌入查表                   |
| 15  | [Patch Merger](operators/15_patch_merger/)               | 视觉 token 下采样合并        |
| 16  | [Vision Block](operators/16_vision_block/)               | 视觉编码器完整 Block         |
| 17  | [Decoder Layer](operators/17_decoder_layer/)             | 文本解码器完整 Layer         |
| 18  | [LM Head](operators/18_lm_head/)                         | 语言模型输出头               |

## 快速开始

```bash
# 1. 安装依赖
pip install torch transformers safetensors huggingface_hub qwen-vl-utils

# 2. 导出激活值 (需要 GPU, 约 5 分钟)
python -m e2e.dump_activations --image demo.jpeg

# 3. 运行全量验证
python -m e2e.run_numpy_e2e
```

## 项目结构

```
operators/NN_name/
  README.md    # 数学原理长文 (400-1200 行, 中文)
  impl.py      # 纯 NumPy 实现 + 验证函数
e2e/
  dump_activations.py   # PyTorch hook 导出中间激活值
  validate.py           # 验证工具函数
  run_numpy_e2e.py      # E2E 全量验证 (20/20 PASS)
```

## 工作流复用

如果你想对其他模型做同样的算子拆解，参考 [AGENTS.md](AGENTS.md) 中定义的完整工作流。
