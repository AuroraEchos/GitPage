---
date: 2026-08-01
category: llm
title: Qwen3-VL
description: 梳理 Qwen3-VL 的视觉编码、DeepStack、多模态位置编码与主干数据流。
---

# Qwen3-VL 笔记
## 1. 基础介绍

Qwen3-VL 是阿里巴巴 Qwen 团队于 2025 年 9 月开始发布的视觉语言模型系列，支持文本、图片、视频和交错多模态上下文。技术报告覆盖 2B、4B、8B、32B Dense 版本，以及 30B-A3B、235B-A22B MoE 版本，原生上下文长度为 256K。官方仓库还说明可以通过扩展配置达到 1M，但这不等同于所有部署默认支持 1M。

Qwen3-VL 家族分为稠密 Dense、MoE 稀疏混合专家两大系列，每个规格提供 Instruct 、Thinking 两个版本：

| 模型规格               | 定位场景                              | 视觉编码器                 |
| ---------------------- | ------------------------------------- | -------------------------- |
| Qwen3-VL-2B / 4B       | 手机 / 嵌入式端侧 Agent、GUI 自动化   | SigLIP-2 300M 轻量编码器   |
| Qwen3-VL-8B / 32B      | 通用服务器多模态、图文推理、OCR       | SigLIP-2 400M 高性能编码器 |
| Qwen3-VL-30B-A3B MoE   | 平衡算力与性能，云端批量推理          | SigLIP-2 400M              |
| Qwen3-VL-235B-A22B MoE | 旗舰通用多模态，复杂 STEM、长视频推理 | SigLIP-2 400M              |

## 2. 核心架构

Qwen3-VL 的核心可以概括为：

- 动态分辨率 SigLIP-2 视觉编码器
- MLP Vision-Language Merger
- Qwen3 Decoder-only LLM
- DeepStack 多层视觉注入
- Interleaved MRoPE 多模态位置编码器

![](Qwen3-VL.jpeg)

早期的视觉模型例如 Qwen-VL 在结构上是在视觉编码器后接 Query Transformer/Cross-Attention Adapter，而 Qwen3-VL 把连续的视觉特征转换成与文本 embedding 同维度的 visual tokens，直接放入语言模型上下文中，再由统一的自回归 Decoder 对文本、图像和视频进行建模。

技术报告将整体设计概括为：视觉编码器把动态原生分辨率输入映射为可变长度视觉 token；DeepStack 把 ViT 多个中间层的特征注入对应 LLM 层；Interleaved MRoPE 为文本、图像和视频分配多维位置频率；视频时间则通过文本时间戳对齐。DeepStack 的目标是保留多层视觉信息，但“完整保留所有细节”仍是过强表述。

官方把它定义为三个主要模块：Vision Encoder、MLP-based Vision-Language Merger 和 Qwen3 LLM；DeepStack、Interleaved MRoPE 和视频文本时间戳是在这个基础上的关键增强。

------

## 3. Qwen3-VL 主干链路数据流模拟

我们构造一个足够小的输入，沿 Hugging Face Transformers 中 `Qwen3VLModel.forward()` 的主要路径走一遍。`main` 分支会持续变化，阅读时应同时核对所安装 Transformers 版本的源码：[modeling_qwen3_vl.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py)。

### 固定一个模拟输入

为了让数字容易手算，我们使用：

- Batch size B = 1
- LLM hidden size D = 4096
- 空间合并系数 spatial_merge_size = 2
- 图像网格 image_grid_thw = [1, 4, 6]
- DeepStack 层数 K = 3

这里 T=1，H=4，W=6，表示图像经过 Patch Embedding 后，在视觉编码器中对应：
$$
T \times H \times W = 1 \times 4 \times 6 = 24
$$
个原始视觉 Patch。

经过 `spatial_merge_size=2` 的 $2\times2$ 空间合并后，进入 LLM 的视觉 Token 数为：
$$
N_{vision} = \frac{T \times H \times W}{2^2} = \frac{24}{4} = 6
$$
所以，我们的文本输入里必须恰好有 6 个图像占位 Token。

### 构造输入 Token 序列

我们构造如下序列：

```
位置  Token                 模态类型
0     <bos>                 text
1     用户问题               text
2     <vision_start>        text

3     <image_pad>           image
4     <image_pad>           image
5     <image_pad>           image
6     <image_pad>           image
7     <image_pad>           image
8     <image_pad>           image

9     <vision_end>          text
10    这                    text
11    是什么                 text
12    ?                     text
```

因此总序列长度：$L=13$

对应的 `input_ids`：$[1, 13]$

为了方便理解，可以用符号表示：

```
input_ids = [
    [
        T0, T1, T2,
        IMG, IMG, IMG, IMG, IMG, IMG,
        T3, T4, T5, T6,
    ]
]
```

### 视觉特征与占位符替换

`pixel_values` 先进入 SigLIP-2 架构的视觉编码器。示例网格包含 24 个 patch；主 merger 按 $2\times2$ 空间块合并并投影到 LLM hidden size，得到：

```text
image_embeds: [6, 4096]
```

视觉编码器还从三个选定的中间层各产生一组 DeepStack 特征；经过各自 merger 后，每组同样对应 6 个视觉位置：

```text
deepstack_visual_embeds: 3 × [6, 4096]
```

模型先把全部 `input_ids` 映射为 `[1, 13, 4096]` 的 token embeddings，再检查 6 个 `<image_pad>` 是否与 6 个主视觉 token 数量一致。通过检查后，使用 image mask 把这些占位符 embedding 替换为 `image_embeds`。占位符本身只是确定视觉特征写入序列的位置，并不是视觉内容。

### Interleaved MRoPE 与 DeepStack

处理器还需要提供图像/视频的模态类型信息。模型据此生成三轴 position IDs；文本位置三轴相同，图像位置则编码合并后网格的高和宽，视频还会包含时间轴。具体张量布局随 Transformers 版本演进，不能只靠 `input_ids` 推出正确的多模态位置。

替换主视觉 embedding 后，统一序列进入 Qwen3 LLM。到达配置指定的前几个层时，模型只在视觉位置上把对应的 DeepStack 特征加到 hidden states：

```text
hidden_states[visual_positions] += deepstack_visual_embeds[layer_index]
```

因此 DeepStack 不是把额外 token 追加到上下文末尾，也不会把序列长度从 13 变大；它是在若干 LLM 层对同一批视觉位置做跨层特征注入。

最后，Causal LM 输出每个序列位置的词表 logits。生成时仍从最后一个有效位置采样下一个文本 token，并使用 KV Cache 增量解码。

参考资料：

1. [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631)
2. [QwenLM/Qwen3-VL 官方仓库](https://github.com/QwenLM/Qwen3-VL)
