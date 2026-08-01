# Qwen3-VL 笔记
### 1. 基础介绍

Qwen3-VL 是阿里云通义千问团队于 2025 年 9 月份推出的新一代开源视觉语言多模态大模型，这是迄今为止 Qwen 系列中最强大的视觉语言模型，完整支持图片、短视频、长视频、混合图文输入，兼顾端侧轻量化部署与云端高性能推理，包含 2B、4B、8B、32B Dense 版本，以及 30B-A3B、235B-A22B MoE 版本，原生上下文长度为 256K，arxiv 链接：https://arxiv.org/pdf/2511.21631。

Qwen3-VL 家族分为稠密 Dense、MoE 稀疏混合专家两大系列，每个规格提供 Instruct 、Thinking 两个版本：

| 模型规格               | 定位场景                              | 视觉编码器                 |
| ---------------------- | ------------------------------------- | -------------------------- |
| Qwen3-VL-2B / 4B       | 手机 / 嵌入式端侧 Agent、GUI 自动化   | SigLIP-2 300M 轻量编码器   |
| Qwen3-VL-8B / 32B      | 通用服务器多模态、图文推理、OCR       | SigLIP-2 400M 高性能编码器 |
| Qwen3-VL-30B-A3B MoE   | 平衡算力与性能，云端批量推理          | SigLIP-2 400M              |
| Qwen3-VL-235B-A22B MoE | 旗舰通用多模态，复杂 STEM、长视频推理 | SigLIP-2 400M              |

### 2. 核心架构

Qwen3-VL 的核心可以概括为：

- 动态分辨率 SigLIP-2 视觉编码器
- MLP Vision-Language Merger
- Qwen3 Decoder-only LLM
- DeepStack 多层视觉注入
- Interleaved MRoPE 多模态位置编码器

![](Qwen3-VL.jpeg)

早期的视觉模型例如 Qwen-VL 在结构上是在视觉编码器后接 Query Transformer/Cross-Attention Adapter，而 Qwen3-VL 把连续的视觉特征转换成与文本 embedding 同维度的 visual tokens，直接放入语言模型上下文中，再由统一的自回归 Decoder 对文本、图像和视频进行建模。

技术报告中对上述框架的原始描述如下：

>The Qwen3-VL framework integrates a vision encoder and a language model decoder to process
>multimodal inputs, including text, images, and video. The vision encoder is specifically designed to
>handle dynamic, native-resolution visual inputs, mapping them to visual tokens of variable length.
>To enhance perceptual capability and preserve rich visual information, we incorporate the pioneering
>DeepStack mechanism, which injects visual tokens from multiple layers of the vision encoder into
>corresponding layers of the LLM. Furthermore, we adopt Interleaved MRoPE to encode positional
>information for multimodal inputs with a balanced frequency spectrum, and introduce text-based
>timestamp tokens to more effectively capture the temporal structure of video sequences.

Qwen3-VL 整体框架融合视觉编码器与大语言模型解码器，用于处理文本、图像、视频等多模态输入。该视觉编码器经过专门优化，可处理任意原生分辨率的动态视觉输入，并将像素信息映射为长度可变的视觉 Token。为提升模型视觉感知能力、完整保留丰富视觉细节，本文提出创新的 DeepStack 多层视觉融合机制：将视觉编码器不同层级输出的视觉特征 Token，分层注入大语言模型的对应解码层。除此之外，模型采用交错式多维旋转位置编码（Interleaved MRoPE），以均衡频谱分布对多模态输入完成位置信息编码；同时引入文本式时间戳 Token，能够更高效地捕捉视频序列的时序逻辑结构。

官方把它定义为三个主要模块：Vision Encoder、MLP-based Vision-Language Merger 和 Qwen3 LLM；DeepStack、Interleaved MRoPE 和视频文本时间戳是在这个基础上的关键增强。

------

### 3. Qwen3-VL 主干链路数据流模拟

我们构造一个足够小、但覆盖完整多模态路径的输入，严格沿着这段 `Qwen3VLModel.forward()` 走一遍。代码来自于：https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py 。

#### 固定一个模拟输入

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

#### 构造输入 Token 序列

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

