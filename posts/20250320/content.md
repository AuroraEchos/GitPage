### *Attention Is All You Need*

*Attention Is All You Need* 是2017年由谷歌大脑（Google Brain）团队发表的一篇里程碑式论文。这篇论文彻底颠覆了传统序列转换（Seq2Seq）任务的技术范式，首次提出了完全不依赖循环神经网络（RNN）或卷积神经网络（CNN）、仅基于注意力机制（Attention Mechanism）的深度学习模型——Transformer。它不仅在机器翻译等核心自然语言处理（NLP）任务上取得了突破性性能，更重新定义了深度学习的技术路线，成为NLP、计算机视觉（CV）、语音识别等多领域的基础架构，直接催生了BERT、GPT、Vision Transformer（ViT）等革命性模型，被誉为“深度学习领域的分水岭”。

### 一、论文的核心背景：传统模型的痛点与突破动机

在Transformer出现之前，序列转换任务（如机器翻译、文本摘要、对话生成等）的主流解决方案依赖**Seq2Seq编码器-解码器架构**，并以RNN（含LSTM、GRU）或CNN为核心组件，同时辅以简单的注意力机制。但这些传统模型存在难以克服的缺陷：

#### 1. 循环神经网络（RNN/LSTM/GRU）的局限性

- **并行性差**：RNN需按序列顺序逐词处理（前一个词的输出作为后一个词的输入），无法对序列中的多个位置进行并行计算，导致训练效率极低，难以处理长文本；

- **长距离依赖建模困难**：尽管LSTM和GRU通过门控机制缓解了“梯度消失/梯度爆炸”问题，但仍无法高效捕捉序列中远距离位置的依赖关系（如长句中首尾词的语义关联），随着序列长度增加，依赖传递的衰减问题依然严重。

#### 2. 卷积神经网络（CNN）的局限性

- 以ByteNet、ConvS2S为代表的CNN-based模型通过卷积核并行处理局部窗口，提升了并行性，但存在两个关键问题：

    - 建模长距离依赖需通过“堆叠多层卷积”或“扩大卷积核感受野”实现，导致计算复杂度随位置间距呈线性或对数增长，效率不高；

    - 卷积的“局部性”本质使其难以直接捕捉全局范围内的位置关联，需通过复杂设计间接弥补。

#### 3. 传统注意力机制的不足

- 早期注意力机制（如Bahdanau Attention、Luong Attention）仅作为RNN/CNN的“辅助组件”，用于增强解码器对编码器输出的选择性关注，并未成为模型的核心驱动力；

- 其设计仍受限于底层RNN/CNN的结构约束，无法充分发挥“全局依赖建模”的潜力。

正是在这样的背景下，谷歌团队提出了一个大胆的设想：**能否彻底抛弃RNN和CNN，让注意力机制成为模型的唯一核心**？Transformer的诞生，正是对这一设想的成功验证——它通过“自注意力（Self-Attention）”机制实现了全局依赖的直接建模，同时通过模块化设计最大化并行计算效率，一举解决了传统模型的核心痛点。

### 二、Transformer的核心创新：架构设计与关键组件

Transformer继承了Seq2Seq的“编码器-解码器”整体框架，但内部结构完全基于注意力机制和Feed-Forward网络，核心创新集中在**自注意力机制、多头注意力、位置编码、残差连接与层归一化**等模块，整体架构清晰且极具扩展性。

#### 1. 整体架构概览

Transformer的架构分为**编码器（Encoder）** 和**解码器（Decoder）** 两部分，二者均由多个相同的层堆叠而成（论文中默认堆叠6层编码器和6层解码器）：

- **编码器**：接收输入序列（如机器翻译中的源语言文本），将其转换为包含全局语义信息的连续向量表示（称为“上下文向量”）；

- **解码器**：以编码器的输出为条件，通过自回归方式（逐词生成，前一个生成的词作为当前输入）生成目标序列（如机器翻译中的目标语言文本）。

#### 2. 编码器（Encoder）：全局语义的并行建模

每个编码器层包含两个核心子层，且每个子层后均接入**残差连接（Residual Connection）** 和**层归一化（Layer Normalization）**，确保模型训练的稳定性（缓解梯度消失）：

##### （1）第一子层：多头自注意力（Multi-Head Self-Attention）

这是编码器的核心，也是Transformer最关键的创新。

- **自注意力的本质**：与传统注意力机制不同，自注意力无需依赖外部信息，仅通过序列内部的“查询（Query, Q）、键（Key, K）、值（Value, V）”计算，直接捕捉序列中任意两个位置的依赖关系——无论它们相距多远，计算复杂度均为O(n²)（n为序列长度），且所有位置的计算可并行进行。

    - 计算逻辑：对于输入序列的每个位置i，通过线性变换生成Q_i、K_i、V_i；通过Q_i与所有位置的K_j计算相似度（注意力分数），经Softmax归一化后，加权求和所有位置的V_j，得到位置i的上下文表示。

- **多头注意力的价值**：将自注意力机制并行执行h次（论文中h=8，即8个“头”），每个头使用不同的线性变换矩阵，捕捉不同维度的依赖关系（如语法依赖、语义关联、位置关联等）；最后将所有头的输出拼接，通过线性变换融合，既丰富了特征表达，又提升了模型的泛化能力。

##### （2）第二子层：Position-wise前馈网络（Position-Wise Feed-Forward Network）

对多头注意力的输出进行逐位置的非线性变换，且每个位置的变换独立进行（不依赖其他位置），进一步增强模型的表达能力。

- 结构：包含两层线性变换和一层ReLU激活函数，公式为：FFN(x) = max(0, xW₁ + b₁)W₂ + b₂；

- 维度：输入和输出维度均为512（论文默认），中间隐藏层维度为2048，通过“升维-激活-降维”的过程捕捉复杂的非线性特征。

#### 3. 解码器（Decoder）：自回归生成与双向依赖融合

每个解码器层在编码器的两个子层基础上，新增了一个“编码器-解码器注意力”子层，共三个子层，同样配备残差连接和层归一化：

##### （1）第一子层：掩码多头自注意力（Masked Multi-Head Self-Attention）

与编码器的多头自注意力类似，但增加了“掩码（Mask）”机制——在计算注意力分数时，屏蔽掉所有“未来位置”的信息（即当前位置i只能关注i及之前的位置），确保解码器的自回归属性（生成第i个词时，无法提前看到第i+1个及以后的词）。

- 掩码实现：通过在注意力分数矩阵中，对未来位置的分数填充“-∞”，经Softmax后权重变为0，从而无法获取未来位置的信息。

##### （2）第二子层：编码器-解码器注意力（Encoder-Decoder Attention）

用于融合编码器的全局语义信息与解码器的当前生成状态，实现“输入序列与输出序列的跨序列依赖建模”。

- 计算逻辑：查询（Q）来自解码器前一层的输出（当前生成序列的上下文表示），键（K）和值（V）来自编码器的最终输出（输入序列的全局表示）；通过Q与K的相似度计算，解码器可选择性地关注输入序列中与当前生成词相关的位置（如翻译时，当前生成的中文词对应源语言的哪个英文词）。

##### （3）第三子层：位置-wise前馈网络

与编码器的前馈网络完全一致，对跨注意力的输出进行逐位置非线性变换。

#### 4. 关键辅助组件：解决“位置信息缺失”与“特征映射”问题

由于Transformer完全基于注意力机制，不包含RNN/CNN的序列结构，无法天然感知输入序列的位置信息（如“我爱吃苹果”和“苹果爱吃我”的语义差异仅来自位置），因此需要通过专门设计弥补：

##### （1）位置编码（Positional Encoding）

- 核心思路：为每个位置分配一个固定的位置向量，与输入词的嵌入向量（Embedding）相加，将位置信息注入到输入特征中；

- 实现方式：采用正弦和余弦函数生成位置编码，公式为：

    - PE(pos, 2i) = sin(pos / 10000^(2i/d_model))

    - PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    其中pos为位置索引，i为维度索引，d_model=512（模型输入输出维度）；

- 优势：正弦余弦函数的周期性的特性使其能够天然建模位置之间的相对关系（而非绝对位置），且可灵活扩展到训练时未见过的更长序列（外推能力强）。

##### （2）词嵌入与权重共享

- 词嵌入（Token Embedding）：将离散的词（Token）转换为连续的向量表示，输入序列和输出序列分别通过独立的嵌入层映射到d_model=512维；

- 权重共享：输入嵌入层、输出嵌入层与最终的Softmax层共享权重矩阵，既减少了模型参数数量（降低过拟合风险），又提升了特征表示的一致性（嵌入向量与Softmax的分类权重保持语义对齐）。

### 三、论文的里程碑意义与深远影响

《Attention Is All You Need》的价值不仅在于提出了一个高性能的模型，更在于它彻底改变了深度学习的技术范式，其影响跨越多个领域：

#### 1. 自然语言处理（NLP）的革命

- Transformer成为NLP的“基础底座”：后续几乎所有主流NLP模型（BERT、GPT、T5、RoBERTa等）均基于Transformer架构，仅通过调整编码器/解码器的使用方式（如BERT用编码器做双向预训练，GPT用解码器做自回归预训练）和预训练任务，就实现了在文本分类、问答、命名实体识别、文本生成等全场景的SOTA（State-of-the-Art）性能；

- 推动“预训练-微调”范式的普及：Transformer的并行性和全局依赖建模能力，使其能够高效处理大规模文本数据，催生了以“海量文本预训练+下游任务微调”为核心的NLP工业化方案，大幅降低了特定任务的模型开发成本。

#### 2. 跨领域的迁移与融合

- 计算机视觉（CV）：2020年，谷歌提出Vision Transformer（ViT），将图像分割为固定大小的“图像块（Patch）”，通过Transformer编码器直接建模图像块的全局依赖，首次证明Transformer在图像分类任务上可超越CNN，开启了“Vision Transformer时代”，后续衍生出Swin Transformer、MAE等模型，成为CV领域的主流架构；

- 语音与多模态：Transformer被广泛应用于语音识别、语音合成、图文生成（如DALL·E）、视频理解等任务，成为连接语言、图像、语音等多模态数据的统一框架。

#### 3. 工程效率的飞跃

- 并行计算能力：Transformer的所有核心计算（自注意力、前馈网络）均可并行执行，相比RNN的串行计算，训练速度提升数倍甚至数十倍，使得训练百亿、千亿参数的超大模型成为可能；

- 模块化设计：Transformer的编码器、解码器、注意力头、前馈网络等组件高度模块化，易于扩展和修改，为后续研究者的创新提供了灵活的基础。

### 四、总结

《Attention Is All You Need》以“极简而强大”的设计理念，用注意力机制重构了序列建模的核心逻辑，解决了传统RNN/CNN在并行性和长距离依赖建模上的根本痛点。它不仅是一篇技术论文，更是深度学习领域的“思想启蒙”——证明了“全局依赖建模”和“并行计算”可通过单一机制实现，为后续超大模型的发展奠定了理论和工程基础。如今，Transformer已成为深度学习的“通用语言”，其影响力仍在持续扩散，推动着人工智能向更高效、更通用的方向发展。这篇论文也因此被公认为“近十年最具影响力的AI论文之一”，深刻改变了人工智能的发展轨迹。

#### Code

```python
"""
Transformer in Pytorch.

Components needed:
    - Positional Encoding
    - Self-Attention
    - Multi-Head Attention
    - Feedforward Layer
    - Encoder Layer
    - Decoder Layer
    - Encoder
    - Decoder
    - Transformer Model

Date: 2025-07-04
Author: Wenhao Liu
"""

import torch
import math
from torch import Tensor
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding for Transformer.

    Args:
        d_model: The dimension of the embedding vector.
        max_len: The maximum length of input sequences.

    Purpose:
        Add positional information to token embeddings so that
        the model can take sequence order into account.
    """
    def __init__(
            self,
            d_model: int,
            max_len: int=5000,
    ) -> None:
        super().__init__()

        # Create a matrix of [max_len, d_model] with positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)    # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)    # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Add positional encoding to the input tensor.

        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]

        Returns:
            Tensor of shape [batch_size, seq_len, d_model] with positional encoding added.
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x
    
class ScaledDotProductAttention(nn.Module):
    """
    Implements scaled dot-product attention mechanism.

    Args:
        None
    
    Purpose:
        Computes attention weights and applies them to values.
        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
    """
    
    def __init__(self) -> None:
        super().__init__()

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            mask: Tensor=None
    ) -> Tensor:
        """
        Args:
            query: Tensor of shape [batch_size, n_heads, seq_len_q, d_k]
            key: Tensor of shape [batch_size, n_heads, seq_len_k, d_k]
            value: Tensor of shape [batch_size, n_heads, seq_len_v, d_v]
            mask: (Optional) Tensor of shape [batch_size, 1, 1, seq_len_k] or broadcastable

        Returns:
            output: Tensor of shape [batch_size, n_heads, seq_len_q, d_v]
            attn: Tensor of attention weights [batch_size, n_heads, seq_len_q, seq_len_k]
        """
        d_k = query.size(-1)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)

        output = torch.matmul(attn, value)

        return output, attn
    
class MultiHeadAttention(nn.Module):
    """
    Implements multi-head attention.

    Args:
        d_model: The dimension of input embedding.
        n_heads: The number of attention heads.

    Purpose:
        Allows the model to jointly attend to information
        from different representation subspaces at different positions.
    """
    def __init__(
            self,
            d_model: int,
            n_heads: int
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads

        # Define linear layers for q, k, v
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # Output linear layer
        self.w_o = nn.Linear(d_model, d_model)

        # Scaled dot-product attention module
        self.attention = ScaledDotProductAttention()

    def forward(
            self, 
            query: Tensor, 
            key: Tensor, 
            value: Tensor, 
            mask: Tensor = None
    ) -> Tensor:
        """
        Args:
            query: Tensor of shape [batch_size, seq_len_q, d_model]
            key: Tensor of shape [batch_size, seq_len_k, d_model]
            value: Tensor of shape [batch_size, seq_len_v, d_model]
            mask: (Optional) Tensor for masking attention scores

        Returns:
            output: Tensor of shape [batch_size, seq_len_q, d_model]
        """
        batch_size = query.size(0)

        # Linear projections and split into heads
        q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        # Apply scaled dot-product attention
        out, attn = self.attention(q, k, v, mask)

        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # Final linear projection
        output = self.w_o(out)

        return output, attn
    
class PositionwiseFeedForward(nn.Module):
    """
    Implements the position-wise feedforward layer.

    Args:
        d_model: The dimension of the input embedding.
        d_ff: The dimension of the hidden layer in the feedforward network.
        dropout: Dropout probability applied after each linear layer.

    Purpose:
        Applies two linear transformations with a ReLU activation in between,
        independently at each position.
    """
    def __init__(
            self,
            d_model: int,
            d_ff: int,
            dropout: float=0.1
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]

        Returns:
            Tensor of shape [batch_size, seq_len, d_model]
        """
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class EncoderLayer(nn.Module):
    """
    Implements a single Transformer encoder layer.

    Args:
        d_model: The dimension of the input embedding.
        n_heads: The number of attention heads.
        d_ff: The dimension of the feedforward network.
        dropout: Dropout probability applied after attention and feedforward layers.

    Purpose:
        Consists of:
            - Multi-head self-attention with residual + layer norm
            - Position-wise feedforward network with residual + layer norm
    """
    def __init__(
            self,
            d_model: int,
            n_heads: int,
            d_ff: int,
            dropout: float=0.1
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(
            self, 
            x: Tensor, 
            mask: Tensor = None
    ) -> Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
            mask: (Optional) Mask tensor for attention

        Returns:
            Tensor of shape [batch_size, seq_len, d_model]
        """
        # Self-attention sub-layer
        attn_out, _ = self.self_attn(x, x, x, mask)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)

        # Feedforward sub-layer
        ff_out = self.feed_forward(x)
        x = x + self.dropout2(ff_out)
        x = self.norm2(x)

        return x

class DecoderLayer(nn.Module):
    """
    Implements a single Transformer decoder layer.

    Args:
        d_model: The dimension of the input embedding.
        n_heads: The number of attention heads.
        d_ff: The dimension of the feedforward network.
        dropout: Dropout probability applied after attention and feedforward layers.

    Purpose:
        Consists of:
            - Masked multi-head self-attention with residual + layer norm
            - Multi-head encoder-decoder attention with residual + layer norm
            - Position-wise feedforward network with residual + layer norm
    """
    def __init__(
            self, 
            d_model: int, 
            n_heads: int, 
            d_ff: int, 
            dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(
            self,
            x: Tensor,
            memory: Tensor,
            tgt_mask: Tensor = None,
            memory_mask: Tensor = None
    ) -> Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, tgt_seq_len, d_model] (decoder input)
            memory: Tensor of shape [batch_size, src_seq_len, d_model] (encoder output)
            tgt_mask: (Optional) mask for decoder self-attention
            memory_mask: (Optional) mask for encoder-decoder attention

        Returns:
            Tensor of shape [batch_size, tgt_seq_len, d_model]
        """
        # Masked self-attention sub-layer
        self_attn_out, _ = self.self_attn(x, x, x, tgt_mask)
        x = x + self.dropout1(self_attn_out)
        x = self.norm1(x)

        # Encoder-decoder attention sub-layer
        cross_attn_out, _ = self.cross_attn(x, memory, memory, memory_mask)
        x = x + self.dropout2(cross_attn_out)
        x = self.norm2(x)

        # Feedforward sub-layer
        ff_out = self.feed_forward(x)
        x = x + self.dropout3(ff_out)
        x = self.norm3(x)

        return x
    
class Encoder(nn.Module):
    """
    Implements the Transformer encoder as a stack of encoder layers.

    Args:
        d_model: The dimension of the input embedding.
        n_heads: The number of attention heads.
        d_ff: The dimension of the feedforward network.
        num_layers: The number of encoder layers to stack.
        dropout: Dropout probability.

    Purpose:
        Applies positional encoding and a sequence of encoder layers.
    """
    def __init__(
            self, 
            d_model: int,
            n_heads: int,
            d_ff: int,
            num_layers: int,
            dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
            self, 
            x: Tensor, 
            mask: Tensor = None
    ) -> Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
            mask: (Optional) mask tensor for padding

        Returns:
            Tensor of shape [batch_size, seq_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return x

class Decoder(nn.Module):
    """
    Implements the Transformer decoder as a stack of decoder layers.

    Args:
        d_model: The dimension of the input embedding.
        n_heads: The number of attention heads.
        d_ff: The dimension of the feedforward network.
        num_layers: The number of decoder layers to stack.
        dropout: Dropout probability.

    Purpose:
        Applies positional encoding and a sequence of decoder layers.
    """
    def __init__(
            self, 
            d_model: int,
            n_heads: int,
            d_ff: int,
            num_layers: int,
            dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
            self, 
            x: Tensor, 
            memory: Tensor, 
            tgt_mask: Tensor = None, 
            memory_mask: Tensor = None
    ) -> Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, tgt_seq_len, d_model]
            memory: Tensor of shape [batch_size, src_seq_len, d_model] (encoder output)
            tgt_mask: (Optional) mask tensor for decoder self-attention
            memory_mask: (Optional) mask tensor for encoder-decoder attention

        Returns:
            Tensor of shape [batch_size, tgt_seq_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        x = self.norm(x)
        return x

class Transformer(nn.Module):
    """
    Implements the full Transformer model for sequence-to-sequence tasks.

    Args:
        src_vocab_size: Vocabulary size of the source language.
        tgt_vocab_size: Vocabulary size of the target language.
        d_model: Dimension of the embedding and model.
        n_heads: Number of attention heads.
        d_ff: Dimension of the feedforward network.
        num_encoder_layers: Number of layers in the encoder.
        num_decoder_layers: Number of layers in the decoder.
        dropout: Dropout probability.
        max_len: Maximum sequence length.

    Purpose:
        End-to-end Transformer with embeddings, positional encodings,
        encoder, decoder, and output projection layer.
    """
    def __init__(
            self, 
            src_vocab_size: int,
            tgt_vocab_size: int,
            d_model: int,
            n_heads: int,
            d_ff: int,
            num_encoder_layers: int,
            num_decoder_layers: int,
            dropout: float = 0.1,
            max_len: int = 5000
    ) -> None:
        super().__init__()

        # Embedding layers
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # Encoder and decoder
        self.encoder = Encoder(d_model, n_heads, d_ff, num_encoder_layers, dropout)
        self.decoder = Decoder(d_model, n_heads, d_ff, num_decoder_layers, dropout)

        # Final output projection layer
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)

        # Dropout on embeddings
        self.dropout = nn.Dropout(dropout)

    def forward(
            self, 
            src: Tensor, 
            tgt: Tensor, 
            src_mask: Tensor = None, 
            tgt_mask: Tensor = None, 
            memory_mask: Tensor = None
    ) -> Tensor:
        """
        Args:
            src: Source input tensor of shape [batch_size, src_seq_len]
            tgt: Target input tensor of shape [batch_size, tgt_seq_len]
            src_mask: (Optional) Mask for source sequence
            tgt_mask: (Optional) Mask for target sequence
            memory_mask: (Optional) Mask for encoder-decoder attention

        Returns:
            Logits of shape [batch_size, tgt_seq_len, tgt_vocab_size]
        """
        # Embed and apply positional encoding to source
        src_emb = self.src_embedding(src) * math.sqrt(self.src_embedding.embedding_dim)
        src_emb = self.dropout(self.pos_encoding(src_emb))

        # Embed and apply positional encoding to target
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.tgt_embedding.embedding_dim)
        tgt_emb = self.dropout(self.pos_encoding(tgt_emb))

        # Encoder
        memory = self.encoder(src_emb, src_mask)

        # Decoder
        out = self.decoder(tgt_emb, memory, tgt_mask, memory_mask)

        # Final projection to vocab size
        logits = self.output_layer(out)

        return logits


# Test the Transformer model
if __name__ == "__main__":
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    d_model = 512
    n_heads = 8
    d_ff = 2048
    num_encoder_layers = 6
    num_decoder_layers = 6
    dropout = 0.1
    max_len = 5000

    model = Transformer(
        src_vocab_size, tgt_vocab_size, d_model, n_heads,
        d_ff, num_encoder_layers, num_decoder_layers, dropout, max_len
    )

    print(model)



```

