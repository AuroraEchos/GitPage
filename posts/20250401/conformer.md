### Conformer: Convolution-augmented Transformer for Speech Recognition

In recent years, both Transformer and CNN architectures have achieved impressive results in Automatic Speech Recognition (ASR). However, each comes with its own limitations. Transformers excel at modeling **global dependencies**, yet often struggle with capturing **fine-grained local features**. On the other hand, CNNs are highly efficient at extracting **local structural information**, but lack the ability to model **long-range temporal context**.

So, can we **combine the strengths of both architectures** to create a model that possesses both the global modeling capability of Transformers and the local sensitivity of CNNs? The answer is yes — and that’s precisely the motivation behind Google Research’s **Conformer (Convolution-augmented Transformer)** architecture.

#### From RNN to CNN/Transformer: The Evolution of ASR Models

RNNs were once the dominant architecture in ASR due to their natural fit for sequential data. However, with the emergence of Transformers, their superior training efficiency and ability to model long-range dependencies gradually displaced RNNs. Meanwhile, CNNs also gained traction in speech recognition, especially in architectures like Jasper and QuartzNet.

Yet, both Transformers and CNNs **fall short when it comes to simultaneously capturing both local and global structures**. ContextNet attempted to address this by introducing Squeeze-and-Excitation modules to extend CNN receptive fields, yielding some improvement. Nevertheless, it still fell short in modeling truly **dynamic global context**, relying only on global average pooling across the sequence.

Recent research has shown that **combining attention mechanisms with convolutional layers yields better results** than using either in isolation. This fusion — between content-based global modeling (attention) and position-based local modeling (convolution) — appears to be a key to more powerful ASR systems.

#### The Proposal of Conformer: Organic Fusion of Attention and Convolution

The Conformer architecture embodies this insight, seamlessly integrating convolutional modules with multi-head self-attention (MHSA) to improve both **parameter efficiency** and **modeling capacity**.

> 📌 Key Hypothesis: **Local and global interactions are equally important**. By combining both, we can build a more effective model.

The authors adopted a **Macaron-style feed-forward network**, where MHSA and the convolution module are "sandwiched" between two FFN layers — forming a new modular unit called the **Conformer Block**.

#### Conformer Architecture Explained

Let’s break down the data flow through the Conformer from the perspective of audio input processing.

The raw audio signal first passes through a **convolutional subsampling layer**, which reduces the sequence length and lowers the computational burden.

Here’s how each Conformer block is structured:

1. **Feed-Forward Module (FFN)** (with half-step residual)
2. **Multi-Head Self-Attention Module (MHSA)**:
   - Employs **relative positional encoding** from Transformer-XL for better generalization to variable-length inputs.
   - Uses **pre-norm residual connections** with dropout to aid deep model training.
3. **Convolution Module**:
   - Includes: Pointwise Conv + GLU → Depthwise Conv → BatchNorm → Swish activation
   - Depthwise convolution efficiently extracts temporal features, while GLU provides a gated control mechanism.
4. **Second FFN Module** (half-step residual)
5. **Final LayerNorm**

This “FFN → MHSA → Conv → FFN” **sandwich structure** was inspired by Macaron-Net, which proposed replacing the single FFN in Transformer blocks with two half-step FFNs — one before and one after attention. This structure, combined with Swish activation and proper normalization, improves both expressivity and stability during training.

#### Performance Highlights

The Conformer achieves state-of-the-art performance on the LibriSpeech benchmark:

| Model                  | Params   | WER (test/test-other) w/o LM | WER (test/test-other) w/ LM |
| ---------------------- | -------- | ---------------------------- | --------------------------- |
| Transformer Transducer | 139M     | 2.4% / 5.6%                  | 2.0% / 4.6%                 |
| **Conformer (Large)**  | **118M** | **2.1% / 4.3%**              | **1.9% / 3.9%**             |

Even the lightweight **Conformer-Small** model with just 10M parameters achieves **2.7% / 6.3%** WER without using any external language model — outperforming other models of similar size such as ContextNet.

#### Ablation Studies: Why is Conformer Effective?

The authors conducted comprehensive ablation studies, revealing the importance of each component:

| Component / Technique               | Critical? | Why                                                    |
| ----------------------------------- | --------- | ------------------------------------------------------ |
| **Convolution Module**              | ✅         | Key to performance gains                               |
| **Macaron-style Dual FFNs**         | ✅         | Outperforms single FFN and improves stability          |
| **Swish Activation**                | ✅         | Converges faster and generalizes better than ReLU      |
| **Relative Positional Encoding**    | ✅         | Enhances robustness to sequence length variation       |
| **Conv After MHSA**                 | ✅         | Outperforms parallel or pre-attention structures       |
| **Depthwise Conv Kernel Size = 32** | ✅         | Optimal balance between receptive field and efficiency |

#### Conclusion

The Conformer architecture presents a highly creative and practical approach to combining the best of CNNs and Transformers for ASR. By jointly modeling **local patterns and global semantics**, it sets a new bar for speech recognition performance while maintaining architectural modularity.

#### Code

```python
import torch
import math
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import Optional, Tuple

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * inputs.sigmoid()
    
class GLU(nn.Module):
    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs: Tensor) -> Tensor:
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()
    
class Linear(nn.Module):
    def __init__(
            self, 
            in_features: int, 
            out_features: int, 
            bias: bool = True
    ) -> None:
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)
    
class View(nn.Module):
    def __init__(self, shape: tuple, contiguous: bool = False):
        super(View, self).__init__()
        self.shape = shape
        self.contiguous = contiguous

    def forward(self, x: Tensor) -> Tensor:
        if self.contiguous:
            x = x.contiguous()
        return x.view(*self.shape)
    
class Transpose(nn.Module):
    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(*self.shape)

class ResidualConnectionModule(nn.Module):
    def __init__(
            self, 
            module: nn.Module, 
            module_factor: float = 1.0, 
            input_factor: float = 1.0
    ) -> None:
        super(ResidualConnectionModule, self).__init__()
        self.module = module
        self.module_factor = module_factor
        self.input_factor = input_factor

    def forward(self, inputs: Tensor) -> Tensor:
        return (self.module(inputs) * self.module_factor) + (inputs * self.input_factor)
    
class Conv2dSubsampling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(Conv2dSubsampling, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
        )
    
    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        outputs = self.sequential(inputs.unsqueeze(1))
        batch_size, channels, subsampled_lengths, sumsampled_dim = outputs.size()

        outputs = outputs.permute(0, 2, 1, 3)
        outputs = outputs.contiguous().view(batch_size, subsampled_lengths, channels * sumsampled_dim)

        output_lengths = input_lengths >> 2
        output_lengths -= 1

        return outputs, output_lengths
    
class FeedForwardModule(nn.Module):
    def __init__(
            self,
            encoder_dim: int = 512,
            expansion_factor: int = 4,
            dropout_p: float = 0.1,
    ) -> None:
        super(FeedForwardModule, self).__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            Linear(encoder_dim, encoder_dim * expansion_factor, bias=True),
            Swish(),
            nn.Dropout(p=dropout_p),
            Linear(encoder_dim * expansion_factor, encoder_dim, bias=True),
            nn.Dropout(p=dropout_p)
        )
    
    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs)
    
class RelPositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 512, max_len: int = 5000) -> None:
        super(RelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x: Tensor) -> None:
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return

        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: Tensor) -> Tensor:
        self.extend_pe(x)
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2 - x.size(1) + 1 : self.pe.size(1) // 2 + x.size(1),
        ]
        return pos_emb
    
class RelativeMultiHeadAttention(nn.Module):
    def __init__(
            self,
            d_model: int = 512,
            num_heads: int = 16,
            dropout_p: float = 0.1
    ) -> None:
        super(RelativeMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        self.d_model = d_model
        self.d_head = int(d_model/num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(self.d_head)

        self.query_proj = Linear(d_model, d_model)
        self.key_proj = Linear(d_model, d_model)
        self.value_proj = Linear(d_model, d_model)
        self.pos_proj = Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout_p)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        self.out_proj = Linear(d_model, d_model)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            pos_embedding: Tensor,
            mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        pos_embedding = self.pos_proj(pos_embedding).view(batch_size, -1, self.num_heads, self.d_head)

        content_score = torch.matmul((query + self.u_bias).transpose(1, 2), key.transpose(2, 3))
        pos_score = torch.matmul((query + self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1))
        pos_score = self._relative_shift(pos_score)

        score = (content_score + pos_score) / self.sqrt_dim

        if mask is not None:
            mask = mask.unsqueeze(1)
            score.masked_fill_(mask, -1e9)

        attn = F.softmax(score, -1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(context)

    def _relative_shift(self, pos_score: Tensor) -> Tensor:
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)[:, :, :, : seq_length2 // 2 + 1]

        return pos_score
    
class MultiHeadedSelfAttentionModule(nn.Module):
    def __init__(
            self, 
            d_model: int, 
            num_heads: int, 
            dropout_p: float = 0.1
    ) -> None:
        super(MultiHeadedSelfAttentionModule, self).__init__()
        self.positional_encoding = RelPositionalEncoding(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = RelativeMultiHeadAttention(d_model, num_heads, dropout_p)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(
            self, 
            inputs: Tensor, 
            mask: Optional[Tensor] = None
    ) -> Tensor:
        batch_size = inputs.size(0)
        pos_embedding = self.positional_encoding(inputs)
        pos_embedding = pos_embedding.repeat(batch_size, 1, 1)

        inputs = self.layer_norm(inputs)
        outputs = self.attention(inputs, inputs, inputs, pos_embedding=pos_embedding, mask=mask)

        return self.dropout(outputs)
    
class DepthwiseConv1D(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            bias: bool = False
    ) -> None:
        super(DepthwiseConv1D, self).__init__()
        assert out_channels % in_channels == 0, "out_channels should be constant multiple of in_channels"
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias,
        )
    
    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)
    
class PointwiseConv1d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            padding: int = 0,
            bias: bool = True,
    ) -> None:
        super(PointwiseConv1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)
    
class ConformerConvModule(nn.Module):
    def __init__(
            self,
            in_channels: int,
            kernel_size: int = 31,
            expansion_factor: int = 2,
            dropout_p: float = 0.1,
    ) -> None:
        super(ConformerConvModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        assert expansion_factor == 2, "Currently, Only Supports expansion_factor 2"

        self.sequential = nn.Sequential(
            nn.LayerNorm(in_channels),
            Transpose(shape=(1, 2)),
            PointwiseConv1d(in_channels, in_channels * expansion_factor, stride=1, padding=0, bias=True),
            GLU(dim=1),
            DepthwiseConv1D(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2),
            nn.BatchNorm1d(in_channels),
            Swish(),
            PointwiseConv1d(in_channels, in_channels, stride=1, padding=0, bias=True),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs).transpose(1, 2)

class ConformerBlock(nn.Module):
    def __init__(
            self,
            encoder_dim: int = 512,
            num_attention_heads: int = 8,
            feed_forward_expansion_factor: int = 4,
            conv_expansion_factor: int = 2,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1,
            conv_kernel_size: int = 31,
            half_step_residual: bool = True,
    ) -> None:
        super(ConformerBlock, self).__init__()
        if half_step_residual:
            self.feed_forward_residual_factor = 0.5
        else:
            self.feed_forward_residual_factor = 1

        self.sequential = nn.Sequential(
            ResidualConnectionModule(
                module=FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                ),
                module_factor=self.feed_forward_residual_factor,
            ),
            ResidualConnectionModule(
                module=MultiHeadedSelfAttentionModule(
                    d_model=encoder_dim,
                    num_heads=num_attention_heads,
                    dropout_p=attention_dropout_p,
                ),
            ),
            ResidualConnectionModule(
                module=ConformerConvModule(
                    in_channels=encoder_dim,
                    kernel_size=conv_kernel_size,
                    expansion_factor=conv_expansion_factor,
                    dropout_p=conv_dropout_p,
                ),
            ),
            ResidualConnectionModule(
                module=FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                ),
                module_factor=self.feed_forward_residual_factor,
            ),
            nn.LayerNorm(encoder_dim),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs)
    
class ConformerEncoder(nn.Module):
    def __init__(
            self,
            input_dim: int = 80,
            encoder_dim: int = 512,
            num_layers: int = 17,
            num_attention_heads: int = 8,
            feed_forward_expansion_factor: int = 4,
            conv_expansion_factor: int = 2,
            input_dropout_p: float = 0.1,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1,
            conv_kernel_size: int = 31,
            half_step_residual: bool = True,
    ) -> None:
        super(ConformerEncoder, self).__init__()
        self.conv_subsample = Conv2dSubsampling(in_channels=1, out_channels=encoder_dim)
        self.input_projection = nn.Sequential(
            Linear(encoder_dim * (((input_dim - 1) // 2 - 1) // 2), encoder_dim),
            nn.Dropout(p=input_dropout_p),
        )
        self.layers = nn.ModuleList([ConformerBlock(
            encoder_dim=encoder_dim,
            num_attention_heads=num_attention_heads,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            conv_expansion_factor=conv_expansion_factor,
            feed_forward_dropout_p=feed_forward_dropout_p,
            attention_dropout_p=attention_dropout_p,
            conv_dropout_p=conv_dropout_p,
            conv_kernel_size=conv_kernel_size,
            half_step_residual=half_step_residual,
        ) for _ in range(num_layers)])

    def count_parameters(self) -> int:
        return sum([p.numel() for p in self.parameters()])
    
    def update_dropout(self, dropout_p: float) -> None:
        for name, child in self.named_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout_p

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        outputs, output_lengths = self.conv_subsample(inputs, input_lengths)
        outputs = self.input_projection(outputs)

        for layer in self.layers:
            outputs = layer(outputs)

        return outputs, output_lengths
    
class Conformer(nn.Module):
    def __init__(
            self,
            num_classes: int,
            input_dim: int = 80,
            encoder_dim: int = 512,
            num_encoder_layers: int = 17,
            num_attention_heads: int = 8,
            feed_forward_expansion_factor: int = 4,
            conv_expansion_factor: int = 2,
            input_dropout_p: float = 0.1,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1,
            conv_kernel_size: int = 31,
            half_step_residual: bool = True,
    ) -> None:
        super(Conformer, self).__init__()
        self.encoder = ConformerEncoder(
            input_dim=input_dim,
            encoder_dim=encoder_dim,
            num_layers=num_encoder_layers,
            num_attention_heads=num_attention_heads,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            conv_expansion_factor=conv_expansion_factor,
            input_dropout_p=input_dropout_p,
            feed_forward_dropout_p=feed_forward_dropout_p,
            attention_dropout_p=attention_dropout_p,
            conv_dropout_p=conv_dropout_p,
            conv_kernel_size=conv_kernel_size,
            half_step_residual=half_step_residual,
        )
        self.fc = Linear(encoder_dim, num_classes, bias=False)

    def count_parameters(self) -> int:
        return self.encoder.count_parameters()

    def update_dropout(self, dropout_p) -> None:
        self.encoder.update_dropout(dropout_p)

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        outputs = self.fc(encoder_outputs)
        outputs = nn.functional.log_softmax(outputs, dim=-1)
        return outputs, encoder_output_lengths
    
if __name__ == "__main__":
    input_dim = 80
    encoder_dim = 144
    num_classes = 10

    model = Conformer(
        input_dim=input_dim,
        encoder_dim=encoder_dim,
        num_classes=num_classes,
        num_encoder_layers=4,
        num_attention_heads=4,
    )

    x = torch.randn(4, 160, input_dim)
    lengths = torch.tensor([160, 150, 140, 130], dtype=torch.long)

    logits, out_lengths = model(x, lengths)

    print("Output logits shape:", logits.shape)     # Expected: (4, T', num_classes)
    print("Output lengths:", out_lengths)           
```

