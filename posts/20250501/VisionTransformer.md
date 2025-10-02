### *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*

**--Google Research, Brain Team**

发表于 ICLR 2021 的 *An Image is Worth 16x16 Words* 论文，首次提出了纯 Transformer 架构在图像分类上的可行性，并展示了它在大规模预训练数据上的强大性能，打破了卷积神经网络长期以来在视觉领域的统治地位。

Transformer架构自2017年问世以来，已经成为自然语言处理（NLP）领域的事实标准。然而，在计算机视觉（CV）领域，CNN 依然占据主导。尽管已有一些研究尝试将自注意力机制（Self-Attention）融入卷积模型，或作为其模块之一使用，但纯Transformer应用于视觉任务的尝试仍显稀缺。

该文提出了“Vision Transformer”（ViT），一个完全抛弃卷积的图像识别架构。其核心思想是：将图像划分为固定尺寸的 patch（如16×16），将每个 patch 当作“单词”处理，通过标准 Transformer 对其进行建模。这一简单而激进的做法，最终在多个图像识别任务中取得了媲美甚至超越 SOTA CNN 模型的结果。

#### 模型设计：

ViT 在模型设计方面最大的特点是：**以最小修改将标准 NLP Transformer 架构直接迁移到图像分类任务中**。

ViT 没有采用 CNN 的局部感知、权重共享等“视觉先验”，而是把图像看作一个**patch 序列**，每个 patch 类似于 NLP 中的“词”。整个模型几乎完全照搬 BERT 的架构，只在输入处理和位置编码上做了适配性调整。

![](/home/wenhaoliu/Project/Github/MyBlog/posts/20250501/ViT.png)

我们从输入到输出，逐步分析其设计构成。

1. **输入表示：图像切分与 patch 嵌入**

   ViT 首先将原始图像 x 划分为固定大小的 patch，我们假设输入图像的维度为 HxWxC。每个 patch 尺寸为 PxP，比如16x16 ，常见配置为 ViT-B/16 。这样一张图片就变成了 N=HW/P^2个 patch 的集合。然后每个 patch 被展平（flatten）成一个向量，输入到一个**可学习的线性层**中，投影到固定维度 D，得到 patch embedding 。这里的关键设计就是：图像 patch = 单词。

2. **[CLS] Token 与位置编码**

   类似于 BERT 中的 [CLS] token，ViT在 patch 序列前添加一个可学习的 class token 向量，用于分类任务。所有patch embedding 加上 position embedding，形成最终输入序列，位置编码是 1D learnable embedding，没有显式使用 2D spatial bias。需要注意的是ViT 并未借助任何硬编码的空间结构，而是依赖 Transformer 自行“学习”空间关系。

3. **编码器结构：标准 Transformer Blocks**

   ViT 使用原始 Transformer Encoder 堆叠多个 block：

   - 每层包含 Multi-Head Self-Attention（MSA）和一个 Feed-Forward Network（MLP）；
   - 每个子层前有 LayerNorm，后有残差连接；
   - 激活函数采用 GELU。

4. **分类头（Head）**

   预训练阶段：使用带一个隐藏层的 MLP 头；

   微调阶段：直接用一个线性分类层。

5. **支持更高分辨率微调（Patch 数量可变）**

   Transformer 原生支持**任意长度序列**，因此 ViT 也可支持**更高分辨率图像**。问题是：原始训练用的位置编码不一定适用于更多 patch 数量。解决方案是：对 pre-trained 的位置编码进行 **2D 双线性插值（interpolation）**，因此 ViT 在微调阶段可以自然支持更高分辨率，无需结构修改。

6. **可选结构：Hybrid ViT（CNN + ViT）**

   另一种替代设计是将 CNN 的中间特征图（如 ResNet stage 3 输出）作为输入，然后在该 feature map 上进行 patch 切分，后续送入 ViT，本质上是借助 CNN 引入部分 inductive bias，提高小数据/低算力时的泛化能力。

总的来看，ViT的设计理念可以总结如下：

| 模块       | ViT 设计选择                              | 与 CNN 的对比                    |
| ---------- | ----------------------------------------- | -------------------------------- |
| 输入处理   | 图像 → patch → flatten → Linear Embedding | CNN 使用局部卷积感受野           |
| 空间信息   | learnable 1D 位置编码                     | CNN 中空间结构隐式保留           |
| 注意机制   | 全局自注意力（multi-head）                | 卷积感知局部邻域                 |
| 特征合成   | 层层全局集成                              | 局部堆叠                         |
| 分类机制   | class token 输出 + MLP 头                 | GAP + 全连接层                   |
| 分辨率适应 | 支持插值扩展 patch 数                     | CNN 需固定输入尺寸或引入 pooling |
| 可扩展性   | 容易堆叠层、加宽通道                      | CNN 通常需定制结构设计           |
| 参数共享   | 无                                        | 卷积核共享空间位置参数           |

ViT 与传统 CNN 最大的不同在于，它不利用图像的二维结构，仅通过序列建模学习空间依赖关系。

#### 性能表现:

ViT 在 ImageNet 上直接训练表现一般，略逊于同规模 ResNet，但这并不意味着失败。研究发现：

- **大数据预训练是关键**：当在更大数据集（如 ImageNet-21k 或 JFT-300M）上预训练后，ViT 在 ImageNet、CIFAR-100 等多个 benchmark 上均超过 SOTA CNN。
- **计算资源更高效**：在相同预训练 FLOPs 下，ViT 的性能优于同级别的 ResNet，尤其在大模型和大数据情形下更显著。
- **迁移性能优越**：在 VTAB（19个小样本分类任务）上，ViT-H/14 获得最高平均准确率（77.6%），优于 BiT 和其他自监督方法。

#### 总结：

*An Image is Worth 16x16 Words* 以一种“极简但有效”的方式，颠覆了图像识别中对卷积的依赖，将 Transformer 的成功从 NLP 延伸到了 CV 领域。它开启了视觉Transformer新时代，为后续如 Swin Transformer、DeiT、BEiT 等工作的爆发奠定了理论与实践基础。

#### 代码：

```python
"""
Vision Transformer in PyTorch.

Components needed:
    - Patch Embedding
    - Positional Encoding
    - Transformer Encoder
    - Classification Head

Date: 2025-07-04
Author: Wenhao Liu
"""

import torch
import torch.nn as nn
from torch import Tensor

class PatchEmbedding(nn.Module):
    """
    Patch Embedding module for Vision Transformer (ViT).

    This module splits an image into non-overlapping patches and projects each patch
    into a high-dimensional embedding space using a convolutional layer. The resulting
    sequence of patch embeddings is suitable for input to a Transformer encoder.

    Args:
        img_size (int): The height and width of the input image. Default: 224.
        patch_size (int): The size (height and width) of each patch. Default: 16.
        in_channels (int): Number of input channels (e.g., 3 for RGB images). Default: 3.
        embed_dim (int): The dimensionality of the patch embedding output. Default: 768.

    Attributes:
        num_patches (int): The total number of patches generated from the input image.
        patch_embed (nn.Conv2d): Convolutional layer to extract and embed patches.
            Acts as a linear projection from flattened patch to embedding dimension.
    """
    def __init__(
            self,
            img_size: int = 224,
            patch_size: int = 16,
            in_channels: int = 3,
            embed_dim: int = 768
    ) -> None:
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2

        # Use Conv2d to divide image into patches and project to embed_dim
        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the PatchEmbedding module.

        Args:
            x (Tensor): Input image tensor of shape (B, C, H, W), where
                B = batch size,
                C = number of channels,
                H = height of image,
                W = width of image.

        Returns:
            Tensor: Output tensor of shape (B, N, D), where
                N = number of patches (H/patch_size * W/patch_size),
                D = embedding dimension.
        """
        # Apply patch embedding: shape becomes (B, D, H', W') where H'=W'=img_size//patch_size
        x = self.patch_embed(x)

        # Flatten spatial dimensions: (B, D, N)
        x = x.flatten(2)

        # Transpose to match transformer input shape: (B, N, D)
        x = x.transpose(1, 2)
        return x

class PositionalEncodingWithClsToken(nn.Module):
    """
    Adds a learnable classification token and positional encoding to the patch embeddings.

    This module is used in Vision Transformers to:
      - prepend a [CLS] token to the sequence of patch embeddings
      - add learnable positional embeddings to each token

    Args:
        num_patches (int): Number of patches (i.e., sequence length before adding [CLS] token).
        embed_dim (int): Dimensionality of patch embeddings and positional embeddings.

    Attributes:
        cls_token (nn.Parameter): Learnable classification token of shape (1, 1, D).
        pos_embed (nn.Parameter): Learnable position embedding of shape (1, N+1, D).
    """
    def __init__(self, num_patches: int, embed_dim: int) -> None:
        super().__init__()

        # Classification token, one per batch
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embeddings for all tokens including [CLS]
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Initialization (optional but helps training stability)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass that prepends the [CLS] token and adds position embeddings.

        Args:
            x (Tensor): Patch embeddings of shape (B, N, D)

        Returns:
            Tensor: Transformer input sequence of shape (B, N+1, D)
        """
        B = x.size(0)  # Batch size

        # Expand cls_token for batch dimension: [B, 1, D]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        # Concatenate [CLS] token to the beginning of patch embeddings
        x = torch.cat((cls_tokens, x), dim=1)  # [B, N+1, D]

        # Add position embeddings
        x = x + self.pos_embed
        return x

class TransformerEncoderBlock(nn.Module):
    """
    A single Transformer Encoder Block used in Vision Transformer (ViT).

    Each block consists of:
      - LayerNorm before Multi-Head Self-Attention (MHSA)
      - MHSA layer (using PyTorch's nn.MultiheadAttention)
      - Residual connection + dropout
      - LayerNorm before Feedforward MLP
      - MLP layer (2-layer feedforward with GELU activation)
      - Residual connection + dropout

    Args:
        embed_dim (int): Input and output embedding dimension.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Expansion factor for hidden dimension in MLP. Default: 4.0.
        dropout (float): Dropout rate applied after attention and MLP. Default: 0.1.

    Attributes:
        norm1 (nn.LayerNorm): Pre-norm for self-attention.
        attn (nn.MultiheadAttention): Multi-head self-attention module.
        norm2 (nn.LayerNorm): Pre-norm for MLP.
        mlp (nn.Sequential): Two-layer feedforward network.
    """
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            dropout: float = 0.1
    ) -> None:
        super().__init__()

        # Pre-Attention LayerNorm
        self.norm1 = nn.LayerNorm(embed_dim)

        # Multi-Head Self-Attention (batch_first=True allows [B, N, D] input)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        # Pre-MLP LayerNorm
        self.norm2 = nn.LayerNorm(embed_dim)

        # MLP (feedforward network): D -> D * r -> D
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the Transformer encoder block.

        Args:
            x (Tensor): Input tensor of shape (B, N, D)

        Returns:
            Tensor: Output tensor of the same shape (B, N, D)
        """
        # Self-attention with residual connection
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]

        # Feedforward MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) implementation in PyTorch.

    Args:
        img_size (int): Input image size (height and width).
        patch_size (int): Size of each patch (assumes square patches).
        in_channels (int): Number of input channels (e.g., 3 for RGB).
        num_classes (int): Number of output classes.
        embed_dim (int): Dimension of patch embeddings and transformer embeddings.
        depth (int): Number of Transformer encoder blocks.
        num_heads (int): Number of attention heads in each block.
        mlp_ratio (float): Expansion ratio for MLP hidden dimension.
        dropout (float): Dropout rate for attention and MLP layers.

    Components:
        - PatchEmbedding
        - PositionalEncodingWithClsToken
        - Stack of TransformerEncoderBlock
        - LayerNorm
        - Classification Head (Linear)
    """
    def __init__(
            self,
            img_size: int = 224,
            patch_size: int = 16,
            in_channels: int = 3,
            num_classes: int = 1000,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            dropout: float = 0.1
    ) -> None:
        super().__init__()

        # 1. Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # 2. Add [CLS] token and position encoding
        self.pos_embedder = PositionalEncodingWithClsToken(num_patches, embed_dim)

        # 3. Stack of Transformer encoder blocks
        self.blocks = nn.Sequential(*[
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # 4. Final layer norm before classification
        self.norm = nn.LayerNorm(embed_dim)

        # 5. Classification head: use CLS token representation
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the Vision Transformer.

        Args:
            x (Tensor): Input image tensor of shape (B, C, H, W)

        Returns:
            Tensor: Logits for classification of shape (B, num_classes)
        """
        x = self.patch_embed(x)                # [B, N, D]
        x = self.pos_embedder(x)               # [B, N+1, D]
        x = self.blocks(x)                     # [B, N+1, D]
        x = self.norm(x)                       # [B, N+1, D]
        cls_token_final = x[:, 0]              # Extract [CLS] token: [B, D]
        logits = self.head(cls_token_final)    # Classification output: [B, num_classes]
        return logits

if __name__ == "__main__":
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1
    )

    dummy_input = torch.randn(2, 3, 224, 224)  # Batch of 2 images
    out = model(dummy_input)
    print("Output shape:", out.shape)  # Expected: [2, 1000]
```

