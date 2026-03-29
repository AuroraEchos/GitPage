### 位置编码

在 Transformer 模型中，由于 Self-Attention 机制本身是置换不变性的，也就是它无法识别输入序列中词语的顺序，我们需要通过 Positional Encoding (位置编码) 为模型引入位置信息。

由 Google 在 *Attention Is All You Need* 中提出的 Sinusoidal Positional Encoding (正弦位置编码) 是一种经典且优雅的绝对位置编码方案。

对于序列中的第 pos 个位置，其编码向量 PE 的第 i 个分量计算如下：
$$
PE_{(pos, 2i)} &= \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right) \\
PE_{(pos, 2i+1)} &= \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$
其中：

- pos：词在句子中的位置（0,1,2,…）。
- i：维度索引（从 0 到 dmodel/2）。
- d_model：嵌入向量的总维度。

正弦位置编码通过不同频率的正弦和余弦函数组合，为序列中的每个位置 $pos$ 生成一个 $d_{model}$ 维的向量。其核心数学特性如下：

1. 对于任何固定的偏移量 $k$，位置 $pos+k$ 的编码 $PE_{pos+k}$ 可以表示为 $PE_{pos}$ 的线性函数。 对于每一对正余弦分量，存在一个仅与 $k$ 有关的变换矩阵 $M^{(k)} \in \mathbb{R}^{2 \times 2}$：
   $$
   \begin{equation}
   \begin{bmatrix} 
   \sin(\omega_i (pos + k)) \\ \cos(\omega_i (pos + k)) 
   \end{bmatrix} = 
   \begin{bmatrix} 
   \cos(\omega_i k) & \sin(\omega_i k) \\ 
   -\sin(\omega_i k) & \cos(\omega_i k) 
   \end{bmatrix}
   \begin{bmatrix} 
   \sin(\omega_i pos) \\ \cos(\omega_i pos) 
   \end{bmatrix}
   \end{equation}
   $$
   其中频率 $\omega_i = 1/10000^{2i/d_{model}}$。这意味着模型可以通过注意力机制中的线性层，轻松捕捉到 token 之间的相对距离信息。

2. 分母项 $10000^{2i/d_{model}}$ 构成了一个几何级数，使得波长在 $2\pi$ 到 $10000 \cdot 2\pi$ 之间变化：

   低维分量 ($i \to 0$)具有高频率，能够捕捉局部的、精细的位置差异。高维分量 ($i \to d_{model}/2$)具有低频率，能够提供长距离的、宏观的位置骨架。这种设计类似于二进制计数器，确保了序列中每个位置的唯一性，同时保持了连续性。

3. 在 Self-Attention 中，位置编码的相似度直接影响注意力权重。两个位置编码的点积 $PE_{pos} \cdot PE_{pos+k}$ 随着距离 $k$ 的增加而呈现衰减趋势：
   $$
   PE_{pos} \cdot PE_{pos+k} = \sum_{i=0}^{d_{model}/2 - 1} \cos\left( \frac{k}{10000^{2i/d_{model}}} \right)
   $$
   这种特性为模型提供了一个先验：物理距离较近的词通常具有更高的相关性，这符合自然语言处理直觉。

正弦位置编码的标准 Pytorch 实现如下：

```python
def get_sinusoidal_positional_embeddings(seq_len: int, d_model: int):
    position = torch.arange(seq_len).unsqueeze(1)  # [SeqLen, 1]
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))  # [D/2]
    
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
    pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度
    
    return pe.unsqueeze(0)  # [1, SeqLen, D]
```

上述代码通过广播机制一次性完成了所有位置的计算，效率极高。它预计算了不同维度对应的“频率系数” div_term 。利用矩阵乘法原理（外积），将每个位置与每个频率配对。

Sinusoidal PE（正弦相加）虽然优雅，但有一个直观的缺点：它将位置信息硬生生地“砸”进词嵌入里。虽然模型能学，但这种方式对**相对位置**的感知是隐式的。

**RoPE (Rotary Positional Embedding，旋转位置编码)** 是目前大模型最主流的位置编码方案。它的核心思想不再是简单的“相加”，而是**“旋转”**。

RoPE 通过在特征维度上施加旋转变换，将位置信息编码到查询向量（Query）和键向量（Key）中。

设输入向量为：
$$
\mathbf{x} = [x_0, x_1, x_2, x_3, \dots, x_{d-2}, x_{d-1}]
$$
将其按相邻两个维度进行分组：
$$
(x_0, x_1), (x_2, x_3), \dots, (x_{d-2}, x_{d-1})
$$
对于第 $i$ 组二维向量，在位置 $pos$ 处施加如下旋转变换：
$$
\begin{bmatrix}
x_{2i}' \\
x_{2i+1}'
\end{bmatrix}
=
\begin{bmatrix}
\cos \theta_i & -\sin \theta_i \\
\sin \theta_i & \cos \theta_i
\end{bmatrix}
\begin{bmatrix}
x_{2i} \\
x_{2i+1}
\end{bmatrix}
$$
其中旋转角度定义为：
$$
\theta_i = pos \cdot \omega_i
$$
频率项 $\omega_i$ 定义为：
$$
\omega_i = 10000^{- \frac{2i}{d}}
$$
因此，每一对维度都会以不同的频率进行旋转，从而在向量中编码位置信息。

这种方法可以看作是在复数域中进行相位旋转，使得注意力机制在计算
$$
\mathbf{Q}\mathbf{K}^\top
$$
时自然引入相对位置信息。

需要特别注意的是，为什么只作用在 Q 和 K？

因为 attention：
$$
\mathbf{Q}\mathbf{K}^\top
$$
旋转后：
$$
(QR)(KR)^T = QRR^TK^T
$$
这里隐含了相对位置编码效果。V 不参与位置编码（否则会破坏内容表达）。

RoPE 的 PyTorch 实现本质就是：

> 把 embedding 的偶数/奇数维配对，当作复数，在不同位置上乘上不同相位，实现一个“位置相关的旋转”。

Sinusoidal Positional Encoding 的结合方式是想加，仅在模型输入层一次性注入，Rotary Positional Embedding 的结合方式是乘法/旋转，在每一层的 Attention 计算时注入。

下面给出 RoPE 的 Pytorch 实现：

```python
class RoPE(nn.Module):
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        # q, k: [Batch, Head, Seq_Len, Head_Dim]
        B, H, L, D = q.shape
        device = q.device

        t = torch.arange(L, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos = emb.cos()[None, None, :, :]  # [1,1,L,D]
        sin = emb.sin()[None, None, :, :]

        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        q_rot = (q * cos) + (rotate_half(q) * sin)
        k_rot = (k * cos) + (rotate_half(k) * sin)
        return q_rot, k_rot
```

将 RoPE 集成到 Attention 里，我们会得到一个更加完整的现代的注意力机制：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class RoPE(nn.Module):
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        # q, k: [Batch, Head, Seq_Len, Head_Dim]
        B, H, L, D = q.shape
        device = q.device

        t = torch.arange(L, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos = emb.cos()[None, None, :, :]  # [1,1,L,D]
        sin = emb.sin()[None, None, :, :]

        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        q_rot = (q * cos) + (rotate_half(q) * sin)
        k_rot = (k * cos) + (rotate_half(k) * sin)
        return q_rot, k_rot

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, dropout: float = 0.0, rope_base: int = 10000):
        super().__init__()
        assert n_heads % n_kv_heads == 0, "n_heads 必须能被 n_kv_heads 整除"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        
        # Q 是全部头，K/V 是 KV 头
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        
        # 位置编码
        self.rope = RoPE(self.head_dim, base=rope_base)
        
        # 输出投影
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout_p = dropout

    def forward(self, x: Tensor, mask: Tensor = None, is_causal: bool = True):
        B, L, _ = x.shape

        # 1. 分别投影 Q/K/V
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)    # [B,Hq,L,D]
        k = self.k_proj(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2) # [B,Hk,L,D]
        v = self.v_proj(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2) # [B,Hk,L,D]

        # 2. RoPE 位置编码
        q, k = self.rope(q, k)

        # 3. GQA 注意力计算
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal,
            enable_gqa=True  
        )

        # 4. 拼接多头 + 输出投影
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.w_o(attn_output)


if __name__ == "__main__":
    # 测试
    batch_size = 1
    seq_len = 1024
    d_model = 768
    n_heads = 8
    n_kv_heads = 2

    x = torch.randn(batch_size, seq_len, d_model)

    gqa = GroupedQueryAttention(d_model, n_heads, n_kv_heads)
    output = gqa(x)
    
    print("GQA+RoPE 输出形状:", output.shape)  # [1, 1024, 768]

```

