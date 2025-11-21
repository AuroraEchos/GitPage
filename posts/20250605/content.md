### *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*

**-- Tri Dao, Daniel Y. Fu etc.**

lashAttention 是一种高效的注意力计算方法，旨在减少传统注意力机制中的内存占用和计算时间。它最初是在 Transformer 模型中提出的，广泛应用于 NLP、计算机视觉等任务中。FlashAttention 的核心理念是通过优化内存访问模式和计算方式，以加速注意力机制的运算。

Transformer 中的自注意力机制（Self-Attention）计算复杂度较高，其时间和空间复杂度为$O(n^2)$，其中$n$为序列长度。每一个 token 都与其他所有 token 进行交互，这在长序列时尤为耗时和内存密集。尽管多头注意力机制能够并行计算多个注意力头，但仍然面临着内存瓶颈，尤其是在 GPU 上处理大规模数据时。

#### FlashAttention 的关键优化

1. **内存访问优化**
    FlashAttention 利用了块状矩阵乘法的技巧，通过分块计算的方式减少了内存访问的冗余。具体来说，它把矩阵乘法分成了小块，避免了传统方法中大量重复的内存读取操作。这种方法减少了对显存带宽的需求，同时降低了内存访问冲突。
2. **降低显存需求**
    FlashAttention 通过巧妙的算法设计，优化了计算图中的内存占用，使得即使在处理较长序列时，也能显著降低内存需求。它使得模型能够处理更长的序列，减少了 GPU 内存的压力。
3. **计算方式改进**
    FlashAttention 使用了更加高效的矩阵乘法和缩放机制。例如，它采用了“带有相对位置编码的缩放点积注意力”方法。通过减少计算中不必要的重复部分，能大幅提高计算速度。
4. **并行计算与分布式加速**
    FlashAttention 强调了高效的并行计算，不仅仅依赖于单一 GPU，还能利用多 GPU 的计算资源来加速模型训练和推理。由于算法的设计具有较好的并行性，可以很好地适应大规模集群的计算需求。

#### 核心计算过程

要详细从数学角度理解 **FlashAttention**，我们需要先回顾一下传统的 **Scaled Dot-Product Attention**，并深入探讨 FlashAttention 如何优化计算过程。

1. 传统的 Scaled Dot-Product Attention

   给定输入的查询$Q$，键$K$，和值$V$，注意力机制的输出可以通过以下公式计算：
   $$
   Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
   $$
   其中：

   $Q\in{R^{n\times{d_k}}}$，查询矩阵。

   $K\in{R^{n\times{d_k}}}$，键矩阵。

   $V\in{R^{n\times{d_v}}}$，值矩阵。

   计算的核心是先计算查询 $Q$ 和键 $K$ 的点积，缩放后通过 **softmax** 得到注意力权重，然后用这些权重加权求和值 $V$。

2. FlashAttention 的优化目标

   FlashAttention 的目标是优化传统方法中的计算和内存瓶颈，特别是在计$QK^T$和 **softmax** 时的内存和带宽需求。

   - 内存优化

     传统方法中，计算$QK^T$和 softmax 时需要占用大量内存。这个矩阵$QK^T$的维度是$n\times{n}$，对于长序列来说，显存需求很高。FlashAttention 使用了 **分块计算**（block-wise computation），将$QK^T$矩阵分为多个小块，这样可以减少一次性加载到内存中的数据量。

   - 分块计算

     FlashAttention 通过分块的方式计算注意力得分，避免一次性计算整个$QK^T$矩阵。设定一个分块大小$B$，那么 $QK^T$可以被拆分为多个小块，每个小块大小为 $B\times{B}$，这样矩阵计算就能局部并行执行。

     对于$QK^T$中的一块$Q_iK^T_j$（其中$i$和$j$是块索引），计算可以写成：
     $$
     Q_iK^T_j=\sum_{k=1}^{B}{Q_{ik}K_{jk}}
     $$
     这将矩阵的乘法分解为多个小规模的矩阵乘法，减少了内存压力。

   - Softmax 计算优化

     在计算 softmax 时，传统方法需要计算整个$QK^T$矩阵的 softmax。FlashAttention 的优化之一是 **分块 softmax 计算**。对于每一块 $Q_iK^T_j$，首先计算每块的 softmax，然后将 softmax 结果与值$V$进行加权求和。

     对于每一块 $Q_iK^T_j$，FlashAttention 使用以下公式计算注意力输出：
     $$
     Attention_i = softmax(\frac{Q_iK^T_j}{\sqrt{d_k}})V_j
     $$
     计算之后，这些注意力输出会被合并成最终的结果。

   - 最终的注意力输出

     最终的 FlashAttention 输出与传统的注意力输出是相同的，只是在计算过程中通过分块计算和优化内存访问来提高效率。输出公式为：
     $$
     Attention(Q, K, V) = \sum_{i=1}^n{softmax(\frac{Q_iK^T}{\sqrt{d_k}})V}
     $$
     

FlashAttention 相比传统方法优化了 **内存使用** 和 **计算效率**，主要通过以下几个步骤：

- **分块矩阵计算**：将矩阵$QK^T$分成小块，减少内存占用。
- **分块 softmax 计算**：对每个小块独立计算 softmax，避免计算整个大矩阵的 softmax。
- **优化内存访问**：减少内存带宽消耗，提高计算速度。

#### 代码

```python
import torch
import torch.nn.functional as F

def standard_attention(Q, K, V):
    """
    标准 Scaled Dot-Product Attention
    Q: [batch, heads, seq_len, d_k]
    K: [batch, heads, seq_len, d_k]
    V: [batch, heads, seq_len, d_v]
    """
    d_k = Q.size(-1)
    
    # [batch, heads, seq_len, seq_len]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    
    attn = F.softmax(scores, dim=-1)
    
    output = torch.matmul(attn, V)  # [batch, heads, seq_len, d_v]
    
    return output

def flash_attention(Q, K, V, block_size=128):
    """
    简化版 FlashAttention (CPU/GPU PyTorch实现)
    Q: [batch, heads, seq_len, d_k]
    K: [batch, heads, seq_len, d_k]
    V: [batch, heads, seq_len, d_v]
    block_size: 分块大小
    """
    B, H, N, D = Q.shape
    _, _, _, Dv = V.shape

    output = torch.zeros((B, H, N, Dv), device=Q.device, dtype=Q.dtype)

    # 遍历 Q 的每个 block
    for start_q in range(0, N, block_size):
        end_q = min(start_q + block_size, N)
        Q_block = Q[:, :, start_q:end_q, :]  # [B, H, Bq, D]

        # 在线 softmax 变量
        m_i = torch.full((B, H, end_q - start_q), float('-inf'), device=Q.device)
        l_i = torch.zeros((B, H, end_q - start_q), device=Q.device)
        acc = torch.zeros((B, H, end_q - start_q, Dv), device=Q.device)

        for start_k in range(0, N, block_size):
            end_k = min(start_k + block_size, N)
            K_block = K[:, :, start_k:end_k, :]  # [B, H, Bk, D]
            V_block = V[:, :, start_k:end_k, :]  # [B, H, Bk, Dv]

            # 局部 attention score
            scores = torch.matmul(Q_block, K_block.transpose(-2, -1)) / (D ** 0.5)  # [B, H, Bq, Bk]

            # 在线 softmax: 更新 m_i 和 l_i
            m_ij = torch.max(scores, dim=-1).values  # [B, H, Bq]
            m_new = torch.maximum(m_i, m_ij)

            exp_mi = torch.exp(m_i - m_new)
            exp_mij = torch.exp(scores - m_new.unsqueeze(-1))

            l_new = exp_mi * l_i + exp_mij.sum(dim=-1)
            acc_new = exp_mi.unsqueeze(-1) * acc + torch.matmul(exp_mij, V_block)

            m_i, l_i, acc = m_new, l_new, acc_new

        # 完成 softmax
        output[:, :, start_q:end_q, :] = acc / l_i.unsqueeze(-1)

    return output
```

