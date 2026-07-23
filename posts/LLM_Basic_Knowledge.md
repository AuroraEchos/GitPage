# LLM 基础知识点

### Q：请描述 Transformer 的整体架构？

Transformer 由 Encoder 和 Decoder 两部分组成，然后其中的每个部分由 N 个相同的层堆叠而成。

Encoder 层首先输入是一个完整的序列，然后进行通过 Embedding 层进行特征向量化，输出为稠密的低维度的连续向量，接着进行位置编码去注入时序信息，因为 Transformer 内部的注意力机制在计算的时候是并行的，天然丢失了位置信息，比如对模型来说，“我爱深度学习”和“深度学习爱我”在没有位置信息时是一样的，所以我们需要让模型知道序列中 Token 的先后顺序，也就是在我们拿到的向量上加上位置编码，接下来就是由 N 个相同的层堆叠，以 Encoder 中一个为例子，首先经过多头注意力，然后是残差连接与层归一化，接着是前馈神经网络，然后再经过一个残差连接与归一化，接着就是堆叠，残差连接就是把这一层的输入直接加到输出上，防止深层网络梯度消失。

Decoder 的整体架构 Encoder 极其相似，也是 N 个相同的层堆叠，输入是目标序列，在训练时，是将真实标签序列整体向右平移一个位置，这个序列同样先通过 Embedding 层进行特征向量化，转化为稠密的低维连续向量，接着通过位置编码注入时序信息，让模型感知目标端 Token 的先后顺序。接下来，就是由 N 个相同的层堆叠。以 Decoder 中的单个层为例，第一步先经过掩码多都注意力，为什么使用掩码，是因为在生成文本时，模型不能看到未来的词，通过掩码，就可以确保模型在预测第 t 个位置的词时，只能看到前 t - 1 个位置的序列信息，然后就是残差连接与层归一化，接下来经过一个交叉注意力，在这部分，注意力机制的 Q 来自 Decoder 上一步的输出，K 和 V 来自于 Encoder 最终的完整输出，通过这一步，Decoder 能够去注意输入序列中的关键信息，将两边的数据连接起来。然后再次进行残差连接与层归一化，接着进行前馈神经网络，对当前融合后的特征进行非线性变换和映射，接着再进行一次残差连接与层归一化。

在经过 N 个这样的 Decoder 层堆叠并输出最终完整的特征序列后，最后再经过一个线型变换层和 Softmax 层，最终输出每个位置预测下一个 token 的概率分布。

------

### Q：为什么 Transformer 比 RNN 要好？

第一点：RNN 是串行计算，无法有效的利用大规模 GPU 集群进行计算。Transformer 中的自注意力机制可以一次性同时计算整个序列中所有 Token 之间的相关性，可以大规模分布式训练。

第二点：长距离信息依赖，RNN 会发生梯度消失或者信息遗忘，但是 Transformer 计算时，序列中的任意两个 Token 都可以通过注意力矩阵直接计算。

第三点：RNN 如果序列的长度为 L，那么第一个词的信息传递到第 L 个词需要跨越 L 步，但是 Transformer 中每个词之间的信息传输路径长度永远为 O(1)。

------

### Q：详细推导 Self-Attention 的计算过程

计算公式：
$$
Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
$$
假设有一个输入序列，长度为 N，词向量维度为 $d_{model}$ ，经过词向量嵌入后，得到一个输入矩阵  $X \in \mathbb{R}^{N \times d_{model}}$ 。

自注意力机制引入了三个可学习的权重矩阵 $W_Q \in \mathbb{R}^{d_{model} \times d_k}$ ，$W_K \in \mathbb{R}^{d_{model} \times d_k}$ ，$W_V \in \mathbb{R}^{d_{model} \times d_v}$ ，分别与 X 相乘，得到三个全新的矩阵：Query（查询矩阵）、Key（键矩阵）和 Value（值矩阵）。

- $$Q = X \cdot W_Q \quad (Q \in \mathbb{R}^{N \times d_k})$$

- $$K = X \cdot W_K \quad (K \in \mathbb{R}^{N \times d_k})$$

- $$V = X \cdot W_V \quad (V \in \mathbb{R}^{N \times d_v})$$

接着计算相似度分数，我们要知道序列中第 i 个词和第 j 个词之间的相关性，使用向量的点积，也就是：
$$
S = Q K^T \quad (S \in \mathbb{R}^{N \times N})
$$
随着维度 $d_k$ 的增大，点积的结果往往会变得很大。这会导致下一步计算 Softmax 时，梯度极其微弱（进入饱和区）。为了避免这个问题，我们需要将分数除以 $\sqrt{d_k}$ 进行缩放：
$$
S_{\text{scaled}} = \frac{Q K^T}{\sqrt{d_k}}
$$
为了将这些分数转化为概率分布（即权重，相加为 1），我们在**行**的方向上应用 Softmax 函数：
$$
A = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) \quad (A \in \mathbb{R}^{N \times N})
$$
此时，矩阵 $A$ 就是 Attention Map（注意力权重矩阵）。其中的第 $i$ 行代表第 $i$ 个词对序列中所有词的注意力权重分配。

最后 step，我们用计算出的权重矩阵 $A$，对包含实际信息的 Value 矩阵 $V$ 进行加权求和：
$$
O = A \cdot V \quad (O \in \mathbb{R}^{N \times d_v})
$$
输出矩阵 $O$ 的每一行，就是融合了上下文信息后，该 Token 的全新向量表示。

在理论上，$d_v$ 可以独立于 $d_k$，甚至不需要等于 $d_{model}$。但在工程实践中（例如标准的 Transformer 架构），为了方便搭建多层残差连接（Residual Connection，需要将输入 $X$ 与输出 $O$ 直接相加：$X + O$），通常直接令 $d_v = d_k$。

------

### Q：为什么要除以 $\sqrt{d_k}$ ？

设 q,k 各维度服从标准正态分布 N(0,1)，$$\mathbb{E}[q_i] = \mathbb{E}[k_i] = 0$$ ，$$\text{Var}(q_i) = \text{Var}(k_i) = 1$$ ，点积运算 $q \cdot k^T$，它实际上是 $d_k$ 个乘积项的和，由于这 $d_k$ 个乘积项之间是相互独立的，根据**方差的可加性**（独立随机变量相加，方差直接相加）：
$$
\text{Var}\left( \sum_{i=1}^{d_k} q_i k_i \right) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = \underbrace{1 + 1 + \dots + 1}_{d_k \text{个}} = d_k
$$
点积结果的总均值为 0，但总方差变成了 $d_k$。方差代表数据的离散程度或波动范围。根据统计学常识，标准差（Standard Deviation）是方差的算术平方根：
$$
\sigma = \sqrt{\text{Var}(q \cdot k^T)} = \sqrt{d_k}
$$
这意味着，随着维度 $d_k$ 的增大，点积结果的波动范围（标准差）会以 $\sqrt{d_k}$ 的速度随之扩大。

Softmax 的公式为 $f(x_i) = \frac{e^{x_i}}{\sum e^{x_j}}$。如果我们不除以 $\sqrt{d_k}$，当 $d_k$ 很大时极大值会把其他指数项压到接近 0 ，Softmax 输出趋近 one-hot，最大值位置概率接近 1，其余全为 0，一旦 Softmax 逼近 One-Hot，由于其输出几乎变成了常数，在该区域的导数（梯度）会趋近于 0。反向传播时，梯度传不回去，整个模型就会停止学习。通过除以 $\sqrt{d_k}$，我们成功把点积结果的方差从 $d_k$ 重新拉回到了 1，完美阻止了 Softmax 进入饱和区。

------

### Q： Self-Attention 的计算复杂度为多少？

对于两个矩阵相乘，$(A \times B) \cdot (B \times C)$ 的基础时间复杂度为 $\mathcal{O}(A \cdot B \cdot C)$。

生成 Q, K, V 矩阵：$3 \cdot \mathcal{O}(N \cdot d^2) \rightarrow \mathcal{O}(N \cdot d^2)$ 。

计算注意力分数：$\mathcal{O}(N \cdot d \cdot N) = \mathcal{O}(N^2 \cdot d)$ 。

Softmax 归一化：每一行有 $N$ 个元素，做一次 Softmax 是 $\mathcal{O}(N)$；一共 $N$ 行，总复杂度为 $\mathcal{O}(N^2)$。

Value 加权聚合成输出：$\mathcal{O}(N \cdot N \cdot d) = \mathcal{O}(N^2 \cdot d)$ 。

最终线性投影：$\mathcal{O}(N \cdot d \cdot d) = \mathcal{O}(N \cdot d^2)$ 。

忽略低阶项和常数项，最终的时间复杂度为：$$\mathcal{O}(N \cdot d^2 + N^2 \cdot d)$$ 。

在绝大多数大模型或标准 NLP 场景中，当序列增长或处理长文本时，**$N$ 的增长远快于 $d$**，或者说 $N^2 \cdot d$ 占据了绝对主导。因此我们通常将注意力机制的时间复杂度挂钩到其致命的瓶颈：**$\mathcal{O}(N^2 \cdot d)$**。

------

### Q：Multi-Attention 的原理和作用

单一的注意力头只能学习一种类型的语义依赖关系，多头注意力把 Query、Key、Value 投影到多个不同的子空间，并行执行多组注意力计算，最后拼接融合，让模型同时捕捉多种不同的关联模式。

假设输入的向量维度为 $d_{model}$ ，头数为 h，每个头的维度 $d_k = d_v = d_{model}/h$ 。

首先进行分头，将 Q，K，V 分别映射到 h 个子空间，得到 h 组 ($Q_i, K_i, V_i$) 。

接着对每一组执行缩放点积注意力：
$$
Attention(Q_i, K_i, V_i) = softmax(\frac{Q_iK_i^T}{\sqrt{d_k}})V_i
$$
然后直接拼接所有头的输出：
$$
Concat = [head_1; head_2;...;head_h]
$$
最后进行线性变换融合：
$$
MultiHead(Q, K, V) = Concat \cdot W_O
$$
其中 $W_O$ 是多头拼接之后的输出投影矩阵，唯一任务是把多个头拼接在一起的向量，重新映射回模型原始维度 $d_{model}$，同时学习融合多头信息。

通常来说 $d_{model}$ 能被头数 h 整除，保证均分维度。

举个 NLP 例子：

    Head1：关注语法主谓关系
    Head2：关注长距离指代
    Head3：关注局部相邻词语义
    Head4：关注同义词语匹配

单个注意力头很难同时学好全部；多头相当于同时启用多组不同关注点。

代码示例如下：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q,K,V: [batch, heads, seq_len, d_k]
    mask: [batch, 1, seq_len, seq_len] 或者 [1,1,seq_len,seq_len]
    """
    d_k = Q.size(-1)

    attn_score = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    if mask is not None:
        attn_score = attn_score.masked_fill(mask == 0, -1e9)

    attn_weight = F.softmax(attn_score, dim=-1)
    output = torch.matmul(attn_weight, V)
    return output, attn_weight


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被头数整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Q, K, V 投影矩阵 W_Q, W_K, W_V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # 输出融合矩阵 W_O
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # q: [B, Lq, d_model]
        # k: [B, Lk, d_model]
        # v: [B, Lv, d_model]

        batch_size = q.shape[0]

        # 线形投影
        Q = self.w_q(q)   # [B, Lq, d_model]
        K = self.w_k(k)   # [B, Lk, d_model]
        V = self.w_v(v)   # [B, Lv, d_model]
        
        # 分头
        # [B, seq_len, n_heads, d_k] -> [B, n_heads, seq_len, d_k]
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 缩放点积注意力
        attn_out, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        
        # 拼接多头
        # [B, n_heads, seq_len, d_k] -> [B, seq_len, n_heads, d_k]
        attn_out = attn_out.transpose(1, 2).contiguous()
        # concat：把所有头拼回 d_model
        attn_out = attn_out.view(batch_size, -1, self.d_model)

        # W_O 融合所有头信息
        output = self.w_o(attn_out)

        return output, attn_weights

if __name__ == "__main__":
    d_model = 512
    n_heads = 8
    mha = MultiHeadAttention(d_model, n_heads)
    batch = 2
    src_len = 12   # encoder序列长度
    tgt_len = 10    # decoder序列长度

    # 1. Encoder输入特征
    x = torch.randn(batch, src_len, d_model)

    print("===== 1. Encoder 自注意力 Q=K=V =====")
    out1, attn1 = mha(x, x, x)
    print("out shape:", out1.shape)      # [2, 12, 512]
    print("attn weight shape:", attn1.shape, "\n") # [2,8,12,12]

    # 构造Decoder因果掩码（下三角mask）
    def get_causal_mask(seq_len):
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask.unsqueeze(0).unsqueeze(0) # [1,1,L,L]
    tgt_mask = get_causal_mask(tgt_len)

    # 2. Decoder 掩码自注意力
    tgt_x = torch.randn(batch, tgt_len, d_model)
    print("===== 2. Decoder 掩码自注意力（带因果mask） =====")
    out2, attn2 = mha(tgt_x, tgt_x, tgt_x, mask=tgt_mask)
    print("out shape:", out2.shape)      # [2, 10, 512]
    print("attn weight shape:", attn2.shape, "\n") # [2,8,10,10]

    # 3. Decoder 交叉注意力
    memory = x   # encoder最终输出作为K,V
    print("===== 3. Decoder 交叉注意力 Cross-Attention =====")
    out3, attn3 = mha(q=tgt_x, k=memory, v=memory)
    print("out shape:", out3.shape)      # [2, 10, 512]
    print("attn weight shape:", attn3.shape) # [2,8,10,12]
```

Encoder 自注意力 / Decoder 掩码自注意力场景，传入 MHA 的原始 q、k、v 初始均为同一特征向量x；经过内部各自独立的线性投影后，得到不同的 Q，K，V；Decoder 交叉注意力场景，q 来自解码器，k/v 来自编码器，三者初始输入不是同一个特征。

只有 Decoder 的掩码自注意力使用下三角形式的因果掩码；Encoder 自注意力、交叉注意力使用 padding mask，和下三角无关。

------

### Q：RoPE 的原理

在 Transformer 架构中，Self-Attention 机制本身是排列不变的，不关注 token 的顺序，因此必须要注入位置信息。

传统位置编码分为两种类型：

- 绝对位置编码：直接加在词向量上，告诉模型这是第几个词，这种编码方式无法直接体现两个词之间的相对距离。
- 相对位置编码：直接在 Attention 的 Score 矩阵上加上与相对距离相关的偏置，也就是直接建模相对位置，但是破坏了词向量本身的独立性，且计算和显存开销比较大。

RoPE 的编码思想是在形式上是对词向量进行绝对位置编码，但是在计算 Attention 的内积的时候，只依赖相对位置，通过向量旋转来实现。

将词向量的每两个维度看作是一个二维平面，根据 token 的绝对位置，在这个平面上旋转一个特定的角度。

假设我们有一个 2 维的词向量 \[ q = \begin{pmatrix} q_0 \\ q_1 \end{pmatrix}, \] 它所在的位置是 $m$。 我们设定一个基础旋转角度 $\theta$。位置 $m$ 对应的总旋转角度就是 $m\theta$。 将向量 $q$ 旋转 $m\theta$ 角度，得到带有位置信息的向量 $q_m$：
$$
 q_m = R_m q = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix} \begin{pmatrix} q_0 \\ q_1 \end{pmatrix} 
$$
同理，对于另一个位置 $n$ 的向量 $k$，旋转后得到 $k_n = R_n k$。 

现在，我们计算 Attention 中核心的内积（点乘）：
$$
\begin{aligned} \langle q_m, k_n \rangle &= q_m^T k_n \\ &= (R_m q)^T (R_n k) \\ &= q^T R_m^T R_n k \end{aligned}
$$
根据旋转矩阵的性质，$R_m^T R_n = R_{n-m}$（即反向旋转 $m$ 再正向旋转 $n$，等价于直接旋转 $n-m$）。 因此： \[ \langle q_m, k_n \rangle = q^T R_{n-m} k = \langle q, k_{n-m} \rangle \]

也就是两个带有绝对位置信息的向量做内积，其结果只与它们的相对位置有关，与绝对位置无关。

实际模型中，词向量的维度 $d$ 通常是几百到几千（如 4096）。RoPE 的做法是：

- 将 $d$ 维向量两两分组，分成 $d/2$ 个二维子空间。
- 为每个子空间分配一个不同的旋转频率 $\theta_i$。
- 频率的设定借鉴了原始的 Sinusoidal 编码：$\theta_i = 10000^{-2i/d},\quad \text{其中 } i \in \{0,1,\dots,d/2-1\}$
- 对每个二维子空间分别进行旋转。

高频部分（$i$ 较小，$\theta_i$ 较大）：旋转角度变化快，用于捕捉短距离的局部位置关系。低频部分（$i$ 较大，$\theta_i$ 较小）：旋转角度变化慢，用于捕捉长距离的全局位置关系。

RoPE 不需要像传统位置编码那样增加额外的 Embedding 查表操作。旋转矩阵可以提前计算好，或者通过极其简单的三角函数和向量加减法实现，甚至可以直接融合到 QKV 的线性投影层的权重中，在推理时几乎不增加计算量。

理论上，由于 RoPE 是基于绝对位置 m 的函数，即使推理时遇到了训练时没有看见过的更长的序列，公式仍然成立。

------

### Q：从原始句子到位置编码前，数据维度的变化？

| 阶段             | 数据形态   | 张量形状 (Shape)      | 数据类型 | 包含的信息                                |
| ---------------- | ---------- | --------------------- | -------- | ----------------------------------------- |
| **原始句子**     | 字符串     | 无                    | String   | 人类可读的文本                            |
| **分词后**       | ID 列表    | `[L]`                 | Int      | 离散的符号索引                            |
| **批处理后**     | ID 矩阵    | **`[B, L]`**          | Int      | 包含 Batch 和 Sequence 维度的纯索引       |
| **Embedding 后** | 向量张量   | **`[B, L, d_model]`** | Float    | **纯语义信息（位置编码前的最终状态）**    |
| **RoPE 后**      | 旋转后张量 | **`[B, L, d_model]`** | Float    | 语义 + 绝对位置信息（准备进入 Attention） |

------

