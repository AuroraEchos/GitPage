### SwiGLU（Swish Gated Linear Unit）

SwiGLU 是 Gated Linear Unit（GLU，门控线性单元）家族中的重要变体，由 PaLM 模型首次提出，目前广泛应用于 LLaMA、GPT-3、PaLM 等主流大模型架构中，核心优势是兼顾非线性表达能力与训练稳定性，缓解梯度消失问题，同时提升模型的推理效率。

在神经网络前向传播过程中，激活函数的核心作用是为网络引入**非线性特性**——若没有激活函数，无论网络有多少层，最终都等价于单一线性变换，无法拟合复杂的非线性数据分布。

传统激活函数的应用形式（通用线性-激活-线性结构）可表示为：
$$
y = Activation(xW_1 + b_1)W_2 + b_2
$$
其中，$$W_1、W_2$$ 为可学习权重矩阵，$$b_1、b_2$$ 为可学习偏置项；$$Activation$$ 为激活函数，常见的有 ReLU、GELU、Sigmoid 等。这种结构的局限性在于，激活函数对输入的“筛选作用”相对单一，无法灵活控制特征的传递强度。

GLU 家族激活函数打破了传统激活函数的应用模式，核心创新是引入**门控机制**——通过一个“门控分支”控制另一个“特征分支”的信息传递，类比于人类大脑的“注意力筛选”，只保留有用的特征，丢弃冗余信息，从而提升模型的表达能力和训练效率。GLU 家族的统一基本公式如下：
$$
GLU(X) = (xW_1 + b_1) \odot \sigma(xW_2 + b_2)
$$
公式说明：

- $$x$$：输入向量（或矩阵，对应批量输入）；
- $$W_1、W_2$$：两路独立的可学习权重矩阵，$$b_1、b_2$$：对应两路的可学习偏置；
- $$\odot$$：Hadamard 积（逐元素相乘），即两个同维度向量/矩阵，对应位置元素相乘；
- $$\sigma$$：门控激活函数，标准 GLU 中使用 Sigmoid 函数，这是门控机制的核心。

Sigmoid 是一种经典的 S 型激活函数，核心作用是将输入值映射到 $$[0,1]$$ 区间，输出值可理解为“门控的开启程度”——输出越接近 1，对应特征分支的信息保留越多；越接近 0，信息保留越少。Sigmoid 函数公式：
$$
S(x) = \frac{1}{1 + e^{-x}}
$$
函数特性：光滑、严格单调、饱和（输入趋于 $$+\infty$$ 时输出趋近于 1，输入趋于 $$-\infty$$ 时输出趋近于 0），但存在梯度消失问题（输入绝对值过大时，导数趋近于 0）。

所有 GLU 变体均遵循上述统一基本公式，**唯一差异仅在于门控激活函数**（即替换公式中的 $$\sigma$$），不同门控函数的选择的核心是平衡“非线性表达”与“梯度稳定性”。

SwiGLU 的核心改进的是：用 **Swish 函数**替代标准 GLU 中的 Sigmoid 函数作为门控，解决了 Sigmoid 函数的梯度消失问题，同时保留门控机制的特征筛选能力，更适合深度大模型的训练。
$$
SwiGLU(X) = (xW_1 + b_1) \odot Swish(xW_2 + b_2)
$$
Swish 函数是由 Google 提出的一种自适应激活函数，兼具 ReLU 的非饱和特性和 Sigmoid 的光滑特性，其公式为：
$$
Swish(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$
函数特性：

- 非饱和性：当 $$x > 0$$ 时，函数值随 x 增大而单调递增，无上限，可缓解梯度消失；
- 光滑性：整体曲线连续可导，优于 ReLU 的分段不可导（x=0 处）；
- 自适应性：输出值与输入 x 相关，不同输入对应不同的“激活强度”，更贴合特征的动态变化。

在实际工程实现（如 PyTorch、TensorFlow）中，为了提升计算效率，通常会对 SwiGLU 进行简化：将输入 x 做一次线性变换后，沿通道维度平分为两路（无需单独定义 $$W_1、W_2$$），分别作为特征分支和门控分支，简化公式如下：
$$
SwiGLU(x) = a \odot Swish(b)
$$
其中，$$xW + b = \text{Concat}(a, b)$$，$$W$$ 为单一权重矩阵，$$a$$ 为特征分支，$$b$$ 为门控分支（经过 Swish 激活）。

相比标准 GLU，解决了 Sigmoid 门控的梯度消失问题；相比 ReGLU、GEGLU，计算量更小，推理速度更快，同时保留了较强的非线性拟合能力；主要用于大语言模型（LLM）的 Feed-Forward Network（FFN，前馈网络）层，是 LLaMA、GPT-3、PaLM 等模型的核心激活函数，也是目前大模型优化中“提升效率与性能”的关键选择之一。

SwiGLU 的 Pytorch 实现如下：

```python
import torch
import torch.nn as nn

class SwiGLU(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        :param input_dim: 输入特征维度
        :param output_dim: 输出特征维度
        """
        super(SwiGLU_Base, self).__init__()
        # 定义两路独立的线性层（对应 W1, b1 和 W2, b2）
        self.linear1 = nn.Linear(input_dim, output_dim)  # 特征分支：xW1 + b1
        self.linear2 = nn.Linear(input_dim, output_dim)  # 门控分支：xW2 + b2
        # 实现 Swish 激活函数
        self.swish = lambda x: x * torch.sigmoid(x)

    def forward(self, x):
        """前向传播：特征分支 × Swish门控分支"""
        # 特征分支：xW1 + b1
        feature_branch = self.linear1(x)
        # 门控分支：先计算 xW2 + b2，再经过 Swish 激活
        gate_branch = self.swish(self.linear2(x))
        # 逐元素相乘，得到 SwiGLU 输出
        return feature_branch * gate_branch  # 等价于 Hadamard 积 ⊙
```

