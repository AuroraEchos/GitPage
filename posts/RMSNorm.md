### RMSNorm

RMSNorm（Root Mean Square Layer Normalization，均方根层归一化）是针对 Transformer 大模型优化的归一化方法，
核心思想是移除均值中心化，仅保留按均方根缩放，在不损失性能的前提下提升计算效率与训练稳定性。

设输入向量 $x \in \mathbb{R}^d$，$\epsilon$ 为防止除零的小常数，$\gamma$ 为可学习缩放参数：

均方根（RMS）定义：
$$
\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2}
$$
RMSNorm 前向传播公式：
$$
\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x) + \epsilon} \cdot \gamma
= \frac{x}{\sqrt{\mathbb{E}[x^2]+\epsilon}} \cdot \gamma
$$
LayerNorm 公式：
$$
\text{LayerNorm}(x) = \frac{x-\mathbb{E}[x]}{\sqrt{\text{Var}(x)+\epsilon}} \cdot \gamma + \beta
$$
从上述两个公式可以直观的看出差异：

- RMSNorm：无均值减法、无偏置 $\beta$、仅做缩放
- LayerNorm：中心化 + 标准化 + 偏置

无均值中心化意味着不改变向量的空间方向，仅缩放模长，且保留了方向信息，更加适配自注意力机制，计算更加清亮，无额外的均值依赖，深层传播更加稳定。

RMSNorm = 仅模长缩放的轻量化归一化}，凭借无中心化、计算简单、梯度稳定、保留方向四大特性，成为 LLaMA 等现代大模型的标配。

RMSNorm 的标准 Pytorch 实现如下：

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # 仅使用可学习缩放参数 gamma，无 beta
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _norm(self, x):
        # 计算均方根：1 / sqrt(E[x²] + eps)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 先做归一化，再乘以可学习权重
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
```

