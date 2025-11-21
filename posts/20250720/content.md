### Focal Loss: 应对类别不平衡

在深度学习的很多任务中，尤其是检测、分类和通信场景预测中，我们常常会遇到一个核心问题：**类别分布不平衡**。

传统的 CrossEntropy Loss 在这种情况下容易被“简单样本”和“多数类”主导，导致模型训练不充分。Focal Loss 正是为了解决这一痛点而提出的。

在目标检测任务中，背景样本数量远大于前景样本。类似的，以波束预测任务为例，不同波束方向的分布也高度不均衡（某些波束方向出现概率极高）。

如果直接用交叉熵损失：

- 易分类样本（置信度高）产生的 Loss 很小，但是数量庞大，主导梯度更新。
- 难分类的样本（模型预测不准确）数量少，在优化过程中被淹没。

这样一来，模型往往倾向于“讨好多数类”，而忽略了困难样本和少数类。



#### **下面我们来进行 Focal Loss 公式推导：**

标准的交叉熵损失：
$$
CE(p_t) = -log(P_t)
$$
其中：

- $p_t$是模型对真实类别的预测概率。

Focal Loss 在此基础上引入了**调制项**：
$$
FL(p_t) = -(1 - p_t)^\gamma log(p_t)
$$

- $(1 - p_t)$：预测概率越高，因子越小；预测概率越低，因子越大。
- $\gamma > 0$：调制强度，通常取2 。

作用：

- 易分类的样本（p_t 高），Loss 被显著削弱。
- 难分类的样本（p_t 低），Loss 保持较大。

在类别不平衡时，再加一个$\alpha$（通常为 0.25~0.75）：
$$
FL(p_t) = - \alpha (1 - p_t)^\gamma log(p_t)
$$
其中：

- $\alpha$用于平衡正负类或少数类和多数类的权重。



#### **下面进行Focal Loss 的 PyTorch 实现：**

对于多分类问题，Focal Loss 可以有两种实现方式：

1. **Softmax 版本**（互斥单标签分类）。
2. **Sigmoid 版本**（每类独立二分类，适合多标签任务或类别高度不均衡时）。

在 PyTorch 中，`torchvision.ops.sigmoid_focal_loss` 实际上实现的是 **sigmoid 版**。

**官方实现：**

```python
import torch
import torchvision

loss = torchvision.ops.sigmoid_focal_loss(
    inputs, targets, alpha=0.25, gamma=2.0, reduction="mean"
)
```

**自定义封装：**

```python
import torch
import torchvision
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, num_classes, gamma=2, alpha=0.25, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        if targets.dim() == 1:
            targets = torch.nn.functional.one_hot(targets, num_classes=self.num_classes)

        return torchvision.ops.sigmoid_focal_loss(
            inputs, targets.float(),
            alpha=self.alpha, gamma=self.gamma,
            reduction=self.reduction
        )
```



#### 超参数选择建议：

- $\gamma$：
  - 0，等价与普通交叉熵。
  - 1~3，常用范围。$\gamma$越大，越强调难样本。
- $\alpha$：
  - 正负样本平衡，检测任务常用 0.25~0.75。
  - 在多分类任务中，可以根据类别频率动态分配。



#### 总结：

Focal Loss 的贡献在于：

- 通过 **调制项** 解决了 **易样本主导问题**。
- 通过 **类别权重** 缓解了 **类别不平衡问题**。
