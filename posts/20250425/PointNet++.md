### *PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space*

**-- Charles R. Qi   Li Yi   Hao Su   Leonidas J. Guibas / Stanford University**

随着自动驾驶、AR/VR、机器人导航等领域的迅猛发展，**点云（Point Cloud）作为三维世界的原始表示形式，越来越成为感知与建图的核心信息源**。然而，点云是非结构化、无序、且分布不均的稀疏数据，使得传统 CNN 等规则网络难以直接应用。

2017 年，Charles Qi 等人提出的 PointNet 为点云处理开创了先河——**通过对每个点独立地提取特征并使用全局池化操作，实现了端到端的点云学习**。但随之而来的问题也逐渐显现：PointNet 缺乏对**局部几何结构的建模能力**，对点云采样密度变化敏感，在复杂任务（如室内语义分割、非均匀采样）中表现受限。

为了解决这些问题，PointNet++ 应运而生。

#### 核心思想：分层特征学习 + 局部区域建模

PointNet++ 的目标非常明确：

> 在保持 PointNet 原生处理无序点云能力的基础上，引入 **局部结构建模** 和 **分层特征抽象机制**，以适应更复杂和多变的点云任务场景。

这一目标通过以下几个核心设计达成：

- **局部邻域的显式建模**

  PointNet++ 通过在点云中构建局部区域（如球形邻域或最近邻），再在这些邻域内使用 PointNet 进行特征提取，实现了点与点之间**空间关系的建模**。

- **分层特征提取机制**

  借鉴 CNN 的思想，PointNet++ 构建了一种自底向上的层次网络结构：每一层网络都对点云进行采样、分组，并抽取更高层的语义特征，从而**逐步抽象局部到全局的信息**。

- **多尺度鲁棒性增强**

  为应对点云密度变化，PointNet++ 引入了多尺度组建机制（Multi-Scale Grouping, MSG），在多个感受野范围内提取特征，从而增强模型对非均匀点分布的鲁棒性。

#### 模型结构

整个 PointNet++ 的网络由以下两个阶段构成：

1. **Set Abstraction (SA) 模块：分层抽象局部特征**

   每个 SA 层完成以下三个操作：

   - **Sampling（采样）**：使用 Farthest Point Sampling（FPS）选择代表性的中心点；

   - **Grouping（分组）**：使用 Ball Query 或 kNN 构建每个中心点的邻域；

   - **PointNet Feature Learning**：在每个局部区域内使用共享 MLP + Pooling 的方式抽取区域特征。

   SA 层可以堆叠多层，每一层处理的点数逐步减少，但语义信息越来越抽象，形成从局部到全局的表达。

2. **Feature Propagation (FP) 模块：密度恢复与精细预测**

   由于 SA 过程不断下采样，模型末端的特征维度较低、密度较稀。为了实现像分割这样的细粒度任务，FP 层对特征进行上采样和融合：

   - 使用加权插值将稀疏特征传播到稠密点；
   - 使用 skip connection 将底层细节与高层语义融合；
   - 多层 MLP 进行融合处理，得到高分辨率输出。

![](/home/wenhaoliu/Project/Github/MyBlog/posts/20250425/PointNet++X2.png)

**多尺度分组策略：MSG and MRG**

为了提升模型适应非均匀密度的能力，PointNet++ 提出了两个增强版本：

- **Multi-Scale Grouping (MSG)**：每个中心点用多个半径分别构造不同邻域，提取多尺度特征后融合；
- **Multi-Resolution Grouping (MRG)**：从不同层级共享特征，适用于大规模点云且计算更高效。

MSG 效果更强，MRG 更轻量，二者可以根据任务需求灵活选择。

![](/home/wenhaoliu/Project/Github/MyBlog/posts/20250425/PointNet++X3.png)

#### 总结

PointNet++ 是 3D 点云领域具有里程碑意义的架构，开启了**直接在点集上进行分层几何学习**的范式，为后续所有基于点表示的模型奠定了基础。无论你是研究 3D 视觉、自动驾驶、机器人感知，还是探索多模态融合，**理解并掌握 PointNet++ 都是必要的技术储备**。

#### 代码

```python
import torch
from torch import Tensor

def index_points(points: Tensor, idx: Tensor) -> Tensor:
    """
    Input:
        points: Tensor of shape (B, N, C)
        idx: Tensor of shape (B, ..., K)
    Return:
        new_points: Tensor of shape (B, ..., K, C)
    """
    B = points.shape[0]
    idx_flatten = idx.reshape(B, -1)

    batch_indices = torch.arange(B, dtype=torch.long, device=points.device).view(B, 1)
    batch_indices = batch_indices.repeat(1, idx_flatten.shape[1])  # shape: (B, L)

    selected = points[batch_indices, idx_flatten]  # shape: (B, ..., K, C)

    new_shape = list(idx.shape) + [points.shape[2]]  # (..., C)
    return selected.view(*new_shape)


def farthest_point_sample(xyz: Tensor, npoint: int) -> Tensor:
    """
    Input:
        xyz: Tensor of shape (B, N, 3)
        npoint: int, number of points to sample
    Return:
        centroids_idx: Tensor of shape (B, npoint)
    """
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)  # 采样索引
    distance = torch.full((B, N), 1e10, device=xyz.device)  # 初始化距离为正无穷
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=xyz.device)  # 初始随机点

    batch_indices = torch.arange(B, device=xyz.device)

    for i in range(npoint):
        centroids[:, i] = farthest  # 记录采样点
        centroid_xyz = xyz[batch_indices, farthest].unsqueeze(1)  # (B, 1, 3)
        dist = torch.sum((xyz - centroid_xyz) ** 2, dim=-1)  # 欧氏距离平方 (B, N)
        mask = dist < distance
        distance[mask] = dist[mask]  # 更新最短距离
        farthest = torch.max(distance, dim=1)[1]  # 下一个最远点

    return centroids

def query_ball_point(radius: float, nsample: int, xyz: Tensor, new_xyz: Tensor) -> Tensor:
    """
    For each new_xyz (center point), find all xyz within radius.
    """
    B, N, _ = xyz.shape
    S = new_xyz.shape[1]

    # 计算距离矩阵：new_xyz (B, S, 3) vs xyz (B, N, 3)
    dists = torch.cdist(new_xyz, xyz, p=2)  # (B, S, N)

    # 找出每个中心点在半径范围内的点（bool mask）
    group_idx = dists.argsort(dim=-1)[:, :, :nsample]  # 先默认取前K个

    # 如果需要严格保证距离限制，进行裁剪（可选）
    # mask = dists <= radius
    # 但这里默认保留最小nsample个点，保持 shape 不变

    return group_idx  # (B, S, nsample)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint: int, radius: float, nsample: int,
                 in_channel: int, mlp_channels: List[int], group_all: bool):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel + 3  # 坐标也作为特征输入
        for out_channel in mlp_channels:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz: torch.Tensor, points: torch.Tensor):
        """
        Input:
            xyz: (B, N, 3) - 原始点坐标
            points: (B, N, C) - 输入点特征（如为None则只有xyz）
        Return:
            new_xyz: (B, npoint, 3) - 中心点坐标
            new_points: (B, npoint, mlp[-1]) - 每个中心点的特征
        """
        B, N, _ = xyz.shape

        if self.group_all:
            new_xyz = torch.zeros(B, 1, 3).to(xyz.device)
            grouped_xyz = xyz.view(B, 1, N, 3)
            if points is not None:
                grouped_points = points.view(B, 1, N, -1)
                new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
            else:
                new_points = grouped_xyz
        else:
            # 1. 采样中心点
            idx = farthest_point_sample(xyz, self.npoint)  # (B, npoint)
            new_xyz = index_points(xyz, idx)                # (B, npoint, 3)

            # 2. 构建邻域
            group_idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)  # (B, npoint, nsample)
            grouped_xyz = index_points(xyz, group_idx)  # (B, npoint, nsample, 3)
            grouped_xyz -= new_xyz.unsqueeze(2)         # 局部相对坐标

            if points is not None:
                grouped_points = index_points(points, group_idx)  # (B, npoint, nsample, C)
                new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)  # (B, npoint, nsample, C+3)
            else:
                new_points = grouped_xyz  # (B, npoint, nsample, 3)

        # 3. 特征提取（MLP + Max Pooling）
        new_points = new_points.permute(0, 3, 1, 2)  # (B, C+3, npoint, nsample)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))  # (B, mlp_out, npoint, nsample)
        new_points = torch.max(new_points, 3)[0]       # Max Pooling over nsample: (B, mlp_out, npoint)
        new_points = new_points.permute(0, 2, 1)       # (B, npoint, mlp_out)

        return new_xyz, new_points

class PointNet2ClassificationSSG(nn.Module):
    def __init__(self, num_classes=40):
        super(PointNet2ClassificationSSG, self).__init__()

        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32,
                                          in_channel=0, mlp_channels=[64, 64, 128],
                                          group_all=False)

        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64,
                                          in_channel=128, mlp_channels=[128, 128, 256],
                                          group_all=False)

        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None,
                                          in_channel=256, mlp_channels=[256, 512, 1024],
                                          group_all=True)

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)

        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, xyz):
        """
        xyz: (B, N, 3)
        """
        B, N, C = xyz.shape
        points = None

        l1_xyz, l1_points = self.sa1(xyz, points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # l3_points: (B, 1, 1024)

        x = l3_points.view(B, 1024)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)

        x = self.fc3(x)  # logits

        return x

if __name__ == "__main__":
    model = PointNet2ClassificationSSG(num_classes=10)
    xyz = torch.rand(8, 1024, 3)  # (B, N, 3)
    logits = model(xyz)
    print("Output logits shape:", logits.shape)  # (8, 10)
```

