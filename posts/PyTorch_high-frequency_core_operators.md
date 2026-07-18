# PyTorch 高频核心算子

这份笔记用于记录一些 Pytorch 框架下的高频算子，Pyotrch 核心其实就三类：

```
形状操作（view/permute）
+
线性代数（matmul）
+
mask + softmax（attention）
```

## 1. 张量形状操作（view、reshape、flatten）

   一个 Tensor 由三个部分组成：storage（底层一维内存）；shape（逻辑形状）；stride（步长，决定如何从 storage 映射到多维）。view/reshape/flatten 本质上都是在“改 shape + stride”，而不是改数据。

   view 是纯视图变换，定义如下：

   ```python
   x = x.view(new_shape)
   ```

   view 不拷贝内存，仅改变张量的“解释方式”，使用时必须满足 Tensor 在内存中是连续的。例如：

   ```python
   x = torch.arange(6)
   print(x)
   x = x.view(2, 3)
   print(x)
   
   # 输出如下：
   tensor([0, 1, 2, 3, 4, 5])
   tensor([[0, 1, 2],
           [3, 4, 5]])
   ```

   permute、transpose 之后的张量不连续，不能直接用 view。连续的定义是：逻辑上相邻的元素，在底层内存里也是相邻的。例如一个张量：

   ```python
   tensor([[0, 1, 2],
           [3, 4, 5]])
   ```

   底层内存是连续的一排：[0,1,2,3,4,5]，逻辑上 0 旁边是 1，内存里 0 旁边也是 1，所以是连续的。如果我们做了 transpose（转置），转置后的逻辑形状变为：

   ```python
   tensor([[0, 3],
           [1, 4],
           [2, 5]])
   ```

   底层内存数据完全没变！还是：[0,1,2,3,4,5]，现在问题来了，逻辑上相邻的数字：0 旁边应该是 3、1 旁边应该是 4、2 旁边应该是 5，但底层内存里：0 旁边是 1、1 旁边是 2、2 旁边是 3，逻辑相邻 ≠ 内存相邻，那么就是不连续。PyTorch 张量内存连续的判定公式如下，可以自己学习理解：
   $$
   \text{stride}[i] = \text{stride}[i+1] \times \text{size}[i+1]
   $$
   reshape 可以看作是一个安全版的 view，定义如下：

   ```python
   x = x.reshape(new_shape)
   ```

   view 要求必须连续，不连续就报错。reshape 不要求连续，不连续就自动拷贝，永远不报错。如果张量内存连续，reshape = view（不拷贝数据，纯改形状）；如果张量内存不连续，reshape 自动调用 contiguous() 拷贝数据，变成连续后再 view。例如：

   ```python
   x = torch.arange(6)
   print(x)
   
   x = x.view(2, 3)
   print(x)
   
   x = x.transpose(0, 1)   # 内存不连续
   print(x)
   
   y = x.contiguous()      # 内存连续
   print(y)
   
   z = y.view(3, 2)        # 内存连续才能改变形状
   print(z)
   
   # 输出：
   tensor([0, 1, 2, 3, 4, 5])
   tensor([[0, 1, 2],
           [3, 4, 5]])
   tensor([[0, 3],
           [1, 4],
           [2, 5]])
   tensor([[0, 3],
           [1, 4],
           [2, 5]])
   tensor([[0, 3],
           [1, 4],
           [2, 5]])
   ```

   下面给出一些工程上的建议：如果场景对性能比较敏感，那么就使用 `view + contiguous`；如果写代码追求稳一些，那么就使用 `reshape`。

   下面介绍一下 flatten，flassten 就是把张量 “拍扁”，变成一维（或从某一维开始压扁），它本质上就是 reshape 的简化版，专门用来展平张量。最简单用法：

   ```python
   x = torch.randn(2, 3, 4)    # shape: [2, 3, 4]
   y = x.flatten()            	# shape: [24]
   ```

   等价于：

   ```python
   y = x.reshape(-1)			# shape: [24]
   ```

   在神经网络中，最常用、最重要的用法：

   ```python
   x.flatten(start_dim=1)
   ```

   意思：从第 1 维开始往后全部压扁，第 0 维保持不动，第 0 维一般是 batch，不能压扁！

   ```python
   x = torch.randn(32, 128, 7, 7)  # [batch, channel, H, W]
   x = x.flatten(start_dim=1)      # [32, 128*7*7] = [32, 6272]
   ```

   这就是CNN 全连接层前的标准操作。

## 2. 张量操作（unsqueeze / squeeze）

   这两个算子专门用来增加 / 删除 “长度 = 1” 的维度，是维度对齐、广播、拼接的必备工具。

   - unsqueeze = 加一维（size=1）
   - squeeze   = 删一维（size=1）

   例如：

   ```python
   import torch
   
   x = torch.randn(3)      # shape: [3]
   y = x.unsqueeze(0)      # 在第0维加一维
   z = x.unsqueeze(1)      # 在第1维加一维
   
   print(y.shape)  # [1, 3]
   print(z.shape)  # [3, 1]
   
   x = torch.randn(1, 3, 1)   	# shape: [1, 3, 1]
   y = x.squeeze()             # 删掉所有 size=1 的维度
   
   print(y.shape)  # [3]
   
   x = torch.randn(1, 3, 1)
   y = x.squeeze(0)   # 只删第0维
   print(y.shape)     # [3, 1]
   ```

## 3. 张量操作（permute / transpose）

   transpose：只能交换两个维度；permute：可以任意重排所有维度。共同点：只改维度顺序、只改 stride，不拷贝底层数据，操作后张量**一定不连续**。例如：

   ```python
   import torch
   # transpose
   x = torch.randn(2, 3, 4)  # shape [2,3,4]
   y = x.transpose(1, 2)     # 交换 1、2 维
   print(y.shape)            # [2,4,3]
   
   # permute
   x = torch.randn(2, 3, 4)  	# [B, C, D]
   y = x.permute(1, 0, 2)     	# 重排：原来1维、0维、2维
   print(y.shape)             	# [3,2,4]
   ```

   深度学习经典通道变换：

   ```python
   # NCHW -> NHWC
   # [B, C, H, W] → [B, H, W, C]
   x.permute(0, 2, 3, 1)
   ```

## 4. 拼接与拆分（cat、stack、split、chunk）

   torch.cat() 在现有维度上拼接，维度数量不变，只在某一维上变长。拼接维度的长度相加，其他维度必须完全相同。

   ```python
   a = torch.randn(3, 4)
   b = torch.randn(3, 4)
   
   c = torch.cat([a, b], dim=0)  # 在第0维拼接
   print(c.shape)  # (6, 4)
   
   d = torch.cat([a, b], dim=1)  # 在第1维拼接
   print(d.shape)  # (3, 8)
   ```

   torch.stack() 新增一个维度堆叠，新建一个维度，把多个张量叠起来。多个张量变成一个 batch，所有张量 shape 必须完全一样。

   ```python
   a = torch.randn(3, 4)
   b = torch.randn(3, 4)
   
   c = torch.stack([a, b], dim=0)
   print(c.shape)  # (2, 3, 4)
   ```

   