### 技术报告：基于 Residual 3D CNN 的雷达波束预测方法

#### 1.引言

随着 **6G 通信** 与 **自动驾驶感知** 的快速发展，高效的波束赋形 (Beamforming) 成为保障高速率与低时延通信的关键技术。然而，传统依赖信道估计的波束选择方法存在计算开销大、时延高的问题。本文提出一种基于 **复数输入与残差三维卷积神经网络 (Residual 3D CNN)** 的雷达波束预测方法，利用雷达回波在 **时域、频域与空间域** 的联合特征，实现高效、低开销的波束选择。在 **DeepSense 6G** 数据集上的实验结果表明，该方法在 Top-5 精度上达到 **90.39%**，平均波束距离为 **1.2985**，显著优于传统基线方法，展示了其在未来智能通信与感知一体化系统中的潜力。请从[此处](https://github.com/AuroraEchos/Radar-Aided-Beam-Prediction/tree/main)获取完整代码。

#### 2.数据预处理

**2.1 雷达信号模型**

雷达原始接收信号可以表示为：
$$
s_{rx}(t) = \sum_{k=1}^{K} \alpha_k e^{j(2\pi f_{d,k} t + \phi_k)},
$$
其中：

- $\alpha_k$为目标反射系数，
- $f_{d,k}$为多普勒频移，
- $\phi_k$为相位。

通过 **快速傅里叶变换 (FFT)**，可在 **距离 (Range)**、**多普勒 (Doppler)** 和 **角度 (Angle)** 维度提取目标特征。

**2.2 Range-Angle-Doppler (RAD) Cube 生成**

1. **距离 FFT**（沿采样点维度）：
   $$
   S_{\text{range}}(r) = \sum_{n=0}^{N-1} s_{rx}(n) e^{-j 2\pi rn/N}
   $$
   
2. **多普勒 FFT**（沿 chirp 维度）：
   $$
   S_{\text{doppler}}(d) = \sum_{m=0}^{M-1} S_{\text{range}}(m) e^{-j 2\pi dm/M}
   $$
   
3. **角度 FFT**（沿天线维度）：
   $$
   S_{\text{angle}}(\theta) = \sum_{a=0}^{A-1} S_{\text{doppler}}(a) e^{-j 2\pi a \sin(\theta)/\lambda}.
   $$

最终得到三维谱：
$$
RAD(\theta, r, d) \in \mathbb{C}^{N_r \times N_\theta \times N_d}
$$
**2.3 实部与虚部作为输入通道**

传统方法取幅度：
$$
|RAD| = \sqrt{\Re(RAD)^2 + \Im(RAD)^2}
$$
但这会丢失相位信息。

在本方法中，采用 **双通道输入**：
$$
X = [\Re(RAD), \Im(RAD)] \in \mathbb{R}^{2 \times N_d \times N_\theta \times N_r}
$$
这样既保留了能量分布，又保留了相位信息，有助于角度估计和多普勒估计。

#### 3.模型设计

**3.1 模型架构**

输入张量：
$$
X \in \mathbb{R}^{B \times 2 \times D \times H \times W}
$$
网络结构如下：

1. **ResidualBlock3D** 堆叠（提取 3D 局部特征）
2. **自适应池化 (Adaptive Pooling)**（统一输出维度，减少参数）
3. **全连接层 + Dropout**（分类器）

**3.2 残差块**

Residual Block 的计算为：
$$
y = \mathcal{F}(x, W) + x
$$
其中$\mathcal{F}(x, W)$为卷积 + BN + ReLU。该结构缓解了 **梯度消失** 和 **网络退化问题**，保证更深的网络仍能有效训练。

**3.3 前向传播**

经过 4 层残差块：
$$
h_1 = \text{ResBlock}(X), \\
h_2 = \text{ResBlock}(h_1), \\
h_3 = \text{ResBlock}(h_2), \\
h_4 = \text{ResBlock}(h_3), \\
$$
池化：
$$
h_p = AdaptiveAvgPool3D(h_4)
$$
展开并分类：
$$
z = softmax(W_2(ReLU(W_1h_p)))
$$
最终输出：
$$
\hat{y} = argmax(z)
$$

#### 4.训练与优化

损失函数采用 **交叉熵损失**：
$$
\mathcal{L} = - \sum_{i=1}^{C} y_i \log(\hat{y}_i)
$$
其中$C=64$为波束数目，$\hat{y}$为 one-hot 标签。

在 DeepSense 6G 数据集中，场景 9 的开发集被划分为训练集、验证集与测试集，比例约为 $7:2:1$。其中测试集严格独立，仅用于最终评估。

采用 Adam 优化器，初始学习率设为 $1\times 10^{-4}$，并使用 $\textit{StepLR} $学习率调度策略，每 10 个 epoch 衰减一次。  批量大小（batch size）设为 $B=64$，训练总轮次为 40。

为了防止过拟合，在全连接层前加入 Dropout 层，丢弃率为 $p=0.5$。此外，批归一化（Batch Normalization）被引入残差块中以稳定训练。

#### 5. 实验结果

**5.1 评估指标**
采用以下指标进行性能评估：

- Top-$k$ 准确率：
$$
\text{Top-}k = \frac{1}{N}\sum_{i=1}^N \mathbf{1}(y_i \in \text{Top-}k(\hat{y}_i)),
$$
其中 $\hat{y}_i$ 为预测概率分布。
- 波束距离 (Beam Distance)：
$$
\text{BeamDist} = \frac{1}{N}\sum_{i=1}^N |y_i - \hat{y}_i^{(1)}|,
$$
其中 $\hat{y}_i^{(1)}$ 表示预测的 Top-1 波束。

**5.2 实验结果**

| Top-1  | Top-2 | Top-3 | Top-4 | Top-5 | BeamDist |
| ------ | ----- | ----- | ----- | ----- | -------- |
| 41.48% | 60.37 | 75.04 | 85.50 | 90.39 | 1.2985   |

结果表明：

1. Top-1 精度为 41.48%，但 Top-5 达到 90.39%，说明模型能够有效提供候选波束集合；
2. BeamDist = 1.2985，误差集中在相邻波束，证明模型捕捉到空间连续性特征；
3. 与传统方法相比，本方法在低复杂度前提下取得更优性能。