### MobileNet series：

The MobileNet series is a collection of lightweight convolutional neural networks (CNNs) proposed by Google, primarily designed for efficient tasks such as image classification, object detection, and semantic segmentation on mobile and embedded devices. The goal of the MobileNet series is to reduce computational costs while maintaining high accuracy, enabling the deployment of deep learning models on resource-constrained devices.

#### MobileNetV1：

In recent years, both academia and industry have shown significant interest in building small yet efficient neural networks. Research in this area can be broadly categorized into two approaches: one involves compressing pre-trained large networks (e.g., through pruning, quantization, etc.), and the other involves directly designing and training small networks. MobileNetV1 introduced a novel category of network architectures aimed at assisting developers in designing small networks tailored to the resource constraints (such as latency and model size) of specific application scenarios. Unlike many studies on small networks that focus solely on reducing the number of model parameters, MobileNetV1 not only aims to reduce model size but also specifically optimizes computational latency, enabling it to excel in resource-constrained environments such as mobile devices.

The core of MobileNetV1 lies in the following three techniques:

- Depthwise Separable Convolution: A replacement for standard convolution that significantly reduces computational complexity.
- Width Multiplier (Thinner Models): Achieves model slimming by reducing the number of channels in the network.
- Resolution Multiplier (Reduced Representation): Reduces computational load by lowering the input resolution.

#### **Depthwise Separable Convolution**:

- **Fundamental Principle**

  Depthwise separable convolution decomposes the standard convolution operation into two simpler operations: depthwise convolution and pointwise convolution.

  1. **Depthwise Convolution**: For each input channel, a separate (k×k) convolution kernel is applied. Here, k is the size of the convolution kernel. This means each input channel is convolved with only one kernel, and there is no cross-channel computation. The result is that each input channel produces one output channel, and these output channels are independent, with no information fusion between them. This operation focuses on extracting features along the spatial dimensions.
  2. **Pointwise Convolution**: A (1×1) convolution kernel is used to linearly combine the output channels from the depthwise convolution into the desired output channels. This step achieves information fusion between channels, generating the final output feature map.

  By separating spatial convolution and channel fusion, depthwise separable convolution significantly reduces computational load while maintaining the model's expressive power.

- **Computational Complexity Analysis**

  Assume the input feature map has dimensions H × W, with C_in input channels and C_out output channels, and the convolution kernel size is k × k. We can compare the computational load (in floating-point operations, FLOPs) between standard convolution and depthwise separable convolution:

  1. **Computational Load of Standard Convolution**:

     **FLOPs_standard = H × W × C_in × C_out × k²** Standard convolution processes both spatial and channel dimensions simultaneously, with computational cost growing quadratically with the number of input and output channels.

  2. **Computational Load of Depthwise Separable Convolution**:

     - **Depthwise Convolution**: **FLOPs_depthwise = H × W × C_in × k²** Each channel is convolved independently, involving only spatial computation.
     - **Pointwise Convolution**: **FLOPs_pointwise = H × W × C_in × C_out × 1 × 1** This convolution only handles linear combinations between channels.
     - **Total Computational Load**: **FLOPs_depthwise-separable = H × W × (C_in × k² + C_in × C_out)**

  **Comparison of Computational Load**: Compared to standard convolution, depthwise separable convolution reduces the computational load by approximately (k² + C_out) / (k² × C_out) times. Typically, when k = 3 (i.e., k² = 9) and C_out is large, the computational load can be reduced by a factor of 8-9. This significant efficiency gain makes MobileNetV1 highly suitable for resource-constrained devices.

  In the MobileNetV1 architecture, depthwise separable convolution is the primary building block. Pointwise convolution (i.e., 1×1 convolution) accounts for the majority of the computational load. Mathematically, 1×1 convolution can be viewed as matrix multiplication: the input feature map is flattened into a 2D matrix (with spatial positions as rows and channels as columns), which is then multiplied by the convolution kernel weight matrix to generate the output feature map. This operation efficiently accomplishes feature fusion between channels.

#### **Width Multiplier:**

Although the base MobileNetV1 already has a small model size and low latency, there may still be a need to further compress the model in certain scenarios. To this end, MobileNetV1 introduces the width multiplier α, which uniformly reduces the network's width at each layer, resulting in a "thinner" model.

- **Fundamental Principle**

  For each layer, the width multiplier α adjusts the number of input channels from M to αM and the number of output channels from N to αN. The value of α ranges from (0, 1], with common values including 1, 0.75, 0.5, and 0.25. When α = 1, it is the standard MobileNetV1; when α < 1, a smaller variant network is generated.

- **Computational Cost Analysis**

  After applying the width multiplier, the computational cost of depthwise separable convolution becomes:

  **FLOPs_α = D_K × D_K × αM × D_F × D_F + αM × αN × D_F × D_F**

  The primary effect of the width multiplier is to reduce the computational load and the number of parameters by a factor of α². For example, when α = 0.5, the computational cost and model size are reduced to approximately 1/4 of the original. This reduction significantly lowers resource requirements while maintaining a certain level of accuracy.

#### **Resolution Multiplier:**

To further reduce the computational load, MobileNetV1 introduces the resolution multiplier ρ, which decreases the spatial dimensions of the feature maps by reducing the input image resolution, thereby lowering the computational cost.

- **Fundamental Principle**

  - The resolution multiplier ρ is a hyperparameter ranging from (0, 1], used to proportionally reduce the input image resolution.
  - The original input resolution is H × W, and after applying the resolution multiplier, it becomes ρH × ρW. For example, if ρ = 0.5, the input resolution is halved.
  - This method directly affects the spatial dimensions of the feature maps at each layer, reducing the computational load across the entire network.

- **Computational Cost Analysis**

  After incorporating the resolution multiplier, the computational cost of depthwise separable convolution becomes:

  **FLOPs_ρ = D_K × D_K × M × (ρD_F) × (ρD_F) + M × N × (ρD_F) × (ρD_F)**

  Simplified:

  **FLOPs_ρ = ρ² × (D_K × D_K × M × D_F × D_F + M × N × D_F × D_F)**

  The primary effect of the resolution multiplier is to reduce the computational load by a factor of ρ². For example, when ρ = 0.5, the computational cost is reduced to 1/4 of the original. However, reducing the resolution may sacrifice some model accuracy, so a trade-off between performance and efficiency is necessary.

The width multiplier α and the resolution multiplier ρ can be used together to further optimize the total computational cost of the model:

**FLOPs_α,ρ = D_K × D_K × αM × (ρD_F) × (ρD_F) + αM × αN × (ρD_F) × (ρD_F)**

The total computational load is reduced by a factor of α² × ρ², providing greater flexibility in adjusting the model size.

#### Code：

```python
"""
Paper: MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
Link: https://arxiv.org/abs/1704.04861
The Number of Layers:
    - 1 Convolutional Layer
    - 13 Depthwise Separable Convolutional Layers(Echo Layer has Batch Normalization and ReLU Activation)
    - 1 Fully Connected Layer
    Total: 1 + 13*2 + 1 = 28 Layers

Parameters:
    - alpha: 1.0  total params: 4.2M
    - alpha: 0.75 total params: 2.6M
    - alpha: 0.5  total params: 1.3M
    - alpha: 0.25 total params: 0.47M

The results of the experiment match the results in the paper.
"""

import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride, alpha=1.0):
        super(DepthwiseSeparableConv, self).__init__()

        in_channels = int(alpha * in_channels)
        out_channels = int(alpha * out_channels)

        # Depthwise Convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)

        # Pointwise Convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.batch_norm1(x)
        x = self.relu1(x)

        x = self.pointwise(x)
        x = self.batch_norm2(x)
        x = self.relu2(x)

        return x
    
class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1000, alpha=1.0):
        super(MobileNetV1, self).__init__()
        self.alpha = alpha

        # Standard Convolution
        self.conv1 = nn.Conv2d(3, int(alpha*32), kernel_size=3, stride=2, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(int(alpha*32))
        self.relu1 = nn.ReLU(inplace=True)

        # Depthwise Separable Convolution
        self.conv2 = DepthwiseSeparableConv(32, 64, 1, alpha)
        self.conv3 = DepthwiseSeparableConv(64, 128, 2, alpha)
        self.conv4 = DepthwiseSeparableConv(128, 128, 1, alpha)
        self.conv5 = DepthwiseSeparableConv(128, 256, 2, alpha)
        self.conv6 = DepthwiseSeparableConv(256, 256, 1, alpha)
        self.conv7 = DepthwiseSeparableConv(256, 512, 2, alpha)
        self.conv8 = DepthwiseSeparableConv(512, 512, 1, alpha)
        self.conv9 = DepthwiseSeparableConv(512, 512, 1, alpha)
        self.conv10 = DepthwiseSeparableConv(512, 512, 1, alpha)
        self.conv11 = DepthwiseSeparableConv(512, 512, 1, alpha)
        self.conv12 = DepthwiseSeparableConv(512, 512, 1, alpha)
        self.conv13 = DepthwiseSeparableConv(512, 1024, 2, alpha)
        self.conv14 = DepthwiseSeparableConv(1024, 1024, 1, alpha)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(alpha*1024), num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224)
    model = MobileNetV1(num_classes=1000, alpha=1.0)
    y = model(x)
    print(f"Output shape: {y.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
```

### MobileNetV2:

MobileNetV2 is an improved version of MobileNetV1, proposed by the Google team in the paper "MobileNetV2: Inverted Residuals and Linear Bottlenecks" (Sandler et al., 2018).

It significantly enhances performance while maintaining lightweight characteristics, primarily relying on the following three key technological innovations:

- **Depthwise Separable Convolution**: Inherited from MobileNetV1, it is used to reduce computational load and the number of parameters.
- **Inverted Residuals**: An improvement on the traditional residual structure (Residual Block), making it more suitable for lightweight networks.
- **Linear Bottlenecks**: Optimizes the feature compression process to avoid information loss caused by non-linear activations.

#### **Depthwise Separable Convolution**:

Refer to the previous section for details.

#### **Inverted Residuals**:

The inverted residual structure is an improvement on the traditional residual structure (Residual Block in ResNet), inspired by the following observations:

**Traditional Residual Structure**:

- The input undergoes a 1×1 convolution to compress the number of channels (bottleneck), followed by a 3×3 convolution to process spatial features, and finally a 1×1 convolution to expand back to the original number of channels.
- Structure: Wide → Narrow → Wide (bottleneck in the middle).
- Information is preserved through a shortcut connection (input added directly to the output).

**Inverted Residual Structure**:

- Designed in reverse: Narrow → Wide → Narrow.
- **Specific Steps**:
  - **Expansion**: Uses a 1×1 convolution to expand the number of input channels from C_in to C_expand (typically C_expand = t × C_in, where t is the expansion factor, with a default value of t=6).
  - **Depthwise Convolution**: Applies a 3×3 depthwise convolution on the expanded high-dimensional features, maintaining the number of channels at C_expand.
  - **Projection**: Uses a 1×1 convolution to compress the number of channels back to C_out.
- If the number of input and output channels is the same, a shortcut connection is added.

The bottleneck design of traditional residuals is suitable for large networks but can lead to information loss in lightweight networks with fewer channels. The inverted residual structure avoids this issue by first expanding the number of channels.

**Computational Complexity Analysis**:

Assuming the input is H×W×C_in and the output is H×W×C_out, with an expansion factor t=6:

- **Expansion**: H×W×C_in×(t×C_in)
- **Depthwise Convolution**: H×W×(t×C_in)×9
- **Projection**: H×W×(t×C_in)×C_out
- **Total Computational Load**: H×W×(t×C_in² + 9×t×C_in + t×C_in×C_out)

Compared to directly using standard convolution, it remains efficient, and the shortcut connection preserves low-dimensional features.

#### **Linear Bottlenecks**

Linear bottlenecks refer to the practice of **not using non-linear activation functions (such as ReLU)** after the projection stage (i.e., the last 1×1 convolution) in the inverted residual structure, instead directly outputting a linear result.

ReLU (max(0, x)) discards the negative part of features, leading to significant information loss in low-dimensional spaces (when the number of channels is small). Experiments show that when expanding to high dimensions and then compressing back to low dimensions, preserving a linear output better retains feature information. The paper hypothesizes that the feature distribution in deep networks is close to a low-dimensional manifold. Operating in high-dimensional space (during the expansion phase) and then linearly projecting to low-dimensional space effectively preserves the manifold structure, whereas non-linear activations may disrupt this structure. Moreover, the high-dimensional features generated during the expansion phase are already rich enough, and the projection phase only requires linear mapping to retain key information. Experiments demonstrate that linear bottlenecks significantly improve model accuracy, especially when the number of channels is small. Compared to MobileNetV1 (which uses ReLU throughout), MobileNetV2 performs better in lightweight scenarios.

#### Code：

```python
"""
Paper: MobileNetV2: Inverted Residuals and Linear Bottlenecks
Link: https://arxiv.org/abs/1801.04381
The Number of Layers:
    - 1 Convolutional Layer
    - 7 Bottlenecks(Echo Bottleneck has Batch Normalization and ReLU Activation)
    - 1 Fully Connected Layer
    Total: 1 + 7*3 + 1 = 23 Layers

Parameters:
    - alpha: 1.0  total params: 3.5M
    - alpha: 0.75 total params: 2.2M
    - alpha: 0.5  total params: 1.2M
    - alpha: 0.25 total params: 0.5M

The results of the experiment match the results in the paper.
"""
import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion_factor):
        super(Bottleneck, self).__init__()
        self.stride = stride
        self.use_shortcut = stride == 1 and in_channels == out_channels
        hidden_dim = int(in_channels * expansion_factor)

        # Expansion Convolution
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        ) if expansion_factor != 1 else nn.Identity()

        # Depthwise Convolution
        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, 
                      padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )

        # Compression Convolution (Linear Bottleneck)
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = x
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.project(out)
        if self.use_shortcut:
            out = out + identity
        return out

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, alpha=1.0, dropout_rate=0.2):
        super(MobileNetV2, self).__init__()
        self.alpha = alpha

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, int(32 * alpha), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(int(32 * alpha)),
            nn.ReLU6(inplace=True)
        )

        bottlenecks = [
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ]

        layers = []
        in_channels = int(32 * alpha)
        for t, c, n, s in bottlenecks:
            out_channels = int(c * alpha)
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(Bottleneck(in_channels, out_channels, stride, t))
                in_channels = out_channels
        self.bottlenecks = nn.Sequential(*layers)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, int(1280 * alpha), kernel_size=1, bias=False),
            nn.BatchNorm2d(int(1280 * alpha)),
            nn.ReLU6(inplace=True)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(int(1280 * alpha), num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bottlenecks(x)
        x = self.conv2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    x = torch.randn(1, 3, 224, 224)
    model = MobileNetV2(num_classes=1000, alpha=0.25)
    y = model(x)
    print(f"Output shape: {y.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
```

### MobileNetV3:

The goal of MobileNetV3 is to optimize neural networks for mobile device CPUs, enabling high performance under conditions of low power consumption and limited computational resources. Building upon the foundations of MobileNetV1 and V2, the paper explores how to enhance the overall technical capabilities of mobile vision tasks by combining automated search algorithms with manual network design.

Compared to its predecessors, the innovations of MobileNetV3 include:

- **Hardware-Aware Optimization**: Designing network architectures by considering the latency and power consumption of actual hardware, such as mobile CPUs.
- **Synergy of Search and Design**: Combining Neural Architecture Search (NAS) and the NetAdapt algorithm to generate an initial model, followed by manual refinements for optimization.
- **Multi-Task Adaptability**: The model is not only used for image classification but also extended to object detection and semantic segmentation.

The original paper proposes two versions:

- **MobileNetV3-Large**: Targeted at high-resource scenarios, aiming for higher accuracy.
- **MobileNetV3-Small**: Targeted at low-resource scenarios, emphasizing efficiency.

#### Key Technological Innovations:

- **Hardware-Aware Neural Architecture Search (NAS)**: Utilizing a platform-aware NAS based on MnasNet, with the search objective being to optimize the trade-off between accuracy and latency. The NetAdapt algorithm is applied to further fine-tune the initial architecture generated by NAS. This results in an efficient initial network that serves as the basis for subsequent manual optimizations.
- **Inverted Residuals and Linear Bottlenecks** (Inherited from V2):
- **Squeeze-and-Excitation (SE) Module**: The SE module generates channel attention weights through global pooling and two fully connected layers, enhancing important features while suppressing less important ones. In MobileNetV3, the SE module is inserted after the depthwise convolution (rather than after the pointwise convolution) to reduce computational overhead. Additionally, the traditional sigmoid function is replaced with hard-sigmoid to improve computational efficiency.
- **Hard-Swish Activation Function**: While the Swish activation function is more powerful than ReLU, it is computationally complex. Hard-Swish is a quantization-friendly version of Swish that retains its non-linear advantages while reducing computational overhead on mobile devices. It is used in the deeper layers and layers with higher channel counts of the network, with ReLU6 still used in the shallower layers.

#### torchvision Provides Official Implementation of MobileNetV3

```python
from torchvision.models import mobilenet_v3_large, mobilenet_v3_small
model = mobilenet_v3_large(pretrained=True)
```

