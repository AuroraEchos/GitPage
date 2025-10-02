### 中文仇恨言论检测

随着社交媒体和互联网的快速发展，仇恨言论已成为全球性问题。它不仅侵犯个人尊严，还可能对社会稳定产生不利影响。因此，自动化检测仇恨言论显得尤为重要。本项目旨在基于深度学习构建一个中文仇恨言论检测系统。我们采用 RoBERTa 模型作为预训练语言编码器，并结合 BiGRU 和 TextCNN 从文本中提取多层次语义特征，进一步提高分类准确率。最终目标是为中文文本中的仇恨言论检测提供一个精准的工具。完整代码可在 [Github](https://github.com/AuroraEchos/HateSpeechDetection) 上获取。

### 方法概述

### 1. RoBERTa：预训练语言模型

RoBERTa（一种鲁棒优化的 BERT 预训练方法）是 BERT 的优化版本，通过在更大数据集上进行扩展训练来提升文本理解能力。我们使用 RoBERTa 作为文本编码器，为下游分类任务生成上下文语义表示。

### 2. BiGRU：双向门控循环单元

为了捕捉文本中的序列信息，我们引入了 BiGRU（双向门控循环单元）。BiGRU 是 GRU 的扩展版本，能够从句子的过去和未来上下文捕获依赖关系。这有助于模型理解文本的句法结构和情感倾向。

### 3. TextCNN：卷积神经网络

TextCNN 是一种基于 CNN 的方法，用于从文本中提取局部 n-gram 特征。通过应用不同大小的多个卷积核，TextCNN 能够捕获多尺度信息，增强模型对局部语义的理解能力。结合 RoBERTa 和 BiGRU，TextCNN 进一步提升了模型的特征提取能力。

### 4. 模型融合：RoBERTa + BiGRU + TextCNN

我们方法的核心是将 RoBERTa、BiGRU 和 TextCNN 结合，发挥各自优势进行文本分类。模型架构设计如下：

- **RoBERTa** 生成上下文词嵌入。
- **BiGRU** 提取序列特征并捕捉长距离依赖。
- **TextCNN** 捕获局部 n-gram 特征，提升短文本检测能力。
- 最后，通过**全连接层 (FC)** 整合这些特征，输出分类结果。

```python
class RoBertFusion(nn.Module):
    def __init__(self, Robert_model, gru_hidden_size=128, num_filters=100, kernel_sizes=[3, 4, 5]):
        super(RoBertFusion, self).__init__()
        self.bert = BertModel.from_pretrained(Robert_model)
        
        self.gru = nn.GRU(self.bert.config.hidden_size, 
                          gru_hidden_size, 
                          bidirectional=True, 
                          batch_first=True)
        
        self.textcnn = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, self.bert.config.hidden_size)) 
            for k in kernel_sizes
        ])

        self.fc = nn.Linear(gru_hidden_size * 2 + num_filters * len(kernel_sizes), 2)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        gru_outputs, _ = self.gru(last_hidden_state)
        gru_outputs = gru_outputs[:, -1, :]

        x = last_hidden_state.unsqueeze(1)
        cnn_outputs = [torch.relu(conv(x)).squeeze(3) for conv in self.textcnn]
        cnn_outputs = [torch.max(out, dim=2)[0] for out in cnn_outputs]
        cnn_outputs = torch.cat(cnn_outputs, dim=1)

        combined_features = torch.cat((gru_outputs, cnn_outputs), dim=1)

        logits = self.fc(combined_features)
        
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss, logits
        return logits
```

### 数据集

本项目使用了开源中文仇恨言论检测数据集 [COLD](https://github.com/thu-coai/COLDataset)，该数据集包含三个部分：

- **训练集（`train.csv`）**：包含用于训练模型的文本样本和标签（0 = 非仇恨言论，1 = 仇恨言论）。
- **验证集（`dev.csv`）**：在训练期间用于验证模型性能并微调超参数。
- **测试集（`test.csv`）**：用于最终评估，以测试模型的泛化能力。

每个 CSV 文件包含两列：

- **TEXT**：输入文本。
- **label**：目标标签（0 或 1）。

### 实验设置

### 1. 环境与工具

- **编程语言**：Python 3.12
- **深度学习框架**：PyTorch 2.5.1，CUDA 12.4
- **预训练模型**：Huggingface Transformers（中文 RoBERTa：`chinese-roberta-wwm-ext`）
- **数据处理工具**：pandas，scikit-learn
- **操作系统**：Ubuntu 22.04
- **硬件**：RTX 4090（24GB）

### 2. 模型训练

我们使用**小批量训练**，批量大小为 64。最大输入长度设置为 128 个 token，以适应硬件限制。模型训练 5 个 epoch，每个 epoch 遍历整个训练集。为防止过拟合，我们基于验证集性能应用**早停法**，并保存最佳模型。

### 3. 超参数设置

- **学习率**：1e-5
- **批量大小**：64
- **最大输入长度**：128
- **Epochs**：5

### 实验结果

在测试集上，我们获得了以下评估指标：

- **准确率**：82.85%
- **精确率**：74.15%
- **召回率**：87.00%
- **F1 分数**：80.06%

这些结果表明，RoBERTa、BiGRU 和 TextCNN 的组合模型在中文仇恨言论检测上表现良好，实现了高准确率并在精确率与召回率之间取得了较好的平衡。特别是高召回率表明模型在识别仇恨言论方面具有较强的敏感性。

### 结果分析

- **高召回率**：模型实现了较高的召回率，意味着它能够有效识别大多数仇恨言论实例。这减少了假阴性，确保了对有害内容的更好覆盖。
- **相对较低的精确率**：尽管模型对仇恨言论敏感，但其精确率略低，表明存在一些假阳性。这可以通过优化模型架构或增加更多训练数据来改进。
- **平衡的 F1 分数**：F1 分数为 80.06%，表明模型在精确率和召回率之间保持了稳健的平衡。

### 结论

本项目通过整合 RoBERTa、BiGRU 和 TextCNN，成功构建了一个高效的中文仇恨言论检测系统。该系统在识别仇恨言论方面表现出色，展现了实际应用价值。随着模型的持续优化和数据的扩展，它有潜力在更多现实场景中应用于内容审核和公共话语分析。