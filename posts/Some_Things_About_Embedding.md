# Embedding 介绍

下面从 Embedding 的基本概念出发，介绍一下对比学习、向量检索、相似度、召回与排序。

---

### Embedding 是什么？

我们首先需要回答一个问题：什么是 Embedding ？

Embedding 的准确定义是把离散的对象映射到连续向量空间中的表示方法，这个对象可以是：

- 文本：query、句子、段落、文档等

- 图片：图像、区域、视觉特征

- 用户：用户画像、行为序列

- 代码：函数、类、仓库片段等

- 多模态内容：图文、视频、语音等

形式上来看，一个 encoder 将输入 $x$ 映射成一个向量：

$$
z = f_{\theta}(x), \quad z \in \mathbb{R}^{d}
$$

其中 $d$ 通常是 $128, 384, 768, 1024, 1536, 3072$ 等维度。

它的核心目标就是语义上相近的对象，在向量空间应该距离更近；语义上不相关的对象，距离应该更远。

例如：

- 如何申请中国签证？

- 中国旅游签证的办理流程。

上述这两个句子表面词完全不同，但是语义很接近，所以它们的 embedding 应该相似。

而：

- 如何申请中国签证？

- Transformer 的注意力机制是什么？

这两个句子明显差异很大，embedding 应该距离较远。

---

### 如何判断两个 Embedding 是否接近？

这涉及到另外一个概念：相似度。向量检索中常见的相似度度量有三类。

- 点积相似度

- 余弦相似度

- 欧式距离

点积的公式为：

$$
sim(q, d) = q^\top d
$$

一般来说，点积越大，通常认为越相似。点积的计算速度非常快，适合大规模的向量检索，很多的向量数据库和 ANN 系统都支持 Maximum Inner Product Search，也就是 MIPS。不过它会受到向量模长的影响，如果某些向量的 norm 很大，那么即使方向不匹配，点积也可能比较大。

余弦相似度看的是两个向量的方向是否一致，而不是长度，数学公式如下：

$$
sim(q, d) = \frac{q^\top d}{\|q\| \|d\|}
$$

如果 embedding 已经做了 L2 normalization：

$$
\hat{q} = \frac{q}{\|q\|},\quad \hat{d} = \frac{d}{\|d\|}
$$

那么：

$$
\hat{q}^\top \hat{d} = \cos(q, d)
$$

也就是说，归一化后的点积等价于余弦相似度。这在文本检索中很常见。

最后一个是欧式距离数学公式如下：

$$
dist(q, d) = \|q - d\|_2
$$

距离越小 ，越相似。对于已经 L2 normalization 的向量，欧氏距离和余弦相似度本质上是等价排序：

$$
\|q - d\|_2^2 = 2 - 2q^\top d
$$

所以在 normalized embedding 上，用 cosine、dot product、L2 distance 通常只是在实现层面不同，排序结果可以一致。

---

### Embedding 是怎么训练出来的呢？

Embedding 模型的训练目标是让语义相关样本靠近，不相关样本远离。现代 embedding 模型最常见的训练范式是 对比学习。

对比学习的基本思想是：给定一个 anchor，让它和 positive 的相似度变高，和 negative 的相似度变低。

在文本检索里，一般有：

- query：如何办理中国旅游签证？

- positive：中国旅游签证需要准备护照、照片、申请表、行程单...

- negative：Transformer 模型使用自注意力机制建模 token 关系...

模型应该学习到：

$$
sim(q, d^+) > sim(q, d^-)
$$

常见的对比学习损失是 InfoNCE 形式：

$$
\mathcal{L} = -\log \frac{\exp\left(sim(q, d^+)/\tau\right)}{\exp\left(sim(q, d^+)/\tau\right) + \sum_i \exp\left(sim(q, d_i^-)/\tau\right)}
$$

其中：

- $q$：query embedding

- $d^+$：正样本文档 embedding

- $d_i^-$：负样本文档 embedding

- $sim(\cdot)$：相似度函数，通常是 dot product 或 cosine

- $\tau$：temperature，控制 softmax 分布的锐度

这个 loss 的作用是：增大 $q$ 和 $d^+$ 的相似度，降低 $q$ 和所有 $d_i^-$ 的相似度，让模型在一批候选中把正样本排到更前面。

在实际训练中，常用 in-batch negatives。

假设一个 batch 中有 $N$ 对 query-document：

$$
(q_1, d_1^+), (q_2, d_2^+), \dots, (q_N, d_N^+)
$$

对于 $q_i$，它自己的 $d_i^+$ 是正样本，其他 $d_j^+$ 都可以视为负样本。相似度矩阵：

$$
S_{ij} = sim(q_i, d_j)
$$

然后对每一行做 cross entropy，目标是让 diagonal 最大：

|     | d1  | d2  | d3  | d4  |
| --- | --- | --- | --- | --- |
| q1  | 高   | 低   | 低   | 低   |
| q2  | 低   | 高   | 低   | 低   |
| q3  | 低   | 低   | 高   | 低   |
| q4  | 低   | 低   | 低   | 高   |

这类训练方式非常高效，因为一个 batch 内天然产生大量负样本。

普通负样本太容易时，模型学不到细粒度区分能力。因此检索模型经常使用 **hard negatives**。

例如：

- query：苹果手机怎么更换电池？

- positive：iPhone 电池的更换流程与费用说明？

- easy negative：如何训练 Transformer 模型

- hard negative：苹果电脑 MacBook 电池维修价格

困难负样本和 query 词面相似，但语义或意图不完全匹配。训练时加入 hard negatives，可以显著提升检索鲁棒性。

常见 hard negative 来源包括：

- BM25 召回但人工标注为不相关的文档
- 旧版 embedding 模型召回但点击率低的结果
- 交叉编码器 reranker 判为低相关的高分候选
- 同主题但不同意图的样本
- 线上日志中的曝光未点击样本

---

### 训练完成后如何进行向量检索？

Embedding 训练好之后，向量检索流程一般是：

1. 离线将所有文档、商品、图片、代码片段编码成 embedding
2. 将 embedding 写入向量索引
3. 用户输入 query 时，实时编码 query embedding
4. 在向量库中查找最近邻
5. 返回 top-k 候选

形式上：

$$
q = f_{\theta}(\text{query}) \\
TopK = \mathop{\arg\max}_{d_i \in D} sim(q, d_i)
$$

如果文档量很小，可以直接暴力计算：

$$
sim(q, d_1), sim(q, d_2), \dots, sim(q, d_N)
$$

然后排序取 top-k。

但当文档量达到百万、千万、亿级时，暴力计算成本太高。这时通常使用 ANN：Approximate Nearest Neighbor，近似最近邻检索。

常见 ANN 索引包括：

- HNSW，全称 Hierarchical Navigable Small World，是当前非常常用的向量检索索引。查询速度快，召回率高，内存占用比较大，适合中大规模在线检索。很多向量数据库都支持 HNSW，例如 Milvus、Qdrant、Weaviate、OpenSearch、Elasticsearch、Vespa 等。

- IVF，全称 Inverted File Index。它先把向量空间聚类成多个中心，然后查询时只搜索离 query 最近的一部分簇。特点是更适合大规模数据，查询速度快，召回率取决于搜索多少个簇，通常和 Product Quantization 结合使用。

- PQ 是 Product Quantization，用较小的码表示高维向量，减少存储与计算成本。特点是显著降低内存占用，适合超大规模向量数据库，但是会损失一定精度。

---

### 什么是召回与排序？

召回就是先把可能相关的候选找出来，在搜索、推荐、RAG 中，**召回** 的目标不是直接给最终答案，而是从海量候选中快速找出一个较小的候选集。

例如：

- 总文档量：100,000,000

- 召回阶段：取 top 1000

- 排序阶段：从 1000 个里选 top 10

召回阶段追求的是：宁可多拿一些候选，也不要漏掉真正相关的结果。

因此召回更看重：Recall@K、覆盖率、延迟、吞吐、成本、索引更新效率，而不是最终排序精度。

召回分为向量召回和关键词召回，关键词召回依赖 token overlap，常见的方法例如 BM25，向量召回依赖语义相似度，适合 RAG 中的语义文档召回，但是向量召回依赖 embedding 模型质量。实际系统中，常常不会只用一种召回方式，而是混合：

- BM25 召回 top 1000

- Embedding 召回 top 1000

- 规则/业务召回若干

- 合并去重

- 进入排序阶段

常见融合方法包括：

线性加权：

$$
score = \alpha \cdot score_{\text{dense}} + (1 - \alpha) \cdot score_{\text{bm25}}
$$

RRF 倒数排序融合：

$$
RRF(d) = \sum_{i} \frac{1}{k + rank_i(d)}
$$

RRF 不太依赖不同检索器分数尺度，工程上很常用。

召回阶段通常比较粗。排序阶段负责更精细地判断相关性。典型检索系统分为：

```
Query
  ↓
多路召回
  ↓
候选合并
  ↓
粗排
  ↓
精排 / Rerank
  ↓
最终结果
```

---

### Bi-Encoder 与 Cross-Encoder

Embedding 检索通常使用 **Bi-Encoder**。

Bi-Encoder 分别编码 query 和 document：

$$
q = f(query) \\
d = g(document)
$$

然后用向量相似度计算相关性：

$$
score(q, d) = q^Td
$$

它的优点是文档 embedding 可以离线预计算，查询时只需要编码 query，可以使用 ANN 快速检索，适合大规模召回。缺点是 query 和 document 之间交互较弱，对细粒度匹配、复杂推理不如 Cross-Encoder。

Cross-Encoder 将 query 和 document 拼在一起输入模型：

`[CLS] query [SEP] document [SEP]`

模型直接输出相关性分数。

优点是 query 和 document 可以充分 token-level interaction ，排序精度高，更能识别细粒度语义、否定、条件、实体关系。缺点就是无法离线预先计算，每个 query-document pair 都要跑一次模型，成本高，延迟大，不适合作为亿级召回器。

所以常见的架构是：

- Bi-Encoder 负责召回 top 100~1000

- Cross-Encoder 负责 rerank top 20~200

---

### RAG 中的 Embedding 检索

在 RAG 中，Embedding 通常用于从知识库中召回相关 chunks。

典型流程：

```
User Query
  ↓
Query Embedding
  ↓
Vector Search
  ↓
Top-k Chunks
  ↓
Rerank / Filter
  ↓
LLM Context
  ↓
Answer
```

一个常见问题是：**相似度高不等于能回答问题**。

例如用户问："张三在 2024 年 Q3 的 OKR 完成情况如何？"

向量检索可能召回："张三 2024 年 Q2 OKR 总结"

它语义很近，但时间不匹配。

所以 RAG 系统中通常还需要：

- metadata filter
- 时间过滤
- 权限过滤
- 结构化字段约束
- reranker
- query rewrite
- multi-query retrieval
- hybrid search
- chunk-level 与 document-level 混合索引

---

### 召回与排序的评价指标

**召回指标：**

Recall@K：

$$
Recall@K = \frac{\text{top K 中相关文档数量}}{\text{所有相关文档数量}}
$$

例如总共有 5 个相关文档，top 10 召回了 3 个：$Recall@10 = 3/5 = 0.6$ 。

Hit@K：如果 top K 中至少有一个相关结果，则为 1，否则为 0。适合问答或单答案场景。

MRR（Mean Reciprocal Rank）：

$$
MRR = \frac{1}{N}\sum_{i} \frac{1}{rank_i}
$$

关注第一个相关结果排在多靠前。

**排序指标：**

Precision@K：

$$
Precision@K = \frac{\text{top K 中相关文档数量}}{K}
$$

NDCG@K：

NDCG 考虑相关性等级和排序位置。高相关结果排得越靠前，NDCG 越高。它适合多等级相关性，例如：不相关、部分相关、相关、高度相关。

MAP（Mean Average Precision）：常用于信息检索任务。

---

### Embedding 系统中的常见问题

1. 语义相似但任务不相关

2. 对数字、时间、版本不敏感

3. Chunk 粒度不合适

4. 向量空间坍缩或分布异常

这些都有对应的解决方案，当然并不能做到完美。

---

### 召回和排序的工程架构

一个工业级检索系统通常长这样：

```
               ┌───────────────┐
               │ User Query     │
               └───────┬───────┘
                       ↓
               ┌───────────────┐
               │ Query Rewrite  │
               └───────┬───────┘
                       ↓
        ┌──────────────┼──────────────┐
        ↓              ↓              ↓
   BM25 Recall   Vector Recall   Rule Recall
        ↓              ↓              ↓
        └──────────────┼──────────────┘
                       ↓
               Candidate Merge
                       ↓
               Dedup / Filter
                       ↓
                 Lightweight Rank
                       ↓
                 Cross-Encoder
                       ↓
                Business Rules
                       ↓
                 Final Top-K
```

在 RAG 中，后面通常还会接：

```
Context Packing
  ↓
LLM Generation
  ↓
Citation / Attribution
  ↓
Answer
```

---

### 简要总结上述概念

| 概念        | 作用                         |
| --------- | -------------------------- |
| Embedding | 把文本、图片、商品等对象映射为向量          |
| 对比学习      | 训练 embedding 空间，使正样本近、负样本远 |
| 相似度       | 衡量两个向量是否接近                 |
| 向量检索      | 在大规模向量库中找最近邻               |
| 召回        | 从海量候选中找出可能相关的一批            |
| 排序        | 对候选做更精细的相关性判断              |

做一个高质量 Embedding 检索系统时，建议重点关注这些问题：

1. Embedding 模型是否匹配领域。通用 embedding 在开放域表现不错，但在法律、医疗、金融、代码、内部知识库中可能不够好。领域数据微调通常很重要。

2. 相似度和归一化是否一致。使用 cosine 时，最好确认向量是否 L2 normalization。索引构建、查询、训练时的 similarity metric 要一致。

3. 负样本质量比样本数量更关键。大量随机负样本不如少量高质量 hard negatives。检索模型的上限往往取决于负样本构造。

4. 不要只依赖向量召回。生产系统里通常需要 hybrid search。BM25 负责精确匹配，dense embedding 负责语义泛化。

5. 召回和排序分开优化。召回阶段关注覆盖率和效率，排序阶段关注最终相关性。不要指望一个向量相似度同时解决所有问题。

6. RAG 中要重视 chunk 和 metadata。很多 RAG 效果差不是 embedding 模型弱，而是 chunk 粒度、metadata filter、索引结构和 rerank 没做好。
