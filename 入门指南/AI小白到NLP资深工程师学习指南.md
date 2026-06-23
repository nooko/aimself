# AI 小白到 NLP 资深工程师学习指南

## 学习路线总览

```
阶段一：数学与编程基础 (2-3个月)
    ↓
阶段二：机器学习基础 (2-3个月)
    ↓
阶段三：深度学习核心 (2-3个月)
    ↓
阶段四：NLP基础与经典方法 (2-3个月)
    ↓
阶段五：预训练语言模型时代 (3-4个月)
    ↓
阶段六：大语言模型与前沿技术 (3-6个月)
    ↓
阶段七：工程化与系统设计 (持续)
```

---

## 阶段一：数学与编程基础

### 数学基础

| 学科 | 核心内容 | 推荐资源 |
|------|----------|----------|
| 线性代数 | 向量、矩阵运算、特征值分解、SVD | MIT 18.06 (Gilbert Strang) |
| 概率统计 | 贝叶斯定理、分布、极大似然估计、假设检验 | 《统计学习方法》前两章 |
| 微积分 | 偏导数、链式法则、梯度、泰勒展开 | 3Blue1Brown 微积分系列 |
| 优化理论 | 梯度下降、凸优化基础、拉格朗日乘数法 | Boyd《Convex Optimization》前几章 |

### 编程基础

- **Python 精通**：数据结构、面向对象、装饰器、生成器、上下文管理器
- **NumPy / Pandas**：矩阵运算、数据处理
- **Linux 基础**：命令行操作、shell脚本、环境管理
- **Git**：版本控制、协作开发

### 实践项目

- 用 NumPy 手写矩阵分解
- 实现梯度下降求解线性回归

---

## 阶段二：机器学习基础

### 核心算法

1. **监督学习**：线性回归、逻辑回归、SVM、决策树、随机森林、XGBoost
2. **无监督学习**：K-Means、PCA、GMM
3. **模型评估**：交叉验证、偏差-方差权衡、正则化、过拟合诊断

### 重要概念

- 损失函数设计
- 特征工程（特征选择、特征变换）
- 模型选择与超参数调优
- 集成学习（Bagging、Boosting、Stacking）

### 推荐资源

- 吴恩达 Coursera Machine Learning / Machine Learning Specialization
- 周志华《机器学习》（西瓜书）
- scikit-learn 官方文档与教程

### 实践项目

- Kaggle 入门竞赛（Titanic、House Prices）
- 用 scikit-learn 完成完整的 ML Pipeline

---

## 阶段三：深度学习核心

### 基础架构

| 模型 | 关键点 |
|------|--------|
| MLP | 反向传播、激活函数、BatchNorm、Dropout |
| CNN | 卷积、池化、ResNet、感受野 |
| RNN/LSTM/GRU | 序列建模、梯度消失、门控机制 |
| Attention | Self-Attention、Multi-Head Attention |
| Transformer | 位置编码、编码器-解码器架构 |

### 框架掌握

- **PyTorch**（首选）：Tensor操作、autograd、nn.Module、DataLoader、分布式训练
- 能手写训练循环、自定义模型、自定义数据集

### 训练技巧

- 学习率调度（Warmup、Cosine Annealing）
- 混合精度训练（AMP）
- 梯度裁剪、梯度累积
- 权重初始化策略

### 推荐资源

- 李宏毅深度学习课程
- 《Dive into Deep Learning》(d2l.ai)
- Andrej Karpathy 系列视频（Neural Networks: Zero to Hero）

### 实践项目

- 手写 Transformer（from scratch）
- 图像分类（CIFAR-10）
- 序列标注（命名实体识别）

---

## 阶段四：NLP 基础与经典方法

### 语言学基础

- 分词、词性标注、句法分析
- 语言模型（N-gram）
- 信息论基础（熵、互信息、困惑度）

### 经典 NLP 方法

1. **文本表示**：TF-IDF、Word2Vec、GloVe、FastText
2. **序列标注**：HMM、CRF、BiLSTM-CRF
3. **文本分类**：TextCNN、TextRNN、Attention机制
4. **序列到序列**：Encoder-Decoder、Attention Mechanism、Beam Search
5. **信息抽取**：关系抽取、事件抽取

### 核心任务

- 命名实体识别（NER）
- 情感分析
- 文本摘要
- 机器翻译
- 问答系统

### 推荐资源

- Stanford CS224N（NLP with Deep Learning）
- Jurafsky & Martin《Speech and Language Processing》
- 《Neural Network Methods for Natural Language Processing》

### 实践项目

- 训练 Word2Vec 并可视化词向量
- BiLSTM-CRF 命名实体识别
- Seq2Seq 机器翻译（英中）

---

## 阶段五：预训练语言模型时代

### 里程碑模型

| 模型 | 年份 | 核心贡献 |
|------|------|----------|
| ELMo | 2018 | 上下文相关词向量 |
| GPT | 2018 | 单向 Transformer + 预训练微调范式 |
| BERT | 2018 | 双向 Transformer、MLM + NSP |
| XLNet | 2019 | 排列语言模型 |
| RoBERTa | 2019 | BERT 训练策略优化 |
| T5 | 2019 | Text-to-Text 统一框架 |
| GPT-2/3 | 2019/2020 | 大规模自回归、In-Context Learning |

### 核心技能

- **预训练**：MLM、CLM、去噪目标
- **微调**：分类、序列标注、生成任务的微调范式
- **Prompt Engineering**：Prompt设计、Few-shot、Chain-of-Thought
- **参数高效微调**：LoRA、Adapter、Prefix-Tuning、P-Tuning

### 工具链

- HuggingFace Transformers / Datasets / Tokenizers
- DeepSpeed / FSDP（分布式训练）
- Weights & Biases（实验跟踪）

### 推荐资源

- 精读论文：Attention Is All You Need、BERT、GPT系列
- HuggingFace 官方课程（huggingface.co/course）
- 各模型官方代码仓库

### 实践项目

- BERT 微调做文本分类 / NER
- 用 T5 做文本摘要
- 实现 LoRA 微调

---

## 阶段六：大语言模型与前沿技术

### LLM 核心知识

1. **模型架构演进**
   - Decoder-Only 架构（GPT系列、LLaMA、Mistral）
   - MoE（Mixture of Experts）
   - 长上下文处理（RoPE、ALiBi、Ring Attention）

2. **训练流程**
   - 预训练数据工程（数据清洗、去重、配比）
   - SFT（Supervised Fine-Tuning）
   - RLHF / DPO / ORPO（对齐技术）

3. **推理优化**
   - KV-Cache
   - 量化（GPTQ、AWQ、GGUF）
   - 推测解码（Speculative Decoding）
   - vLLM、TensorRT-LLM

4. **RAG（检索增强生成）**
   - 向量数据库（Milvus、Chroma、FAISS）
   - Embedding 模型
   - 检索策略（Hybrid Search、Reranking）
   - 文档解析与分块

5. **Agent 与工具使用**
   - Function Calling
   - ReAct / Plan-and-Execute
   - Multi-Agent 框架

### 前沿方向

- 多模态大模型（Vision-Language Model）
- 代码生成与推理
- 长文本理解与生成
- 模型蒸馏与小模型
- 合成数据

### 推荐资源

- LLaMA / Mistral / Qwen 系列论文与代码
- Andrej Karpathy: Let's build GPT from scratch
- LangChain / LlamaIndex 文档
- Sebastian Raschka《Build a Large Language Model From Scratch》

---

## 阶段七：工程化与系统设计

### 模型服务化

- 模型部署（Triton Inference Server、vLLM、TGI）
- API 设计（流式输出、并发控制、负载均衡）
- 模型版本管理与 A/B 测试
- 监控与告警（延迟、吞吐、错误率）

### 数据工程

- 数据标注系统设计
- 数据飞轮（Data Flywheel）
- 主动学习（Active Learning）
- 数据质量评估

### 系统设计能力

- 搜索系统（Query理解 → 召回 → 排序 → 重排）
- 对话系统（意图识别 → 槽位填充 → 对话管理 → 回复生成）
- 内容审核系统
- 知识图谱与问答系统

### 软技能

- 技术方案设计与评审
- 论文阅读与复现能力
- 开源社区参与
- 技术写作与分享

---

## 学习原则与建议

### 核心原则

1. **论文驱动**：每个阶段精读 5-10 篇核心论文，理解动机而非只看结果
2. **代码先行**：每学一个模型/算法，必须动手实现或微调
3. **项目驱动**：用真实项目串联知识点，而非孤立学习
4. **迭代深入**：先广度再深度，每轮学习都比上一轮更深

### 时间分配建议

```
理论学习：30%
代码实践：40%
论文阅读：20%
社区交流：10%
```

### 推荐学习节奏

- 工作日：每天 1-2 小时专注学习
- 周末：3-4 小时项目实践
- 每月：精读 2-3 篇论文
- 每季度：完成一个端到端项目

### 必读论文清单（Top 20）

1. Word2Vec (Mikolov et al., 2013)
2. GloVe (Pennington et al., 2014)
3. Sequence to Sequence (Sutskever et al., 2014)
4. Attention Is All You Need (Vaswani et al., 2017)
5. ELMo (Peters et al., 2018)
6. GPT (Radford et al., 2018)
7. BERT (Devlin et al., 2018)
8. XLNet (Yang et al., 2019)
9. RoBERTa (Liu et al., 2019)
10. T5 (Raffel et al., 2019)
11. GPT-3 (Brown et al., 2020)
12. LoRA (Hu et al., 2021)
13. InstructGPT / RLHF (Ouyang et al., 2022)
14. Chain-of-Thought (Wei et al., 2022)
15. LLaMA (Touvron et al., 2023)
16. DPO (Rafailov et al., 2023)
17. Mistral / Mixtral (Jiang et al., 2023)
18. RAG (Lewis et al., 2020)
19. ReAct (Yao et al., 2022)
20. Qwen Technical Report (2023/2024)

---

## 能力自测检查表

### 初级 NLP 工程师

- [ ] 能用 PyTorch 搭建并训练基础模型
- [ ] 理解 Transformer 架构每个组件
- [ ] 能用 HuggingFace 微调 BERT 做下游任务
- [ ] 掌握基本文本预处理与特征工程
- [ ] 了解常见 NLP 任务与评估指标

### 中级 NLP 工程师

- [ ] 能从零实现 Transformer 并调通训练
- [ ] 熟练使用 LoRA 等参数高效微调方法
- [ ] 掌握分布式训练（DDP / FSDP）
- [ ] 能设计并实现完整 NLP Pipeline
- [ ] 能读懂并复现前沿论文
- [ ] 具备模型部署与优化经验

### 资深 NLP 工程师

- [ ] 深入理解 LLM 训练全流程（预训练→SFT→对齐）
- [ ] 能设计大规模 NLP 系统架构
- [ ] 掌握推理优化（量化、KV-Cache、推测解码等）
- [ ] 具备 RAG / Agent 系统设计与落地经验
- [ ] 能主导技术选型与方案设计
- [ ] 对领域前沿有持续跟踪与判断力
- [ ] 能指导团队并推动技术影响力

---

## 社区与资源汇总

| 类别 | 资源 |
|------|------|
| 课程 | CS224N, CS231N, 李宏毅, fast.ai |
| 书籍 | d2l.ai, 西瓜书, Speech & Language Processing |
| 论文 | arXiv, Papers With Code, Semantic Scholar |
| 代码 | GitHub, HuggingFace Hub |
| 竞赛 | Kaggle, 天池, CLUE Benchmark |
| 社区 | Reddit r/MachineLearning, 知乎, PaperWeekly |
| 工具 | W&B, TensorBoard, Jupyter, VS Code |

---

> **最后一点建议**：NLP 领域发展极快，保持好奇心和持续学习的习惯比任何单一技术都重要。不要追求面面俱到，找到你的深度方向，成为这个方向的专家。
