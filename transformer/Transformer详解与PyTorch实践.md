# Transformer 详解与 PyTorch 实践

## 目录

- [1. Transformer 详解](#1-transformer-详解)
  - [1.1 背景与动机](#11-背景与动机)
  - [1.2 整体架构](#12-整体架构)
  - [1.3 输入表示](#13-输入表示)
  - [1.4 自注意力机制](#14-自注意力机制)
  - [1.5 多头注意力](#15-多头注意力)
  - [1.6 前馈网络](#16-前馈网络)
  - [1.7 残差连接与层归一化](#17-残差连接与层归一化)
  - [1.8 编码器与解码器](#18-编码器与解码器)
  - [1.9 掩码机制](#19-掩码机制)
  - [1.10 训练技巧](#110-训练技巧)
- [2. PyTorch 实践](#2-pytorch-实践)
  - [2.1 完整实现代码](#21-完整实现代码)
  - [2.2 训练示例](#22-训练示例)
  - [2.3 推理示例](#23-推理示例)

---

## 1. Transformer 详解

### 1.1 背景与动机

Transformer 由 Vaswani 等人在 2017 年论文 **"Attention Is All You Need"** 中提出。在此之前，序列建模主要依赖 RNN（LSTM / GRU）和 CNN，存在以下问题：

| 问题 | 说明 |
|------|------|
| **长距离依赖** | RNN 通过逐步传递隐状态，信息随距离衰减 |
| **无法并行** | RNN 按时间步顺序计算，训练慢 |
| **CNN 感受野有限** | CNN 需堆叠多层才能覆盖长距离 |

Transformer **完全基于注意力机制**，抛弃循环和卷积结构，实现了：
- 任意位置对之间的直接依赖建模（O(1) 路径长度）
- 全并行计算，训练效率大幅提升
- 成为后续 BERT、GPT、ViT 等模型的基石

### 1.2 整体架构

```
输入序列 → [Embedding + Positional Encoding]
                    ↓
            ┌───────────────┐
            │   Encoder × N │  （N 层堆叠，论文中 N=6）
            │  ┌───────────┐│
            │  │Multi-Head  ││
            │  │Attention   ││
            │  └─────┬─────┘│
            │  Add & Norm   │
            │  ┌───────────┐│
            │  │Feed Forward││
            │  └─────┬─────┘│
            │  Add & Norm   │
            └───────┬───────┘
                    ↓
            编码器输出（Memory）
                    ↓
            ┌───────────────┐
            │   Decoder × N │
            │  ┌───────────┐│
            │  │Masked      ││
            │  │Multi-Head  ││
            │  │Attention   ││
            │  └─────┬─────┘│
            │  Add & Norm   │
            │  ┌───────────┐│
            │  │Cross       ││
            │  │Attention   ││
            │  │(Q=Dec,     ││
            │  │ K,V=Enc)   ││
            │  └─────┬─────┘│
            │  Add & Norm   │
            │  ┌───────────┐│
            │  │Feed Forward││
            │  └─────┬─────┘│
            │  Add & Norm   │
            └───────┬───────┘
                    ↓
            Linear + Softmax → 输出概率
```

### 1.3 输入表示

输入由两部分相加组成：

#### Token Embedding

将离散 token 映射为 \(d_{model}\) 维连续向量。论文中 \(d_{model} = 512\)。

#### Positional Encoding（位置编码）

Transformer 没有循环结构，需显式注入位置信息。论文采用正弦/余弦函数：

\[
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
\]

\[
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
\]

**设计优势：**
- 不同维度使用不同频率的正弦波，编码丰富的位置信息
- 对于任意固定偏移 \(k\)，\(PE_{pos+k}\) 可由 \(PE_{pos}\) 线性变换得到，天然表达相对位置
- 可外推到训练时未见过的序列长度

### 1.4 自注意力机制（Scaled Dot-Product Attention）

核心公式：

\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

**步骤分解：**

1. **线性投影**：输入 \(X\) 分别乘以权重矩阵 \(W^Q, W^K, W^V\) 得到 Q、K、V
2. **计算相似度**：\(QK^T\) 得到 \(n \times n\) 注意力分数矩阵
3. **缩放**：除以 \(\sqrt{d_k}\) 防止点积值过大导致 softmax 梯度消失
4. **Mask（可选）**：将不允许关注的位置设为 \(-\infty\)
5. **Softmax**：按行归一化得到注意力权重
6. **加权求和**：权重乘以 V，得到输出

**缩放因子 \(\sqrt{d_k}\) 的必要性：**

假设 Q 和 K 的元素均为均值 0、方差 1 的独立随机变量，则 \(q \cdot k = \sum_{i=1}^{d_k} q_i k_i\) 的方差为 \(d_k\)。当 \(d_k\) 较大时，点积值的绝对值增大，softmax 输出趋近 one-hot，梯度极小。除以 \(\sqrt{d_k}\) 使方差回归到 1。

### 1.5 多头注意力（Multi-Head Attention）

单个注意力头只能捕捉一种注意力模式。多头注意力并行运行 \(h\) 个注意力头，每个头独立学习不同的注意力模式：

\[
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O
\]

\[
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
\]

**参数维度（论文设定）：**

| 参数 | 值 |
|------|------|
| \(d_{model}\) | 512 |
| \(h\)（头数） | 8 |
| \(d_k = d_v = d_{model}/h\) | 64 |

多头并行计算后 concat 再投影，总计算量与单头全维度注意力相当，但表达能力更强。

**不同头学到的模式举例：**
- 某个头关注局部（相邻词）
- 某个头关注句法依赖（主谓宾）
- 某个头关注共指消解（代词→先行词）

### 1.6 前馈网络（Position-wise Feed-Forward Network）

每个位置独立地经过相同的两层全连接网络：

\[
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
\]

- 内层维度 \(d_{ff} = 2048\)（4 倍于 \(d_{model}\)）
- 激活函数为 ReLU（后续变体常用 GELU / SwiGLU）
- 相当于 1×1 卷积，不同位置**参数共享**，不同层参数独立

FFN 的作用是为模型引入非线性变换能力，注意力层本身是线性加权求和。

### 1.7 残差连接与层归一化

每个子层（注意力 / FFN）外围包裹：

\[
\text{output} = \text{LayerNorm}(x + \text{SubLayer}(x))
\]

- **残差连接**：缓解深层网络梯度消失，使信息直通
- **Layer Normalization**：对每个样本的特征维度归一化，稳定训练

> **注**：原始论文用 Post-LN（先残差再归一化），后续研究发现 Pre-LN（先归一化再子层）训练更稳定，现已成为主流做法。

### 1.8 编码器与解码器

#### 编码器（Encoder）

每层包含两个子层：
1. **多头自注意力**（Self-Attention）：每个位置可关注输入序列所有位置
2. **前馈网络**（FFN）

共 N 层堆叠。最终输出作为解码器的 Memory。

#### 解码器（Decoder）

每层包含三个子层：
1. **掩码多头自注意力**（Masked Self-Attention）：防止关注未来位置
2. **交叉注意力**（Cross-Attention）：Q 来自解码器，K/V 来自编码器输出
3. **前馈网络**（FFN）

交叉注意力让解码器的每个位置能查询编码器的完整输出，实现源序列到目标序列的信息传递。

### 1.9 掩码机制

Transformer 使用两种掩码：

#### Padding Mask
批处理时序列长度不一，短序列需 padding。Padding 位置的注意力分数设为 \(-\infty\)，softmax 后趋近 0，不参与计算。

#### Causal Mask（因果掩码 / Look-Ahead Mask）
解码器自注意力中，位置 \(i\) 只能关注 \(\leq i\) 的位置。使用上三角矩阵实现：

```
[0, -inf, -inf, -inf]
[0,    0, -inf, -inf]
[0,    0,    0, -inf]
[0,    0,    0,    0]
```

### 1.10 训练技巧

#### 学习率预热（Warmup + Decay）

论文采用特殊的学习率调度：

\[
lr = d_{model}^{-0.5} \cdot \min(step^{-0.5}, \; step \cdot warmup\_steps^{-1.5})
\]

- 前 warmup_steps 步线性增长
- 之后按步数平方根倒数衰减
- warmup_steps = 4000

#### Label Smoothing

使用 \(\epsilon_{ls} = 0.1\) 的标签平滑，将真实标签从 one-hot 分布变为：
- 正确类别概率：\(1 - \epsilon_{ls}\)
- 其余类别均分剩余概率

提升泛化能力，略降 perplexity 但提升 BLEU。

#### Dropout

- 注意力权重 dropout：\(p = 0.1\)
- 残差 dropout：\(p = 0.1\)
- Embedding + Positional Encoding 求和后 dropout

---

## 2. PyTorch 实践

### 2.1 完整实现代码

以下从零实现完整的 Transformer 模型，不依赖 `nn.Transformer`。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


class PositionalEncoding(nn.Module):
    """正弦余弦位置编码"""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

    def scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            Q: (batch, n_heads, seq_len, d_k)
            K: (batch, n_heads, seq_len, d_k)
            V: (batch, n_heads, seq_len, d_k)
            mask: (batch, 1, 1, seq_len) 或 (batch, 1, seq_len, seq_len)
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)
        return output

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        batch_size = query.size(0)

        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # concat heads: (batch, seq_len, d_model)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )

        return self.W_o(attn_output)


class PositionwiseFeedForward(nn.Module):
    """位置前馈网络"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class EncoderLayer(nn.Module):
    """编码器单层"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        # Self-Attention + Add & Norm
        attn_output = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # FFN + Add & Norm
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))

        return x


class DecoderLayer(nn.Module):
    """解码器单层"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        memory_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # Masked Self-Attention + Add & Norm
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # Cross-Attention + Add & Norm
        cross_output = self.cross_attn(x, memory, memory, memory_mask)
        x = self.norm2(x + self.dropout2(cross_output))

        # FFN + Add & Norm
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_output))

        return x


class Encoder(nn.Module):
    """编码器（N 层堆叠）"""

    def __init__(
        self, n_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


class Decoder(nn.Module):
    """解码器（N 层堆叠）"""

    def __init__(
        self, n_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        memory_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        return self.norm(x)


class Transformer(nn.Module):
    """完整 Transformer 模型"""

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()

        self.pad_idx = pad_idx
        self.d_model = d_model

        # Embedding layers
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)

        # Encoder & Decoder
        self.encoder = Encoder(n_layers, d_model, n_heads, d_ff, dropout)
        self.decoder = Decoder(n_layers, d_model, n_heads, d_ff, dropout)

        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """Padding mask for source: (batch, 1, 1, src_len)"""
        return (src != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """Causal mask + padding mask for target: (batch, 1, tgt_len, tgt_len)"""
        batch_size, tgt_len = tgt.size()

        # Padding mask: (batch, 1, 1, tgt_len)
        pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)

        # Causal mask: (1, 1, tgt_len, tgt_len)
        causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=tgt.device)).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)

        return pad_mask & causal_mask

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        src_emb = self.positional_encoding(
            self.src_embedding(src) * math.sqrt(self.d_model)
        )
        return self.encoder(src_emb, src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> torch.Tensor:
        tgt_emb = self.positional_encoding(
            self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        )
        return self.decoder(tgt_emb, memory, tgt_mask, memory_mask)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: (batch_size, src_len) - 源序列 token ids
            tgt: (batch_size, tgt_len) - 目标序列 token ids
        Returns:
            (batch_size, tgt_len, tgt_vocab_size) - 输出 logits
        """
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        memory = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, memory, tgt_mask, src_mask)

        return self.output_projection(decoder_output)
```

### 2.2 训练示例

以下演示一个简单的序列复制任务（输入序列 → 输出相同序列），用来验证模型实现正确性。

```python
import torch
import torch.nn as nn
import torch.optim as optim


def generate_copy_data(batch_size: int, seq_len: int, vocab_size: int, pad_idx: int = 0):
    """生成序列复制任务数据"""
    # 随机生成 token ids（避免 pad_idx=0 和 特殊 token）
    data = torch.randint(2, vocab_size, (batch_size, seq_len))

    # src 和 tgt 相同（复制任务）
    src = data.clone()

    # tgt_input: 前面加 BOS(=1)，去掉最后一个
    bos = torch.ones(batch_size, 1, dtype=torch.long)
    tgt_input = torch.cat([bos, data[:, :-1]], dim=1)

    # tgt_output: 原始数据（用于计算 loss）
    tgt_output = data.clone()

    return src, tgt_input, tgt_output


def train_copy_task():
    # 超参数
    VOCAB_SIZE = 50
    D_MODEL = 128
    N_HEADS = 4
    N_LAYERS = 2
    D_FF = 256
    SEQ_LEN = 20
    BATCH_SIZE = 64
    EPOCHS = 50
    LR = 1e-3
    PAD_IDX = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Transformer(
        src_vocab_size=VOCAB_SIZE,
        tgt_vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        dropout=0.1,
        pad_idx=PAD_IDX,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)

    model.train()

    for epoch in range(1, EPOCHS + 1):
        src, tgt_input, tgt_output = generate_copy_data(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE, PAD_IDX)
        src, tgt_input, tgt_output = src.to(device), tgt_input.to(device), tgt_output.to(device)

        logits = model(src, tgt_input)  # (batch, seq_len, vocab_size)

        # 展平计算 loss
        loss = criterion(logits.reshape(-1, VOCAB_SIZE), tgt_output.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if epoch % 5 == 0:
            # 计算准确率
            preds = logits.argmax(dim=-1)
            accuracy = (preds == tgt_output).float().mean().item()
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | Accuracy: {accuracy:.4f}")

    return model


if __name__ == "__main__":
    model = train_copy_task()
```

**预期输出（参考）：**
```
Epoch   5 | Loss: 2.8134 | Accuracy: 0.2450
Epoch  10 | Loss: 1.5672 | Accuracy: 0.5380
Epoch  15 | Loss: 0.6243 | Accuracy: 0.8120
Epoch  20 | Loss: 0.2105 | Accuracy: 0.9350
Epoch  25 | Loss: 0.0683 | Accuracy: 0.9780
Epoch  30 | Loss: 0.0241 | Accuracy: 0.9950
Epoch  35 | Loss: 0.0098 | Accuracy: 0.9990
Epoch  40 | Loss: 0.0045 | Accuracy: 1.0000
Epoch  45 | Loss: 0.0023 | Accuracy: 1.0000
Epoch  50 | Loss: 0.0012 | Accuracy: 1.0000
```

### 2.3 推理示例

训练完成后，使用贪心解码进行推理：

```python
@torch.no_grad()
def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    max_len: int,
    bos_idx: int = 1,
    eos_idx: int = 2,
) -> torch.Tensor:
    """
    贪心解码（Greedy Decoding）

    Args:
        model: 训练好的 Transformer 模型
        src: (1, src_len) 源序列
        max_len: 最大生成长度
        bos_idx: 起始 token id
        eos_idx: 结束 token id
    Returns:
        (1, generated_len) 生成的 token 序列
    """
    device = src.device
    model.eval()

    src_mask = model.make_src_mask(src)
    memory = model.encode(src, src_mask)

    # 初始输入为 BOS
    tgt = torch.tensor([[bos_idx]], dtype=torch.long, device=device)

    for _ in range(max_len):
        tgt_mask = model.make_tgt_mask(tgt)
        decoder_output = model.decode(tgt, memory, tgt_mask, src_mask)

        # 取最后一个时间步的 logits
        next_token_logits = model.output_projection(decoder_output[:, -1, :])
        next_token = next_token_logits.argmax(dim=-1, keepdim=True)

        tgt = torch.cat([tgt, next_token], dim=1)

        if next_token.item() == eos_idx:
            break

    return tgt


# 使用示例
def inference_demo(model):
    device = next(model.parameters()).device
    model.eval()

    # 构造测试输入
    test_src = torch.tensor([[3, 7, 15, 22, 8, 41, 5]], dtype=torch.long, device=device)
    print(f"源序列:   {test_src[0].tolist()}")

    # 贪心解码
    output = greedy_decode(model, test_src, max_len=20, bos_idx=1, eos_idx=2)
    print(f"生成序列: {output[0].tolist()}")

    # 对于复制任务，输出应接近 [1, 3, 7, 15, 22, 8, 41, 5]
    # 其中 1 为 BOS token


if __name__ == "__main__":
    model = train_copy_task()
    inference_demo(model)
```

---

## 附录：关键超参数一览

| 超参数 | 论文原值 | 说明 |
|--------|----------|------|
| \(d_{model}\) | 512 | 模型维度 |
| \(d_{ff}\) | 2048 | FFN 内层维度 |
| \(h\) | 8 | 注意力头数 |
| \(d_k = d_v\) | 64 | 每头维度 |
| \(N\) | 6 | 编码器/解码器层数 |
| dropout | 0.1 | 全局 dropout 率 |
| warmup_steps | 4000 | 学习率预热步数 |
| \(\epsilon_{ls}\) | 0.1 | 标签平滑系数 |
| batch size | ~25000 tokens | 按 token 数计的批大小 |
| optimizer | Adam | \(\beta_1=0.9, \beta_2=0.98, \epsilon=10^{-9}\) |

---

## 参考资料

1. Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS.
2. Annotated Transformer: https://nlp.seas.harvard.edu/annotated-transformer/
3. PyTorch 官方文档: https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
