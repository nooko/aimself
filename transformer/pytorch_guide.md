# PyTorch 使用指南

## PyTorch 是什么？

PyTorch 是 Meta（Facebook）开发的开源**深度学习框架**。核心用途：

- **训练神经网络** — 图像识别、NLP、语音、推荐系统等
- **研究原型** — 动态计算图，调试方便，学术界首选
- **生产部署** — 通过 TorchScript / ONNX 导出模型上线
- **科学计算** — GPU 加速的张量运算，替代 NumPy

---

## 安装

```bash
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

## 核心概念

| 概念 | 说明 |
|------|------|
| `Tensor` | 多维数组，类似 NumPy ndarray，支持 GPU 加速 |
| `autograd` | 自动微分引擎，追踪计算图 |
| `nn.Module` | 神经网络基类，所有模型继承它 |
| `DataLoader` | 批量加载数据，支持多进程 |
| `Optimizer` | 参数更新策略（SGD, Adam 等） |

---

## 基本训练流程

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 1. 定义模型
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)

# 2. 初始化
model = Model().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 3. 训练循环
for epoch in range(10):
    for inputs, labels in dataloader:
        inputs, labels = inputs.cuda(), labels.cuda()

        optimizer.zero_grad()              # 清梯度
        outputs = model(inputs)            # 前向
        loss = criterion(outputs, labels)  # 算损失
        loss.backward()                    # 反向传播
        optimizer.step()                   # 更新参数
```

---

## 常用操作

```python
# Tensor 创建
x = torch.randn(3, 4)             # 随机正态
x = torch.zeros(3, 4)             # 全零
x = torch.from_numpy(np_array)    # 从 NumPy

# GPU 操作
x = x.to('cuda')                  # 移到 GPU
x = x.cpu()                       # 回 CPU

# 保存/加载
torch.save(model.state_dict(), 'model.pth')
model.load_state_dict(torch.load('model.pth'))

# 推理模式（关闭梯度）
with torch.no_grad():
    pred = model(input)
```

### 关键提示

- `model.train()` / `model.eval()` — 切换训练/评估模式（影响 Dropout、BatchNorm）
- `loss.item()` — 取标量值，避免显存泄漏
- `torch.cuda.amp` — 混合精度训练，省显存加速

---

## 一、Transformer 实现

Transformer 是当前 NLP 和视觉领域主流架构（GPT、BERT、ViT 均基于它）。

### 核心组件

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        B, L, _ = q.shape
        # 投影 + 拆分多头
        q = self.W_q(q).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(k).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(v).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 缩放点积注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        # 拼接多头
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.W_o(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(self.norm1(x), self.norm1(x), self.norm1(x), mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=6, d_ff=2048, max_len=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        x = self.embed(x) + self.pos_embed(pos)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return self.head(x)
```

### 关键点

| 要素 | 说明 |
|------|------|
| 位置编码 | 可学习 Embedding 或正弦固定编码 |
| Causal Mask | 解码时用上三角矩阵遮挡未来 token |
| Pre-Norm vs Post-Norm | 上例用 Pre-Norm（训练更稳定） |
| `nn.MultiheadAttention` | PyTorch 内置版本，生产可直接用 |

```python
# 生成 causal mask
def causal_mask(size):
    return torch.tril(torch.ones(size, size)).unsqueeze(0).unsqueeze(0)
```

---

## 二、分布式训练

单卡放不下或训练太慢时，需多卡/多机并行。

### 方案对比

| 方案 | 适用场景 | 复杂度 |
|------|---------|--------|
| `DataParallel` (DP) | 单机多卡，快速原型 | 低 |
| `DistributedDataParallel` (DDP) | 单机/多机，生产首选 | 中 |
| `FSDP` | 超大模型，显存不够 | 高 |
| DeepSpeed / Megatron | 百亿参数级 | 高 |

### DDP 最小示例

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def train(rank, world_size):
    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    model = Model().cuda(rank)
    model = DDP(model, device_ids=[rank])

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        sampler.set_epoch(epoch)  # 确保每 epoch shuffle 不同
        for inputs, labels in dataloader:
            inputs, labels = inputs.cuda(rank), labels.cuda(rank)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()

    dist.destroy_process_group()
```

### 启动方式

```bash
# 单机 4 卡
torchrun --nproc_per_node=4 train.py

# 多机（2 节点各 4 卡）
torchrun --nnodes=2 --node_rank=0 --master_addr=10.0.0.1 --master_port=29500 --nproc_per_node=4 train.py
```

### 关键点

- **梯度同步**：DDP 自动 AllReduce 梯度，各卡参数保持一致
- **混合精度**：配合 `torch.cuda.amp.GradScaler` 省显存、加速
- **梯度累积**：小显存模拟大 batch → 每 N 步才 `optimizer.step()`
- **checkpoint**：只在 rank 0 保存，避免重复写入

```python
# 混合精度 + DDP
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    output = model(inputs)
    loss = criterion(output, labels)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## 三、自定义数据集

PyTorch 数据管道：`Dataset` → `DataLoader` → 训练循环。

### Map-style Dataset

```python
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        for label, cls_name in enumerate(sorted(os.listdir(root_dir))):
            cls_dir = os.path.join(root_dir, cls_name)
            for fname in os.listdir(cls_dir):
                self.samples.append((os.path.join(cls_dir, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
```

### NLP 文本数据集

```python
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.tokenizer.encode(self.texts[idx], max_length=self.max_len, truncation=True)
        # 输入 = tokens[:-1], 标签 = tokens[1:]（语言模型）
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y
```

### 自定义 collate_fn（处理变长序列）

```python
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    xs, ys = zip(*batch)
    xs_padded = pad_sequence(xs, batch_first=True, padding_value=0)
    ys_padded = pad_sequence(ys, batch_first=True, padding_value=-100)  # -100 被 CrossEntropyLoss 忽略
    return xs_padded, ys_padded

dataloader = DataLoader(dataset, batch_size=32, shuffle=True,
                        collate_fn=collate_fn, num_workers=4, pin_memory=True)
```

### Iterable-style Dataset（大文件流式读取）

```python
import json

class StreamDataset(torch.utils.data.IterableDataset):
    def __init__(self, file_path):
        self.file_path = file_path

    def __iter__(self):
        with open(self.file_path, 'r') as f:
            for line in f:
                yield self.process(line)

    def process(self, line):
        data = json.loads(line)
        return torch.tensor(data['input']), torch.tensor(data['label'])
```

### DataLoader 关键参数

| 参数 | 说明 |
|------|------|
| `num_workers` | 多进程加载，通常设 4-8 |
| `pin_memory=True` | 锁页内存，加速 CPU→GPU 传输 |
| `prefetch_factor` | 每 worker 预加载批次数 |
| `persistent_workers=True` | 避免每 epoch 重建 worker 进程 |
| `drop_last=True` | 丢弃最后不完整 batch（DDP 必须） |

---

## 总结

| 方向 | 核心要点 |
|------|---------|
| Transformer | Multi-Head Attention + FFN + 残差 + LayerNorm，堆叠 N 层 |
| 分布式训练 | DDP 为主，FSDP 应对超大模型，torchrun 启动 |
| 自定义数据集 | 继承 Dataset，实现 `__len__` + `__getitem__`，collate 处理变长 |
