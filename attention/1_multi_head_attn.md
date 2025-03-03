## 2. Multi-Head Attention（多头注意力）的实现过程

### 基本概念
Multi-Head Attention 是 Transformer 模型的核心，通过并行计算多个注意力头，增强模型捕捉输入序列中不同位置关系的能力。它在单头注意力基础上扩展而来。

### 实现步骤
假设输入 X 的维度为 $n × d_{model}$：

1. **线性变换生成 Q、K、V**：
    - $Q = X × W^Q$，$K = X × W^K$，$V = X × W^V$，其中 $W^Q$、$W^K$、$W^V$ 是可学习权重矩阵。
    - 通常 $d_k = d_v = d_{model} / h$（h 为头数）。
2. **分割成多头**：将 Q、K、V 分成 h 个子矩阵，每个头的维度为 $n × (d_k/h)$。
3. **计算注意力**：对每个头 i，计算 $head_i = Attention(Q_i, K_i, V_i)$。
4. **拼接**：将 h 个头的输出拼接为 $MultiHead(Q, K, V) = Concat(head_1, ..., head_h)$。
5. 线性变换输出：将拼接结果与 $W^O$ 相乘，输出维度为 $n × d_{model}$。

### 数学公式（纯文本描述）

- $MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W^O$。
- 每个 $head_i = Attention(Q × W_i^Q, K × W_i^K, V × W_i^V)$。

### 代码实现（PyTorch）
```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)
        return output
```

### 单头与多头的区别
- 单头：输出由 $W^V$ 直接生成，无需 $W^O$，因为只有一个视角。
- 多头：需要 $W^O$ 融合 h 个头的子空间表示。