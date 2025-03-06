## 1. Single-Head Attention（单头注意力）的实现过程

### 基本概念
单头注意力是注意力机制的基本形式，通过计算查询 Q、键 K 和值 V 之间的关系，动态为输入序列中的每个位置分配注意力权重。它是多头注意力的基础。

### 实现步骤
假设输入 Q、K、V 的维度分别为 $n × d_k$, $m × d_k$ 和 $m × d_v$：
1. **相似度计算**：Q 与 K 的转置相乘，得到 $n × m $的矩阵，表示查询和键之间的原始相似度。
2. **缩放**：将结果除以 d_k 的平方根，防止点积值过大。
3. **Softmax 归一化**：对缩放结果应用 Softmax，得到 $n × m$ 的注意力权重矩阵。
4. **加权求和**：注意力权重与 V 相乘，输出 Attention(Q, K, V)，维度为 $n × d_v$。

### 数学公式（纯文本描述）
- $Attention(Q, K, V) = Softmax((Q × K^T) / sqrt(d_k)) × V$。

### 代码实现（PyTorch）
```python
import torch
import torch.nn as nn
import math

class SingleHeadAttention(nn.Module):
    def __init__(self, d_model, d_k=None, d_v=None):
        super(SingleHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k if d_k is not None else d_model
        self.d_v = d_v if d_v is not None else d_model
        self.W_q = nn.Linear(d_model, self.d_k)
        self.W_k = nn.Linear(d_model, self.d_k)
        self.W_v = nn.Linear(d_model, self.d_v)
        
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        return output
```

### 特点
- 优势：简单高效，适合小型任务。
- 局限：单一视角，表达能力有限。