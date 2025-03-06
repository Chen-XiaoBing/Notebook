## 3. Multi-Query Attention（MQA，多查询注意力）的实现过程

### 基本概念

MQA 是 Multi-Head Attention 的优化变体，所有头共享 K 和 V，但 Q 仍独立计算，旨在降低内存和计算开销，尤其
适合自回归解码。

### 实现步骤
假设输入 X 的维度为 $n × d_{model}$：
1. 生成 Q、K、V：
    - $Q_i = X × W_i^Q$（每个头独立生成 Q）。
    - $K = X × W^K，V = X × W^V$（所有头共享）。
    - $d_k = d_v = d_{model} / h$。
2. 计算注意力：每个头 i 计算 $head_i = Attention(Q_i, K, V)$。
3. 拼接和投影: $Output = Concat(head_1, ..., head_h) × W^O$, 输出维度为 $n × d_{model}$。

### 数学公式（纯文本描述）
- $MultiQuery(Q, K, V) = Concat(head_1, ..., head_h) × W^O$。
- $每个 head_i = Attention(Q_i, K, V)$。

### 代码实现（PyTorch）
```python
import torch
import torch.nn as nn
import math

class MultiQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiQueryAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        self.W_qs = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(num_heads)])
        self.W_k = nn.Linear(d_model, self.d_k)
        self.W_v = nn.Linear(d_model, self.d_v)
        self.W_o = nn.Linear(self.d_v * num_heads, d_model)
        
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        Qs = torch.stack([W_q(Q) for W_q in self.W_qs], dim=1)
        K = self.W_k(K)
        V = self.W_v(V)
        
        scores = torch.matmul(Qs, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_v)
        output = self.W_o(context)
        return output
```

### 与 MHA 的对比
- MHA：每个头有独立的 $Q_i$, $K_i$, $V_i$。
-  MQA：仅 $Q_i$ 独立，K 和 V 共享。
- 优势：内存占用降低 h 倍，适合推理阶段。
- 局限：表达能力不如 MHA。