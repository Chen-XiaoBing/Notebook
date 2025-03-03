## 4. Grouped-Query Attention（GQA，分组查询注意力）的实现过程

### 基本概念
GQA 是 MQA 和 MHA 之间的折中方案，将 h 个头分成 g 组，每组共享一组 K 和 V，而 Q 仍独立计算。它在效率和表达能力之间取得平衡。

### 实现步骤
假设输入 X 的维度为 $n × d_{model}$，头数为 h，分组数为 g：
1. 生成 Q、K、V：
    - $Q_i = X × W_i^Q$（h 个独立的 Q）。
    - 将 h 个头分成 g 组，每组生成一组 $K_j = X × W_j^K$ 和 $V_j = X × W_j^V$（j = 1, ..., g）。
    - 每组内的头共享相同的 $K_j$ 和 $V_j$。
2. 计算注意力：每个头 i 使用所属组的 $K_j$ 和 $V_j$，计算 $head_i = Attention(Q_i, K_j, V_j)$。
3. 拼接和投影：$Output = Concat(head_1, ..., head_h) × W^O$，输出维度为 $n × d_{model}$。

### 数学公式（纯文本描述）
- $GroupedQuery(Q, K, V) = Concat(head_1, ..., head_h) × W^O$。
- 每个 $head_i = Attention(Q_i, K_j, V_j)$，其中 j 是头 i 所属的组。

### 代码实现（PyTorch）
以下是简化的 GQA 实现（假设 g = 2）：

```python
import torch
import torch.nn as nn
import math

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_groups):
        super(GroupedQueryAttention, self).__init__()
        assert d_model % num_heads == 0 and num_heads % num_groups == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        self.heads_per_group = num_heads // num_groups
        
        self.W_qs = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(num_heads)])
        self.W_ks = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(num_groups)])
        self.W_vs = nn.ModuleList([nn.Linear(d_model, self.d_v) for _ in range(num_groups)])
        self.W_o = nn.Linear(self.d_v * num_heads, d_model)
        
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        Qs = torch.stack([W_q(Q) for W_q in self.W_qs], dim=1)  # (batch, num_heads, seq, d_k)
        Ks = [W_k(K) for W_k in self.W_ks]  # 列表：num_groups 个 (batch, seq, d_k)
        Vs = [W_v(V) for W_v in self.W_vs]  # 列表：num_groups 个 (batch, seq, d_v)
        
        heads = []
        for i in range(self.num_heads):
            group_idx = i // self.heads_per_group
            scores = torch.matmul(Qs[:, i:i+1], Ks[group_idx].transpose(-2, -1)) / math.sqrt(self.d_k)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            attn = torch.softmax(scores, dim=-1)
            head = torch.matmul(attn, Vs[group_idx])
            heads.append(head)
        
        context = torch.cat(heads, dim=1).view(batch_size, -1, self.num_heads * self.d_v)
        output = self.W_o(context)
        return output
```

### 与 MQA 和 MHA 的对比
- MHA：g = h，每个头独立 KV。
- MQA：g = 1，所有头共享一组 KV。
- GQA：1 < g < h，折中方案。
- 优势：比 MQA 表达能力强，比 MHA 更高效。

### 总结
- 单头注意力：简单高效，单一视角。
- 多头注意力：通过多头和 $W^O$ 增强表达能力。
- MQA：共享 K 和 V，优化效率。
- GQA：分组共享 KV，平衡效率与性能。