import torch
import torch.nn as nn
import torch.nn.functional as F
from moe import BasicMoE

class MoEConfig:
    def __init__(self, hidden_dim, num_experts, top_k, shared_experts, dropout, device):
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.shared_experts = shared_experts
        self.dropout = dropout
        self.device = device

class MoERouter(nn.Module):
    def __init__(self, config: MoEConfig):
        super(MoERouter, self).__init__()
        self.config = config
        self.top_k = config.top_k
        self.shared_experts = config.shared_experts
        self.num_experts = config.num_experts
        self.gate = nn.Linear(config.hidden_dim, config.num_experts)

    def forward(self, x):
        # x shape:[batch_size * seq_len, hidden_dim]
        # 后面只会选top_k个专家
        router_logits = self.gate(x)
        router_prob = F.softmax(router_logits, dim=1)
        
        # 计算top_k个专家的输出，top_k是可以反向传播优化的
        # topk()函数默认返回的是topk的值和对应的索引
        # router_weights: [batch_size * seq_len, top_k], top_k维度对应的是top_k个专家的权重
        # expert_indices: [batch_size * seq_len, top_k], top_k维度对应的是top_k个专家的索引
        # dim=-1表示在router_prob的最后一个维度上进行topk
        router_weights, expert_indices = torch.topk(router_prob, self.top_k, dim=-1)
        router_weights = router_weights / torch.sum(router_weights, dim=-1, keepdim=True)
        # to(x.dtype)表示将router_weights的类型转换为x的类型，比如x是float16，那么router_weights也是float16
        router_weights = router_weights.to(x.dtype)

        # F.one_hot()函数会将expert_indices转换为one-hot编码
        # 例如，如果expert_indices=[0, 1, 2], num_classes=4, 那么one_hot编码为[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
        expert_mask = F.one_hot(
            expert_indices,
            num_classes=self.num_experts
        ) # expert_mask shape:[batch_size * seq_len, top_k, num_experts]

        # permute()函数会根据2, 1, 0的顺序重新排列expert_mask的形状
        # 将其变形为[num_experts, top_k, batch_size * seq_len]
        expert_mask = expert_mask.permute(2, 1, 0)
        
        return router_logits, router_weights, expert_indices, expert_mask

class SparseMoE(nn.Module):
    def __init__(self, config: MoEConfig):
        super(SparseMoE, self).__init__()
        self.config = config
        self.top_k = config.top_k
        self.shared_experts = config.shared_experts
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_dim
        self.device = config.device
        self.dropout = config.dropout

        self.experts = nn.ModuleList(
            [BasicMoE(self.hidden_dim, self.hidden_dim) for _ in range(self.num_experts)]
        )
        self.router = MoERouter(config)

    def forward(self, x):
        # x shape:[batch_size, seq_len, hidden_dim]
        batch_size, seq_len, hidden_dim = x.size()
        # MoE的输入是Token维度的，因此需要将x变形为[batch_size * seq_len, hidden_dim]
        # view()函数会根据batch_size * seq_len和hidden_dim的大小重新排列x的形状
        x = x.view(batch_size * seq_len, hidden_dim)

        # 计算路由概率
        router_logits, router_weights, expert_indices, expert_mask = self.router(x)
        # expert_mask shape:[num_experts, top_k, batch_size * seq_len]
        