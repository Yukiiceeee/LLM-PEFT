import torch
import torch.nn as nn
import torch.nn.functional as F
from moe import BasicExpert

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
        # 变形前的含义是：对于每一个 token，前 top_k 是哪些专家；
        # 变形后的含义是：对于每一个专家，把它作为 top_k 选择的 token 是哪些；
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
            [BasicExpert(self.hidden_dim, self.hidden_dim) for _ in range(self.num_experts)]
        )
        self.router = MoERouter(config)

    def forward(self, x):
        # x shape:[batch_size, seq_len, hidden_dim]6
        batch_size, seq_len, hidden_dim = x.size()
        # MoE的输入是Token维度的，因此需要将x变形为[batch_size * seq_len, hidden_dim]
        # view()函数会根据batch_size * seq_len和hidden_dim的大小重新排列x的形状
        x = x.view(batch_size * seq_len, hidden_dim)

        # 计算路由概率
        router_logits, router_weights, expert_indices, expert_mask = self.router(x)
        # expert_mask shape:[num_experts, top_k, batch_size * seq_len]
        # 这个mask表示对于每一个专家，把它作为top_k选择的token是哪些
        final_output = torch.zeros(
            (batch_size * seq_len, hidden_dim),
            dtype=x.dtype
        )

        for expert_idx in range(self.num_experts):
            expert = self.experts[expert_idx]
            # current_expert_mask shape:[top_k, batch_size * seq_len]
            # 这个mask表示对于当前专家，把它作为top_k选择的token是哪些
            current_expert_mask = expert_mask[expert_idx]
            
            # torch.where()函数会返回current_expert_mask中为True的元素的索引
            # 也就是返回，对于当前专家，是由这个专家计算的token的索引
            # top_idx 返回这个token选择当前专家是作为top_k的第几个
            # token_idx 返回这个token在batch_size * seq_len中的索引
            top_idx, token_idx = torch.where(current_expert_mask)
            # 假设 top_k 是 3，那么 top_idx 就是 0 or 1 or 2
            # token_idx 是 token 在 batch_size * seq_len 中的索引
            # top_idx 用来选 weight
            # token_idx 用来选 x
            
            current_x = x.unsqueeze(0)[:,token_idx,:].reshape(-1, hidden_dim)
            current_x = expert(current_x)
            # current_x shape:[selected_token_num, hidden_dim]
            current_weight = router_weights[token_idx, top_idx]
            # current_weight shape:[selected_token_num]
            current_weight = current_weight.unsqueeze(-1)
            # current_weight shape:[selected_token_num, 1]
            # 这里广播机制把所有weight都和对应的x相乘了
            current_hidden_state = current_weight * current_x
            # 这里计算出来的是对应这个专家参与计算的token的输出，因此要把这个输出加回原来的所有token输出里
            final_output.index_add_(
                0,
                token_idx,
                current_hidden_state.to(x.dtype)
            )
        # 计算完每个专家，把形状还原
        final_output = final_output.reshape(batch_size, seq_len, hidden_dim)

        # router_logits用来计算loss
        return final_output, router_logits

def main():
    config = MoEConfig(hidden_dim=256, num_experts=8, top_k=4, shared_experts=True, dropout=0.1, device="cuda")
    sparse_moe = SparseMoE(config)
    x = torch.randn(3, 6, 256)
    output = sparse_moe(x)
    print(output[0].shape)
    print(output[1].shape)

if __name__ == "__main__":
    main()