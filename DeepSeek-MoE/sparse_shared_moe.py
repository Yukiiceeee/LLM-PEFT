import torch
import torch.nn as nn
import torch.nn.functional as F
from moe import BasicExpert
from sparse_moe import MoEConfig, SparseMoE

class SharedMoE(nn.Module):
    def __init__(self, config: MoEConfig):
        super(SharedMoE, self).__init__()
        self.config = config
        self.top_k = config.top_k
        self.shared_experts = config.shared_experts
        self.num_experts = config.num_experts
        
        self.routed_moe = SparseMoE(config)
        self.shared_moe = nn.ModuleList(
            [BasicExpert(config.hidden_dim, config.hidden_dim) for _ in range(config.num_experts)]
        )
    
    # x shape:[batch_size, seq_len, hidden_dim]
    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.size()
        # 每个专家的输出shape:[batch_size, seq_len, hidden_dim]
        shared_experts_output_list = [
            expert(x) for expert in self.shared_moe
        ]
        # 要把所有专家的输出累加，先通过stack拓展expert_num维度
        # 然后再对expert_num维度求和
        # 这里不能直接用concat，因为concat只是在当前维度拼接，相当于只在batch维度把输出拼接起来
        # 而stack是创建一个新的维度，然后把所有输出堆叠起来
        shared_experts_output = torch.stack(shared_experts_output_list, dim=0)
        shared_experts_output = shared_experts_output.sum(dim=0)
        sparse_experts_output, sparse_logits = self.routed_moe(x)

        output = shared_experts_output + sparse_experts_output

        return output, sparse_logits
    
def main():
    config = MoEConfig(
        hidden_dim=256,
        num_experts=4,
        top_k=2,
        shared_experts=True,
        dropout=0.1,
        device="cpu"
    )
    model = SharedMoE(config)
    x = torch.randn(3, 6, 256)
    output, logits = model(x)
    print(output.shape)
    print(logits.shape)
    
if __name__ == "__main__":
    main()
        