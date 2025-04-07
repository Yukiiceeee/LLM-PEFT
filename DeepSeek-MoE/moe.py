import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicExpert(nn.Module):
    def __init__(self, in_features, out_features):
        super(BasicExpert, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.expert = nn.Linear(in_features, out_features)
        
    def forward(self, x):
        return self.expert(x)
    
class BasicRouter(nn.Module):
    def __init__(self, in_features, num_experts):
        super(BasicRouter, self).__init__()
        self.num_experts = num_experts
        self.in_features = in_features
        self.router = nn.Linear(in_features, num_experts)

    def forward(self, x):
        return self.router(x)
    
class BasicMoE(nn.Module):
    def __init__(self, in_features, out_features, num_experts):
        super(BasicMoE, self).__init__()
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        self.router = BasicRouter(in_features, num_experts)
        self.experts = nn.ModuleList(
            [
                BasicExpert(in_features, out_features, num_experts)
                for _ in range(num_experts)
            ]
        )
        
    def forward(self, x): # x: [batch_size, in_features]
        experts_weights = self.router(x)
        experts_weights = F.softmax(experts_weights, dim=1)
        # weighes: [batch_size, num_experts]
        # 因此，需要expert_output的shape为：[batch_size, num_experts, out_features]
        expert_output_list = [
            expert(x) for expert in self.experts
        ]
        expert_outputs = [
            expert_output.unsqueeze(1) for expert_output in expert_output_list
        ]
        expert_output = torch.concat(expert_outputs, dim=1)
        # expert_output: [batch_size, num_experts, out_features]
        experts_weights = experts_weights.unsqueeze(1)
        # print(experts_weights.shape)
        # print(expert_output.shape)

        output = experts_weights @ expert_output
        output = output.squeeze(1)
        return output
    
def test():
    x = torch.randn(4, 512)
    moe = BasicMoE(512, 128, 8)
    output = moe(x)
    print(output.shape)

if __name__ == "__main__":
    test()