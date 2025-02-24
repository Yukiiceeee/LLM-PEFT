import torch
import torch.nn as nn
import torch.nn.functional as F
class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        return self.fc(x)
    
class MoELayer(nn.Module):
    def _init_(self, num_experts, in_features, out_features):
        super(MoELayer, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([Linear(in_features, out_features) for _ in range(num_experts)])
        self.gate = nn.Linear(in_features, num_experts)

    def forward(self, x):
        # dim=-1表示在gate(x)输出结果的最后一个维度（对应专家数量）上进行softmax
        gate_scores = F.softmax(self.gate(x), dim=-1)
        # 把每个专家的输出堆到一个输出向量再加权
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)
        output = torch.bmm(gate_scores.unsqueeze(1), expert_outputs).squeeze(1)
        return output
    
input_size = 5
output_size = 3
num_experts = 4
batch_size = 10

model = MoELayer(num_experts, input_size, output_size)

demo = torch.randn(batch_size, input_size)

output = model(demo)
print(output.shape)
