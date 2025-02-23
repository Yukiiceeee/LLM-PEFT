import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoraLinear(nn.Module):
    def _init_(self, in_features, out_features, merge, rank=16, alpha=16, dropout=0.5):
        super(LoraLinear, self)._init_()
        self.in_features = in_features
        self.out_features = out_features
        self.merge = merge
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
    
        self.linear = nn.Linear(in_features, out_features)
        if rank > 0:
            # 构建权重参数
            self.lora_b = nn.Parameter(torch.zeros(out_features, rank))
            self.lora_a = nn.Parameter(torch.zeros(rank, in_features))
            self.scale = self.alpha / rank
            self.linear.weight.requires_grad = False

        if dropout > 0:
            self.dropout = nn.Dropout(self.dropout)
        else:
            self.dropout = nn.Identity()
        
        self.initial_weights()
    
    def initial_weights(self):
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x):
        if self.rank > 0 and self.merge:
            # 可以看到这边lora_b是out_features * rank的矩阵，lora_a是rank * in_features的矩阵，乘完后是out_features * in_features的矩阵
            # 那么weight就是out_features * in_features的矩阵，刚好和linear.weight的形状一样
            # 而wx+b，因此x就是in_features * batch_size的矩阵
            output = F.linear(x, weight=self.linear.weight + self.lora_b @ self.lora_a * self.scale, bias=self.linear.bias)
            output = self.dropout(output)
            return output
        else:
            return self.linear(x)

        
