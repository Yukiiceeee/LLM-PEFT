import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
import random
import math
from torch import Tensor

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        # 减去均值，除以标准差，加偏移，乘以缩放（可训练的缩放因子参数）
        # 输入x的形状为(batch_size, seq_len, d_model)
        mean = x.mean(-1, keepdim=True)
        # 无偏估计，并且保持维度不变
        var = x.var(-1, keepdim=True, unbiased=False)
        # 归一化
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        # 缩放和平移
        return self.gamma * x_norm + self.beta
    
    