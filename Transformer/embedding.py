import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
import random
import math

random_torch = torch.randn(4, 4)
print(random_torch)

from torch import Tensor

# 将输入的词汇表索引转换为指定维度的Embedding
# 每个token的词索引升维到d_model维，padding_idx=1表示填充词的索引为1
# 继承nn.Embedding在训练中前向传播，反向传播，更新参数
class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)

# 通过位置编码计算输入每个词生成的正弦余弦位置编码
# 创建的是固定不变的位置编码，在训练中不更新，直接基于公式计算这个序列长度的位置编码矩阵
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super(PositionalEmbedding, self).__init__()
        # 初始化一个大小为(max_len, d_model)的零矩阵
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False
        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        _2i = torch.arange(0, d_model, step=2, device=device).float()
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        # x 形状: [batch_size, seq_len]，也就是词索引
        batch_size, seq_len = x.size()
        # 返回适合当前序列长度的位置编码
        return self.encoding[:seq_len, :]

# 嵌入层的输入是经过tokenizer处理后的词索引，输出是词的Embedding和位置编码
class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        super(TransformerEmbedding, self).__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_embedding = PositionalEmbedding(d_model, max_len, device)
        self.dropout = nn.Dropout(p=drop_prob)
    
    def forward(self, x):
        token_embedding = self.token_embedding(x)
        positional_embedding = self.positional_embedding(x)
        return self.dropout(token_embedding + positional_embedding)
        