import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
import random
import math
from torch import Tensor
import os
import sys
sys.path.append('/d2/mxy/LLM-PEFT')
from Transformer.multi_attention import MultiHeadAttention as multi_attention
from Transformer.layernorm import LayerNorm as layernorm
from Transformer.embedding import TokenEmbedding as embedding

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, hidden_dim, num_heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attn = multi_attention.MultiHeadAttention(d_model, num_heads)
        self.norm1 = layernorm.LayerNorm(d_model)
        self,dropout1 = nn.Dropout(dropout)
        self.ffn = PositionwiseFeedForward(d_model, hidden_dim, dropout)
        self.norm2 = layernorm.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        _x = x
        x = self.attn(x, x, x, mask)
        x = self.dropout1(x)
        # 残差连接
        x = self.norm1(x + _x)
        _x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x
    
class Encoder(nn.Module):
    def __init__(self, voc_size, max_len, d_model, hidden_dim, num_heads, num_layers, dropout=0.1, device=None):
        super(Encoder, self).__init__()
        self.embed = embedding.Embedding(voc_size, d_model, max_len, dropout, device)
        self.layers = nn.ModuleList([EncoderLayer(d_model, hidden_dim, num_heads, dropout) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x