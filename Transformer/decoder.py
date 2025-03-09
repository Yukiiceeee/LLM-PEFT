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
import multi_attention
import layernorm
import embedding

class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.attn1 = multi_attention.MultiHeadAttention(d_model, num_heads)
        self.norm1 = layernorm.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.cross_attn = multi_attention.MultiHeadAttention(d_model, num_heads)
        self.norm2 = layernorm.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.ffn = embedding.PositionwiseFeedForward(d_model, ffn_hidden, dropout)
        self.norm3 = layernorm.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    # t_mask是目标序列的掩码，s_mask是源序列的掩码
    # enc是编码器输出，dec是解码器输出
    def forward(self, dec, enc, t_mask, s_mask):
        _x = dec
        x = self.attn1(_x, _x, _x, t_mask)
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        _x = x
        x = self.cross_attn(x, enc, enc, s_mask)
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        x = self.ffn(x)
        x = self.dropout3(x)
        x = self.norm3(x + _x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, hidden_dim, num_heads, num_layers, dropout=0.1, device=None):
        super(Decoder, self).__init__()
        self.embed = embedding.TransformerEmbedding(dec_voc_size, d_model, max_len, dropout, device)
        self.layers = nn.ModuleList([DecoderLayer(d_model, hidden_dim, num_heads, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, dec_voc_size)
    
    def forward(self, dec, enc, t_mask, s_mask):
        dec = self.embed(dec)
        for layer in self.layers:
            dec = layer(dec, enc, t_mask, s_mask)
        return self.fc(dec)
        