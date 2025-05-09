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
from Transformer.encoder import Encoder as encoder
from Transformer.decoder import Decoder as decoder

class Transformer(nn.Module):
    # src_pad_idx: 源序列的填充索引
    # tgt_pad_idx: 目标序列的填充索引
    # 填充索引是单个数值，在词表中用来表示填充符号padding token的索引值，通常设为0或-1
    # src_voc_size: 源序列的词汇表大小
    # tgt_voc_size: 目标序列的词汇表大小
    # max_len: 序列的最大长度
    # d_model: 词嵌入的维度

    def __init__(self, src_pad_idx, tgt_pad_idx, src_voc_size, tgt_voc_size, max_len, d_model, hidden_dim, num_heads, num_layers, dropout=0.1, device=None):
        super(Transformer, self).__init__()
        self.encoder = encoder.Encoder(src_voc_size, max_len, d_model, hidden_dim, num_heads, num_layers, dropout, device)
        self.decoder = decoder.Decoder(tgt_voc_size, max_len, d_model, hidden_dim, num_heads, num_layers, dropout, device)
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.device = device

    def make_pad_mask(self, q, k, pad_idx_q, pad_idx_k):
        len_q, len_k = q.size(1), k.size(1)
        # 先用ne()操作转换为二值矩阵，然后通过unsqueeze()操作在q的维度上扩展维度
        # 最后通过repeat()操作将q的维度扩展为(batch_size, 1, len_q, len_k)
        # 这个操作是为了将mask矩阵的维度和score分数矩阵的维度对齐
        q = q.ne(pad_idx_q).unsqueeze(1).unsqueeze(3)
        q = q.repeat(1, 1, 1, len_k)
        k = k.ne(pad_idx_k).unsqueeze(1).unsqueeze(2)
        k = k.repeat(1, 1, len_q, 1)
        mask = q & k
        return mask
    
    def make_casual_mask(self, q, k):
        mask = torch.tril(torch.ones(q.size(1), k.size(1))).type(torch.BoolTensor).to(self.device)
        return mask
    
    def forward(self, src, tgt):
        src_mask = self.make_pad_mask(src, src, self.src_pad_idx, self.src_pad_idx)
        # 注意*是按元素乘，相当于把两种mask的作用效果叠加了
        tgt_mask = self.make_pad_mask(tgt, tgt, self.tgt_pad_idx, self.tgt_pad_idx)*self.make_casual_mask(tgt, tgt)
        enc = self.encoder(src, src_mask)
        dec = self.decoder(tgt, enc, tgt_mask, src_mask)
        return dec
