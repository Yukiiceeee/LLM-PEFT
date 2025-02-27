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
class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)
    



