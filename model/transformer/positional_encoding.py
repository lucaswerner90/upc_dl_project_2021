import torch
from torch import tensor
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        """
        Args:
            embed_dim (int): Embedding dimensionality
            max_len (int): Maximum length of a sequence to expect
        """
        super(PositionalEncoding, self).__init__()
        # Create matrix of (T x embed_dim) representing the positional encoding
        # for max_len inputs
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe, persistent=False)
    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return x