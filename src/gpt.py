import torch
import torch.nn as nn


class PositionalEncoding(nn.Embedding):
    pass

class CausalSelfAttention(nn.Module):
    def __init__(self, seq_len: int, d: int):
        super().__init__()
        self.seq_len = seq_len
        self.d = d

        self.wq = nn.Linear(seq_len, d)
        self.wk = nn.Linear(seq_len, d)
        self.wv = nn.Linear(seq_len, d)

    def forward(self, x):
        q = self.wq(x)
        k = self.wk(x)
        a = 
        v = 


class Dropout(nn.Module):
    pass


class MutliHeadSelfAttention(nn.Module):
    pass

class LayerNorm(nn.module):
    pass

class FeedForward(nn.Module):
    pass


class SubBlock(nn.Module):
    pass

class GPT2(nn.Module):
    pass


