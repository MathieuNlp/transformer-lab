import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

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
        v = self.wv(x)

        attn_weight = q @ k.transpose(-2, -1)
        scaled_attn_weight = attn_weight / np.sqrt(self.d)
        scaled_attn_weight = F.softmax(scaled_attn_weight)
        attn = scaled_attn_weight @ v.transpose(-2, -1)

        return attn



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


