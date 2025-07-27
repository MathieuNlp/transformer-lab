import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Embedding):
    pass

class CausalSelfAttention(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.d = d

        self.wq = nn.Linear(self.d, self.d, bias=False)
        self.wk = nn.Linear(self.d, self.d, bias=False)
        self.wv = nn.Linear(self.d, self.d, bias=False)

    def forward(self, x: torch.tensor) -> torch.tensor:
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        attn_weight = q @ k.transpose(-2, -1)
        scaled_attn_weight = attn_weight / np.sqrt(self.d)
        scaled_attn_weight = F.softmax(scaled_attn_weight, dim=-2)
        attn = torch.triu(scaled_attn_weight).transpose(-2, -1) @ v

        return attn



class Dropout(nn.Module):
    pass


class MutliHeadSelfAttention(nn.Module):
    pass

class LayerNorm(nn.Module):
    pass

class FeedForward(nn.Module):
    pass


class SubBlock(nn.Module):
    pass

class GPT2(nn.Module):
    pass


