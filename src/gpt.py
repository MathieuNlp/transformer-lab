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
        causal_scaled_attn_weight = torch.triu(scaled_attn_weight).transpose(-2, -1)
        scaled_attn_weight = F.softmax(causal_scaled_attn_weight, dim=-2)
        attn = torch.triu(scaled_attn_weight).transpose(-2, -1) @ v

        return attn

class MutliHeadSelfAttention(nn.Module):
    def __init__(self, n_head: int, d: int):
        super().__init__()
        self.n_head = n_head
        self.d_head = d // n_head

    def forward(self, x):
        pass


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout_rate: float):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        ff = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff),
            nn.Dropout(self.dropout_rate),
            nn.ReLU(),
            nn.Linear(self.d_ff, self.d_model),
            nn.Dropout(self.dropout_rate)
        )

        return ff(x)
        

class SubBlock(nn.Module):
    pass

class GPT2(nn.Module):
    pass


