import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Embedding):
    pass

class CausalSelfAttention(nn.Module):
    def __init__(self, d_dim: int):
        super().__init__()
        self.d_dim = d_dim

        self.wq = nn.Linear(self.d_dim, self.d_dim, bias=True)
        self.wk = nn.Linear(self.d_dim, self.d_dim, bias=True)
        self.wv = nn.Linear(self.d_dim, self.d_dim, bias=True)

    def forward(self, x: torch.tensor) -> torch.tensor:
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        attn_weight = q @ k.transpose(-2, -1)
        scaled_attn_weight = attn_weight / np.sqrt(self.d_dim)
        causal_scaled_attn_weight = torch.triu(scaled_attn_weight).transpose(-2, -1)
        scaled_attn_weight = F.softmax(causal_scaled_attn_weight, dim=-2)
        attn = torch.triu(scaled_attn_weight).transpose(-2, -1) @ v

        return attn

class MutliHeadSelfAttention(nn.Module):
    def __init__(self, num_heads: int, seq_len:int, d_dim: int):
        super().__init__()
        self.d_dim = d_dim
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.head_dim = self.d_dim // self.num_heads

        self.wq = nn.Linear(self.d_dim, self.d_dim, bias=True)
        self.wk = nn.Linear(self.d_dim, self.d_dim, bias=True)
        self.wv = nn.Linear(self.d_dim, self.d_dim, bias=True)
        self.wo = nn.Linear(self.d_dim, self.d_dim, bias=True)

    def forward(self, x):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        batch_size = q.shape[0]


        q = q.view(batch_size, self.seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, self.seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, self.seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)


        attn_weight = q @ k.transpose(-2, -1)
        scaled_attn_weight = attn_weight / np.sqrt(self.d_dim)
        causal_scaled_attn_weight = torch.triu(scaled_attn_weight).transpose(-2, -1)
        scaled_attn_weight = F.softmax(causal_scaled_attn_weight, dim=-2)
        attn = torch.triu(scaled_attn_weight).transpose(-2, -1) @ v

        attn = attn.permute(0, 2, 1, 3)
        concat_attn = attn.reshape(batch_size, self.seq_len, self.num_heads*self.head_dim)
        print(concat_attn.shape)
        attn_proj = self.wo(concat_attn)

        return attn_proj

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


