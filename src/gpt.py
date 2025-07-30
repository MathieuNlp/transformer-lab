import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleDict


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

    def forward(self, x: torch.tensor) -> torch.tensor:
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
        attn_proj = self.wo(concat_attn)

        return attn_proj

class FeedForward(nn.Module):
    def __init__(self, d_dim: int, d_ff: int, dropout_rate: float):
        super().__init__()
        self.d_dim = d_dim
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        ff = nn.Sequential(
            nn.Linear(self.d_dim, self.d_ff),
            nn.Dropout(self.dropout_rate),
            nn.ReLU(),
            nn.Linear(self.d_ff, self.d_dim),
            nn.Dropout(self.dropout_rate)
        )

        return ff(x)
        
class LayerNorm(nn.Module):
    def __init__(self, d_dim: int, eps: float):
        super().__init__()
        self.d_dim = d_dim
        self.eps = eps
        self.ln = nn.LayerNorm(normalized_shape=self.d_dim, eps=self.eps)
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.ln(x)

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, num_head: int, seq_len: int, d_dim: int, eps: float):
        super().__init__()
        self.num_head = num_head
        self.seq_len = seq_len
        self.d_dim = d_dim
        self.eps = eps

        self.mha = MutliHeadSelfAttention(self.num_head, self.seq_len, self.d_dim)
        self.layer_norm = LayerNorm(self.d_dim, self.eps)

        self.block = nn.Sequential(
            self.layer_norm,
            self.mha
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        return x + self.block(x)


class FeedForwardBlock(nn.Module):
    def __init__(self, d_dim: int, d_ff: int, dropout_rate: float, eps: float):
        super().__init__()
        self.d_dim = d_dim
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.eps = eps

        self.ff = FeedForward(self.d_dim, self.d_ff, self.dropout_rate)
        self.layer_norm = LayerNorm(self.d_dim, self.eps)

        self.block = nn.Sequential(
            self.layer_norm,
            self.ff
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        return x + self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, num_head: int, seq_len: int, d_dim: int, d_ff: int, dropout_rate: float, eps: float):
        super().__init__()
        self.mha_block = MultiHeadAttentionBlock(num_head, seq_len, d_dim, eps)
        self.ff_block = FeedForwardBlock(d_dim, d_ff, dropout_rate, eps)

        self.block = nn.Sequential(
            self.mha_block,
            self.ff_block
        )

    def forward(self, x):
        return self.block(x)


class GPT2(nn.Module):
    def __init__(self, num_block: int, num_head: int, seq_len: int, d_dim: int, d_ff: int, dropout_rate: float, eps: float):
        super().__init__()
        self.num_block = num_block
        self.num_head = num_head
        self.seq_len = seq_len
        self.d_dim = d_dim
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.eps = eps

        self.model = nn.ModuleList()

        for i in range(self.num_block):
            decoder_block = DecoderBlock(self.num_head, self.seq_len, self.d_dim, self.d_ff, self.dropout_rate, self.eps)
            self.model.add_module(f"block_{i}", decoder_block)

    def forward(self, x):
        pass

if __name__ == "__main__":
    model = GPT2(num_block=12,
                 num_head=8,
                 seq_len=100,
                 d_dim=512,
                 d_ff=2048,
                 dropout_rate=0.8,
                 eps=1e-05
                 )
    print(model)