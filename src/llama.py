from IPython import embed
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.functional import softmax
import numpy as np

class SelfAttention(nn.Module):
    def __init__(self, embed_dim: int, q_dim=None, k_dim=None, v_dim=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_dim = q_dim if q_dim else embed_dim
        self.k_dim = k_dim if q_dim else embed_dim
        self.v_dim = v_dim if q_dim else embed_dim

        self.wq = nn.Linear(self.q_dim, self.q_dim, bias=False)
        self.wk = nn.Linear(self.k_dim, self.q_dim, bias=False)
        self.wv = nn.Linear(self.v_dim, self.q_dim, bias=False)

    def forward(self, x):
        q = self.wq(x)
        k = self.wv(x)
        v = self.wv(x)

        attn_weights = torch.matmul(q, k.view(x.shape[0], self.k_dim, -1))
        scaled_attn_weights = attn_weights / np.sqrt(self.k_dim)
        sm_attn = torch.tril(softmax(scaled_attn_weights, dim=-1))
        attn = torch.matmul(sm_attn.contiguous(), v)

        return attn

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_head: int, q_dim=None, k_dim=None, v_dim=None, o_dim=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_head = num_head
        self.head_dim = self.embed_dim // self.num_head
        self.q_dim = q_dim if q_dim else self.embed_dim
        self.k_dim = k_dim if q_dim else self.embed_dim
        self.v_dim = v_dim if q_dim else self.embed_dim
        self.o_dim = o_dim if o_dim else self.embed_dim

        self.wq = nn.Linear(self.q_dim, self.q_dim, bias=False)
        self.wk = nn.Linear(self.k_dim, self.q_dim, bias=False)
        self.wv = nn.Linear(self.v_dim, self.q_dim, bias=False)
        self.wo = nn.Linear(self.o_dim, self.o_dim, bias=False)


    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        q = self.wq(x).view(batch_size, self.num_head, seq_len, self.head_dim)
        k = self.wk(x).view(batch_size, self.num_head, seq_len, self.head_dim)
        v = self.wv(x).view(batch_size, self.num_head, seq_len, self.head_dim)

        attn_weights = torch.matmul(q, k.view(batch_size, self.num_head, self.head_dim, seq_len))
        normalized_attn_weights = attn_weights / np.sqrt(self.q_dim)
        sftm_attn_weights = softmax(normalized_attn_weights, dim=-1)
        attn = torch.matmul(sftm_attn_weights, v)
        attn_concat = attn.view(batch_size, seq_len,self.embed_dim).contiguous()
        final_attn = self.wo(attn_concat)

        return final_attn

class SwiGLU(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, dropout_rate: float):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        self.f = nn.Sequential(
            nn.Linear(self.embed_dim, self.hidden_dim),
            SwiGLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            SwiGLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.embed_dim),
            SwiGLU()
            )

    def forward(self, x):
        return self.f(x)




if __name__ == "__main__":
    x = torch.rand((5, 10, 512))
    mha = MultiheadAttention(embed_dim=512, num_head=4)
    res = mha(x)
    print(res.shape)

