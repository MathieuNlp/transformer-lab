from IPython import embed
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.functional import softmax
import numpy as np

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, q_dim=None, k_dim=None, v_dim=None):
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

if __name__ == "__main__":
    x = torch.rand((5, 10, 512))
    sa = SelfAttention(embed_dim=512)
    res = sa(x)
    print(res.shape)

