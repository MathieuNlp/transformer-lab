from gpt import CausalSelfAttention
import torch
import torch.nn as nn
from torch.nn.functional import softmax


x = torch.rand(3, 100, 521)
attn = torch.rand(3, 100, 100)

csa = CausalSelfAttention(521)

res = csa(x)

print(res.shape)