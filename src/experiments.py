from gpt import MutliHeadSelfAttention
from gpt import FeedForward
import torch
import torch.nn as nn
from torch.nn.functional import softmax


x = torch.rand(3, 100, 521)
attn = torch.rand(3, 100, 100)

csa = MutliHeadSelfAttention(4, 100, 512)
ff = FeedForward(521, 2048, 0.8)

res = csa(x)
print(res.shape)