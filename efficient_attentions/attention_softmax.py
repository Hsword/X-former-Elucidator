import torch
import torch.nn as nn
import math


class SoftmaxSelfAttention(nn.Module):
    def __init__(self, config, lyr=None):
        super().__init__()
        self.num_head = config["num_head"]

        self.drop_attn = nn.Dropout(p = config["dropout_prob"])
        self.head_dim = config["head_dim"]

    def forward(self, Q, K, V, mask):
        dot = torch.matmul(Q, torch.transpose(K, -2, -1))
        dot = dot / math.sqrt(self.head_dim)
        dot = dot - 1e6 * (1 - mask[:, None, None, :])

        attn = nn.functional.softmax(dot, dim = -1)
        attn = self.drop_attn(attn)

        X = torch.matmul(attn, V)
        
        return X