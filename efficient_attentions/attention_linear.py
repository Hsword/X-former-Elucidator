import torch
import torch.nn as nn

def kernelized_attention(Q, K, V, eps = 1e-6):
        '''
        Q -> (batch, head, maxlen, qkdim)
        K -> (batch, head, maxlen, qkdim)
        V -> (batch, head, maxlen, vdim)
        '''

        K_sum = K.sum(dim = -2)
        denom = 1. / torch.clamp_min(torch.einsum('...nd,...d->...n', Q, K_sum.type_as(Q)), eps)
        KV = torch.einsum('...nd,...ne->...de', K, V)
        out = torch.einsum('...de,...nd,...n->...ne', KV, Q, denom)
        return out


class LinearSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, Q, K, V, mask):
        Q = nn.functional.elu(Q) + 1
        K = (nn.functional.elu(K) + 1)

        return kernelized_attention(Q, K * mask[:, None, :, None], V * mask[:, None, :, None])
