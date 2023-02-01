import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch import Tensor
from typing import Optional

class CosformerSelfAttention(nn.Module):
    def __init__(
        self,
        config
    ):
        super().__init__()
        
        self.head_dim = config["head_dim"]
        self.embed_dim = config["embedding_dim"]
        self.kdim = config["embedding_dim"]
        self.vdim = config["embedding_dim"]
        self.num_heads = config["num_head"]
        self.act_fun = config["act_fun"]
        if "reweighting" in config and not config["reweighting"]:
            self.reweighting = False
        else:
            self.reweighting = True

        assert (self.embed_dim % self.num_heads == 0), "embed_dim must be divisible by num_heads"

    def get_index(self, seq_len):
        index = np.pi / 2 * torch.arange(1, seq_len + 1).reshape(1, -1, 1)

        return nn.Parameter(index, requires_grad=False)

    def get_act_fun(self, act_fun):
        if act_fun == "relu":
            return F.relu
        elif act_fun == "elu":
            return 1 + F.elu

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor] = None,
        eps: Optional[float] = 1e-6,
    ):
        """Input shape: Sequence x Batch x Embedding
        Args:
            X (Tensor): `(N, L, E)`
            query (Tensor): `(N, L, E)` where L is the target sequence length, N is the batch size,
            E is the embedding dimension.
            key (Tensor): `(N, S, E)` where S is the source sequence length, N is the batch size,
            E is the embedding dimension.
            value (Tensor): `(N, S, E)` where S is the source sequence length, N is the batch size,
            E is the embedding dimension.
            mask (Optional[Tensor], optional): typically used to implement causal attention, 
            where the mask prevents the attention from looking forward in time (default: None).
        """
        num_heads = self.num_heads
        head_dim = self.head_dim
        bsz = k.size(0)
        src_len = q.size(1)
        tgt_len = k.size(1)

        # activation
        if self.act_fun == "relu":
            q = F.relu(q)
            k = F.relu(k)
        elif self.act_fun == "elu":
            q = F.elu(q) + 1
            k = F.elu(k) + 1
        else:
            assert(False)

        k = k * mask[:, :, None]
        v = v * mask[:, :, None]

        q = q.reshape(bsz, src_len, num_heads, head_dim).transpose(1, 2).reshape(-1, src_len, head_dim)
        k = k.reshape(bsz, tgt_len, num_heads, head_dim).transpose(1, 2).reshape(-1, tgt_len, head_dim)
        v = v.reshape(bsz, tgt_len, num_heads, head_dim).transpose(1, 2).reshape(-1, tgt_len, head_dim)
        
        if self.reweighting:
            # cos transform
            m = max(src_len, tgt_len)
            # get index and send to cuda
            weight_index = self.get_index(m).to(q)
            # (N * h, L, 2 * d)
            q_ = torch.cat([q * torch.sin(weight_index[:, :tgt_len, :] / m), q * torch.cos(weight_index[:, :tgt_len, :] / m)], dim=-1)
            # (N * h, S, 2 * d)
            k_ = torch.cat([k * torch.sin(weight_index[:, :src_len, :] / m), k * torch.cos(weight_index[:, :src_len, :] / m)], dim=-1)
        else:
            q_ = q
            k_ = k

        # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, 2 * d, d)
        kv_ = torch.einsum('nld,nlm->ndm', k_, v)
        # (N * h, L, 2 * d) (N * h, 2 * d) -> (N * h, L)
        z_ = 1. / torch.clamp_min(torch.einsum('nld,nd->nl', q_, torch.sum(k_, dim=-2)), eps)
        # (N * h, L, 2 * d) (N * h, d, 2 * d) (N * h, L) -> (N * h, L, d)
        attn_output = torch.einsum('nld,ndm,nl->nlm', q_, kv_, z_)
        # (N * h, L, d) -> (N, h, L, d)
        attn_output = attn_output.reshape(bsz, num_heads, tgt_len, head_dim)

        return attn_output
