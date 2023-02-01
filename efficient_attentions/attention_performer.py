import torch
import torch.nn as nn
from performer_pytorch import FastAttention

class PerformerSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.head_dim = config["head_dim"]
        self.rp_dim = config["rp_dim"]
        self.generalized_attention = config["generalized_attention"]
        self.kernel_type = config["kernel_type"]
        if self.kernel_type == "relu":
            self.attn_fn = FastAttention(
                dim_heads=self.head_dim,
                nb_features=self.rp_dim,
                causal=False,
                generalized_attention=self.generalized_attention,
                kernel_fn = nn.ReLU(),
                out_qkv=self.vis
            )
        elif self.kernel_type == "exp":
            self.attn_fn = FastAttention(
                dim_heads=self.head_dim,
                nb_features=self.rp_dim,
                causal=False,
                generalized_attention=self.generalized_attention,
                kernel_fn = torch.exp,
                out_qkv=self.vis
            )

    def forward(self, Q, K, V, mask):
        out = self.attn_fn(Q, K * mask[:, None, :, None], V * mask[:, None, :, None])

        return out

    def extra_repr(self):
        return f'rp_dim={self.rp_dim}, kernel_type={self.kernel_type}'
