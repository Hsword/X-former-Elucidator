import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from .model import Embeddings

import os
import sys
curr_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.dirname(curr_path)))
sys.path.append(root_path)
from efficient_attentions.attention_synthesizer import FactorizedRandomSelfAttention


class SyntheticAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.grad_checkpointing = config["attention_grad_checkpointing"]

        self.hidden_dim = config["hidden_dim"]
        self.head_dim = config["head_dim"]
        self.num_head = config["num_head"]

        self.attn_type = config["attn_type"]

        self.W_v = nn.Linear(self.hidden_dim, self.num_head * self.head_dim)

        self.attn = FactorizedRandomSelfAttention(config)

        self.W_o = nn.Linear(self.num_head * self.head_dim, self.hidden_dim)
    
    def forward(self, X, mask):
        V = self.split_heads(self.W_v(X))

        with torch.cuda.amp.autocast(enabled = False):
            if self.grad_checkpointing:
                attn_out = checkpoint(self.attn, V.float(), mask.float())
            else:
                attn_out = self.attn(V.float(), mask.float())
        attn_out = self.combine_heads(attn_out)

        out = self.W_o(attn_out)

        return out
    
    def combine_heads(self, X):
        X = X.transpose(1, 2)
        X = X.reshape(X.size(0), X.size(1), self.num_head * self.head_dim)
        return X
    
    def split_heads(self, X):
        X = X.reshape(X.size(0), X.size(1), self.num_head, self.head_dim)
        X = X.transpose(1, 2)
        return X


class SynthesizerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
    
        self.hidden_dim = config["hidden_dim"]
        self.ff_dim = config["ff_dim"]

        self.mha = SyntheticAttention(config)

        self.dropout1 = torch.nn.Dropout(p = config["dropout_prob"])
        self.norm1 = nn.LayerNorm(self.hidden_dim)

        self.ff = nn.Sequential(
            nn.Linear(self.hidden_dim, self.ff_dim),
            nn.GELU(),
            nn.Linear(self.ff_dim, self.hidden_dim)
        )

        self.dropout2 = torch.nn.Dropout(p = config["dropout_prob"])
        self.norm2 = nn.LayerNorm(self.hidden_dim)

    def forward(self, X, mask):
        mha_out = X + self.dropout1(self.mha(self.norm1(X), mask))
        ff_out = mha_out + self.dropout2(self.ff(self.norm2(mha_out)))

        return ff_out


class SynthesizerModel(nn.Module):
    def __init__(self, config, use_mask_token=False):
        super().__init__()

        self.num_layers = config["num_layers"]
        self.hidden_dim = config["hidden_dim"]
        self.embeddings = Embeddings(config, use_mask_token=use_mask_token)
        self.norm = nn.LayerNorm(self.hidden_dim)

        for idx in range(self.num_layers):
            setattr(self, f"synthesizer_vit_block_{idx}", SynthesizerBlock(config))

    def forward(self, pixel_values, bool_masked_pos=None):
        X = self.embeddings(pixel_values, bool_masked_pos)

        batch_size, seq_len, _ = X.size()
        mask = torch.ones((batch_size, seq_len), device=X.device)

        for idx in range(self.num_layers):
            encoder = getattr(self, f"synthesizer_vit_block_{idx}")
            X = encoder(X, mask)
        
        X = self.norm(X)

        return X
