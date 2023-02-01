import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from .model import Embeddings

import os
import sys
curr_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.dirname(curr_path)))
sys.path.append(root_path)
from efficient_attentions.attention_reformer import ReformerSelfAttention


class ReformerAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.grad_checkpointing = True if "attention_grad_checkpointing" in config and config["attention_grad_checkpointing"] else False

        self.hidden_dim = config["hidden_dim"]
        self.head_dim = config["head_dim"]
        self.num_head = config["num_head"]

        self.attn_type = config["attn_type"]

        self.W_q = nn.Linear(self.hidden_dim, self.num_head * self.head_dim)
        self.W_v = nn.Linear(self.hidden_dim, self.num_head * self.head_dim)

        self.attn = ReformerSelfAttention(config, self.W_q, self.W_v)

        self.W_o = nn.Linear(self.num_head * self.head_dim, self.hidden_dim)

    def forward(self, X, mask):
        with torch.cuda.amp.autocast(enabled = False):
            if self.grad_checkpointing:
                attn_out = checkpoint(self.attn, X.float(), mask.float())
            else:
                attn_out = self.attn(X.float(), mask.float())

        out = self.W_o(attn_out)
        return out


class ReformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.hidden_dim = config["hidden_dim"]
        self.ff_dim = config["ff_dim"]

        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.mha = ReformerAttention(config)
        self.dropout1 = nn.Dropout(p=config["dropout_prob"])

        self.norm2 = nn.LayerNorm(self.hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(self.hidden_dim, self.ff_dim),
            nn.GELU(),
            nn.Dropout(p=config["dropout_prob"]),
            nn.Linear(self.ff_dim, self.hidden_dim),
            nn.Dropout(p=config["dropout_prob"])
        )

    def forward(self, X, mask):
        mha_out = X + self.dropout1(self.mha(self.norm1(X), mask))
        ff_out = mha_out + self.ff(self.norm2(mha_out))

        return ff_out


class ReformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_layers = config["num_layers"]
        self.hidden_dim = config["hidden_dim"]
        self.pooling_mode = config["pooling_mode"]

        self.embeddings = Embeddings(config)
        
        for idx in range(self.num_layers):
            setattr(self, f"reformer_block_{idx}", ReformerBlock(config))
        
        self.norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, input_ids, mask=None):
        
        X = self.embeddings(input_ids)

        if mask is None:
            mask = torch.ones_like(input_ids)
        
        for idx in range(self.num_layers):
            encoder = getattr(self, f"reformer_block_{idx}")
            X = encoder(X, mask)
        
        # X = self.norm(X)
        X = self.norm(X) * mask[:, :, None]

        return X
