# Based on Huggingface modeling_vit.py

import torch
import torch.nn as nn
import math
from torch.utils.checkpoint import checkpoint
import numpy as np
import collections.abc


def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)


# Not support interpolate pos encoding yet.
class Embeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings.
    Optionally, also the mask tokens.
    """
    def __init__(self, config, use_mask_token=False, drop_last=False):
        super().__init__()

        self.use_mask_token = use_mask_token
        self.drop_last = drop_last
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config["embedding_dim"]))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config["embedding_dim"])) if self.use_mask_token else None
        self.patch_embeddings = PatchEmbeddings(
            image_size=config["image_size"],
            patch_size=config["patch_size"],
            num_channels=config["num_channels"],
            embed_dim=config["embedding_dim"],
        )
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config["embedding_dim"])) if not self.drop_last else nn.Parameter(torch.zeros(1, num_patches, config["embedding_dim"]))
        
        self.has_project = config["embedding_dim"] != config["hidden_dim"]
        if self.has_project:
            self.dense = nn.Linear(config["embedding_dim"], config["hidden_dim"])
        
        self.dropout = nn.Dropout(p=config["dropout_prob"])


    def forward(self, pixel_values, bool_masked_pos=None):
        batch_size, num_channels, height, width = pixel_values.size()
        X = self.patch_embeddings(pixel_values)
        batch_size, seq_len, _ = X.size()

        if bool_masked_pos is not None:
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            X = X * (1.0 - mask) + mask_tokens * mask
        
        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        X = torch.cat((cls_tokens, X), dim=1)
        if self.drop_last:
            X = X[:, :-1, :]

        X = X + self.position_embeddings

        if self.has_project:
            X = self.dense(X)

        X = self.dropout(X)

        return X


class PatchEmbeddings(nn.Module):
    """
    Image to Patch Embbedding.
    """

    def __init__(self, image_size=64, patch_size=2, num_channels=3, embed_dim=256):
        super().__init__()
        image_size = to_2tuple(image_size)
        patch_size = to_2tuple(patch_size)
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
    
        self.projection = nn.Conv2d(
            num_channels, embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size)
    
    def forward(self, pixel_values):        
        X = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return X


class SoftmaxSelfAttention(nn.Module):
    def __init__(self, config, lyr):
        super().__init__()
        self.print_attn = ("print_attn" in config and config["print_attn"])
        self.lyr = lyr
        self.drop_attn = nn.Dropout(p = config["dropout_prob"])
        self.head_dim = config["head_dim"]

    def forward(self, Q, K, V):
        dot = torch.matmul(Q, torch.transpose(K, -2, -1))
        dot = dot / math.sqrt(self.head_dim)

        attn = nn.functional.softmax(dot, dim = -1)
        attn = self.drop_attn(attn)

        X = torch.matmul(attn, V)
        return X


class Attention(nn.Module):
    def __init__(self, config, lyr):
        super().__init__()

        self.grad_checkpointing = config["attention_grad_checkpointing"]

        self.hidden_dim = config["hidden_dim"]
        self.head_dim = config["head_dim"]
        self.num_head = config["num_head"]

        self.attn_type = config["attn_type"]

        self.W_q = nn.Linear(self.hidden_dim, self.num_head * self.head_dim)
        self.W_k = nn.Linear(self.hidden_dim, self.num_head * self.head_dim)
        self.W_v = nn.Linear(self.hidden_dim, self.num_head * self.head_dim)

        self.attn = SoftmaxSelfAttention(config, lyr)

        self.W_o = nn.Linear(self.num_head * self.head_dim, self.hidden_dim)

    def forward(self, X):
        Q = self.split_heads(self.W_q(X))
        K = self.split_heads(self.W_k(X))
        V = self.split_heads(self.W_v(X))
        with torch.cuda.amp.autocast(enabled = False):
            if self.grad_checkpointing:
                attn_out = checkpoint(self.attn, Q.float(), K.float(), V.float())
            else:
                attn_out = self.attn(Q.float(), K.float(), V.float())
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


class Block(nn.Module):
    def __init__(self, config, lyr):
        super().__init__()

        self.hidden_dim = config["hidden_dim"]
        self.ff_dim = config["ff_dim"]

        self.mha = Attention(config, lyr)

        self.dropout1 = torch.nn.Dropout(p = config["dropout_prob"])
        self.norm1 = nn.LayerNorm(self.hidden_dim)

        self.ff = nn.Sequential(
            nn.Linear(self.hidden_dim, self.ff_dim),
            nn.GELU(),
            nn.Linear(self.ff_dim, self.hidden_dim)
        )

        self.dropout2 = torch.nn.Dropout(p = config["dropout_prob"])
        self.norm2 = nn.LayerNorm(self.hidden_dim)

    def forward(self, X):
        mha_out = X + self.dropout1(self.mha(self.norm1(X)))
        ff_out = mha_out + self.dropout2(self.ff(self.norm2(mha_out)))

        return ff_out


class Model(nn.Module):
    def __init__(self, config, use_mask_token=False):
        super().__init__()

        self.num_layers = config["num_layers"]
        self.hidden_dim = config["hidden_dim"]
        self.embeddings = Embeddings(config, use_mask_token=use_mask_token)
        self.norm = nn.LayerNorm(self.hidden_dim)

        for idx in range(self.num_layers):
            setattr(self, f"vit_block_{idx}", Block(config, idx))

    def forward(self, pixel_values, bool_masked_pos=None):
        X = self.embeddings(pixel_values, bool_masked_pos)

        for idx in range(self.num_layers):
            encoder = getattr(self, f"vit_block_{idx}")
            X = encoder(X)
        
        X = self.norm(X)

        return X
