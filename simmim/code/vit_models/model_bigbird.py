import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from .model import Embeddings

import os
import sys
curr_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.dirname(curr_path)))
sys.path.append(root_path)
from efficient_attentions.attention_bigbird import BigBirdSelfAttention


class BigBirdAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.grad_checkpointing = config["attention_grad_checkpointing"]

        self.hidden_dim = config["hidden_dim"]
        self.head_dim = config["head_dim"]
        self.num_head = config["num_head"]
        self.block_size = config["block_size"]

        self.attn_type = config["attn_type"]

        self.W_q = nn.Linear(self.hidden_dim, self.num_head * self.head_dim)
        self.W_k = nn.Linear(self.hidden_dim, self.num_head * self.head_dim)
        self.W_v = nn.Linear(self.hidden_dim, self.num_head * self.head_dim)

        self.attn = BigBirdSelfAttention(config, self.W_q, self.W_k, self.W_v)

        self.W_o = nn.Linear(self.num_head * self.head_dim, self.hidden_dim)
    
    def forward(self, X, band_mask, from_mask, to_mask, blocked_encoder_mask):
        # pad to fit seq_len % block_size == 0
        orig_len = X.shape[-2]
        pad_len = self.block_size - (orig_len % self.block_size)
        X_pad = (0, 0, 0, pad_len)
        X = F.pad(X, X_pad)

        with torch.cuda.amp.autocast(enabled = False):
            if self.grad_checkpointing:
                attn_out = checkpoint(self.attn, X.float(), band_mask, from_mask, to_mask, blocked_encoder_mask)
            else:
                attn_out = self.attn(X.float(), band_mask, from_mask, to_mask, blocked_encoder_mask)

        out = self.W_o(attn_out)
        # back to original shape
        out = out[:, :orig_len]

        return out


class BigBirdBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
    
        self.hidden_dim = config["hidden_dim"]
        self.ff_dim = config["ff_dim"]

        self.mha = BigBirdAttention(config)

        self.dropout1 = torch.nn.Dropout(p = config["dropout_prob"])
        self.norm1 = nn.LayerNorm(self.hidden_dim)

        self.ff = nn.Sequential(
            nn.Linear(self.hidden_dim, self.ff_dim),
            nn.GELU(),
            nn.Linear(self.ff_dim, self.hidden_dim)
        )

        self.dropout2 = torch.nn.Dropout(p = config["dropout_prob"])
        self.norm2 = nn.LayerNorm(self.hidden_dim)

    def forward(self, X, band_mask, from_mask, to_mask, blocked_encoder_mask):
        mha_out = X + self.dropout1(self.mha(self.norm1(X), band_mask, from_mask, to_mask, blocked_encoder_mask))
        ff_out = mha_out + self.dropout2(self.ff(self.norm2(mha_out)))

        return ff_out


class BigBirdModel(nn.Module):
    def __init__(self, config, use_mask_token=False):
        super().__init__()

        self.num_layers = config["num_layers"]
        self.block_size = config["block_size"]
        self.hidden_dim = config["hidden_dim"]
        self.embeddings = Embeddings(config, use_mask_token=use_mask_token)
        self.norm = nn.LayerNorm(self.hidden_dim)

        for idx in range(self.num_layers):
            setattr(self, f"bigbird_vit_block_{idx}", BigBirdBlock(config))

    def forward(self, pixel_values, bool_masked_pos=None):
        X = self.embeddings(pixel_values, bool_masked_pos)

        batch_size, seq_len, _ = X.size()
        mask = torch.ones((batch_size, seq_len), device=X.device)

        # pad to fit seq_len % block_size == 0
        orig_len = seq_len
        pad_len = self.block_size - (orig_len % self.block_size)
        mask_pad = (0, pad_len)
        mask = F.pad(mask, mask_pad)
        
        blocked_encoder_mask, band_mask, from_mask, to_mask = self.create_masks_for_block_sparse_attn(
            mask, self.block_size
        )

        for idx in range(self.num_layers):
            encoder = getattr(self, f"bigbird_vit_block_{idx}")
            X = encoder(
                X,
                band_mask = band_mask,
                from_mask = from_mask,
                to_mask = to_mask,
                blocked_encoder_mask = blocked_encoder_mask
            )
        
        X = self.norm(X)
        
        return X
    
    # implemented in transformers 4.17.0
    @staticmethod
    def create_masks_for_block_sparse_attn(attention_mask: torch.Tensor, block_size: int):

        batch_size, seq_length = attention_mask.size()
        assert (
            seq_length % block_size == 0
        ), f"Sequence length must be multiple of block size, but sequence length is {seq_length}, while block size is {block_size}."

        def create_band_mask_from_inputs(from_blocked_mask, to_blocked_mask):
            """
            Create 3D attention mask from a 2D tensor mask.

            Args:
                from_blocked_mask: 2D Tensor of shape [batch_size,
                from_seq_length//from_block_size, from_block_size].
                to_blocked_mask: int32 Tensor of shape [batch_size,
                to_seq_length//to_block_size, to_block_size].

            Returns:
                float Tensor of shape [batch_size, 1, from_seq_length//from_block_size-4, from_block_size,
                3*to_block_size].
            """
            exp_blocked_to_pad = torch.cat(
                [to_blocked_mask[:, 1:-3], to_blocked_mask[:, 2:-2], to_blocked_mask[:, 3:-1]], dim=2
            )
            band_mask = torch.einsum("blq,blk->blqk", from_blocked_mask[:, 2:-2], exp_blocked_to_pad)
            band_mask.unsqueeze_(1)
            return band_mask

        blocked_encoder_mask = attention_mask.view(batch_size, seq_length // block_size, block_size)
        band_mask = create_band_mask_from_inputs(blocked_encoder_mask, blocked_encoder_mask)

        from_mask = attention_mask.view(batch_size, 1, seq_length, 1)
        to_mask = attention_mask.view(batch_size, 1, 1, seq_length)

        return blocked_encoder_mask, band_mask, from_mask, to_mask
