import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.drop_attn = nn.Dropout(p=config["dropout_prob"])
        self.head_dim = config["head_dim"]
        self.block_size = config["block_size"]
    
    def forward(self, Q, K, V, mask):
        '''
        Q, K, V: [bs, head, len, dim]
        '''
        bs, head_num, seq_len, head_dim = Q.shape
        block_size = self.block_size
        head_dim = self.head_dim
        num_block = (seq_len + (block_size - 1)) // block_size

        orig_len = seq_len
        if orig_len % self.block_size == 0:
            pad_len = 0
        else:
            pad_len = self.block_size - (orig_len % self.block_size)
        X_pad = (0, 0, 0, pad_len)
        Q = F.pad(Q, X_pad)
        K = F.pad(K, X_pad)
        V = F.pad(V, X_pad)
        mask = F.pad(mask, (0, pad_len))
        cur_len = orig_len + pad_len

        Q = Q.view(bs, head_num, num_block, block_size, head_dim)
        K = K.view(bs, head_num, num_block, block_size, head_dim)
        V = V.view(bs, head_num, num_block, block_size, head_dim)

        attn_scores = torch.einsum("...xd, ...yd -> ...xy", Q, K)
        
        remove_mask = (mask == 0.)[:, None, :, None]
        float_mask = remove_mask.type_as(Q).masked_fill(
            remove_mask, -10000.0
        )
        ones_mask = float_mask.new_ones(size=float_mask.size())

        float_mask = float_mask.view(bs, 1, num_block, block_size, 1)
        ones_mask = ones_mask.view(bs, 1, num_block, block_size, 1)
        blocked_local_mask = torch.einsum("...xd, ...yd -> ...xy", ones_mask, float_mask)
        
        attn_scores += blocked_local_mask

        attn_scores = attn_scores.reshape(bs, head_num, cur_len, block_size)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = torch.masked_fill(attn_probs, remove_mask, 0.0)

        del attn_scores

        attn_probs = self.drop_attn(attn_probs)

        out = torch.matmul(
            attn_probs.reshape(bs, head_num, num_block, block_size, block_size),
            V
        )
        out = out.reshape(bs, head_num, cur_len, head_dim)[:, :, :orig_len, :]
        return out