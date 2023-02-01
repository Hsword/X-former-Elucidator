import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SparseSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.drop_attn = nn.Dropout(p=config["dropout_prob"])
        self.head_dim = config["head_dim"]
        self.block_size = config["block_size"]
        self.stride_c = config["stride_c"]
        self.has_block_global = not self.stride_c == -1
        if "use_cls_token" in config and config["use_cls_token"]:
            self.use_cls_token = True
        else:
            self.use_cls_token = False
        if (not "is_decoder" in config) or config["is_decoder"]:
            self.is_decoder = True
        else:
            self.is_decoder = False

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask):
        '''
        query, key, value: [bs, head, len, dim]
        '''
        assert(not self.is_decoder)
        # assert(self.use_cls_token)
        bs, head_num, seq_len, head_dim = query.shape
        block_size = self.block_size
        num_block = (seq_len + (block_size - 1)) // block_size

        orig_len = seq_len
        pad_len = num_block * block_size - orig_len
        X_pad = (0, 0, 0, pad_len)
        query = F.pad(query, X_pad)
        key = F.pad(key, X_pad)
        value = F.pad(value, X_pad)
        mask = F.pad(mask, (0, pad_len))
        cur_len = orig_len + pad_len
        
        # block local
        blocked_query = query.view(bs, head_num, num_block, block_size, head_dim)
        blocked_key = key.view(bs, head_num, num_block, block_size, head_dim)
        block_local_attn_scores = torch.einsum("shbxd,shbyd->shbxy", (blocked_query, blocked_key)).reshape(bs, head_num, cur_len, block_size)

        # mask for block local
        remove_mask = (mask == 0.)[:, None, :, None]
        float_mask = remove_mask.type_as(query).masked_fill(
            remove_mask, -10000.0
        )
        ones_mask = float_mask.new_ones(size=float_mask.size())

        float_mask = float_mask.view(bs, 1, num_block, block_size, 1)
        ones_mask = ones_mask.view(bs, 1, num_block, block_size, 1)
        blocked_local_mask = torch.einsum("...xd, ...yd -> ...xy", ones_mask, float_mask).reshape(bs, 1, cur_len, block_size)

        block_local_attn_scores += blocked_local_mask


        # block global
        if self.has_block_global:
            is_index_global_attn_unmasked = self._get_global_index(cur_len, block_size, self.stride_c)
            is_index_global_attn_unmasked = torch.tensor(is_index_global_attn_unmasked, device=mask.device, dtype=mask.dtype)
            
            is_index_global_attn = torch.logical_and(mask, is_index_global_attn_unmasked)
            # above can be replaced by mask * in_index_global_attn_nonzero_unmasked
            
            num_global_attn_indices = is_index_global_attn.long().sum(dim=1)
            max_num_global_attn_indices = num_global_attn_indices.max()
            is_index_global_attn_nonzero = is_index_global_attn.nonzero(as_tuple=True)
            
            is_local_index_global_attn = torch.arange(
                max_num_global_attn_indices, device=is_index_global_attn.device
            ) < num_global_attn_indices.unsqueeze(dim=-1)
            # location of the non-padding values within global attention indices
            is_local_index_global_attn_nonzero = is_local_index_global_attn.nonzero(as_tuple=True)
            # location of the padding values within global attention indices
            is_local_index_no_global_attn_nonzero = (is_local_index_global_attn == 0).nonzero(as_tuple=True)
            

            # global_key
            global_key = key.new_zeros(
                bs, head_num, max_num_global_attn_indices, head_dim
            )
            global_key[is_local_index_global_attn_nonzero[0], :, is_local_index_global_attn_nonzero[1]] \
                = key[is_index_global_attn_nonzero[0], :, is_index_global_attn_nonzero[1]]
            
            block_global_attn_scores = torch.einsum("shnd,shzd->shnz", (query, global_key))
            
            block_global_attn_scores[
                is_local_index_no_global_attn_nonzero[0], :, :, is_local_index_no_global_attn_nonzero[1]
            ] = -10000.
        
        # combine local and global scores
        if self.has_block_global:
            attn_scores = torch.cat(
                [block_local_attn_scores, block_global_attn_scores], dim=-1
            )
            del block_local_attn_scores
            del block_global_attn_scores
        else:
            attn_scores = block_local_attn_scores

        attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32)

        # apply dropout
        attn_probs = self.drop_attn(attn_probs)

        # block local
        block_local_attn_probs = attn_probs[:, :, :, :block_size].reshape(bs, head_num, num_block, block_size, block_size)
        blocked_value = value.view(bs, head_num, num_block, block_size, head_dim)
        block_local_attn_outputs = torch.matmul(
            block_local_attn_probs, blocked_value
        ).reshape(bs, head_num, cur_len, head_dim)
        
        # block global
        if self.has_block_global:
            block_global_attn_probs = attn_probs[:, :, :, block_size:]
            global_value = value.new_zeros(
                bs, head_num, max_num_global_attn_indices, head_dim
            )
            global_value[is_local_index_global_attn_nonzero[0], :, is_local_index_global_attn_nonzero[1]] \
                = value[is_index_global_attn_nonzero[0], :, is_index_global_attn_nonzero[1]]

            block_global_attn_outputs = torch.matmul(
                block_global_attn_probs, global_value
            )#.reshape(bs, head_num, cur_len, head_dim)
        else:
            block_global_attn_outputs = 0.
        
        attn_outputs = block_local_attn_outputs + block_global_attn_outputs
        return attn_outputs[:, :, :orig_len, :]

    @staticmethod
    def _get_global_index(seq_len, block_size, stride_c):
        r = np.arange(seq_len)
        k = r[np.newaxis, :]
        remainder = np.remainder(k, block_size)
        is_index_global_attn_unmasked = np.logical_or(remainder == 0, remainder >= block_size - stride_c)
        
        return is_index_global_attn_unmasked
        