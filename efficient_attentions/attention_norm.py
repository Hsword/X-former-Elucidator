import torch
import torch.nn as nn
import efficient_attentions.utils.helpers as hp

def norm_kernelized_attention(Q, K, V, norm):
        '''
        Q -> (batch, head, maxlen, qkdim)
        K -> (batch, head, maxlen, qkdim)
        V -> (batch, head, maxlen, vdim)
        '''

        KV = torch.einsum('...nd,...ne->...de', K, V)
        out = torch.einsum("...de, ...nd -> ...ne", KV, Q)
        out = norm(out)
        return out

class NormSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_head = config["num_head"]
        self.head_dim = config["head_dim"]
        if "attn_norm_type" not in config:
            self.norm = hp.get_norm_fn("simple_rms_norm")(self.num_head * self.head_dim)
        else:
            self.norm = hp.get_norm_fn(config["attn_norm_type"])(self.num_head * self.head_dim)
        
        if "kernel_type" not in config:
            self.act = hp.get_act_fn("1+elu")
        else:
            self.act = hp.get_act_fn(config["kernel_type"])
        
        
    def forward(self, Q, K, V, mask):
        Q = self.act(Q)
        K = self.act(K)

        out = norm_kernelized_attention(Q, K * mask[:, None, :, None], V * mask[:, None, :, None], self.norm)
        return out
    