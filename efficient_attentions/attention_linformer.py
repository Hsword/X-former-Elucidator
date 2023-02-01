import torch
import torch.nn as nn
import math

class LinformerSelfAttention(nn.Module):
    projection_matrix = None

    def __init__(self, config):
        super().__init__()

        self.num_head = config["num_head"]
        self.head_dim = config["head_dim"]
        self.linformer_k = config["linformer_k"]
        if "max_seq_len" in config:
            self.seq_len = config["max_seq_len"]
        else:
            self.seq_len = (config["image_size"] // config["patch_size"])**2 + 1

        if LinformerSelfAttention.projection_matrix is not None and LinformerSelfAttention.projection_matrix.size(1) == self.seq_len:
            self.E = LinformerSelfAttention.projection_matrix
        else:
            LinformerSelfAttention.projection_matrix = nn.Parameter(torch.Tensor(self.linformer_k, self.seq_len))
            nn.init.normal_(LinformerSelfAttention.projection_matrix, std = 0.02)
            self.E = LinformerSelfAttention.projection_matrix

    def forward(self, Q, K, V, mask):
        K = torch.matmul(self.E, K * mask[:, None, :, None])
        V = torch.matmul(self.E, V * mask[:, None, :, None])

        dot = torch.matmul(Q, torch.transpose(K, -2, -1))
        dot = dot / math.sqrt(self.head_dim)

        attn = nn.functional.softmax(dot, dim = -1)

        X = torch.matmul(attn, V)

        return X

    def extra_repr(self):
        return f'linformer_k={self.linformer_k}'
