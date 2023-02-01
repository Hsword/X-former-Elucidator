import torch
import torch.nn as nn

import os
import sys
curr_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.dirname(curr_path)))
sys.path.append(root_path)
from efficient_attentions.attention_softmax import SoftmaxSelfAttention


class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.word_embeddings = nn.Embedding(config["vocab_size"], config["embedding_dim"])
        self.position_embeddings = nn.Embedding(config["max_seq_len"] + 2, config["embedding_dim"])
        self.token_type_embeddings = nn.Embedding(config["num_sen_type"], config["embedding_dim"])

        nn.init.normal_(self.word_embeddings.weight, std = 0.02)
        nn.init.normal_(self.position_embeddings.weight, std = 0.02)
        nn.init.normal_(self.token_type_embeddings.weight, std = 0.02)

        self.has_project = config["embedding_dim"] != config["hidden_dim"]
        if self.has_project:
            self.dense = nn.Linear(config["embedding_dim"], config["hidden_dim"])

        self.norm = nn.LayerNorm(config["hidden_dim"])
        self.dropout = nn.Dropout(p = config["dropout_prob"])

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.size()

        position_ids = torch.arange(seq_len, dtype = torch.long, device = input_ids.device)[None, :].repeat(batch_size, 1) + 2
        type_ids = torch.zeros(input_ids.size(), dtype = torch.long, device = input_ids.device)

        X_token = self.word_embeddings(input_ids)
        X_pos = self.position_embeddings(position_ids)
        X_seq = self.token_type_embeddings(type_ids)
        X = X_token + X_pos + X_seq

        if self.has_project:
            X = self.dense(X)

        X = self.norm(X)
        X = self.dropout(X)

        return X


class Attention(nn.Module):
    def __init__(self, config, lyr):
        super().__init__()

        self.hidden_dim = config["hidden_dim"]
        self.head_dim = config["head_dim"]
        self.num_head = config["num_head"]

        self.W_q = nn.Linear(self.hidden_dim, self.num_head * self.head_dim)
        self.W_k = nn.Linear(self.hidden_dim, self.num_head * self.head_dim)
        self.W_v = nn.Linear(self.hidden_dim, self.num_head * self.head_dim)

        self.attn = SoftmaxSelfAttention(config, lyr)

        self.W_o = nn.Linear(self.num_head * self.head_dim, self.hidden_dim)

    def forward(self, X, mask):
        Q = self.split_heads(self.W_q(X))
        K = self.split_heads(self.W_k(X))
        V = self.split_heads(self.W_v(X))

        attn_out = self.attn(Q.float(), K.float(), V.float(), mask.float())
        
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

        self.msa = Attention(config, lyr)

        self.dropout1 = nn.Dropout(p = config["dropout_prob"])
        self.norm1 = nn.LayerNorm(self.hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.ff_dim),
            nn.GELU(),
            nn.Linear(self.ff_dim, self.hidden_dim)
        )

        self.dropout2 = nn.Dropout(p = config["dropout_prob"])
        self.norm2 = nn.LayerNorm(self.hidden_dim)

    def forward(self, X, mask):
        msa_out = self.norm1(X + self.dropout1(self.msa(X, mask)))
        ffn_out = self.norm2(msa_out + self.dropout2(self.ffn(msa_out)))

        return ffn_out


class BlockNoMSA(nn.Module):
    def __init__(self, config, lyr):
        super().__init__()

        self.hidden_dim = config["hidden_dim"]
        self.ff_dim = config["ff_dim"]

        self.dropout1 = nn.Dropout(p = config["dropout_prob"])
        self.norm1 = nn.LayerNorm(self.hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.ff_dim),
            nn.GELU(),
            nn.Linear(self.ff_dim, self.hidden_dim)
        )

        self.dropout2 = nn.Dropout(p = config["dropout_prob"])
        self.norm2 = nn.LayerNorm(self.hidden_dim)

    def forward(self, X, mask):
        msa_out = self.norm1(X + self.dropout1(X))
        ffn_out = self.norm2(msa_out + self.dropout2(self.ffn(msa_out)))

        return ffn_out


class BlockNoFFN(nn.Module):
    def __init__(self, config, lyr):
        super().__init__()

        self.hidden_dim = config["hidden_dim"]

        self.msa = Attention(config, lyr)

        self.dropout1 = nn.Dropout(p = config["dropout_prob"])
        self.norm1 = nn.LayerNorm(self.hidden_dim)

        self.dropout2 = nn.Dropout(p = config["dropout_prob"])
        self.norm2 = nn.LayerNorm(self.hidden_dim)

    def forward(self, X, mask):
        msa_out = self.norm1(X + self.dropout1(self.msa(X, mask)))
        ffn_out = self.norm2(msa_out + self.dropout2(msa_out))

        return ffn_out


class BlockNoMSAAndFFN(nn.Module):
    def __init__(self, config, lyr):
        super().__init__()

        self.hidden_dim = config["hidden_dim"]

        self.dropout1 = nn.Dropout(p = config["dropout_prob"])
        self.norm1 = nn.LayerNorm(self.hidden_dim)

        self.dropout2 = nn.Dropout(p = config["dropout_prob"])
        self.norm2 = nn.LayerNorm(self.hidden_dim)

    def forward(self, X, mask):
        msa_out = self.norm1(X + self.dropout1(X))
        ffn_out = self.norm2(msa_out + self.dropout2(msa_out))

        return ffn_out


class BlockEmpty(nn.Module):
    def __init__(self, config, lyr):
        super().__init__()

    def forward(self, X, mask):
        return X


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.mode = config["mode"]

        self.num_layers = config["num_layers"]
        self.embeddings = Embeddings(config)

        for idx in range(self.num_layers):
            if self.mode == "normal":
                setattr(self, f"transformer_block_{idx}", Block(config, idx))
            elif self.mode == "no msa":
                setattr(self, f"transformer_block_{idx}", BlockNoMSA(config, idx))
            elif self.mode == "no ffn":
                setattr(self, f"transformer_block_{idx}", BlockNoFFN(config, idx))
            elif self.mode == "no msa and ffn":
                setattr(self, f"transformer_block_{idx}", BlockNoMSAAndFFN(config, idx))
            elif self.mode == "no block":
                setattr(self, f"transformer_block_{idx}", BlockEmpty(config, idx))

    def forward(self, input_ids, mask = None):
        X = self.embeddings(input_ids)

        if mask is None:
            mask = torch.ones_like(input_ids)

        for idx in range(self.num_layers):
            encoder = getattr(self, f"transformer_block_{idx}")
            X = encoder(X, mask)
        
        return X
