import torch
import torch.nn as nn
import math
from torch.utils.checkpoint import checkpoint


class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config["embedding_dim"] == config["hidden_dim"]

        self.word_embeddings = nn.Embedding(config["vocab_size"], config["embedding_dim"])
        self.position_embeddings = nn.Embedding(config["max_seq_len"], config["embedding_dim"])

        nn.init.normal_(self.word_embeddings.weight, std = 0.02)
        nn.init.normal_(self.position_embeddings.weight, std = 0.02)

        self.dropout = nn.Dropout(p = config["dropout_prob"])

    def forward(self, input_ids):

        batch_size, seq_len = input_ids.size()
        position_ids = torch.arange(seq_len, dtype = torch.long, device = input_ids.device)[None, :].repeat(batch_size, 1)
        X_token = self.word_embeddings(input_ids)
        X_pos = self.position_embeddings(position_ids)

        X = X_token + X_pos
        X = self.dropout(X)

        return X


class SoftmaxSelfAttention(nn.Module):
    def __init__(self, config, lyr):
        super().__init__()
        self.lyr = lyr
        self.drop_attn = nn.Dropout(p = config["dropout_prob"])
        self.head_dim = config["head_dim"]

    def forward(self, Q, K, V, mask, output_QK=False):
        dot = torch.matmul(Q, torch.transpose(K, -2, -1))
        dot = dot / math.sqrt(self.head_dim)
        dot = dot - 1e6 * (1 - mask[:, None, None, :])

        attn = nn.functional.softmax(dot, dim = -1)
        attn = self.drop_attn(attn)

        X = torch.matmul(attn, V)

        if output_QK:
            return X, (Q, K, mask)
        else:
            return X


class Attention(nn.Module):
    def __init__(self, config, lyr):
        super().__init__()

        self.grad_checkpointing = True if "attention_grad_checkpointing" in config and config["attention_grad_checkpointing"] else False

        self.hidden_dim = config["hidden_dim"]
        self.head_dim = config["head_dim"]
        self.num_head = config["num_head"]

        self.attn_type = config["attn_type"]

        self.W_q = nn.Linear(self.hidden_dim, self.num_head * self.head_dim)
        self.W_k = nn.Linear(self.hidden_dim, self.num_head * self.head_dim)
        self.W_v = nn.Linear(self.hidden_dim, self.num_head * self.head_dim)

        self.attn = SoftmaxSelfAttention(config, lyr)

        self.W_o = nn.Linear(self.num_head * self.head_dim, self.hidden_dim)

    def forward(self, X, mask, output_QK=False):

        Q = self.split_heads(self.W_q(X))
        K = self.split_heads(self.W_k(X))
        V = self.split_heads(self.W_v(X))
        with torch.cuda.amp.autocast(enabled = False):
            if output_QK:
                if self.grad_checkpointing:
                    attn_out, QK = checkpoint(self.attn, Q.float(), K.float(), V.float(), mask.float(), output_QK=True)
                else:
                    attn_out, QK = self.attn(Q.float(), K.float(), V.float(), mask.float(), output_QK=True)
            else:
                if self.grad_checkpointing:
                    attn_out = checkpoint(self.attn, Q.float(), K.float(), V.float(), mask.float())
                else:
                    attn_out = self.attn(Q.float(), K.float(), V.float(), mask.float())
        attn_out = self.combine_heads(attn_out)

        out = self.W_o(attn_out)

        if output_QK:
            return out, QK
        else:
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

        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.mha = Attention(config, lyr)
        self.dropout1 = nn.Dropout(p=config["dropout_prob"])

        self.norm2 = nn.LayerNorm(self.hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(self.hidden_dim, self.ff_dim),
            nn.GELU(),
            nn.Dropout(p=config["dropout_prob"]),
            nn.Linear(self.ff_dim, self.hidden_dim),
            nn.Dropout(p=config["dropout_prob"])
        )

    def forward(self, X, mask, output_QK=False):
        if output_QK:
            mha_out, QK = self.mha(self.norm1(X), mask, output_QK=True)
        else:
            mha_out = self.mha(self.norm1(X), mask)
        mha_out = X + self.dropout1(mha_out)
        ff_out = mha_out + self.ff(self.norm2(mha_out))

        if output_QK:
            return ff_out, QK
        else:
            return ff_out


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_layers = config["num_layers"]
        self.hidden_dim = config["hidden_dim"]
        self.pooling_mode = config["pooling_mode"]

        self.embeddings = Embeddings(config)
        
        for idx in range(self.num_layers):
            setattr(self, f"transformer_block_{idx}", Block(config, idx))
        
        self.norm = nn.LayerNorm(self.hidden_dim)

        self.output_QK = "output_QK" in config and config["output_QK"]

    def forward(self, input_ids, mask=None):
        
        X = self.embeddings(input_ids)

        if mask is None:
            mask = torch.ones_like(input_ids)
        
        if self.output_QK:
            list_QK = list()
            for idx in range(self.num_layers):
                encoder = getattr(self, f"transformer_block_{idx}")
                X, QK = encoder(X, mask, output_QK=True)
                list_QK.append(QK)
        else:
            for idx in range(self.num_layers):
                encoder = getattr(self, f"transformer_block_{idx}")
                X = encoder(X, mask)
        
        # X = self.norm(X)
        X = self.norm(X) * mask[:, :, None]

        if self.output_QK:
            return X, list_QK
        else:
            return X
