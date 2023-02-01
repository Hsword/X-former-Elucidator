import torch
import torch.nn as nn

class FactorizedRandomSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_type = config["attn_type"].split("-")[0]
        self.vis = ("vis" in config and config["vis"])
        self.lyr = config["layer"] if "layer" in config else None

        self.num_head = config["num_head"]
        self.factor_k = config["factor_k"]
        self.synthesizer_mode = config["synthesizer_mode"]
        if "max_seq_len" in config:
            self.seq_len = config["max_seq_len"]
        else:
            self.seq_len = (config["image_size"] // config["patch_size"])**2 + 1

        self.R1 = nn.Parameter(torch.Tensor(self.num_head, self.seq_len, self.factor_k))
        self.R2 = nn.Parameter(torch.Tensor(self.num_head, self.factor_k, self.seq_len))
        self.drop_attn = torch.nn.Dropout(p = config["dropout_prob"])

        torch.nn.init.normal_(self.R1, std = 0.02)
        torch.nn.init.normal_(self.R2, std = 0.02)

    def forward(self, V, mask):
        random_weights = torch.matmul(self.R1, self.R2)
        random_weights = torch.unsqueeze(random_weights, 0).repeat(V.shape[0], 1, 1, 1)
        random_weights = random_weights - 1e6 * (1 - mask[:, None, None, :])

        attn = nn.functional.softmax(random_weights, dim = -1)
        attn = self.drop_attn(attn)

        X = torch.matmul(attn, V)
        return X
    
    def extra_repr(self):
        return f'{self.synthesizer_mode}_k={self.factor_k}'
