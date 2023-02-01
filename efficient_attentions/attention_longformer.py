from transformers.models.longformer.modeling_longformer import LongformerSelfAttention as LSelfAttention
from transformers.models.longformer.modeling_longformer import LongformerConfig

class LongformerSelfAttention(LSelfAttention):
    def __init__(self, config, query, key, value):
        
        self.window_size = config["window_size"]
        self.first_token_view = config["first_token_view"]
        if "use_cls_token" in config and config["use_cls_token"]:
            self.first_token_view = True
        longformer_config = LongformerConfig()
        longformer_config.num_attention_heads = config["num_head"]
        longformer_config.hidden_size = config["hidden_dim"]
        longformer_config.attention_window = [config["window_size"]]

        super().__init__(longformer_config, 0)

        self.query.weight = query.weight
        self.query_global.weight = query.weight

        self.key.weight = key.weight
        self.key_global.weight = key.weight

        self.value.weight = value.weight
        self.value_global.weight = value.weight

        self.query.bias = query.bias
        self.query_global.bias = query.bias

        self.key.bias = key.bias
        self.key_global.bias = key.bias

        self.value.bias = value.bias
        self.value_global.bias = value.bias

    def forward(self, X, mask):
        mask = mask - 1
        if self.first_token_view:
            mask[:, 0] = 1
        is_index_masked = mask < 0
        is_index_global_attn = mask > 0
        is_global_attn = is_index_global_attn.flatten().any().item()

        (out,) = super().forward(
            hidden_states = X,
            attention_mask = mask,
            is_index_masked = is_index_masked,
            is_index_global_attn = is_index_global_attn,
            is_global_attn = is_global_attn
        )
        
        return out

    def extra_repr(self):
        return f'window_size={self.window_size}'
