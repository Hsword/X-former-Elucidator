from transformers.models.reformer.modeling_reformer import LSHSelfAttention, ReformerConfig

class ReformerSelfAttention(LSHSelfAttention):
    def __init__(self, config, query, value):
        
        self.num_hash = config["num_hash"]
        self.chunk_len = config["chunk_len"]
        reformer_config = ReformerConfig()
        reformer_config.attention_head_size = config["head_dim"]
        reformer_config.num_attention_heads = config["num_head"]
        reformer_config.attn_layers = ["lsh"]
        reformer_config.num_hashes = config["num_hash"]
        reformer_config.lsh_attn_chunk_length = config["chunk_len"]
        reformer_config.is_decoder = False
        if "max_seq_len" in config:
            reformer_config.max_position_embeddings = config["max_seq_len"]
        else:
            orig_len = (config["image_size"] // config["patch_size"])**2 + 1
            pad_len = self.chunk_len - (orig_len % self.chunk_len)
            reformer_config.max_position_embeddings = orig_len + pad_len
        reformer_config.hidden_size = config["hidden_dim"]
        super().__init__(reformer_config)
        self.query_key.weight = query.weight
        self.value.weight = value.weight

    def forward(self, X, mask):
        lsh_attn_out = super().forward(
            hidden_states=X,
            attention_mask=mask,
        )
        out = lsh_attn_out.hidden_states
        
        return out

    def extra_repr(self):
        return f'num_hash={self.num_hash}'
