from transformers.models.big_bird.modeling_big_bird import BigBirdBlockSparseAttention, BigBirdConfig

class BigBirdSelfAttention(BigBirdBlockSparseAttention):
    def __init__(self, config, query, key, value):
        
        self.block_size = config["block_size"]
        self.num_random_blocks = config["num_random_blocks"] # default 3
        big_bird_config = BigBirdConfig()
        if "max_seq_len" in config:
            big_bird_config.max_position_embeddings = config["max_seq_len"]
        else:
            orig_seqlen = (config["image_size"] // config["patch_size"])**2 + 1
            extra_seqlen = self.block_size - (orig_seqlen % self.block_size)
            big_bird_config.max_position_embeddings = orig_seqlen + extra_seqlen
        big_bird_config.num_attention_heads = config["num_head"]
        big_bird_config.block_size = config["block_size"]
        big_bird_config.num_random_blocks = config["num_random_blocks"]
        big_bird_config.hidden_size = config["hidden_dim"]
        
        super().__init__(big_bird_config)

        self.query.weight = query.weight
        self.key.weight = key.weight
        self.value.weight = value.weight

        self.query.bias = query.bias
        self.key.bias = key.bias
        self.value.bias = value.bias

    def forward(self, X, band_mask, from_mask, to_mask, blocked_encoder_mask):
        (out,) = super().forward(
            hidden_states = X,
            band_mask=band_mask,
            from_mask=from_mask,
            to_mask=to_mask,
            from_blocked_mask=blocked_encoder_mask,
            to_blocked_mask=blocked_encoder_mask
        )

        return out
    
    def extra_repr(self):
        return f'block_size={self.block_size}'