from fast_transformers.attention import ImprovedClusteredAttention

class ClusteredSelfAttention(ImprovedClusteredAttention):
    def __init__(self, config):
        
        self.num_clusters = config["num_clusters"]
        self.top_k = config["topk"] # default 32
        self.bits = config["bits"] # default 32
        super().__init__(
            self.num_clusters,
            attention_dropout = config["dropout_prob"],
            bits = config["bits"],
            topk = config["topk"],
            attn_vis=self.vis
        )
    
    def forward(self, Q, K, V, attn_mask, query_lengths, key_lengths):
        out = super().forward(
            Q, K, V,
            attn_mask,
            query_lengths,
            key_lengths
        )
        
        return out
    
    def extra_repr(self):
        return f'num_clusters={self.num_clusters}'
        