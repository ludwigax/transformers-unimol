from transformers import PretrainedConfig

class UnimolConfig(PretrainedConfig):

    model_type = "unimol"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=30, # TODO
        hidden_size=512,
        intermediate_size=2048,
        num_hidden_layers=15,
        num_attention_heads=64,
        num_key_value_heads=None,
        hidden_mlp="enc_mlp",
        hidden_norm="ln",
        hidden_act="silu",
        embed_norm="ln",
        head_act="tanh",
        num_kernels = 128,
        num_edge_types = 900,
        max_seq_len = 512,
        initializer_range=0.02,
        norm_eps=1e-6,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        attention_bias=True,
        attention_dropout=0.0,
        mlp_bias=True,
        post_norm=True,
        num_labels=1,
        problem_type="regression",
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.hidden_mlp = hidden_mlp
        self.hidden_norm = hidden_norm
        self.hidden_act = hidden_act
        self.embed_norm = embed_norm
        self.head_act = head_act
        self.num_kernels = num_kernels
        self.num_edge_types = num_edge_types
        self.max_seq_len = max_seq_len
        self.initializer_range = initializer_range
        self.norm_eps = norm_eps
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.post_norm = post_norm

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            num_labels=num_labels,
            problem_type=problem_type,
            **kwargs,
        )