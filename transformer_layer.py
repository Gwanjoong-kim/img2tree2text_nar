import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerDecoderLayerBase(nn.Module):

    def __init__(
        self,
        embed_dim,
        ffn_embed_dim,
        attention_heads,
        num_heads,
        dropout,
        activation_fn='relu',
        normalize_before=False,
        no_encoder_attn=False,
        add_bias_kv=False,
        add_zero_attn=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.head_dim = embed_dim // attention_heads
        self.dropout = dropout
        self.dropout_module = nn.Dropout(dropout)
        self.activation_fn = getattr(F, activation_fn)
        self.normalize_before = normalize_before

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, attention_heads, dropout, add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn
        )
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)

        # Encoder attention
        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = nn.MultiheadAttention(
                embed_dim, attention_heads, dropout
            )
            self.encoder_attn_layer_norm = nn.LayerNorm(embed_dim)

        # Feed Forward Network
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def residual_connection(self, x, residual):
        return residual + x

    def forward(
        self,
        x,                        # [batch_size, seq_len, embed_dim]
        encoder_out=None,         # [batch_size, seq_len_enc, embed_dim]
        encoder_padding_mask=None,
        self_attn_mask=None,
        self_attn_padding_mask=None,
        need_attn=False,
        need_head_weights=False,
    ):
        residual = x

        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # Self-attention
        x, self_attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            num_heads=self.attention_heads,
            key_padding_mask=None,
            attn_mask=self_attn_mask,
            need_weights=False,
        )
        x = self.dropout_module(x)
        x = residual + x

        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # Encoder attention
        if encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=None,
                need_weights=need_attn,
            )
            x = self.dropout_module(x)
            x = residual + x

            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
        else:
            attn = None

        # Feed-forward network
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x

        if not self.normalize_before:
            x = self.final_layer_norm(x)

        return x, attn, None

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn