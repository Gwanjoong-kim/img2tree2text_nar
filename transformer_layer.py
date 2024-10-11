import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerDecoderLayerBase(nn.Module):

    def __init__(
        self,
        embed_dim,
        ffn_embed_dim,
        num_heads,
        dropout,
        activation_fn,
        normalize_before
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.dropout_module = nn.Dropout(dropout)
        self.activation_fn = getattr(F, activation_fn)
        self.normalize_before = normalize_before
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.cross_attn_layer_norm = nn.LayerNorm(embed_dim)
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)

        # Feed Forward Network
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def residual_connection(self, x, residual):
        return residual + x
    
    def forward(
        self,
        x,                  
        encoder_out        
    ):
    
        residual = x
        
        if self.normalize_before:
            x = self.cross_attn_layer_norm(x)
                
        # before cross_attn x: torch.Size([10, 1, 1024]) encoder_out: torch.Size([4800, 1, 1024])

        # Apply attention with the mask
        x, _ = self.cross_attn(
            query=x,
            key=encoder_out,
            value=encoder_out,
        )
                
        x = self.dropout_module(x)
        x = residual + x
                
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x
        )
        
        x = self.dropout_module(x)
        x = residual + x
        
        # Feed-forward network
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        

        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x, _

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn