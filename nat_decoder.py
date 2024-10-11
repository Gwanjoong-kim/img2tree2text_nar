import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformer_layer import TransformerDecoderLayerBase

class PositionalEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx, learned=False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.learned = learned
        if self.learned:
            self.weight = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        else:
            self.register_buffer('weight', self._create_sinusoidal_embeddings(num_embeddings, embedding_dim, padding_idx))

    def _create_sinusoidal_embeddings(self, num_embeddings, embedding_dim, padding_idx):
        position = torch.arange(0, num_embeddings).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))
        embeddings = torch.zeros(num_embeddings, embedding_dim)
        embeddings[:, 0::2] = torch.sin(position * div_term)
        embeddings[:, 1::2] = torch.cos(position * div_term)
        if padding_idx is not None:
            embeddings[padding_idx, :] = 0
        return embeddings

    def forward(self, input_ids):
        positions = torch.arange(0, input_ids.size(1), dtype=torch.long, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(input_ids.size(0), -1)
        if self.learned:
            pos_emb = self.weight(positions)
        else: 
            pos_emb = self.weight[positions]
            
        print(f"positions: {positions.shape}") 
        pos_emb = pos_emb.to(input_ids.device)
        return pos_emb

# Transformer Decoder
class NATransformerDecoder(nn.Module):
    def __init__(self, dictionary, cfg, output_projection=None):
        super().__init__()

        self.embed_dim = cfg.decoder_embed_dim
        self.max_target_length = cfg.max_target_length
        self.dictionary = dictionary
        self.padding_idx = dictionary['<pad>']
        self.bos_idx = dictionary['<bos>']
        self.eos_idx = dictionary['<eos>']
        self.vocab_size = len(dictionary)
        self.embed_scale = 1.0 if cfg.no_scale_embedding else math.sqrt(self.embed_dim)

        # Token Embedding
        self.embed_tokens = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embed_dim,
            padding_idx=self.padding_idx
        )

        # Positional Embedding
        self.embed_positions = PositionalEmbedding(
            num_embeddings=self.max_target_length,
            embedding_dim=self.embed_dim,
            padding_idx=self.padding_idx,
            learned=cfg.decoder_learned_pos,
        )

        # Output Projection
        self.output_projection = nn.Linear(self.embed_dim, self.vocab_size, bias=False)
        nn.init.normal_(
            self.output_projection.weight, mean=0, std=self.embed_dim ** -0.5
        )

        # Decoder layers
        self.layers = nn.ModuleList(
            [self.build_decoder_layer(cfg, no_encoder_attn=False) for _ in range(cfg.decoder_layers)]
        )

    def build_decoder_layer(self, cfg, no_encoder_attn=False):
        decoder_layer = TransformerDecoderLayerBase(
            num_heads=cfg.decoder_num_heads,
            embed_dim=cfg.decoder_embed_dim,
            ffn_embed_dim=cfg.decoder_ffn_embed_dim,
            attention_heads=cfg.decoder_attention_heads,
            dropout=cfg.dropout,
            activation_fn=cfg.activation_fn,
            normalize_before=cfg.decoder_normalize_before,
            no_encoder_attn=no_encoder_attn,
            add_bias_kv=False,
            add_zero_attn=False
        )
        return decoder_layer

    def forward_embedding(self, decoder_input_ids):
        # Get positional embeddings
        positions = self.embed_positions(decoder_input_ids)

        # Token embeddings with positional embeddings
        x = self.embed_scale * self.embed_tokens(decoder_input_ids)
        x += positions

        # Padding mask
        decoder_padding_mask = decoder_input_ids.eq(self.padding_idx)

        return x, decoder_padding_mask

    def extract_features(self, encoder_out):
        # Prepare decoder input IDs based on the target length
        batch_size, _, _ = encoder_out['encoder_out'].shape
        max_target_length = self.max_target_length # Assuming target length equals encoder sequence length
        device = encoder_out['encoder_out'].device

        # Generate placeholder input IDs (e.g., use BOS token)
        decoder_input_ids = torch.arange(max_target_length, device=device).unsqueeze(0).expand(batch_size, -1)
        print(f"decoder_input_ids: {decoder_input_ids.shape}")

        # Embed tokens and get padding mask
        x, decoder_padding_mask = self.forward_embedding(decoder_input_ids)

        for layer in self.layers:
            x, attn = layer(  
                x,
                encoder_out=encoder_out["encoder_out"],
                encoder_padding_mask=encoder_out.get("encoder_padding_mask", None),
                self_attn_mask=None,
                self_attn_padding_mask=decoder_padding_mask,
                need_attn=False,
                need_head_weights=False,
            )
        # Final linear projection for output
        decoder_output = self.output_projection(x)

        return {
            "x": decoder_output,
            "attn": attn,
            "learnable_query": None,  # Update as per your implementation
            "linear_layer": self.output_projection.weight,
        }

    def forward(self, encoder_out):
        # Extract features from the decoder
        res = self.extract_features(encoder_out)

        # Return log softmax of the predicted sequence
        return {
            "real_res": F.log_softmax(res["x"], dim=-1),
            "learnable_query": res["learnable_query"],
            "linear_layer": res["linear_layer"],
        }

    @staticmethod
    def base_architecture(args):
        # Set default architecture configurations

        args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
        args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
        args.decoder_layers = getattr(args, "decoder_layers", 6)
        args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
        args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
        args.decoder_learned_pos = getattr(args, "decoder_learned_pos", True)
        args.attention_dropout = getattr(args, "attention_dropout", 0.1)
        args.activation_dropout = getattr(args, "activation_dropout", 0.1)
        args.activation_fn = getattr(args, "activation_fn", "relu")
        args.dropout = getattr(args, "dropout", 0.1)
        args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
        args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
        args.share_decoder_input_output_embed = getattr(args, "share_decoder_input_output_embed", False)
        args.max_target_length = getattr(args, "max_target_length", 10)
        args.decoder_num_heads = getattr(args, "decoder_num_heads", 64)
        args.cross_self_attention = getattr(args, "cross_self_attention", False)
        args.no_scale_embedding = getattr(args, "no_scale_embedding", False)