import torch
import torch.nn as nn
import torch.nn.functional as F

class ARDecodingModel(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_layers, max_nodes, max_token_length):
        super(ARDecodingModel, self).__init__()
        
        # Embedding dimensions
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.max_nodes = max_nodes
        self.max_token_length = max_token_length
        self.head_dim = hidden_dim // num_heads
        
        # Decoder embedding (for positional information)
        self.node_embedding = nn.Embedding(4, hidden_dim)  # Node Type (0, 1, 2, 3)
        self.parent_node_embedding = nn.Embedding(max_nodes, hidden_dim)  # Parent node index embedding
        self.token_length_embedding = nn.Embedding(max_token_length, hidden_dim)  # Token length embedding
        
        # Transformer decoder layers with batch_first=True
        self.decoder_layers = nn.ModuleList(
            [nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True) for _ in range(num_layers)]
        )
        
        # Multihead Attention for Cross Attention with Encoder outputs
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        
        # Linear heads for output prediction
        self.type_head = nn.Linear(hidden_dim, 4)  # Node Type prediction (0 ~ 3)
        self.parent_head = nn.Linear(hidden_dim, max_nodes)  # Parent node index prediction
        self.token_length_head = nn.Linear(hidden_dim, max_token_length)  # Token length prediction (for Node Types 1 and 2)
        
        # 이전에 예측된 node types를 저장하기 위한 리스트
        self.node_types_so_far = None  # 초기화는 forward 함수에서 처리

    def forward(self, encoder_output, decoder_input, node_types_so_far=None, decoder_mask=None):
        """
        encoder_output: (batch_size, max_nodes, hidden_dim) - output from encoder (shape: (batch_size, 4800, 1024))
        decoder_input: (batch_size, current_step, hidden_dim) - the previous predicted sequence inputs (Auto-regressive)
        node_types_so_far: (batch_size, current_step) - previously predicted node types
        """
        batch_size, seq_len, _ = decoder_input.size()
        
        # Initialize node_types_so_far if not provided
        if node_types_so_far is None:
            device = decoder_input.device
            node_types_so_far = torch.zeros(batch_size, 0, dtype=torch.long, device=device)  # 초기에는 빈 텐서
    
        # Apply cross attention: Query comes from decoder, Key and Value come from encoder output
        attn_output, _ = self.cross_attention(
            query = decoder_input,   # (batch_size, current_step, hidden_dim)
            key = encoder_output,    # (batch_size, max_nodes, hidden_dim)
            value = encoder_output   # (batch_size, max_nodes, hidden_dim)
        )    
        
        seq_len = decoder_input.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(decoder_input.device)

        # Pass the output through decoder layers (AR decoding step)
        for decoder_layer in self.decoder_layers:
            attn_output = decoder_layer(attn_output, encoder_output, tgt_mask=tgt_mask)
        
        # Use the final transformer output to predict the three values
        node_type_logits = self.type_head(attn_output)  # (batch_size, seq_len, 4)
        parent_node_logits = self.parent_head(attn_output)  # (batch_size, seq_len, max_nodes)
        token_length_logits = self.token_length_head(attn_output)  # (batch_size, seq_len, max_token_length)
        
        # Compute predictions
        node_type = torch.argmax(node_type_logits, dim=-1)  # (batch_size, seq_len)
        parent_node_index = torch.argmax(parent_node_logits, dim=-1)  # (batch_size, seq_len)
        token_length = torch.argmax(token_length_logits, dim=-1)  # (batch_size, seq_len)
        
        # Update node_types_so_far with the newly predicted node_type
        # node_types_so_far = torch.cat([node_types_so_far, node_type], dim=1)  # (batch_size, current_step)

        # Get parent node types using parent_node_index
        max_index = node_types_so_far.size(1) - 1  # 현재까지의 최대 인덱스
        parent_node_index_clipped = parent_node_index.clamp(min=0, max=max_index)  # 인덱스 범위 조정

        # 부모 노드의 타입을 가져옵니다.
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, seq_len)  # (batch_size, seq_len)
        parent_node_type = node_types_so_far[batch_indices, parent_node_index_clipped]  # (batch_size, seq_len)

        # 부모 노드 타입에 따라 가능한 자식 노드 타입을 정의합니다.
        # 부모 노드 타입별로 가능한 자식 노드 타입을 정의합니다.
        possible_child_types = torch.zeros(batch_size, seq_len, 4, device=decoder_input.device)  # (batch_size, seq_len, num_node_types)

        for i in range(4):
            mask = torch.zeros(4, device=decoder_input.device)
            if i == 0:  # ObjectNode
                mask[[1, 3]] = 1  # 가능: KeyValueNode, ArrayNode
            elif i == 1:  # KeyValueNode
                mask[[ ]] = 1  # 자식 없음
            elif i == 2:  # ValueNode
                mask[[ ]] = 1  # 자식 없음
            elif i == 3:  # ArrayNode
                mask[[1, 2]] = 1  # 가능: KeyValueNode, ValueNode
            # 부모 노드 타입이 i인 곳에 마스크 적용
            indices = (parent_node_type == i)
            possible_child_types[indices] = mask

        # 불가능한 노드 타입의 로짓을 매우 작은 값으로 설정하여 선택되지 않도록 합니다.
        node_type_logits = node_type_logits.masked_fill(possible_child_types == 0, float(-1))
        
        # 마스크 적용 후 다시 노드 타입 예측
        node_type = torch.argmax(node_type_logits, dim=-1)  # (batch_size, seq_len)

        # Token Length는 Node Type이 1 또는 2인 경우에만 유효합니다.
        token_length = torch.where(
            (node_type == 1) | (node_type == 2),
            token_length,  # 해당 노드 타입의 경우 token_length 유지
            torch.zeros_like(token_length)  # 다른 경우는 0으로 설정
        )

        # 필요에 따라 token_length_logits도 마찬가지로 마스킹할 수 있습니다.
        token_length_logits = torch.where()(
            ((node_type.unsqueeze(-1) == 1) | (node_type.unsqueeze(-1) == 2)),
            token_length_logits,
            torch.full_like(token_length_logits, float(-1))
        )
        
        print(f"Node_type_logits: {node_type_logits}")
        print(f"Parent_node_logits: {parent_node_logits}")

        # 최종 반환값
        return [node_type_logits, parent_node_logits, token_length_logits, node_type, parent_node_index, token_length, node_types_so_far]

# Example usage
# if __name__ == "__main__":
    
#     batch_size = 32
#     encoder_output = torch.randn(batch_size, 4800, 1024)  # Example encoder output
#     decoder_input = torch.randn(batch_size, 1, 1024)  # Example decoder input (AR step)

#     model = ARDecodingModel()
#     node_types_so_far = None  # 초기에는 None으로 시작

#     node_type_logits, parent_node_logits, token_length_logits, node_type, parent_node_index, token_length, node_types_so_far = model(
#         encoder_output, decoder_input, node_types_so_far
#     )

#     # 출력 확인
#     print("Node Type Shape:", node_type.shape)  # (batch_size, seq_len)
#     print("Parent Node Index Shape:", parent_node_index.shape)  # (batch_size, seq_len)
#     print("Token Length Shape:", token_length.shape)  # (batch_size, seq_len)
#     print("Node Types So Far Shape:", node_types_so_far.shape)  # (batch_size, current_step)