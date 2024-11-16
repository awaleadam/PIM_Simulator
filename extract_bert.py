import torch
from transformers import BertModel, BertConfig

# Initialize BERT configuration and model
config = BertConfig()
model = BertModel(config)

file = open("bert_configs.txt","w")

# Example input batch (batch size B, sequence length L, hidden size D)
B, L = 1, 128  # Batch size of 1 and a sequence length of 128 tokens
input_ids = torch.randint(0, config.vocab_size, (B, L))  # Random token IDs for testing

# Forward pass to obtain hidden states
outputs = model(input_ids=input_ids)
hidden_states = outputs.last_hidden_state  # Shape: (B, L, D)

print("=== BERT Layer Dimensions and Operations ===")

# 1. Token Embedding Layer
embedding_dim = config.hidden_size
token_embeddings = model.embeddings.word_embeddings
print(f"Token Embedding Layer:")
print(f"  Input shape: (B, L) = ({B}, {L})")
print(f"  Token embedding lookup: (B, L) -> (B, L, {embedding_dim})\n")

# 2. Position Embedding Layer
position_embedding_dim = model.embeddings.position_embeddings.weight.shape
print(f"Position Embedding:")
print(f"  Position embedding tensor shape: {position_embedding_dim}")
print(f"  Position embeddings are added element-wise to token embeddings: (B, L, {embedding_dim}) + (L, {embedding_dim})\n")

file.write(f"ElementADD:{L},{embedding_dim}\n")

# 3. Transformer Encoder Layers
for i, encoder_layer in enumerate(model.encoder.layer):
    print(f"--- Encoder Layer {i+1} ---")

    # Multi-Head Self-Attention
    attention = encoder_layer.attention.self
    num_heads = attention.num_attention_heads
    head_dim = attention.attention_head_size

    print(f"Multi-Head Self-Attention:")
    # Queries, Keys, and Values projection for each head
    print(f"  Query, Key, Value projection (MatMul): (B, L, {embedding_dim}) * ({embedding_dim}, {num_heads * head_dim})")
    print(f"  Output shape (Q, K, V): (B, {num_heads}, L, {head_dim})")

    file.write(f"MatMul:{L},{embedding_dim},{embedding_dim},{num_heads * head_dim}\n")
    file.write(f"MatMul:{L},{embedding_dim},{embedding_dim},{num_heads * head_dim}\n")
    file.write(f"MatMul:{L},{embedding_dim},{embedding_dim},{num_heads * head_dim}\n")

    # Attention Scores calculation (Q x K^T)
    print(f"  Attention score calculation (MatMul): (B, {num_heads}, L, {head_dim}) * (B, {num_heads}, {head_dim}, L)")
    print(f"  Attention scores shape: (B, {num_heads}, L, L)")

    file.write(f"MatMul:{L},{head_dim*num_heads},{head_dim*num_heads},{L}\n")

    # Attention-weighted values (scores x V)
    print(f"  Attention-weighted value calculation (MatMul): (B, {num_heads}, L, L) * (B, {num_heads}, L, {head_dim})")
    print(f"  Multi-Head Output shape after concatenation: (B, L, {embedding_dim})\n")

    file.write(f"MatMul:{head_dim*num_heads},{L},{L},{head_dim*num_heads}\n")

    # Feed-Forward Network
    ffn = encoder_layer.intermediate
    expansion_factor = ffn.dense.weight.shape[0] // embedding_dim

    print(f"Feed-Forward Network:")
    print(f"  First linear layer (MatMul): (B, L, {embedding_dim}) * ({embedding_dim}, {embedding_dim * expansion_factor})")
    print(f"  Output shape: (B, L, {embedding_dim * expansion_factor})")

    file.write(f"MatMul:{L},{embedding_dim},{embedding_dim},{embedding_dim * expansion_factor}\n")

    print(f"  Second linear layer (MatMul): (B, L, {embedding_dim * expansion_factor}) * ({embedding_dim * expansion_factor}, {embedding_dim})")
    print(f"  Output shape: (B, L, {embedding_dim})\n")

    file.write(f"MatMul:{L},{embedding_dim * expansion_factor},{embedding_dim * expansion_factor},{embedding_dim}\n")

# 4. Pooling (for BERT’s classification tasks, if [CLS] token is used)
print("Pooling Layer (CLS Token):")
cls_token_dim = hidden_states[:, 0].shape  # Shape of [CLS] token
print(f"  Final [CLS] Token Representation shape: {cls_token_dim}\n")
