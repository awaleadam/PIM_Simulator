import torch
from transformers import ViTModel, ViTConfig

# Initialize ViT configuration and model
config = ViTConfig()
model = ViTModel(config)

# Example input batch (batch size B, height H, width W, channels C)
B, H, W, C = 1, config.image_size, config.image_size, 3
input_tensor = torch.randn(B, C, H, W)

# Forward pass to obtain hidden states
outputs = model(pixel_values=input_tensor)
hidden_states = outputs.last_hidden_state  # Shape: (B, L, D)

print("=== Vision Transformer Layer Dimensions and Operations ===")

# 1. Patch Embedding Layer
patch_embedding = model.embeddings.patch_embeddings
num_patches = (config.image_size // config.patch_size) ** 2
embedding_dim = config.hidden_size

print(f"Patch Embedding:")
print(f"  Input shape: ({B}, {H}, {W}, {C})")
print(f"  Flattening operation to patches: ({B}, {num_patches}, {config.patch_size**2 * C})")
print(f"  Linear projection (MatMul): ({B}, {num_patches}, {config.patch_size**2 * C}) * ({config.patch_size**2 * C}, {embedding_dim})")
print(f"  Output after projection: ({B}, {num_patches}, {embedding_dim})\n")

# 2. Position Embedding Layer
position_embedding_dim = model.embeddings.position_embeddings.shape
print(f"Position Embedding:")
print(f"  Position embedding tensor shape: {position_embedding_dim}")
print(f"  Position embeddings are added element-wise to patch embeddings: ({B}, {num_patches}, {embedding_dim}) + ({num_patches}, {embedding_dim})\n")

# 3. Transformer Encoder Layers
for i, encoder_layer in enumerate(model.encoder.layer):
    print(f"--- Encoder Layer {i+1} ---")

    # Multi-Head Self-Attention
    attention = encoder_layer.attention.attention
    num_heads = attention.num_attention_heads
    head_dim = attention.attention_head_size

    # Queries, Keys, Values projection for each head
    print(f"Multi-Head Self-Attention:")
    print(f"  Query, Key, Value projection (MatMul): ({B}, {num_patches}, {embedding_dim}) * ({embedding_dim}, {num_heads * head_dim})")
    print(f"  Output (Q, K, V) shape: ({B}, {num_heads}, {num_patches}, {head_dim})")

    # Attention Scores calculation (Q x K^T)
    print(f"  Attention score calculation (MatMul): ({B}, {num_heads}, {num_patches}, {head_dim}) * ({B}, {num_heads}, {head_dim}, {num_patches})")
    print(f"  Attention scores shape: ({B}, {num_heads}, {num_patches}, {num_patches})")

    # Attention-weighted values (scores x V)
    print(f"  Attention-weighted value calculation (MatMul): ({B}, {num_heads}, {num_patches}, {num_patches}) * ({B}, {num_heads}, {num_patches}, {head_dim})")
    print(f"  Multi-Head Output shape after concatenation: ({B}, {num_patches}, {embedding_dim})\n")

    # Feed-Forward Network
    ffn = encoder_layer.intermediate
    expansion_factor = ffn.dense.weight.shape[0] // embedding_dim

    print(f"Feed-Forward Network:")
    print(f"  First linear layer (MatMul): ({B}, {num_patches}, {embedding_dim}) * ({embedding_dim}, {embedding_dim * expansion_factor})")
    print(f"  Output shape: ({B}, {num_patches}, {embedding_dim * expansion_factor})")

    print(f"  Second linear layer (MatMul): ({B}, {num_patches}, {embedding_dim * expansion_factor}) * ({embedding_dim * expansion_factor}, {embedding_dim})")
    print(f"  Output shape: ({B}, {num_patches}, {embedding_dim})\n")

# 4. Classification Head
print("Classification Head:")
cls_token_dim = hidden_states[:, 0].shape  # Shape of [CLS] token
num_classes = config.num_labels  # Typically this is set when the model is fine-tuned

print(f"  Final [CLS] Token Representation shape: {cls_token_dim}")
print(f"  Classification head (MatMul): ({cls_token_dim}) * ({embedding_dim}, {num_classes})")
print(f"  Output shape: ({B}, {num_classes})\n")
