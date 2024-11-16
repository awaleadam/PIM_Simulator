
'''
# Import necessary libraries
import torch
from timm import create_model  # Using `timm` for easy ViT access

# Load a pretrained Vision Transformer (ViT) model
model = create_model("vit_base_patch16_224", pretrained=True)

# Define a hook to capture layer input and output sizes
layer_info = []

def hook(module, input, output):
    layer_info.append({
        "layer": module.__class__.__name__,
        "input_shape": tuple(input[0].shape),
        "output_shape": tuple(output.shape),
        "operation": type(module).__name__
    })

# Register hooks for each layer
for layer in model.modules():
    layer.register_forward_hook(hook)

# Create a random tensor representing a batch of images (e.g., batch size 1, 3 channels, 224x224 pixels)
sample_input = torch.randn(1, 3, 224, 224)

# Pass the input through the model to collect layer information
model.eval()
with torch.no_grad():
    model(sample_input)

# Display collected layer information
for idx, info in enumerate(layer_info):
    print(f"Layer {idx + 1}:")
    print(f"    Operation: {info['operation']}")
    print(f"    Input Shape: {info['input_shape']}")
    print(f"    Output Shape: {info['output_shape']}")
    print()
'''



import torch
from transformers import ViTModel, ViTConfig


def getConvOperations(height, width, kernel_height, kernel_width, stride, padding, input_channels, output_channels):
    flatten = kernel_height*kernel_width*input_channels
    h = ((height+2*padding-kernel_height)/(stride)) + 1 
    w = ((width+2*padding-kernel_width)/(stride)) + 1 
    
    patches = int(h*w) 

    m1 = [output_channels, flatten]
    m2 = [flatten, patches]

    #m1 = [patches, flatten]
    #m2 = [flatten, output_channels]

    return (m1, m2)

file = open("vit_configs.txt","w")


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

# 1. Patch Embedding Layer with Convolution Dimensions
patch_embedding = model.embeddings.patch_embeddings
num_patches = (config.image_size // config.patch_size) ** 2
embedding_dim = config.hidden_size

print(f"Patch Embedding (Conv2D with kernel size {config.patch_size} and stride {config.patch_size}):")
print(f"  Input shape: ({B}, {C}, {H}, {W})")
print(f"  Convolution operation to extract patches: Conv2D({C} channels, {embedding_dim} channels, kernel_size={config.patch_size}, stride={config.patch_size})")
print(f"  Output after convolution (patches): ({B}, {embedding_dim}, {H // config.patch_size}, {W // config.patch_size})")
stride = 16
kernel_height = 16
kernel_width = 16
input_channels = 3
padding = 0 
input_height = 224
input_width = 224
kernels = 768

conv_out = getConvOperations(input_height, input_width, kernel_height, kernel_width, stride, padding, input_channels, kernels)


file.write(f"MatMul:{conv_out[0][0]},{conv_out[0][1]},{conv_out[1][0]},{conv_out[1][1]}\n")

# Flatten to patches and project to embedding dimension
print(f"  Flattening patches to sequence shape: ({B}, {num_patches}, {config.patch_size**2 * C})")
print(f"  Linear projection (MatMul): ({B}, {num_patches}, {config.patch_size**2 * C}) * ({config.patch_size**2 * C}, {embedding_dim})")
print(f"  Output after projection: ({B}, {num_patches}, {embedding_dim})\n")

file.write(f"MatMul:196,768,768,768\n")

# 2. Position Embedding Layer
position_embedding_dim = model.embeddings.position_embeddings.shape
print(f"Position Embedding:")
print(f"  Position embedding tensor shape: {position_embedding_dim}")
print(f"  Position embeddings are added element-wise to patch embeddings: ({B}, {num_patches}, {embedding_dim}) + ({num_patches}, {embedding_dim})\n")

file.write(f"ElementAdd:196,768,196,768\n")

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

    file.write(f"MatMul:{num_patches},{embedding_dim},{embedding_dim},{num_heads * head_dim}\n")
    file.write(f"MatMul:{num_patches},{embedding_dim},{embedding_dim},{num_heads * head_dim}\n")
    file.write(f"MatMul:{num_patches},{embedding_dim},{embedding_dim},{num_heads * head_dim}\n")

    # Attention Scores calculation (Q x K^T)
    print(f"  Attention score calculation (MatMul): ({B}, {num_heads}, {num_patches}, {head_dim}) * ({B}, {num_heads}, {head_dim}, {num_patches})")
    print(f"  Attention scores shape: ({B}, {num_heads}, {num_patches}, {num_patches})")

    file.write(f"MatMul:{num_patches},{head_dim*num_heads},{head_dim*num_heads},{num_patches}\n")

    # Attention-weighted values (scores x V)
    print(f"  Attention-weighted value calculation (MatMul): ({B}, {num_heads}, {num_patches}, {num_patches}) * ({B}, {num_heads}, {num_patches}, {head_dim})")
    print(f"  Multi-Head Output shape after concatenation: ({B}, {num_patches}, {embedding_dim})\n")

    file.write(f"MatMul:{head_dim*num_heads},{num_patches},{num_patches},{head_dim*num_heads}\n")

    # Feed-Forward Network
    ffn = encoder_layer.intermediate
    expansion_factor = ffn.dense.weight.shape[0] // embedding_dim

    print(f"Feed-Forward Network:")
    print(f"  First linear layer (MatMul): ({B}, {num_patches}, {embedding_dim}) * ({embedding_dim}, {embedding_dim * expansion_factor})")
    print(f"  Output shape: ({B}, {num_patches}, {embedding_dim * expansion_factor})")

    file.write(f"MatMul:{num_patches},{embedding_dim},{embedding_dim},{embedding_dim * expansion_factor}\n")

    print(f"  Second linear layer (MatMul): ({B}, {num_patches}, {embedding_dim * expansion_factor}) * ({embedding_dim * expansion_factor}, {embedding_dim})")
    print(f"  Output shape: ({B}, {num_patches}, {embedding_dim})\n")

    file.write(f"MatMul:{num_patches},{embedding_dim * expansion_factor},{embedding_dim * expansion_factor},{embedding_dim}\n")

# 4. Classification Head
print("Classification Head:")
cls_token_dim = hidden_states[:, 0].shape  # Shape of [CLS] token
num_classes = config.num_labels  # Typically this is set when the model is fine-tuned

print(f"  Final [CLS] Token Representation shape: {cls_token_dim}")
print(f"  Classification head (MatMul): ({cls_token_dim}) * ({embedding_dim}, {num_classes})")
print(f"  Output shape: ({B}, {num_classes})\n")


file.write(f"MatMul:1,768,768,2\n")

file.close()