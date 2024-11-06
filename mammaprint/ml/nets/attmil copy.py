import torch
from torch import nn
import torch.optim as optim
import logging
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, C // num_heads]
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * (1.0 / (k.shape[-1] ** 0.5))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class GatedAttention(nn.Module):
    def __init__(self, feature_dim, attention_dim=256):
        super(GatedAttention, self).__init__()
        self.attention_V = nn.Linear(feature_dim, attention_dim)
        self.attention_U = nn.Linear(feature_dim, attention_dim)
        self.attention_weights = nn.Linear(attention_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        A_V = self.attention_V(x)
        A_U = self.attention_U(x)
        A = torch.tanh(A_V) * self.sigmoid(A_U)
        A = self.attention_weights(A)
        return A

class AttMILModel2(nn.Module):
    def __init__(self, feature_dim=512, num_heads=8, num_layers=2):
        super(AttMILModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(feature_dim, num_heads) for _ in range(num_layers)
        ])

        self.attention = GatedAttention(feature_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        self.logger.debug(f'Original input to classifier: {x.shape}')
        batch_size, num_tiles, features = x.shape

        for layer in self.transformer_layers:
            x = layer(x)

        x_flat = x.view(batch_size * num_tiles, features)
        attention_weights = self.attention(x_flat)
        attention_weights = attention_weights.view(batch_size, num_tiles)
        self.logger.debug(f'Attention weights shape: {attention_weights.shape}')

        attention_weights = torch.softmax(attention_weights, dim=1)
        weights = attention_weights.view(batch_size, num_tiles, 1)
        self.logger.debug(f'Attention weights after unsqueeze: {weights.shape}')

        weighted_features = torch.sum(x * weights, dim=1)
        self.logger.debug(f'Weighted features shape: {weighted_features.shape}')

        output = self.classifier(weighted_features)
        self.logger.debug(f'Bag-level predictions: {output.shape}')
        self.logger.debug(f'Prediction: {output}')
        return output, attention_weights
