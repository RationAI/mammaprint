# Inspired by: https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py 
# Attention-based Deep Multiple Instance Learning
# by Maximilian Ilse (ilse.maximilian@gmail.com), Jakub M. Tomczak (jakubmkt@gmail.com) and Max Welling
# Overview
# PyTorch implementation of paper "Attention-based Deep Multiple Instance Learning":
# Ilse, M., Tomczak, J. M., & Welling, M. (2018). Attention-based Deep Multiple Instance Learning. arXiv preprint arXiv:1802.04712.

import torch
from torch import nn
import logging
import torchvision.models as models

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
  
class GatedAttention(nn.Module):
    def __init__(self, feature_dim, attention_dim=512):
        super(GatedAttention, self).__init__()
        self.attention_V = nn.Linear(feature_dim, attention_dim)
        self.attention_U = nn.Linear(feature_dim, attention_dim)
        self.attention_weights = nn.Linear(attention_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: [batch_size * num_tiles, features]
        A_V = self.attention_V(x)  # Shape: [batch_size * num_tiles, attention_dim]
        A_U = self.attention_U(x)  # Shape: [batch_size * num_tiles, attention_dim]
        A = torch.tanh(A_V) * self.sigmoid(A_U)  # Shape: [batch_size * num_tiles, attention_dim]
        A = self.attention_weights(A)  # Shape: [batch_size * num_tiles, 1]
        return A

class AttMILModel(nn.Module):
    def __init__(self):
        super(AttMILModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        # Layer normalization for input stability
        self.norm = nn.LayerNorm(512)
        self.attention = GatedAttention(512)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
        
        # Initialize classifier weights
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        # self.logger.debug(f'Original input to classifier: {x.shape}')
        batch_size, num_tiles, features = x.shape

        # Layer normalization for input stability
        x = self.norm(x)  # [B, N, C]
        
        # Flatten to process each tile individually
        x_flat = x.view(batch_size * num_tiles, features)  # Shape: [batch_size * num_tiles, features]
        
        # Get attention weights
        attention_weights = self.attention(x_flat)  # Shape: [batch_size * num_tiles, 1]
        attention_weights = attention_weights.view(batch_size, num_tiles)  # Shape: [batch_size, num_tiles]
        # self.logger.debug(f'Attention weights shape: {attention_weights.shape}')
        
        # Normalize attention weights
        attention_weights = torch.softmax(attention_weights, dim=1)  # Shape: [batch_size, num_tiles]
        
        # Reshape weights to apply to each feature vector
        weights = attention_weights.view(batch_size, num_tiles, 1)  # Shape: [batch_size, num_tiles, 1]
        # self.logger.debug(f'Attention weights after unsqueeze: {weights.shape}')
        
        # Weighted sum of features across the tiles
        weighted_features = torch.sum(x * weights, dim=1)  # Shape: [batch_size, features]
        # self.logger.debug(f'Weighted features shape: {weighted_features.shape}')
        
        # Final classification
        output = self.classifier(weighted_features)  # Shape: [batch_size, 1]
        # self.logger.debug(f'Bag-level predictions: {output.shape}')
        # self.logger.debug(f'Prediction: {output}')
        return output, attention_weights

# import torch
# from torch import nn
# import torch.optim as optim
# import logging
# from torch.optim.lr_scheduler import ReduceLROnPlateau

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# class AttMILModel(nn.Module):
#     def __init__(self, feature_dim=512, num_heads=8, dropout=0.1):
#         super(AttMILModel, self).__init__()
#         self.logger = logging.getLogger(self.__class__.__name__)
        
#         # Layer normalization for input stability
#         self.norm = nn.LayerNorm(feature_dim)
        
#         # Learnable query vector
#         self.query = nn.Parameter(torch.randn(1, 1, feature_dim))
#         nn.init.xavier_uniform_(self.query)
        
#         # Multihead Attention layer
#         self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        
#         # Classifier
#         self.classifier = nn.Sequential(
#             nn.Linear(feature_dim, 256),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(256, 1)
#         )
        
#         # Initialize classifier weights
#         for layer in self.classifier:
#             if isinstance(layer, nn.Linear):
#                 nn.init.xavier_uniform_(layer.weight)
#                 if layer.bias is not None:
#                     nn.init.zeros_(layer.bias)
        
#     def forward(self, x):
#         """
#         Forward pass of the AttMILModel.

#         Args:
#             x (Tensor): Input tensor of shape [batch_size, num_tiles, feature_dim]

#         Returns:
#             output (Tensor): Predictions of shape [batch_size, 1]
#             attention_weights (Tensor): Attention weights of shape [batch_size, num_tiles]
#         """
#         self.logger.debug(f'Input shape: {x.shape}')  # [B, N, C]
        
#         # Extract dimensions
#         batch_size, num_tiles, feature_dim = x.shape
        
#         # Normalize input features
#         x = self.norm(x)  # [B, N, C]
#         self.logger.debug(f'Normalized input shape: {x.shape}')
        
#         # Expand the query to match the batch size
#         query = self.query.expand(batch_size, -1, -1)  # [B, 1, C]
#         self.logger.debug(f'Expanded query shape: {query.shape}')
        
#         # Apply Multihead Attention with the learnable query
#         attn_output, attn_weights = self.attention(query=query, key=x, value=x)  # attn_weights: [B, 1, N]
#         self.logger.debug(f'Attention weights shape: {attn_weights.shape}')
        
#         # Squeeze to remove the sequence dimension
#         attn_weights = attn_weights.squeeze(1)  # [B, N]
#         self.logger.debug(f'Squeezed attention weights shape: {attn_weights.shape}')
        
#         # Weighted sum of features
#         weights = attn_weights.unsqueeze(-1)  # [B, N, 1]
#         weighted_features = torch.sum(x * weights, dim=1)  # [B, C]
#         self.logger.debug(f'Weighted features shape: {weighted_features.shape}')
        
#         # Classification
#         output = self.classifier(weighted_features)  # [B, 1]
#         self.logger.debug(f'Output shape: {output.shape}')
        
#         return output, attn_weights
