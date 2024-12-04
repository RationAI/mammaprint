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
    def __init__(self, feature_dim, attention_dim=2048):
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
    def __init__(self, feature_dim=2048, classifier_dims= [1024, 512, 256, 1], dropout=0.1):
        super(AttMILModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        # Layer normalization for input stability
        self.norm = nn.LayerNorm(feature_dim)
        self.attention = GatedAttention(feature_dim)
        # Build classifier layers
        layers = []
        input_dim = feature_dim
        for i, dim in enumerate(classifier_dims):
            layers.append(nn.Linear(input_dim, dim))
            if i < len(classifier_dims) - 1:  # Skip activation/dropout for the last layer
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            input_dim = dim
        self.classifier = nn.Sequential(*layers)
        
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
        
        # Normalize attention weights
        attention_weights = torch.softmax(attention_weights, dim=1)  # Shape: [batch_size, num_tiles]
        
        # Reshape weights to apply to each feature vector
        weights = attention_weights.view(batch_size, num_tiles, 1)  # Shape: [batch_size, num_tiles, 1]
        
        # Weighted sum of features across the tiles
        weighted_features = torch.sum(x * weights, dim=1)  # Shape: [batch_size, features]
        
        # Final classification
        output = self.classifier(weighted_features)  # Shape: [batch_size, 1]
        return output, attention_weights
