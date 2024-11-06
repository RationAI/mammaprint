import torch
from torch import nn
import torch.optim as optim
import logging
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AttMILModel(nn.Module):
    def __init__(self, feature_dim=512, num_heads=8, dropout=0.1):
        super(AttMILModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Layer normalization for input stability
        self.norm = nn.LayerNorm(feature_dim)
        
        # Learnable query vector
        self.query = nn.Parameter(torch.randn(1, 1, feature_dim))
        nn.init.xavier_uniform_(self.query)
        
        # Multihead Attention layer
        self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
        
        # Initialize classifier weights
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        
    def forward(self, x):
        """
        Forward pass of the AttMILModel.

        Args:
            x (Tensor): Input tensor of shape [batch_size, num_tiles, feature_dim]

        Returns:
            output (Tensor): Predictions of shape [batch_size, 1]
            attention_weights (Tensor): Attention weights of shape [batch_size, num_tiles]
        """
        self.logger.debug(f'Input shape: {x.shape}')  # [B, N, C]
        
        # Extract dimensions
        batch_size, num_tiles, feature_dim = x.shape
        
        # Normalize input features
        x = self.norm(x)  # [B, N, C]
        self.logger.debug(f'Normalized input shape: {x.shape}')
        
        # Expand the query to match the batch size
        query = self.query.expand(batch_size, -1, -1)  # [B, 1, C]
        self.logger.debug(f'Expanded query shape: {query.shape}')
        
        # Apply Multihead Attention with the learnable query
        attn_output, attn_weights = self.attention(query=query, key=x, value=x)  # attn_weights: [B, 1, N]
        self.logger.debug(f'Attention weights shape: {attn_weights.shape}')
        
        # Squeeze to remove the sequence dimension
        attn_weights = attn_weights.squeeze(1)  # [B, N]
        self.logger.debug(f'Squeezed attention weights shape: {attn_weights.shape}')
        
        # Weighted sum of features
        weights = attn_weights.unsqueeze(-1)  # [B, N, 1]
        weighted_features = torch.sum(x * weights, dim=1)  # [B, C]
        self.logger.debug(f'Weighted features shape: {weighted_features.shape}')
        
        # Classification
        output = self.classifier(weighted_features)  # [B, 1]
        self.logger.debug(f'Output shape: {output.shape}')
        
        return output, attn_weights