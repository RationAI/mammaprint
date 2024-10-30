import torch
from torch import nn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GatedAttention(nn.Module):
    def __init__(self, feature_dim, attention_dim=512, temperature=0.5, lambda_reg=1e-3):
        super(GatedAttention, self).__init__()
        self.attention_V = nn.Linear(feature_dim, attention_dim)
        self.attention_U = nn.Linear(feature_dim, attention_dim)
        self.attention_weights = nn.Linear(attention_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.temperature = temperature  # Temperature for scaling softmax
        self.norm = nn.LayerNorm(attention_dim)  # Layer normalization for stability
        self.lambda_reg = lambda_reg  # Regularization term for entropy
        self.attention_entropy = 0  # Placeholder for tracking entropy

    def forward(self, x):
        # x shape: [batch_size * num_tiles, features]
        A_V = self.attention_V(x)  # Shape: [batch_size * num_tiles, attention_dim]
        A_U = self.attention_U(x)  # Shape: [batch_size * num_tiles, attention_dim]
        
        # Compute gated attention scores and apply layer normalization
        A = self.norm(torch.tanh(A_V) * self.sigmoid(A_U))  # Shape: [batch_size * num_tiles, attention_dim]
        
        # Compute unnormalized attention weights and apply temperature scaling
        A = self.attention_weights(A) / self.temperature  # Shape: [batch_size * num_tiles, 1]
        
        # Reshape and normalize attention weights using softmax
        attention_weights = torch.softmax(A, dim=0)  # Shape: [batch_size * num_tiles, 1]
        
        # Calculate entropy for regularization monitoring (optional)
        self.attention_entropy = -torch.sum(attention_weights * torch.log(torch.clamp(attention_weights, min=1e-8))).item()
        
        return attention_weights

class AttMILModel(nn.Module):
    def __init__(self):
        super(AttMILModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.attention = GatedAttention(512, temperature=0.3, lambda_reg=1e-3)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )


    def forward(self, x):
        batch_size, num_tiles, features = x.shape
        
        # Flatten to process each tile individually
        x_flat = x.view(batch_size * num_tiles, features)
        
        # Get attention weights
        attention_weights = self.attention(x_flat)  # Shape: [batch_size * num_tiles, 1]
        attention_weights = attention_weights.view(batch_size, num_tiles)  # Reshape to [batch_size, num_tiles]
        
        # Normalize attention weights across tiles
        attention_weights = torch.softmax(attention_weights, dim=1)  # Shape: [batch_size, num_tiles]
        
        # Reshape weights to apply to each feature vector
        weights = attention_weights.view(batch_size, num_tiles, 1)  # Shape: [batch_size, num_tiles, 1]
        
        # Weighted sum of features across the tiles
        weighted_features = torch.sum(x * weights, dim=1)  # Shape: [batch_size, features]
        
        # Final classification
        output = self.classifier(weighted_features)  # Shape: [batch_size, 1]
        self.logger.debug(f'Prediction: {output}')

        # Log the entropy for analysis (optional)
        self.logger.debug(f'Attention entropy: {self.attention.attention_entropy}')
        
        return output
