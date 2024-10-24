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
        self.attention = GatedAttention(512)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        self.logger.debug(f'Original input to classifier: {x.shape}')
        batch_size, num_tiles, features = x.shape
        
        # Flatten to process each tile individually
        x_flat = x.view(batch_size * num_tiles, features)  # Shape: [batch_size * num_tiles, features]
        
        # Get attention weights
        attention_weights = self.attention(x_flat)  # Shape: [batch_size * num_tiles, 1]
        attention_weights = attention_weights.view(batch_size, num_tiles)  # Shape: [batch_size, num_tiles]
        self.logger.debug(f'Attention weights shape: {attention_weights.shape}')
        
        # Normalize attention weights
        attention_weights = torch.softmax(attention_weights, dim=1)  # Shape: [batch_size, num_tiles]
        
        # Reshape weights to apply to each feature vector
        weights = attention_weights.view(batch_size, num_tiles, 1)  # Shape: [batch_size, num_tiles, 1]
        self.logger.debug(f'Attention weights after unsqueeze: {weights.shape}')
        
        # Weighted sum of features across the tiles
        weighted_features = torch.sum(x * weights, dim=1)  # Shape: [batch_size, features]
        self.logger.debug(f'Weighted features shape: {weighted_features.shape}')
        
        # Final classification
        output = self.classifier(weighted_features)  # Shape: [batch_size, 1]
        self.logger.debug(f'Bag-level predictions: {output.shape}')
        self.logger.debug(f'Prediction: {output}')
        return output
