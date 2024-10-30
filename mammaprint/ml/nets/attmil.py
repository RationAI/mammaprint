import torch
from torch import nn
import logging
import torchvision.models as models

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
  
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.concat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channel, reduction)
        self.sa = SpatialAttention(kernel_size)

    def get_channel_attention_conv(self):
        return self.ca.fc[-1]

    def get_spatial_attention_conv(self):
        return self.sa.conv

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x
    
class FullyConnectedAttention(nn.Module):
    def __init__(self, feature_dim, reduction=8):
        super(FullyConnectedAttention, self).__init__()
        self.fc_attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // reduction),
            nn.ReLU(),
            nn.Linear(feature_dim // reduction, feature_dim),
            nn.Sigmoid()
        )
        self.fc_scoring = nn.Sequential(
            nn.Linear(feature_dim, 1),
            nn.Tanh()  # Ensures the output is between -1 and 1, which can be useful for scoring
        )

    def forward(self, x):
        attention_weights = self.fc_attention(x)
        x = x * attention_weights
        scores = self.fc_scoring(x)
        return x, scores

class GatedAttention(nn.Module):
    def __init__(self, feature_dim, attention_dim=256):
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
        self.classifier = nn.Linear(512, 1)

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
