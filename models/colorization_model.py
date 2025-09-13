"""
Model architecture for image colorization using VGG16 + Custom CNN.
VGG16 is used as a frozen feature extractor, followed by a custom CNN for A&B channel prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights
import math

class VGG16FeatureExtractor(nn.Module):
    """VGG16 Feature Extractor (frozen weights)."""
    
    def __init__(self, pretrained=True):
        super(VGG16FeatureExtractor, self).__init__()
        
        # Load pre-trained VGG16
        if pretrained:
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        else:
            vgg = vgg16(weights=None)
        
        # Extract feature layers (conv layers only, excluding classifier)
        self.features = vgg.features
        
        # Freeze all parameters
        for param in self.features.parameters():
            param.requires_grad = False
        
        # Modify first layer to accept single channel (L channel) instead of 3 channels
        # We'll replicate the L channel 3 times to match VGG16's expected input
        
    def forward(self, x):
        # Input x shape: (batch_size, 1, 224, 224) - L channel only
        # Replicate L channel to create 3-channel input for VGG16
        x = x.repeat(1, 3, 1, 1)  # Shape: (batch_size, 3, 224, 224)
        
        # Extract features
        features = self.features(x)  # Shape: (batch_size, 512, 7, 7)
        
        return features

class ColorizationCNN(nn.Module):
    """Custom CNN for predicting A and B channels from VGG16 features."""
    
    def __init__(self, input_channels=512):
        super(ColorizationCNN, self).__init__()
        
        # Decoder network to predict A and B channels
        self.decoder = nn.Sequential(
            # First upsampling block
            nn.ConvTranspose2d(input_channels, 256, kernel_size=4, stride=2, padding=1),  # 7x7 -> 14x14
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Second upsampling block
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 14x14 -> 28x28
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Third upsampling block
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 28x28 -> 56x56
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Fourth upsampling block
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 56x56 -> 112x112
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Fifth upsampling block
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 112x112 -> 224x224
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # Final layer to output A and B channels
            nn.Conv2d(16, 2, kernel_size=3, stride=1, padding=1),  # 224x224, 2 channels (A, B)
            nn.Sigmoid()  # Output range [0, 1] for normalized A and B channels
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Input x shape: (batch_size, 512, 7, 7) - VGG16 features
        ab_channels = self.decoder(x)  # Shape: (batch_size, 2, 224, 224)
        return ab_channels

class ImageColorizationModel(nn.Module):
    """Complete Image Colorization Model combining VGG16 + Custom CNN."""
    
    def __init__(self, pretrained_vgg=True):
        super(ImageColorizationModel, self).__init__()
        
        # VGG16 feature extractor (frozen)
        self.feature_extractor = VGG16FeatureExtractor(pretrained=pretrained_vgg)
        
        # Custom CNN for colorization
        self.colorization_cnn = ColorizationCNN(input_channels=512)
        
    def forward(self, L_channel):
        # Input: L channel (batch_size, 1, 224, 224)
        
        # Extract features using VGG16
        features = self.feature_extractor(L_channel)  # (batch_size, 512, 7, 7)
        
        # Predict A and B channels
        ab_channels = self.colorization_cnn(features)  # (batch_size, 2, 224, 224)
        
        return ab_channels
    
    def predict_full_image(self, L_channel):
        """Predict and return full LAB image."""
        with torch.no_grad():
            ab_channels = self.forward(L_channel)
            
            # Combine L and AB channels
            lab_image = torch.cat([L_channel, ab_channels], dim=1)  # (batch_size, 3, 224, 224)
            
            return lab_image, ab_channels

def create_model(pretrained=True, device='cuda'):
    """Create and initialize the colorization model."""
    model = ImageColorizationModel(pretrained_vgg=pretrained)
    
    # Move to device
    if device == 'cuda' and torch.cuda.is_available():
        model = model.cuda()
        print(f"Model moved to GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("Model running on CPU")
    
    return model, device

def count_parameters(model):
    """Count the number of trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    return total_params, trainable_params

def test_model():
    """Test the model architecture."""
    print("Testing Image Colorization Model...")
    
    # Create model
    model, device = create_model(pretrained=True, device='cpu')  # Use CPU for testing
    
    # Count parameters
    count_parameters(model)
    
    # Test with dummy input
    batch_size = 4
    L_input = torch.randn(batch_size, 1, 224, 224)
    
    print(f"\nInput L channel shape: {L_input.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        ab_output = model(L_input)
        lab_output, _ = model.predict_full_image(L_input)
    
    print(f"Output AB channels shape: {ab_output.shape}")
    print(f"Output LAB image shape: {lab_output.shape}")
    print(f"AB channels range: [{ab_output.min():.3f}, {ab_output.max():.3f}]")
    
    # Test feature extractor separately
    features = model.feature_extractor(L_input)
    print(f"VGG16 features shape: {features.shape}")
    
    print("✓ Model architecture test completed successfully!")
    
    return model

if __name__ == "__main__":
    # Test the model
    model = test_model()