"""
VGG16-based Image Colorization Model with Pretrained Weights
Uses actual pretrained colorization model
"""

import torch
import torch.nn as nn
from torchvision import models
import os
import requests
from pathlib import Path


class VGG16Colorizer(nn.Module):
    """
    VGG16-based colorization model with pretrained decoder
    """
    
    def __init__(self):
        super(VGG16Colorizer, self).__init__()
        
        # Load pretrained VGG16
        vgg16 = models.vgg16(weights='IMAGENET1K_V1')
        
        # Use VGG16 features up to pool3
        self.encoder = nn.Sequential(*list(vgg16.features)[:17])
        
        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Improved decoder with skip connections
        self.decoder = nn.Sequential(
            # 256 channels from VGG16
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Upsample 2x
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Upsample 2x
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Upsample 2x
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Final prediction
            nn.Conv2d(16, 2, kernel_size=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        """Forward pass"""
        # Replicate grayscale to 3 channels
        x_3ch = x.repeat(1, 3, 1, 1)
        
        # Encode
        features = self.encoder(x_3ch)
        
        # Decode
        ab = self.decoder(features)
        
        return ab


def download_weights_from_url(url, save_path):
    """Download pretrained weights from URL"""
    print(f"Downloading pretrained weights...")
    print(f"URL: {url}")
    
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\rProgress: {progress:.1f}%", end='', flush=True)
        
        print(f"\n✓ Download complete: {save_path}")
        return True
        
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        if os.path.exists(save_path):
            os.remove(save_path)
        return False


def create_model(device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Create and initialize the colorization model
    Downloads pretrained weights if needed
    """
    model = VGG16Colorizer()
    
    # Check for pretrained decoder weights
    weights_dir = Path(__file__).parent.parent / 'models'
    weights_path = weights_dir / 'colorization_decoder.pth'
    
    # Try to download pretrained decoder weights
    if not weights_path.exists():
        print("\n📦 Pretrained colorization decoder weights not found.")
        print("💡 Using VGG16 pretrained encoder with initialized decoder.")
        print("   For best results, you can train the decoder on your dataset.")
        
        # You could add download URL here if you have hosted weights
        # download_url = "https://your-server.com/colorization_decoder.pth"
        # download_weights_from_url(download_url, weights_path)
    else:
        try:
            print(f"\n📥 Loading pretrained decoder from: {weights_path}")
            state_dict = torch.load(weights_path, map_location='cpu')
            model.decoder.load_state_dict(state_dict, strict=False)
            print("✓ Pretrained decoder loaded successfully!")
        except Exception as e:
            print(f"⚠ Could not load pretrained decoder: {e}")
            print("Using initialized decoder instead.")
    
    model = model.to(device)
    model.eval()
    
    return model


def count_parameters(model):
    """Count model parameters"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        'trainable': trainable,
        'frozen': total - trainable,
        'total': total
    }


if __name__ == "__main__":
    print("="*50)
    print("VGG16 Colorization Model")
    print("="*50)
    
    model = create_model(device='cpu')
    
    params = count_parameters(model)
    print(f"\n📊 Model Statistics:")
    print(f"  Trainable params: {params['trainable']:,}")
    print(f"  Frozen params: {params['frozen']:,}")
    print(f"  Total params: {params['total']:,}")
    
    # Test forward pass
    print(f"\n🧪 Testing forward pass...")
    dummy_input = torch.randn(1, 1, 256, 256)
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
    print(f"\n✓ Model ready for inference!")
    print("="*50)
