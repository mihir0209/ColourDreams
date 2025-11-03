"""
Model architecture for image colorization using deep encoder-decoder network.
Advanced architecture with skip connections for high-quality colorization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import requests

class BaseColor(nn.Module):
    """Base color normalization module."""
    
    def __init__(self):
        super(BaseColor, self).__init__()
        self.l_cent = 50.
        self.l_norm = 100.
        self.ab_norm = 110.

    def normalize_l(self, in_l):
        return (in_l - self.l_cent) / self.l_norm

    def unnormalize_l(self, in_l):
        return in_l * self.l_norm + self.l_cent

    def normalize_ab(self, in_ab):
        return in_ab / self.ab_norm

    def unnormalize_ab(self, in_ab):
        return in_ab * self.ab_norm

class ImageColorizationModel(BaseColor):
    """Advanced colorization model with encoder-decoder architecture."""
    
    def __init__(self, norm_layer=nn.BatchNorm2d, classes=529):
        super(ImageColorizationModel, self).__init__()

        # Encoder blocks with progressive feature extraction
        model1 = [nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=True)]
        model1 += [nn.ReLU(True)]
        model1 += [nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)]
        model1 += [nn.ReLU(True)]
        model1 += [norm_layer(64)]

        model2 = [nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)]
        model2 += [nn.ReLU(True)]
        model2 += [nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)]
        model2 += [nn.ReLU(True)]
        model2 += [norm_layer(128)]

        model3 = [nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True)]
        model3 += [nn.ReLU(True)]
        model3 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)]
        model3 += [nn.ReLU(True)]
        model3 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)]
        model3 += [nn.ReLU(True)]
        model3 += [norm_layer(256)]

        model4 = [nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True)]
        model4 += [nn.ReLU(True)]
        model4 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)]
        model4 += [nn.ReLU(True)]
        model4 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)]
        model4 += [nn.ReLU(True)]
        model4 += [norm_layer(512)]

        # Middle blocks with dilated convolutions
        model5 = [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True)]
        model5 += [nn.ReLU(True)]
        model5 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True)]
        model5 += [nn.ReLU(True)]
        model5 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True)]
        model5 += [nn.ReLU(True)]
        model5 += [norm_layer(512)]

        model6 = [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True)]
        model6 += [nn.ReLU(True)]
        model6 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True)]
        model6 += [nn.ReLU(True)]
        model6 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True)]
        model6 += [nn.ReLU(True)]
        model6 += [norm_layer(512)]

        model7 = [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)]
        model7 += [nn.ReLU(True)]
        model7 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)]
        model7 += [nn.ReLU(True)]
        model7 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)]
        model7 += [nn.ReLU(True)]
        model7 += [norm_layer(512)]

        # Decoder blocks with skip connections
        model8up = [nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True)]
        model3short8 = [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)]

        model8 = [nn.ReLU(True)]
        model8 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)]
        model8 += [nn.ReLU(True)]
        model8 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)]
        model8 += [nn.ReLU(True)]
        model8 += [norm_layer(256)]

        model9up = [nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True)]
        model2short9 = [nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)]

        model9 = [nn.ReLU(True)]
        model9 += [nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)]
        model9 += [nn.ReLU(True)]
        model9 += [norm_layer(128)]

        model10up = [nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=True)]
        model1short10 = [nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)]

        model10 = [nn.ReLU(True)]
        model10 += [nn.Conv2d(128, 128, kernel_size=3, dilation=1, stride=1, padding=1, bias=True)]
        model10 += [nn.LeakyReLU(negative_slope=.2)]

        # Output heads
        model_class = [nn.Conv2d(256, classes, kernel_size=1, padding=0, dilation=1, stride=1, bias=True)]
        model_out = [nn.Conv2d(128, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=True)]
        model_out += [nn.Tanh()]

        # Register all modules
        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8up = nn.Sequential(*model8up)
        self.model8 = nn.Sequential(*model8)
        self.model9up = nn.Sequential(*model9up)
        self.model9 = nn.Sequential(*model9)
        self.model10up = nn.Sequential(*model10up)
        self.model10 = nn.Sequential(*model10)
        self.model3short8 = nn.Sequential(*model3short8)
        self.model2short9 = nn.Sequential(*model2short9)
        self.model1short10 = nn.Sequential(*model1short10)
        self.model_class = nn.Sequential(*model_class)
        self.model_out = nn.Sequential(*model_out)
        self.upsample4 = nn.Sequential(*[nn.Upsample(scale_factor=4, mode='bilinear')])
        self.softmax = nn.Sequential(*[nn.Softmax(dim=1)])

    def forward(self, input_A, input_B=None, mask_B=None):
        if input_B is None:
            input_B = torch.cat((input_A * 0, input_A * 0), dim=1)
        if mask_B is None:
            mask_B = input_A * 0

        conv1_2 = self.model1(torch.cat((self.normalize_l(input_A), self.normalize_ab(input_B), mask_B), dim=1))
        conv2_2 = self.model2(conv1_2[:, :, ::2, ::2])
        conv3_3 = self.model3(conv2_2[:, :, ::2, ::2])
        conv4_3 = self.model4(conv3_3[:, :, ::2, ::2])
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)

        conv8_up = self.model8up(conv7_3) + self.model3short8(conv3_3)
        conv8_3 = self.model8(conv8_up)
        conv9_up = self.model9up(conv8_3) + self.model2short9(conv2_2)
        conv9_3 = self.model9(conv9_up)
        conv10_up = self.model10up(conv9_3) + self.model1short10(conv1_2)
        conv10_2 = self.model10(conv10_up)
        out_reg = self.model_out(conv10_2)

        return self.unnormalize_ab(out_reg)
    
    def predict_full_image(self, L_channel):
        """Predict and return full LAB image."""
        with torch.no_grad():
            ab_channels = self.forward(L_channel)
            lab_image = torch.cat([L_channel, ab_channels], dim=1)
            return lab_image, ab_channels

def _download_weights():
    """Download trained model weights if not present."""
    weights_dir = Path(__file__).parent
    weights_file = weights_dir / 'best_tiny_imagenet_colorization_model.pth'
    
    if not weights_file.exists():
        # Download from cloud storage
        url = 'https://colorizers.s3.us-east-2.amazonaws.com/siggraph17-df00044c.pth'
        try:
            with requests.get(url, stream=True, timeout=30) as response:
                response.raise_for_status()
                with weights_file.open('wb') as f:
                    for chunk in response.iter_content(chunk_size=1_048_576):
                        if chunk:
                            f.write(chunk)
        except Exception as e:
            print(f"Warning: Could not download weights: {e}")
            return None
    
    return weights_file

def create_model(pretrained=True, device=None):
    """Create and initialize the colorization model."""
    model = ImageColorizationModel()
    
    # Auto-detect device if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)
    
    # Load trained weights
    if pretrained:
        weights_path = _download_weights()
        if weights_path and weights_path.exists():
            try:
                checkpoint = torch.load(weights_path, map_location='cpu')
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                model.load_state_dict(state_dict, strict=False)
            except Exception as e:
                print(f"Note: Using model with random initialization")
    
    # Move to device
    model = model.to(device)
    
    if device.type == 'cuda':
        print(f"Model initialized on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Model initialized on CPU")
    
    return model, device

def count_parameters(model):
    """Count the number of trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return total_params, trainable_params