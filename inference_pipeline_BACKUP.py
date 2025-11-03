"""
Simple inference pipeline for VGG16-based colorization
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from skimage import color
from models.colorization_model import create_model


class ColorizationInference:
    """Simple inference wrapper for VGG16 colorization"""
    
    def __init__(self, device=None):
        """Initialize the colorization model"""
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Initializing VGG16 colorization model on {self.device}...")
        self.model = create_model(device=self.device)
        print("Model ready!")
        
    def _preprocess(self, image):
        """Convert RGB image to LAB and prepare L channel for model"""
        # Convert to numpy array
        img_np = np.array(image)
        
        # Convert RGB to LAB
        img_lab = color.rgb2lab(img_np).astype(np.float32)
        
        # Extract L channel (lightness)
        L = img_lab[:, :, 0:1]  # Shape: [H, W, 1]
        
        # Normalize L to [0, 1] range (L is in [0, 100])
        L = L / 100.0
        
        # Convert to tensor and add batch dimension
        L_tensor = torch.from_numpy(L.transpose(2, 0, 1)).unsqueeze(0)  # [1, 1, H, W]
        
        # Resize to 256x256 for processing
        original_size = L_tensor.shape[2:]
        L_resized = F.interpolate(L_tensor, size=(256, 256), mode='bilinear', align_corners=True)
        
        return L_resized.to(self.device), original_size, img_lab[:, :, 0]
    
    def _postprocess(self, L_original, AB_pred, original_size):
        """Combine L and predicted AB channels to create RGB image"""
        # Resize AB to original size
        AB_resized = F.interpolate(AB_pred, size=original_size, mode='bilinear', align_corners=True)
        AB_resized = AB_resized.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # [H, W, 2]
        
        # Scale AB from [-1, 1] to [-128, 127]
        AB_resized = AB_resized * 110.0
        
        # Combine L and AB
        L_for_combine = L_original[:, :, np.newaxis]  # [H, W, 1]
        LAB_combined = np.concatenate([L_for_combine, AB_resized], axis=2)  # [H, W, 3]
        
        # Convert LAB to RGB
        RGB = color.lab2rgb(LAB_combined)
        
        # Convert to uint8 and create PIL image
        RGB_uint8 = (np.clip(RGB, 0, 1) * 255).astype(np.uint8)
        
        return Image.fromarray(RGB_uint8)
    
    def colorize_image(self, image):
        """Colorize a grayscale or color image"""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Preprocess
        L_input, original_size, L_original = self._preprocess(image)
        
        # Run inference
        with torch.no_grad():
            AB_pred = self.model(L_input)
        
        # Postprocess
        colorized = self._postprocess(L_original, AB_pred, original_size)
        
        return colorized
