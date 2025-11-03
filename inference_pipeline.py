"""
Inference pipeline using official richzhang/colorization models
"""

import torch
import numpy as np
from PIL import Image
import sys
from pathlib import Path

# Add colorizers to path
colorizers_path = Path(__file__).parent / "colorizers"
sys.path.insert(0, str(colorizers_path))

from models.colorization_model import create_model
import colorizers


class ColorizationInference:
    """Inference wrapper using official pretrained colorization"""
    
    def __init__(self, device=None):
        """
        Initialize the colorization model
        
        Args:
            device: Device to run on (auto-detected if None)
            model_type: 'eccv16' or 'siggraph17'
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Initializing colorization model on {self.device}...")
        self.model = create_model(device=str(self.device))
        print("Model ready!")
        
    def colorize_image(self, image):
        """
        Colorize an image using the official model
        
        Args:
            image: PIL Image (RGB or grayscale)
            
        Returns:
            Colorized PIL Image
        """
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array (always RGB)
        img_np = np.array(image)

        # Use official preprocessing that handles resizing and LAB conversion
        tens_l_orig, tens_l_rs = colorizers.preprocess_img(img_np, HW=(256, 256))
        tens_l_rs = tens_l_rs.to(self.device)

        # Run inference on resized L channel
        with torch.no_grad():
            out_ab = self.model(tens_l_rs)

        # Convert network output back to original resolution
        img_rgb = colorizers.postprocess_tens(tens_l_orig, out_ab.cpu())

        # Convert to uint8 image
        img_rgb_uint8 = (np.clip(img_rgb, 0, 1) * 255).astype(np.uint8)

        return Image.fromarray(img_rgb_uint8)


if __name__ == "__main__":
    print("Testing colorization inference...")
    
    # Create inference object
    inference = ColorizationInference(device='cpu')
    
    # Create test image
    test_img = Image.new('RGB', (256, 256), color=(128, 128, 128))
    
    # Colorize
    result = inference.colorize_image(test_img)
    
    print(f"Input: {test_img.size}")
    print(f"Output: {result.size}")
    print("✓ Inference working!")
