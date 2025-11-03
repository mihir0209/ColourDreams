"""
Clean inference pipeline for image colorization.
Handles image preprocessing, model inference, and postprocessing.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage import color

from models.colorization_model import create_model


class ColorizationInference:
    """Main inference class for image colorization."""
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """Initialize the inference pipeline.
        
        Args:
            model_path: Optional path to model weights. If None, uses default.
            device: Device to run on ('cuda', 'cpu', or None for auto-detect)
        """
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Create and load model
        self.model, _ = create_model(pretrained=True, device=self.device)
        self.model.eval()
        
        print(f"Inference ready on {self.device}")
    
    def _load_image(self, image_input: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """Load image from various input types."""
        if isinstance(image_input, str):
            img = np.asarray(Image.open(image_input))
        elif isinstance(image_input, Image.Image):
            img = np.asarray(image_input.convert('RGB'))
        else:
            img = np.asarray(image_input)
        
        # Ensure RGB
        if img.ndim == 2:
            img = np.tile(img[:, :, None], 3)
        
        return img
    
    def _resize_image(self, img: np.ndarray, target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
        """Resize image to target size."""
        pil_img = Image.fromarray(img)
        resized = pil_img.resize((target_size[1], target_size[0]), resample=Image.Resampling.LANCZOS)
        return np.asarray(resized)
    
    def _preprocess(self, img_rgb: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """Preprocess RGB image to L channel tensors."""
        # Get original and resized versions
        img_rgb_resized = self._resize_image(img_rgb, (256, 256))
        
        # Convert to LAB
        img_lab_orig = color.rgb2lab(img_rgb)
        img_lab_resized = color.rgb2lab(img_rgb_resized)
        
        # Extract L channels
        img_l_orig = img_lab_orig[:, :, 0]
        img_l_resized = img_lab_resized[:, :, 0]
        
        # Convert to tensors
        tens_orig_l = torch.Tensor(img_l_orig)[None, None, :, :]
        tens_resized_l = torch.Tensor(img_l_resized)[None, None, :, :]
        
        return tens_orig_l, tens_resized_l, img_rgb
    
    def _postprocess(self, tens_orig_l: torch.Tensor, out_ab: torch.Tensor) -> np.ndarray:
        """Combine L and predicted AB channels to create RGB image."""
        HW_orig = tens_orig_l.shape[2:]
        HW = out_ab.shape[2:]
        
        # Resize AB to match original if needed
        if HW_orig[0] != HW[0] or HW_orig[1] != HW[1]:
            out_ab_orig = F.interpolate(out_ab, size=HW_orig, mode='bilinear', align_corners=False)
        else:
            out_ab_orig = out_ab
        
        # Combine L and AB
        out_lab_orig = torch.cat((tens_orig_l, out_ab_orig), dim=1)
        
        # Convert to RGB
        rgb_output = color.lab2rgb(out_lab_orig.data.cpu().numpy()[0, ...].transpose((1, 2, 0)))
        
        return rgb_output
    
    def colorize_image(self, image: Union[str, Image.Image, np.ndarray]) -> Tuple[Optional[Image.Image], Optional[str]]:
        """Colorize an image.
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
        
        Returns:
            Tuple of (colorized PIL Image, error message)
        """
        try:
            # Load and preprocess
            img_rgb = self._load_image(image)
            tens_orig_l, tens_resized_l, _ = self._preprocess(img_rgb)
            
            # Move to device
            tens_orig_l = tens_orig_l.to(self.device)
            tens_resized_l = tens_resized_l.to(self.device)
            
            # Inference
            with torch.no_grad():
                out_ab = self.model(tens_resized_l)
            
            # Postprocess
            rgb_output = self._postprocess(tens_orig_l, out_ab)
            rgb_output = np.clip(rgb_output, 0.0, 1.0)
            
            # Convert to PIL
            colorized_img = Image.fromarray((rgb_output * 255).astype(np.uint8))
            
            return colorized_img, None
            
        except Exception as e:
            return None, str(e)
