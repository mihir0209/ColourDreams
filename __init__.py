"""
Image Colorization using Transfer Learning VGG16 Model

A deep learning-based image colorization system that transforms grayscale images 
into vibrant colored images using transfer learning with VGG16 and a custom CNN architecture.
"""

__version__ = "1.0.0"
__author__ = "MiHiR"
__email__ = "hahaha@hahaha.com"
__description__ = "AI-powered image colorization using VGG16 and custom CNN"

from .models.colorization_model import ImageColorizationModel, create_model
from .data.preprocessing import ColorDataset, create_data_loaders

__all__ = [
    "ImageColorizationModel",
    "create_model", 
    "ColorDataset",
    "create_data_loaders",
]