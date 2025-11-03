"""
Legacy inference script - deprecated.
Use inference_pipeline.py for production inference.
"""

import warnings
warnings.warn(
    "This inference.py is deprecated. Use inference_pipeline.ColorizationInference instead.",
    DeprecationWarning,
    stacklevel=2
)

# For backwards compatibility, import from new module
from inference_pipeline import ColorizationInference

__all__ = ['ColorizationInference']

# Import the exact same model architecture used in training
class VGG16FeatureExtractor(nn.Module):
    """VGG16 feature extractor with frozen layers as per project requirements."""
    
    def __init__(self, freeze_layers=True):
        super(VGG16FeatureExtractor, self).__init__()
        
        # Load pretrained VGG16
        vgg16_model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        
        # Modify first conv layer to accept single channel (L channel)
        original_first_layer = vgg16_model.features[0]
        self.first_conv = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        
        # Initialize new layer with averaged RGB weights
        with torch.no_grad():
            # Average the 3-channel weights to 1-channel
            self.first_conv.weight = nn.Parameter(
                original_first_layer.weight.mean(dim=1, keepdim=True)
            )
            self.first_conv.bias = original_first_layer.bias
        
        # Extract VGG16 features (excluding first layer)
        self.vgg_features = nn.Sequential(
            self.first_conv,
            *list(vgg16_model.features.children())[1:]
        )
        
        # Freeze VGG16 layers as per project requirements
        if freeze_layers:
            for param in self.vgg_features.parameters():
                param.requires_grad = False
        
        # Add adaptive pooling to handle different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
    def forward(self, x):
        """Extract features using frozen VGG16 layers."""
        # Input: L channel (batch_size, 1, 224, 224)
        features = self.vgg_features(x)  # Output: (batch_size, 512, 7, 7)
        features = self.adaptive_pool(features)  # Ensure (batch_size, 512, 7, 7)
        return features

class CustomCNNDecoder(nn.Module):
    """Custom CNN decoder for predicting A and B channels."""
    
    def __init__(self, input_channels=512):
        super(CustomCNNDecoder, self).__init__()
        
        self.decoder = nn.Sequential(
            # Upsample from 7x7 to 14x14
            nn.ConvTranspose2d(input_channels, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Upsample from 14x14 to 28x28
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Upsample from 28x28 to 56x56
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Upsample from 56x56 to 112x112
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Upsample from 112x112 to 224x224
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # Final layer: predict A and B channels
            nn.Conv2d(16, 2, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Output range [-1, 1] for A and B channels
        )
    
    def forward(self, x):
        """Forward pass through decoder.
        Args:
            x: Input features (B, 512, 7, 7)
        Returns:
            ab: A and B channels (B, 2, 224, 224)
        """
        return self.decoder(x)

class TinyImageNetColorizationModel(nn.Module):
    """Complete colorization model: VGG16 + Custom CNN Decoder."""
    
    def __init__(self, freeze_layers=True):
        super(TinyImageNetColorizationModel, self).__init__()
        
        self.feature_extractor = VGG16FeatureExtractor(freeze_layers=freeze_layers)
        self.decoder = CustomCNNDecoder(input_channels=512)
    
    def forward(self, L):
        """Forward pass.
        Args:
            L: L channel input (B, 1, 224, 224)
        Returns:
            ab: Predicted A and B channels (B, 2, 224, 224)
        """
        # Extract features using VGG16
        features = self.feature_extractor(L)
        
        # Decode to A and B channels
        ab = self.decoder(features)
        
        return ab

class ColorizationInference:
    """Main class for image colorization inference."""
    
    def __init__(self, model_path, device=None):
        """Initialize the inference pipeline.
        
        Args:
            model_path: Path to the trained model (.pth file)
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    
    def _load_model(self, model_path):
        """Load the trained model."""
        print(f"Loading model from {model_path}")
        
        # Create model
        model = TinyImageNetColorizationModel(freeze_layers=True)
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
            print(f"Training loss: {checkpoint.get('train_loss', 'unknown')}")
            print(f"Validation loss: {checkpoint.get('val_loss', 'unknown')}")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded model weights")
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def _rgb_to_lab(self, rgb_image):
        """Convert RGB image to LAB colorspace.
        
        Args:
            rgb_image: RGB image as numpy array (H, W, 3)
        Returns:
            lab_image: LAB image as numpy array (H, W, 3)
        """
        # Convert RGB (uint8) to LAB using skimage for consistency with training
        rgb_norm = rgb_image.astype(np.float32) / 255.0
        lab_image = skcolor.rgb2lab(rgb_norm)
        return lab_image
    
    def _lab_to_rgb(self, lab_image):
        """Convert LAB image to RGB colorspace.
        
        Args:
            lab_image: LAB image as numpy array (H, W, 3)
        Returns:
            rgb_image: RGB image as numpy array (H, W, 3)
        """
        # Convert LAB (float) to RGB using skimage
        rgb_image = skcolor.lab2rgb(lab_image)
        return (rgb_image * 255).astype(np.uint8)
    
    def _preprocess_image(self, image_path):
        """Preprocess input image.
        
        Args:
            image_path: Path to input image
        Returns:
            L_tensor: L channel as tensor (1, 1, 224, 224)
            original_size: Original image size (width, height)
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        # Convert to numpy
        image_np = np.array(image)
        
        # Convert to LAB
        lab_image = self._rgb_to_lab(image_np)

        # Extract L channel (range 0-100)
        L_channel = lab_image[:, :, 0]

        # Convert to PIL for transforms
        L_pil = Image.fromarray((L_channel / 100.0 * 255).astype(np.uint8), mode='L')

        # Apply transforms and renormalize to match training ((L-50)/50)
        L_tensor = self.transform(L_pil).unsqueeze(0)
        L_tensor = (L_tensor * 100.0 - 50.0) / 50.0

        return L_tensor.to(self.device), original_size
    
    def _postprocess_output(self, L_tensor, ab_tensor, original_size):
        """Postprocess model output to RGB image.
        
        Args:
            L_tensor: L channel tensor (1, 1, 224, 224)
            ab_tensor: Predicted AB channels tensor (1, 2, 224, 224)
            original_size: Original image size (width, height)
        Returns:
            rgb_image: Final RGB image as PIL Image
        """
        # Move to CPU and convert to numpy
        L_np = L_tensor.cpu().squeeze().numpy()
        ab_np = ab_tensor.cpu().squeeze().numpy()

        # Denormalize to LAB ranges used during training
        L_np = (L_np * 50.0) + 50.0
        ab_np = ab_np * 128.0

        lab_image = np.zeros((224, 224, 3), dtype=np.float32)
        lab_image[:, :, 0] = np.clip(L_np, 0, 100)
        lab_image[:, :, 1] = np.clip(ab_np[0], -128, 127)
        lab_image[:, :, 2] = np.clip(ab_np[1], -128, 127)

        rgb_image = skcolor.lab2rgb(lab_image)
        rgb_image = np.clip(rgb_image, 0.0, 1.0)
        rgb_pil = Image.fromarray((rgb_image * 255).astype(np.uint8))
        
        # Resize back to original size
        rgb_pil = rgb_pil.resize(original_size, Image.LANCZOS)
        
        return rgb_pil
    
    def colorize_image(self, image_path, output_path=None):
        """Colorize a single image.
        
        Args:
            image_path: Path to input grayscale/color image
            output_path: Path to save colorized image (optional)
        Returns:
            colorized_image: PIL Image of colorized result
        """
        # Preprocess
        L_tensor, original_size = self._preprocess_image(image_path)
        
        # Inference
        with torch.no_grad():
            ab_tensor = self.model(L_tensor)
        
        # Postprocess
        colorized_image = self._postprocess_output(L_tensor, ab_tensor, original_size)
        
        # Save if output path provided
        if output_path:
            colorized_image.save(output_path)
            print(f"Colorized image saved to: {output_path}")
        
        return colorized_image
    
    def colorize_batch(self, input_dir, output_dir, extensions=('.jpg', '.jpeg', '.png', '.bmp')):
        """Colorize all images in a directory.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save colorized images
            extensions: Valid image extensions
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Find all images
        image_files = []
        for ext in extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        print(f"Found {len(image_files)} images to colorize")
        
        # Process each image
        for i, image_file in enumerate(image_files, 1):
            try:
                # Generate output filename
                output_file = output_path / f"colorized_{image_file.name}"
                
                # Colorize
                print(f"[{i}/{len(image_files)}] Processing: {image_file.name}")
                self.colorize_image(str(image_file), str(output_file))
                
            except Exception as e:
                print(f"Error processing {image_file.name}: {e}")
        
        print(f"Batch colorization completed! Results saved to: {output_dir}")

def main():
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(description='Image Colorization Inference')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model (.pth file)')
    parser.add_argument('--input', type=str, required=True, help='Input image or directory path')
    parser.add_argument('--output', type=str, help='Output image or directory path')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], help='Device to use (auto-detect if not specified)')
    parser.add_argument('--batch', action='store_true', help='Process all images in input directory')
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = ColorizationInference(args.model, args.device)
    
    if args.batch:
        # Batch processing
        output_dir = args.output or f"{args.input}_colorized"
        inference.colorize_batch(args.input, output_dir)
    else:
        # Single image processing
        output_path = args.output or f"colorized_{Path(args.input).name}"
        inference.colorize_image(args.input, output_path)

if __name__ == "__main__":
    main()