"""
FastAPI backend for image colorization service.
Handles image upload, processing, and inference using the trained model.
"""

from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import torch
import cv2
import numpy as np
from PIL import Image
import io
import base64
import os
import sys
import uvicorn
from typing import Optional
import logging

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.colorization_model import create_model
from data.preprocessing import lab_to_rgb_tensor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Image Colorization API",
    description="AI-powered image colorization using VGG16 + Custom CNN",
    version="1.0.0"
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
templates = Jinja2Templates(directory="frontend/templates")

# Global variables for model
model = None
device = None

class ImageProcessor:
    """Handle image processing operations."""
    
    @staticmethod
    def preprocess_image(image_bytes: bytes) -> torch.Tensor:
        """Convert uploaded image to L channel tensor."""
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array
            image_np = np.array(image)
            
            # Resize to 224x224
            image_resized = cv2.resize(image_np, (224, 224))
            
            # Convert RGB to LAB
            lab_image = cv2.cvtColor(image_resized, cv2.COLOR_RGB2LAB)
            
            # Extract L channel and normalize
            L_channel = lab_image[:, :, 0].astype(np.float32) / 100.0
            
            # Convert to tensor
            L_tensor = torch.from_numpy(L_channel).unsqueeze(0).unsqueeze(0)  # (1, 1, 224, 224)
            
            return L_tensor, image_resized
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise HTTPException(status_code=400, detail="Invalid image format")
    
    @staticmethod
    def postprocess_results(L_tensor: torch.Tensor, AB_tensor: torch.Tensor, original_image: np.ndarray) -> dict:
        """Convert model outputs to displayable images."""
        try:
            # Move tensors to CPU
            L_cpu = L_tensor.cpu().squeeze(0)  # (1, 224, 224)
            AB_cpu = AB_tensor.cpu().squeeze(0)  # (2, 224, 224)
            
            # Convert to numpy
            L_np = L_cpu.squeeze(0).numpy()  # (224, 224)
            AB_np = AB_cpu.permute(1, 2, 0).numpy()  # (224, 224, 2)
            
            # Denormalize
            L_denorm = L_np * 100.0
            AB_denorm = AB_np * 255.0 - 128.0
            
            # Combine LAB channels
            lab_image = np.zeros((224, 224, 3), dtype=np.float32)
            lab_image[:, :, 0] = L_denorm
            lab_image[:, :, 1:] = AB_denorm
            
            # Clip values to valid ranges
            lab_image[:, :, 0] = np.clip(lab_image[:, :, 0], 0, 100)
            lab_image[:, :, 1:] = np.clip(lab_image[:, :, 1:], -128, 127)
            
            # Convert LAB to RGB
            lab_uint8 = lab_image.astype(np.uint8)
            rgb_colorized = cv2.cvtColor(lab_uint8, cv2.COLOR_LAB2RGB)
            
            # Convert images to base64 for web display
            results = {
                'original': ImageProcessor.numpy_to_base64(original_image),
                'lab': ImageProcessor.numpy_to_base64(lab_uint8),
                'rgb_colorized': ImageProcessor.numpy_to_base64(rgb_colorized),
                'grayscale': ImageProcessor.numpy_to_base64(cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY))
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error postprocessing results: {e}")
            raise HTTPException(status_code=500, detail="Error processing model outputs")
    
    @staticmethod
    def numpy_to_base64(image_np: np.ndarray) -> str:
        """Convert numpy array to base64 string."""
        try:
            # Convert to PIL Image
            if len(image_np.shape) == 2:  # Grayscale
                pil_image = Image.fromarray(image_np, mode='L')
            else:  # RGB
                pil_image = Image.fromarray(image_np, mode='RGB')
            
            # Convert to bytes
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            buffer.seek(0)
            
            # Encode to base64
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return f"data:image/png;base64,{image_base64}"
            
        except Exception as e:
            logger.error(f"Error converting to base64: {e}")
            return ""

def load_model(model_path: str = "checkpoints/best_model.pth"):
    """Load the trained colorization model."""
    global model, device
    
    try:
        # Determine device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Loading model on device: {device}")
        
        # Create model
        model, device = create_model(pretrained=True, device=device)
        
        # Load trained weights if available
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded trained model from {model_path}")
        else:
            logger.warning(f"Trained model not found at {model_path}. Using pretrained VGG16 only.")
        
        model.eval()
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    load_model()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/colorize")
async def colorize_image(file: UploadFile = File(...)):
    """Colorize an uploaded grayscale image."""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read file
        file_bytes = await file.read()
        
        # Preprocess image
        L_tensor, original_image = ImageProcessor.preprocess_image(file_bytes)
        
        # Move to device
        L_tensor = L_tensor.to(device)
        
        # Inference
        with torch.no_grad():
            AB_predicted = model(L_tensor)
        
        # Postprocess results
        results = ImageProcessor.postprocess_results(L_tensor, AB_predicted, original_image)
        
        return JSONResponse(content={
            "success": True,
            "images": results,
            "message": "Image colorized successfully"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in colorize endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown"
    }

@app.get("/model-info")
async def model_info():
    """Get model information."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            "model_type": "VGG16 + Custom CNN",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": total_params - trainable_params,
            "device": str(device),
            "input_size": "224x224",
            "color_space": "LAB"
        }
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving model information")

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )