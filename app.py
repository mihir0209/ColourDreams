"""
Flask backend for image colorization service.
Handles image upload, processing, and inference using the trained Tiny ImageNet model.
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import torch
import cv2
import numpy as np
from PIL import Image
import io
import base64
import os
import sys
import logging
from werkzeug.utils import secure_filename
import tempfile
from pathlib import Path
# from direct_inference import DirectModelInference

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference_pipeline import ColorizationInference

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'temp_uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for model
inference_model = None

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_model():
    """Initialize the colorization model."""
    global inference_model
    
    try:
        # Initialize the deep learning colorization model
        logger.info("🚀 Initializing advanced colorization model...")
        
        inference_model = ColorizationInference()
        logger.info("✅ Model loaded successfully!")
            
        return True
        
    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        return False

def image_to_base64(image):
    """Convert PIL Image to base64 string."""
    try:
        buffer = io.BytesIO()
        if image.mode == 'L':
            image.save(buffer, format='PNG')
        else:
            image.save(buffer, format='JPEG')
        buffer.seek(0)
        
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        format_type = 'png' if image.mode == 'L' else 'jpeg'
        return f"data:image/{format_type};base64,{image_base64}"
        
    except Exception as e:
        logger.error(f"Error converting image to base64: {e}")
        return None

@app.route('/')
def home():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": inference_model is not None,
        "trained_model_loaded": inference_model is not None
    })

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get model information."""
    if inference_model is None:
        return jsonify({"error": "Model not loaded"}), 503
    
    return jsonify({
        "model_type": "Deep Encoder-Decoder Network",
        "architecture": "Advanced skip-connection colorization model",
        "device": str(inference_model.device),
        "input_size": "256x256 (auto-resized)",
        "color_space": "LAB"
    })

@app.route('/colorize', methods=['POST'])
def colorize_image():
    """Colorize an uploaded image."""
    try:
        # Check if any model is available
        if inference_model is None:
            return jsonify({"error": "No colorization models available"}), 503
        
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Check file type
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type. Allowed: PNG, JPG, JPEG, GIF, BMP"}), 400
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{filename}")
        file.save(temp_path)
        
        try:
            # Load original image for comparison
            original_image = Image.open(temp_path).convert('RGB')
            original_size = original_image.size
            
            # Create grayscale version
            grayscale_image = original_image.convert('L')
            
            # Use your trained model
            colorized_image, error = inference_model.colorize_image(temp_path)
            if error:
                return jsonify({"error": f"Colorization failed: {error}"}), 500
            
            method_used = "Advanced Deep Learning Model"
            
            # Convert images to base64 for web display
            results = {
                'original': image_to_base64(original_image),
                'grayscale': image_to_base64(grayscale_image),
                'colorized': image_to_base64(colorized_image)
            }
            
            # Clean up temp file
            os.remove(temp_path)
            
            return jsonify({
                "success": True,
                "images": results,
                "message": f"Image colorized successfully using {method_used}",
                "original_size": original_size,
                "method": method_used
            })
            
        except Exception as e:
            # Clean up temp file if error occurs
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e
            
    except Exception as e:
        logger.error(f"Error in colorize endpoint: {e}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

@app.route('/colorize-batch', methods=['POST'])
def colorize_batch():
    """Colorize multiple images."""
    try:
        # Check if model is loaded
        if inference_model is None:
            return jsonify({"error": "Model not initialized"}), 503
        
        # Check if files are present
        if 'files' not in request.files:
            return jsonify({"error": "No files uploaded"}), 400
        
        files = request.files.getlist('files')
        
        if not files or all(f.filename == '' for f in files):
            return jsonify({"error": "No files selected"}), 400
        
        results = []
        
        for i, file in enumerate(files):
            if file and allowed_file(file.filename):
                try:
                    # Save uploaded file temporarily
                    filename = secure_filename(file.filename)
                    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"batch_{i}_{filename}")
                    file.save(temp_path)
                    
                    # Load original image
                    original_image = Image.open(temp_path).convert('RGB')
                    grayscale_image = original_image.convert('L')
                    
                    # Colorize
                    colorized_image, error = inference_model.colorize_image(temp_path)
                    if error:
                        raise RuntimeError(error)
                    
                    # Convert to base64
                    result = {
                        'filename': filename,
                        'original': image_to_base64(original_image),
                        'grayscale': image_to_base64(grayscale_image),
                        'colorized': image_to_base64(colorized_image),
                        'success': True
                    }
                    
                    results.append(result)
                    
                    # Clean up
                    os.remove(temp_path)
                    
                except Exception as e:
                    logger.error(f"Error processing {file.filename}: {e}")
                    results.append({
                        'filename': file.filename,
                        'success': False,
                        'error': str(e)
                    })
                    
                    # Clean up on error
                    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"batch_{i}_{filename}")
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
        
        return jsonify({
            "success": True,
            "message": f"Processed {len(results)} images",
            "results": results
        })
        
    except Exception as e:
        logger.error(f"Error in batch colorize endpoint: {e}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({"error": "File too large. Maximum size is 16MB"}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors."""
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    # Initialize model on startup
    print("Initializing colorization model...")
    if init_model():
        print("✓ Model loaded successfully!")
        print("Starting Flask server...")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("✗ Failed to load model. Please check the model file path.")
        print("Expected model location: models/best_tiny_imagenet_colorization_model.pth")