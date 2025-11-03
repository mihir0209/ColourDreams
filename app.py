"""
Flask backend for VGG16-based image colorization service.
Simple and efficient colorization using pretrained VGG16.
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import numpy as np
from PIL import Image
import io
import base64
import os
import sys
import logging
from werkzeug.utils import secure_filename

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
    """Initialize the VGG16 colorization model."""
    global inference_model
    
    try:
        logger.info("🎨 Initializing VGG16 colorization model...")
        inference_model = ColorizationInference()
        logger.info("✅ Model ready!")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        return False

@app.route('/')
def home():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": inference_model is not None
    })

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get model information."""
    if inference_model is None:
        return jsonify({"error": "Model not loaded"}), 503
    
    return jsonify({
        "model_type": "VGG16 Colorization Network",
        "architecture": "VGG16 + Decoder",
        "device": str(inference_model.device),
        "input_size": "256x256",
        "color_space": "LAB"
    })

@app.route('/colorize', methods=['POST'])
def colorize_image():
    """Colorize an uploaded image using VGG16."""
    try:
        if inference_model is None:
            return jsonify({"error": "Model not loaded"}), 503
        
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type"}), 400
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{filename}")
        file.save(temp_path)
        
        try:
            # Load image
            original_image = Image.open(temp_path).convert('RGB')
            
            # Colorize
            colorized_image = inference_model.colorize_image(original_image)
            
            # Convert original to base64
            original_buffer = io.BytesIO()
            original_image.save(original_buffer, format='JPEG')
            original_b64 = base64.b64encode(original_buffer.getvalue()).decode('utf-8')
            
            # Convert colorized to base64
            colorized_buffer = io.BytesIO()
            colorized_image.save(colorized_buffer, format='JPEG')
            colorized_b64 = base64.b64encode(colorized_buffer.getvalue()).decode('utf-8')
            
            # Clean up
            os.remove(temp_path)
            
            return jsonify({
                "status": "success",
                "original_base64": original_b64,
                "colorized_base64": colorized_b64,
                "message": "Image colorized successfully with VGG16"
            })
            
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e
            
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

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
    print("\n" + "="*50)
    print("🎨 VGG16 Image Colorization Server")
    print("="*50)
    print("Initializing model...")
    if init_model():
        print("✓ Model loaded successfully!")
        print("\n🚀 Starting Flask server on http://localhost:5000")
        print("="*50 + "\n")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("✗ Failed to load model.")
        sys.exit(1)