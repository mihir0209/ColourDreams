# Project Structure

```
image-colorization/
├── 📄 README.md                    # Main project documentation
├── 📄 LICENSE                      # MIT License
├── 📄 CONTRIBUTING.md              # Contribution guidelines
├── 📄 requirements.txt             # Python dependencies
├── 📄 .gitignore                   # Git ignore rules
│
├── 🚀 app.py                       # FastAPI web application
├── 🚀 quickstart.py                # Quick start script
├── 🚀 download_dataset.py          # Dataset download utility
├── 🧪 test_setup.py                # System testing script
│
├── 📁 models/                      # Model architecture
│   └── colorization_model.py      # VGG16 + Custom CNN model
│
├── 📁 data/                        # Data processing
│   └── preprocessing.py            # RGB to LAB conversion, data loaders
│
├── 📁 training/                    # Training pipeline
│   └── train.py                   # Training loop and utilities
│
├── 📁 frontend/                    # Web interface
│   ├── templates/
│   │   └── index.html             # Main HTML template
│   └── static/
│       ├── css/
│       │   └── style.css          # Custom styling
│       └── js/
│           └── app.js             # Frontend JavaScript
│
└── 📁 example_images/              # Example images for demo
```

## File Descriptions

### Core Application Files
- **`app.py`** - FastAPI backend server with REST API endpoints
- **`quickstart.py`** - Easy-to-use script for training and running the app
- **`download_dataset.py`** - Automated Tiny-ImageNet dataset downloader

### Model Components
- **`models/colorization_model.py`** - Complete model architecture:
  - VGG16 feature extractor (frozen)
  - Custom CNN decoder for A&B channel prediction
  - Model creation and testing utilities

### Data Pipeline
- **`data/preprocessing.py`** - Data processing pipeline:
  - RGB to LAB color space conversion
  - Image resizing to 224×224
  - Dataset splitting and data loaders
  - Color space conversion utilities

### Training Infrastructure
- **`training/train.py`** - Complete training system:
  - Training and validation loops
  - Loss functions (MSE + L1)
  - Model checkpointing
  - Learning rate scheduling
  - Training visualization

### Web Interface
- **`frontend/templates/index.html`** - Modern, responsive web interface
- **`frontend/static/css/style.css`** - Custom styling with Bootstrap
- **`frontend/static/js/app.js`** - Frontend functionality and API communication

### Utilities
- **`test_setup.py`** - Comprehensive system testing
- **`requirements.txt`** - All Python dependencies
- **`.gitignore`** - Excludes datasets, checkpoints, and cache files

## Generated During Runtime

### Data Directories (excluded from Git)
```
dataset/                            # Downloaded Tiny-ImageNet
processed_data/                     # Preprocessed data splits
checkpoints/                        # Model checkpoints and best models
test_images/                        # Test images for debugging
```

### Model Outputs
```
checkpoints/
├── best_model.pth                 # Best performing model
├── checkpoint_epoch_N.pth         # Regular checkpoints
└── training_history.png           # Training loss curves
```

## Key Features

- **🧠 AI Model**: VGG16 + Custom CNN for image colorization
- **🎨 Color Science**: LAB color space for perceptual accuracy
- **🌐 Web Interface**: Modern, responsive design with Bootstrap
- **⚡ Fast API**: RESTful API with automatic documentation
- **📊 Training**: Complete pipeline with monitoring and checkpointing
- **🧪 Testing**: Comprehensive test suite for validation
- **📚 Documentation**: Detailed guides and code documentation

## Usage Patterns

1. **Quick Start**: `python quickstart.py all`
2. **Training Only**: `python quickstart.py train`
3. **Web App Only**: `python quickstart.py web`
4. **Testing**: `python quickstart.py test`

This structure ensures easy development, deployment, and contribution while maintaining clean separation of concerns.