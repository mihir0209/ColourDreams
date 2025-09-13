# Image Colorization using Transfer Learning VGG16 Model

A deep learning-based image colorization system that transforms grayscale images into vibrant colored images using transfer learning with VGG16 and a custom CNN architecture.

![Project Demo](https://img.shields.io/badge/Project-AI%20Image%20Colorization-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-teal)

## 🎯 Project Overview

This project implements an advanced image colorization system that uses:
- **VGG16** as a pre-trained feature extractor (frozen weights)
- **Custom CNN** for predicting A & B channels in LAB color space
- **LAB color space** for perceptually accurate colorization
- **Tiny-ImageNet dataset** for training (200 classes, 100K images)
- **Modern web interface** for easy deployment and interaction

## 🏗️ Architecture

```
Input (Grayscale) → VGG16 Features → Custom CNN → A&B Channels → RGB Output
     224×224            512×7×7        Decoder      2×224×224     224×224
```

### Model Components:
1. **VGG16 Feature Extractor**: Pre-trained on ImageNet, frozen weights
2. **Custom CNN Decoder**: Upsampling network to predict A&B channels
3. **Color Space Conversion**: RGB ↔ LAB transformations
4. **Post-processing**: Combine L channel with predicted A&B channels

## 📊 Dataset

- **Source**: Tiny-ImageNet-200
- **Size**: 100,000 training images across 200 classes
- **Split**: 80% train, 15% validation, 5% test
- **Preprocessing**: RGB to LAB conversion, resize to 224×224

## 🛠️ Technology Stack

### Backend
- **Python 3.8+**
- **PyTorch 2.0+** - Deep learning framework
- **FastAPI** - Modern web framework
- **OpenCV** - Image processing
- **scikit-image** - Advanced image operations
- **NumPy, Matplotlib** - Data manipulation and visualization

### Frontend
- **HTML5, CSS3, JavaScript**
- **Bootstrap 5** - Responsive design
- **Font Awesome** - Icons

### MLOps
- **Model checkpointing** and versioning
- **Training monitoring** with loss tracking
- **Automated data pipeline**

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone https://github.com/mihir0209/ColourDreams.git
cd CColourDreams
pip install -r requirements.txt
```

### 2. Download Dataset
```bash
python download_dataset.py
```

### 3. Preprocess Data
```bash
python data/preprocessing.py
```

### 4. Train Model
```bash
python training/train.py
```

### 5. Run Web Application
```bash
python app.py
```

Visit `http://localhost:8000` to access the web interface.

## 📁 Project Structure

```
image-colorization/
├── app.py                          # FastAPI web application
├── requirements.txt                # Python dependencies
├── download_dataset.py             # Dataset download script
├── README.md                       # Project documentation
│
├── data/
│   └── preprocessing.py            # Data preprocessing pipeline
│
├── models/
│   └── colorization_model.py      # Model architecture
│
├── training/
│   └── train.py                   # Training pipeline
│
├── frontend/
│   ├── templates/
│   │   └── index.html             # Main web interface
│   └── static/
│       ├── css/
│       │   └── style.css          # Custom styling
│       └── js/
│           └── app.js             # Frontend JavaScript
│
├── dataset/                        # Downloaded dataset
│   └── tiny-imagenet-200/
│
├── processed_data/                 # Preprocessed data
│   └── dataset_splits.json
│
└── checkpoints/                    # Model checkpoints
    ├── best_model.pth
    └── training_history.png
```

## 🎮 Usage

### Web Interface
1. Open your browser and navigate to `http://localhost:8000`
2. Upload a grayscale image (JPG, PNG formats supported)
3. Click "Colorize Image" 
4. View results showing:
   - Original image
   - Grayscale input
   - LAB colorized output
   - RGB colorized output

### API Usage
```python
import requests

# Upload and colorize image
files = {'file': open('grayscale_image.jpg', 'rb')}
response = requests.post('http://localhost:8000/colorize', files=files)
result = response.json()
```

### Programmatic Usage
```python
from models.colorization_model import create_model
from data.preprocessing import ImageProcessor

# Load model
model, device = create_model(pretrained=True)
model.load_state_dict(torch.load('checkpoints/best_model.pth'))

# Process image
processor = ImageProcessor()
L_tensor = processor.preprocess_image(image_bytes)
colorized = model(L_tensor)
```

## 🔬 Technical Details

### Model Architecture
- **Input**: L channel (1×224×224)
- **VGG16 Features**: 512×7×7 feature maps
- **Decoder**: 5 transposed convolution layers
- **Output**: A&B channels (2×224×224)
- **Activation**: Sigmoid for normalized outputs

### Training Configuration
- **Optimizer**: Adam (lr=1e-4)
- **Loss Function**: Combined MSE + L1 loss
- **Batch Size**: 16
- **Epochs**: 50
- **Scheduler**: ReduceLROnPlateau

### Performance Optimizations
- **Gradient clipping** to prevent exploding gradients
- **Mixed precision training** support
- **Memory-efficient data loading**
- **Model checkpointing** every 5 epochs

## 📈 Training Results

The model training provides:
- Training and validation loss curves
- Learning rate scheduling visualization
- Model performance metrics
- Best model checkpointing

## 🎨 Color Space Details

### LAB Color Space Benefits:
- **Perceptually uniform**: Changes in color values correspond to visual perception
- **Separate luminance**: L channel contains brightness information
- **Color channels**: A (green-red) and B (blue-yellow) channels contain color information
- **Better training**: Easier for neural networks to learn color relationships

### Conversion Process:
1. RGB → LAB conversion during preprocessing
2. L channel used as model input
3. A&B channels predicted by the model
4. LAB → RGB conversion for final output

## 🔧 API Endpoints

### Main Endpoints
- `GET /` - Web interface
- `POST /colorize` - Colorize uploaded image
- `GET /health` - Health check
- `GET /model-info` - Model information

### Response Format
```json
{
  "success": true,
  "images": {
    "original": "data:image/png;base64,...",
    "grayscale": "data:image/png;base64,...",
    "lab": "data:image/png;base64,...",
    "rgb_colorized": "data:image/png;base64,..."
  },
  "message": "Image colorized successfully"
}
```

## 🧪 Testing

### Unit Tests
```bash
python -m pytest tests/
```

### Model Testing
```bash
python models/colorization_model.py
```

### API Testing
```bash
curl -X POST "http://localhost:8000/colorize" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@test_image.jpg"
```

## 📚 References

1. **Zhang, R., Isola, P., & Efros, A. A. (2016)**. "Colorful image colorization." *European conference on computer vision (ECCV)*. [Paper](https://arxiv.org/abs/1603.08511)

2. **Iizuka, S., Simo-Serra, E., & Ishikawa, H. (2016)**. "Let there be color!: joint end-to-end learning of global and local image priors for automatic image colorization." *ACM Transactions on Graphics (TOG)*. [Paper](https://hi.cs.waseda.ac.jp/~iizuka/projects/colorization/en/)

3. **Simonyan, K., & Zisserman, A. (2014)**. "Very deep convolutional networks for large-scale image recognition." *International Conference on Learning Representations (ICLR)*. [Paper](https://arxiv.org/abs/1409.1556)

4. **Deng, J., et al. (2009)**. "ImageNet: A large-scale hierarchical image database." *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*. [Paper](http://www.image-net.org/papers/imagenet_cvpr09.pdf)

5. **Le, H., & Borji, A. (2017)**. "What are the receptive, effective receptive, and projective fields of neurons in convolutional neural networks?" *arXiv preprint*. [Paper](https://arxiv.org/abs/1705.07049)

## 🎯 Future Enhancements

- [ ] **GAN-based colorization** for more realistic results
- [ ] **Attention mechanisms** for better feature focusing
- [ ] **User-guided colorization** with color hints
- [ ] **Batch processing** capabilities
- [ ] **Model quantization** for mobile deployment
- [ ] **Real-time video colorization**
- [ ] **Advanced evaluation metrics** (LPIPS, FID)

## 🐛 Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce batch size in training configuration
   - Use gradient accumulation

2. **Model not loading**
   - Check checkpoint file path
   - Verify model architecture compatibility

3. **Poor colorization quality**
   - Ensure sufficient training epochs
   - Check dataset quality and diversity
   - Adjust loss function weights

4. **Web interface not loading**
   - Check FastAPI server status
   - Verify port availability (8000)
   - Check static file paths

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 👥 Authors

- **MiHiR** - *Initial work* - [YourProfile](https://github.com/mihir0209)

## 🙏 Acknowledgments

- Thanks to the PyTorch team for the excellent deep learning framework
- Tiny-ImageNet dataset creators for providing quality training data
- VGG16 authors for the foundational architecture
- FastAPI creators for the modern web framework
- Bootstrap team for the responsive design framework

---

**Note**: This project is for educational and research purposes. The colorization quality depends on the training data and model architecture choices.