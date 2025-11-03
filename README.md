# 🎨 AI Image Colorization with VGG16# 🌈 Image Colorization with Deep Learning



**Transform grayscale images into vibrant color using VGG16 deep learning****Transform grayscale images into vibrant color using Advanced Encoder-Decoder Architecture**



![VGG16 Colorization](https://img.shields.io/badge/Model-VGG16-blue.svg)![Colorization Example](https://via.placeholder.com/800x200/1a1a1a/ffffff?text=Grayscale+→+AI+Colorization+→+Full+Color)

![Framework](https://img.shields.io/badge/Framework-PyTorch-red.svg)

![Python](https://img.shields.io/badge/Python-3.8+-green.svg)## 🚀 **Quick Start**



## 🚀 Quick Start**⚡ Fast and easy colorization with pre-trained model**



Get started in under 2 minutes!### **Getting Started:**

1. � **Install** dependencies: `pip install -r requirements.txt`

```bash2. 🚀 **Run** the app: `python app.py`

# 1. Install dependencies3. 🌐 **Open** browser at `http://localhost:5000`

pip install -r requirements.txt4. � **Upload** grayscale images

5. 🌈 **Download** colorized results

# 2. Run the application

python app.py## 🎯 **Project Overview**



# 3. Open your browser### **What it does:**

# Visit http://localhost:5000- 🖼️ **Input**: Grayscale images

```- 🧠 **Process**: Deep encoder-decoder network with skip connections

- 🌈 **Output**: Realistic color images

## ✨ Features

### **Key Features:**

- 🧠 **VGG16 Architecture** - Industry-standard pretrained network- ✅ **Advanced Architecture**: Multi-layer encoder-decoder design

- 🎨 **LAB Color Space** - Perceptually uniform color processing- ✅ **LAB Color Space**: Perceptually uniform color processing

- ⚡ **Fast Inference** - Real-time colorization on GPU/CPU- ✅ **Skip Connections**: Preserves fine details during colorization

- 🌐 **Beautiful Web UI** - Modern, responsive interface- ✅ **Auto-optimization**: Detects hardware and optimizes settings

- 🔒 **Privacy First** - Images processed locally, never stored- ✅ **Web Interface**: Flask + Bootstrap frontend



## 🎯 What It Does## 📁 **Project Structure**



This application uses the power of deep learning to automatically add realistic colors to black & white images:```

image-colorization/

1. **Upload** a grayscale image├── �️ app.py                             # Web interface (Flask)

2. **VGG16** extracts meaningful features├── 🧠 models/                             # Model architecture

3. **Decoder network** predicts color channels│   ├── colorization_model.py             # Deep encoder-decoder network

4. **Download** your colorized image!│   └── best_tiny_imagenet_colorization_model.pth  # Trained weights

├── 📊 data/                               # Data processing

## 🏗️ Architecture├── � templates/                          # Web UI templates

├── � static/                             # CSS and JavaScript

```└── 📋 requirements.txt                    # Dependencies

Input (Grayscale) → VGG16 Features → Decoder → Color Prediction → Output (Color)```

     [L channel]      [256 filters]    [Upsampling]    [AB channels]    [RGB Image]

```## 🖥️ **Usage**



### Technical Details### **Web Interface:**

```bash

- **Feature Extractor**: VGG16 (pretrained on ImageNet)# Install dependencies

- **Processing**: LAB color space for better perceptual resultspip install -r requirements.txt

- **Input Resolution**: 256x256 (auto-resized)

- **Output**: Full-color RGB image# Run the application

- **Backend**: Flask REST APIpython app.py

- **Frontend**: Modern HTML5/CSS3/JavaScript

# Open browser at http://localhost:5000

## 📦 Installation```



### Requirements### **Upload and Colorize:**

1. Click "Choose File" to select a grayscale image

- Python 3.8 or higher2. Click "Colorize" to process the image

- PyTorch 2.0+3. View and download the colorized result

- 4GB+ RAM (8GB recommended)

- GPU optional (CUDA support for faster processing)## 🎛️ **Technical Details**



### Step-by-Step### **Architecture:**

- **Encoder**: Deep convolutional layers with dilated convolutions

```bash- **Decoder**: Multi-scale feature aggregation with skip connections

# Clone the repository- **Input**: L channel (lightness) from LAB color space

git clone https://github.com/mihir0209/ColourDreams.git- **Output**: AB channels (color) predictions

cd ColourDreams/image-colorization- **Color Space**: LAB for perceptually uniform color representation



# Install dependencies### **Model Specifications:**

pip install -r requirements.txt- **Input Resolution**: 256x256 (automatically resized)

- **Output Resolution**: Matches input resolution

# Run the server- **Processing**: Efficient CPU/GPU inference

python app.py- **Color Accuracy**: Trained on diverse image datasets

```

## 🎨 **Example Results**

## 🎨 Usage

The model learns to colorize various objects:

### Web Interface (Recommended)- 🌸 **Flowers**: Realistic petal colors

- 🏞️ **Landscapes**: Natural sky and vegetation

1. Start the server: `python app.py`- 🐕 **Animals**: Proper fur and eye colors

2. Open browser: `http://localhost:5000`- 🏠 **Objects**: Context-aware colorization

3. Drag & drop or click to upload an image

4. Click "Colorize My Image"## 🔧 **Requirements**

5. Download your colorized result!

```

### Python APItorch >= 2.0.0

torchvision >= 0.15.0

```pythonnumpy

from inference_pipeline import ColorizationInferencePillow

from PIL import Imagescikit-image

flask

# Initialize modelrequests

colorizer = ColorizationInference()```



# Load image## 🎯 **Features**

img = Image.open('grayscale_photo.jpg')

### **1. 🌐 Web Interface**

# Colorize- Upload images via browser

colorized = colorizer.colorize_image(img)- Real-time colorization

- User-friendly interface

# Save result- Download colorized results

colorized.save('colorized_photo.jpg')

```### **2. 🧠 Advanced Model**

- Deep encoder-decoder architecture

## 🛠️ Project Structure- Skip connections for detail preservation

- LAB color space processing

```- CPU and GPU support

image-colorization/

├── app.py                      # Flask web server### **3. � High-Quality Results**

├── inference_pipeline.py       # Inference wrapper- Natural-looking colors

├── models/- Context-aware colorization

│   └── colorization_model.py  # VGG16 + Decoder architecture- Works on various image types

├── templates/

│   └── index.html             # Modern web UI## 📝 **Citation**

├── temp_uploads/              # Temporary file storage

└── requirements.txt           # Python dependenciesIf you use this project, please cite:

``````

@misc{image-colorization-2025,

## 🧪 How It Works  title={Image Colorization with Deep Learning},

  author={Your Name},

### 1. Color Space Conversion  year={2025},

  url={https://github.com/mihir0209/ColourDreams}

We use LAB color space instead of RGB:}

- **L channel**: Lightness (0-100)```

- **A channel**: Green to Red (-128 to 127)

- **B channel**: Blue to Yellow (-128 to 127)## 📄 **License**



This separation allows the model to focus on predicting color (A & B) while preserving the original brightness (L).MIT License - see [LICENSE](LICENSE) file for details.



### 2. VGG16 Feature Extraction## 🤝 **Contributing**



VGG16, pretrained on ImageNet, extracts rich semantic features from the grayscale input. We use layers up to pool3 to maintain spatial resolution.Contributions welcome! Please feel free to submit issues and pull requests.



### 3. Decoder Network---



A lightweight CNN decoder upsamples the features and predicts the AB color channels using:**🌈 Transform your grayscale memories into vibrant color!**
- Convolutional layers for feature transformation
- Bilinear upsampling for resolution recovery
- Tanh activation for bounded color predictions

### 4. RGB Reconstruction

The predicted AB channels are combined with the original L channel and converted back to RGB for display.

## 📊 Requirements File

```txt
torch>=2.0.0
torchvision>=0.15.0
numpy
Pillow
scikit-image
flask
flask-cors
Werkzeug
```

## 🎓 Model Details

| Component | Details |
|-----------|---------|
| **Architecture** | VGG16 + Custom Decoder |
| **Parameters** | ~15M (decoder only, VGG16 frozen) |
| **Input Size** | 256×256 |
| **Color Space** | LAB |
| **Framework** | PyTorch |
| **Inference Time** | ~0.1s (GPU) / ~0.5s (CPU) |

## 🌟 Example Results

The model can colorize various types of images:
- 📷 **Old photographs** - Bring family memories to life
- 🏞️ **Landscapes** - Natural skies and vegetation
- 👤 **Portraits** - Realistic skin tones
- 🏛️ **Architecture** - Context-aware building colors

## 🔧 API Endpoints

### POST /colorize
Upload and colorize an image

**Request:**
```javascript
FormData {
  file: <image file>
}
```

**Response:**
```json
{
  "status": "success",
  "original_base64": "...",
  "colorized_base64": "...",
  "message": "Image colorized successfully with VGG16"
}
```

### GET /health
Check server status

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### GET /model-info
Get model information

**Response:**
```json
{
  "model_type": "VGG16 Colorization Network",
  "architecture": "VGG16 + Decoder",
  "device": "cuda",
  "input_size": "256x256",
  "color_space": "LAB"
}
```

## 💡 Tips for Best Results

- Use **clear, well-lit** grayscale images
- **Higher resolution** inputs generally work better
- The model works best on **natural scenes**
- For best results, images should have **good contrast**

## 🚧 Limitations

- May produce unexpected colors for unusual objects
- Performance depends on input image quality
- Not trained specifically for artistic/creative colorization
- Works best with photographic content

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests
- Improve documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **VGG16**: [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
- **PyTorch**: Deep learning framework
- **Flask**: Web framework
- **Bootstrap**: UI components

## 📧 Contact

For questions or feedback:
- GitHub: [@mihir0209](https://github.com/mihir0209)
- Repository: [ColourDreams](https://github.com/mihir0209/ColourDreams)

---

**Made with ❤️ using VGG16 and PyTorch**
