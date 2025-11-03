# 🌈 Image Colorization with Deep Learning

**Transform grayscale images into vibrant color using Advanced Encoder-Decoder Architecture**

![Colorization Example](https://via.placeholder.com/800x200/1a1a1a/ffffff?text=Grayscale+→+AI+Colorization+→+Full+Color)

## 🚀 **Quick Start**

**⚡ Fast and easy colorization with pre-trained model**

### **Getting Started:**
1. � **Install** dependencies: `pip install -r requirements.txt`
2. 🚀 **Run** the app: `python app.py`
3. 🌐 **Open** browser at `http://localhost:5000`
4. � **Upload** grayscale images
5. 🌈 **Download** colorized results

## 🎯 **Project Overview**

### **What it does:**
- 🖼️ **Input**: Grayscale images
- 🧠 **Process**: Deep encoder-decoder network with skip connections
- 🌈 **Output**: Realistic color images

### **Key Features:**
- ✅ **Advanced Architecture**: Multi-layer encoder-decoder design
- ✅ **LAB Color Space**: Perceptually uniform color processing
- ✅ **Skip Connections**: Preserves fine details during colorization
- ✅ **Auto-optimization**: Detects hardware and optimizes settings
- ✅ **Web Interface**: Flask + Bootstrap frontend

## 📁 **Project Structure**

```
image-colorization/
├── �️ app.py                             # Web interface (Flask)
├── 🧠 models/                             # Model architecture
│   ├── colorization_model.py             # Deep encoder-decoder network
│   └── best_tiny_imagenet_colorization_model.pth  # Trained weights
├── 📊 data/                               # Data processing
├── � templates/                          # Web UI templates
├── � static/                             # CSS and JavaScript
└── 📋 requirements.txt                    # Dependencies
```

## 🖥️ **Usage**

### **Web Interface:**
```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py

# Open browser at http://localhost:5000
```

### **Upload and Colorize:**
1. Click "Choose File" to select a grayscale image
2. Click "Colorize" to process the image
3. View and download the colorized result

## 🎛️ **Technical Details**

### **Architecture:**
- **Encoder**: Deep convolutional layers with dilated convolutions
- **Decoder**: Multi-scale feature aggregation with skip connections
- **Input**: L channel (lightness) from LAB color space
- **Output**: AB channels (color) predictions
- **Color Space**: LAB for perceptually uniform color representation

### **Model Specifications:**
- **Input Resolution**: 256x256 (automatically resized)
- **Output Resolution**: Matches input resolution
- **Processing**: Efficient CPU/GPU inference
- **Color Accuracy**: Trained on diverse image datasets

## 🎨 **Example Results**

The model learns to colorize various objects:
- 🌸 **Flowers**: Realistic petal colors
- 🏞️ **Landscapes**: Natural sky and vegetation
- 🐕 **Animals**: Proper fur and eye colors
- 🏠 **Objects**: Context-aware colorization

## 🔧 **Requirements**

```
torch >= 2.0.0
torchvision >= 0.15.0
numpy
Pillow
scikit-image
flask
requests
```

## 🎯 **Features**

### **1. 🌐 Web Interface**
- Upload images via browser
- Real-time colorization
- User-friendly interface
- Download colorized results

### **2. 🧠 Advanced Model**
- Deep encoder-decoder architecture
- Skip connections for detail preservation
- LAB color space processing
- CPU and GPU support

### **3. � High-Quality Results**
- Natural-looking colors
- Context-aware colorization
- Works on various image types

## 📝 **Citation**

If you use this project, please cite:
```
@misc{image-colorization-2025,
  title={Image Colorization with Deep Learning},
  author={Your Name},
  year={2025},
  url={https://github.com/mihir0209/ColourDreams}
}
```

## 📄 **License**

MIT License - see [LICENSE](LICENSE) file for details.

## 🤝 **Contributing**

Contributions welcome! Please feel free to submit issues and pull requests.

---

**🌈 Transform your grayscale memories into vibrant color!**