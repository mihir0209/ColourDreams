"""
Quick Start Script for Image Colorization Project
This script provides easy commands to train the model and run the web interface.
"""

import subprocess
import sys
import os
import argparse
import time

def run_command(command, description):
    """Run a command with proper error handling."""
    print(f"\n🚀 {description}")
    print(f"Command: {command}")
    print("-" * 50)
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=False)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with error code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⏹️  {description} interrupted by user")
        return False

def check_requirements():
    """Check if all requirements are installed."""
    print("🔍 Checking requirements...")
    
    # Map pip package names to their import names
    package_mapping = {
        'torch': 'torch',
        'torchvision': 'torchvision', 
        'fastapi': 'fastapi',
        'uvicorn': 'uvicorn',
        'opencv-python': 'cv2',
        'pillow': 'PIL',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'scikit-learn': 'sklearn',
        'tqdm': 'tqdm',
        'requests': 'requests'
    }
    
    missing_packages = []
    for pip_name, import_name in package_mapping.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(pip_name)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Please run: pip install -r requirements.txt")
        return False
    else:
        print("✅ All required packages are installed")
        return True

def train_model(epochs=None, batch_size=None):
    """Start model training."""
    print("🎯 Starting Model Training...")
    
    # Check if dataset exists
    if not os.path.exists("dataset/tiny-imagenet-200"):
        print("❌ Dataset not found. Downloading dataset first...")
        if not run_command("python download_dataset.py", "Downloading Tiny-ImageNet dataset"):
            return False
    
    # Check if data is preprocessed
    if not os.path.exists("processed_data/dataset_splits.json"):
        print("❌ Preprocessed data not found. Running preprocessing...")
        if not run_command("python data/preprocessing.py", "Preprocessing dataset"):
            return False
    
    # Run training
    command = "python training/train.py"
    return run_command(command, "Training model")

def run_web_app(port=8000):
    """Start the web application."""
    print("🌐 Starting Web Application...")
    
    command = f"python app.py"
    print(f"\n📱 The web interface will be available at: http://localhost:{port}")
    print("Press Ctrl+C to stop the server")
    
    return run_command(command, "Running web application")

def quick_test():
    """Run quick system test."""
    print("🧪 Running Quick System Test...")
    return run_command("python test_setup.py", "System test")

def main():
    parser = argparse.ArgumentParser(description="Quick Start Script for Image Colorization")
    parser.add_argument('action', choices=['test', 'train', 'web', 'all'], 
                       help='Action to perform')
    parser.add_argument('--epochs', type=int, default=50, 
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=16, 
                       help='Batch size for training (default: 16)')
    parser.add_argument('--port', type=int, default=8000, 
                       help='Port for web application (default: 8000)')
    
    args = parser.parse_args()
    
    print("🎨 Image Colorization Project - Quick Start")
    print("=" * 50)
    
    # Check requirements first
    if not check_requirements():
        sys.exit(1)
    
    if args.action == 'test':
        success = quick_test()
        
    elif args.action == 'train':
        success = train_model(args.epochs, args.batch_size)
        
    elif args.action == 'web':
        success = run_web_app(args.port)
        
    elif args.action == 'all':
        print("🚀 Running complete pipeline...")
        
        # Test system
        if not quick_test():
            print("❌ System test failed. Please check the setup.")
            sys.exit(1)
        
        # Train model
        if not train_model(args.epochs, args.batch_size):
            print("❌ Training failed. Please check the logs.")
            sys.exit(1)
        
        # Ask user if they want to run web app
        response = input("\n🤔 Training completed! Do you want to start the web interface? (y/n): ")
        if response.lower() in ['y', 'yes']:
            success = run_web_app(args.port)
        else:
            print("✅ Training completed successfully!")
            print(f"💡 You can start the web interface later with: python quickstart.py web")
            success = True
    
    if success:
        print("\n🎉 Operation completed successfully!")
    else:
        print("\n❌ Operation failed. Please check the errors above.")
        sys.exit(1)

def show_help():
    """Show detailed help information."""
    help_text = """
🎨 Image Colorization Project - Quick Start Guide

Available Commands:
==================

1. Test System:
   python quickstart.py test
   - Runs comprehensive system test
   - Verifies all components are working

2. Train Model:
   python quickstart.py train [--epochs 50] [--batch-size 16]
   - Downloads dataset if needed
   - Preprocesses data if needed
   - Starts model training

3. Run Web App:
   python quickstart.py web [--port 8000]
   - Starts the FastAPI web interface
   - Access at http://localhost:8000

4. Complete Pipeline:
   python quickstart.py all
   - Runs test, then training, then optionally web app
   - Complete end-to-end execution

Examples:
=========
python quickstart.py test                    # Test system
python quickstart.py train --epochs 100      # Train for 100 epochs
python quickstart.py web --port 8080         # Run web app on port 8080
python quickstart.py all                     # Complete pipeline

Manual Steps:
=============
If you prefer to run steps manually:
1. python download_dataset.py               # Download dataset
2. python data/preprocessing.py             # Preprocess data
3. python training/train.py                 # Train model
4. python app.py                           # Run web app

Troubleshooting:
===============
- Check requirements: pip install -r requirements.txt
- Verify GPU availability: nvidia-smi (if using CUDA)
- Check disk space: Dataset needs ~500MB, training needs ~2GB
- Memory issues: Reduce batch size in training config

For more information, see README.md
    """
    print(help_text)

if __name__ == "__main__":
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] in ['--help', '-h', 'help']):
        show_help()
    else:
        main()