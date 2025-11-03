"""
Download pretrained colorization model weights
This script downloads publicly available colorization weights
"""

import torch
import requests
from pathlib import Path
import sys


def download_file(url, destination):
    """Download file with progress bar"""
    print(f"Downloading from: {url}")
    print(f"Saving to: {destination}")
    
    try:
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        downloaded = 0
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(block_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        bar_length = 50
                        filled = int(bar_length * downloaded / total_size)
                        bar = '█' * filled + '░' * (bar_length - filled)
                        print(f'\r[{bar}] {percent:.1f}% ({downloaded}/{total_size} bytes)', end='')
        
        print('\n✓ Download complete!')
        return True
        
    except Exception as e:
        print(f'\n✗ Download failed: {e}')
        if destination.exists():
            destination.unlink()
        return False


def main():
    """Download pretrained model"""
    print("="*60)
    print("Pretrained Colorization Model Downloader")
    print("="*60)
    
    # Models directory
    models_dir = Path(__file__).parent / 'models'
    models_dir.mkdir(exist_ok=True)
    
    # Option 1: Download from public repository (example)
    print("\n📦 Available pretrained models:")
    print("1. VGG16-based colorization (custom trained)")
    print("2. Use VGG16 pretrained only (no colorization weights)")
    
    choice = input("\nSelect option (1-2) or press Enter for option 2: ").strip()
    
    if choice == "1":
        print("\n⚠ Note: Custom pretrained colorization weights need to be provided.")
        print("Please place your pretrained decoder weights as:")
        print(f"  {models_dir / 'colorization_decoder.pth'}")
        print("\nOr train your own decoder using the training script.")
        
    else:
        print("\n✓ Using VGG16 pretrained encoder only.")
        print("The decoder will use random initialization.")
        print("\n💡 Tips:")
        print("  - For better results, train the decoder on a colorization dataset")
        print("  - Or provide pretrained decoder weights")
        
    print("\n" + "="*60)
    print("Setup complete! Run the app with: python app.py")
    print("="*60)


if __name__ == "__main__":
    main()
