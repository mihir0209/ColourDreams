#!/usr/bin/env python3
"""
Tiny-ImageNet Dataset Download Script for Image Colorization Project
Author: GitHub Copilot
Date: September 2025

This script downloads and extracts the Tiny-ImageNet dataset for use in
the image colorization project using VGG16 + custom CNN architecture.
"""

import os
import urllib.request
import zipfile
import tarfile
import shutil
from pathlib import Path
import sys

def download_file(url, filepath, description="Downloading"):
    """Download a file with progress tracking."""
    def progress_hook(block_num, block_size, total_size):
        if total_size > 0:
            percent = min(100, (block_num * block_size * 100) // total_size)
            sys.stdout.write(f"\r{description}: {percent}% complete")
            sys.stdout.flush()
    
    try:
        urllib.request.urlretrieve(url, filepath, progress_hook)
        print(f"\n✓ Downloaded: {filepath}")
        return True
    except Exception as e:
        print(f"\n✗ Error downloading {url}: {e}")
        return False

def extract_archive(archive_path, extract_to):
    """Extract archive file (zip or tar)."""
    try:
        print(f"Extracting {archive_path}...")
        
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.suffix in ['.tar', '.gz']:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
        
        print(f"✓ Extracted to: {extract_to}")
        return True
    except Exception as e:
        print(f"✗ Error extracting {archive_path}: {e}")
        return False

def organize_dataset(dataset_dir):
    """Organize the dataset into train/val structure."""
    print("Organizing dataset structure...")
    
    # Tiny-ImageNet comes with train/ and val/ folders
    train_dir = dataset_dir / "tiny-imagenet-200" / "train"
    val_dir = dataset_dir / "tiny-imagenet-200" / "val"
    
    # Create organized structure
    organized_dir = dataset_dir / "organized"
    organized_train = organized_dir / "train"
    organized_val = organized_dir / "val"
    organized_test = organized_dir / "test"
    
    organized_dir.mkdir(exist_ok=True)
    organized_train.mkdir(exist_ok=True)
    organized_val.mkdir(exist_ok=True)
    organized_test.mkdir(exist_ok=True)
    
    # Process training data
    if train_dir.exists():
        print("Processing training data...")
        for class_dir in train_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                images_dir = class_dir / "images"
                
                if images_dir.exists():
                    target_class_dir = organized_train / class_name
                    target_class_dir.mkdir(exist_ok=True)
                    
                    # Copy images
                    for img_file in images_dir.glob("*.JPEG"):
                        shutil.copy2(img_file, target_class_dir / img_file.name)
    
    # Process validation data (Tiny-ImageNet val structure is different)
    if val_dir.exists():
        print("Processing validation data...")
        val_annotations = val_dir / "val_annotations.txt"
        val_images = val_dir / "images"
        
        if val_annotations.exists() and val_images.exists():
            # Read annotations to get class mappings
            with open(val_annotations, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        img_name = parts[0]
                        class_name = parts[1]
                        
                        # Create class directory in validation
                        target_class_dir = organized_val / class_name
                        target_class_dir.mkdir(exist_ok=True)
                        
                        # Copy image
                        src_img = val_images / img_name
                        if src_img.exists():
                            shutil.copy2(src_img, target_class_dir / img_name)
    
    # Create a small test set from validation (20% of val data)
    print("Creating test split...")
    for class_dir in organized_val.iterdir():
        if class_dir.is_dir():
            images = list(class_dir.glob("*.JPEG"))
            test_count = len(images) // 5  # 20% for test
            
            if test_count > 0:
                test_class_dir = organized_test / class_dir.name
                test_class_dir.mkdir(exist_ok=True)
                
                # Move some images to test
                for img in images[:test_count]:
                    shutil.move(str(img), str(test_class_dir / img.name))
    
    print("✓ Dataset organization complete!")
    return organized_dir

def verify_dataset(dataset_dir):
    """Verify the dataset structure and count images."""
    print("\nVerifying dataset...")
    
    splits = ['train', 'val', 'test']
    total_images = 0
    
    for split in splits:
        split_dir = dataset_dir / split
        if split_dir.exists():
            split_count = len(list(split_dir.rglob("*.JPEG")))
            total_images += split_count
            print(f"{split.capitalize()}: {split_count} images")
        else:
            print(f"{split.capitalize()}: Not found")
    
    print(f"Total images: {total_images}")
    return total_images > 0

def main():
    """Main function to download and setup Tiny-ImageNet dataset."""
    print("🎨 Image Colorization Dataset Setup")
    print("=" * 50)
    
    # Setup paths
    project_root = Path(__file__).parent
    dataset_dir = project_root / "dataset"
    dataset_dir.mkdir(exist_ok=True)
    
    # Tiny-ImageNet download URL
    tiny_imagenet_url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    archive_path = dataset_dir / "tiny-imagenet-200.zip"
    
    print(f"Dataset directory: {dataset_dir}")
    print(f"Download URL: {tiny_imagenet_url}")
    
    # Check if dataset already exists
    organized_dir = dataset_dir / "organized"
    if organized_dir.exists() and any(organized_dir.iterdir()):
        print("✓ Dataset already exists and is organized!")
        if verify_dataset(organized_dir):
            print("✓ Dataset verification passed!")
            return True
        else:
            print("⚠ Dataset verification failed, re-downloading...")
            shutil.rmtree(organized_dir)
    
    # Download dataset
    print(f"\n📥 Downloading Tiny-ImageNet dataset...")
    print("Note: This is approximately 237MB and may take a few minutes.")
    
    if not download_file(tiny_imagenet_url, archive_path, "Downloading Tiny-ImageNet"):
        print("❌ Failed to download dataset!")
        return False
    
    # Extract dataset
    print(f"\n📦 Extracting dataset...")
    if not extract_archive(archive_path, dataset_dir):
        print("❌ Failed to extract dataset!")
        return False
    
    # Organize dataset
    print(f"\n📁 Organizing dataset structure...")
    organized_dir = organize_dataset(dataset_dir)
    
    # Verify dataset
    if verify_dataset(organized_dir):
        print("\n✅ Dataset setup complete!")
        print(f"Dataset location: {organized_dir}")
        print("\nDataset structure:")
        print("dataset/organized/")
        print("├── train/     # Training images (by class)")
        print("├── val/       # Validation images (by class)")
        print("└── test/      # Test images (by class)")
        
        # Clean up archive
        try:
            archive_path.unlink()
            print(f"✓ Cleaned up archive file: {archive_path.name}")
        except:
            pass
        
        return True
    else:
        print("❌ Dataset verification failed!")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
    
    print(f"\n🎉 Ready to start training your image colorization model!")
    print("Next steps:")
    print("1. Run the preprocessing script to prepare data for training")
    print("2. Train the VGG16 + CNN model")
    print("3. Deploy with FastAPI frontend")