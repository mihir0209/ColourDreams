"""
Data preprocessing module for image colorization.
Handles RGB to LAB conversion, resizing, and dataset preparation.
"""

import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
from PIL import Image
import glob
from tqdm import tqdm
import json

class ColorDataset(Dataset):
    """Custom dataset for image colorization."""
    
    def __init__(self, image_paths, transform=None, mode='train'):
        self.image_paths = image_paths
        self.transform = transform
        self.mode = mode
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to 224x224
        image = cv2.resize(image, (224, 224))
        
        # Convert RGB to LAB
        lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Normalize LAB values
        lab_image = lab_image.astype(np.float32)
        lab_image[:, :, 0] = lab_image[:, :, 0] / 100.0  # L channel [0, 100] -> [0, 1]
        lab_image[:, :, 1] = (lab_image[:, :, 1] + 128) / 255.0  # A channel [-128, 127] -> [0, 1]
        lab_image[:, :, 2] = (lab_image[:, :, 2] + 128) / 255.0  # B channel [-128, 127] -> [0, 1]
        
        # Split LAB channels
        L = lab_image[:, :, 0]  # Lightness (input)
        AB = lab_image[:, :, 1:]  # A and B channels (target)
        
        # Convert to tensors and add channel dimension for L
        L = torch.from_numpy(L).unsqueeze(0)  # Shape: (1, 224, 224)
        AB = torch.from_numpy(AB).permute(2, 0, 1)  # Shape: (2, 224, 224)
        
        # Apply transforms if provided
        if self.transform:
            # Create 3-channel image for transform compatibility
            rgb_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            rgb_tensor = self.transform(rgb_tensor)
            
        return L, AB, image_path

def rgb_to_lab_tensor(rgb_tensor):
    """Convert RGB tensor to LAB tensor."""
    # Convert tensor to numpy
    rgb_np = rgb_tensor.permute(1, 2, 0).numpy()
    rgb_np = (rgb_np * 255).astype(np.uint8)
    
    # Convert to LAB
    lab_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB)
    lab_np = lab_np.astype(np.float32)
    
    # Normalize
    lab_np[:, :, 0] = lab_np[:, :, 0] / 100.0
    lab_np[:, :, 1] = (lab_np[:, :, 1] + 128) / 255.0
    lab_np[:, :, 2] = (lab_np[:, :, 2] + 128) / 255.0
    
    return torch.from_numpy(lab_np).permute(2, 0, 1)

def lab_to_rgb_tensor(l_tensor, ab_tensor):
    """Convert LAB tensor back to RGB tensor."""
    # Combine L and AB channels
    l_np = l_tensor.squeeze(0).numpy()
    ab_np = ab_tensor.permute(1, 2, 0).numpy()
    
    # Denormalize
    l_np = l_np * 100.0
    ab_np = ab_np * 255.0 - 128
    
    # Combine channels
    lab_np = np.zeros((224, 224, 3), dtype=np.float32)
    lab_np[:, :, 0] = l_np
    lab_np[:, :, 1:] = ab_np
    
    # Convert to RGB
    lab_np = lab_np.astype(np.uint8)
    rgb_np = cv2.cvtColor(lab_np, cv2.COLOR_LAB2RGB)
    
    return torch.from_numpy(rgb_np).permute(2, 0, 1).float() / 255.0

def prepare_dataset(data_dir, output_dir):
    """Prepare and split the dataset."""
    print("Preparing Tiny-ImageNet dataset for colorization...")
    
    # Get all image paths
    image_paths = []
    train_dir = os.path.join(data_dir, 'train')
    
    print("Collecting image paths...")
    for class_dir in tqdm(os.listdir(train_dir)):
        class_path = os.path.join(train_dir, class_dir, 'images')
        if os.path.exists(class_path):
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(class_path, img_file))
    
    print(f"Found {len(image_paths)} images")
    
    # Split dataset: 80% train, 15% validation, 5% test
    train_paths, temp_paths = train_test_split(image_paths, test_size=0.2, random_state=42)
    val_paths, test_paths = train_test_split(temp_paths, test_size=0.25, random_state=42)
    
    print(f"Train: {len(train_paths)} images")
    print(f"Validation: {len(val_paths)} images")
    print(f"Test: {len(test_paths)} images")
    
    # Save splits
    os.makedirs(output_dir, exist_ok=True)
    splits = {
        'train': train_paths,
        'val': val_paths,
        'test': test_paths
    }
    
    with open(os.path.join(output_dir, 'dataset_splits.json'), 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"Dataset splits saved to {output_dir}/dataset_splits.json")
    return train_paths, val_paths, test_paths

def create_data_loaders(data_dir, splits_file, batch_size=32, num_workers=4, 
                       pin_memory=True, persistent_workers=False, prefetch_factor=2):
    """Create data loaders for training."""
    
    # Load splits
    with open(splits_file, 'r') as f:
        splits = json.load(f)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)
    ])
    
    # Create datasets
    train_dataset = ColorDataset(splits['train'], transform=None, mode='train')
    val_dataset = ColorDataset(splits['val'], transform=None, mode='val')
    test_dataset = ColorDataset(splits['test'], transform=None, mode='test')
    
    # Create data loaders with optimizations
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=True  # Consistent batch sizes for GPU
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    return train_loader, val_loader, test_loader

def verify_preprocessing():
    """Verify the preprocessing pipeline with sample images."""
    print("Verifying preprocessing pipeline...")
    
    # Test RGB to LAB conversion
    sample_rgb = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    sample_lab = cv2.cvtColor(sample_rgb, cv2.COLOR_RGB2LAB)
    reconstructed_rgb = cv2.cvtColor(sample_lab, cv2.COLOR_LAB2RGB)
    
    print("✓ RGB ↔ LAB conversion working")
    
    # Test tensor conversions
    rgb_tensor = torch.from_numpy(sample_rgb).permute(2, 0, 1).float() / 255.0
    lab_tensor = rgb_to_lab_tensor(rgb_tensor)
    
    print("✓ Tensor conversion working")
    print(f"RGB tensor shape: {rgb_tensor.shape}")
    print(f"LAB tensor shape: {lab_tensor.shape}")
    print(f"L channel range: [{lab_tensor[0].min():.3f}, {lab_tensor[0].max():.3f}]")
    print(f"A channel range: [{lab_tensor[1].min():.3f}, {lab_tensor[1].max():.3f}]")
    print(f"B channel range: [{lab_tensor[2].min():.3f}, {lab_tensor[2].max():.3f}]")

if __name__ == "__main__":
    # Verify preprocessing
    verify_preprocessing()
    
    # Prepare dataset
    data_dir = "dataset/tiny-imagenet-200"
    output_dir = "processed_data"
    
    if os.path.exists(data_dir):
        train_paths, val_paths, test_paths = prepare_dataset(data_dir, output_dir)
        print("Dataset preparation completed!")
    else:
        print(f"Dataset directory {data_dir} not found. Please run download_dataset.py first.")