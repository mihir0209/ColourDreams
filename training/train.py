"""
Training pipeline for image colorization model.
Includes training loop, loss functions, validation, and checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import time
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.colorization_model import create_model, count_parameters
from data.preprocessing import create_data_loaders, lab_to_rgb_tensor

class ColorizationLoss(nn.Module):
    """Custom loss function for image colorization."""
    
    def __init__(self, alpha=1.0, beta=0.1):
        super(ColorizationLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.alpha = alpha  # Weight for MSE loss
        self.beta = beta    # Weight for L1 loss
        
    def forward(self, predicted_ab, target_ab):
        # Combine MSE and L1 losses for better training stability
        mse = self.mse_loss(predicted_ab, target_ab)
        l1 = self.l1_loss(predicted_ab, target_ab)
        
        total_loss = self.alpha * mse + self.beta * l1
        return total_loss, mse, l1

class Trainer:
    """Training class for image colorization model."""
    
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # Loss function
        self.criterion = ColorizationLoss(alpha=config['alpha'], beta=config['beta'])
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        # Create checkpoint directory
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        
    def train_epoch(self, epoch):
        """Train the model for one epoch."""
        self.model.train()
        running_loss = 0.0
        running_mse = 0.0
        running_l1 = 0.0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["num_epochs"]}')
        
        for batch_idx, (L_batch, AB_batch, _) in enumerate(progress_bar):
            # Move to device
            L_batch = L_batch.to(self.device)
            AB_batch = AB_batch.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            predicted_AB = self.model(L_batch)
            
            # Calculate loss
            total_loss, mse_loss, l1_loss = self.criterion(predicted_AB, AB_batch)
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Statistics
            running_loss += total_loss.item()
            running_mse += mse_loss.item()
            running_l1 += l1_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'MSE': f'{mse_loss.item():.4f}',
                'L1': f'{l1_loss.item():.4f}'
            })
            
            # Free memory
            del L_batch, AB_batch, predicted_AB, total_loss, mse_loss, l1_loss
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Calculate average losses
        avg_loss = running_loss / len(self.train_loader)
        avg_mse = running_mse / len(self.train_loader)
        avg_l1 = running_l1 / len(self.train_loader)
        
        return avg_loss, avg_mse, avg_l1
    
    def validate_epoch(self, epoch):
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        running_mse = 0.0
        running_l1 = 0.0
        
        with torch.no_grad():
            for L_batch, AB_batch, _ in tqdm(self.val_loader, desc='Validation'):
                # Move to device
                L_batch = L_batch.to(self.device)
                AB_batch = AB_batch.to(self.device)
                
                # Forward pass
                predicted_AB = self.model(L_batch)
                
                # Calculate loss
                total_loss, mse_loss, l1_loss = self.criterion(predicted_AB, AB_batch)
                
                # Statistics
                running_loss += total_loss.item()
                running_mse += mse_loss.item()
                running_l1 += l1_loss.item()
                
                # Free memory
                del L_batch, AB_batch, predicted_AB, total_loss, mse_loss, l1_loss
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Calculate average losses
        avg_loss = running_loss / len(self.val_loader)
        avg_mse = running_mse / len(self.val_loader)
        avg_l1 = running_l1 / len(self.val_loader)
        
        return avg_loss, avg_mse, avg_l1
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.config['checkpoint_dir'], 
            f'checkpoint_epoch_{epoch+1}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config['checkpoint_dir'], 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"✓ Best model saved at epoch {epoch+1}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.learning_rates = checkpoint.get('learning_rates', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_epoch = checkpoint.get('best_epoch', 0)
        
        return checkpoint['epoch']
    
    def plot_training_history(self):
        """Plot training history."""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot losses
        axes[0].plot(self.train_losses, label='Train Loss', color='blue')
        axes[0].plot(self.val_losses, label='Validation Loss', color='red')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot learning rate
        axes[1].plot(self.learning_rates, label='Learning Rate', color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['checkpoint_dir'], 'training_history.png'))
        plt.show()
    
    def train(self, resume_from=None):
        """Main training loop."""
        start_epoch = 0
        
        # Resume from checkpoint if provided
        if resume_from and os.path.exists(resume_from):
            start_epoch = self.load_checkpoint(resume_from) + 1
            print(f"Resumed training from epoch {start_epoch}")
        
        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Total epochs: {self.config['num_epochs']}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Learning rate: {self.config['learning_rate']}")
        
        # Count parameters
        count_parameters(self.model)
        
        training_start_time = time.time()
        
        for epoch in range(start_epoch, self.config['num_epochs']):
            epoch_start_time = time.time()
            
            # Train
            train_loss, train_mse, train_l1 = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_mse, val_l1 = self.validate_epoch(epoch)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Record history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.learning_rates.append(current_lr)
            
            # Check for best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
            
            # Save checkpoint
            if (epoch + 1) % self.config['save_frequency'] == 0:
                self.save_checkpoint(epoch, is_best=is_best)
            
            epoch_time = time.time() - epoch_start_time
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']} Summary:")
            print(f"Train Loss: {train_loss:.4f} (MSE: {train_mse:.4f}, L1: {train_l1:.4f})")
            print(f"Val Loss: {val_loss:.4f} (MSE: {val_mse:.4f}, L1: {val_l1:.4f})")
            print(f"Learning Rate: {current_lr:.6f}")
            print(f"Epoch Time: {epoch_time:.2f}s")
            print(f"Best Val Loss: {self.best_val_loss:.4f} (Epoch {self.best_epoch+1})")
            print("-" * 50)
        
        total_time = time.time() - training_start_time
        print(f"\nTraining completed in {total_time/3600:.2f} hours")
        
        # Save final model
        self.save_checkpoint(self.config['num_epochs']-1, is_best=False)
        
        # Plot training history
        self.plot_training_history()

def create_training_config():
    """Create training configuration."""
    config = {
        # Model parameters
        'num_epochs': 50,
        'batch_size': 16,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        
        # Loss function parameters
        'alpha': 1.0,  # MSE weight
        'beta': 0.1,   # L1 weight
        
        # Training parameters
        'save_frequency': 5,  # Save checkpoint every N epochs
        'checkpoint_dir': 'checkpoints',
        
        # Data parameters
        'num_workers': 4,
        'pin_memory': True,
    }
    
    return config

def main():
    """Main training function."""
    # Configuration
    config = create_training_config()
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create model
    model, device = create_model(pretrained=True, device=device)
    
    # Create data loaders
    splits_file = "processed_data/dataset_splits.json"
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir="dataset",
        splits_file=splits_file,
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, device, config)
    
    # Start training
    trainer.train()
    
    print("Training completed!")

if __name__ == "__main__":
    main()