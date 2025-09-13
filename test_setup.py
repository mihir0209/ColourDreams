"""
Quick test script to verify the complete setup works.
This script tests model loading, data preprocessing, and basic inference.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import requests
import time

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_model_architecture():
    """Test if the model can be created and run inference."""
    print("🧪 Testing Model Architecture...")
    
    try:
        from models.colorization_model import create_model, count_parameters
        
        # Create model
        model, device = create_model(pretrained=True, device='cpu')
        print(f"✅ Model created successfully on {device}")
        
        # Count parameters
        total_params, trainable_params = count_parameters(model)
        print(f"✅ Model has {trainable_params:,} trainable parameters")
        
        # Test forward pass
        dummy_input = torch.randn(1, 1, 224, 224)
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✅ Forward pass successful - Output shape: {output.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def test_data_preprocessing():
    """Test data preprocessing pipeline."""
    print("\n📊 Testing Data Preprocessing...")
    
    try:
        from data.preprocessing import verify_preprocessing
        
        verify_preprocessing()
        print("✅ Data preprocessing test passed")
        return True
        
    except Exception as e:
        print(f"❌ Data preprocessing test failed: {e}")
        return False

def test_dataset_availability():
    """Check if dataset is available."""
    print("\n💾 Checking Dataset Availability...")
    
    dataset_path = "dataset/tiny-imagenet-200"
    splits_path = "processed_data/dataset_splits.json"
    
    if os.path.exists(dataset_path):
        print("✅ Tiny-ImageNet dataset found")
        dataset_ok = True
    else:
        print("⚠️  Tiny-ImageNet dataset not found - run download_dataset.py first")
        dataset_ok = False
    
    if os.path.exists(splits_path):
        print("✅ Dataset splits found")
        splits_ok = True
    else:
        print("⚠️  Dataset splits not found - run data/preprocessing.py first")
        splits_ok = False
    
    return dataset_ok and splits_ok

def test_web_interface():
    """Test if the web interface can be started."""
    print("\n🌐 Testing Web Interface...")
    
    try:
        # Import the app
        from app import app
        print("✅ FastAPI app imported successfully")
        
        # Check if templates and static files exist
        templates_exist = os.path.exists("frontend/templates/index.html")
        static_css_exist = os.path.exists("frontend/static/css/style.css")
        static_js_exist = os.path.exists("frontend/static/js/app.js")
        
        if templates_exist and static_css_exist and static_js_exist:
            print("✅ Frontend files found")
            return True
        else:
            print("❌ Some frontend files missing")
            return False
            
    except Exception as e:
        print(f"❌ Web interface test failed: {e}")
        return False

def test_training_pipeline():
    """Test if training pipeline can be imported and configured."""
    print("\n🏃 Testing Training Pipeline...")
    
    try:
        from training.train import create_training_config, ColorizationLoss
        
        # Test configuration
        config = create_training_config()
        print(f"✅ Training configuration created")
        print(f"   - Batch size: {config['batch_size']}")
        print(f"   - Learning rate: {config['learning_rate']}")
        print(f"   - Epochs: {config['num_epochs']}")
        
        # Test loss function
        loss_fn = ColorizationLoss()
        dummy_pred = torch.randn(1, 2, 224, 224)
        dummy_target = torch.randn(1, 2, 224, 224)
        
        total_loss, mse_loss, l1_loss = loss_fn(dummy_pred, dummy_target)
        print(f"✅ Loss function working - Total: {total_loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training pipeline test failed: {e}")
        return False

def create_test_image():
    """Create a simple test image for testing."""
    print("\n🖼️  Creating Test Image...")
    
    try:
        # Create a simple test image
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        test_image_pil = Image.fromarray(test_image)
        
        # Save test image
        os.makedirs("test_images", exist_ok=True)
        test_image_pil.save("test_images/test_image.jpg")
        
        print("✅ Test image created at test_images/test_image.jpg")
        return True
        
    except Exception as e:
        print(f"❌ Test image creation failed: {e}")
        return False

def run_complete_test():
    """Run all tests and provide summary."""
    print("🚀 Starting Complete System Test...\n")
    
    test_results = {
        "Model Architecture": test_model_architecture(),
        "Data Preprocessing": test_data_preprocessing(),
        "Dataset Availability": test_dataset_availability(),
        "Web Interface": test_web_interface(),
        "Training Pipeline": test_training_pipeline(),
        "Test Image Creation": create_test_image()
    }
    
    print("\n" + "="*50)
    print("📋 TEST SUMMARY")
    print("="*50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<30} {status}")
        if result:
            passed += 1
    
    print("="*50)
    print(f"Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! Your setup is ready.")
        print("\nNext steps:")
        print("1. Run 'python training/train.py' to start training")
        print("2. Run 'python app.py' to start the web interface")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

def quick_demo():
    """Run a quick demo if everything is working."""
    print("\n🎬 Running Quick Demo...")
    
    try:
        from models.colorization_model import create_model
        
        # Create model
        model, device = create_model(pretrained=True, device='cpu')
        
        # Create dummy grayscale input
        dummy_l = torch.randn(1, 1, 224, 224)
        
        # Run inference
        model.eval()
        with torch.no_grad():
            predicted_ab = model(dummy_l)
        
        print(f"✅ Demo successful!")
        print(f"   Input shape (L channel): {dummy_l.shape}")
        print(f"   Output shape (AB channels): {predicted_ab.shape}")
        print(f"   Output range: [{predicted_ab.min():.3f}, {predicted_ab.max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False

if __name__ == "__main__":
    # Run complete test
    all_passed = run_complete_test()
    
    # Run demo if all tests passed
    if all_passed:
        quick_demo()
    
    print("\n🏁 Testing completed!")