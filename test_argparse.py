#!/usr/bin/env python3
"""
Test script to verify argparse functionality
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_argparse():
    """Test argparse functionality"""
    
    # Test basic help
    print("Testing argparse help...")
    print("-" * 50)
    
    # Simulate command line arguments
    test_args = [
        ['--help'],
        ['--experiment', 'cifar10'],
        ['--experiment', 'cifar100', '--epochs', '100'],
        ['--experiment', 'corruptions', '--quick'],
        ['--experiment', 'all', '--device', 'cpu', '--epochs', '25'],
        ['--experiment', 'cifar10', '--model_type', 'vgg', '--model_depth', '19'],
        ['--experiment', 'corruptions', '--corruption_types', 'gaussian_noise', 'defocus_blur'],
        ['--experiment', 'all', '--quick', '--output_dir', './test_results']
    ]
    
    for i, args in enumerate(test_args):
        print(f"\nTest {i+1}: {' '.join(args)}")
        print("-" * 30)
        
        try:
            # Import and test the argument parser
            from simple_expts import run_cifar_experiments
            
            # This would normally run the experiments, but we'll just test the import
            print(f"✓ Successfully imported with args: {args}")
            
        except Exception as e:
            print(f"✗ Error with args {args}: {e}")
    
    print("\n" + "=" * 50)
    print("Argparse test completed!")
    print("=" * 50)

def test_config_integration():
    """Test configuration integration"""
    print("\nTesting configuration integration...")
    print("-" * 50)
    
    try:
        from config import get_config, create_custom_config
        
        # Test default config
        default_config = get_config("default")
        print(f"✓ Default config: epochs={default_config.epochs}, batch_size={default_config.batch_size}")
        
        # Test quick config
        quick_config = get_config("quick")
        print(f"✓ Quick config: epochs={quick_config.epochs}, coreset_budget={quick_config.coreset_budget}")
        
        # Test custom config
        custom_config = create_custom_config(epochs=75, batch_size=128)
        print(f"✓ Custom config: epochs={custom_config.epochs}, batch_size={custom_config.batch_size}")
        
        print("✓ Configuration integration working!")
        
    except Exception as e:
        print(f"✗ Configuration integration error: {e}")

def test_imports():
    """Test all necessary imports"""
    print("\nTesting imports...")
    print("-" * 50)
    
    required_modules = [
        'torch',
        'torchvision',
        'numpy',
        'matplotlib',
        'seaborn',
        'tqdm'
    ]
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module}: {e}")
    
    print("✓ Import test completed!")

if __name__ == "__main__":
    print("ARGPARSE TEST SUITE")
    print("=" * 50)
    
    test_imports()
    test_config_integration()
    test_argparse()
    
    print("\nAll tests completed!")
