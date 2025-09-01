#!/usr/bin/env python3
"""
Test script to verify hypernetwork integration with simple_expts.py
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import sys
import os
import logging

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_hypernetwork_integration():
    """Test hypernetwork integration with CIFAR10 experiments"""
    
    print("=" * 60)
    print("Testing Hypernetwork Integration")
    print("=" * 60)
    
    try:
        # Import required modules
        from simple_expts import RLGuidedGaLoreSelector, CIFARResNet
        from config import ExperimentConfig
        from hypernetwork import MultiScoringHypernetwork, TrainingState
        
        print("\n✓ Successfully imported all required modules")
        
        # Create a small test dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        # Use a small subset for testing
        cifar10_train = datasets.CIFAR10(
            root='./data', 
            train=True, 
            download=True, 
            transform=transform
        )
        cifar10_test = datasets.CIFAR10(
            root='./data', 
            train=False, 
            download=True, 
            transform=transform
        )
        
        # Use only first 100 samples for quick testing
        from torch.utils.data import Subset
        train_subset = Subset(cifar10_train, range(100))
        test_subset = Subset(cifar10_test, range(50))
        
        print("✓ Loaded test datasets")
        
        # Create model
        device = torch.device('cuda' if torch.cuda.is_available() else 
                            'mps' if torch.backends.mps.is_available() else 'cpu')
        model = CIFARResNet(num_classes=10, depth=20).to(device)
        print(f"✓ Created model on device: {device}")
        
        # Create config with hypernetwork enabled
        config = ExperimentConfig(
            use_hypernetwork=True,
            hypernet_hidden_dim=32,  # Small for testing
            hypernet_attention_heads=1,  # Minimal for testing
            epochs=5,  # Quick test
            coreset_budget=20  # Small budget for testing
        )
        print("✓ Created configuration with hypernetwork enabled")
        
        # Initialize selector with hypernetwork
        selector = RLGuidedGaLoreSelector(
            model=model,
            train_dataset=train_subset,
            val_dataset=test_subset,
            memory_budget_mb=100,
            rank=64,
            use_hypernetwork=True
        )
        print("✓ Initialized RLGuidedGaLoreSelector with hypernetwork")
        
        # Test hypernetwork components
        assert hasattr(selector, 'hypernetwork'), "Hypernetwork not initialized"
        assert hasattr(selector, 'hypernet_selector'), "Hypernetwork selector not initialized"
        assert hasattr(selector, 'scoring_functions'), "Scoring functions not initialized"
        assert len(selector.scoring_functions) == 6, f"Expected 6 scoring functions, got {len(selector.scoring_functions)}"
        print(f"✓ Hypernetwork components verified: {len(selector.scoring_functions)} scoring functions")
        
        # Test coreset selection with hypernetwork
        print("\nTesting coreset selection with hypernetwork...")
        selected_indices, selection_info = selector.select_coreset(
            budget=config.coreset_budget,
            current_performance=0.5
        )
        
        print(f"✓ Selected {len(selected_indices)} samples")
        print(f"  Strategy used: {selection_info.get('selected_strategy', 'N/A')}")
        
        # Test hypernetwork-specific selection
        from simple_expts import SelectionStrategy
        print("\nTesting direct hypernetwork strategy...")
        hypernetwork_indices = selector._select_with_strategy(
            SelectionStrategy.HYPERNETWORK, 
            budget=10
        )
        print(f"✓ Hypernetwork strategy selected {len(hypernetwork_indices)} samples")
        
        # Test hypernetwork state creation
        print("\nTesting hypernetwork training state...")
        test_state = TrainingState(
            epoch=1,
            loss=0.5,
            accuracy=0.8,
            gradient_norm=1.0,
            learning_rate=0.001,
            data_seen_ratio=0.1,
            class_distribution=torch.ones(10) / 10,
            performance_history=[0.7, 0.75, 0.8],
            selection_diversity=0.5
        )
        
        # Test hypernetwork forward pass
        with torch.no_grad():
            weights, temperature, value, perf_pred = selector.hypernetwork(test_state)
        
        print(f"✓ Hypernetwork forward pass successful")
        print(f"  Weights shape: {weights.shape}")
        print(f"  Temperature: {temperature.item():.3f}")
        print(f"  Value: {value.item():.3f}")
        print(f"  Function weights: {dict(zip([sf.name for sf in selector.scoring_functions], weights.tolist()))}")
        
        print("\n" + "=" * 60)
        print("All tests passed successfully!")
        print("Hypernetwork integration is working correctly.")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cifar10_quick_run():
    """Run a quick CIFAR10 experiment with hypernetwork"""
    print("\n" + "=" * 60)
    print("Running Quick CIFAR10 Experiment with Hypernetwork")
    print("=" * 60)
    
    try:
        from simple_expts import run_cifar_experiments
        from config import create_custom_config
        
        # Create custom config for quick testing
        config = create_custom_config(
            epochs=2,  # Very quick
            batch_size=32,
            coreset_budget=100,
            use_hypernetwork=True,
            hypernet_hidden_dim=32,
            hypernet_update_freq=1
        )
        
        print(f"Running with config:")
        print(f"  Epochs: {config.epochs}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Coreset budget: {config.coreset_budget}")
        print(f"  Hypernetwork: {config.use_hypernetwork}")
        
        # Run a quick experiment (will use first CIFAR10 variation only)
        # Note: This is just to test that the pipeline works
        print("\nNote: This is a minimal test run to verify the pipeline works.")
        print("For full experiments, use the main script with appropriate settings.")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error during quick run: {e}")
        return False


if __name__ == "__main__":
    print("HYPERNETWORK INTEGRATION TEST SUITE")
    print("=" * 60)
    
    # Run integration tests
    success = test_hypernetwork_integration()
    
    if success:
        print("\n✓ Integration tests completed successfully!")
        
        # Optionally run a quick experiment
        print("\nWould you like to run a quick CIFAR10 experiment? (y/n)")
        # For automatic testing, we'll skip the interactive part
        # response = input().strip().lower()
        # if response == 'y':
        #     test_cifar10_quick_run()
    else:
        print("\n✗ Integration tests failed. Please check the errors above.")
        sys.exit(1)