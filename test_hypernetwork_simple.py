#!/usr/bin/env python3
"""
Simple test script to verify hypernetwork integration without downloading data
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test all hypernetwork imports"""
    print("Testing imports...")
    try:
        from simple_expts import RLGuidedGaLoreSelector, CIFARResNet, SelectionStrategy
        from hypernetwork import (
            MultiScoringHypernetwork,
            SubmodularMultiScoringSelector,
            TrainingState,
            GradientMagnitudeScoring,
            DiversityScoring,
            UncertaintyScoring,
            BoundaryScoring,
            InfluenceScoring,
            ForgetScoring
        )
        from config import ExperimentConfig
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_hypernetwork_components():
    """Test hypernetwork component initialization"""
    print("\nTesting hypernetwork components...")
    
    try:
        from hypernetwork import (
            MultiScoringHypernetwork,
            TrainingState,
            GradientMagnitudeScoring,
            DiversityScoring,
            UncertaintyScoring
        )
        from simple_expts import CIFARResNet
        
        # Create a simple model
        model = CIFARResNet(num_classes=10, depth=20)
        device = torch.device('cpu')  # Use CPU for testing
        model.to(device)
        
        # Create scoring functions
        scoring_functions = [
            GradientMagnitudeScoring(model, device),
            UncertaintyScoring(model, device),
        ]
        
        # Create hypernetwork
        hypernetwork = MultiScoringHypernetwork(
            scoring_functions=scoring_functions,
            state_dim=19,
            hidden_dim=32,
            attention_heads=1
        )
        
        print(f"✓ Created hypernetwork with {len(scoring_functions)} scoring functions")
        
        # Test forward pass
        test_state = TrainingState(
            epoch=1,
            loss=0.5,
            accuracy=0.8,
            gradient_norm=1.0,
            learning_rate=0.001,
            data_seen_ratio=0.1,
            class_distribution=np.ones(10) / 10,
            performance_history=[0.7, 0.75, 0.8],
            selection_diversity=0.5
        )
        
        with torch.no_grad():
            weights, temperature, value, perf_pred = hypernetwork(test_state)
        
        print(f"✓ Forward pass successful")
        print(f"  Weights: {weights.tolist()}")
        print(f"  Temperature: {temperature.item():.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_selector_integration():
    """Test integration with RLGuidedGaLoreSelector"""
    print("\nTesting selector integration...")
    
    try:
        from simple_expts import RLGuidedGaLoreSelector, CIFARResNet
        from torch.utils.data import TensorDataset
        
        # Create dummy dataset
        dummy_data = torch.randn(100, 3, 32, 32)
        dummy_labels = torch.randint(0, 10, (100,))
        train_dataset = TensorDataset(dummy_data, dummy_labels)
        test_dataset = TensorDataset(dummy_data[:50], dummy_labels[:50])
        
        # Create model
        model = CIFARResNet(num_classes=10, depth=20)
        
        # Initialize selector with hypernetwork
        selector = RLGuidedGaLoreSelector(
            model=model,
            train_dataset=train_dataset,
            val_dataset=test_dataset,
            memory_budget_mb=100,
            rank=64,
            use_hypernetwork=True
        )
        
        # Check components
        assert hasattr(selector, 'hypernetwork'), "Hypernetwork not found"
        assert hasattr(selector, 'hypernet_selector'), "Hypernet selector not found"
        assert len(selector.scoring_functions) == 6, f"Expected 6 scoring functions, got {len(selector.scoring_functions)}"
        
        print(f"✓ Selector initialized with hypernetwork")
        print(f"  Scoring functions: {[sf.name for sf in selector.scoring_functions]}")
        
        # Test selection
        selected_indices, info = selector.select_coreset(budget=10, current_performance=0.5)
        print(f"✓ Selected {len(selected_indices)} samples")
        print(f"  Strategy: {info.get('selected_strategy', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_integration():
    """Test configuration with hypernetwork settings"""
    print("\nTesting configuration...")
    
    try:
        from config import ExperimentConfig, create_custom_config
        
        # Test default config
        config = ExperimentConfig()
        assert hasattr(config, 'use_hypernetwork'), "Missing use_hypernetwork"
        assert hasattr(config, 'hypernet_hidden_dim'), "Missing hypernet_hidden_dim"
        
        print(f"✓ Default config has hypernetwork settings")
        print(f"  use_hypernetwork: {config.use_hypernetwork}")
        print(f"  hypernet_hidden_dim: {config.hypernet_hidden_dim}")
        
        # Test custom config
        custom = create_custom_config(
            use_hypernetwork=True,
            hypernet_hidden_dim=128,
            hypernet_update_freq=3
        )
        
        print(f"✓ Custom config created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False

if __name__ == "__main__":
    print("HYPERNETWORK INTEGRATION TEST (Simple)")
    print("=" * 60)
    
    all_passed = True
    
    # Run tests
    all_passed &= test_imports()
    all_passed &= test_hypernetwork_components()
    all_passed &= test_selector_integration()
    all_passed &= test_config_integration()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed!")
        print("\nHypernetwork integration is working correctly.")
        print("\nTo run experiments with hypernetwork:")
        print("  python simple_expts.py --experiment cifar10 --epochs 50")
        print("\nThe hypernetwork will be used automatically for faster convergence.")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        sys.exit(1)