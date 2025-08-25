#!/usr/bin/env python3
"""
Example usage of the CIFAR framework for quick testing
"""

import torch
import torch.nn as nn
from simple_expts import (
    CIFARVariations, 
    CIFARResNet, 
    RLGuidedGaLoreSelector,
    run_single_cifar_experiment
)
from torch.utils.data import DataLoader
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quick_cifar10_test():
    """Quick test with CIFAR10 standard dataset"""
    logger.info("Running quick CIFAR10 test...")
    
    # Get CIFAR10 standard dataset
    cifar10_variations = CIFARVariations.get_cifar10_variations("./data")
    dataset_info = cifar10_variations['cifar10_standard']
    
    logger.info(f"Using dataset: {dataset_info['name']}")
    
    # Create model
    model = CIFARResNet(num_classes=10, depth=20)
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model = model.to(device)
    
    logger.info(f"Model created on {device}")
    
    # Initialize selector
    selector = RLGuidedGaLoreSelector(
        model=model,
        train_dataset=dataset_info['train'],
        val_dataset=dataset_info['test'],
        memory_budget_mb=500,  # Smaller budget for quick test
        rank=128
    )
    
    logger.info("Selector initialized")
    
    # Run quick experiment (fewer epochs)
    result = run_single_cifar_experiment(
        selector=selector,
        model=model,
        train_dataset=dataset_info['train'],
        val_dataset=dataset_info['test'],
        dataset_name="cifar10_standard_quick",
        device=device,
        epochs=10,  # Quick test with 10 epochs
        coreset_budget=500  # Smaller coreset
    )
    
    logger.info(f"Quick test completed!")
    logger.info(f"Final accuracy: {result['final_val_accuracy']:.2f}%")
    logger.info(f"Best accuracy: {result['best_val_accuracy']:.2f}%")
    logger.info(f"Phase transitions detected: {result['num_phase_transitions']}")
    
    return result

def test_corruption_robustness():
    """Test robustness to a few corruption types"""
    logger.info("Testing corruption robustness...")
    
    # Get base CIFAR10 dataset
    cifar10_train = CIFARVariations.get_cifar10_variations("./data")['cifar10_standard']['train']
    
    # Test a few corruption types
    corruption_types = ['gaussian_noise', 'defocus_blur', 'brightness']
    severities = [1, 3]
    
    results = {}
    
    for corruption_type in corruption_types:
        for severity in severities:
            logger.info(f"Testing {corruption_type} severity {severity}")
            
            # Create corrupted dataset
            corrupted_train = CIFARVariations.create_corrupted_cifar10(
                cifar10_train, corruption_type, severity
            )
            
            # Create model
            model = CIFARResNet(num_classes=10, depth=20)
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            model = model.to(device)
            
            # Initialize selector
            selector = RLGuidedGaLoreSelector(
                model=model,
                train_dataset=corrupted_train,
                val_dataset=cifar10_train,  # Use clean data for validation
                memory_budget_mb=500,
                rank=128
            )
            
            # Run quick experiment
            result = run_single_cifar_experiment(
                selector=selector,
                model=model,
                train_dataset=corrupted_train,
                val_dataset=cifar10_train,
                dataset_name=f"cifar10_{corruption_type}_sev{severity}_quick",
                device=device,
                epochs=5,  # Very quick test
                coreset_budget=300
            )
            
            results[f"{corruption_type}_sev{severity}"] = result
            
            # Clean up
            del model, selector
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif device == "mps":
                # MPS doesn't have empty_cache, but we can force garbage collection
                import gc
                gc.collect()
    
    # Print results
    logger.info("\nCorruption robustness results:")
    for key, result in results.items():
        logger.info(f"{key}: {result['final_val_accuracy']:.2f}% accuracy")
    
    return results

def test_strategy_selection():
    """Test the RL-guided strategy selection"""
    logger.info("Testing RL-guided strategy selection...")
    
    # Get CIFAR10 dataset
    cifar10_variations = CIFARVariations.get_cifar10_variations("./data")
    dataset_info = cifar10_variations['cifar10_standard']
    
    # Create model
    model = CIFARResNet(num_classes=10, depth=20)
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model = model.to(device)
    
    # Initialize selector
    selector = RLGuidedGaLoreSelector(
        model=model,
        train_dataset=dataset_info['train'],
        val_dataset=dataset_info['test'],
        memory_budget_mb=500,
        rank=128
    )
    
    # Test strategy selection for a few epochs
    for epoch in range(5):
        logger.info(f"\nEpoch {epoch + 1}:")
        
        # Simulate performance
        current_performance = 50.0 + epoch * 5.0  # Simulated improvement
        
        # Select coreset
        selected_indices, selection_info = selector.select_coreset(500, current_performance)
        
        logger.info(f"  Selected strategy: {selection_info['selected_strategy']}")
        logger.info(f"  Phase: {selection_info['current_phase']}")
        logger.info(f"  Compression ratio: {selection_info['compression_ratio']:.3f}")
        
        if selection_info['phase_transition']:
            logger.info(f"  Phase transition detected! Confidence: {selection_info['transition_confidence']:.3f}")
    
    logger.info("Strategy selection test completed!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick CIFAR framework tests')
    parser.add_argument('--test', type=str, default='all',
                       choices=['quick', 'corruption', 'strategy', 'all'],
                       help='Type of test to run')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS doesn't have manual seed functions
        pass
    
    try:
        if args.test in ['quick', 'all']:
            quick_cifar10_test()
        
        if args.test in ['corruption', 'all']:
            test_corruption_robustness()
        
        if args.test in ['strategy', 'all']:
            test_strategy_selection()
            
        logger.info("All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
