#!/usr/bin/env python3
"""
DataComp MODE Training Script
Server-ready implementation with YAML config support
"""

import argparse
import yaml
import torch
import os
import random
import numpy as np
from pathlib import Path
from datetime import datetime

# Import from mode_vlm_experiment
from mode_vlm_experiment import (
    DataCompMODEConfig,
    DataCompMODE,
    FakeDataset
)


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def yaml_to_datacomp_config(yaml_config):
    """Convert YAML config to DataCompMODEConfig object"""

    # Extract values from YAML and map to correct field names
    config = DataCompMODEConfig(
        # Dataset
        dataset_size=yaml_config['dataset']['size'],
        selection_budget=yaml_config['selection']['budget'],

        # Models
        proxy_model=yaml_config['model']['clip_model'],
        main_model=yaml_config['model']['clip_model'],

        # Selection strategy
        selection_strategy=yaml_config['selection']['strategy'],

        # Proxy weights (use selection weights from YAML)
        proxy_weight=yaml_config['selection']['nuclear_norm_weight'],
        checkpoint_weight=yaml_config['selection']['clip_score_weight'],

        # Training
        num_epochs=yaml_config['training']['epochs'],
        batch_size=yaml_config['training']['batch_size'],
        learning_rate=yaml_config['training']['learning_rate'],
        warmup_steps=yaml_config['training']['warmup_steps'],

        # Efficiency
        selection_batch_size=yaml_config['selection']['batch_size'],
        mixed_precision=yaml_config['model']['mixed_precision'],
        device=yaml_config['model']['device'],

        # Cache
        cache_dir=yaml_config['output']['checkpoint_dir'],
        cache_size_gb=yaml_config['cache']['max_size_mb'] / 1024.0,  # Convert MB to GB

        # Logging
        enable_tensorboard=yaml_config['logging']['tensorboard'],
        tensorboard_dir=yaml_config['logging']['tensorboard_dir'],
        log_frequency=yaml_config['logging']['log_frequency']
    )

    return config


def create_dataset(yaml_config):
    """Create dataset based on config"""
    dataset_name = yaml_config['dataset']['name']
    dataset_size = yaml_config['dataset']['size']

    if dataset_name == "datacomp":
        # For now, use synthetic data
        # TODO: Replace with actual DataComp dataset loading
        print(f"Creating synthetic dataset with {dataset_size} samples")
        dataset = FakeDataset(size=dataset_size)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented yet")

    return dataset


def setup_output_dirs(yaml_config):
    """Create output directories"""
    output_dir = Path(yaml_config['output']['results_dir'])
    checkpoint_dir = Path(yaml_config['output']['checkpoint_dir'])

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create experiment subdirectory with timestamp
    exp_name = yaml_config['experiment']['name']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = output_dir / f"{exp_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    return exp_dir


def main():
    parser = argparse.ArgumentParser(description="Train DataComp MODE model")
    parser.add_argument(
        "--config",
        type=str,
        default="datacomp_config.yaml",
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        help="Override config values (e.g., training.epochs=20 model.device=cpu)"
    )
    args = parser.parse_args()

    # Load config
    print(f"Loading config from: {args.config}")
    yaml_config = load_config(args.config)

    # Apply command-line overrides
    if args.override:
        for override in args.override:
            key_path, value = override.split("=")
            keys = key_path.split(".")

            # Navigate to the nested dict
            target = yaml_config
            for key in keys[:-1]:
                target = target[key]

            # Convert value to appropriate type
            try:
                # Try to evaluate as Python literal
                value = eval(value)
            except:
                # Keep as string if evaluation fails
                pass

            target[keys[-1]] = value
            print(f"Override: {key_path} = {value}")

    # Set random seed
    set_seed(yaml_config['experiment']['seed'])
    print(f"Random seed set to: {yaml_config['experiment']['seed']}")

    # Setup output directories
    exp_dir = setup_output_dirs(yaml_config)
    print(f"Experiment directory: {exp_dir}")

    # Save config to experiment directory
    config_save_path = exp_dir / "config.yaml"
    with open(config_save_path, 'w') as f:
        yaml.dump(yaml_config, f, default_flow_style=False, sort_keys=False)
    print(f"Config saved to: {config_save_path}")

    # Convert YAML config to DataCompMODEConfig
    config = yaml_to_datacomp_config(yaml_config)

    # Create dataset
    dataset = create_dataset(yaml_config)
    print(f"Dataset size: {len(dataset)}")

    # Create pipeline
    print("\nInitializing DataComp MODE pipeline...")
    pipeline = DataCompMODE(config)

    print("\n" + "="*80)
    print("DataComp MODE Pipeline Ready")
    print("="*80)
    print(f"Experiment: {yaml_config['experiment']['name']}")
    print(f"Description: {yaml_config['experiment']['description']}")
    print(f"Strategy: {config.selection_strategy}")
    print(f"Selection budget: {config.selection_budget * 100:.1f}%")
    print(f"Training epochs: {config.num_epochs}")
    print(f"Device: {config.device}")
    print("="*80)
    print()

    # Run training based on strategy
    strategy = yaml_config['selection']['strategy']

    if strategy == "one_shot":
        print("Running ONE-SHOT strategy...")
        pipeline.run_one_shot(dataset)
    elif strategy == "iterative":
        print("Running ITERATIVE strategy...")
        pipeline.run_iterative(dataset)
    elif strategy == "hybrid":
        print("Running HYBRID strategy...")
        # TODO: Implement hybrid strategy
        raise NotImplementedError("Hybrid strategy not implemented yet")
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    print("\nTraining complete!")
    print(f"Results saved to: {exp_dir}")

    # Save final results summary
    results = {
        'experiment': yaml_config['experiment']['name'],
        'config': yaml_config,
        'dataset_size': len(dataset),
        'selected_samples': int(len(dataset) * config.selection_budget),
        'training_epochs': config.num_epochs,
        'timestamp': datetime.now().isoformat()
    }

    results_path = exp_dir / "results_summary.yaml"
    with open(results_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)
    print(f"Results summary saved to: {results_path}")


if __name__ == "__main__":
    main()
