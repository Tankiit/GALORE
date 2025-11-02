#!/usr/bin/env python3
"""
Launcher script for ResNet18 experiments with different modes

Usage Examples:
    # Quick test with reduced settings
    python run_resnet18_experiments.py --quick_test

    # Full comprehensive experiments (warning: takes many hours)
    python run_resnet18_experiments.py --full_run --epochs 50 --num_runs 5

    # Custom configuration
    python run_resnet18_experiments.py --datasets cifar10 cifar100 --budgets 10 30 50 --epochs 20

    # Use pre-defined config file
    python run_resnet18_experiments.py --config resnet18_config_test.json --num_runs 2

    # Dry run to see what would be executed
    python run_resnet18_experiments.py --full_run --dry_run
"""

import argparse
import subprocess
import sys
from pathlib import Path

def run_quick_test():
    """Run a quick test with minimal configurations"""
    cmd = [
        sys.executable, "resnet18_multi_config_experiments.py",
        "--quick_test",
        "--epochs", "5",
        "--num_sequential_runs", "2",
        "--batch_size", "64"
    ]

    print("Running ResNet18 Quick Test...")
    print("This will test ResNet18 on CIFAR-10 and MNIST with a few budget configurations")
    print("Command:", " ".join(cmd))

    return subprocess.run(cmd)

def run_full_experiments(args):
    """Run full comprehensive experiments"""
    cmd = [
        sys.executable, "resnet18_multi_config_experiments.py",
        "--epochs", str(args.epochs),
        "--num_sequential_runs", str(args.num_runs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr)
    ]

    if args.dry_run:
        cmd.append("--dry_run")

    print("Running ResNet18 Comprehensive Experiments...")
    print("This will test ResNet18 on multiple datasets with various budget configurations")
    print("WARNING: This may take several hours to complete!")
    print("Command:", " ".join(cmd))

    if not args.dry_run:
        confirm = input("Continue? (y/N): ").lower().strip()
        if confirm != 'y':
            print("Aborted by user")
            return None

    return subprocess.run(cmd)

def run_custom_experiments(args):
    """Run experiments with custom configurations"""
    cmd = [
        sys.executable, "mascs_multi_models.py",
        "--run_multiple",
        "--model", "resnet18",
        "--epochs", str(args.epochs),
        "--num_sequential_runs", str(args.num_runs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr)
    ]

    if args.datasets:
        cmd.extend(["--datasets"] + args.datasets)

    if args.budgets:
        cmd.extend(["--percentages"] + [str(b) for b in args.budgets])

    print("Running Custom ResNet18 Experiments...")
    print("Command:", " ".join(cmd))

    return subprocess.run(cmd)

def run_config_file_experiments(args):
    """Run experiments from a configuration file"""
    config_file = Path(args.config)
    if not config_file.exists():
        print(f"Configuration file {config_file} not found!")
        return None

    cmd = [
        sys.executable, "mascs_multi_models.py",
        "--run_multiple",
        "--experiment_config", str(config_file),
        "--num_sequential_runs", str(args.num_runs)
    ]

    print(f"Running ResNet18 Experiments from config file: {config_file}")
    print("Command:", " ".join(cmd))

    return subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description='ResNet18 Experiment Launcher')

    # Experiment modes
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--quick_test', action='store_true', help='Run quick test with minimal configurations')
    mode_group.add_argument('--full_run', action='store_true', help='Run full comprehensive experiments')
    mode_group.add_argument('--custom', action='store_true', help='Run custom experiments with specified parameters')
    mode_group.add_argument('--config', type=str, help='Run experiments from JSON configuration file')

    # Experiment parameters
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--num_runs', type=int, default=3, help='Number of sequential runs per configuration')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')

    # Custom mode parameters
    parser.add_argument('--datasets', nargs='+', type=str,
                       choices=['cifar10', 'cifar100', 'mnist', 'fashionmnist', 'svhn'],
                       help='Datasets to use (for custom mode)')
    parser.add_argument('--budgets', nargs='+', type=float,
                       help='Budget percentages to use (for custom mode)')

    # Utility options
    parser.add_argument('--dry_run', action='store_true', help='Show what would be run without executing')

    args = parser.parse_args()

    # Check if the required files exist
    required_files = ['mascs_multi_models.py', 'resnet18_multi_config_experiments.py']
    missing_files = [f for f in required_files if not Path(f).exists()]

    if missing_files:
        print("Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease ensure all files are in the current directory.")
        return 1

    # Run the appropriate experiment mode
    try:
        if args.quick_test:
            result = run_quick_test()
        elif args.full_run:
            result = run_full_experiments(args)
        elif args.custom:
            if not args.datasets or not args.budgets:
                print("Custom mode requires --datasets and --budgets arguments")
                return 1
            result = run_custom_experiments(args)
        elif args.config:
            result = run_config_file_experiments(args)

        if result is not None:
            return result.returncode
        else:
            return 1

    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
        return 1
    except Exception as e:
        print(f"Error running experiments: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())