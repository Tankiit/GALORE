#!/usr/bin/env python3
"""
ResNet18 Multi-Configuration Experiments with MASCS

This script runs comprehensive experiments with ResNet18 across multiple configurations:
- Different datasets (CIFAR-10, CIFAR-100, MNIST, FashionMNIST, SVHN)
- Different budget percentages (5%, 10%, 20%, 30%, 50%, 70%, 100%)
- Multiple sequential runs for statistical significance
- Comprehensive logging and analysis

Usage:
    python resnet18_multi_config_experiments.py --epochs 20 --num_sequential_runs 3
    python resnet18_multi_config_experiments.py --quick_test  # For testing with reduced settings
"""

import argparse
import json
import sys
from pathlib import Path
from mascs_multi_models import ExperimentManager, ModelManager, DatasetManager
import numpy as np

class ResNet18ExperimentSuite:
    """Comprehensive experiment suite for ResNet18 across multiple configurations"""

    def __init__(self, log_dir='./experiments_resnet18', quick_test=False):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.quick_test = quick_test
        self.exp_manager = ExperimentManager(log_dir)

        # Define experiment configurations
        self.datasets = ['cifar10', 'cifar100', 'mnist', 'fashionmnist', 'svhn']
        self.budget_percentages = [5, 10, 20, 30, 50, 70, 100]  # Full range of budgets

        if quick_test:
            self.datasets = ['cifar10', 'mnist']  # Reduced for testing
            self.budget_percentages = [10, 30, 100]  # Reduced for testing
            print("Running in QUICK TEST mode with reduced configurations")

    def generate_all_configs(self, base_config):
        """Generate all ResNet18 configurations"""
        configs = []

        for dataset in self.datasets:
            for budget_percentage in self.budget_percentages:
                config = base_config.copy()
                config.update({
                    'model': 'resnet18',
                    'dataset': dataset,
                    'budget_percentage': budget_percentage,
                    'experiment_type': 'resnet18_multi_config'
                })
                configs.append(config)

        return configs

    def run_comprehensive_experiments(self, base_config, num_sequential_runs=3):
        """Run the complete experiment suite"""
        print("="*80)
        print("RESNET18 COMPREHENSIVE EXPERIMENT SUITE")
        print("="*80)
        print(f"Datasets: {self.datasets}")
        print(f"Budget percentages: {self.budget_percentages}")
        print(f"Sequential runs per config: {num_sequential_runs}")
        print(f"Total experiments: {len(self.datasets) * len(self.budget_percentages) * num_sequential_runs}")
        print("="*80)

        # Generate all configurations
        configs = self.generate_all_configs(base_config)

        print(f"\nGenerated {len(configs)} configurations")
        print("Starting experiments...")

        # Run all experiments
        results = self.exp_manager.run_multiple_experiments(configs, num_sequential_runs)

        # Analyze and save results
        self.analyze_results(results, configs)

        return results

    def analyze_results(self, results, configs):
        """Comprehensive analysis of experiment results"""
        print("\n" + "="*80)
        print("EXPERIMENT ANALYSIS")
        print("="*80)

        # Filter successful results
        successful_results = [r for r in results if 'best_val_accuracy' in r]

        if not successful_results:
            print("No successful experiments found!")
            return

        # Group results by dataset and budget
        analysis = {}

        for result in successful_results:
            config = result['config']
            dataset = config['dataset']
            budget = config['budget_percentage']
            accuracy = result['best_val_accuracy']

            key = f"{dataset}_{budget}%"
            if key not in analysis:
                analysis[key] = []
            analysis[key].append(accuracy)

        # Calculate statistics for each configuration
        stats = []
        for key, accuracies in analysis.items():
            dataset, budget_str = key.split('_')
            budget = float(budget_str.rstrip('%'))

            stats.append({
                'dataset': dataset,
                'budget_percentage': budget,
                'num_runs': len(accuracies),
                'mean_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies),
                'min_accuracy': np.min(accuracies),
                'max_accuracy': np.max(accuracies),
                'median_accuracy': np.median(accuracies)
            })

        # Sort by dataset and budget
        stats.sort(key=lambda x: (x['dataset'], x['budget_percentage']))

        # Print detailed results
        print("\nDETAILED RESULTS:")
        print("-" * 100)
        print(f"{'Dataset':<12} {'Budget%':<8} {'Runs':<5} {'Mean±Std':<15} {'Median':<8} {'Min':<8} {'Max':<8}")
        print("-" * 100)

        for stat in stats:
            print(f"{stat['dataset']:<12} {stat['budget_percentage']:<8.0f} "
                  f"{stat['num_runs']:<5} {stat['mean_accuracy']:<7.2f}±{stat['std_accuracy']:<5.2f} "
                  f"{stat['median_accuracy']:<8.2f} {stat['min_accuracy']:<8.2f} {stat['max_accuracy']:<8.2f}")

        # Dataset-wise analysis
        print("\n" + "="*60)
        print("DATASET-WISE ANALYSIS")
        print("="*60)

        dataset_analysis = {}
        for stat in stats:
            dataset = stat['dataset']
            if dataset not in dataset_analysis:
                dataset_analysis[dataset] = []
            dataset_analysis[dataset].append(stat)

        for dataset, dataset_stats in dataset_analysis.items():
            print(f"\n{dataset.upper()}:")
            print(f"  Number of budget configurations: {len(dataset_stats)}")

            # Find best and worst budget configurations
            best_config = max(dataset_stats, key=lambda x: x['mean_accuracy'])
            worst_config = min(dataset_stats, key=lambda x: x['mean_accuracy'])

            print(f"  Best configuration: {best_config['budget_percentage']:.0f}% budget "
                  f"({best_config['mean_accuracy']:.2f}% accuracy)")
            print(f"  Worst configuration: {worst_config['budget_percentage']:.0f}% budget "
                  f"({worst_config['mean_accuracy']:.2f}% accuracy)")

            # Analyze efficiency (accuracy per % of data used)
            efficiency_scores = []
            for stat in dataset_stats:
                if stat['budget_percentage'] > 0:
                    efficiency = stat['mean_accuracy'] / stat['budget_percentage']
                    efficiency_scores.append((stat['budget_percentage'], efficiency))

            if efficiency_scores:
                best_efficiency = max(efficiency_scores, key=lambda x: x[1])
                print(f"  Most efficient: {best_efficiency[0]:.0f}% budget "
                      f"(efficiency: {best_efficiency[1]:.3f} acc/% data)")

        # Budget efficiency analysis across all datasets
        print("\n" + "="*60)
        print("BUDGET EFFICIENCY ANALYSIS")
        print("="*60)

        budget_analysis = {}
        for stat in stats:
            budget = stat['budget_percentage']
            if budget not in budget_analysis:
                budget_analysis[budget] = []
            budget_analysis[budget].append(stat['mean_accuracy'])

        print(f"{'Budget%':<8} {'Datasets':<8} {'Mean Acc':<10} {'Std Acc':<8} {'Efficiency':<10}")
        print("-" * 50)

        for budget in sorted(budget_analysis.keys()):
            accuracies = budget_analysis[budget]
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            efficiency = mean_acc / budget if budget > 0 else 0

            print(f"{budget:<8.0f} {len(accuracies):<8} {mean_acc:<10.2f} {std_acc:<8.2f} {efficiency:<10.3f}")

        # Save analysis to files
        self.save_analysis_results(stats, successful_results)

        # Generate recommendations
        self.generate_recommendations(stats, dataset_analysis, budget_analysis)

    def save_analysis_results(self, stats, results):
        """Save analysis results to files"""
        # Save detailed statistics
        import pandas as pd

        stats_df = pd.DataFrame(stats)
        stats_df.to_csv(self.log_dir / 'resnet18_analysis_stats.csv', index=False)

        # Save all results
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.log_dir / 'resnet18_all_results.csv', index=False)

        # Save analysis summary
        summary = {
            'total_experiments': len(results),
            'successful_experiments': len([r for r in results if 'best_val_accuracy' in r]),
            'datasets_tested': list(set([r['config']['dataset'] for r in results if 'config' in r])),
            'budget_percentages_tested': list(set([r['config']['budget_percentage'] for r in results if 'config' in r])),
            'overall_best_accuracy': max([r['best_val_accuracy'] for r in results if 'best_val_accuracy' in r]),
            'overall_mean_accuracy': np.mean([r['best_val_accuracy'] for r in results if 'best_val_accuracy' in r])
        }

        with open(self.log_dir / 'resnet18_experiment_summary.json', 'w') as f:
            json.dump(summary, f, indent=4)

        print(f"\nAnalysis results saved to {self.log_dir}/")
        print(f"  - resnet18_analysis_stats.csv: Detailed statistics")
        print(f"  - resnet18_all_results.csv: All experiment results")
        print(f"  - resnet18_experiment_summary.json: Summary statistics")

    def generate_recommendations(self, stats, dataset_analysis, budget_analysis):
        """Generate recommendations based on experiment results"""
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)

        # Find overall best configurations
        best_overall = max(stats, key=lambda x: x['mean_accuracy'])
        print(f"1. BEST OVERALL CONFIGURATION:")
        print(f"   Dataset: {best_overall['dataset']}")
        print(f"   Budget: {best_overall['budget_percentage']:.0f}%")
        print(f"   Accuracy: {best_overall['mean_accuracy']:.2f}% ± {best_overall['std_accuracy']:.2f}%")

        # Find most efficient configurations (best accuracy per % of data)
        efficiency_scores = []
        for stat in stats:
            if stat['budget_percentage'] > 0:
                efficiency = stat['mean_accuracy'] / stat['budget_percentage']
                efficiency_scores.append((stat, efficiency))

        best_efficiency = max(efficiency_scores, key=lambda x: x[1])
        print(f"\n2. MOST EFFICIENT CONFIGURATION:")
        print(f"   Dataset: {best_efficiency[0]['dataset']}")
        print(f"   Budget: {best_efficiency[0]['budget_percentage']:.0f}%")
        print(f"   Accuracy: {best_efficiency[0]['mean_accuracy']:.2f}%")
        print(f"   Efficiency: {best_efficiency[1]:.3f} accuracy per % of data")

        # Dataset-specific recommendations
        print(f"\n3. DATASET-SPECIFIC RECOMMENDATIONS:")
        for dataset, dataset_stats in dataset_analysis.items():
            best_for_dataset = max(dataset_stats, key=lambda x: x['mean_accuracy'])
            most_efficient_for_dataset = max(dataset_stats,
                                           key=lambda x: x['mean_accuracy']/x['budget_percentage'] if x['budget_percentage'] > 0 else 0)

            print(f"   {dataset.upper()}:")
            print(f"     Best accuracy: {best_for_dataset['budget_percentage']:.0f}% budget "
                  f"({best_for_dataset['mean_accuracy']:.2f}%)")
            print(f"     Most efficient: {most_efficient_for_dataset['budget_percentage']:.0f}% budget "
                  f"({most_efficient_for_dataset['mean_accuracy']:.2f}%)")

        # Budget recommendations
        print(f"\n4. BUDGET RECOMMENDATIONS:")
        budget_efficiency = []
        for budget, accuracies in budget_analysis.items():
            if budget > 0:
                mean_acc = np.mean(accuracies)
                efficiency = mean_acc / budget
                budget_efficiency.append((budget, mean_acc, efficiency))

        # Sort by efficiency
        budget_efficiency.sort(key=lambda x: x[2], reverse=True)

        print("   Top 3 most efficient budget levels across all datasets:")
        for i, (budget, mean_acc, efficiency) in enumerate(budget_efficiency[:3]):
            print(f"     {i+1}. {budget:.0f}% budget: {mean_acc:.2f}% accuracy "
                  f"(efficiency: {efficiency:.3f})")

        # Save recommendations
        recommendations = {
            'best_overall': {
                'dataset': best_overall['dataset'],
                'budget_percentage': best_overall['budget_percentage'],
                'accuracy': best_overall['mean_accuracy'],
                'std': best_overall['std_accuracy']
            },
            'most_efficient': {
                'dataset': best_efficiency[0]['dataset'],
                'budget_percentage': best_efficiency[0]['budget_percentage'],
                'accuracy': best_efficiency[0]['mean_accuracy'],
                'efficiency_score': best_efficiency[1]
            },
            'dataset_specific_best': {
                dataset: {
                    'budget_percentage': max(stats, key=lambda x: x['mean_accuracy'])['budget_percentage'],
                    'accuracy': max(stats, key=lambda x: x['mean_accuracy'])['mean_accuracy']
                } for dataset, stats in dataset_analysis.items()
            },
            'top_efficient_budgets': [
                {
                    'budget_percentage': budget,
                    'mean_accuracy': mean_acc,
                    'efficiency_score': efficiency
                } for budget, mean_acc, efficiency in budget_efficiency[:3]
            ]
        }

        with open(self.log_dir / 'resnet18_recommendations.json', 'w') as f:
            json.dump(recommendations, f, indent=4)

        print(f"\nRecommendations saved to {self.log_dir}/resnet18_recommendations.json")

def main():
    parser = argparse.ArgumentParser(description='ResNet18 Multi-Configuration Experiments with MASCS')

    # Experiment configuration
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_sequential_runs', type=int, default=3, help='Number of sequential runs per configuration')

    # Data and logging
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--log_dir', type=str, default='./experiments_resnet18', help='Experiment log directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')

    # MASCS configuration
    parser.add_argument('--use_gradient_cache', action='store_true', default=True, help='Enable gradient caching')
    parser.add_argument('--gradient_batch_accumulation', type=int, default=8, help='Gradient batch accumulation')

    # Testing and debugging
    parser.add_argument('--quick_test', action='store_true', help='Run with reduced configurations for testing')
    parser.add_argument('--dry_run', action='store_true', help='Show what would be run without actually running')

    # Analysis only
    parser.add_argument('--analyze_only', action='store_true', help='Only run analysis on existing results')
    parser.add_argument('--existing_results_dir', type=str, help='Directory with existing results to analyze')

    args = parser.parse_args()

    # Create experiment suite
    exp_suite = ResNet18ExperimentSuite(log_dir=args.log_dir, quick_test=args.quick_test)

    if args.analyze_only:
        print("Analysis-only mode - not implemented yet")
        print("Please run experiments first, then analysis will be performed automatically")
        return

    # Base configuration for all experiments
    base_config = {
        'batch_size': args.batch_size,
        'lr': args.lr,
        'epochs': args.epochs,
        'data_dir': args.data_dir,
        'device': args.device,
        'use_gradient_cache': args.use_gradient_cache,
        'gradient_batch_accumulation': args.gradient_batch_accumulation
    }

    if args.dry_run:
        configs = exp_suite.generate_all_configs(base_config)
        print(f"DRY RUN: Would run {len(configs)} configurations with {args.num_sequential_runs} runs each")
        print(f"Total experiments: {len(configs) * args.num_sequential_runs}")
        print("\nConfigurations:")
        for i, config in enumerate(configs[:5]):  # Show first 5
            print(f"  {i+1}. {config['model']} on {config['dataset']} with {config['budget_percentage']}% budget")
        if len(configs) > 5:
            print(f"  ... and {len(configs) - 5} more configurations")
        return

    # Run the comprehensive experiment suite
    try:
        results = exp_suite.run_comprehensive_experiments(base_config, args.num_sequential_runs)

        print("\n" + "="*80)
        print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Total experiments run: {len(results)}")
        print(f"Check {args.log_dir}/ for detailed results and analysis")

    except KeyboardInterrupt:
        print("\nExperiments interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during experiments: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()