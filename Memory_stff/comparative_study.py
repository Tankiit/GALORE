#!/usr/bin/env python3
"""
Small-Scale Comparative Study: MASCS vs Baseline Methods
Shows clear advantages of MASCS with detailed reward/performance tracking per epoch
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import argparse
from datetime import datetime
import pandas as pd

# Import MASCS components
from mascs_mdp import (
    MemoryAugmentedCoresetSelector, 
    create_model_and_transforms,
    get_dataset_loaders
)

class BaselineSelector:
    """Simple baseline coreset selectors for comparison"""
    
    def __init__(self, dataset, budget):
        self.dataset = dataset
        self.budget = budget
        self.name = "baseline"
    
    def select_random(self):
        """Random selection baseline"""
        indices = np.random.choice(len(self.dataset), self.budget, replace=False)
        return indices, None, None
    
    def select_uncertainty(self, model, dataloader):
        """Uncertainty-based selection (simple entropy)"""
        model.eval()
        uncertainties = []
        all_indices = []
        
        with torch.no_grad():
            for batch_idx, (data, targets, indices) in enumerate(dataloader):
                outputs = model(data)
                probs = torch.softmax(outputs, dim=1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                
                uncertainties.extend(entropy.cpu().numpy())
                all_indices.extend(indices.numpy())
        
        # Select top uncertain samples
        uncertainty_indices = np.argsort(uncertainties)[-self.budget:]
        selected = np.array(all_indices)[uncertainty_indices]
        
        return selected, uncertainties, None

def run_experiment(method, dataset_name, architecture, budget, epochs, device='mps'):
    """Run single experiment with detailed tracking"""
    print(f"\n🧪 Running {method} on {dataset_name} ({architecture}, budget={budget}, epochs={epochs})")
    
    # Setup
    results = {
        'method': method,
        'dataset': dataset_name, 
        'architecture': architecture,
        'budget': budget,
        'epochs': epochs,
        'epoch_data': []
    }
    
    # Get data and model
    train_loader, test_loader, num_classes = get_dataset_loaders(
        dataset_name, batch_size=64, data_percentage=100, 
        data_path="/Users/tanmoy/research/data"
    )
    
    model, transform = create_model_and_transforms(architecture, num_classes)
    model = model.to(device)
    
    # Initialize method
    if method == 'MASCS':
        selector = MemoryAugmentedCoresetSelector(
            dataset=train_loader.dataset,
            budget=budget,
            feature_extractor=model,
            num_classes=num_classes,
            device=device
        )
        current_coreset = np.random.choice(len(train_loader.dataset), budget, replace=False)
    elif method == 'Random':
        baseline = BaselineSelector(train_loader.dataset, budget)
        current_coreset, _, _ = baseline.select_random()
    elif method == 'Uncertainty':
        baseline = BaselineSelector(train_loader.dataset, budget)
        current_coreset, _, _ = baseline.select_uncertainty(model, train_loader)
    
    # Training loop with detailed tracking
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        print(f"  Epoch {epoch+1}/{epochs}...")
        
        # Create coreset dataloader
        coreset_dataset = torch.utils.data.Subset(train_loader.dataset, current_coreset)
        coreset_loader = torch.utils.data.DataLoader(
            coreset_dataset, batch_size=64, shuffle=True
        )
        
        # Training phase
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(coreset_loader):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_acc = 100. * correct / total
        avg_train_loss = train_loss / len(coreset_loader)
        
        # Testing phase
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        test_acc = 100. * correct / total
        avg_test_loss = test_loss / len(test_loader)
        
        # Calculate reward/improvement
        if method == 'MASCS':
            # MASCS-specific reward calculation
            if epoch == 0:
                reward = test_acc  # Initial performance
                strategy_used = 'initial'
            else:
                prev_acc = results['epoch_data'][-1]['test_acc']
                reward = test_acc - prev_acc  # Improvement
                
                # Select new coreset using MASCS
                val_improvement = test_acc - prev_acc
                current_coreset, scores, all_scores = selector.select_coreset(
                    model, criterion, current_coreset, 'balance', val_improvement, coreset_loader
                )
                strategy_used = 'balance'  # Using balance strategy for consistency
        else:
            # For baselines, reward is just test accuracy
            reward = test_acc
            strategy_used = method.lower()
        
        # Store epoch data
        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_acc': train_acc,
            'test_loss': avg_test_loss,
            'test_acc': test_acc,
            'reward': reward,
            'strategy': strategy_used,
            'coreset_size': len(current_coreset)
        }
        
        results['epoch_data'].append(epoch_data)
        
        print(f"    Train: {train_acc:.2f}% | Test: {test_acc:.2f}% | Reward: {reward:.3f}")
    
    # Final results
    final_epoch = results['epoch_data'][-1]
    results['final_train_acc'] = final_epoch['train_acc']
    results['final_test_acc'] = final_epoch['test_acc']
    results['best_test_acc'] = max([ed['test_acc'] for ed in results['epoch_data']])
    results['avg_reward'] = np.mean([ed['reward'] for ed in results['epoch_data']])
    results['total_reward'] = sum([ed['reward'] for ed in results['epoch_data']])
    
    return results

def create_comparative_plots(all_results, output_dir):
    """Create comprehensive comparative visualizations"""
    print("\n📊 Creating comparative plots...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for plotting
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Test Accuracy Over Epochs
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('MASCS vs Baselines: Comprehensive Comparison', fontsize=16, fontweight='bold')
    
    # Group by dataset
    datasets = list(set([r['dataset'] for r in all_results]))
    methods = list(set([r['method'] for r in all_results]))
    
    # Plot 1: Test Accuracy Progression
    ax1 = axes[0, 0]
    for dataset in datasets:
        for method in methods:
            dataset_method_results = [r for r in all_results if r['dataset'] == dataset and r['method'] == method]
            if dataset_method_results:
                result = dataset_method_results[0]
                epochs = [ed['epoch'] for ed in result['epoch_data']]
                test_accs = [ed['test_acc'] for ed in result['epoch_data']]
                ax1.plot(epochs, test_accs, marker='o', linewidth=2, 
                        label=f"{method} ({dataset})", markersize=4)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('Test Accuracy Progression')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Reward Over Epochs  
    ax2 = axes[0, 1]
    for dataset in datasets:
        for method in methods:
            dataset_method_results = [r for r in all_results if r['dataset'] == dataset and r['method'] == method]
            if dataset_method_results:
                result = dataset_method_results[0]
                epochs = [ed['epoch'] for ed in result['epoch_data']]
                rewards = [ed['reward'] for ed in result['epoch_data']]
                ax2.plot(epochs, rewards, marker='s', linewidth=2, 
                        label=f"{method} ({dataset})", markersize=4)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Reward')
    ax2.set_title('Reward Progression (Higher = Better)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Final Performance Comparison
    ax3 = axes[1, 0]
    method_performance = defaultdict(list)
    method_labels = []
    
    for method in methods:
        method_results = [r for r in all_results if r['method'] == method]
        final_accs = [r['final_test_acc'] for r in method_results]
        method_performance[method] = final_accs
        method_labels.append(f"{method}\n(n={len(final_accs)})")
    
    box_data = [method_performance[method] for method in methods]
    box_plot = ax3.boxplot(box_data, labels=methods, patch_artist=True)
    
    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax3.set_ylabel('Final Test Accuracy (%)')
    ax3.set_title('Final Performance Distribution')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Total Reward Comparison
    ax4 = axes[1, 1]
    method_rewards = defaultdict(list)
    
    for method in methods:
        method_results = [r for r in all_results if r['method'] == method]
        total_rewards = [r['total_reward'] for r in method_results]
        method_rewards[method] = total_rewards
    
    box_data = [method_rewards[method] for method in methods]
    box_plot = ax4.boxplot(box_data, labels=methods, patch_artist=True)
    
    # Color the boxes
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax4.set_ylabel('Total Reward')
    ax4.set_title('Cumulative Reward Distribution')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mascs_comparative_study.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Performance Summary Table
    summary_data = []
    for method in methods:
        method_results = [r for r in all_results if r['method'] == method]
        if method_results:
            final_accs = [r['final_test_acc'] for r in method_results]
            total_rewards = [r['total_reward'] for r in method_results]
            
            summary_data.append({
                'Method': method,
                'Experiments': len(method_results),
                'Mean Final Accuracy': np.mean(final_accs),
                'Std Final Accuracy': np.std(final_accs),
                'Best Final Accuracy': np.max(final_accs),
                'Mean Total Reward': np.mean(total_rewards),
                'Std Total Reward': np.std(total_rewards)
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, 'comparative_summary.csv'), index=False)
    
    return summary_df

def main():
    parser = argparse.ArgumentParser(description='Small-Scale MASCS Comparative Study')
    parser.add_argument('--datasets', nargs='+', default=['CIFAR10', 'FashionMNIST'], 
                       help='Datasets to test')
    parser.add_argument('--methods', nargs='+', default=['MASCS', 'Random', 'Uncertainty'],
                       help='Methods to compare')
    parser.add_argument('--budget', type=int, default=1000,
                       help='Coreset budget')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--architecture', type=str, default='resnet18',
                       help='Model architecture')
    parser.add_argument('--device', type=str, default='mps',
                       help='Device to use')
    parser.add_argument('--output_dir', type=str, default='./comparative_study_results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    print("🔬 MASCS Comparative Study")
    print("=" * 50)
    print(f"Datasets: {args.datasets}")
    print(f"Methods: {args.methods}")
    print(f"Budget: {args.budget}")
    print(f"Epochs: {args.epochs}")
    print(f"Architecture: {args.architecture}")
    print(f"Device: {args.device}")
    
    # Run experiments
    all_results = []
    total_experiments = len(args.datasets) * len(args.methods)
    experiment_count = 0
    
    for dataset in args.datasets:
        for method in args.methods:
            experiment_count += 1
            print(f"\n[{experiment_count}/{total_experiments}] {method} on {dataset}")
            
            try:
                result = run_experiment(
                    method=method,
                    dataset_name=dataset,
                    architecture=args.architecture,
                    budget=args.budget,
                    epochs=args.epochs,
                    device=args.device
                )
                all_results.append(result)
                
                print(f"  ✅ Final Test Acc: {result['final_test_acc']:.2f}%")
                print(f"  ✅ Total Reward: {result['total_reward']:.3f}")
                
            except Exception as e:
                print(f"  ❌ Failed: {str(e)}")
                continue
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save results
    with open(os.path.join(args.output_dir, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create comparative analysis
    summary_df = create_comparative_plots(all_results, args.output_dir)
    
    # Print summary
    print("\n" + "=" * 60)
    print("🏆 COMPARATIVE STUDY RESULTS")
    print("=" * 60)
    print(summary_df.to_string(index=False, float_format='%.3f'))
    
    # Calculate MASCS advantage
    if 'MASCS' in args.methods and len(args.methods) > 1:
        mascs_results = [r for r in all_results if r['method'] == 'MASCS']
        baseline_results = [r for r in all_results if r['method'] != 'MASCS']
        
        if mascs_results and baseline_results:
            mascs_avg_acc = np.mean([r['final_test_acc'] for r in mascs_results])
            baseline_avg_acc = np.mean([r['final_test_acc'] for r in baseline_results])
            
            advantage = mascs_avg_acc - baseline_avg_acc
            print(f"\n💡 MASCS Advantage: +{advantage:.2f}% test accuracy over baselines")
            
            mascs_avg_reward = np.mean([r['total_reward'] for r in mascs_results])
            baseline_avg_reward = np.mean([r['total_reward'] for r in baseline_results])
            reward_advantage = mascs_avg_reward - baseline_avg_reward
            print(f"💡 MASCS Reward Advantage: +{reward_advantage:.3f} total reward over baselines")
    
    print(f"\n📁 Results saved to: {args.output_dir}")
    print("🎯 Key files:")
    print("   - mascs_comparative_study.png: Visual comparison")
    print("   - comparative_summary.csv: Performance summary")
    print("   - all_results.json: Detailed results")

if __name__ == '__main__':
    main()