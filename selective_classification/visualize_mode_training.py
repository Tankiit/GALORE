#!/usr/bin/env python3
"""
Visualization script for MODE training results.

Creates publication-quality plots for:
1. Training convergence curves
2. Strategy weight evolution
3. Zero-shot evaluation comparison
4. Data efficiency plots

Usage:
    python visualize_mode_training.py --results_dir mode_hypernetwork_output
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import seaborn as sns

# Set publication-quality plot style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 11


def load_training_results(results_dir: Path) -> Dict:
    """Load training results from JSON"""
    results_path = results_dir / 'training_results.json'

    if not results_path.exists():
        raise FileNotFoundError(f"Results not found: {results_path}")

    with open(results_path, 'r') as f:
        results = json.load(f)

    return results


def plot_convergence_curves(results: Dict, output_dir: Path):
    """
    Plot training convergence curves.

    Shows:
    - Training loss over epochs
    - Zero-shot ImageNet accuracy
    - COCO retrieval metrics
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Training Loss
    if 'loss' in results:
        ax = axes[0, 0]
        epochs = range(1, len(results['loss']) + 1)
        ax.plot(epochs, results['loss'], linewidth=2, marker='o', label='Training Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Convergence')
        ax.grid(True, alpha=0.3)
        ax.legend()

    # Plot 2: ImageNet Zero-Shot Accuracy
    if 'top1_accuracy' in results:
        ax = axes[0, 1]
        epochs = range(1, len(results['top1_accuracy']) + 1)
        ax.plot(epochs, results['top1_accuracy'], linewidth=2, marker='s',
               label='Top-1 Accuracy', color='green')
        if 'top5_accuracy' in results:
            ax.plot(epochs, results['top5_accuracy'], linewidth=2, marker='^',
                   label='Top-5 Accuracy', color='blue')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('ImageNet Zero-Shot Classification')
        ax.grid(True, alpha=0.3)
        ax.legend()

    # Plot 3: COCO Image-to-Text Retrieval
    if 'I2T_R@1' in results:
        ax = axes[1, 0]
        epochs = range(1, len(results['I2T_R@1']) + 1)
        ax.plot(epochs, results['I2T_R@1'], linewidth=2, marker='o',
               label='R@1', color='purple')
        if 'I2T_R@5' in results:
            ax.plot(epochs, results['I2T_R@5'], linewidth=2, marker='s',
                   label='R@5', color='orange')
        if 'I2T_R@10' in results:
            ax.plot(epochs, results['I2T_R@10'], linewidth=2, marker='^',
                   label='R@10', color='red')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Recall (%)')
        ax.set_title('COCO Image-to-Text Retrieval')
        ax.grid(True, alpha=0.3)
        ax.legend()

    # Plot 4: COCO Text-to-Image Retrieval
    if 'T2I_R@1' in results:
        ax = axes[1, 1]
        epochs = range(1, len(results['T2I_R@1']) + 1)
        ax.plot(epochs, results['T2I_R@1'], linewidth=2, marker='o',
               label='R@1', color='teal')
        if 'T2I_R@5' in results:
            ax.plot(epochs, results['T2I_R@5'], linewidth=2, marker='s',
                   label='R@5', color='brown')
        if 'T2I_R@10' in results:
            ax.plot(epochs, results['T2I_R@10'], linewidth=2, marker='^',
                   label='R@10', color='pink')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Recall (%)')
        ax.set_title('COCO Text-to-Image Retrieval')
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    output_path = output_dir / 'convergence_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved convergence curves: {output_path}")
    plt.close()


def plot_strategy_evolution(strategy_weights_history: List[Dict], output_dir: Path):
    """
    Plot evolution of MODE strategy weights over training.

    Shows how the hypernetwork adapts its selection strategy.
    """
    if not strategy_weights_history:
        print("No strategy weights found to plot")
        return

    # Extract strategy names and weights over epochs
    strategy_names = list(strategy_weights_history[0].keys())
    num_epochs = len(strategy_weights_history)

    # Create matrix: [num_strategies, num_epochs]
    weights_matrix = np.zeros((len(strategy_names), num_epochs))

    for epoch_idx, weights in enumerate(strategy_weights_history):
        for strategy_idx, name in enumerate(strategy_names):
            weights_matrix[strategy_idx, epoch_idx] = weights[name]

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Line plot of strategy weights
    ax = axes[0]
    epochs = range(1, num_epochs + 1)

    for idx, name in enumerate(strategy_names):
        ax.plot(epochs, weights_matrix[idx], linewidth=2, marker='o',
               label=name.replace('_', ' ').title())

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Strategy Weight')
    ax.set_title('MODE Strategy Weight Evolution')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', ncol=2)

    # Plot 2: Heatmap of strategy weights
    ax = axes[1]
    im = ax.imshow(weights_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')

    # Set ticks and labels
    ax.set_xticks(range(num_epochs))
    ax.set_xticklabels(range(1, num_epochs + 1))
    ax.set_yticks(range(len(strategy_names)))
    ax.set_yticklabels([name.replace('_', ' ').title() for name in strategy_names])

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Strategy')
    ax.set_title('Strategy Weight Heatmap')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Weight', rotation=270, labelpad=20)

    plt.tight_layout()
    output_path = output_dir / 'strategy_evolution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved strategy evolution: {output_path}")
    plt.close()


def plot_data_efficiency_comparison(results_dict: Dict[str, Dict], output_dir: Path):
    """
    Compare data efficiency across different methods.

    Args:
        results_dict: Dict mapping method names to their results
                     e.g., {'MODE-30%': results1, 'Random-30%': results2, 'Full': results3}
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Final ImageNet accuracy comparison
    ax = axes[0]
    methods = list(results_dict.keys())
    accuracies = [results_dict[m]['top1_accuracy'][-1] for m in methods if 'top1_accuracy' in results_dict[m]]

    if accuracies:
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        bars = ax.bar(range(len(methods)), accuracies, color=colors[:len(methods)])

        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_ylabel('ImageNet Top-1 Accuracy (%)')
        ax.set_title('Zero-Shot Classification Performance')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')

    # Plot 2: COCO retrieval comparison
    ax = axes[1]
    if all('I2T_R@1' in results_dict[m] for m in methods):
        x = np.arange(len(methods))
        width = 0.35

        i2t_r1 = [results_dict[m]['I2T_R@1'][-1] for m in methods]
        t2i_r1 = [results_dict[m]['T2I_R@1'][-1] for m in methods]

        bars1 = ax.bar(x - width/2, i2t_r1, width, label='Image→Text R@1', color='#FF6B6B')
        bars2 = ax.bar(x + width/2, t2i_r1, width, label='Text→Image R@1', color='#4ECDC4')

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_ylabel('Recall@1 (%)')
        ax.set_title('COCO Retrieval Performance')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / 'data_efficiency_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved data efficiency comparison: {output_path}")
    plt.close()


def plot_convergence_speed_comparison(results_dict: Dict[str, Dict], output_dir: Path,
                                     target_accuracy: float = 40.0):
    """
    Compare convergence speed across methods.

    Shows steps/epochs needed to reach target accuracy.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for method_name, results in results_dict.items():
        if 'top1_accuracy' not in results:
            continue

        accuracies = results['top1_accuracy']
        epochs = range(1, len(accuracies) + 1)

        ax.plot(epochs, accuracies, linewidth=2, marker='o', label=method_name)

        # Mark where target accuracy is reached
        try:
            target_epoch = next(i for i, acc in enumerate(accuracies) if acc >= target_accuracy)
            ax.axvline(target_epoch + 1, linestyle='--', alpha=0.5)
            ax.text(target_epoch + 1, target_accuracy - 2,
                   f'{method_name}\n{target_epoch + 1} epochs',
                   rotation=90, va='top', fontsize=9)
        except StopIteration:
            pass

    # Add target line
    ax.axhline(target_accuracy, color='red', linestyle=':', linewidth=2,
              label=f'Target ({target_accuracy}%)', alpha=0.7)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('ImageNet Top-1 Accuracy (%)')
    ax.set_title(f'Convergence Speed Comparison (Target: {target_accuracy}%)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    plt.tight_layout()
    output_path = output_dir / 'convergence_speed.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved convergence speed comparison: {output_path}")
    plt.close()


def generate_results_table(results_dict: Dict[str, Dict], output_dir: Path):
    """Generate LaTeX table of results for paper"""

    table_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Zero-shot evaluation results on ImageNet and COCO}",
        "\\label{tab:main_results}",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Method & ImageNet & ImageNet & COCO & COCO & Speedup \\\\",
        "       & Top-1 & Top-5 & I2T R@1 & T2I R@1 & \\\\",
        "\\midrule"
    ]

    for method_name, results in results_dict.items():
        img_top1 = results.get('top1_accuracy', [0])[-1] if 'top1_accuracy' in results else 0
        img_top5 = results.get('top5_accuracy', [0])[-1] if 'top5_accuracy' in results else 0
        i2t_r1 = results.get('I2T_R@1', [0])[-1] if 'I2T_R@1' in results else 0
        t2i_r1 = results.get('T2I_R@1', [0])[-1] if 'T2I_R@1' in results else 0

        speedup = "1.0×"  # Placeholder - compute from actual data

        line = f"{method_name} & {img_top1:.1f} & {img_top5:.1f} & {i2t_r1:.1f} & {t2i_r1:.1f} & {speedup} \\\\"
        table_lines.append(line)

    table_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])

    # Save to file
    table_path = output_dir / 'results_table.tex'
    with open(table_path, 'w') as f:
        f.write('\n'.join(table_lines))

    print(f"Saved LaTeX table: {table_path}")

    # Also print to console
    print("\n" + "="*80)
    print("RESULTS TABLE")
    print("="*80)
    print(f"{'Method':<20} | {'ImageNet':<15} | {'COCO I2T':<10} | {'COCO T2I':<10}")
    print(f"{'':20} | {'Top-1':>7} {'Top-5':>7} | {'R@1':>10} | {'R@1':>10}")
    print("-"*80)

    for method_name, results in results_dict.items():
        img_top1 = results.get('top1_accuracy', [0])[-1] if 'top1_accuracy' in results else 0
        img_top5 = results.get('top5_accuracy', [0])[-1] if 'top5_accuracy' in results else 0
        i2t_r1 = results.get('I2T_R@1', [0])[-1] if 'I2T_R@1' in results else 0
        t2i_r1 = results.get('T2I_R@1', [0])[-1] if 'T2I_R@1' in results else 0

        print(f"{method_name:<20} | {img_top1:>7.1f} {img_top5:>7.1f} | {i2t_r1:>10.1f} | {t2i_r1:>10.1f}")

    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Visualize MODE training results")
    parser.add_argument('--results_dir', type=str, default='mode_hypernetwork_output',
                       help='Directory containing training results')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for plots (default: same as results_dir)')
    parser.add_argument('--compare_methods', type=str, nargs='+',
                       help='Directories of other methods to compare (for data efficiency plots)')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir

    print(f"\nLoading results from: {results_dir}")

    # Load main results
    try:
        results = load_training_results(results_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you've run training first!")
        return

    # Create plots
    print("\nGenerating visualizations...")

    # 1. Convergence curves
    plot_convergence_curves(results, output_dir)

    # 2. Strategy evolution (if available)
    if 'strategy_weights' in results:
        plot_strategy_evolution(results['strategy_weights'], output_dir)

    # 3. Comparison plots (if other methods specified)
    if args.compare_methods:
        results_dict = {'MODE': results}

        for method_dir in args.compare_methods:
            method_path = Path(method_dir)
            method_name = method_path.name
            try:
                method_results = load_training_results(method_path)
                results_dict[method_name] = method_results
            except FileNotFoundError:
                print(f"Warning: Could not load results from {method_dir}")

        if len(results_dict) > 1:
            plot_data_efficiency_comparison(results_dict, output_dir)
            plot_convergence_speed_comparison(results_dict, output_dir)
            generate_results_table(results_dict, output_dir)

    print(f"\n✅ All visualizations saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - convergence_curves.png")
    print("  - strategy_evolution.png (if MODE used)")
    print("  - data_efficiency_comparison.png (if comparison methods provided)")
    print("  - convergence_speed.png (if comparison methods provided)")
    print("  - results_table.tex (LaTeX table for paper)")


if __name__ == '__main__':
    main()
