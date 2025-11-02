#!/usr/bin/env python3
"""
Batch TensorBoard Log Analysis for MASCS
Automatically discovers and analyzes all TensorBoard log directories
"""

import os
import sys
import glob
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import the analysis modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from mascs_mdp import TensorBoardAnalyzer, IntermediateTracker
    print("✅ Successfully imported MASCS analysis modules")
except ImportError as e:
    print(f"❌ Error importing MASCS modules: {e}")
    sys.exit(1)

class BatchLogAnalyzer:
    """Batch analyzer for multiple TensorBoard log directories"""
    
    def __init__(self, base_log_dir: str, output_dir: str = './batch_analysis'):
        self.base_log_dir = base_log_dir
        self.output_dir = output_dir
        self.all_experiments = {}
        self.summary_data = []
        os.makedirs(output_dir, exist_ok=True)
        
    def discover_log_directories(self):
        """Automatically discover all TensorBoard log directories"""
        log_dirs = []
        
        # Look for directories that contain event files
        for root, dirs, files in os.walk(self.base_log_dir):
            # Check if directory contains TensorBoard event files
            if any(f.startswith('events.out.tfevents') for f in files):
                log_dirs.append(root)
        
        print(f"🔍 Discovered {len(log_dirs)} TensorBoard log directories:")
        for i, log_dir in enumerate(log_dirs, 1):
            rel_path = os.path.relpath(log_dir, self.base_log_dir)
            print(f"  {i:2d}. {rel_path}")
        
        return log_dirs
    
    def analyze_single_experiment(self, log_dir: str):
        """Analyze a single experiment log directory"""
        experiment_name = os.path.basename(log_dir)
        print(f"📊 Analyzing: {experiment_name}")
        
        try:
            # Create individual analyzer for this experiment
            exp_output_dir = os.path.join(self.output_dir, f"individual_{experiment_name}")
            analyzer = TensorBoardAnalyzer(os.path.dirname(log_dir), exp_output_dir)
            
            # Extract data from this specific experiment
            analyzer.data = {}
            analyzer.extract_tensorboard_data()
            
            # Filter to just this experiment
            if experiment_name in analyzer.data:
                experiment_data = {experiment_name: analyzer.data[experiment_name]}
                analyzer.data = experiment_data
                
                # Generate individual analysis
                analyzer.create_training_curves(save_plots=True)
                analyzer.create_strategy_analysis(save_plots=True)
                analyzer.create_score_distribution_plots(save_plots=True)
                summary_df = analyzer.export_summary_data(f'{experiment_name}_summary.csv')
                
                # Store data for batch analysis
                self.all_experiments[experiment_name] = analyzer.data[experiment_name]
                
                # Extract key metrics
                scalars = analyzer.data[experiment_name]['scalars']
                metrics = {'experiment': experiment_name}
                
                # Extract final values
                for metric_name, tb_name in [
                    ('final_train_loss', 'Training/Loss'),
                    ('final_train_acc', 'Training/Accuracy'),
                    ('final_test_loss', 'Testing/Loss'),
                    ('final_test_acc', 'Testing/Accuracy')
                ]:
                    if tb_name in scalars and scalars[tb_name]['values']:
                        metrics[metric_name] = scalars[tb_name]['values'][-1]
                    else:
                        metrics[metric_name] = None
                
                # Extract strategy metrics
                if 'Strategy/Reward' in scalars:
                    rewards = scalars['Strategy/Reward']['values']
                    metrics['avg_reward'] = sum(rewards) / len(rewards) if rewards else 0
                    metrics['max_reward'] = max(rewards) if rewards else 0
                    metrics['min_reward'] = min(rewards) if rewards else 0
                    metrics['reward_std'] = pd.Series(rewards).std() if len(rewards) > 1 else 0
                
                # Parse experiment details from name
                parts = experiment_name.replace('mascs_', '').split('_')
                if len(parts) >= 3:
                    metrics['dataset'] = parts[0]
                    metrics['architecture'] = parts[1]
                    metrics['data_percentage'] = float(parts[2].replace('%', '')) if '%' in parts[2] else None
                
                self.summary_data.append(metrics)
                print(f"  ✅ Completed analysis for {experiment_name}")
                
            else:
                print(f"  ⚠️  No data found for {experiment_name}")
                
        except Exception as e:
            print(f"  ❌ Error analyzing {experiment_name}: {e}")
    
    def create_batch_comparative_plots(self):
        """Create comprehensive comparative plots across all experiments"""
        if not self.summary_data:
            print("❌ No summary data available for batch analysis")
            return
        
        df = pd.DataFrame(self.summary_data)
        print(f"📈 Creating batch comparative analysis with {len(df)} experiments...")
        
        # Create comprehensive comparison plots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Final test accuracy comparison
        plt.subplot(3, 3, 1)
        if 'final_test_acc' in df.columns and df['final_test_acc'].notna().any():
            valid_data = df[df['final_test_acc'].notna()]
            plt.barh(range(len(valid_data)), valid_data['final_test_acc'])
            plt.yticks(range(len(valid_data)), valid_data['experiment'], rotation=0, fontsize=8)
            plt.xlabel('Final Test Accuracy (%)')
            plt.title('Final Test Accuracy by Experiment')
            plt.grid(True, alpha=0.3)
        
        # 2. Architecture comparison
        plt.subplot(3, 3, 2)
        if 'architecture' in df.columns and 'final_test_acc' in df.columns:
            arch_data = df.groupby('architecture')['final_test_acc'].agg(['mean', 'std']).reset_index()
            plt.bar(arch_data['architecture'], arch_data['mean'], yerr=arch_data['std'], capsize=5)
            plt.xlabel('Architecture')
            plt.ylabel('Test Accuracy (%)')
            plt.title('Performance by Architecture')
            plt.xticks(rotation=45)
        
        # 3. Dataset comparison
        plt.subplot(3, 3, 3)
        if 'dataset' in df.columns and 'final_test_acc' in df.columns:
            dataset_data = df.groupby('dataset')['final_test_acc'].agg(['mean', 'std']).reset_index()
            plt.bar(dataset_data['dataset'], dataset_data['mean'], yerr=dataset_data['std'], capsize=5)
            plt.xlabel('Dataset')
            plt.ylabel('Test Accuracy (%)')
            plt.title('Performance by Dataset')
        
        # 4. Data percentage vs performance
        plt.subplot(3, 3, 4)
        if 'data_percentage' in df.columns and 'final_test_acc' in df.columns:
            pct_data = df.groupby('data_percentage')['final_test_acc'].agg(['mean', 'std']).reset_index()
            plt.errorbar(pct_data['data_percentage'], pct_data['mean'], yerr=pct_data['std'], 
                        marker='o', capsize=5)
            plt.xlabel('Data Percentage (%)')
            plt.ylabel('Test Accuracy (%)')
            plt.title('Performance vs Data Percentage')
            plt.grid(True, alpha=0.3)
        
        # 5. Reward distribution
        plt.subplot(3, 3, 5)
        if 'avg_reward' in df.columns and df['avg_reward'].notna().any():
            plt.hist(df['avg_reward'].dropna(), bins=15, alpha=0.7, edgecolor='black')
            plt.xlabel('Average Reward')
            plt.ylabel('Frequency')
            plt.title('Distribution of Average Rewards')
        
        # 6. Training vs test accuracy correlation
        plt.subplot(3, 3, 6)
        if 'final_train_acc' in df.columns and 'final_test_acc' in df.columns:
            valid_data = df[df['final_train_acc'].notna() & df['final_test_acc'].notna()]
            if not valid_data.empty:
                plt.scatter(valid_data['final_train_acc'], valid_data['final_test_acc'], alpha=0.7)
                plt.plot([0, 100], [0, 100], 'r--', alpha=0.5)  # Perfect correlation line
                plt.xlabel('Final Training Accuracy (%)')
                plt.ylabel('Final Test Accuracy (%)')
                plt.title('Training vs Test Accuracy')
                plt.grid(True, alpha=0.3)
        
        # 7. Architecture vs Dataset heatmap
        plt.subplot(3, 3, 7)
        if 'architecture' in df.columns and 'dataset' in df.columns and 'final_test_acc' in df.columns:
            pivot_data = df.pivot_table(values='final_test_acc', 
                                      index='architecture', 
                                      columns='dataset', 
                                      aggfunc='mean')
            if not pivot_data.empty:
                sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='viridis')
                plt.title('Architecture vs Dataset Performance')
        
        # 8. Reward vs Accuracy correlation
        plt.subplot(3, 3, 8)
        if 'avg_reward' in df.columns and 'final_test_acc' in df.columns:
            valid_data = df[df['avg_reward'].notna() & df['final_test_acc'].notna()]
            if not valid_data.empty:
                plt.scatter(valid_data['avg_reward'], valid_data['final_test_acc'], alpha=0.7)
                plt.xlabel('Average Reward')
                plt.ylabel('Final Test Accuracy (%)')
                plt.title('Reward vs Test Accuracy')
                plt.grid(True, alpha=0.3)
        
        # 9. Performance summary table (text)
        plt.subplot(3, 3, 9)
        plt.axis('off')
        
        # Create summary statistics text
        summary_text = "EXPERIMENT SUMMARY\n" + "="*20 + "\n"
        summary_text += f"Total Experiments: {len(df)}\n"
        
        if 'final_test_acc' in df.columns and df['final_test_acc'].notna().any():
            best_acc = df['final_test_acc'].max()
            avg_acc = df['final_test_acc'].mean()
            worst_acc = df['final_test_acc'].min()
            summary_text += f"Best Test Acc: {best_acc:.2f}%\n"
            summary_text += f"Avg Test Acc: {avg_acc:.2f}%\n"
            summary_text += f"Worst Test Acc: {worst_acc:.2f}%\n"
        
        if 'avg_reward' in df.columns and df['avg_reward'].notna().any():
            best_reward = df['avg_reward'].max()
            avg_reward = df['avg_reward'].mean()
            summary_text += f"Best Avg Reward: {best_reward:.4f}\n"
            summary_text += f"Overall Avg Reward: {avg_reward:.4f}\n"
        
        # Add architecture counts
        if 'architecture' in df.columns:
            summary_text += "\nArchitectures:\n"
            arch_counts = df['architecture'].value_counts()
            for arch, count in arch_counts.head(5).items():
                summary_text += f"  {arch}: {count}\n"
        
        plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'batch_comparative_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return df
    
    def generate_comprehensive_report(self, df: pd.DataFrame):
        """Generate a comprehensive analysis report"""
        report_path = os.path.join(self.output_dir, 'comprehensive_batch_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MASCS BATCH EXPERIMENT ANALYSIS REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # Overview
            f.write("EXPERIMENT OVERVIEW\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total experiments analyzed: {len(df)}\n")
            f.write(f"Base log directory: {self.base_log_dir}\n")
            f.write(f"Analysis output directory: {self.output_dir}\n\n")
            
            # Performance rankings
            if 'final_test_acc' in df.columns and df['final_test_acc'].notna().any():
                f.write("TOP 10 PERFORMING EXPERIMENTS (by Test Accuracy)\n")
                f.write("-" * 50 + "\n")
                top_10 = df.nlargest(10, 'final_test_acc')
                for i, (_, row) in enumerate(top_10.iterrows(), 1):
                    f.write(f"{i:2d}. {row['experiment']:40s} {row['final_test_acc']:6.2f}%\n")
                f.write("\n")
            
            # Architecture analysis
            if 'architecture' in df.columns and 'final_test_acc' in df.columns:
                f.write("ARCHITECTURE PERFORMANCE ANALYSIS\n")
                f.write("-" * 35 + "\n")
                arch_stats = df.groupby('architecture')['final_test_acc'].agg(['count', 'mean', 'std', 'min', 'max'])
                arch_stats = arch_stats.sort_values('mean', ascending=False)
                
                f.write(f"{'Architecture':<20} {'Count':<6} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8}\n")
                f.write("-" * 70 + "\n")
                for arch, stats in arch_stats.iterrows():
                    f.write(f"{arch:<20} {stats['count']:<6.0f} {stats['mean']:<8.2f} {stats['std']:<8.2f} {stats['min']:<8.2f} {stats['max']:<8.2f}\n")
                f.write("\n")
            
            # Dataset analysis
            if 'dataset' in df.columns and 'final_test_acc' in df.columns:
                f.write("DATASET DIFFICULTY ANALYSIS\n")
                f.write("-" * 27 + "\n")
                dataset_stats = df.groupby('dataset')['final_test_acc'].agg(['count', 'mean', 'std', 'min', 'max'])
                dataset_stats = dataset_stats.sort_values('mean', ascending=False)
                
                f.write(f"{'Dataset':<15} {'Count':<6} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8}\n")
                f.write("-" * 65 + "\n")
                for dataset, stats in dataset_stats.iterrows():
                    f.write(f"{dataset:<15} {stats['count']:<6.0f} {stats['mean']:<8.2f} {stats['std']:<8.2f} {stats['min']:<8.2f} {stats['max']:<8.2f}\n")
                f.write("\n")
            
            # Strategy effectiveness
            if 'avg_reward' in df.columns and df['avg_reward'].notna().any():
                f.write("STRATEGY EFFECTIVENESS ANALYSIS\n")
                f.write("-" * 32 + "\n")
                reward_stats = df['avg_reward'].describe()
                f.write(f"Average reward statistics:\n")
                f.write(f"  Count: {reward_stats['count']:.0f}\n")
                f.write(f"  Mean:  {reward_stats['mean']:.4f}\n")
                f.write(f"  Std:   {reward_stats['std']:.4f}\n")
                f.write(f"  Min:   {reward_stats['min']:.4f}\n")
                f.write(f"  Max:   {reward_stats['max']:.4f}\n\n")
                
                # Best reward experiments
                if len(df) >= 5:
                    f.write("Top 5 experiments by average reward:\n")
                    top_rewards = df.nlargest(5, 'avg_reward')
                    for i, (_, row) in enumerate(top_rewards.iterrows(), 1):
                        f.write(f"  {i}. {row['experiment']}: {row['avg_reward']:.4f}\n")
                    f.write("\n")
            
            # Data efficiency analysis
            if 'data_percentage' in df.columns and 'final_test_acc' in df.columns:
                f.write("DATA EFFICIENCY ANALYSIS\n")
                f.write("-" * 24 + "\n")
                pct_stats = df.groupby('data_percentage')['final_test_acc'].agg(['count', 'mean', 'std'])
                pct_stats = pct_stats.sort_values('data_percentage')
                
                f.write(f"{'Data %':<8} {'Count':<6} {'Mean Acc':<10} {'Std':<8}\n")
                f.write("-" * 35 + "\n")
                for pct, stats in pct_stats.iterrows():
                    f.write(f"{pct:<8.0f} {stats['count']:<6.0f} {stats['mean']:<10.2f} {stats['std']:<8.2f}\n")
                f.write("\n")
            
            # Files generated
            f.write("FILES GENERATED\n")
            f.write("-" * 15 + "\n")
            f.write("📊 batch_comparative_analysis.png - Comprehensive comparison plots\n")
            f.write("📋 batch_summary.csv - Complete experiment metrics\n")
            f.write("📁 individual_* directories - Per-experiment detailed analysis\n")
            f.write("📝 comprehensive_batch_report.txt - This report\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("Batch analysis completed successfully!\n")
            f.write("=" * 80 + "\n")
        
        return report_path

def main():
    parser = argparse.ArgumentParser(description='Batch TensorBoard Log Analysis for MASCS')
    parser.add_argument('--log_dir', type=str, required=True,
                       help='Base directory containing TensorBoard log subdirectories')
    parser.add_argument('--output_dir', type=str, default='./batch_analysis',
                       help='Directory to save batch analysis outputs')
    parser.add_argument('--intermediate_dir', type=str, default=None,
                       help='Directory containing intermediate tracking files')
    parser.add_argument('--pattern', type=str, default=None,
                       help='Filter log directories by pattern (e.g., "CIFAR10" or "resnet")')
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.log_dir):
        print(f"❌ Log directory '{args.log_dir}' does not exist")
        sys.exit(1)
    
    print(f"🚀 Starting batch analysis of TensorBoard logs...")
    print(f"📁 Base log directory: {args.log_dir}")
    print(f"📊 Output directory: {args.output_dir}")
    
    # Initialize batch analyzer
    batch_analyzer = BatchLogAnalyzer(args.log_dir, args.output_dir)
    
    # Discover all log directories
    log_dirs = batch_analyzer.discover_log_directories()
    
    if not log_dirs:
        print("❌ No TensorBoard log directories found")
        sys.exit(1)
    
    # Filter by pattern if provided
    if args.pattern:
        original_count = len(log_dirs)
        log_dirs = [d for d in log_dirs if args.pattern.lower() in d.lower()]
        print(f"🔍 Filtered by pattern '{args.pattern}': {len(log_dirs)}/{original_count} directories")
    
    # Analyze each experiment
    print(f"\n📊 Analyzing {len(log_dirs)} experiments...")
    for i, log_dir in enumerate(log_dirs, 1):
        print(f"\n[{i:2d}/{len(log_dirs)}] ", end="")
        batch_analyzer.analyze_single_experiment(log_dir)
    
    # Create batch comparative analysis
    print(f"\n🔄 Creating batch comparative analysis...")
    df = batch_analyzer.create_batch_comparative_plots()
    
    # Save complete summary
    if df is not None and not df.empty:
        summary_path = os.path.join(args.output_dir, 'batch_summary.csv')
        df.to_csv(summary_path, index=False)
        print(f"💾 Saved complete summary to: {summary_path}")
        
        # Generate comprehensive report
        report_path = batch_analyzer.generate_comprehensive_report(df)
        print(f"📝 Generated comprehensive report: {report_path}")
    
    # Analyze intermediate tracking data if available
    if args.intermediate_dir and os.path.exists(args.intermediate_dir):
        print(f"\n🔄 Analyzing intermediate tracking data...")
        tracker = IntermediateTracker(args.intermediate_dir)
        tracker.load_intermediate(args.intermediate_dir)
        tracker.analyze_strategy_effectiveness()
    
    print(f"\n✅ Batch analysis complete!")
    print(f"📁 All outputs saved to: {args.output_dir}")

if __name__ == "__main__":
    main()