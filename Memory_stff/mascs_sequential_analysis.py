#!/usr/bin/env python3
"""
MASCS Sequential Analysis for CIFAR-10 with ResNet18
Demonstrates how memory models track score patterns and use attention mechanisms
"""

import numpy as np
import torch
import torch.nn as nn
import json
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import time
from collections import deque, defaultdict
from datetime import datetime
import os

class EnhancedMemoryModel:
    """Enhanced memory model with attention and caching for MASCS"""
    
    def __init__(self, num_samples, memory_window=100, device='cpu'):
        self.num_samples = num_samples
        self.memory_window = memory_window
        self.device = device
        
        # Per-sample memory with enhanced tracking
        self.sample_memories = [deque(maxlen=memory_window) for _ in range(num_samples)]
        self.selection_history = [[] for _ in range(num_samples)]
        self.validation_improvements = [[] for _ in range(num_samples)]
        
        # Attention-based pattern recognition
        self.pattern_attention = nn.MultiheadAttention(
            embed_dim=6,  # 6 score types
            num_heads=2,
            dropout=0.1,
            batch_first=True
        ).to(device)
        
        # Pattern cache for faster retrieval
        self.pattern_cache = {}
        self.cache_timestamps = {}
        
    def update_sample_memory(self, sample_idx, scores, selected, validation_improvement, epoch):
        """Update memory with current scores and selection status"""
        memory_vector = [
            scores['S_U'], scores['S_B'], scores['S_G'], scores['S_F'],
            scores['S_D'], scores['S_C'],  # Score values
            float(selected), validation_improvement, epoch  # Additional metadata
        ]
        self.sample_memories[sample_idx].append(memory_vector)
        
        if selected:
            self.selection_history[sample_idx].append(epoch)
            self.validation_improvements[sample_idx].append(validation_improvement)
            
    def compute_temporal_features(self, sample_idx):
        """Compute temporal features from memory history"""
        if len(self.sample_memories[sample_idx]) < 2:
            return {
                'volatility': 0.0,
                'gradient_trend': 0.0,
                'forgetting_frequency': 0.0,
                'selection_impact': 0.0,
                'staleness': float(len(self.sample_memories[sample_idx]))
            }
            
        # Extract score histories
        memory_array = np.array(list(self.sample_memories[sample_idx]))
        uncertainty_history = memory_array[:, 0]  # S_U
        gradient_history = memory_array[:, 2]     # S_G
        forgetting_history = memory_array[:, 3]   # S_F
        epoch_history = memory_array[:, 8]        # epochs
        
        # Compute temporal features
        volatility = np.var(uncertainty_history) if len(uncertainty_history) > 1 else 0.0
        
        if len(gradient_history) > 1:
            time_points = np.arange(len(gradient_history))
            gradient_trend = np.polyfit(time_points, gradient_history, 1)[0]
        else:
            gradient_trend = 0.0
            
        forgetting_frequency = np.mean(forgetting_history) if len(forgetting_history) > 0 else 0.0
        
        selection_impact = np.mean(self.validation_improvements[sample_idx]) if self.validation_improvements[sample_idx] else 0.0
        
        # Staleness: epochs since last selection
        if self.selection_history[sample_idx]:
            staleness = epoch_history[-1] - self.selection_history[sample_idx][-1]
        else:
            staleness = float(len(self.sample_memories[sample_idx]))
            
        return {
            'volatility': volatility,
            'gradient_trend': gradient_trend,
            'forgetting_frequency': forgetting_frequency,
            'selection_impact': selection_impact,
            'staleness': staleness
        }
    
    def get_sequential_patterns(self, sample_idx, window_size=10):
        """Extract sequential patterns using attention mechanism"""
        cache_key = f"pattern_{sample_idx}_{len(self.sample_memories[sample_idx])}"
        
        # Check cache first (5 minute TTL)
        if cache_key in self.pattern_cache:
            cache_age = time.time() - self.cache_timestamps[cache_key]
            if cache_age < 300:  # 5 minutes
                return self.pattern_cache[cache_key]
        
        if len(self.sample_memories[sample_idx]) < 2:
            pattern_vector = torch.zeros(6, device=self.device)
            self.pattern_cache[cache_key] = pattern_vector
            self.cache_timestamps[cache_key] = time.time()
            return pattern_vector
            
        # Get recent history
        recent_memory = list(self.sample_memories[sample_idx])[-window_size:]
        memory_tensor = torch.tensor([[entry[i] for i in range(6)] for entry in recent_memory], 
                                   dtype=torch.float32, device=self.device)
        
        # Apply attention to find important patterns
        if len(memory_tensor) > 1:
            # Self-attention on score sequences
            attended_scores, attention_weights = self.pattern_attention(
                memory_tensor.unsqueeze(0),  # Query
                memory_tensor.unsqueeze(0),  # Key
                memory_tensor.unsqueeze(0)   # Value
            )
            pattern_vector = attended_scores.squeeze(0)[-1]  # Most recent attended pattern
        else:
            pattern_vector = memory_tensor[-1]  # Just the last scores if no history
            
        # Cache the result
        self.pattern_cache[cache_key] = pattern_vector
        self.cache_timestamps[cache_key] = time.time()
        
        return pattern_vector

class CacheSimulation:
    """Simulate cache effectiveness for MASCS"""
    
    def __init__(self, max_cache_size=1000):
        self.max_cache_size = max_cache_size
        self.cache = {}
        self.access_times = {}
        self.hit_count = 0
        self.miss_count = 0
        
    def get(self, key):
        """Get item from cache"""
        current_time = time.time()
        
        if key in self.cache:
            self.hit_count += 1
            self.access_times[key] = current_time
            return self.cache[key]
        else:
            self.miss_count += 1
            return None
            
    def put(self, key, value):
        """Put item in cache"""
        current_time = time.time()
        
        # Clean up if cache is full
        if len(self.cache) >= self.max_cache_size:
            self._cleanup()
            
        self.cache[key] = value
        self.access_times[key] = current_time
        
    def _cleanup(self):
        """Remove least recently used items"""
        if not self.access_times:
            return
            
        # Sort by access time and remove oldest 20%
        sorted_items = sorted(self.access_times.items(), key=lambda x: x[1])
        remove_count = max(1, int(self.max_cache_size * 0.2))
        
        for i in range(min(remove_count, len(sorted_items))):
            key_to_remove = sorted_items[i][0]
            if key_to_remove in self.cache:
                del self.cache[key_to_remove]
            del self.access_times[key_to_remove]
            
    def get_hit_rate(self):
        """Calculate cache hit rate"""
        total = self.hit_count + self.miss_count
        if total == 0:
            return 0
        return self.hit_count / total
        
    def get_stats(self):
        """Get cache statistics"""
        return {
            'hit_rate': self.get_hit_rate(),
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'cache_size': len(self.cache),
            'max_cache_size': self.max_cache_size
        }

def simulate_sample_scores(epoch, strategy='explore'):
    """
    Simulate how scores change for a sample over time based on strategy
    """
    base_scores = {}
    
    if strategy == 'explore':
        # High uncertainty, diversity-focused samples
        base_scores['S_U'] = 0.8 + 0.1 * np.sin(epoch * 0.5) + np.random.normal(0, 0.05)
        base_scores['S_B'] = 0.6 + 0.2 * np.cos(epoch * 0.3) + np.random.normal(0, 0.03)
        base_scores['S_G'] = 0.4 + 0.1 * np.sin(epoch * 0.7) + np.random.normal(0, 0.04)
        base_scores['S_F'] = 0.3 + 0.1 * np.random.random()
        base_scores['S_D'] = 0.9 - 0.1 * (epoch / 15) + np.random.normal(0, 0.02)
        base_scores['S_C'] = 0.5 + 0.2 * np.sin(epoch * 0.4) + np.random.normal(0, 0.03)
        
    elif strategy == 'exploit':
        # High gradient, forgetting-focused samples
        base_scores['S_U'] = 0.4 + 0.1 * np.random.random()
        base_scores['S_B'] = 0.5 + 0.1 * np.cos(epoch * 0.4) + np.random.normal(0, 0.02)
        base_scores['S_G'] = 0.8 + 0.1 * np.sin(epoch * 0.6) + np.random.normal(0, 0.03)
        base_scores['S_F'] = 0.7 + 0.2 * (epoch / 15) + np.random.normal(0, 0.04)
        base_scores['S_D'] = 0.3 + 0.1 * np.random.random()
        base_scores['S_C'] = 0.4 + 0.1 * np.sin(epoch * 0.3) + np.random.normal(0, 0.02)
        
    elif strategy == 'balance':
        # Class balance focused
        base_scores['S_U'] = 0.5 + 0.1 * np.random.random()
        base_scores['S_B'] = 0.4 + 0.1 * np.cos(epoch * 0.2) + np.random.normal(0, 0.02)
        base_scores['S_G'] = 0.5 + 0.1 * np.sin(epoch * 0.4) + np.random.normal(0, 0.03)
        base_scores['S_F'] = 0.4 + 0.1 * np.random.random()
        base_scores['S_D'] = 0.6 + 0.1 * np.sin(epoch * 0.5) + np.random.normal(0, 0.02)
        base_scores['S_C'] = 0.9 - 0.2 * (epoch / 15) + np.random.normal(0, 0.01)
        
    else:  # mixed/default
        base_scores['S_U'] = 0.6 + 0.2 * np.sin(epoch * 0.3) + np.random.normal(0, 0.03)
        base_scores['S_B'] = 0.5 + 0.2 * np.cos(epoch * 0.4) + np.random.normal(0, 0.02)
        base_scores['S_G'] = 0.6 + 0.2 * np.sin(epoch * 0.5) + np.random.normal(0, 0.03)
        base_scores['S_F'] = 0.5 + 0.2 * np.cos(epoch * 0.6) + np.random.normal(0, 0.02)
        base_scores['S_D'] = 0.6 + 0.2 * np.sin(epoch * 0.7) + np.random.normal(0, 0.02)
        base_scores['S_C'] = 0.6 + 0.2 * np.cos(epoch * 0.8) + np.random.normal(0, 0.02)
    
    # Ensure scores are between 0 and 1
    for key in base_scores:
        base_scores[key] = np.clip(base_scores[key], 0, 1)
        
    return base_scores

def main():
    """Main function to run MASCS sequential analysis"""
    
    # Create output directory
    output_dir = "sequential_analysis_output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("🎯 MASCS Sequential Analysis: CIFAR-10 with ResNet18")
    print("=" * 60)
    
    # Simulation parameters
    num_epochs = 15
    budget = 1000  # Coreset budget
    num_samples = 1000  # Simulate subset for demo
    
    print(f"Dataset: CIFAR-10 (simulating {num_samples} samples)")
    print(f"Model: ResNet18")
    print(f"Coreset Budget: {budget} samples")
    print(f"Training Epochs: {num_epochs}")
    print()
    
    # Initialize components
    memory_model = EnhancedMemoryModel(num_samples, memory_window=100, device='cpu')
    cache_sim = CacheSimulation(max_cache_size=1000)
    
    # Define score types
    score_types = ['S_U (Uncertainty)', 'S_B (Boundary)', 'S_G (Gradient)', 
                   'S_F (Forgetting)', 'S_D (Diversity)', 'S_C (Class Balance)']
    
    # Track sample evolution
    sample_id = 123  # Example sample to track in detail
    sample_scores_log = []
    temporal_features_log = []
    attention_weights_log = []
    cache_stats_log = []
    
    print(f"📊 Tracking Sample #{sample_id} Over {num_epochs} Epochs")
    print("-" * 50)
    
    # Simulate training epochs with different strategies
    strategy_sequence = ['explore', 'explore', 'exploit', 'exploit', 'exploit', 
                        'balance', 'balance', 'explore', 'explore', 'mixed',
                        'mixed', 'exploit', 'exploit', 'balance', 'mixed']
    
    # Log files
    detailed_log_file = os.path.join(output_dir, "detailed_sample_tracking.csv")
    temporal_log_file = os.path.join(output_dir, "temporal_features.csv")
    cache_log_file = os.path.join(output_dir, "cache_performance.csv")
    summary_file = os.path.join(output_dir, "analysis_summary.json")
    
    # Initialize CSV files
    with open(detailed_log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'strategy', 'S_U', 'S_B', 'S_G', 'S_F', 'S_D', 'S_C', 'selected', 'improvement'])
    
    with open(temporal_log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'volatility', 'gradient_trend', 'forgetting_frequency', 'selection_impact', 'staleness'])
    
    with open(cache_log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'hit_rate', 'hit_count', 'miss_count', 'cache_size'])
    
    # Simulate epochs
    for epoch in range(num_epochs):
        strategy = strategy_sequence[epoch]
        
        # Simulate scores for all samples
        epoch_scores = {}
        for i in range(num_samples):
            epoch_scores[i] = simulate_sample_scores(epoch, strategy)
        
        # Simulate selection (simplified)
        selected_samples = np.random.choice(num_samples, budget, replace=False)
        
        # Update memory for all samples
        for i in range(num_samples):
            selected = i in selected_samples
            improvement = np.random.normal(0.03, 0.01) if selected else 0.0
            memory_model.update_sample_memory(i, epoch_scores[i], selected, improvement, epoch)
            
            # Simulate cache access
            cache_key = f"temporal_{i}_{epoch}"
            cached_value = cache_sim.get(cache_key)
            if cached_value is None:
                # Cache miss - compute and store
                computation_result = memory_model.compute_temporal_features(i)
                cache_sim.put(cache_key, computation_result)
        
        # Log detailed tracking for our sample
        sample_scores = epoch_scores[sample_id]
        selected = sample_id in selected_samples
        improvement = np.random.normal(0.03, 0.01) if selected else 0.0
        
        # Log to CSV
        with open(detailed_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, strategy, 
                           sample_scores['S_U'], sample_scores['S_B'], sample_scores['S_G'],
                           sample_scores['S_F'], sample_scores['S_D'], sample_scores['S_C'],
                           int(selected), improvement])
        
        # Compute and log temporal features for our sample
        temporal_features = memory_model.compute_temporal_features(sample_id)
        with open(temporal_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, temporal_features['volatility'], temporal_features['gradient_trend'],
                           temporal_features['forgetting_frequency'], temporal_features['selection_impact'],
                           temporal_features['staleness']])
        
        # Log cache statistics
        cache_stats = cache_sim.get_stats()
        with open(cache_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, cache_stats['hit_rate'], cache_stats['hit_count'],
                           cache_stats['miss_count'], cache_stats['cache_size']])
        
        # Store for visualization
        sample_scores_log.append({
            'epoch': epoch,
            'strategy': strategy,
            'scores': sample_scores.copy(),
            'selected': selected,
            'improvement': improvement
        })
        temporal_features_log.append({
            'epoch': epoch,
            'features': temporal_features.copy()
        })
        cache_stats_log.append(cache_stats.copy())
        
        if epoch % 3 == 0:  # Print progress every 3 epochs
            print(f"Epoch {epoch:2d} ({strategy:7s}): "
                  f"S_U={sample_scores['S_U']:.2f}, S_G={sample_scores['S_G']:.2f}, "
                  f"Selected={'✓' if selected else '✗'}")
    
    # Compute attention patterns for our sample
    print(f"\n🔍 Computing attention patterns for sample {sample_id}...")
    attention_patterns = []
    for epoch in range(0, num_epochs, 2):  # Every 2 epochs
        if epoch < len(sample_scores_log):
            pattern = memory_model.get_sequential_patterns(sample_id)
            attention_patterns.append({
                'epoch': epoch,
                'pattern': pattern.detach().cpu().numpy().tolist()
            })
    
    # Create visualizations
    print(f"\n📊 Creating visualizations...")
    
    # Set up plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('MASCS Sequential Analysis: Sample Tracking', fontsize=16)
    
    # Plot 1: Score evolution
    ax1 = axes[0, 0]
    epochs = [entry['epoch'] for entry in sample_scores_log]
    for score_type in ['S_U', 'S_B', 'S_G', 'S_F', 'S_D', 'S_C']:
        scores = [entry['scores'][score_type] for entry in sample_scores_log]
        ax1.plot(epochs, scores, marker='o', label=score_type, linewidth=2, markersize=4)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Score Value')
    ax1.set_title('Score Evolution Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Temporal features
    ax2 = axes[0, 1]
    epochs = [entry['epoch'] for entry in temporal_features_log]
    features = ['volatility', 'gradient_trend', 'forgetting_frequency', 'selection_impact', 'staleness']
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for feature, color in zip(features, colors):
        values = [entry['features'][feature] for entry in temporal_features_log]
        ax2.plot(epochs, values, marker='s', label=feature, linewidth=2, color=color)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Feature Value')
    ax2.set_title('Temporal Feature Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Cache performance
    ax3 = axes[1, 0]
    epochs = list(range(len(cache_stats_log)))
    hit_rates = [entry['hit_rate'] for entry in cache_stats_log]
    
    ax3.plot(epochs, hit_rates, marker='o', linewidth=2, markersize=6, color='green')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Cache Hit Rate')
    ax3.set_title('Cache Performance Over Time')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    # Plot 4: Selection pattern
    ax4 = axes[1, 1]
    epochs = [entry['epoch'] for entry in sample_scores_log]
    selected = [int(entry['selected']) for entry in sample_scores_log]
    
    ax4.bar(epochs, selected, color=['red' if s == 0 else 'green' for s in selected])
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Selected (0=No, 1=Yes)')
    ax4.set_title('Sample Selection Pattern')
    ax4.set_ylim(0, 1.2)
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, "sequential_analysis_plots.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create strategy weight visualization
    strategy_weights = {
        'explore':  {'S_U': 0.7, 'S_B': 0.6, 'S_G': 0.5, 'S_F': 0.3, 'S_D': 0.8, 'S_C': 0.4},
        'exploit':  {'S_U': 0.6, 'S_B': 0.5, 'S_G': 0.8, 'S_F': 0.7, 'S_D': 0.3, 'S_C': 0.4},
        'balance':  {'S_U': 0.3, 'S_B': 0.4, 'S_G': 0.3, 'S_F': 0.2, 'S_D': 0.6, 'S_C': 0.9}
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    score_labels = ['S_U', 'S_B', 'S_G', 'S_F', 'S_D', 'S_C']
    x_pos = np.arange(len(score_labels))
    width = 0.25
    
    for i, (strategy, weights) in enumerate(strategy_weights.items()):
        weight_values = [weights[stype] for stype in score_labels]
        ax.bar(x_pos + i*width, weight_values, width, label=strategy.capitalize())
    
    ax.set_xlabel('Score Types')
    ax.set_ylabel('Weight Value')
    ax.set_title('Strategy Weight Optimization')
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(score_labels)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    strategy_plot_file = os.path.join(output_dir, "strategy_weights.png")
    plt.savefig(strategy_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary report
    final_cache_stats = cache_sim.get_stats()
    without_cache_time = sum([stats['hit_count'] * 0.01 + stats['miss_count'] * 0.1 
                             for stats in cache_stats_log])
    with_cache_time = sum([stats['hit_count'] * 0.001 + stats['miss_count'] * 0.1 
                          for stats in cache_stats_log])
    time_saved = without_cache_time - with_cache_time
    speedup = without_cache_time / with_cache_time if with_cache_time > 0 else 1
    
    summary_data = {
        "analysis_timestamp": datetime.now().isoformat(),
        "simulation_parameters": {
            "num_epochs": num_epochs,
            "num_samples": num_samples,
            "budget": budget
        },
        "sample_tracking": {
            "tracked_sample_id": sample_id,
            "final_scores": sample_scores_log[-1]['scores'] if sample_scores_log else {},
            "times_selected": sum(1 for entry in sample_scores_log if entry['selected'])
        },
        "temporal_features": {
            "final_features": temporal_features_log[-1]['features'] if temporal_features_log else {}
        },
        "cache_performance": {
            "final_hit_rate": final_cache_stats['hit_rate'],
            "total_hits": final_cache_stats['hit_count'],
            "total_misses": final_cache_stats['miss_count'],
            "time_saved_seconds": time_saved,
            "speedup_factor": speedup
        },
        "strategy_weights": strategy_weights
    }
    
    # Save summary
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    # Print final summary
    print(f"\n✅ Analysis Complete!")
    print(f"📊 Results saved to: {output_dir}")
    print(f"📈 Final cache hit rate: {final_cache_stats['hit_rate']:.2%}")
    print(f"⏱️  Time saved with caching: {time_saved:.2f}s ({speedup:.1f}x speedup)")
    print(f"📁 Output files:")
    print(f"   - {detailed_log_file}")
    print(f"   - {temporal_log_file}")
    print(f"   - {cache_log_file}")
    print(f"   - {summary_file}")
    print(f"   - {plot_file}")
    print(f"   - {strategy_plot_file}")
    
    return summary_data

if __name__ == "__main__":
    main()