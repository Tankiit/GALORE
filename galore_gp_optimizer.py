import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import torch
import torch.nn as nn
from collections import defaultdict
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class GALOREStrategyOptimizer:
    """
    GP-based strategy weight optimizer specifically designed for GALORE framework
    """
    
    def __init__(self, phase_detector, selection_policy, galore_config):
        self.phase_detector = phase_detector
        self.selection_policy = selection_policy
        self.galore_config = galore_config
        
        # Define the scoring functions from GALORE
        self.score_names = ['gradient_magnitude', 'diversity', 'uncertainty', 
                           'boundary', 'forgetting', 'class_balance']
        
        # Search space for each strategy weight
        self.search_space = [
            Real(0.0, 1.0, name=score) for score in self.score_names
        ]
        
        # Storage for phase-specific optimal weights
        self.phase_weights = {
            'early': None,
            'middle': None,
            'late': None,
            'transition': None
        }
        
        # History tracking
        self.optimization_history = defaultdict(list)
        
    def detect_current_phase(self, metrics):
        """Use GALORE's PhaseTransitionDetector to identify current phase"""
        # This integrates with your existing phase detection
        phase_info = self.phase_detector.analyze_phase(
            gradient_norm=metrics['gradient_norm'],
            loss_curvature=metrics['loss_curvature'],
            gradient_alignment=metrics['gradient_alignment']
        )
        return phase_info['phase']
    
    def evaluate_strategy_weights(self, weights_list, phase, dataset_info):
        """
        Evaluate strategy weights in context of GALORE framework
        """
        weights_dict = {
            name: weight for name, weight in zip(self.score_names, weights_list)
        }
        
        # Create a temporary selection policy with these weights
        temp_policy = self.create_temp_policy(weights_dict, phase)
        
        # Run mini-experiment with GaLore compression
        with self.galore_config.compression_context():
            # Select subset using the weights
            selected_indices = temp_policy.select_batch(
                dataset=dataset_info['data'],
                budget=dataset_info['budget']
            )
            
            # Train for few epochs and measure performance
            performance = self.quick_evaluate(
                selected_indices, 
                dataset_info,
                num_epochs=5  # Quick evaluation
            )
        
        # Consider multiple objectives based on phase
        if phase == 'early':
            # Early phase: prioritize diversity and exploration
            diversity_score = self.compute_diversity_score(selected_indices)
            final_score = 0.7 * performance['accuracy'] + 0.3 * diversity_score
            
        elif phase == 'middle':
            # Middle phase: balance performance and efficiency
            compression_efficiency = performance['galore_compression_ratio']
            final_score = (0.6 * performance['accuracy'] + 
                          0.2 * compression_efficiency + 
                          0.2 * performance['convergence_speed'])
            
        elif phase == 'late':
            # Late phase: focus on fine-tuning and hard samples
            hard_sample_coverage = self.compute_hard_sample_coverage(selected_indices)
            final_score = 0.8 * performance['accuracy'] + 0.2 * hard_sample_coverage
            
        else:  # transition
            # During transitions: maintain stability
            stability_score = 1 - performance['loss_variance']
            final_score = 0.5 * performance['accuracy'] + 0.5 * stability_score
        
        return final_score
    
    def create_temp_policy(self, weights_dict, phase):
        """Create a temporary policy with given weights"""
        # This will be implemented based on your selection policy structure
        temp_policy = type(self.selection_policy)(
            state_dim=self.selection_policy.state_dim,
            num_strategies=self.selection_policy.num_strategies
        )
        temp_policy.set_weights(weights_dict)
        return temp_policy
    
    def quick_evaluate(self, selected_indices, dataset_info, num_epochs=5):
        """Quick evaluation of selected subset"""
        # Simplified evaluation - you'll want to connect this to your actual training
        performance = {
            'accuracy': np.random.random() * 0.3 + 0.6,  # Placeholder
            'galore_compression_ratio': np.random.random() * 0.2 + 0.8,
            'convergence_speed': np.random.random() * 0.3 + 0.5,
            'loss_variance': np.random.random() * 0.1
        }
        return performance
    
    def compute_diversity_score(self, selected_indices):
        """Compute diversity of selected samples"""
        # Placeholder - implement based on your diversity metric
        return np.random.random() * 0.3 + 0.6
    
    def compute_hard_sample_coverage(self, selected_indices):
        """Compute coverage of hard samples"""
        # Placeholder - implement based on your hard sample identification
        return np.random.random() * 0.3 + 0.5
    
    def optimize_phase_specific_weights(self, phase, dataset_info, n_calls=30):
        """
        Optimize weights for a specific training phase
        """
        print(f"\nOptimizing weights for {phase} phase...")
        
        def objective(weights_list):
            score = self.evaluate_strategy_weights(weights_list, phase, dataset_info)
            # Track history
            self.optimization_history[phase].append({
                'weights': weights_list,
                'score': score
            })
            return -score  # Minimize negative score
        
        # Run GP optimization
        result = gp_minimize(
            func=objective,
            dimensions=self.search_space,
            n_calls=n_calls,
            n_initial_points=10,
            acq_func='EI',  # Expected Improvement
            noise='gaussian',
            random_state=42
        )
        
        # Store optimal weights
        optimal_weights = {
            name: weight for name, weight in zip(self.score_names, result.x)
        }
        self.phase_weights[phase] = optimal_weights
        
        print(f"Optimal weights for {phase} phase:")
        for name, weight in optimal_weights.items():
            print(f"  {name}: {weight:.3f}")
        
        return optimal_weights, result
    
    def adaptive_weight_update(self, current_metrics):
        """
        Dynamically update weights based on current training state
        Integrates with GALORE's adaptive selection policy
        """
        # Detect current phase
        phase = self.detect_current_phase(current_metrics)
        
        # Get base weights for this phase
        if self.phase_weights[phase] is None:
            # Use default if not optimized yet
            base_weights = self.get_default_weights(phase)
        else:
            base_weights = self.phase_weights[phase]
        
        # Apply dynamic adjustments based on recent performance
        adjusted_weights = self.apply_performance_adjustments(
            base_weights, 
            current_metrics
        )
        
        return adjusted_weights
    
    def get_default_weights(self, phase):
        """Get default weights for a phase"""
        defaults = {
            'early': {'gradient_magnitude': 0.3, 'diversity': 0.4, 'uncertainty': 0.1,
                     'boundary': 0.1, 'forgetting': 0.05, 'class_balance': 0.05},
            'middle': {'gradient_magnitude': 0.35, 'diversity': 0.2, 'uncertainty': 0.15,
                      'boundary': 0.15, 'forgetting': 0.1, 'class_balance': 0.05},
            'late': {'gradient_magnitude': 0.2, 'diversity': 0.1, 'uncertainty': 0.2,
                    'boundary': 0.2, 'forgetting': 0.2, 'class_balance': 0.1},
            'transition': {'gradient_magnitude': 0.25, 'diversity': 0.25, 'uncertainty': 0.15,
                          'boundary': 0.15, 'forgetting': 0.1, 'class_balance': 0.1}
        }
        return defaults.get(phase, defaults['middle'])
    
    def apply_performance_adjustments(self, base_weights, metrics):
        """
        Fine-tune weights based on recent training dynamics
        """
        adjusted = base_weights.copy()
        
        # If gradient variance is high, increase gradient_magnitude weight
        if metrics.get('gradient_variance', 0) > metrics.get('gradient_variance_avg', 1):
            adjusted['gradient_magnitude'] *= 1.1
        
        # If seeing many forgetting events, increase forgetting weight
        if metrics.get('forgetting_rate', 0) > 0.1:
            adjusted['forgetting'] *= 1.2
            
        # If class imbalance detected, boost class_balance
        if metrics.get('class_imbalance', 0) > 0.3:
            adjusted['class_balance'] *= 1.3
        
        # Normalize to sum to 1
        total = sum(adjusted.values())
        if total > 0:
            for k in adjusted:
                adjusted[k] /= total
            
        return adjusted
    
    def integrate_with_strategy_discovery(self):
        """
        Use GP optimization results to guide strategy discovery engine
        """
        # Analyze which weight combinations work best
        successful_combinations = []
        
        for phase, history in self.optimization_history.items():
            # Find top 10% performing weight combinations
            sorted_history = sorted(history, key=lambda x: x['score'], reverse=True)
            top_10_percent = sorted_history[:max(1, len(sorted_history) // 10)]
            
            for item in top_10_percent:
                successful_combinations.append({
                    'phase': phase,
                    'weights': item['weights'],
                    'score': item['score']
                })
        
        # Extract patterns for strategy discovery
        patterns = self.extract_weight_patterns(successful_combinations)
        
        # Generate new composite strategies
        new_strategies = self.generate_composite_strategies(patterns)
        
        return new_strategies
    
    def extract_weight_patterns(self, combinations):
        """
        Identify patterns in successful weight combinations
        """
        patterns = {
            'correlated_scores': [],
            'phase_specific_dominance': {},
            'weight_ranges': {}
        }
        
        if not combinations:
            return patterns
        
        # Find correlations between scores
        weight_matrix = np.array([c['weights'] for c in combinations])
        if len(weight_matrix) > 1:
            correlation = np.corrcoef(weight_matrix.T)
            
            # Identify strongly correlated score pairs
            for i in range(len(self.score_names)):
                for j in range(i+1, len(self.score_names)):
                    if abs(correlation[i, j]) > 0.7:
                        patterns['correlated_scores'].append(
                            (self.score_names[i], self.score_names[j], correlation[i, j])
                        )
        
        # Find phase-specific dominant scores
        for phase in ['early', 'middle', 'late']:
            phase_combos = [c for c in combinations if c.get('phase') == phase]
            if phase_combos:
                avg_weights = np.mean([c['weights'] for c in phase_combos], axis=0)
                dominant_idx = np.argmax(avg_weights)
                patterns['phase_specific_dominance'][phase] = self.score_names[dominant_idx]
        
        return patterns
    
    def generate_composite_strategies(self, patterns):
        """
        Generate new strategies based on discovered patterns
        """
        new_strategies = []
        
        # Create correlation-based strategies
        for score1, score2, corr in patterns['correlated_scores']:
            if corr > 0:
                # Positive correlation - combine them
                strategy = {
                    'name': f'{score1}_{score2}_synergy',
                    'weights': {s: 0.1 for s in self.score_names}
                }
                strategy['weights'][score1] = 0.4
                strategy['weights'][score2] = 0.4
                new_strategies.append(strategy)
        
        # Create phase-transition strategies
        for phase, dominant_score in patterns['phase_specific_dominance'].items():
            strategy = {
                'name': f'{phase}_optimized',
                'weights': {s: 0.1 for s in self.score_names}
            }
            strategy['weights'][dominant_score] = 0.5
            new_strategies.append(strategy)
        
        return new_strategies
    
    def save_optimization_results(self, filepath='galore_gp_weights.json'):
        """Save optimized weights and patterns"""
        results = {
            'phase_weights': self.phase_weights,
            'optimization_history': dict(self.optimization_history),
            'discovered_strategies': self.integrate_with_strategy_discovery()
        }
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            return obj
        
        results = convert_to_serializable(results)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
    
    def visualize_optimization(self):
        """Visualize GP optimization results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        phases = ['early', 'middle', 'late', 'transition']
        
        for idx, (ax, phase) in enumerate(zip(axes.flat, phases)):
            if phase in self.optimization_history:
                history = self.optimization_history[phase]
                
                if history:
                    # Plot optimization progress
                    scores = [h['score'] for h in history]
                    ax.plot(scores, 'bo-', alpha=0.5, label='Evaluated')
                    ax.plot(np.maximum.accumulate(scores), 'r-', linewidth=2, label='Best')
                    
                    ax.set_title(f'{phase.capitalize()} Phase Optimization')
                    ax.set_xlabel('Iteration')
                    ax.set_ylabel('Score')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('galore_gp_optimization.png')
        plt.show()


# Integration with your existing GALORE code
class EnhancedGALOREFramework:
    """
    Enhanced GALORE with GP-based weight optimization
    """
    
    def __init__(self, phase_detector, selection_policy, galore_config, 
                 num_epochs=100, coreset_size=1000):
        # Keep your existing components
        self.phase_detector = phase_detector
        self.selection_policy = selection_policy
        self.galore_config = galore_config
        self.num_epochs = num_epochs
        self.coreset_size = coreset_size
        
        # Add GP optimizer
        self.gp_optimizer = GALOREStrategyOptimizer(
            self.phase_detector,
            self.selection_policy,
            self.galore_config
        )
        
        # Track metrics
        self.current_metrics = {}
        
    def initialize_optimal_weights(self, val_dataset=None):
        """Run GP optimization for each phase using validation data"""
        print("Initializing optimal weights for each phase...")
        
        # Use provided validation dataset or create a small one
        if val_dataset is None:
            val_dataset = self.prepare_validation_dataset()
        
        for phase in ['early', 'middle', 'late', 'transition']:
            # Simulate phase conditions
            phase_dataset_info = self.simulate_phase_conditions(val_dataset, phase)
            
            # Optimize weights for this phase
            self.gp_optimizer.optimize_phase_specific_weights(
                phase, 
                phase_dataset_info,
                n_calls=30  # Adjust based on compute budget
            )
        
        # Save results
        self.gp_optimizer.save_optimization_results()
        
        # Generate visualization
        self.gp_optimizer.visualize_optimization()
    
    def prepare_validation_dataset(self):
        """Prepare a small validation dataset for optimization"""
        # This should be implemented based on your dataset structure
        return {
            'data': None,  # Your validation data
            'budget': 100  # Small budget for quick evaluation
        }
    
    def simulate_phase_conditions(self, dataset, phase):
        """Simulate conditions for a specific training phase"""
        # Adjust dataset characteristics based on phase
        phase_conditions = {
            'early': {'gradient_variance': 0.5, 'loss_variance': 0.3},
            'middle': {'gradient_variance': 0.3, 'loss_variance': 0.2},
            'late': {'gradient_variance': 0.1, 'loss_variance': 0.1},
            'transition': {'gradient_variance': 0.4, 'loss_variance': 0.25}
        }
        
        dataset_info = {
            'data': dataset,
            'budget': self.coreset_size // 10,  # Use smaller budget for optimization
            'phase_metrics': phase_conditions.get(phase, {})
        }
        
        return dataset_info
    
    def compute_current_metrics(self):
        """Compute current training metrics"""
        # This should integrate with your existing metric computation
        metrics = {
            'gradient_norm': 0.0,
            'loss_curvature': 0.0,
            'gradient_alignment': 0.0,
            'gradient_variance': 0.0,
            'gradient_variance_avg': 0.0,
            'forgetting_rate': 0.0,
            'class_imbalance': 0.0,
            'phase': 'middle'
        }
        return metrics
    
    def get_dominant_strategy(self, weights):
        """Get the dominant strategy from weights"""
        max_weight = max(weights.values())
        for name, weight in weights.items():
            if weight == max_weight:
                return name
        return 'mixed'
    
    def train_step(self, selected_indices, dataset, model, optimizer):
        """Single training step with selected indices"""
        # This should integrate with your existing training logic
        subset = torch.utils.data.Subset(dataset, selected_indices)
        loader = torch.utils.data.DataLoader(subset, batch_size=32, shuffle=True)
        
        train_loss = 0.0
        for batch_idx, (data, target) in enumerate(loader):
            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        return {'loss': train_loss / len(loader)}
    
    def evaluate(self, model, test_loader):
        """Evaluate model performance"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        return correct / total if total > 0 else 0
    
    def run_enhanced_experiment(self, dataset, model, optimizer, test_loader):
        """
        Run GALORE experiment with GP-optimized weights
        """
        results = {
            'dataset': 'experiment',
            'phases': [],
            'final_accuracy': 0,
            'strategy_usage': defaultdict(int)
        }
        
        # Initialize weights if not done
        if all(v is None for v in self.gp_optimizer.phase_weights.values()):
            self.initialize_optimal_weights()
        
        # Training loop with adaptive weight updates
        for epoch in range(self.num_epochs):
            # Get current metrics
            metrics = self.compute_current_metrics()
            
            # Get optimized weights for current state
            weights = self.gp_optimizer.adaptive_weight_update(metrics)
            
            # Update selection policy with new weights (this depends on your policy implementation)
            # self.selection_policy.update_weights(weights)
            
            # Select coreset indices
            selected_indices = list(range(min(self.coreset_size, len(dataset))))
            
            # Train with selected subset
            train_metrics = self.train_step(selected_indices, dataset, model, optimizer)
            
            # Track results
            if epoch % 10 == 0:
                results['phases'].append({
                    'epoch': epoch,
                    'phase': metrics['phase'],
                    'weights': weights
                })
            
            results['strategy_usage'][self.get_dominant_strategy(weights)] += 1
            
            # Log progress
            if epoch % 10 == 0:
                accuracy = self.evaluate(model, test_loader)
                logger.info(f"Epoch {epoch}: Accuracy = {accuracy:.4f}")
        
        results['final_accuracy'] = self.evaluate(model, test_loader)
        
        return results