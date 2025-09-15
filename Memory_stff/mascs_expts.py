import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict, deque
import argparse
import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import warnings
import random
import sys
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from scipy.stats import dirichlet, beta, norm
from scipy.special import logsumexp
warnings.filterwarnings('ignore')


class BayesianWeightOptimizer:
    """
    Bayesian approach to learning optimal weight sequences for coreset selection.
    Maintains beliefs about weight effectiveness and updates based on observed outcomes.
    """
    
    def __init__(self, 
                 score_names: List[str] = ['S_U', 'S_B', 'S_G', 'S_F', 'S_D', 'S_C'],
                 sequence_length: int = 5,
                 memory_capacity: int = 1000):
        
        self.score_names = score_names
        self.n_scores = len(score_names)
        self.sequence_length = sequence_length
        
        # Bayesian priors - start with uninformative Dirichlet prior
        self.prior_alpha = np.ones(self.n_scores)  
        
        # Memory of sequences and their outcomes
        self.sequence_memory = deque(maxlen=memory_capacity)
        
        # Posterior parameters for different contexts
        self.context_posteriors = defaultdict(lambda: self.prior_alpha.copy())
        
        # Track uncertainty in our beliefs
        self.epistemic_uncertainty = defaultdict(float)
        
    def encode_context(self, state: Dict) -> str:
        """Encode current state into a context identifier"""
        loss_level = 'high' if state.get('loss', 0) > 1.0 else 'low'
        epoch_stage = 'early' if state.get('epoch', 0) < 20 else 'late'
        accuracy_level = 'low' if state.get('accuracy', 0) < 0.5 else 'high'
        return f"{loss_level}_loss_{epoch_stage}_epoch_{accuracy_level}_acc"
    
    def sample_weight_sequence(self, context: str, n_samples: int = 1) -> List[np.ndarray]:
        """Sample weight sequences from current posterior belief"""
        posterior_alpha = self.context_posteriors[context]
        exploration_bonus = self.epistemic_uncertainty[context]
        effective_alpha = posterior_alpha + exploration_bonus
        
        sequences = []
        for _ in range(n_samples):
            sequence = []
            for t in range(self.sequence_length):
                weights = dirichlet.rvs(effective_alpha, size=1)[0]
                
                if t > 0 and len(sequence) > 0:
                    prev_weights = sequence[-1]
                    transition_noise = dirichlet.rvs(np.ones(self.n_scores) * 10, size=1)[0]
                    weights = 0.7 * weights + 0.3 * prev_weights + 0.1 * transition_noise
                    weights /= weights.sum()
                
                sequence.append(weights)
            sequences.append(np.array(sequence))
        
        return sequences if n_samples > 1 else sequences[0]
    
    def update_posterior(self, 
                        context: str,
                        weight_sequence: np.ndarray,
                        outcomes: List[float],
                        delays: List[int]):
        """Update posterior beliefs based on observed sequence performance"""
        self.sequence_memory.append({
            'context': context,
            'sequence': weight_sequence.copy(),
            'outcomes': outcomes,
            'delays': delays,
            'total_reward': sum(outcomes)
        })
        
        self._update_context_posterior(context)
        self._update_epistemic_uncertainty(context)
    
    def _update_context_posterior(self, context: str):
        """Update posterior using all relevant sequences from memory"""
        relevant_sequences = [
            mem for mem in self.sequence_memory 
            if mem['context'] == context
        ]
        
        if not relevant_sequences:
            return
        
        successful_seqs = [
            mem for mem in relevant_sequences 
            if mem['total_reward'] > 0
        ]
        
        if not successful_seqs:
            return
        
        total_weight = sum(mem['total_reward'] for mem in successful_seqs)
        posterior_alpha = self.prior_alpha.copy()
        
        for mem in successful_seqs:
            weight = mem['total_reward'] / total_weight
            avg_weights = np.mean(mem['sequence'], axis=0)
            posterior_alpha += weight * avg_weights * len(successful_seqs)
        
        self.context_posteriors[context] = posterior_alpha
    
    def _update_epistemic_uncertainty(self, context: str):
        """Update uncertainty about the best weights for this context"""
        relevant_sequences = [
            mem for mem in self.sequence_memory 
            if mem['context'] == context
        ]
        
        if len(relevant_sequences) < 3:
            self.epistemic_uncertainty[context] = 1.0
            return
        
        outcomes = [mem['total_reward'] for mem in relevant_sequences]
        outcome_variance = np.var(outcomes) if len(outcomes) > 1 else 1.0
        
        sequences = np.array([mem['sequence'].flatten() for mem in relevant_sequences])
        sequence_variance = np.mean(np.var(sequences, axis=0))
        
        self.epistemic_uncertainty[context] = np.sqrt(outcome_variance * sequence_variance)


class GPStrategyWeightOptimizer:
    """
    Uses Gaussian Process-based Bayesian Optimization to find optimal
    strategy weights for dataset selection
    """
    
    def __init__(self, score_names=['S_U', 'S_B', 'S_G', 'S_F', 'S_D', 'S_C']):
        self.score_names = score_names
        self.n_scores = len(score_names)
        
        # Define search space - weights between 0 and 1
        self.search_space = [
            Real(0.0, 1.0, name=score) for score in score_names
        ]
        
        # Store optimization history
        self.history = defaultdict(list)
        self.best_weights = None
        self.best_performance = -np.inf
        self.validation_performance_fn = None
        
    def set_validation_function(self, validation_fn):
        """Set the validation function that evaluates weight performance"""
        self.validation_performance_fn = validation_fn
        
    def evaluate_weights(self, weights_dict, model=None, dataset=None, budget=None):
        """
        Evaluate a specific weight combination
        """
        if self.validation_performance_fn is not None:
            return self.validation_performance_fn(weights_dict, model, dataset, budget)
        
        # Default simulation for demo/testing
        true_optimal = {'S_U': 0.7, 'S_B': 0.6, 'S_G': 0.8, 
                       'S_F': 0.4, 'S_D': 0.5, 'S_C': 0.6}
        
        distance = sum((weights_dict[k] - true_optimal[k])**2 
                      for k in self.score_names)
        
        noise = np.random.normal(0, 0.02)
        performance = 0.95 - 0.3 * distance + noise
        
        return np.clip(performance, 0, 1)
    
    def objective(self, weights_list):
        """Objective function for GP optimization"""
        weights_dict = {
            name: weight for name, weight in zip(self.score_names, weights_list)
        }
        
        performance = self.evaluate_weights(weights_dict)
        
        # Store in history
        self.history['weights'].append(weights_dict)
        self.history['performance'].append(performance)
        
        # Update best if needed
        if performance > self.best_performance:
            self.best_performance = performance
            self.best_weights = weights_dict.copy()
        
        return -performance  # Minimize negative performance
    
    def optimize_per_strategy(self, strategies=['explore', 'exploit', 'refresh', 'balance', 'focus'], 
                            n_calls_per_strategy=20, verbose=False):
        """
        Optimize weights for each strategy separately with different objectives
        """
        strategy_weights = {}
        
        for strategy in strategies:
            if verbose:
                print(f"Optimizing weights for strategy: {strategy}")
            
            # Reset history for this strategy
            self.history = defaultdict(list)
            self.best_performance = -np.inf
            
            # Define strategy-specific objective
            def strategy_objective(weights_list):
                weights_dict = {
                    name: weight for name, weight in 
                    zip(self.score_names, weights_list)
                }
                
                base_performance = self.evaluate_weights(weights_dict)
                
                # Add strategy-specific bonuses
                if strategy == 'explore':
                    # Favor uncertainty and diversity
                    bonus = (weights_dict['S_U'] + weights_dict['S_D']) / 2
                elif strategy == 'exploit':
                    # Favor gradient and forgetting
                    bonus = (weights_dict['S_G'] + weights_dict['S_F']) / 2
                elif strategy == 'refresh':
                    # Favor diversity and boundary samples
                    bonus = (weights_dict['S_D'] + weights_dict['S_B']) / 2
                elif strategy == 'balance':
                    # Favor class balance
                    bonus = weights_dict['S_C']
                elif strategy == 'focus':
                    # Favor forgetting and gradient
                    bonus = (weights_dict['S_F'] + weights_dict['S_G']) / 2
                else:
                    bonus = 0
                
                return -(base_performance + 0.1 * bonus)
            
            # Run optimization for this strategy
            result = gp_minimize(
                func=strategy_objective,
                dimensions=self.search_space,
                n_calls=n_calls_per_strategy,
                n_initial_points=min(10, n_calls_per_strategy // 2),
                acq_func='EI',
                random_state=42 + strategies.index(strategy),
                noise='gaussian'
            )
            
            # Store best weights for this strategy
            best_weights_list = result.x
            strategy_weights[strategy] = {
                name: weight for name, weight in 
                zip(self.score_names, best_weights_list)
            }
            
            if verbose:
                print(f"Best weights for {strategy}: {self.format_weights(strategy_weights[strategy])}")
        
        return strategy_weights
    
    def format_weights(self, weights_dict):
        """Pretty print weights"""
        return ", ".join([f"{k}: {v:.3f}" for k, v in weights_dict.items()])


class MemoryAugmentedCoresetSelector:
    """Memory-Augmented Strategic Coreset Selection (MASCS) with MDP formulation"""
    
    def __init__(self, dataset, budget, feature_extractor, num_classes, 
                 memory_window=100, device='cuda' if torch.cuda.is_available() else 'cpu',
                 state_dim=64, action_dim=5, gamma=0.95):
        self.dataset = dataset
        self.budget = budget
        self.feature_extractor = feature_extractor
        self.num_classes = num_classes
        self.memory_window = memory_window
        self.device = device
        self.gamma = gamma  # MDP discount factor
        
        # Initialize memory for each sample (MDP state history)
        self.sample_memories = [deque(maxlen=memory_window) for _ in range(len(dataset))]
        self.selection_history = defaultdict(list)
        self.validation_improvements = defaultdict(list)
        self.reward_history = []
        self.state_history = []
        self.action_history = []
        
        # Historical effect buffer for temporal credit assignment
        self.effect_buffer = deque(maxlen=1000)
        self.strategy_effect_patterns = {
            'explore': {'typical_delay': 3, 'effect_duration': 5},
            'exploit': {'typical_delay': 1, 'effect_duration': 2},
            'refresh': {'typical_delay': 4, 'effect_duration': 6},
            'balance': {'typical_delay': 2, 'effect_duration': 3},
            'focus': {'typical_delay': 2, 'effect_duration': 4}
        }
        
        # Feature cache to avoid recomputation
        self.feature_cache = {}
        self.feature_staleness = {}
        self.model_version = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Initialize GP optimizer for strategy weights
        self.gp_optimizer = GPStrategyWeightOptimizer()
        
        # Initialize Bayesian weight optimizer
        self.bayesian_optimizer = BayesianWeightOptimizer()
        self.current_bayesian_sequence = None
        self.bayesian_sequence_start = None
        self.bayesian_sequence_outcomes = []
        
        # MDP Strategy definitions (actions) - will be optimized by GP
        self.strategies = self.initialize_strategy_weights()
        
        self.strategy_names = list(self.strategies.keys())
        self.current_strategy = 'explore'  # Current MDP action
        
        # Transformer policy network for MDP action selection
        self.policy_network = TransformerPolicyNetwork(state_dim, action_dim).to(device)
        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=1e-4)
        
        # Value network for policy gradient
        self.value_network = ValueNetwork(state_dim).to(device)
        self.value_optimizer = torch.optim.Adam(self.value_network.parameters(), lr=1e-4)
        
        # Temporal feature extractor
        self.temporal_encoder = nn.TransformerEncoderLayer(
            d_model=8, nhead=4, dim_feedforward=32, dropout=0.1, batch_first=True
        ).to(device)
    
    def initialize_strategy_weights(self):
        """Initialize strategy weights using GP optimization or defaults"""
        # Default weights as fallback
        default_strategies = {
            'explore': {'S_U': 0.7, 'S_B': 0.6, 'S_G': 0.5, 'S_F': 0.3, 'S_D': 0.8, 'S_C': 0.4},
            'exploit': {'S_U': 0.6, 'S_B': 0.5, 'S_G': 0.8, 'S_F': 0.7, 'S_D': 0.3, 'S_C': 0.4},
            'refresh': {'S_U': 0.4, 'S_B': 0.3, 'S_G': 0.4, 'S_F': 0.2, 'S_D': 0.7, 'S_C': 0.5},
            'balance': {'S_U': 0.3, 'S_B': 0.4, 'S_G': 0.3, 'S_F': 0.2, 'S_D': 0.6, 'S_C': 0.9},
            'focus': {'S_U': 0.5, 'S_B': 0.7, 'S_G': 0.6, 'S_F': 0.9, 'S_D': 0.2, 'S_C': 0.3}
        }
        
        return default_strategies
    
    def get_bayesian_weights(self, state: Dict, epoch: int) -> Dict[str, float]:
        """Get weights from Bayesian optimizer"""
        context = self.bayesian_optimizer.encode_context(state)
        
        # Check if we need a new sequence
        if (self.current_bayesian_sequence is None or 
            epoch - self.bayesian_sequence_start >= self.bayesian_optimizer.sequence_length):
            
            # Get new optimal sequence
            self.current_bayesian_sequence = self.bayesian_optimizer.sample_weight_sequence(context)
            self.bayesian_sequence_start = epoch
            self.bayesian_sequence_outcomes = []
        
        # Get weights for current position in sequence
        sequence_position = epoch - self.bayesian_sequence_start
        current_weights = self.current_bayesian_sequence[sequence_position]
        
        # Convert to dictionary
        weight_dict = {
            score_name: weight 
            for score_name, weight in zip(self.bayesian_optimizer.score_names, current_weights)
        }
        
        return weight_dict
    
    def update_bayesian_outcomes(self, epoch: int, performance_change: float):
        """Update Bayesian model with observed outcomes"""
        if self.current_bayesian_sequence is not None:
            delay = epoch - self.bayesian_sequence_start
            self.bayesian_sequence_outcomes.append({
                'delay': delay,
                'performance': performance_change
            })
            
            # If sequence is complete, update posterior
            if delay >= self.bayesian_optimizer.sequence_length - 1:
                context = self.bayesian_optimizer.encode_context({'epoch': epoch})
                outcomes = [o['performance'] for o in self.bayesian_sequence_outcomes]
                delays = [o['delay'] for o in self.bayesian_sequence_outcomes]
                
                self.bayesian_optimizer.update_posterior(
                    context, self.current_bayesian_sequence, outcomes, delays
                )
    
    def optimize_strategy_weights(self, model, val_loader, n_optimization_calls=15, verbose=False):
        """
        Optimize strategy weights using GP-based Bayesian optimization
        """
        if verbose:
            print("Optimizing strategy weights using Gaussian Process...")
        
        # Use historical effect patterns to inform optimization
        successful_patterns = self._analyze_successful_patterns()
        
        # Define validation function for GP optimizer
        def validation_fn(weights_dict, model_ref, dataset_ref, budget_ref):
            # Bias evaluation based on historical success
            historical_bonus = 0
            for past_success in successful_patterns:
                similarity = self._compute_weight_similarity(weights_dict, past_success['weights'])
                historical_bonus += similarity * past_success['average_improvement']
            
            # Create a temporary strategy with these weights
            temp_strategy = {'temp': weights_dict}
            
            # Select coreset using these weights
            current_coreset = np.random.choice(len(self.dataset), self.budget, replace=False)
            selected_indices, _, _ = self.select_coreset(
                model, nn.CrossEntropyLoss(), current_coreset, 
                strategy='temp', dataloader=val_loader
            )
            
            # Quick evaluation - train for a few steps and measure validation accuracy
            temp_model = create_model(type(self.dataset).__name__.lower().replace('dataset', ''), 
                                    self.num_classes, self.device)
            temp_model.load_state_dict(model.state_dict())
            temp_optimizer = torch.optim.Adam(temp_model.parameters(), lr=1e-4)
            
            # Create coreset loader
            coreset_dataset = torch.utils.data.Subset(self.dataset, selected_indices)
            coreset_loader = DataLoader(coreset_dataset, batch_size=64, shuffle=True)
            
            # Quick training
            temp_model.train()
            for i, (x, y) in enumerate(coreset_loader):
                if i >= 3:  # Only a few batches for quick evaluation
                    break
                x, y = x.to(self.device), y.to(self.device)
                temp_optimizer.zero_grad()
                outputs = temp_model(x)
                loss = nn.CrossEntropyLoss()(outputs, y)
                loss.backward()
                temp_optimizer.step()
            
            # Evaluate on validation set
            temp_model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for i, (x, y) in enumerate(val_loader):
                    if i >= 5:  # Quick evaluation
                        break
                    x, y = x.to(self.device), y.to(self.device)
                    outputs = temp_model(x)
                    _, predicted = torch.max(outputs.data, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()
            
            accuracy = correct / total if total > 0 else 0.0
            
            # Include historical bonus
            return accuracy + 0.1 * historical_bonus
        
        # Set up GP optimizer with validation function
        self.gp_optimizer.set_validation_function(validation_fn)
        
        # Optimize weights for each strategy
        optimized_weights = self.gp_optimizer.optimize_per_strategy(
            strategies=list(self.strategies.keys()),
            n_calls_per_strategy=n_optimization_calls,
            verbose=verbose
        )
        
        # Update strategies with optimized weights
        self.strategies.update(optimized_weights)
        
        if verbose:
            print("Strategy weight optimization complete!")
            for strategy, weights in optimized_weights.items():
                print(f"{strategy}: {self.gp_optimizer.format_weights(weights)}")
        
        return optimized_weights
    
    def _extract_features_with_cache(self, model, dataloader):
        """Extract features with caching to avoid recomputation"""
        features_list = []
        labels_list = []
        
        model.eval()
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(dataloader):
                batch_size = x.size(0)
                start_idx = batch_idx * dataloader.batch_size
                
                # Check cache for each sample in batch
                cached_features = []
                uncached_indices = []
                
                for i in range(batch_size):
                    sample_idx = start_idx + i
                    cache_key = f"{sample_idx}_v{self.model_version}"
                    
                    if cache_key in self.feature_cache:
                        cached_features.append((i, self.feature_cache[cache_key]))
                        self.cache_hits += 1
                    else:
                        uncached_indices.append(i)
                        self.cache_misses += 1
                
                # Extract only uncached features
                if uncached_indices:
                    x_uncached = x[uncached_indices].to(self.device)
                    new_features = self.feature_extractor(model, x_uncached).cpu()
                    
                    # Update cache
                    for local_idx, feat in zip(uncached_indices, new_features):
                        sample_idx = start_idx + local_idx
                        cache_key = f"{sample_idx}_v{self.model_version}"
                        self.feature_cache[cache_key] = feat
                        self.feature_staleness[cache_key] = self.model_version
                        
                        # Manage cache size
                        if len(self.feature_cache) > 10000:  # Max cache size
                            # Remove oldest entries
                            oldest_key = min(self.feature_cache.keys(), 
                                           key=lambda k: self.feature_staleness.get(k, 0))
                            del self.feature_cache[oldest_key]
                            if oldest_key in self.feature_staleness:
                                del self.feature_staleness[oldest_key]
                
                # Combine cached and new features in correct order
                batch_features = torch.zeros((batch_size, new_features.size(1) if uncached_indices else cached_features[0][1].size(0)))
                for i, feat in cached_features:
                    batch_features[i] = feat
                if uncached_indices:
                    for local_idx, feat in zip(uncached_indices, new_features):
                        batch_features[local_idx] = feat
                
                features_list.append(batch_features.numpy())
                labels_list.append(y.numpy())
        
        # Log cache performance periodically
        if len(self.selection_history) % 10 == 0 and (self.cache_hits + self.cache_misses) > 0:
            hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses)
            print(f"Feature cache hit rate: {hit_rate:.2%}")
        
        return np.vstack(features_list), np.concatenate(labels_list)
    
    def _analyze_successful_patterns(self):
        """Analyze which weight patterns led to improvements"""
        successful_patterns = []
        
        for selection in self.effect_buffer:
            if 'future_effects' not in selection:
                continue
                
            total_improvement = sum(effect['reward'] for effect in selection['future_effects'] 
                                  if effect['reward'] > 0)
            
            if total_improvement > 0.05:  # Significant improvement threshold
                avg_delay = np.mean([e['delay'] for e in selection['future_effects']]) if selection['future_effects'] else 0
                successful_patterns.append({
                    'strategy': selection['strategy'],
                    'weights': selection['strategy_weights'],
                    'average_improvement': total_improvement / len(selection['future_effects']) if selection['future_effects'] else 0,
                    'average_delay': avg_delay
                })
        
        return successful_patterns
    
    def _compute_weight_similarity(self, weights1, weights2):
        """Compute similarity between two weight configurations"""
        keys = set(weights1.keys()) & set(weights2.keys())
        if not keys:
            return 0
        
        diffs = [abs(weights1[k] - weights2[k]) for k in keys]
        return np.exp(-np.mean(diffs))  # Exponential similarity
    
    def _compute_temporal_credits(self, current_epoch, current_reward):
        """Compute credit assignment for past decisions"""
        credits = {}
        
        for past_selection in self.effect_buffer:
            past_epoch = past_selection['epoch']
            if past_epoch >= current_epoch:
                continue
                
            delay = current_epoch - past_epoch
            strategy = past_selection['strategy']
            
            # Strategy-specific expected delay
            if strategy in self.strategy_effect_patterns:
                expected_delay = self.strategy_effect_patterns[strategy]['typical_delay']
                effect_duration = self.strategy_effect_patterns[strategy]['effect_duration']
                
                # Credit based on delay match and duration
                if delay <= effect_duration:
                    # Temporal decay
                    time_decay = self.gamma ** delay
                    
                    # Delay match (highest when actual delay matches expected)
                    delay_match = np.exp(-0.5 * ((delay - expected_delay) ** 2))
                    
                    # Strategy weight magnitude (how confident was the decision)
                    weight_magnitude = np.std(list(past_selection['strategy_weights'].values()))
                    
                    credit = current_reward * time_decay * delay_match * weight_magnitude
                    credits[past_epoch] = credit
                    
                    # Update our understanding of strategy delays
                    if abs(current_reward) > 0.01:  # Significant effect
                        self._update_strategy_patterns(strategy, delay, current_reward)
        
        return credits
    
    def _update_strategy_patterns(self, strategy, observed_delay, observed_effect):
        """Update our model of how strategies affect future performance"""
        if strategy not in self.strategy_effect_patterns:
            return
        
        pattern = self.strategy_effect_patterns[strategy]
        
        # Simple exponential moving average update
        alpha = 0.1  # Learning rate
        
        # Update typical delay
        pattern['typical_delay'] = (1 - alpha) * pattern['typical_delay'] + alpha * observed_delay
        
        # Update effect duration if this is a longer effect than we've seen
        if observed_delay > pattern['effect_duration'] and abs(observed_effect) > 0.01:
            pattern['effect_duration'] = max(pattern['effect_duration'], observed_delay)
    
    def _compute_state_improvement(self, state, next_state):
        """Compute improvement between states"""
        # Simple L2 distance for now - could be more sophisticated
        return -torch.norm(next_state - state).item()
    
    def encode_state(self, performance_metrics: Dict, memory_stats: Dict) -> torch.Tensor:
        """Encode current state for MDP policy"""
        state_features = []
        
        # Performance metrics
        state_features.extend([
            performance_metrics.get('loss', 0.0),
            performance_metrics.get('accuracy', 0.0),
            performance_metrics.get('val_loss', 0.0),
            performance_metrics.get('val_accuracy', 0.0)
        ])
        
        # Memory statistics for each strategy
        for strategy in self.strategy_names:
            strategy_scores = memory_stats.get(strategy, [])
            if strategy_scores:
                state_features.extend([
                    np.mean(strategy_scores),
                    np.std(strategy_scores),
                    np.percentile(strategy_scores, 90)
                ])
            else:
                state_features.extend([0.0, 0.0, 0.0])
        
        # Recent reward history
        recent_rewards = self.reward_history[-10:] if len(self.reward_history) >= 10 else self.reward_history
        if recent_rewards:
            state_features.extend([
                np.mean(recent_rewards),
                np.std(recent_rewards) if len(recent_rewards) > 1 else 0.0
            ])
        else:
            state_features.extend([0.0, 0.0])
        
        # Pad to state_dim
        while len(state_features) < 64:
            state_features.append(0.0)
        
        return torch.tensor(state_features[:64], dtype=torch.float32).to(self.device)
    
    def select_strategy(self, state: torch.Tensor) -> str:
        """Select strategy using transformer policy network"""
        with torch.no_grad():
            action_probs = self.policy_network(state.unsqueeze(0))
            action_dist = torch.distributions.Categorical(action_probs)
            action_idx = action_dist.sample().item()
            return self.strategy_names[action_idx]
    
    def update_policy(self, state: torch.Tensor, action: str, reward: float, next_state: torch.Tensor):
        """Update policy network using policy gradient with temporal credit assignment"""
        action_idx = self.strategy_names.index(action)
        
        # Update effect buffer with observed outcomes
        current_epoch = len(self.selection_history)
        for past_selection in self.effect_buffer:
            if past_selection['epoch'] < current_epoch:
                delay = current_epoch - past_selection['epoch']
                past_selection['future_effects'].append({
                    'delay': delay,
                    'reward': reward,
                    'state_improvement': self._compute_state_improvement(state, next_state)
                })
        
        # Compute temporal credit for past decisions
        credits = self._compute_temporal_credits(current_epoch, reward)
        
        # Compute advantage with temporal credits
        with torch.no_grad():
            value = self.value_network(state.unsqueeze(0)).item()
            next_value = self.value_network(next_state.unsqueeze(0)).item()
            
            # Include temporal credits in advantage
            temporal_bonus = credits.get(current_epoch - 1, 0) if credits else 0
            advantage = reward + temporal_bonus + self.gamma * next_value - value
        
        # Update value network
        value_pred = self.value_network(state.unsqueeze(0))
        value_target = reward + self.gamma * next_value
        value_loss = F.mse_loss(value_pred, torch.tensor([value_target]).to(self.device))
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        # Update policy network
        action_probs = self.policy_network(state.unsqueeze(0))
        action_prob = action_probs[0, action_idx]
        policy_loss = -torch.log(action_prob) * advantage
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        return policy_loss.item(), value_loss.item()
    
    def compute_uncertainty_score(self, model, x):
        """Compute uncertainty score via prediction entropy"""
        with torch.no_grad():
            logits = model(x)
            probabilities = F.softmax(logits, dim=1)
            entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)
            return entropy.cpu().numpy()
    
    def compute_boundary_score(self, model, x):
        """Compute boundary score via prediction margin"""
        with torch.no_grad():
            logits = model(x)
            probabilities = F.softmax(logits, dim=1)
            top2 = torch.topk(probabilities, 2, dim=1).values
            margin = top2[:, 0] - top2[:, 1]
            return (1 - margin).cpu().numpy()
    
    def compute_gradient_score(self, model, x, y, loss_fn):
        """Compute gradient magnitude score"""
        model.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        
        grad_norms = []
        for param in model.parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.detach().norm(2).item())
        
        return np.mean(grad_norms) if grad_norms else 0.0
    
    def compute_forgetting_score(self, sample_idx, current_correct):
        """Compute forgetting score based on history"""
        if not self.sample_memories[sample_idx]:
            return 0.0
        
        last_memory = self.sample_memories[sample_idx][-1]
        was_correct = last_memory[3]  # S_F in memory
        return 1.0 if was_correct and not current_correct else 0.0
    
    def compute_diversity_score(self, features, current_coreset_features):
        """Compute diversity score via distance to nearest coreset sample"""
        if len(current_coreset_features) == 0:
            return np.ones(len(features))
        
        nbrs = NearestNeighbors(n_neighbors=1).fit(current_coreset_features)
        distances, _ = nbrs.kneighbors(features)
        return distances.flatten()
    
    def compute_class_balance_score(self, labels, current_coreset_labels):
        """Compute class balance score"""
        if len(current_coreset_labels) == 0:
            return np.ones(len(labels))
        
        class_counts = np.bincount(current_coreset_labels, minlength=self.num_classes)
        scores = []
        for label in labels:
            scores.append(1.0 / (class_counts[label] + 1))
        return np.array(scores)
    
    def extract_features(self, model, dataloader):
        """Extract features from the model's penultimate layer"""
        features, labels = [], []
        model.eval()
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                feat = self.feature_extractor(model, x)
                features.append(feat.cpu().numpy())
                labels.append(y.numpy())
        
        return np.vstack(features), np.concatenate(labels)
    
    def update_memory(self, sample_idx, scores, selected, validation_improvement=0.0):
        """Update memory for a sample with current scores and status"""
        memory_vector = [
            scores['S_U'], scores['S_B'], scores['S_G'], scores['S_F'],
            float(selected), validation_improvement
        ]
        memory_vector.extend([0.0] * (8 - len(memory_vector)))
        self.sample_memories[sample_idx].append(memory_vector)
        
        if selected:
            self.selection_history[sample_idx].append(len(self.sample_memories[sample_idx]) - 1)
            self.validation_improvements[sample_idx].append(validation_improvement)
    
    def compute_temporal_features(self, sample_idx):
        """Compute temporal features from memory using transformer"""
        memory = list(self.sample_memories[sample_idx])
        if not memory:
            return {
                'volatility': 0.0,
                'gradient_trend': 0.0,
                'forgetting_frequency': 0.0,
                'selection_impact': 0.0,
                'staleness': 0.0
            }
        
        memory_tensor = torch.tensor(memory, dtype=torch.float32).unsqueeze(0).to(self.device)
        encoded = self.temporal_encoder(memory_tensor)
        
        uncertainty_history = memory_tensor[0, :, 0].cpu().numpy()
        gradient_history = memory_tensor[0, :, 2].cpu().numpy()
        forgetting_history = memory_tensor[0, :, 3].cpu().numpy()
        selection_history = memory_tensor[0, :, 4].cpu().numpy()
        
        volatility = np.var(uncertainty_history) if len(uncertainty_history) > 1 else 0.0
        
        if len(gradient_history) > 1:
            time_points = np.arange(len(gradient_history))
            gradient_trend = np.polyfit(time_points, gradient_history, 1)[0]
        else:
            gradient_trend = 0.0
        
        forgetting_frequency = np.mean(forgetting_history)
        
        selected_indices = np.where(selection_history > 0.5)[0]
        if len(selected_indices) > 0:
            improvements = [self.validation_improvements[sample_idx][i] for i in range(min(len(selected_indices), len(self.validation_improvements[sample_idx])))]
            selection_impact = np.mean(improvements) if improvements else 0.0
        else:
            selection_impact = 0.0
        
        if len(selection_history) > 0:
            last_selection = np.where(selection_history > 0.5)[0]
            staleness = len(selection_history) - last_selection[-1] if len(last_selection) > 0 else len(selection_history)
        else:
            staleness = 0.0
        
        return {
            'volatility': volatility,
            'gradient_trend': gradient_trend,
            'forgetting_frequency': forgetting_frequency,
            'selection_impact': selection_impact,
            'staleness': staleness
        }
    
    def compute_temporal_bonus(self, temporal_features, strategy):
        """Compute temporal bonus based on strategy"""
        if strategy == 'explore':
            return temporal_features['volatility'] * 0.7 + temporal_features['staleness'] * 0.3
        elif strategy == 'exploit':
            return temporal_features['selection_impact'] * 0.8 + temporal_features['gradient_trend'] * 0.2
        elif strategy == 'refresh':
            return temporal_features['staleness'] * 0.9 + temporal_features['volatility'] * 0.1
        elif strategy == 'balance':
            return temporal_features['forgetting_frequency'] * 0.5 + temporal_features['selection_impact'] * 0.5
        elif strategy == 'focus':
            return temporal_features['forgetting_frequency'] * 0.8 + temporal_features['gradient_trend'] * 0.2
        else:
            return 0.0
    
    def select_coreset(self, model, loss_fn, current_coreset_indices, strategy=None, 
                      validation_improvement=0.0, dataloader=None):
        """Select a new coreset based on the specified strategy"""
        if dataloader is None:
            dataloader = DataLoader(self.dataset, batch_size=64, shuffle=False)
        
        if strategy is None:
            strategy = self.current_strategy
        
        # Track this selection in effect buffer
        selection_entry = {
            'epoch': len(self.effect_buffer),
            'strategy': strategy,
            'strategy_weights': self.strategies[strategy].copy() if strategy != 'temp' else self.gp_optimizer.history['weights'][-1] if self.gp_optimizer.history['weights'] else self.strategies['explore'].copy(),
            'validation_improvement': validation_improvement,
            'future_effects': []
        }
        self.effect_buffer.append(selection_entry)
        
        current_coreset = [self.dataset[i] for i in current_coreset_indices]
        if current_coreset:
            current_coreset_features, current_coreset_labels = self.extract_features(
                model, DataLoader(current_coreset, batch_size=64)
            )
        else:
            current_coreset_features, current_coreset_labels = np.array([]), np.array([])
        
        # Use cached features when possible
        all_features, all_labels = self._extract_features_with_cache(model, dataloader)
        
        all_scores = {
            'S_U': np.zeros(len(self.dataset)),
            'S_B': np.zeros(len(self.dataset)),
            'S_G': np.zeros(len(self.dataset)),
            'S_F': np.zeros(len(self.dataset)),
            'S_D': np.zeros(len(self.dataset)),
            'S_C': np.zeros(len(self.dataset))
        }
        
        model.train()
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)
            start_idx = batch_idx * dataloader.batch_size
            end_idx = min(start_idx + len(x), len(self.dataset))
            sample_indices = list(range(start_idx, end_idx))
            
            all_scores['S_U'][start_idx:end_idx] = self.compute_uncertainty_score(model, x)
            all_scores['S_B'][start_idx:end_idx] = self.compute_boundary_score(model, x)
            
            for i, (xi, yi) in enumerate(zip(x, y)):
                sample_idx = start_idx + i
                all_scores['S_G'][sample_idx] = self.compute_gradient_score(
                    model, xi.unsqueeze(0), yi.unsqueeze(0), loss_fn
                )
            
            with torch.no_grad():
                logits = model(x)
                predictions = torch.argmax(logits, dim=1)
                correct = (predictions == y).cpu().numpy()
            
            for i, sample_idx in enumerate(sample_indices):
                all_scores['S_F'][sample_idx] = self.compute_forgetting_score(sample_idx, correct[i])
        
        all_scores['S_D'] = self.compute_diversity_score(all_features, current_coreset_features)
        all_scores['S_C'] = self.compute_class_balance_score(all_labels, current_coreset_labels)
        
        # Normalize scores
        for key in all_scores:
            if np.max(all_scores[key]) > np.min(all_scores[key]):
                all_scores[key] = (all_scores[key] - np.min(all_scores[key])) / (
                    np.max(all_scores[key]) - np.min(all_scores[key]))
        
        # Compute final scores with strategy weights and temporal bonus
        if strategy == 'temp':
            # Handle temporary strategy from GP optimization
            strategy_weights = self.gp_optimizer.history['weights'][-1] if self.gp_optimizer.history['weights'] else self.strategies['explore']
        else:
            strategy_weights = self.strategies[strategy]
        final_scores = np.zeros(len(self.dataset))
        
        for i in range(len(self.dataset)):
            if strategy == 'temp':
                # For temporary strategy, weights are directly keyed by score type
                weighted_sum = sum(
                    strategy_weights[score_type] * all_scores[score_type][i] 
                    for score_type in ['S_U', 'S_B', 'S_G', 'S_F', 'S_D', 'S_C']
                )
            else:
                # For normal strategies, use the original format
                weighted_sum = sum(
                    strategy_weights[score_type] * all_scores[score_type][i] 
                    for score_type in ['S_U', 'S_B', 'S_G', 'S_F', 'S_D', 'S_C']
                )
            
            temporal_features = self.compute_temporal_features(i)
            temporal_bonus = self.compute_temporal_bonus(temporal_features, strategy)
            
            final_scores[i] = weighted_sum + 0.1 * temporal_bonus
            
            sample_scores = {score_type: all_scores[score_type][i] for score_type in all_scores}
            self.update_memory(i, sample_scores, False)
        
        selected_indices = np.argsort(final_scores)[-self.budget:]
        
        # Update memory for selected samples
        for idx in selected_indices:
            if self.sample_memories[idx]:
                memory = list(self.sample_memories[idx][-1])
                memory[4] = 1.0  # Mark as selected
                memory[5] = validation_improvement
                self.sample_memories[idx][-1] = memory
        
        return selected_indices, final_scores, all_scores
    
    def on_epoch_end(self):
        """Call this after each training epoch to update caches and patterns"""
        # Invalidate stale cache entries
        self.model_version += 1
        
        # Clean up old cache entries
        if self.model_version % 10 == 0:
            keys_to_remove = []
            for key in self.feature_cache:
                try:
                    version = int(key.split('_v')[1])
                    if self.model_version - version > 5:
                        keys_to_remove.append(key)
                except:
                    pass
            
            for key in keys_to_remove:
                del self.feature_cache[key]
                if key in self.feature_staleness:
                    del self.feature_staleness[key]
        
        # Analyze and print temporal patterns periodically
        if self.model_version % 20 == 0:
            print("\n=== Temporal Effect Analysis ===")
            for strategy, pattern in self.strategy_effect_patterns.items():
                print(f"{strategy}: typical delay={pattern['typical_delay']:.1f}, "
                      f"duration={pattern['effect_duration']}")


class TransformerPolicyNetwork(nn.Module):
    """Transformer-based policy network for strategy selection"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128, num_heads=8, num_layers=3):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.embedding = nn.Linear(state_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*2,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        # state: (batch_size, state_dim)
        embedded = self.embedding(state).unsqueeze(1)  # (batch_size, 1, hidden_dim)
        transformed = self.transformer(embedded)  # (batch_size, 1, hidden_dim)
        output = self.output_layer(transformed.squeeze(1))  # (batch_size, action_dim)
        return F.softmax(output, dim=-1)


class ValueNetwork(nn.Module):
    """Value network for policy gradient"""
    
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        return self.network(state)


def get_dataset(name: str, data_dir: str = './data'):
    """Load dataset by name"""
    if name.lower() == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform
        )
        return train_dataset, test_dataset, 10
    
    elif name.lower() == 'cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = torchvision.datasets.CIFAR100(
            root=data_dir, train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=data_dir, train=False, download=True, transform=transform
        )
        return train_dataset, test_dataset, 100
    
    elif name.lower() == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = torchvision.datasets.MNIST(
            root=data_dir, train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root=data_dir, train=False, download=True, transform=transform
        )
        return train_dataset, test_dataset, 10
    
    else:
        raise ValueError(f"Dataset {name} not supported")


def create_model(dataset_name: str, num_classes: int, device: str):
    """Create model based on dataset"""
    if dataset_name.lower() in ['cifar10', 'cifar100']:
        model = torchvision.models.resnet18(pretrained=False, num_classes=num_classes)
    elif dataset_name.lower() == 'mnist':
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    else:
        raise ValueError(f"Model for {dataset_name} not implemented")
    
    return model.to(device)


def simple_feature_extractor(model, x):
    """Extract features from model"""
    if hasattr(model, 'extract_features'):
        return model.extract_features(x)
    else:
        # For ResNet, remove final layer
        if hasattr(model, 'fc'):
            features = model(x)
            return features
        else:
            return x.flatten(1)


def train_model(model, train_loader, optimizer, loss_fn, device):
    """Train model for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        outputs = model(x)
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
    
    return total_loss / len(train_loader), 100.0 * correct / total


def evaluate_model(model, test_loader, loss_fn, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = loss_fn(outputs, y)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    return total_loss / len(test_loader), 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser(
        description='Memory-Augmented Strategic Coreset Selection (MASCS)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Dataset and Model Configuration
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument('--datasets', nargs='+', default=['CIFAR10'],
                           choices=['CIFAR10', 'CIFAR100', 'MNIST'],
                           help='Datasets to run experiments on')
    data_group.add_argument('--data_dir', type=str, default='./data',
                           help='Directory to store datasets')
    data_group.add_argument('--budget', type=int, default=5000,
                           help='Coreset budget (number of samples)')

    # Training Configuration
    training_group = parser.add_argument_group('Training Configuration')
    training_group.add_argument('--epochs', type=int, default=50,
                               help='Number of training epochs')
    training_group.add_argument('--batch_size', type=int, default=64,
                               help='Batch size for training')
    training_group.add_argument('--lr', type=float, default=1e-3,
                               help='Learning rate')
    training_group.add_argument('--device', type=str, default='auto',
                               choices=['auto', 'cuda', 'cpu', 'mps'],
                               help='Device to use for training')
    training_group.add_argument('--seed', type=int, default=42,
                               help='Random seed for reproducibility')

    # MASCS Configuration
    mascs_group = parser.add_argument_group('MASCS Configuration')
    mascs_group.add_argument('--memory_window', type=int, default=100,
                            help='Memory window size for each sample')
    mascs_group.add_argument('--state_dim', type=int, default=64,
                            help='State dimension for MDP policy network')
    mascs_group.add_argument('--action_dim', type=int, default=5,
                            help='Action dimension (number of strategies)')
    mascs_group.add_argument('--gamma', type=float, default=0.95,
                            help='MDP discount factor')

    # Optimization Configuration
    opt_group = parser.add_argument_group('Strategy Optimization')
    opt_group.add_argument('--optimize_weights', action='store_true',
                          help='Use GP to optimize strategy weights')
    opt_group.add_argument('--gp_calls', type=int, default=15,
                          help='Number of GP optimization calls per strategy')
    opt_group.add_argument('--weight_optimization_epoch', type=int, default=10,
                          help='Epoch at which to run weight optimization')
    opt_group.add_argument('--use_bayesian', action='store_true',
                          help='Use Bayesian weight optimization')

    # Logging and Output Configuration
    output_group = parser.add_argument_group('Output and Logging')
    output_group.add_argument('--log_dir', type=str, default='./logs',
                             help='Directory to store tensorboard logs')
    output_group.add_argument('--save_results', type=str, default='mascs_results.json',
                             help='File to save final results')
    output_group.add_argument('--save_intermediate', type=str,
                             help='Directory to save intermediate results and checkpoints')
    output_group.add_argument('--save_frequency', type=int, default=10,
                             help='Save intermediate results every N epochs')
    output_group.add_argument('--log_level', type=str, default='INFO',
                             choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                             help='Logging level')
    output_group.add_argument('--tensorboard_comment', type=str, default='',
                             help='Comment to add to tensorboard log directory name')

    # Experiment Configuration
    exp_group = parser.add_argument_group('Experiment Configuration')
    exp_group.add_argument('--resume_from', type=str,
                          help='Resume experiment from checkpoint directory')
    exp_group.add_argument('--experiment_name', type=str,
                          help='Custom experiment name for logging')
    exp_group.add_argument('--tags', nargs='*', default=[],
                          help='Tags to add to experiment for organization')
    exp_group.add_argument('--notes', type=str, default='',
                          help='Experiment notes/description')
    
    args = parser.parse_args()

    # Set device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device

    # Setup logging
    import logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)

    logger.info(f"Starting MASCS Experiments")
    logger.info(f"Device: {device}")
    logger.info(f"Arguments: {vars(args)}")

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Create necessary directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    # Create intermediate directory if specified
    if args.save_intermediate:
        os.makedirs(args.save_intermediate, exist_ok=True)
        logger.info(f"Intermediate results will be saved to: {args.save_intermediate}")

        # Save experiment metadata
        metadata = {
            'experiment_name': args.experiment_name or f"mascs_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'command_line_args': vars(args),
            'device': device,
            'start_time': datetime.now().isoformat(),
            'tags': args.tags,
            'notes': args.notes,
            'pytorch_version': torch.__version__,
            'python_version': sys.version
        }

        metadata_file = os.path.join(args.save_intermediate, 'experiment_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    # Check for resume capability
    resume_epoch = 0
    if args.resume_from and os.path.exists(args.resume_from):
        checkpoint_file = os.path.join(args.resume_from, 'experiment_checkpoint.json')
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
                resume_epoch = checkpoint_data.get('current_epoch', 0)
                logger.info(f"Resuming from epoch {resume_epoch}")
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")
                resume_epoch = 0
    
    results = {}
    
    # Run experiments on each dataset
    for dataset_name in tqdm.tqdm(args.datasets, desc="Datasets"):
        print(f"\n{'='*50}")
        print(f"Running MASCS on {dataset_name}")
        print(f"{'='*50}")
        
        # Load dataset
        try:
            train_dataset, test_dataset, num_classes = get_dataset(dataset_name, args.data_dir)
        except ValueError as e:
            print(f"Error loading {dataset_name}: {e}")
            continue
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Create model
        model = create_model(dataset_name, num_classes, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        loss_fn = nn.CrossEntropyLoss()
        
        # Initialize MASCS selector
        selector = MemoryAugmentedCoresetSelector(
            train_dataset, args.budget, simple_feature_extractor, 
            num_classes, args.memory_window, device
        )
        
        # Initialize tensorboard logger with enhanced naming
        exp_name = args.experiment_name or f"mascs_{dataset_name.lower()}"
        if args.tensorboard_comment:
            exp_name += f"_{args.tensorboard_comment}"
        exp_name += f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        log_path = os.path.join(args.log_dir, exp_name)
        writer = SummaryWriter(log_path)

        # Log experiment hyperparameters to tensorboard
        writer.add_hparams(
            {
                'budget': args.budget,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.lr,
                'memory_window': args.memory_window,
                'state_dim': getattr(args, 'state_dim', 64),
                'gamma': getattr(args, 'gamma', 0.95),
                'optimize_weights': args.optimize_weights,
                'use_bayesian': args.use_bayesian,
                'seed': args.seed
            },
            {},
            run_name=exp_name
        )
        
        # Initialize random coreset
        current_coreset = np.random.choice(len(train_dataset), args.budget, replace=False)
        
        # Training loop
        dataset_results = {
            'train_losses': [],
            'train_accuracies': [],
            'test_losses': [],
            'test_accuracies': [],
            'strategies_used': [],
            'rewards': [],
            'policy_losses': [],
            'value_losses': [],
            'temporal_credits': []
        }
        
        prev_performance = {'accuracy': 0.0, 'loss': float('inf')}
        weights_optimized = False
        
        for epoch in tqdm.tqdm(range(args.epochs), desc=f"Training {dataset_name}"):
            # Create coreset dataloader
            coreset_dataset = torch.utils.data.Subset(train_dataset, current_coreset)
            coreset_loader = DataLoader(coreset_dataset, batch_size=args.batch_size, shuffle=True)
            
            # Train on coreset
            train_loss, train_acc = train_model(model, coreset_loader, optimizer, loss_fn, device)
            
            # Evaluate on test set
            test_loss, test_acc = evaluate_model(model, test_loader, loss_fn, device)
            
            # Compute performance metrics
            current_performance = {
                'loss': train_loss,
                'accuracy': train_acc,
                'val_loss': test_loss,
                'val_accuracy': test_acc,
                'epoch': epoch
            }
            
            # Compute memory statistics for state encoding
            memory_stats = {}
            for strategy in selector.strategy_names:
                strategy_scores = []
                for i in range(len(train_dataset)):
                    temporal_features = selector.compute_temporal_features(i)
                    score = selector.compute_temporal_bonus(temporal_features, strategy)
                    strategy_scores.append(score)
                memory_stats[strategy] = strategy_scores
            
            # Encode current state
            current_state = selector.encode_state(current_performance, memory_stats)
            
            # Optimize strategy weights if requested
            if args.optimize_weights and epoch == args.weight_optimization_epoch and not weights_optimized:
                print(f"\nOptimizing strategy weights at epoch {epoch}...")
                try:
                    optimized_weights = selector.optimize_strategy_weights(
                        model, test_loader, args.gp_calls, verbose=True
                    )
                    weights_optimized = True
                    
                    # Log optimized weights to tensorboard
                    for strategy, weights in optimized_weights.items():
                        for score_type, weight in weights.items():
                            writer.add_scalar(f'OptimizedWeights/{strategy}_{score_type}', weight, epoch)
                    
                except Exception as e:
                    print(f"Weight optimization failed: {e}")
                    print("Continuing with default weights...")
            
            # Select strategy using MDP policy or Bayesian
            if args.use_bayesian and epoch > 5:
                # Use Bayesian weight optimization
                bayesian_weights = selector.get_bayesian_weights(current_performance, epoch)
                selector.strategies['bayesian'] = bayesian_weights
                strategy = 'bayesian'
            else:
                strategy = selector.select_strategy(current_state)
            selector.current_strategy = strategy
            
            # Compute reward (improvement in validation accuracy)
            reward = current_performance['val_accuracy'] - prev_performance['accuracy']
            selector.reward_history.append(reward)
            
            # Update Bayesian optimizer if used
            if args.use_bayesian and epoch > 5:
                selector.update_bayesian_outcomes(epoch, reward)
            
            # Update policy if we have a previous state
            if len(selector.state_history) > 0:
                prev_state = selector.state_history[-1]
                prev_action = selector.action_history[-1]
                policy_loss, value_loss = selector.update_policy(
                    prev_state, prev_action, reward, current_state
                )
                dataset_results['policy_losses'].append(policy_loss)
                dataset_results['value_losses'].append(value_loss)
                
                # Log policy updates
                writer.add_scalar(f'Policy/Loss', policy_loss, epoch)
                writer.add_scalar(f'Policy/Value_Loss', value_loss, epoch)
            
            # Store state and action
            selector.state_history.append(current_state)
            selector.action_history.append(strategy)
            
            # Select new coreset using the chosen strategy
            val_improvement = current_performance['val_accuracy'] - prev_performance['accuracy']
            new_coreset, scores, all_scores = selector.select_coreset(
                model, loss_fn, current_coreset, strategy, val_improvement, train_loader
            )
            current_coreset = new_coreset
            
            # Update selector state after training
            selector.on_epoch_end()
            
            # Store results
            dataset_results['train_losses'].append(train_loss)
            dataset_results['train_accuracies'].append(train_acc)
            dataset_results['test_losses'].append(test_loss)
            dataset_results['test_accuracies'].append(test_acc)
            dataset_results['strategies_used'].append(strategy)
            dataset_results['rewards'].append(reward)

            # Log metrics to tensorboard
            writer.add_scalar(f'Train/Loss', train_loss, epoch)
            writer.add_scalar(f'Train/Accuracy', train_acc, epoch)
            writer.add_scalar(f'Test/Loss', test_loss, epoch)
            writer.add_scalar(f'Test/Accuracy', test_acc, epoch)
            writer.add_scalar(f'Strategy/Reward', reward, epoch)
            writer.add_scalar(f'Strategy/Current', selector.strategy_names.index(strategy), epoch)

            # Log strategy weights
            if strategy in selector.strategies:
                current_strategy_weights = selector.strategies[strategy]
                for score_type, weight in current_strategy_weights.items():
                    writer.add_scalar(f'Weights/{strategy}_{score_type}', weight, epoch)

            # Log temporal credits if available
            if len(selector.effect_buffer) > 0:
                recent_effects = [e for e in selector.effect_buffer[-10:] if 'future_effects' in e and e['future_effects']]
                if recent_effects:
                    avg_credit = np.mean([np.mean([f['reward'] for f in e['future_effects']]) for e in recent_effects])
                    writer.add_scalar(f'TemporalCredit/Average', avg_credit, epoch)

            # Save intermediate results
            if hasattr(args, 'save_intermediate') and args.save_intermediate:
                save_intermediate_results(args.save_intermediate, dataset_name, epoch, args.epochs,
                                        current_performance, dataset_results, strategy)

            prev_performance = current_performance

            # Progress update
            if epoch % 10 == 0 or epoch == args.epochs - 1:
                print(f"Epoch {epoch+1}/{args.epochs} - Train: {train_acc:.2f}% | Test: {test_acc:.2f}% | Strategy: {strategy}")

        writer.close()

        # Store final results
        results[dataset_name] = {
            'final_train_accuracy': dataset_results['train_accuracies'][-1],
            'final_test_accuracy': dataset_results['test_accuracies'][-1],
            'best_test_accuracy': max(dataset_results['test_accuracies']),
            'final_train_loss': dataset_results['train_losses'][-1],
            'final_test_loss': dataset_results['test_losses'][-1],
            'min_test_loss': min(dataset_results['test_losses']),
            'strategies_used': dataset_results['strategies_used'],
            'total_reward': sum(dataset_results['rewards']),
            'average_reward': np.mean(dataset_results['rewards']),
            'training_history': dataset_results
        }

        print(f"\n{dataset_name} Results:")
        print(f"Final Test Accuracy: {results[dataset_name]['final_test_accuracy']:.2f}%")
        print(f"Best Test Accuracy: {results[dataset_name]['best_test_accuracy']:.2f}%")
        print(f"Total Reward: {results[dataset_name]['total_reward']:.3f}")

        # Save intermediate checkpoint at the end of each dataset
        if hasattr(args, 'save_intermediate') and args.save_intermediate:
            save_experiment_checkpoint(args.save_intermediate, dataset_name, 'completed', args.epochs, args.epochs)

    # Save final results
    with open(args.save_results, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nAll experiments completed! Results saved to {args.save_results}")
    print(f"Tensorboard logs saved to {args.log_dir}")

    return results


def save_intermediate_results(save_dir: str, dataset_name: str, epoch: int, total_epochs: int,
                            performance: dict, results: dict, strategy: str):
    """Save intermediate experiment results"""
    os.makedirs(save_dir, exist_ok=True)

    intermediate_data = {
        'dataset': dataset_name,
        'current_epoch': epoch,
        'total_epochs': total_epochs,
        'current_performance': performance,
        'current_strategy': strategy,
        'training_history': {
            'train_losses': results['train_losses'],
            'train_accuracies': results['train_accuracies'],
            'test_losses': results['test_losses'],
            'test_accuracies': results['test_accuracies'],
            'strategies_used': results['strategies_used'],
            'rewards': results['rewards']
        },
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'status': 'completed' if epoch >= total_epochs - 1 else 'running'
    }

    # Save to intermediate directory
    intermediate_file = os.path.join(save_dir, f'{dataset_name}_epoch_{epoch}.json')
    with open(intermediate_file, 'w') as f:
        json.dump(intermediate_data, f, indent=2)

    # Also save/update the main checkpoint file
    checkpoint_file = os.path.join(save_dir, 'experiment_checkpoint.json')
    checkpoint_data = {
        'dataset': dataset_name,
        'current_epoch': epoch,
        'total_epochs': total_epochs,
        'status': 'completed' if epoch >= total_epochs - 1 else 'running',
        'best_accuracy': max(results['test_accuracies']) if results['test_accuracies'] else 0.0,
        'current_accuracy': performance['val_accuracy'],
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)


def save_experiment_checkpoint(save_dir: str, dataset_name: str, status: str, epoch: int, total_epochs: int):
    """Save experiment checkpoint for resume capability"""
    os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        'dataset': dataset_name,
        'status': status,
        'current_epoch': epoch,
        'total_epochs': total_epochs,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    checkpoint_file = os.path.join(save_dir, 'experiment_checkpoint.json')
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)


if __name__ == "__main__":
    import time
    main()