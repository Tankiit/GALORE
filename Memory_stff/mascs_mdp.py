import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import timm
import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict, deque
import argparse
import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import warnings
import random
from typing import Dict, List, Tuple, Optional
import json
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
warnings.filterwarnings('ignore')


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
                 memory_window=100, device='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'),
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
        
        # Caching for performance optimization
        self.cache = {
            'features': {},  # Cache extracted features per epoch
            'scores': {},    # Cache computed scores per epoch/strategy
            'temporal_features': {},  # Cache temporal features per sample
            'memory_stats': {},  # Cache memory statistics per epoch
            'last_epoch': -1,  # Track last cached epoch
        }
        
        # Initialize GP optimizer for strategy weights
        self.gp_optimizer = GPStrategyWeightOptimizer()
        
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
    
    def optimize_strategy_weights(self, model, val_loader, n_optimization_calls=15, verbose=False):
        """
        Optimize strategy weights using GP-based Bayesian optimization
        """
        if verbose:
            print("Optimizing strategy weights using Gaussian Process...")
        
        # Define validation function for GP optimizer
        def validation_fn(weights_dict, model_ref, dataset_ref, budget_ref):
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
            return accuracy
        
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
        """Update policy network using policy gradient"""
        action_idx = self.strategy_names.index(action)
        
        # Compute advantage
        with torch.no_grad():
            value = self.value_network(state.unsqueeze(0)).item()
            next_value = self.value_network(next_state.unsqueeze(0)).item()
            advantage = reward + self.gamma * next_value - value
        
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
        model.eval()  # Set to eval mode to avoid BatchNorm issues
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
    
    def extract_features(self, model, dataloader, epoch=None, cache_key=None):
        """Extract features from the model's penultimate layer with caching"""
        if cache_key and cache_key in self.cache['features']:
            return self.cache['features'][cache_key]
            
        features, labels = [], []
        model.eval()
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                feat = self.feature_extractor(model, x)
                features.append(feat.cpu().numpy())
                labels.append(y.numpy())
        
        result = np.vstack(features), np.concatenate(labels)
        
        # Cache the result if cache_key provided
        if cache_key:
            self.cache['features'][cache_key] = result
            
        return result
    
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
        """Compute temporal features from memory using transformer with caching"""
        # Check cache first
        cache_key = f"temporal_{sample_idx}_{len(self.sample_memories[sample_idx])}"
        if cache_key in self.cache['temporal_features']:
            return self.cache['temporal_features'][cache_key]
            
        memory = list(self.sample_memories[sample_idx])
        if not memory:
            result = {
                'volatility': 0.0,
                'gradient_trend': 0.0,
                'forgetting_frequency': 0.0,
                'selection_impact': 0.0,
                'staleness': 0.0
            }
            self.cache['temporal_features'][cache_key] = result
            return result
        
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
        if len(selected_indices) > 0 and sample_idx in self.validation_improvements:
            improvements = self.validation_improvements[sample_idx][:len(selected_indices)]
            selection_impact = np.mean(improvements) if improvements else 0.0
        else:
            selection_impact = 0.0
        
        if len(selection_history) > 0:
            last_selection = np.where(selection_history > 0.5)[0]
            staleness = len(selection_history) - last_selection[-1] if len(last_selection) > 0 else len(selection_history)
        else:
            staleness = 0.0
        
        result = {
            'volatility': volatility,
            'gradient_trend': gradient_trend,
            'forgetting_frequency': forgetting_frequency,
            'selection_impact': selection_impact,
            'staleness': staleness
        }
        
        # Cache the result
        self.cache['temporal_features'][cache_key] = result
        return result
    
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
        
        current_coreset = [self.dataset[i] for i in current_coreset_indices]
        if current_coreset:
            current_coreset_features, current_coreset_labels = self.extract_features(
                model, DataLoader(current_coreset, batch_size=64), cache_key=f"coreset_{hash(tuple(current_coreset_indices))}"
            )
        else:
            current_coreset_features, current_coreset_labels = np.array([]), np.array([])
        
        all_features, all_labels = self.extract_features(model, dataloader, cache_key=f"all_features_{hash(str(model.state_dict()))}")
        
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
            
            # Compute batch gradient score (much faster than individual)
            batch_grad_score = self.compute_gradient_score(model, x, y, loss_fn)
            all_scores['S_G'][start_idx:end_idx] = batch_grad_score
            
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
    
    def cleanup_cache(self, current_epoch):
        """Clean up old cache entries to prevent memory bloat"""
        # Keep only last 3 epochs of cached data
        epochs_to_keep = 3
        
        # Clean memory stats cache
        keys_to_remove = []
        for key in self.cache['memory_stats']:
            if key.startswith('memory_stats_'):
                epoch_num = int(key.split('_')[-1])
                if current_epoch - epoch_num > epochs_to_keep:
                    keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.cache['memory_stats'][key]
        
        # Clean temporal features cache if it gets too large
        if len(self.cache['temporal_features']) > 10000:
            # Keep only most recent entries
            items = list(self.cache['temporal_features'].items())
            self.cache['temporal_features'] = dict(items[-5000:])
        
        # Clean features cache if it gets too large  
        if len(self.cache['features']) > 100:
            # Keep only most recent entries
            items = list(self.cache['features'].items())
            self.cache['features'] = dict(items[-50:])


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


def get_dataset(name: str, data_dir: str = './data', data_percentage: float = 100.0):
    """Load dataset by name with configurable data percentage"""
    
    # Define transforms for different datasets
    if name.lower() in ['cifar10', 'cifar100']:
        transform = transforms.Compose([
            transforms.Resize(224),  # Resize for TIMM models
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet normalization
        ])
    elif name.lower() == 'mnist':
        transform = transforms.Compose([
            transforms.Resize(224),  # Resize for TIMM models
            transforms.Grayscale(num_output_channels=3),  # Convert to RGB
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    elif name.lower() == 'fashionmnist':
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    elif name.lower() == 'svhn':
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    else:
        # Default transform for other datasets
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    
    # Load datasets
    if name.lower() == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform
        )
        num_classes = 10
        
    elif name.lower() == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100(
            root=data_dir, train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=data_dir, train=False, download=True, transform=transform
        )
        num_classes = 100
        
    elif name.lower() == 'mnist':
        train_dataset = torchvision.datasets.MNIST(
            root=data_dir, train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root=data_dir, train=False, download=True, transform=transform
        )
        num_classes = 10
        
    elif name.lower() == 'fashionmnist':
        train_dataset = torchvision.datasets.FashionMNIST(
            root=data_dir, train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            root=data_dir, train=False, download=True, transform=transform
        )
        num_classes = 10
        
    elif name.lower() == 'svhn':
        train_dataset = torchvision.datasets.SVHN(
            root=data_dir, split='train', download=True, transform=transform
        )
        test_dataset = torchvision.datasets.SVHN(
            root=data_dir, split='test', download=True, transform=transform
        )
        num_classes = 10
        
    elif name.lower() == 'stl10':
        train_dataset = torchvision.datasets.STL10(
            root=data_dir, split='train', download=True, transform=transform
        )
        test_dataset = torchvision.datasets.STL10(
            root=data_dir, split='test', download=True, transform=transform
        )
        num_classes = 10
        
    else:
        raise ValueError(f"Dataset {name} not supported. Supported: CIFAR10, CIFAR100, MNIST, FashionMNIST, SVHN, STL10")
    
    # Apply data percentage sampling
    if data_percentage < 100.0:
        train_size = int(len(train_dataset) * (data_percentage / 100.0))
        indices = np.random.choice(len(train_dataset), train_size, replace=False)
        train_dataset = Subset(train_dataset, indices)
        print(f"Using {data_percentage}% of training data: {len(train_dataset)} samples")
    
    return train_dataset, test_dataset, num_classes


def create_model(architecture: str, num_classes: int, device: str, pretrained: bool = False):
    """Create model using TIMM architectures"""
    try:
        # Create model using TIMM
        model = timm.create_model(
            architecture, 
            pretrained=pretrained, 
            num_classes=num_classes
        )
        print(f"Created {architecture} model with {num_classes} classes (pretrained={pretrained})")
        return model.to(device)
    except Exception as e:
        print(f"Error creating {architecture}: {e}")
        print("Falling back to ResNet18...")
        # Fallback to ResNet18 if TIMM model fails
        model = timm.create_model('resnet18', pretrained=pretrained, num_classes=num_classes)
        return model.to(device)

def get_available_architectures():
    """Get list of popular TIMM architectures suitable for the datasets"""
    return [
        # ResNet family
        'resnet18', 'resnet34', 'resnet50',
        # EfficientNet family  
        'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
        # Vision Transformer family
        'vit_tiny_patch16_224', 'vit_small_patch16_224', 'vit_base_patch16_224',
        # MobileNet family
        'mobilenetv3_small_100', 'mobilenetv3_large_100',
        # DenseNet family
        'densenet121', 'densenet169',
        # RegNet family
        'regnetx_002', 'regnetx_004', 'regnetx_006',
        # ConvNeXt family
        'convnext_tiny', 'convnext_small',
        # Swin Transformer family
        'swin_tiny_patch4_window7_224', 'swin_small_patch4_window7_224'
    ]


def simple_feature_extractor(model, x):
    """Extract features from TIMM model"""
    try:
        # Use TIMM's forward_features method if available
        if hasattr(model, 'forward_features'):
            features = model.forward_features(x)
            # Handle different output formats
            if isinstance(features, tuple):
                features = features[0]  # Take first element if tuple
            # Flatten if needed
            if len(features.shape) > 2:
                features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
            return features
        else:
            # Fallback: use full forward pass
            with torch.no_grad():
                features = model(x)
            return features
    except Exception as e:
        print(f"Feature extraction error: {e}, using fallback")
        # Ultimate fallback
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
    parser = argparse.ArgumentParser(description='Memory-Augmented Strategic Coreset Selection (MASCS)')
    parser.add_argument('--datasets', nargs='+', 
                       default=['CIFAR10'], 
                       choices=['CIFAR10', 'CIFAR100', 'MNIST', 'FashionMNIST', 'SVHN', 'STL10'],
                       help='Datasets to run experiments on')
    parser.add_argument('--architectures', nargs='+', 
                       default=['resnet18'], 
                       help='Model architectures to use (TIMM model names)')
    parser.add_argument('--data_percentages', nargs='+', type=float,
                       default=[100.0],
                       help='Percentage of training data to use (e.g., 10 20 30 50 70 100)')
    parser.add_argument('--budget', type=int, default=5000, 
                       help='Coreset budget (number of samples)')
    parser.add_argument('--epochs', type=int, default=50, 
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, 
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, 
                       help='Learning rate')
    parser.add_argument('--memory_window', type=int, default=100, 
                       help='Memory window size for each sample')
    parser.add_argument('--data_dir', type=str, default='/Users/tanmoy/research/data', 
                       help='Directory to store datasets')
    parser.add_argument('--log_dir', type=str, default='./logs', 
                       help='Directory to store tensorboard logs')
    parser.add_argument('--device', type=str, default='auto', 
                       help='Device to use (cuda/mps/cpu/auto)')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed')
    parser.add_argument('--save_results', type=str, default='mascs_results.json', 
                       help='File to save results')
    parser.add_argument('--optimize_weights', action='store_true',
                       help='Use GP to optimize strategy weights')
    parser.add_argument('--gp_calls', type=int, default=15,
                       help='Number of GP optimization calls per strategy')
    parser.add_argument('--weight_optimization_epoch', type=int, default=10,
                       help='Epoch at which to run weight optimization')
    parser.add_argument('--pretrained', action='store_true',
                       help='Use pretrained models')
    parser.add_argument('--list_architectures', action='store_true',
                       help='List available architectures and exit')
    
    args = parser.parse_args()
    
    # List architectures if requested
    if args.list_architectures:
        print("Available architectures:")
        for arch in get_available_architectures():
            print(f"  - {arch}")
        return
    
    # Set device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create log directory
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    
    results = {}
    
    # Run experiments on each combination
    total_experiments = len(args.datasets) * len(args.architectures) * len(args.data_percentages)
    experiment_count = 0
    
    for dataset_name in args.datasets:
        for architecture in args.architectures:
            for data_percentage in args.data_percentages:
                experiment_count += 1
                experiment_key = f"{dataset_name}_{architecture}_{data_percentage}%"
                
                print(f"\n{'='*80}")
                print(f"Experiment {experiment_count}/{total_experiments}: {experiment_key}")
                print(f"Dataset: {dataset_name} | Architecture: {architecture} | Data: {data_percentage}%")
                print(f"{'='*80}")
                
                # Load dataset with specified data percentage
                try:
                    train_dataset, test_dataset, num_classes = get_dataset(
                        dataset_name, args.data_dir, data_percentage
                    )
                except ValueError as e:
                    print(f"Error loading {dataset_name}: {e}")
                    continue
                
                # Create data loaders
                train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
                
                # Create model with specified architecture
                model = create_model(architecture, num_classes, device, args.pretrained)
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
                loss_fn = nn.CrossEntropyLoss()
                
                # Initialize MASCS selector
                selector = MemoryAugmentedCoresetSelector(
                    train_dataset, args.budget, simple_feature_extractor, 
                    num_classes, args.memory_window, device
                )
                
                # Initialize tensorboard logger
                log_path = os.path.join(args.log_dir, f"mascs_{experiment_key}")
                writer = SummaryWriter(log_path)
                
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
                    'value_losses': []
                }
                
                prev_performance = {'accuracy': 0.0, 'loss': float('inf')}
                weights_optimized = False
                
                for epoch in tqdm.tqdm(range(args.epochs), desc=f"Training {experiment_key}"):
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
                'val_accuracy': test_acc
            }
            
            # Compute memory statistics for state encoding with caching
            cache_key = f"memory_stats_{epoch}"
            if cache_key in selector.cache['memory_stats']:
                memory_stats = selector.cache['memory_stats'][cache_key]
            else:
                memory_stats = {}
                # Sample subset for efficiency (instead of all 50K samples)
                sample_indices = np.random.choice(len(train_dataset), min(1000, len(train_dataset)), replace=False)
                for strategy in selector.strategy_names:
                    strategy_scores = []
                    for i in sample_indices:
                        temporal_features = selector.compute_temporal_features(i)
                        score = selector.compute_temporal_bonus(temporal_features, strategy)
                        strategy_scores.append(score)
                    memory_stats[strategy] = strategy_scores
                selector.cache['memory_stats'][cache_key] = memory_stats
            
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
            
            # Select strategy using MDP policy
            strategy = selector.select_strategy(current_state)
            selector.current_strategy = strategy
            
            # Compute reward (improvement in validation accuracy)
            reward = current_performance['val_accuracy'] - prev_performance['accuracy']
            selector.reward_history.append(reward)
            
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
            
            # Store results
            dataset_results['train_losses'].append(train_loss)
            dataset_results['train_accuracies'].append(train_acc)
            dataset_results['test_losses'].append(test_loss)
            dataset_results['test_accuracies'].append(test_acc)
            dataset_results['strategies_used'].append(strategy)
            dataset_results['rewards'].append(reward)
            
            # Log to tensorboard
            writer.add_scalar(f'Training/Loss', train_loss, epoch)
            writer.add_scalar(f'Training/Accuracy', train_acc, epoch)
            writer.add_scalar(f'Testing/Loss', test_loss, epoch)
            writer.add_scalar(f'Testing/Accuracy', test_acc, epoch)
            writer.add_scalar(f'Strategy/Reward', reward, epoch)
            writer.add_scalar(f'Strategy/Action', selector.strategy_names.index(strategy), epoch)
            
            # Log score distributions
            for score_type, score_values in all_scores.items():
                writer.add_histogram(f'Scores/{score_type}', score_values, epoch)
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{args.epochs}: "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                      f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, "
                      f"Strategy: {strategy}, Reward: {reward:.4f}")
            
            prev_performance = current_performance
            
            # Clean up cache every few epochs to prevent memory bloat
            if epoch % 5 == 0:
                selector.cleanup_cache(epoch)
        
                writer.close()
                results[experiment_key] = dataset_results
        
                print(f"\nFinal Results for {experiment_key}:")
        print(f"Final Test Accuracy: {dataset_results['test_accuracies'][-1]:.2f}%")
        print(f"Best Test Accuracy: {max(dataset_results['test_accuracies']):.2f}%")
        print(f"Average Reward: {np.mean(dataset_results['rewards']):.4f}")
        
        # Strategy usage statistics
        strategy_counts = {}
        for strategy in dataset_results['strategies_used']:
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        print("Strategy Usage:")
        for strategy, count in strategy_counts.items():
            percentage = (count / len(dataset_results['strategies_used'])) * 100
            print(f"  {strategy}: {count} times ({percentage:.1f}%)")
        
        # Show optimized weights if used
        if args.optimize_weights and weights_optimized:
            print("\nOptimized Strategy Weights:")
            for strategy, weights in selector.strategies.items():
                print(f"  {strategy}: {selector.gp_optimizer.format_weights(weights)}")
    
    # Save results
    with open(args.save_results, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for dataset, data in results.items():
            json_results[dataset] = {}
            for key, value in data.items():
                if isinstance(value, list):
                    json_results[dataset][key] = value
                else:
                    json_results[dataset][key] = str(value)
        
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to {args.save_results}")
    print(f"Tensorboard logs saved to {args.log_dir}")


if __name__ == "__main__":
    main()