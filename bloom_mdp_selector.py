"""
Bloom Filter + MDP-based Dataset Selection
==========================================

Implements memory-efficient dataset selection using:
1. Bloom filters for duplicate detection
2. MDP for adaptive strategy selection
3. Neural cache with compression
4. Sample memory tracking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import hashlib
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
from dataclasses import dataclass
import random
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class SelectionStrategy(Enum):
    EXPLORE = "π_explore"      # Focus on volatile/uncertain samples
    EXPLOIT = "π_exploit"      # Select proven high-impact samples
    REFRESH = "π_refresh"      # Choose stale samples
    BALANCE = "π_balance"      # Maintain diversity
    FOCUS = "π_focus"          # Target difficult samples

@dataclass
class SampleMemory:
    """Memory for individual sample across time"""
    uncertainty_history: List[float]
    gradient_history: List[float] 
    selection_history: List[bool]
    impact_history: List[float]
    last_selected: int
    volatility: float = 0.0
    staleness: int = 0

class BloomFilter:
    """Probabilistic data structure for duplicate detection"""
    
    def __init__(self, capacity: int, error_rate: float = 0.01):
        self.capacity = capacity
        self.error_rate = error_rate
        self.bit_array_size = self._calculate_bit_array_size()
        self.hash_count = self._calculate_hash_count()
        self.bit_array = [False] * self.bit_array_size
        
    def _calculate_bit_array_size(self) -> int:
        return int(-self.capacity * np.log(self.error_rate) / (np.log(2) ** 2))
    
    def _calculate_hash_count(self) -> int:
        return int(self.bit_array_size * np.log(2) / self.capacity)
    
    def _hash(self, item: str, seed: int) -> int:
        return int(hashlib.md5(f"{item}{seed}".encode()).hexdigest(), 16) % self.bit_array_size
    
    def add(self, item: str):
        """Add item to bloom filter"""
        for i in range(self.hash_count):
            index = self._hash(item, i)
            self.bit_array[index] = True
    
    def __contains__(self, item: str) -> bool:
        """Check if item might be in the set"""
        for i in range(self.hash_count):
            index = self._hash(item, i)
            if not self.bit_array[index]:
                return False
        return True

class NeuralCache:
    """Memory-efficient neural cache with compression"""
    
    def __init__(self, cache_size: int = 10000, feature_dim: int = 512):
        self.cache_size = cache_size
        self.feature_dim = feature_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        
        # Compressed storage
        self.keys = torch.zeros(cache_size, feature_dim, dtype=torch.float16, device=self.device)
        self.values = torch.zeros(cache_size, dtype=torch.uint8, device=self.device)
        self.scores_quantized = torch.zeros(cache_size, dtype=torch.uint16, device=self.device)
        self.timestamps = torch.zeros(cache_size, dtype=torch.int32, device=self.device)
        
        # Metadata
        self.ptr = 0
        self.age = 0
        self.score_min = 0.0
        self.score_max = 1.0
        
        # Bloom filter for fast duplicate detection
        self.bloom_filter = BloomFilter(capacity=cache_size * 2, error_rate=0.01)
    
    def _get_feature_hash(self, features: torch.Tensor) -> str:
        """Generate hash for feature vector"""
        return hashlib.md5(features.cpu().numpy().tobytes()).hexdigest()
    
    def add_batch(self, features: torch.Tensor, values: torch.Tensor, scores: torch.Tensor):
        """Add batch of samples to cache with compression"""
        # Compress features to float16
        features_compressed = features.half()
        
        # Update score range for quantization
        self.score_min = min(self.score_min, scores.min().item())
        self.score_max = max(self.score_max, scores.max().item())
        
        # Quantize scores to uint16
        if self.score_max > self.score_min:
            scores_normalized = (scores - self.score_min) / (self.score_max - self.score_min)
        else:
            scores_normalized = torch.zeros_like(scores)
        scores_quantized = (scores_normalized * 65535).to(torch.uint16)
        
        batch_size = features.size(0)
        
        for i in range(batch_size):
            # Check bloom filter first
            feature_hash = self._get_feature_hash(features[i])
            
            if feature_hash not in self.bloom_filter:
                # Find replacement position
                if self.ptr < self.cache_size:
                    idx = self.ptr
                    self.ptr += 1
                else:
                    # Replace oldest entry
                    idx = self.timestamps.argmin().item()
                
                # Add to cache
                self.keys[idx] = features_compressed[i]
                self.values[idx] = values[i].to(torch.uint8)
                self.scores_quantized[idx] = scores_quantized[i]
                self.timestamps[idx] = self.age
                self.age += 1
                self.bloom_filter.add(feature_hash)

class MDPState:
    """State representation for MDP"""
    
    def __init__(self):
        self.performance_metrics = {}
        self.memory_statistics = {}
        self.exploration_potential = 0.0
        self.average_staleness = 0.0
        self.reward_history = deque(maxlen=100)
    
    def to_tensor(self) -> torch.Tensor:
        """Convert state to tensor representation"""
        state_vector = [
            self.performance_metrics.get('accuracy', 0.0),
            self.performance_metrics.get('loss', 0.0),
            self.memory_statistics.get('volatility_mean', 0.0),
            self.memory_statistics.get('volatility_std', 0.0),
            self.exploration_potential,
            self.average_staleness,
            np.mean(self.reward_history) if self.reward_history else 0.0
        ]
        return torch.tensor(state_vector, dtype=torch.float32)

class SelectionPolicy(nn.Module):
    """Neural network policy for MDP"""
    
    def __init__(self, state_dim: int = 7, hidden_dim: int = 128, n_strategies: int = 5):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_strategies),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

class DatasetSelectorMDP:
    """Main MDP-based dataset selector with bloom filtering"""
    
    def __init__(self, 
                 dataset_size: int,
                 cache_size: int = 10000,
                 feature_dim: int = 512,
                 learning_rate: float = 1e-3,
                 epsilon: float = 0.3,
                 epsilon_decay: float = 0.995,
                 gamma: float = 0.95):
        
        self.dataset_size = dataset_size
        self.cache_size = cache_size
        self.feature_dim = feature_dim
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        
        # Components
        self.neural_cache = NeuralCache(cache_size, feature_dim)
        self.policy = SelectionPolicy()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Memory tracking for each sample
        self.sample_memories = {i: SampleMemory([], [], [], [], -1) for i in range(dataset_size)}
        
        # MDP state
        self.current_state = MDPState()
        
        # Strategy weights for each action
        self.strategy_weights = {
            SelectionStrategy.EXPLORE: {'uncertainty': 1.0, 'volatility': 0.8, 'gradient': 0.3},
            SelectionStrategy.EXPLOIT: {'impact': 1.0, 'gradient': 0.7, 'uncertainty': 0.2},
            SelectionStrategy.REFRESH: {'staleness': 1.0, 'diversity': 0.5},
            SelectionStrategy.BALANCE: {'diversity': 1.0, 'uncertainty': 0.5, 'impact': 0.5},
            SelectionStrategy.FOCUS: {'difficulty': 1.0, 'gradient': 0.8}
        }
        
        # Training history
        self.training_step = 0
        self.reward_history = []
        
    def _compute_sample_scores(self, features: torch.Tensor, indices: List[int], 
                              model=None, targets=None) -> Dict[str, torch.Tensor]:
        """Compute multiple scoring functions for samples"""
        batch_size = features.size(0)
        device = features.device
        
        scores = {
            'uncertainty': torch.randn(batch_size, device=device).abs(),  # Placeholder
            'gradient': torch.randn(batch_size, device=device).abs(),     # Placeholder
            'diversity': torch.randn(batch_size, device=device).abs(),    # Placeholder
            'difficulty': torch.randn(batch_size, device=device).abs(),   # Placeholder
            'impact': torch.zeros(batch_size, device=device),
            'staleness': torch.zeros(batch_size, device=device),
            'volatility': torch.zeros(batch_size, device=device)
        }
        
        # Compute real scores if model is provided
        if model is not None:
            scores['uncertainty'] = self._compute_uncertainty(model, features)
            if targets is not None:
                scores['gradient'] = self._compute_gradient_magnitude(model, features, targets)
            scores['difficulty'] = self._compute_difficulty(model, features)
            scores['diversity'] = self._compute_diversity(features)
        
        # Fill memory-based scores
        for i, idx in enumerate(indices):
            if idx < len(self.sample_memories):
                memory = self.sample_memories[idx]
                scores['impact'][i] = np.mean(memory.impact_history) if memory.impact_history else 0.0
                scores['staleness'][i] = memory.staleness
                scores['volatility'][i] = memory.volatility
            
        return scores
    
    def _compute_uncertainty(self, model, features):
        """Compute uncertainty using MC dropout"""
        model.train()  # Enable dropout
        predictions = []
        
        for _ in range(5):  # Reduced for efficiency
            with torch.no_grad():
                pred = F.softmax(model(features), dim=1)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        uncertainty = torch.var(predictions, dim=0).mean(dim=1)
        return uncertainty
    
    def _compute_gradient_magnitude(self, model, features, targets):
        """Compute gradient magnitude for influence estimation"""
        features = features.clone().detach().requires_grad_(True)
        output = model(features)
        loss = F.cross_entropy(output, targets)
        
        gradients = torch.autograd.grad(loss, features, retain_graph=False)[0]
        gradient_magnitude = torch.norm(gradients.view(features.size(0), -1), dim=1)
        return gradient_magnitude
    
    def _compute_difficulty(self, model, features):
        """Compute sample difficulty based on prediction confidence"""
        model.eval()
        with torch.no_grad():
            outputs = F.softmax(model(features), dim=1)
            max_probs, _ = torch.max(outputs, dim=1)
            difficulty = 1.0 - max_probs  # Lower confidence = higher difficulty
        return difficulty
    
    def _compute_diversity(self, features):
        """Compute diversity score based on feature distances"""
        # Normalize features
        features_norm = F.normalize(features, p=2, dim=1)
        
        # Compute pairwise distances
        distances = torch.cdist(features_norm, features_norm)
        
        # Average distance to k nearest neighbors
        k = min(5, features.size(0) - 1)
        if k > 0:
            knn_distances, _ = torch.topk(distances, k+1, dim=1, largest=False)
            diversity = knn_distances[:, 1:].mean(dim=1)  # Exclude self
        else:
            diversity = torch.zeros(features.size(0), device=features.device)
        
        return diversity
    
    def _update_sample_memories(self, indices: List[int], scores: Dict[str, torch.Tensor], selected_mask: torch.Tensor):
        """Update memory for processed samples"""
        for i, idx in enumerate(indices):
            if idx < len(self.sample_memories):
                memory = self.sample_memories[idx]
                
                # Update histories
                memory.uncertainty_history.append(scores['uncertainty'][i].item())
                memory.gradient_history.append(scores['gradient'][i].item())
                memory.selection_history.append(selected_mask[i].item())
                
                # Update derived metrics
                if len(memory.uncertainty_history) > 1:
                    memory.volatility = np.var(memory.uncertainty_history[-10:])  # Recent volatility
                
                memory.staleness = self.training_step - memory.last_selected if memory.last_selected >= 0 else self.training_step
                
                if selected_mask[i]:
                    memory.last_selected = self.training_step
    
    def _update_mdp_state(self, performance_metrics: Dict[str, float]):
        """Update MDP state based on current training metrics"""
        self.current_state.performance_metrics.update(performance_metrics)
        
        # Compute memory statistics
        all_volatilities = [mem.volatility for mem in self.sample_memories.values()]
        self.current_state.memory_statistics = {
            'volatility_mean': np.mean(all_volatilities) if all_volatilities else 0.0,
            'volatility_std': np.std(all_volatilities) if all_volatilities else 0.0
        }
        
        # Compute exploration potential
        never_selected = sum(1 for mem in self.sample_memories.values() if mem.last_selected == -1)
        self.current_state.exploration_potential = never_selected / self.dataset_size
        
        # Compute average staleness
        staleness_values = [mem.staleness for mem in self.sample_memories.values()]
        self.current_state.average_staleness = np.mean(staleness_values) if staleness_values else 0.0
    
    def _select_strategy(self, state: torch.Tensor) -> SelectionStrategy:
        """Select strategy using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            # Random exploration
            return random.choice(list(SelectionStrategy))
        else:
            # Policy-based selection
            with torch.no_grad():
                action_probs = self.policy(state)
                action_idx = torch.multinomial(action_probs, 1).item()
                return list(SelectionStrategy)[action_idx]
    
    def _compute_final_scores(self, scores: Dict[str, torch.Tensor], strategy: SelectionStrategy) -> torch.Tensor:
        """Combine scores based on selected strategy"""
        weights = self.strategy_weights[strategy]
        final_scores = torch.zeros_like(scores['uncertainty'])
        
        for score_type, weight in weights.items():
            if score_type in scores:
                final_scores += weight * scores[score_type]
        
        return final_scores
    
    def select_batch(self, 
                    features: torch.Tensor, 
                    indices: List[int], 
                    budget: int,
                    performance_metrics: Dict[str, float],
                    model=None,
                    targets=None) -> Tuple[List[int], SelectionStrategy]:
        """Main batch selection method"""
        
        # Update MDP state
        self._update_mdp_state(performance_metrics)
        state_tensor = self.current_state.to_tensor()
        
        # Select strategy
        strategy = self._select_strategy(state_tensor)
        
        # Compute scores for all samples
        scores = self._compute_sample_scores(features, indices, model, targets)
        
        # Combine scores based on strategy
        final_scores = self._compute_final_scores(scores, strategy)
        
        # Select top-k samples
        _, top_indices = torch.topk(final_scores, min(budget, len(indices)))
        selected_indices = [indices[i] for i in top_indices.tolist()]
        
        # Create selection mask for memory update
        selected_mask = torch.zeros(len(indices), dtype=torch.bool)
        selected_mask[top_indices] = True
        
        # Update sample memories
        self._update_sample_memories(indices, scores, selected_mask)
        
        # Add to neural cache (only novel samples)
        self.neural_cache.add_batch(features, torch.tensor(indices), final_scores)
        
        # Update training step
        self.training_step += 1
        self.epsilon *= self.epsilon_decay
        
        logger.info(f"Selected {len(selected_indices)} samples using strategy: {strategy.value}")
        
        return selected_indices, strategy
    
    def update_policy(self, reward: float, next_performance_metrics: Dict[str, float]):
        """Update policy based on reward"""
        self.current_state.reward_history.append(reward)
        self.reward_history.append(reward)
        
        # Simple policy gradient update
        if len(self.reward_history) > 10:
            # Compute advantage
            baseline = np.mean(self.reward_history[-10:])
            advantage = reward - baseline
            
            # Update policy
            state_tensor = self.current_state.to_tensor()
            action_probs = self.policy(state_tensor)
            
            # Simplified loss
            loss = -torch.log(action_probs.max()) * advantage
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics"""
        return {
            'training_step': self.training_step,
            'epsilon': self.epsilon,
            'cache_utilization': self.neural_cache.ptr / self.neural_cache.cache_size,
            'exploration_potential': self.current_state.exploration_potential,
            'average_staleness': self.current_state.average_staleness,
            'recent_rewards': list(self.current_state.reward_history)[-10:] if self.current_state.reward_history else [],
            'bloom_filter_size': len([b for b in self.neural_cache.bloom_filter.bit_array if b])
        }


# Integration helper for GALORE framework
class GALOREMDPIntegration:
    """Integration layer for GALORE framework"""
    
    def __init__(self, dataset_size: int, feature_dim: int = 512, config=None):
        self.mdp_selector = DatasetSelectorMDP(
            dataset_size=dataset_size,
            cache_size=config.cache_size if config else 10000,
            feature_dim=feature_dim,
            learning_rate=config.mdp_lr if config else 1e-3,
            epsilon=config.epsilon if config else 0.3,
            epsilon_decay=config.epsilon_decay if config else 0.995
        )
        
        self.previous_performance = {'accuracy': 0.0, 'loss': float('inf')}
        
    def select_coreset(self, model, train_dataset, budget, current_performance=None):
        """Select coreset using MDP strategy"""
        if current_performance is None:
            current_performance = self.previous_performance
        
        # Extract features for dataset
        features = self._extract_features(model, train_dataset)
        indices = list(range(len(train_dataset)))
        
        # Select using MDP
        selected_indices, strategy = self.mdp_selector.select_batch(
            features, indices, budget, current_performance, model
        )
        
        # Compute reward if we have performance improvement
        if self.previous_performance['accuracy'] > 0:
            accuracy_improvement = current_performance['accuracy'] - self.previous_performance['accuracy']
            loss_improvement = self.previous_performance['loss'] - current_performance['loss']
            reward = 0.7 * accuracy_improvement + 0.3 * loss_improvement
            
            # Update policy
            self.mdp_selector.update_policy(reward, current_performance)
        
        self.previous_performance = current_performance.copy()
        
        return selected_indices, strategy
    
    def _extract_features(self, model, dataset, batch_size=256):
        """Extract features from dataset using model"""
        model.eval()
        features_list = []
        
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            for data, _ in loader:
                # Get features from second-to-last layer
                features = model.extract_features(data) if hasattr(model, 'extract_features') else data
                features_list.append(features)
        
        return torch.cat(features_list, dim=0)
    
    def get_statistics(self):
        """Get MDP selector statistics"""
        return self.mdp_selector.get_statistics()


# Example usage specific to GALORE
def integrate_with_galore_experiments():
    """Example of integrating MDP selector with GALORE experiments"""
    
    # This would be called from simple_expts.py
    from torch.utils.data import DataLoader, Subset
    
    def enhanced_coreset_selection(selector, model, train_dataset, budget, performance_metrics):
        """Enhanced coreset selection using MDP"""
        
        # Initialize MDP integration if not exists
        if not hasattr(selector, 'mdp_integration'):
            selector.mdp_integration = GALOREMDPIntegration(
                dataset_size=len(train_dataset),
                feature_dim=512  # Adjust based on your model
            )
        
        # Select coreset using MDP
        selected_indices, strategy = selector.mdp_integration.select_coreset(
            model, train_dataset, budget, performance_metrics
        )
        
        # Log statistics
        stats = selector.mdp_integration.get_statistics()
        logger.info(f"MDP Stats - Cache: {stats['cache_utilization']:.2%}, "
                   f"Exploration: {stats['exploration_potential']:.2%}, "
                   f"Strategy: {strategy.value}")
        
        return selected_indices, {'strategy': strategy.value, 'stats': stats}
    
    return enhanced_coreset_selection


if __name__ == "__main__":
    # Test the implementation
    print("Testing MDP Dataset Selector...")
    
    # Create dummy dataset
    dataset_size = 1000
    feature_dim = 128
    
    # Initialize selector
    selector = DatasetSelectorMDP(
        dataset_size=dataset_size,
        cache_size=100,
        feature_dim=feature_dim
    )
    
    # Simulate selection
    for epoch in range(5):
        print(f"\n=== Epoch {epoch} ===")
        
        # Simulate batch
        batch_size = 100
        indices = random.sample(range(dataset_size), batch_size)
        features = torch.randn(batch_size, feature_dim)
        
        # Performance metrics
        performance = {
            'accuracy': 0.7 + epoch * 0.05,
            'loss': 2.0 - epoch * 0.2
        }
        
        # Select subset
        selected, strategy = selector.select_batch(
            features, indices, 20, performance
        )
        
        print(f"Selected {len(selected)} samples using {strategy.value}")
        
        # Update policy with reward
        reward = random.uniform(0.0, 0.1)
        selector.update_policy(reward, performance)
        
        # Print statistics
        stats = selector.get_statistics()
        print(f"Stats: {stats}")
    
    print("\nTest completed successfully!")