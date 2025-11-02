"""
PyTorch RL-compatible MDP formulation for Coreset Selection
Clean implementation using standard RL interfaces
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import gym
from gym import spaces
from collections import defaultdict, Counter


# =============================================================================
# MDP State Definition
# =============================================================================

@dataclass
class CoresetMDPState:
    """
    MDP State for coreset selection
    Designed to be simple and compatible with gym environments
    """
    # Core performance metrics (5 features)
    train_accuracy: float = 0.0
    val_accuracy: float = 0.0  
    train_loss: float = 10.0
    val_loss: float = 10.0
    loss_improvement: float = 0.0  # Recent improvement
    
    # Training progress (3 features)
    epoch_ratio: float = 0.0  # epoch / total_epochs
    samples_ratio: float = 0.0  # selected_samples / budget
    selection_round: int = 0  # Which selection round we're in
    
    # Class balance (10 features for CIFAR-10)
    class_proportions: List[float] = None  # Proportion of each class
    class_entropy: float = 0.0  # Entropy of class distribution
    
    # Strategy performance (12 features - 2 per strategy)
    strategy_rewards: List[float] = None  # Recent rewards for each strategy
    strategy_usage: List[float] = None  # How much each strategy is used
    
    # Simple diversity metrics (3 features)
    selection_diversity: float = 0.5  # How diverse recent selections are
    feature_spread: float = 0.5  # Spread in feature space
    redundancy: float = 0.0  # How redundant selections are
    
    def __post_init__(self):
        """Initialize lists with defaults if None"""
        if self.class_proportions is None:
            self.class_proportions = [0.0] * 10  # For CIFAR-10
        if self.strategy_rewards is None:
            self.strategy_rewards = [0.0] * 6  # 6 strategies
        if self.strategy_usage is None:
            self.strategy_usage = [1.0/6] * 6  # Equal initial usage
    
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array for gym compatibility"""
        features = []
        
        # Performance (5)
        features.extend([
            self.train_accuracy,
            self.val_accuracy,
            np.clip(self.train_loss / 10.0, 0, 1),  # Normalize
            np.clip(self.val_loss / 10.0, 0, 1),
            np.clip(self.loss_improvement, -1, 1)
        ])
        
        # Progress (3)
        features.extend([
            self.epoch_ratio,
            self.samples_ratio,
            min(self.selection_round / 20.0, 1.0)  # Normalize rounds
        ])
        
        # Class distribution (11 = 10 proportions + 1 entropy)
        features.extend(self.class_proportions)
        features.append(self.class_entropy)
        
        # Strategy info (12)
        features.extend(self.strategy_rewards)
        features.extend(self.strategy_usage)
        
        # Diversity (3)
        features.extend([
            self.selection_diversity,
            self.feature_spread,
            self.redundancy
        ])
        
        return np.array(features, dtype=np.float32)
    
    @property
    def dim(self) -> int:
        """State dimension"""
        return len(self.to_numpy())


# =============================================================================
# Gym Environment for Coreset Selection
# =============================================================================

class CoresetSelectionEnv(gym.Env):
    """
    OpenAI Gym-compatible environment for coreset selection
    """
    
    def __init__(self, 
                 dataset,
                 model_fn,  # Function to create a new model
                 budget: int,
                 val_dataset=None,
                 num_classes: int = 10,
                 selection_batch_size: int = 100,
                 device='cuda'):
        super().__init__()
        
        self.dataset = dataset
        self.model_fn = model_fn
        self.budget = budget
        self.val_dataset = val_dataset
        self.num_classes = num_classes
        self.selection_batch_size = selection_batch_size
        self.device = device
        
        # Define action and observation spaces
        # Action: weights for 6 strategies (normalized to sum to 1)
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(6,), dtype=np.float32
        )
        
        # Observation: MDP state
        dummy_state = CoresetMDPState()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(dummy_state.dim,), 
            dtype=np.float32
        )
        
        # Environment state
        self.selected_indices = set()
        self.current_state = None
        self.model = None
        self.selection_round = 0
        self.episode_rewards = []
        
        # Strategy scoring functions (placeholders - implement your actual ones)
        self.scoring_functions = {
            'S_U': self._uncertainty_scores,
            'S_D': self._diversity_scores,
            'S_C': self._class_balance_scores,
            'S_B': self._boundary_scores,
            'S_G': self._gradient_scores,
            'S_F': self._forgetting_scores
        }
        
    def reset(self):
        """Reset environment for new episode"""
        self.selected_indices = set()
        self.selection_round = 0
        self.episode_rewards = []
        
        # Create new model
        self.model = self.model_fn().to(self.device)
        
        # Create initial state
        self.current_state = CoresetMDPState(
            epoch_ratio=0.0,
            samples_ratio=0.0,
            selection_round=0
        )
        
        return self.current_state.to_numpy()
    
    def step(self, action):
        """
        Execute action (strategy weights) and return next state
        
        Args:
            action: numpy array of shape (6,) with strategy weights
            
        Returns:
            observation: next state
            reward: reward for this transition
            done: whether episode is finished
            info: additional information
        """
        # Normalize action to ensure it sums to 1
        action = np.abs(action)  # Ensure positive
        action = action / (action.sum() + 1e-8)
        
        # Determine how many samples to select this round
        remaining = self.budget - len(self.selected_indices)
        n_samples = min(self.selection_batch_size, remaining)
        
        if n_samples <= 0:
            # No more samples to select
            return self.current_state.to_numpy(), 0.0, True, {}
        
        # Select samples using weighted strategies
        new_indices = self._select_samples_weighted(action, n_samples)
        self.selected_indices.update(new_indices)
        
        # Train model on current coreset (simplified - just a few steps)
        train_metrics = self._quick_train_eval()
        
        # Compute class distribution
        class_dist = self._compute_class_distribution()
        
        # Create next state
        next_state = CoresetMDPState(
            train_accuracy=train_metrics['train_acc'],
            val_accuracy=train_metrics['val_acc'],
            train_loss=train_metrics['train_loss'],
            val_loss=train_metrics['val_loss'],
            loss_improvement=train_metrics['val_loss'] - self.current_state.val_loss,
            epoch_ratio=min(self.selection_round * 5 / 100, 1.0),  # Assume 100 epochs total
            samples_ratio=len(self.selected_indices) / self.budget,
            selection_round=self.selection_round + 1,
            class_proportions=class_dist['proportions'],
            class_entropy=class_dist['entropy'],
            strategy_rewards=self._compute_strategy_rewards(action, train_metrics),
            strategy_usage=action.tolist(),
            selection_diversity=self._compute_diversity_metric(),
            feature_spread=np.random.random() * 0.5 + 0.25,  # Placeholder
            redundancy=np.random.random() * 0.3  # Placeholder
        )
        
        # Compute reward
        reward = self._compute_reward(self.current_state, next_state, len(new_indices))
        self.episode_rewards.append(reward)
        
        # Update state
        self.current_state = next_state
        self.selection_round += 1
        
        # Check if done
        done = len(self.selected_indices) >= self.budget
        
        # Info dict
        info = {
            'selected_total': len(self.selected_indices),
            'selected_this_round': len(new_indices),
            'val_accuracy': train_metrics['val_acc'],
            'strategy_weights': action
        }
        
        return next_state.to_numpy(), reward, done, info
    
    def _select_samples_weighted(self, weights, n_samples):
        """Select samples using weighted combination of strategies"""
        strategy_names = ['S_U', 'S_D', 'S_C', 'S_B', 'S_G', 'S_F']
        
        # Allocate samples to each strategy
        allocations = {}
        remaining = n_samples
        
        for i, (strategy, weight) in enumerate(zip(strategy_names, weights)):
            if remaining <= 0:
                allocations[strategy] = 0
                continue
                
            alloc = int(weight * n_samples)
            alloc = min(alloc, remaining)
            allocations[strategy] = alloc
            remaining -= alloc
        
        # Give remaining to highest weight strategy
        if remaining > 0:
            best_strategy = strategy_names[np.argmax(weights)]
            allocations[best_strategy] += remaining
        
        # Select samples for each strategy
        new_indices = set()
        
        for strategy, count in allocations.items():
            if count == 0:
                continue
                
            # Get scores from strategy
            scores = self.scoring_functions[strategy]()
            
            # Mask already selected
            mask = np.ones(len(self.dataset), dtype=bool)
            mask[list(self.selected_indices)] = False
            mask[list(new_indices)] = False
            
            available_indices = np.where(mask)[0]
            if len(available_indices) == 0:
                continue
                
            available_scores = scores[mask]
            
            # Select top scoring
            n_select = min(count, len(available_indices))
            if n_select > 0:
                top_idx = np.argpartition(available_scores, -n_select)[-n_select:]
                selected = available_indices[top_idx]
                new_indices.update(selected.tolist())
        
        return new_indices
    
    def _quick_train_eval(self):
        """Quick training and evaluation on current coreset"""
        if len(self.selected_indices) == 0:
            return {
                'train_acc': 0.0,
                'val_acc': 0.0,
                'train_loss': 10.0,
                'val_loss': 10.0
            }
        
        # In practice, you would actually train here
        # For now, return mock improvements
        base_acc = len(self.selected_indices) / self.budget * 0.8
        return {
            'train_acc': base_acc + np.random.random() * 0.1,
            'val_acc': base_acc - 0.05 + np.random.random() * 0.1,
            'train_loss': 10.0 - base_acc * 5,
            'val_loss': 10.0 - (base_acc - 0.05) * 5
        }
    
    def _compute_class_distribution(self):
        """Compute class distribution of selected samples"""
        if len(self.selected_indices) == 0:
            return {
                'proportions': [0.0] * self.num_classes,
                'entropy': 0.0
            }
        
        # Count classes
        class_counts = Counter()
        for idx in self.selected_indices:
            _, label = self.dataset[idx]
            class_counts[label] += 1
        
        # Compute proportions
        total = len(self.selected_indices)
        proportions = [class_counts.get(i, 0) / total for i in range(self.num_classes)]
        
        # Compute entropy
        entropy = 0.0
        for p in proportions:
            if p > 0:
                entropy -= p * np.log(p)
        
        # Normalize entropy
        max_entropy = np.log(self.num_classes)
        entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return {
            'proportions': proportions,
            'entropy': entropy
        }
    
    def _compute_reward(self, prev_state: CoresetMDPState, 
                       curr_state: CoresetMDPState, 
                       n_new_samples: int) -> float:
        """
        Compute reward for the transition
        
        Components:
        - Performance improvement
        - Efficiency (improvement per sample)
        - Class balance (entropy)
        - Exploration bonus
        """
        # Performance improvement
        acc_improvement = curr_state.val_accuracy - prev_state.val_accuracy
        perf_reward = np.tanh(10 * acc_improvement)  # Scale and bound
        
        # Efficiency: improvement per sample
        if n_new_samples > 0:
            efficiency = acc_improvement / (n_new_samples / self.budget)
            eff_reward = np.tanh(5 * efficiency)
        else:
            eff_reward = 0.0
        
        # Class balance reward (high entropy is good)
        balance_reward = curr_state.class_entropy * 0.2
        
        # Exploration bonus for trying different strategies
        strategy_entropy = 0.0
        for usage in curr_state.strategy_usage:
            if usage > 0:
                strategy_entropy -= usage * np.log(usage)
        exploration_bonus = strategy_entropy * 0.1
        
        # Combine rewards
        total_reward = (
            0.5 * perf_reward +
            0.3 * eff_reward +
            0.1 * balance_reward +
            0.1 * exploration_bonus
        )
        
        return float(total_reward)
    
    def _compute_strategy_rewards(self, weights, metrics):
        """Estimate per-strategy rewards (simplified)"""
        # In practice, track which samples each strategy selected
        # and their contribution to performance
        base_reward = metrics['val_acc'] - self.current_state.val_accuracy
        
        # Distribute reward proportionally with some noise
        rewards = []
        for w in weights:
            reward = base_reward * w + np.random.normal(0, 0.01)
            rewards.append(float(reward))
        
        return rewards
    
    def _compute_diversity_metric(self):
        """Simple diversity metric based on class distribution"""
        if len(self.selected_indices) < 2:
            return 0.5
        
        # Use class entropy as a proxy for diversity
        class_dist = self._compute_class_distribution()
        return class_dist['entropy']
    
    # Placeholder scoring functions - implement your actual ones
    def _uncertainty_scores(self):
        """Compute uncertainty scores for all samples"""
        # In practice: use model predictions
        return np.random.rand(len(self.dataset))
    
    def _diversity_scores(self):
        """Compute diversity scores"""
        # In practice: use feature distances
        return np.random.rand(len(self.dataset))
    
    def _class_balance_scores(self):
        """Compute class balance scores"""
        # Prefer underrepresented classes
        scores = np.ones(len(self.dataset))
        
        if len(self.selected_indices) > 0:
            class_counts = Counter()
            for idx in self.selected_indices:
                _, label = self.dataset[idx]
                class_counts[label] += 1
            
            # Higher scores for underrepresented classes
            for i in range(len(self.dataset)):
                _, label = self.dataset[i]
                count = class_counts.get(label, 0)
                scores[i] = 1.0 / (count + 1)
        
        return scores
    
    def _boundary_scores(self):
        """Compute boundary proximity scores"""
        # In practice: use prediction margins
        return np.random.rand(len(self.dataset))
    
    def _gradient_scores(self):
        """Compute gradient-based importance scores"""
        # In practice: use gradient magnitudes
        return np.random.rand(len(self.dataset))
    
    def _forgetting_scores(self):
        """Compute forgetting scores"""
        # In practice: track prediction changes
        return np.random.rand(len(self.dataset))


# =============================================================================
# Simple Policy Network
# =============================================================================

class CoresetPolicyNetwork(nn.Module):
    """
    Simple MLP policy network for strategy weight prediction
    """
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 6),  # 6 strategies
            nn.Softmax(dim=-1)
        )
        
    def forward(self, state):
        return self.net(state)


# =============================================================================
# Training Loop Example
# =============================================================================

def train_with_rl(env, policy_net, num_episodes=100, lr=1e-3):
    """
    Simple training loop using REINFORCE
    """
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    
    all_rewards = []
    all_accuracies = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_log_probs = []
        episode_rewards = []
        
        done = False
        while not done:
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Get action from policy
            action_probs = policy_net(state_tensor)
            
            # Sample action (or use deterministic for evaluation)
            if episode < num_episodes * 0.8:  # Explore during training
                dist = torch.distributions.Categorical(action_probs[0])
                action_sample = dist.sample()
                log_prob = dist.log_prob(action_sample)
                
                # Convert to strategy weights
                action = torch.zeros(6)
                action[action_sample] = 1.0
                action = action.numpy()
                
                # Add some noise for exploration
                noise = np.random.dirichlet(np.ones(6) * 0.5)
                action = 0.8 * action + 0.2 * noise
            else:
                # Deterministic evaluation
                action = action_probs[0].detach().numpy()
                log_prob = torch.log(action_probs[0].max())
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            episode_rewards.append(reward)
            episode_log_probs.append(log_prob)
            
            state = next_state
            
            # Log progress
            if 'val_accuracy' in info:
                all_accuracies.append(info['val_accuracy'])
        
        all_rewards.append(episode_reward)
        
        # REINFORCE update
        if len(episode_rewards) > 0:
            # Compute returns
            returns = []
            G = 0
            for r in reversed(episode_rewards):
                G = r + 0.99 * G
                returns.insert(0, G)
            
            returns = torch.FloatTensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            # Policy gradient
            policy_loss = 0
            for log_prob, G in zip(episode_log_probs, returns):
                policy_loss += -log_prob * G
            
            # Update
            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()
        
        # Log
        if episode % 10 == 0:
            print(f"Episode {episode}: Reward = {episode_reward:.4f}, "
                  f"Samples = {len(env.selected_indices)}/{env.budget}")
    
    return all_rewards, all_accuracies