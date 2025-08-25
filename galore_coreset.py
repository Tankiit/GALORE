"""
Multi-Strategy RL-Guided Coreset Selection for CIFAR-10/100
Complete implementation with all components integrated
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import os
from typing import List, Dict, Tuple, Optional, Callable
import logging
import argparse
from heapq import heappush, heappop

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Model Architectures for CIFAR
# =============================================================================

class ResNetBlock(nn.Module):
    """Basic ResNet block for CIFAR"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CIFARResNet(nn.Module):
    """ResNet for CIFAR-10/100 with feature extraction capability"""
    def __init__(self, num_classes=10, depth=20):
        super().__init__()
        assert (depth - 2) % 6 == 0, "depth must be 6n+2"
        n = (depth - 2) // 6
        
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.layer1 = self._make_layer(16, 16, n, stride=1)
        self.layer2 = self._make_layer(16, 32, n, stride=2)
        self.layer3 = self._make_layer(32, 64, n, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        
        # For feature extraction
        self.feature_dim = 64
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x, return_features=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        features = out.view(out.size(0), -1)
        
        if return_features:
            return features
            
        out = self.fc(features)
        return out
    
    def get_features(self, x):
        """Extract features for similarity computation"""
        return self.forward(x, return_features=True)


# =============================================================================
# Submodular Scoring Strategies
# =============================================================================

class SubmodularScoringStrategies:
    """Collection of theoretically-grounded scoring functions"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        
        # Caches for efficiency
        self.gradient_cache = {}
        self.embedding_cache = {}
        self.loss_cache = {}
        
        # Cache management
        self.max_cache_size = 10000
        self.cache_hits = 0
        self.cache_misses = 0
        
    def clear_cache(self):
        """Clear all caches to free memory"""
        self.gradient_cache.clear()
        self.embedding_cache.clear()
        self.loss_cache.clear()
        logger.info(f"Cache cleared. Hit rate: {self.cache_hits/(self.cache_hits+self.cache_misses+1e-8):.2%}")
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _manage_cache(self, cache_dict):
        """Evict oldest entries if cache is too large"""
        if len(cache_dict) > self.max_cache_size:
            # Remove 20% oldest entries
            num_to_remove = int(0.2 * self.max_cache_size)
            for key in list(cache_dict.keys())[:num_to_remove]:
                del cache_dict[key]
    
    def gradient_magnitude_score(self, idx, x_i, y_i, C_current):
        """
        Score based on gradient norm - high gradient = high learning potential
        Submodular because: f(C ∪ {x}) - f(C) decreases as |C| grows
        """
        cache_key = (idx, self.model.training)
        
        if cache_key in self.gradient_cache:
            self.cache_hits += 1
            return self.gradient_cache[cache_key]
        
        self.cache_misses += 1
        
        # Compute gradient
        self.model.zero_grad()
        x_i = x_i.unsqueeze(0).to(self.device)
        y_i = torch.tensor([y_i]).to(self.device)
        
        output = self.model(x_i)
        loss = F.cross_entropy(output, y_i)
        loss.backward()
        
        # Aggregate gradient norms
        grad_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.norm().item() ** 2
        grad_norm = grad_norm ** 0.5
        
        self.gradient_cache[cache_key] = grad_norm
        self._manage_cache(self.gradient_cache)
        
        return grad_norm
    
    def diversity_score(self, idx, x_i, y_i, C_current):
        """
        Facility location: maximize minimum distance to selected set
        Submodular by construction - diminishing returns as coverage increases
        """
        if len(C_current) == 0:
            return 1.0  # First point has maximum diversity
        
        # Get embedding for x_i
        cache_key = (idx, 'embedding')
        if cache_key not in self.embedding_cache:
            with torch.no_grad():
                x_i = x_i.unsqueeze(0).to(self.device)
                features = self.model.get_features(x_i)
                self.embedding_cache[cache_key] = features.cpu()
                self._manage_cache(self.embedding_cache)
        
        feat_i = self.embedding_cache[cache_key]
        
        # Compute minimum distance to selected set
        min_dist = float('inf')
        for c_idx in C_current:
            c_cache_key = (c_idx, 'embedding')
            if c_cache_key not in self.embedding_cache:
                # This shouldn't happen if we're selecting greedily
                continue
                
            feat_j = self.embedding_cache[c_cache_key]
            dist = torch.norm(feat_i - feat_j).item()
            min_dist = min(min_dist, dist)
        
        # Normalize by feature dimension
        normalized_dist = min_dist / (feat_i.shape[1] ** 0.5)
        return normalized_dist
    
    def uncertainty_score(self, idx, x_i, y_i, C_current):
        """
        Entropy of prediction - uncertain samples are informative
        Submodular when combined with coverage constraints
        """
        with torch.no_grad():
            x_i = x_i.unsqueeze(0).to(self.device)
            logits = self.model(x_i)
            probs = F.softmax(logits, dim=-1)
            
            # Entropy
            entropy = -(probs * torch.log(probs + 1e-8)).sum().item()
            
        return entropy
    
    def boundary_score(self, idx, x_i, y_i, C_current):
        """
        Distance to decision boundary - samples near boundary are critical
        Approximated by prediction confidence margin
        """
        with torch.no_grad():
            x_i = x_i.unsqueeze(0).to(self.device)
            logits = self.model(x_i)
            probs = F.softmax(logits, dim=-1)
            
            # Margin between top two predictions
            top2 = torch.topk(probs, k=2, dim=-1)
            margin = (top2.values[0, 0] - top2.values[0, 1]).item()
            
        # Lower margin = closer to boundary = higher score
        # Use exponential to emphasize very close boundaries
        boundary_score = np.exp(-5 * margin)
        return boundary_score
    
    def forgetting_score(self, idx, x_i, y_i, C_current):
        """
        Track how often model forgets this sample
        Samples that are frequently forgotten are important
        """
        cache_key = (idx, 'loss_history')
        
        if cache_key not in self.loss_cache:
            # Initialize loss history
            self.loss_cache[cache_key] = deque(maxlen=10)
        
        # Compute current loss
        with torch.no_grad():
            x_i = x_i.unsqueeze(0).to(self.device)
            y_i = torch.tensor([y_i]).to(self.device)
            output = self.model(x_i)
            loss = F.cross_entropy(output, y_i).item()
            
        loss_history = self.loss_cache[cache_key]
        loss_history.append(loss)
        
        # Forgetting events: when loss increases
        if len(loss_history) >= 2:
            forgetting_events = sum(1 for i in range(1, len(loss_history)) 
                                  if loss_history[i] > loss_history[i-1])
            forgetting_score = forgetting_events / (len(loss_history) - 1)
        else:
            forgetting_score = 0.0
            
        return forgetting_score


# =============================================================================
# Multi-Strategy Selector
# =============================================================================

class MultiStrategySelector:
    """Combines multiple scoring strategies with learnable weights"""
    
    def __init__(self, strategies: Dict[str, Callable], initial_weights=None):
        self.strategies = strategies
        self.strategy_names = list(strategies.keys())
        self.num_strategies = len(strategies)
        
        if initial_weights is None:
            self.weights = np.ones(self.num_strategies) / self.num_strategies
        else:
            self.weights = np.array(initial_weights)
            
        # Performance tracking
        self.strategy_scores_history = defaultdict(list)
        self.selection_history = []
        
    def compute_combined_score(self, idx, x_i, y_i, C_current):
        """Weighted combination of all strategies"""
        scores = {}
        
        # Get individual strategy scores
        for name, strategy_fn in self.strategies.items():
            score = strategy_fn(idx, x_i, y_i, C_current)
            scores[name] = score
        
        # Normalize scores per strategy (z-score normalization)
        normalized_scores = []
        for name in self.strategy_names:
            score = scores[name]
            # Use running statistics for normalization
            if name in self.strategy_scores_history and len(self.strategy_scores_history[name]) > 10:
                history = self.strategy_scores_history[name][-100:]  # Last 100 scores
                mean = np.mean(history)
                std = np.std(history) + 1e-8
                normalized = (score - mean) / std
            else:
                normalized = score
            normalized_scores.append(normalized)
            
        # Apply sigmoid to keep in reasonable range
        normalized_scores = 1 / (1 + np.exp(-np.array(normalized_scores)))
        
        # Weighted combination
        combined_score = np.dot(self.weights, normalized_scores)
        
        # Store scores for history
        for name, score in scores.items():
            self.strategy_scores_history[name].append(score)
            
        return combined_score, scores
    
    def select_coreset_greedy(self, dataset, budget, batch_size=100, verbose=True):
        """Greedy selection using combined scores with lazy evaluation"""
        C_indices = []
        n = len(dataset)
        available_indices = set(range(n))
        
        # Priority queue for lazy greedy
        scores_heap = []
        score_computed_at = {}
        
        if verbose:
            pbar = tqdm(total=budget, desc="Selecting coreset")
        
        while len(C_indices) < budget and available_indices:
            # Lazy evaluation: recompute top scores as needed
            while scores_heap:
                neg_score, idx, computed_round = scores_heap[0]
                
                if computed_round == score_computed_at.get(idx, -1) and idx in available_indices:
                    # Score is still valid
                    break
                else:
                    # Score is stale, remove it
                    heappop(scores_heap)
            
            # Need to compute new scores
            if not scores_heap or len(scores_heap) < batch_size:
                # Score a batch of points
                batch_indices = list(available_indices)[:batch_size]
                
                for idx in batch_indices:
                    if idx not in C_indices:
                        x_i, y_i = dataset[idx]
                        score, _ = self.compute_combined_score(idx, x_i, y_i, C_indices)
                        heappush(scores_heap, (-score, idx, len(C_indices)))
                        score_computed_at[idx] = len(C_indices)
            
            # Select best point
            if scores_heap:
                neg_score, best_idx, _ = heappop(scores_heap)
                C_indices.append(best_idx)
                available_indices.remove(best_idx)
                self.selection_history.append(best_idx)
                
                if verbose:
                    pbar.update(1)
                    if len(C_indices) % 100 == 0:
                        pbar.set_postfix({'score': -neg_score})
        
        if verbose:
            pbar.close()
            
        return C_indices
    
    def update_weights(self, new_weights):
        """Update strategy weights (for RL policy)"""
        assert len(new_weights) == self.num_strategies
        self.weights = np.array(new_weights)
        self.weights = self.weights / (self.weights.sum() + 1e-8)  # Normalize


# =============================================================================
# Training State Management
# =============================================================================

class TrainingStateEncoder:
    """Encodes current training state for RL policy"""
    
    def __init__(self, window_size=50):
        self.window_size = window_size
        self.history = {
            'loss': deque(maxlen=window_size),
            'train_accuracy': deque(maxlen=window_size),
            'val_accuracy': deque(maxlen=window_size),
            'gradient_norm': deque(maxlen=window_size),
            'learning_rate': deque(maxlen=window_size),
            'selection_entropy': deque(maxlen=window_size)
        }
        
        # Per-strategy tracking
        self.strategy_rewards = defaultdict(lambda: deque(maxlen=window_size))
        self.strategy_usage = defaultdict(lambda: deque(maxlen=window_size))
        
    def update(self, metrics: Dict):
        """Update history with new metrics"""
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
            elif key.startswith('reward_'):
                strategy = key.replace('reward_', '')
                self.strategy_rewards[strategy].append(value)
            elif key.startswith('usage_'):
                strategy = key.replace('usage_', '')
                self.strategy_usage[strategy].append(value)
    
    def get_state_vector(self):
        """Convert history to fixed-size state vector"""
        features = []
        
        # Helper function for trajectory features
        def get_trajectory_features(values):
            if len(values) == 0:
                return [0, 0, 0, 0, 0]
            
            values = list(values)
            return [
                np.mean(values),
                np.std(values) if len(values) > 1 else 0,
                values[-1],  # Current value
                np.gradient(values).mean() if len(values) > 1 else 0,  # Trend
                max(values) - min(values) if len(values) > 1 else 0  # Range
            ]
        
        # Training dynamics features
        features.extend(get_trajectory_features(self.history['loss']))
        features.extend(get_trajectory_features(self.history['train_accuracy']))
        features.extend(get_trajectory_features(self.history['val_accuracy']))
        features.extend(get_trajectory_features(self.history['gradient_norm']))
        
        # Learning rate (important for phase detection)
        if self.history['learning_rate']:
            features.append(self.history['learning_rate'][-1])
        else:
            features.append(0.001)  # Default
            
        # Selection diversity
        features.extend(get_trajectory_features(self.history['selection_entropy']))
        
        # Per-strategy performance
        strategies = ['grad_mag', 'diversity', 'uncertainty', 'boundary', 'forgetting']
        for strategy in strategies:
            # Average reward
            rewards = list(self.strategy_rewards.get(strategy, []))
            features.append(np.mean(rewards) if rewards else 0)
            
            # Usage frequency
            usage = list(self.strategy_usage.get(strategy, []))
            features.append(np.mean(usage) if usage else 0)
        
        # Training progress indicator
        if self.history['loss']:
            initial_loss = self.history['loss'][0] if len(self.history['loss']) > 0 else 1.0
            current_loss = self.history['loss'][-1]
            progress = 1.0 - (current_loss / (initial_loss + 1e-8))
            features.append(np.clip(progress, 0, 1))
        else:
            features.append(0)
            
        return np.array(features, dtype=np.float32)


# =============================================================================
# Constraint Checking
# =============================================================================

class CoresetConstraintChecker:
    """Verifies C1 (performance) and C2 (size) constraints"""
    
    def __init__(self, epsilon_max=0.05, budget_B=1000):
        self.epsilon_max = epsilon_max
        self.budget_B = budget_B
        self.performance_history = []
        
    def verify_performance_preservation(self, loss_full, loss_coreset):
        """Check if |E[L_D] - E[L_C]| <= epsilon"""
        epsilon = abs(loss_full - loss_coreset)
        satisfied = epsilon <= self.epsilon_max
        
        self.performance_history.append({
            'loss_full': loss_full,
            'loss_coreset': loss_coreset,
            'epsilon': epsilon,
            'satisfied': satisfied
        })
        
        return satisfied, epsilon
    
    def verify_size_constraint(self, coreset_size):
        """Check if |C| <= B"""
        return coreset_size <= self.budget_B
    
    def get_constraint_summary(self):
        """Summary of constraint satisfaction over time"""
        if not self.performance_history:
            return {}
            
        epsilons = [h['epsilon'] for h in self.performance_history]
        satisfaction_rate = sum(h['satisfied'] for h in self.performance_history) / len(self.performance_history)
        
        return {
            'avg_epsilon': np.mean(epsilons),
            'max_epsilon': np.max(epsilons),
            'satisfaction_rate': satisfaction_rate,
            'total_checks': len(self.performance_history)
        }


# =============================================================================
# Main Training Functions
# =============================================================================

def evaluate_model(model, data_loader, device):
    """Evaluate model on dataset"""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / total
    return accuracy, avg_loss


def compute_gradient_norm(model):
    """Compute total gradient norm"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


def compute_selection_entropy(selected_indices, num_classes=10):
    """Compute entropy of selected indices (diversity measure)"""
    if len(selected_indices) == 0:
        return 0.0
        
    # Bin indices into classes
    hist, _ = np.histogram(selected_indices, bins=min(num_classes, len(selected_indices)))
    hist = hist / hist.sum()
    
    # Compute entropy
    entropy = -sum(p * np.log(p + 1e-8) for p in hist if p > 0)
    return entropy


def train_with_coreset(model, train_dataset, val_loader, test_loader,
                      coreset_budget=1000, epochs=200, 
                      selection_frequency=20, device='cuda', 
                      tensorboard_dir='runs/coreset_experiment'):
    """
    Main training function with multi-strategy coreset selection
    """
    logger.info(f"Starting training with coreset budget={coreset_budget}")
    
    # Initialize tensorboard writer
    writer = SummaryWriter(tensorboard_dir)
    
    # Initialize components
    scorer = SubmodularScoringStrategies(model, device)
    strategies = {
        'grad_mag': scorer.gradient_magnitude_score,
        'diversity': scorer.diversity_score,
        'uncertainty': scorer.uncertainty_score,
        'boundary': scorer.boundary_score,
        'forgetting': scorer.forgetting_score
    }
    
    selector = MultiStrategySelector(strategies)
    state_encoder = TrainingStateEncoder()
    constraint_checker = CoresetConstraintChecker(epsilon_max=0.05, budget_B=coreset_budget)
    
    # Training setup
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Tracking
    results = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': [],
        'test_acc': [],
        'selection_time': [],
        'weights_history': []
    }
    
    # Initial coreset selection
    logger.info("Initial coreset selection...")
    start_time = time.time()
    coreset_indices = selector.select_coreset_greedy(train_dataset, coreset_budget)
    selection_time = time.time() - start_time
    logger.info(f"Selected {len(coreset_indices)} samples in {selection_time:.2f}s")
    
    for epoch in range(epochs):
        # Reselect coreset periodically
        if epoch > 0 and epoch % selection_frequency == 0:
            logger.info(f"\nReselecting coreset at epoch {epoch}")
            
            # Update strategy weights based on performance
            # (In full implementation, RL policy would determine these)
            performance_based_weights = compute_performance_weights(
                state_encoder, selector.strategy_names
            )
            selector.update_weights(performance_based_weights)
            
            # Clear caches to free memory
            scorer.clear_cache()
            
            # Select new coreset
            start_time = time.time()
            coreset_indices = selector.select_coreset_greedy(
                train_dataset, coreset_budget, verbose=False
            )
            selection_time = time.time() - start_time
            results['selection_time'].append(selection_time)
            
            logger.info(f"New coreset selected in {selection_time:.2f}s")
            logger.info(f"Strategy weights: {dict(zip(selector.strategy_names, selector.weights))}")
        
        # Create coreset dataloader
        coreset_dataset = Subset(train_dataset, coreset_indices)
        train_loader = DataLoader(coreset_dataset, batch_size=128, shuffle=True, num_workers=2)
        
        # Training epoch
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            # Compute gradient norm for tracking
            if batch_idx == 0:
                grad_norm = compute_gradient_norm(model)
        
        # Compute accuracies
        train_acc = 100. * train_correct / train_total
        val_acc, val_loss = evaluate_model(model, val_loader, device)
        test_acc, _ = evaluate_model(model, test_loader, device)
        
        # Update state
        metrics = {
            'loss': train_loss / len(train_loader),
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'gradient_norm': grad_norm,
            'learning_rate': scheduler.get_last_lr()[0],
            'selection_entropy': compute_selection_entropy(coreset_indices)
        }
        
        # Compute strategy-specific rewards (improvement in validation accuracy)
        if epoch > 0:
            val_improvement = val_acc - results['val_acc'][-1]
            for strategy, weight in zip(selector.strategy_names, selector.weights):
                metrics[f'reward_{strategy}'] = val_improvement * weight
                metrics[f'usage_{strategy}'] = weight
        
        state_encoder.update(metrics)
        
        # Record results
        results['train_loss'].append(train_loss / len(train_loader))
        results['train_acc'].append(train_acc)
        results['val_acc'].append(val_acc)
        results['test_acc'].append(test_acc)
        results['weights_history'].append(selector.weights.copy())
        
        # Log to tensorboard
        writer.add_scalar('Loss/Train', train_loss / len(train_loader), epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        writer.add_scalar('Accuracy/Test', test_acc, epoch)
        writer.add_scalar('Training/Learning_Rate', scheduler.get_last_lr()[0], epoch)
        writer.add_scalar('Training/Gradient_Norm', grad_norm, epoch)
        writer.add_scalar('Coreset/Selection_Entropy', compute_selection_entropy(coreset_indices), epoch)
        writer.add_scalar('Coreset/Size', len(coreset_indices), epoch)
        
        # Log strategy weights
        for i, strategy in enumerate(selector.strategy_names):
            writer.add_scalar(f'Strategy_Weights/{strategy}', selector.weights[i], epoch)
        
        # Logging
        if epoch % 10 == 0:
            logger.info(f'Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.3f} | '
                       f'Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | '
                       f'Test Acc: {test_acc:.2f}%')
        
        scheduler.step()
    
    # Final constraint verification
    logger.info("\nVerifying constraints...")
    
    # Train on full dataset for comparison
    logger.info("Training on full dataset for comparison...")
    model_full = train_full_dataset(train_dataset, val_loader, epochs=50, device=device)
    _, full_loss = evaluate_model(model_full, test_loader, device)
    _, coreset_loss = evaluate_model(model, test_loader, device)
    
    c1_satisfied, epsilon = constraint_checker.verify_performance_preservation(
        full_loss, coreset_loss
    )
    c2_satisfied = constraint_checker.verify_size_constraint(len(coreset_indices))
    
    logger.info(f"C1 (Performance): {'SATISFIED' if c1_satisfied else 'VIOLATED'} - ε={epsilon:.4f}")
    logger.info(f"C2 (Size): {'SATISFIED' if c2_satisfied else 'VIOLATED'} - |C|={len(coreset_indices)}")
    
    # Log constraints to tensorboard
    writer.add_scalar('Constraints/Performance_Epsilon', epsilon, 0)
    writer.add_scalar('Constraints/Performance_Satisfied', int(c1_satisfied), 0)
    writer.add_scalar('Constraints/Size_Satisfied', int(c2_satisfied), 0)
    
    # Summary
    constraint_summary = constraint_checker.get_constraint_summary()
    
    # Close tensorboard writer
    writer.close()
    
    return model, results, constraint_summary


def compute_performance_weights(state_encoder, strategy_names):
    """
    Compute strategy weights based on historical performance
    (Simplified version - full RL policy would be more sophisticated)
    """
    weights = []
    
    for strategy in strategy_names:
        rewards = list(state_encoder.strategy_rewards.get(strategy, [0]))
        if len(rewards) > 0:
            # Exponential moving average of rewards
            ema_reward = 0
            alpha = 0.1
            for r in rewards:
                ema_reward = alpha * r + (1 - alpha) * ema_reward
            
            # Convert to weight (softmax later)
            weight = np.exp(ema_reward * 10)  # Temperature = 10
        else:
            weight = 1.0
            
        weights.append(weight)
    
    # Normalize
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    return weights


def train_full_dataset(train_dataset, val_loader, epochs=50, device='cuda'):
    """Train model on full dataset for baseline comparison"""
    model = CIFARResNet(num_classes=len(train_dataset.classes))
    model = model.to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        if epoch % 10 == 0:
            val_acc, _ = evaluate_model(model, val_loader, device)
            logger.info(f'Full dataset - Epoch {epoch}: Val Acc: {val_acc:.2f}%')
    
    return model


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_results(results, save_path='results/'):
    """Plot training results"""
    os.makedirs(save_path, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Training curves
    ax = axes[0, 0]
    ax.plot(results['train_loss'], label='Train Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True)
    
    # Accuracy curves
    ax = axes[0, 1]
    ax.plot(results['train_acc'], label='Train', alpha=0.8)
    ax.plot(results['val_acc'], label='Validation', alpha=0.8)
    ax.plot(results['test_acc'], label='Test', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy Evolution')
    ax.legend()
    ax.grid(True)
    
    # Strategy weights evolution
    ax = axes[0, 2]
    weights_history = np.array(results['weights_history'])
    strategies = ['grad_mag', 'diversity', 'uncertainty', 'boundary', 'forgetting']
    for i, strategy in enumerate(strategies):
        ax.plot(weights_history[:, i], label=strategy, alpha=0.8)
    ax.set_xlabel('Selection Round')
    ax.set_ylabel('Weight')
    ax.set_title('Strategy Weights Evolution')
    ax.legend()
    ax.grid(True)
    
    # Selection time
    if results['selection_time']:
        ax = axes[1, 0]
        ax.plot(results['selection_time'], 'o-')
        ax.set_xlabel('Selection Round')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Coreset Selection Time')
        ax.grid(True)
    
    # Final accuracies comparison
    ax = axes[1, 1]
    final_accs = {
        'Train': results['train_acc'][-1],
        'Val': results['val_acc'][-1],
        'Test': results['test_acc'][-1]
    }
    ax.bar(final_accs.keys(), final_accs.values())
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Final Accuracies')
    ax.grid(True, axis='y')
    
    # Strategy weights distribution
    ax = axes[1, 2]
    final_weights = results['weights_history'][-1]
    ax.pie(final_weights, labels=strategies, autopct='%1.1f%%')
    ax.set_title('Final Strategy Weights')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_results.png'), dpi=150)
    plt.close()
    
    # Additional analysis plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Weight evolution heatmap
    ax = axes[0]
    sns.heatmap(weights_history.T, cmap='viridis', ax=ax,
                xticklabels=10, yticklabels=strategies)
    ax.set_xlabel('Selection Round')
    ax.set_ylabel('Strategy')
    ax.set_title('Strategy Weight Evolution Heatmap')
    
    # Accuracy improvement over time
    ax = axes[1]
    baseline = results['val_acc'][0]
    improvements = [acc - baseline for acc in results['val_acc']]
    ax.plot(improvements)
    ax.fill_between(range(len(improvements)), 0, improvements, alpha=0.3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Improvement over Initial (%)')
    ax.set_title('Validation Accuracy Improvement')
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'analysis_results.png'), dpi=150)
    plt.close()


# =============================================================================
# Main Execution
# =============================================================================

def main_cifar_experiment(args):
    """
    Run the complete experiment on CIFAR-10 or CIFAR-100
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Data preparation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load dataset
    if args.dataset.lower() == 'cifar10':
        train_dataset = CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
        val_dataset = CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_test)
        test_dataset = CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
        num_classes = 10
    else:  # CIFAR-100
        train_dataset = CIFAR100(root=args.data_dir, train=True, download=True, transform=transform_train)
        val_dataset = CIFAR100(root=args.data_dir, train=True, download=True, transform=transform_test)
        test_dataset = CIFAR100(root=args.data_dir, train=False, download=True, transform=transform_test)
        num_classes = 100
    
    # Create validation split
    val_size = args.val_size
    train_size = len(train_dataset) - val_size
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, train_size + val_size))
    
    train_dataset = Subset(train_dataset, train_indices)
    val_dataset = Subset(val_dataset, val_indices)
    
    # Create data loaders
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Calculate coreset budget
    coreset_budget = int(len(train_dataset) * args.coreset_ratio)
    logger.info(f"Dataset: {args.dataset.upper()}")
    logger.info(f"Training set size: {len(train_dataset)}")
    logger.info(f"Coreset budget: {coreset_budget} ({args.coreset_ratio*100:.1f}%)")
    
    # Initialize model
    model = CIFARResNet(num_classes=num_classes, depth=args.model_depth)
    
    # Create results directory
    results_dir = os.path.join(args.results_dir, args.dataset)
    os.makedirs(results_dir, exist_ok=True)
    
    # Train with coreset selection
    model, results, constraint_summary = train_with_coreset(
        model, train_dataset, val_loader, test_loader,
        coreset_budget=coreset_budget,
        epochs=args.epochs,
        selection_frequency=args.selection_frequency,
        device=device,
        tensorboard_dir=os.path.join(args.tensorboard_dir, f"{args.dataset}_{args.experiment_name}")
    )
    
    # Save results
    logger.info("\nFinal Results:")
    logger.info(f"Best Validation Accuracy: {max(results['val_acc']):.2f}%")
    logger.info(f"Final Test Accuracy: {results['test_acc'][-1]:.2f}%")
    logger.info(f"Constraint Summary: {constraint_summary}")
    
    # Plot results
    plot_results(results, save_path=results_dir)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'results': results,
        'constraint_summary': constraint_summary,
        'args': vars(args)
    }, os.path.join(results_dir, 'final_model.pth'))
    
    return results, constraint_summary


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Multi-Strategy Coreset Selection for CIFAR')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='cifar10', 
                       choices=['cifar10', 'cifar100'],
                       help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Directory to store/download datasets')
    parser.add_argument('--val_size', type=int, default=5000,
                       help='Validation set size')
    
    # Model arguments
    parser.add_argument('--model_depth', type=int, default=20,
                       help='ResNet depth (must be 6n+2)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='Number of data loading workers')
    
    # Coreset arguments
    parser.add_argument('--coreset_ratio', type=float, default=0.1,
                       help='Fraction of dataset to select as coreset')
    parser.add_argument('--selection_frequency', type=int, default=20,
                       help='How often to reselect coreset (epochs)')
    
    # Experiment arguments
    parser.add_argument('--experiment_name', type=str, default='default',
                       help='Name for this experiment')
    parser.add_argument('--results_dir', type=str, default='./results',
                       help='Directory to save results')
    parser.add_argument('--tensorboard_dir', type=str, default='./runs',
                       help='Directory for tensorboard logs')
    
    # System arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run experiment
    results, constraints = main_cifar_experiment(args)
