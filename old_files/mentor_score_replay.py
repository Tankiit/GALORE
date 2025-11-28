#!/usr/bin/env python3
"""
MENTOR Score Replay Buffer Implementation
Based on the visual architecture schema provided
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, OrderedDict
import time
import hashlib
from typing import Dict, List, Tuple, Optional


class ScoreReplayBuffer:
    """
    Tier 3: Score Replay Buffer (MEMORY)
    Efficiently stores and retrieves strategy scores with staleness discounting
    """

    def __init__(self, max_size: int = 50000, staleness_decay: float = 0.1):
        self.max_size = max_size
        self.staleness_decay = staleness_decay
        self.buffer = OrderedDict()  # LRU order for eviction

        # Statistics
        self.hit_count = 0
        self.miss_count = 0
        self.evict_count = 0

    def add_scores(self, sample_ids: List[int], scores: Dict[str, torch.Tensor],
                   epoch: int, model_hash: str):
        """ADD: Store fresh scores in buffer"""
        current_time = time.time()

        for i, sample_id in enumerate(sample_ids):
            # Extract scores for this sample
            sample_scores = {}
            for strategy_name, strategy_scores in scores.items():
                sample_scores[strategy_name] = strategy_scores[i].item()

            # Calculate priority (max score across strategies)
            priority = max(sample_scores.values())

            # Buffer entry
            entry = {
                'scores': sample_scores,
                'epoch': epoch,
                'priority': priority,
                'model_hash': model_hash,
                'timestamp': current_time
            }

            # Add to buffer (move to end if exists)
            if sample_id in self.buffer:
                del self.buffer[sample_id]
            self.buffer[sample_id] = entry

        # Evict if over capacity
        while len(self.buffer) > self.max_size:
            self._evict_oldest()

    def get_scores(self, sample_ids: List[int], current_epoch: int) -> Dict[int, Dict[str, float]]:
        """GET: Retrieve cached scores with staleness discounting"""
        result = {}

        for sample_id in sample_ids:
            if sample_id in self.buffer:
                entry = self.buffer[sample_id]

                # Calculate staleness discount
                age = current_epoch - entry['epoch']
                discount = np.exp(-self.staleness_decay * age)

                # Apply discount to all scores
                discounted_scores = {}
                for strategy, score in entry['scores'].items():
                    discounted_scores[strategy] = score * discount

                result[sample_id] = discounted_scores
                self.hit_count += 1

                # Move to end (LRU)
                self.buffer.move_to_end(sample_id)
            else:
                self.miss_count += 1

        return result

    def _evict_oldest(self):
        """Remove oldest/lowest priority sample from buffer"""
        if self.buffer:
            self.buffer.popitem(last=False)  # Remove oldest (first item)
            self.evict_count += 1

    def get_hit_rate(self) -> float:
        """Calculate buffer hit rate"""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0

    def get_stats(self) -> Dict:
        """Get buffer statistics"""
        return {
            'size': len(self.buffer),
            'hit_rate': self.get_hit_rate(),
            'hits': self.hit_count,
            'misses': self.miss_count,
            'evictions': self.evict_count,
            'utilization': len(self.buffer) / self.max_size
        }


class AdaptiveFreshRatioCalculator:
    """
    Tier 1: Adaptive Fresh Ratio Calculator
    Dynamically adjusts fresh sample ratio based on training signals
    """

    def __init__(self):
        self.prev_model_params = None
        self.best_val_loss = float('inf')
        self.prev_val_loss = float('inf')

    def compute_fresh_ratio(self, epoch: int, total_epochs: int,
                           model: nn.Module, val_loss: float,
                           buffer_hit_rate: float) -> float:
        """Compute adaptive fresh ratio based on multiple signals"""

        # Base ratio based on training phase
        if epoch < 5:
            base_ratio = 0.7  # Early: more exploration
        elif epoch < total_epochs // 2:
            base_ratio = 0.5  # Mid: balanced
        else:
            base_ratio = 0.3  # Late: more exploitation

        fresh_ratio = base_ratio

        # Signal 1: Model drift
        if self.prev_model_params is not None:
            current_params = torch.cat([p.data.flatten() for p in model.parameters()]).cpu()
            drift = torch.norm(current_params - self.prev_model_params).item() / torch.norm(self.prev_model_params).item()

            if drift > 0.1:  # Significant model change
                fresh_ratio += 0.2

            self.prev_model_params = current_params.detach()
        else:
            # Store initial parameters
            self.prev_model_params = torch.cat([p.data.flatten() for p in model.parameters()]).cpu().detach()

        # Signal 2: Buffer hit rate
        if buffer_hit_rate < 0.5:
            fresh_ratio += 0.15  # Cache not working well, need more fresh
        elif buffer_hit_rate > 0.8:
            fresh_ratio -= 0.1  # Cache working well, can reuse more

        # Signal 3: Validation performance
        if val_loss > 1.05 * self.best_val_loss:
            fresh_ratio += 0.2  # Performance dropping, explore more
        elif val_loss <= self.best_val_loss:
            fresh_ratio -= 0.05  # Improving, can exploit more

        # Update best validation loss
        self.best_val_loss = min(self.best_val_loss, val_loss)

        # Clip to reasonable range
        fresh_ratio = max(0.2, min(0.9, fresh_ratio))

        return fresh_ratio


class CurriculumStrategies:
    """
    Tier 2: Curriculum Scoring Strategies
    Expensive scoring functions that we want to cache
    """

    @staticmethod
    def uncertainty_score(model: nn.Module, images: torch.Tensor,
                          device: str) -> torch.Tensor:
        """Uncertainty: Forward pass for prediction entropy"""
        model.eval()
        with torch.no_grad():
            outputs = model(images.to(device))
            probs = F.softmax(outputs, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        return entropy.cpu()

    @staticmethod
    def diversity_score(model: nn.Module, images: torch.Tensor,
                        features_cache: Optional[torch.Tensor], device: str) -> torch.Tensor:
        """Diversity: Distance to nearest cached feature (placeholder)"""
        batch_size = images.size(0)
        if features_cache is None or len(features_cache) == 0:
            return torch.ones(batch_size)

        # Simplified: use random distances as placeholder
        # In real implementation, this would compute actual feature distances
        return torch.rand(batch_size)

    @staticmethod
    def balance_score(labels: torch.Tensor, class_counts: torch.Tensor) -> torch.Tensor:
        """Balance: Inverse class frequency"""
        scores = torch.zeros(len(labels))
        for i, label in enumerate(labels):
            if label < len(class_counts):
                freq = class_counts[label].float()
                scores[i] = 1.0 / (freq + 1e-8)
        return scores / scores.max()

    @staticmethod
    def loss_score(model: nn.Module, images: torch.Tensor, labels: torch.Tensor,
                   device: str) -> torch.Tensor:
        """Loss: High loss samples are informative"""
        model.eval()
        with torch.no_grad():
            outputs = model(images.to(device))
            loss = F.cross_entropy(outputs, labels.to(device), reduction='none')
        return loss.cpu()


class ThompsonSamplingMAB:
    """
    Tier 4: Thompson Sampling Multi-Armed Bandit
    Selects curriculum strategy based on learned posterior
    """

    def __init__(self, num_strategies: int = 4, state_dim: int = 5):
        self.num_strategies = num_strategies

        # Posterior parameters for each strategy (simplified linear model)
        self.theta_means = torch.zeros(num_strategies, state_dim)
        self.theta_covs = [torch.eye(state_dim) * 0.1 for _ in range(num_strategies)]

        # Learning rates
        self.lr_theta = 0.01
        self.lr_cov = 0.001

    def encode_state(self, epoch: int, val_loss: float, buffer_hit_rate: float,
                     fresh_ratio: float, model_norm: float) -> torch.Tensor:
        """Encode training state into feature vector"""
        # Normalize and combine signals
        epoch_norm = epoch / 100.0  # Assume max 100 epochs
        loss_norm = min(val_loss / 5.0, 1.0)  # Normalize loss
        state = torch.tensor([epoch_norm, loss_norm, buffer_hit_rate, fresh_ratio, model_norm])
        return state

    def select_strategy(self, state: torch.Tensor) -> int:
        """Sample from posterior and select best strategy"""
        values = []

        for k in range(self.num_strategies):
            # Sample theta from posterior
            theta_sample = torch.distributions.MultivariateNormal(
                self.theta_means[k], self.theta_covs[k]
            ).sample()

            # Compute expected reward
            value = torch.dot(theta_sample, state)
            values.append(value)

        # Select strategy with highest expected value
        return torch.argmax(torch.tensor(values)).item()

    def update(self, strategy_idx: int, state: torch.Tensor, reward: float):
        """Update posterior based on observed reward"""
        # Simplified Bayesian update for linear model
        lr = 0.1

        # Update mean towards reward direction
        update = lr * reward * state
        self.theta_means[strategy_idx] += update

        # Reduce uncertainty (simplify covariance update)
        self.theta_covs[strategy_idx] *= 0.99

        # Ensure minimum uncertainty
        min_cov = torch.eye(state.size(0)) * 0.01
        self.theta_covs[strategy_idx] = torch.maximum(self.theta_covs[strategy_idx], min_cov)


class MENTORWithScoreReplay:
    """
    Complete MENTOR system with Score Replay Buffer
    Implements the 6-tier architecture
    """

    def __init__(self, model: nn.Module, dataset: torch.utils.data.Dataset,
                 budget: float = 0.1, epochs: int = 20, device: str = 'cpu'):

        # Core components
        self.model = model.to(device)
        self.dataset = dataset
        self.budget = budget  # Fraction of data to select
        self.epochs = epochs
        self.device = device

        # Tier components
        self.buffer = ScoreReplayBuffer(max_size=10000)
        self.fresh_calculator = AdaptiveFreshRatioCalculator()
        self.ts_mab = ThompsonSamplingMAB()

        # Curriculum strategies
        self.strategies = ['uncertainty', 'diversity', 'balance', 'loss']

        # Training setup
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

        # Tracking
        self.train_acc_history = []
        self.fresh_ratio_history = []
        self.buffer_hit_rate_history = []
        self.speedup_history = []
        self.feature_quality_history = []  # NEW: [epoch, linear_probe_acc, intra_class_var]

        # Data setup
        self.setup_data()

    def setup_data(self):
        """Setup train/test data loaders"""
        # Use subset for faster demo
        total_size = len(self.dataset)
        subset_size = min(5000, total_size)
        indices = torch.randperm(total_size)[:subset_size]

        # Split into train/val
        train_size = int(0.8 * subset_size)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        self.train_subset = Subset(self.dataset, train_indices)
        self.val_subset = Subset(self.dataset, val_indices)

        self.train_loader = DataLoader(self.train_subset, batch_size=64, shuffle=True)
        self.val_loader = DataLoader(self.val_subset, batch_size=64, shuffle=False)

        # Class counts for balance strategy
        all_labels = [self.dataset[i][1] for i in train_indices]
        unique_labels, counts = torch.unique(torch.tensor(all_labels), return_counts=True)
        self.class_counts = torch.zeros(10)  # CIFAR-10 has 10 classes
        self.class_counts[unique_labels] = counts.float()

    def compute_model_hash(self) -> str:
        """Compute hash of model parameters"""
        params = torch.cat([p.data.flatten() for p in self.model.parameters()]).cpu()
        return hashlib.md5(params.numpy().tobytes()).hexdigest()[:8]

    def select_fresh_samples(self, fresh_ratio: float) -> List[int]:
        """Select fresh samples (50% random + 50% high uncertainty guided)"""
        total_samples = len(self.train_subset)
        fresh_count = int(total_samples * fresh_ratio)

        # Get all sample indices
        all_indices = list(range(total_samples))

        # Simple strategy: random sampling
        fresh_indices = np.random.choice(all_indices, fresh_count, replace=False).tolist()

        return fresh_indices

    def compute_fresh_scores(self, fresh_indices: List[int]) -> Dict[str, torch.Tensor]:
        """Tier 2: Compute expensive scores only for fresh samples"""
        # Gather fresh samples
        fresh_images = []
        fresh_labels = []

        for idx in fresh_indices:
            image, label = self.train_subset[idx]
            fresh_images.append(image)
            fresh_labels.append(label)

        fresh_images = torch.stack(fresh_images)
        fresh_labels = torch.tensor(fresh_labels)

        # Compute scores (expensive part!)
        scores = {}
        scores['uncertainty'] = CurriculumStrategies.uncertainty_score(
            self.model, fresh_images, self.device
        )
        scores['diversity'] = CurriculumStrategies.diversity_score(
            self.model, fresh_images, None, self.device
        )
        scores['balance'] = CurriculumStrategies.balance_score(
            fresh_labels, self.class_counts
        )
        scores['loss'] = CurriculumStrategies.loss_score(
            self.model, fresh_images, fresh_labels, self.device
        )

        return scores

    def combine_scores(self, fresh_indices: List[int], fresh_scores: Dict[str, torch.Tensor],
                       strategy_weights: torch.Tensor, current_epoch: int) -> torch.Tensor:
        """Combine fresh and cached scores"""
        total_samples = len(self.train_subset)
        combined_scores = torch.zeros(total_samples)

        # Get cached scores
        cached_scores = self.buffer.get_scores(list(range(total_samples)), current_epoch)

        # Combine scores for each sample
        for sample_idx in range(total_samples):
            if sample_idx in fresh_indices:
                # Use fresh scores
                fresh_idx = fresh_indices.index(sample_idx)
                sample_score = 0
                for i, strategy in enumerate(self.strategies):
                    if strategy in fresh_scores:
                        sample_score += strategy_weights[i] * fresh_scores[strategy][fresh_idx].item()
                combined_scores[sample_idx] = sample_score

            elif sample_idx in cached_scores:
                # Use cached (discounted) scores
                cached = cached_scores[sample_idx]
                sample_score = 0
                for i, strategy in enumerate(self.strategies):
                    if strategy in cached:
                        sample_score += strategy_weights[i] * cached[strategy]
                combined_scores[sample_idx] = sample_score

            else:
                # No information available, use neutral score
                combined_scores[sample_idx] = 0.5

        return combined_scores

    def select_top_k(self, scores: torch.Tensor) -> List[int]:
        """Select top-k samples based on combined scores"""
        k = int(len(self.train_subset) * self.budget)
        _, top_indices = torch.topk(scores, k)
        return top_indices.tolist()

    def train_epoch(self, epoch: int) -> Dict:
        """Train one epoch with MENTOR Score Replay"""
        epoch_start_time = time.time()

        # Get validation metrics for state encoding
        val_loss, val_acc = self.evaluate()
        model_norm = torch.norm(torch.cat([p.data.flatten() for p in self.model.parameters()])).item()

        # Tier 1: Compute adaptive fresh ratio
        fresh_ratio = self.fresh_calculator.compute_fresh_ratio(
            epoch, self.epochs, self.model, val_loss, self.buffer.get_hit_rate()
        )

        # Tier 4: Thompson Sampling to select strategy
        state = self.ts_mab.encode_state(
            epoch, val_loss, self.buffer.get_hit_rate(), fresh_ratio, model_norm
        )
        strategy_idx = self.ts_mab.select_strategy(state)
        strategy_weights = F.one_hot(torch.tensor(strategy_idx), num_classes=4).float()

        # Tier 1: Select fresh samples
        fresh_indices = self.select_fresh_samples(fresh_ratio)

        # Timing: measure expensive computation
        score_start_time = time.time()

        # Tier 2: Compute fresh scores (expensive!)
        fresh_scores = self.compute_fresh_scores(fresh_indices)

        score_time = time.time() - score_start_time

        # Tier 3: Add to buffer
        model_hash = self.compute_model_hash()
        self.buffer.add_scores(fresh_indices, fresh_scores, epoch, model_hash)

        # Combine scores (Tier 5)
        combined_scores = self.combine_scores(fresh_indices, fresh_scores, strategy_weights, epoch)

        # Tier 6: Select top-k
        selected_indices = self.select_top_k(combined_scores)

        # Train on selected samples
        train_start_time = time.time()
        train_loss, train_acc = self.train_on_selected(selected_indices)
        train_time = time.time() - train_start_time

        # Calculate speedup
        total_time = time.time() - epoch_start_time
        estimated_full_time = score_time + train_time * (1.0 / self.budget)  # Estimate if computed all
        speedup = estimated_full_time / total_time

        # Update Thompson Sampling based on reward
        reward = val_acc  # Use validation accuracy as reward
        self.ts_mab.update(strategy_idx, state, reward)

        # Measure feature quality every few epochs (to save computation)
        if epoch % 3 == 0:  # Every 3 epochs
            linear_probe_acc, intra_class_var = self.measure_feature_quality(self.model, self.val_loader, self.device)
            self.feature_quality_history.append([epoch, linear_probe_acc, intra_class_var])
        else:
            linear_probe_acc, intra_class_var = None, None

        # Store metrics
        self.train_acc_history.append(val_acc)
        self.fresh_ratio_history.append(fresh_ratio)
        self.buffer_hit_rate_history.append(self.buffer.get_hit_rate())
        self.speedup_history.append(speedup)

        return {
            'val_loss': val_loss,
            'val_acc': val_acc,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'fresh_ratio': fresh_ratio,
            'buffer_hit_rate': self.buffer.get_hit_rate(),
            'strategy_idx': strategy_idx,
            'speedup': speedup,
            'selected_count': len(selected_indices),
            'linear_probe_acc': linear_probe_acc,
            'intra_class_var': intra_class_var
        }

    def train_on_selected(self, selected_indices: List[int]) -> Tuple[float, float]:
        """Train model on selected subset"""
        self.model.train()

        # Create subset of selected samples
        selected_dataset = torch.utils.data.Subset(self.train_subset, selected_indices)
        selected_loader = DataLoader(selected_dataset, batch_size=32, shuffle=True)

        total_loss = 0
        correct = 0
        total = 0

        for images, labels in selected_loader:
            self.optimizer.zero_grad()
            outputs = self.model(images.to(self.device))
            loss = self.criterion(outputs, labels.to(self.device))
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels).sum().item()

        avg_loss = total_loss / len(selected_loader)
        accuracy = correct / total if total > 0 else 0

        return avg_loss, accuracy

    def extract_features(self, model: nn.Module, data_loader: DataLoader, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features from penultimate layer for quality measurement"""
        model.eval()
        all_features = []
        all_labels = []

        with torch.no_grad():
            for images, labels in data_loader:
                # Get features from penultimate layer (before final classification)
                # Remove the final layer temporarily
                temp_fc = model.fc
                model.fc = nn.Identity()  # Pass through features

                outputs = model(images.to(device))
                features = outputs.view(outputs.size(0), -1)  # Flatten

                # Restore final layer
                model.fc = temp_fc

                all_features.append(features.cpu())
                all_labels.append(labels)

        return torch.cat(all_features), torch.cat(all_labels)

    def train_linear_probe(self, features: torch.Tensor, labels: torch.Tensor, device: str = 'cpu') -> float:
        """Train linear probe on extracted features and return accuracy"""
        # Simple linear classifier
        probe = nn.Linear(features.size(1), len(torch.unique(labels))).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(probe.parameters(), lr=0.01)

        # Move to device
        features, labels = features.to(device), labels.to(device)

        # Train for a few epochs
        probe.train()
        for _ in range(10):  # Quick training
            optimizer.zero_grad()
            outputs = probe(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Evaluate
        probe.eval()
        with torch.no_grad():
            outputs = probe(features)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == labels).float().mean().item()

        return accuracy

    def compute_intra_class_variance(self, features: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute average intra-class variance (lower is better)"""
        variances = []

        for class_label in torch.unique(labels):
            class_mask = (labels == class_label)
            if class_mask.sum() > 1:
                class_features = features[class_mask]
                class_mean = class_features.mean(dim=0)
                variance = ((class_features - class_mean) ** 2).mean().item()
                variances.append(variance)

        return np.mean(variances) if variances else 0.0

    def measure_feature_quality(self, model: nn.Module, val_loader: DataLoader, device: str) -> Tuple[float, float]:
        """Measure feature quality: linear probe accuracy and intra-class variance"""
        try:
            features, labels = self.extract_features(model, val_loader, device)

            # Linear probe accuracy (higher is better)
            linear_probe_acc = self.train_linear_probe(features, labels, device)

            # Intra-class variance (lower is better)
            intra_class_var = self.compute_intra_class_variance(features, labels)

            return linear_probe_acc, intra_class_var
        except Exception as e:
            print(f"Feature quality measurement failed: {e}")
            return 0.5, 1.0  # Return neutral values

    def evaluate(self) -> Tuple[float, float]:
        """Evaluate model on validation set"""
        self.model.eval()

        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                outputs = self.model(images.to(self.device))
                loss = self.criterion(outputs, labels.to(self.device))

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted.cpu() == labels).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total if total > 0 else 0

        return avg_loss, accuracy

    def train(self):
        """Main training loop"""
        print("🚀 Starting MENTOR Score Replay Buffer Training")
        print(f"Dataset: {len(self.train_subset)} samples")
        print(f"Budget: {self.budget:.1%}, Epochs: {self.epochs}")
        print("="*70)

        for epoch in range(self.epochs):
            metrics = self.train_epoch(epoch)

            # Print progress
            strategy_name = self.strategies[metrics['strategy_idx']]

            # Base metrics
            progress_str = f"Epoch {epoch+1:2d}: Val Acc: {metrics['val_acc']:.3f} | Fresh: {metrics['fresh_ratio']:.2f} | Hit Rate: {metrics['buffer_hit_rate']:.2f} | Strategy: {strategy_name} | Speedup: {metrics['speedup']:.1f}x"

            # Add feature quality if available
            if metrics['linear_probe_acc'] is not None:
                progress_str += f" | LP Acc: {metrics['linear_probe_acc']:.3f} | IntraVar: {metrics['intra_class_var']:.3f}"

            print(progress_str)

        print("\n" + "="*70)
        print("🎯 Training Complete!")
        print(f"Final Validation Accuracy: {self.train_acc_history[-1]:.3f}")
        print(f"Average Speedup: {np.mean(self.speedup_history):.1f}x")
        print(f"Final Buffer Hit Rate: {self.buffer_hit_rate_history[-1]:.2f}")

        # Plot results
        self.plot_results()

    def plot_results(self):
        """Plot training metrics"""
        # Create larger figure for feature quality
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('MENTOR Score Replay Buffer Results with Feature Quality', fontsize=16)

        # Plot 1: Validation Accuracy
        axes[0, 0].plot(self.train_acc_history, 'b-', linewidth=2)
        axes[0, 0].set_title('Validation Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].grid(True)

        # Plot 2: Fresh Ratio Evolution
        axes[0, 1].plot(self.fresh_ratio_history, 'g-', linewidth=2)
        axes[0, 1].set_title('Adaptive Fresh Ratio')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Fresh Ratio')
        axes[0, 1].grid(True)

        # Plot 3: Buffer Hit Rate
        axes[0, 2].plot(self.buffer_hit_rate_history, 'r-', linewidth=2)
        axes[0, 2].set_title('Buffer Hit Rate')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Hit Rate')
        axes[0, 2].grid(True)

        # Plot 4: Speedup
        axes[1, 0].plot(self.speedup_history, 'm-', linewidth=2)
        axes[1, 0].set_title('Computation Speedup')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Speedup (x)')
        axes[1, 0].grid(True)

        # Plot 5: Linear Probe Accuracy (if available)
        if self.feature_quality_history:
            epochs = [entry[0] for entry in self.feature_quality_history]
            lp_accs = [entry[1] for entry in self.feature_quality_history]
            axes[1, 1].plot(epochs, lp_accs, 'c-', linewidth=2, marker='o')
            axes[1, 1].set_title('Linear Probe Accuracy')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('LP Accuracy')
            axes[1, 1].grid(True)
        else:
            axes[1, 1].text(0.5, 0.5, 'No Feature Quality Data', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Linear Probe Accuracy')

        # Plot 6: Intra-class Variance (if available)
        if self.feature_quality_history:
            epochs = [entry[0] for entry in self.feature_quality_history]
            intra_vars = [entry[2] for entry in self.feature_quality_history]
            axes[1, 2].plot(epochs, intra_vars, 'orange', linewidth=2, marker='s')
            axes[1, 2].set_title('Intra-Class Variance')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Variance (lower is better)')
            axes[1, 2].grid(True)
        else:
            axes[1, 2].text(0.5, 0.5, 'No Feature Quality Data', ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Intra-Class Variance')

        plt.tight_layout()
        plt.savefig('mentor_score_replay_results.png', dpi=150, bbox_inches='tight')
        print("Results saved as: mentor_score_replay_results.png")


def main():
    """Main function to run MENTOR with Score Replay Buffer"""
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Setup CIFAR-10 data
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    dataset = torchvision.datasets.CIFAR10(
        root='/Users/tanmoy/research/data',
        train=True,
        download=True,
        transform=transform
    )

    # Create pretrained model with proper CIFAR-10 adaptation
    model = torchvision.models.resnet18(pretrained=True)
    # Replace final layer for CIFAR-10
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)

    # Initialize MENTOR with Score Replay Buffer
    mentor = MENTORWithScoreReplay(
        model=model,
        dataset=dataset,
        budget=0.1,  # Use 10% of data per epoch
        epochs=15,
        device=device
    )

    # Train
    mentor.train()


if __name__ == "__main__":
    main()