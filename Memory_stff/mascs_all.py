import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
from collections import deque, defaultdict
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List, Tuple, Optional, Union
import faiss
import time
import argparse
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import json
import datetime
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
import warnings
warnings.filterwarnings('ignore')

class MASCSOptimized:
    def __init__(self, dataset, budget, feature_extractor, num_classes,
                 memory_window=100, device='cuda' if torch.cuda.is_available() else 'cpu',
                 state_dim=3, action_dim=6, gamma=0.95, use_gradient_cache=True,
                 gradient_batch_accumulation=8):
        self.dataset = dataset
        self.budget = budget
        self.feature_extractor = feature_extractor
        self.num_classes = num_classes
        self.memory_window = memory_window
        self.device = device
        self.model_version = 0
        self.use_gradient_cache = use_gradient_cache
        self.gradient_batch_accumulation = gradient_batch_accumulation
        
        # Memory structures
        self.sample_memories = [None] * len(dataset)
        self.memory_buffer = deque(maxlen=memory_window * 100)
        self.selection_history = defaultdict(int)
        self.validation_improvements = defaultdict(list)
        
        # Caching
        self._cached_features = None
        self._cached_labels = None
        self._cached_model_version = -1

        # Gradient caching
        self._cached_gradient_scores = None
        self._cached_gradient_model_version = -1
        
        # Strategy components
        self.strategy_names = ['S_U', 'S_B', 'S_G', 'S_F', 'S_D', 'S_C']
        self.current_weights = {name: 1.0/len(self.strategy_names) for name in self.strategy_names}
        
        # Policy network
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        ).to(device)
        
        self.value_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(device)
        
        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=1e-4)
        self.value_optimizer = torch.optim.Adam(self.value_network.parameters(), lr=1e-4)
        self.gamma = gamma
        
        # Bayesian optimization setup
        self.bayesian_sequence_outcomes = []
        self.current_bayesian_sequence = None
        self.bayesian_sequence_start = 0
        
        # Performance tracking
        self.reward_history = np.array([])
        self.state_history = []
        self.performance_history = []

    def _extract_features_with_cache(self, model, dataloader):
        """Efficient feature extraction with better caching"""
        if (self._cached_features is not None and 
            self._cached_labels is not None and 
            self._cached_model_version == self.model_version):
            return self._cached_features, self._cached_labels
        
        features_list = []
        labels_list = []
        
        model.eval()
        with torch.no_grad():
            for x, y in tqdm(dataloader, desc="Extracting features"):
                x = x.to(self.device)
                feat = self.feature_extractor(model, x)
                features_list.append(feat.cpu())
                labels_list.append(y)
        
        # Cache results
        self._cached_features = torch.cat(features_list, dim=0).numpy()
        self._cached_labels = torch.cat(labels_list, dim=0).numpy()
        self._cached_model_version = self.model_version
        
        return self._cached_features, self._cached_labels

    def compute_gradient_scores_cached(self, model, dataloader, loss_fn, cache_dir=None, batch_accumulation=4):
        """Compute gradient scores with batched caching for better efficiency"""
        # Check memory cache first
        if (self._cached_gradient_scores is not None and
            self._cached_gradient_model_version == self.model_version):
            print("Using memory-cached gradient scores")
            return self._cached_gradient_scores

        # Check disk cache if cache_dir provided
        cache_file = None
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, f"gradient_scores_v{self.model_version}.npy")
            if os.path.exists(cache_file):
                print(f"Loading gradient scores from disk cache: {cache_file}")
                self._cached_gradient_scores = np.load(cache_file)
                self._cached_gradient_model_version = self.model_version
                return self._cached_gradient_scores

        print(f"Computing gradient scores with batch accumulation (batches={batch_accumulation})")
        all_gradient_scores = []
        original_training = model.training
        model.train()

        try:
            accumulated_batches = []
            accumulated_labels = []

            for batch_idx, (x_batch, y_batch) in enumerate(tqdm(dataloader, desc="Computing gradients")):
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

                accumulated_batches.append(x_batch)
                accumulated_labels.append(y_batch)

                # Process accumulated batches when we reach the accumulation limit or end of data
                if len(accumulated_batches) == batch_accumulation or batch_idx == len(dataloader) - 1:
                    # Concatenate all accumulated batches
                    combined_x = torch.cat(accumulated_batches, dim=0)
                    combined_y = torch.cat(accumulated_labels, dim=0)

                    model.zero_grad()
                    with torch.enable_grad():
                        logits = model(combined_x)
                        loss = loss_fn(logits, combined_y)
                        loss.backward()

                    # Compute gradient norm for the combined batch
                    total_grad_norm = 0
                    for param in model.parameters():
                        if param.grad is not None:
                            total_grad_norm += param.grad.norm(2).item()

                    # Distribute gradient score across all samples in combined batch
                    avg_grad_score = total_grad_norm / len(combined_x)

                    # Assign scores to individual batches
                    start_idx = 0
                    for orig_batch in accumulated_batches:
                        batch_size = len(orig_batch)
                        batch_scores = [avg_grad_score] * batch_size
                        all_gradient_scores.extend(batch_scores)
                        start_idx += batch_size

                    # Clear accumulated batches
                    accumulated_batches = []
                    accumulated_labels = []

        finally:
            model.train(original_training)

        # Cache the results in memory
        self._cached_gradient_scores = np.array(all_gradient_scores)
        self._cached_gradient_model_version = self.model_version

        # Also save to disk if cache_dir provided
        if cache_file is not None:
            np.save(cache_file, self._cached_gradient_scores)
            print(f"Saved gradient scores to disk cache: {cache_file}")

        return self._cached_gradient_scores

    def compute_gradient_score_batch(self, model, x_batch, y_batch, loss_fn):
        # This method is now just a wrapper - actual computation happens in cached version
        return []  # Will be filled by cached computation

    def update_memory(self, sample_idx, scores, selected, validation_improvement=0.0):
        """More memory-efficient memory update"""
        memory_vector = np.array([
            scores['S_U'], scores['S_B'], scores['S_G'], scores['S_F'],
            float(selected), validation_improvement, 0.0, 0.0
        ], dtype=np.float32)
        
        if self.sample_memories[sample_idx] is None:
            self.sample_memories[sample_idx] = deque(maxlen=self.memory_window)
        
        self.sample_memories[sample_idx].append(memory_vector)
        self.memory_buffer.append((sample_idx, memory_vector))
        
        if selected:
            self.selection_history[sample_idx] = self.selection_history.get(sample_idx, 0) + 1
            self.validation_improvements[sample_idx] = self.validation_improvements.get(sample_idx, []) + [validation_improvement]

    def compute_diversity_score(self, features, current_coreset_features):
        """Efficient diversity score computation"""
        if len(current_coreset_features) == 0:
            return np.ones(len(features))
        
        # Use FAISS for faster nearest neighbor search if available
        try:
            index = faiss.IndexFlatL2(current_coreset_features.shape[1])
            index.add(current_coreset_features.astype(np.float32))
            distances, _ = index.search(features.astype(np.float32), 1)
            return distances.flatten()
        except ImportError:
            # Fallback to scikit-learn
            nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(current_coreset_features)
            distances, _ = nbrs.kneighbors(features)
            return distances.flatten()

    def compute_class_balance_score(self, labels, current_coreset_labels):
        """Efficient class balance score computation"""
        if len(current_coreset_labels) == 0:
            return np.ones(len(labels))
        
        # Use numpy bincount for efficiency
        class_counts = np.bincount(current_coreset_labels, minlength=self.num_classes)
        inv_class_counts = 1.0 / (class_counts + 1)  # Avoid division by zero
        return inv_class_counts[labels]

    def get_bayesian_weights(self, state: Dict, epoch: int) -> Dict[str, float]:
        """Efficient Bayesian weight sampling"""
        # Simplified implementation - in practice you'd use a Bayesian optimization library
        context = self._encode_context(state)
        
        # Check if we need a new sequence (with early termination)
        if (self.current_bayesian_sequence is None or 
            epoch - self.bayesian_sequence_start >= 10 or
            len(self.bayesian_sequence_outcomes) > 0 and np.mean([o['performance'] for o in self.bayesian_sequence_outcomes]) < -0.1):
            
            # Early termination for poorly performing sequences
            self.current_bayesian_sequence = self._sample_weight_sequence(context)
            self.bayesian_sequence_start = epoch
            self.bayesian_sequence_outcomes = []
        
        # Get weights for current position in sequence
        sequence_position = epoch - self.bayesian_sequence_start
        current_weights = self.current_bayesian_sequence[sequence_position]
        
        # Convert to dictionary
        return {score_name: weight for score_name, weight in 
                zip(self.strategy_names, current_weights)}
    
    def _encode_context(self, state):
        """Encode state for Bayesian optimization"""
        # Simplified implementation
        return np.array([state.get(k, 0) for k in ['accuracy', 'loss', 'diversity']])
    
    def _sample_weight_sequence(self, context):
        """Sample a sequence of weight vectors"""
        # Simplified implementation - in practice you'd use Bayesian optimization
        seq_length = 10
        weights = []
        for i in range(seq_length):
            # Sample weights from a Dirichlet distribution
            w = np.random.dirichlet(np.ones(len(self.strategy_names)))
            weights.append(w)
        return weights

    def compute_temporal_features(self, sample_idx):
        """Efficient temporal feature computation"""
        memory = self.sample_memories[sample_idx]
        if memory is None or len(memory) == 0:
            return {
                'volatility': 0.0,
                'gradient_trend': 0.0,
                'forgetting_frequency': 0.0,
                'selection_impact': 0.0,
                'staleness': 0.0
            }
        
        # Convert to numpy array for efficient computation
        memory_array = np.array(list(memory))
        
        # Compute features efficiently
        uncertainty_history = memory_array[:, 0]
        gradient_history = memory_array[:, 2]
        forgetting_history = memory_array[:, 3]
        selection_history = memory_array[:, 4]
        
        volatility = np.var(uncertainty_history) if len(uncertainty_history) > 1 else 0.0
        
        if len(gradient_history) > 1:
            time_points = np.arange(len(gradient_history))
            gradient_trend = np.polyfit(time_points, gradient_history, 1)[0]
        else:
            gradient_trend = 0.0
        
        forgetting_frequency = np.mean(forgetting_history)
        
        selected_indices = np.where(selection_history > 0.5)[0]
        if len(selected_indices) > 0:
            improvements = memory_array[selected_indices, 5]
            selection_impact = np.mean(improvements) if len(improvements) > 0 else 0.0
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

    def update_policy(self, state: torch.Tensor, action: str, reward: float, next_state: torch.Tensor):
        """Efficient policy update with batch processing"""
        action_idx = self.strategy_names.index(action)
        
        # Batch temporal credit assignment
        current_epoch = len(self.performance_history)
        credits = self._compute_temporal_credits(current_epoch, reward)
        
        # Compute advantage with temporal credits
        with torch.no_grad():
            value = self.value_network(state.unsqueeze(0)).item()
            next_value = self.value_network(next_state.unsqueeze(0)).item()
            
            # Include temporal credits in advantage
            temporal_bonus = credits.get(current_epoch - 1, 0) if credits else 0
            advantage = reward + temporal_bonus + self.gamma * next_value - value
        
        # Batch update value network
        value_pred = self.value_network(state.unsqueeze(0))
        value_target = torch.tensor([reward + self.gamma * next_value], device=self.device)
        value_loss = F.mse_loss(value_pred, value_target)
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        # Batch update policy network
        action_probs = self.policy_network(state.unsqueeze(0))
        action_prob = action_probs[0, action_idx]
        policy_loss = -torch.log(action_prob) * advantage
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        return policy_loss.item(), value_loss.item()
    
    def _compute_temporal_credits(self, current_epoch, reward):
        """Compute temporal credits for recent actions"""
        credits = {}
        window = min(5, current_epoch)
        for i in range(1, window + 1):
            # Assign credit based on recency and performance improvement
            credits[current_epoch - i] = reward * (self.gamma ** i)
        return credits

    def select_coreset(self, model, dataloader, loss_fn, current_coreset=None, epoch=0):
        """Optimized coreset selection"""
        start_time = time.time()
        
        # Extract features with caching
        features, labels = self._extract_features_with_cache(model, dataloader)
        
        # Initialize scores
        n_samples = len(self.dataset)
        all_scores = {name: np.zeros(n_samples) for name in self.strategy_names}
        
        # Get current coreset features if available
        current_coreset_features = np.array([])
        current_coreset_labels = np.array([])
        if current_coreset is not None and len(current_coreset) > 0:
            current_coreset_features = np.vstack([features[i] for i in current_coreset])
            current_coreset_labels = np.array([labels[i] for i in current_coreset])
        
        # Compute gradient scores using cached method
        cache_dir = "./gradient_cache" if self.use_gradient_cache else None
        all_scores['S_G'] = self.compute_gradient_scores_cached(
            model, dataloader, loss_fn, cache_dir, self.gradient_batch_accumulation
        )

        # Compute other scores in batches
        model.eval()
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(tqdm(dataloader, desc="Computing other scores")):
                x, y = x.to(self.device), y.to(self.device)
                start_idx = batch_idx * dataloader.batch_size
                end_idx = min(start_idx + len(x), n_samples)

                # Compute uncertainty and boundary scores
                all_scores['S_U'][start_idx:end_idx] = self.compute_uncertainty_score(model, x).cpu().numpy()
                all_scores['S_B'][start_idx:end_idx] = self.compute_boundary_score(model, x).cpu().numpy()
        
        # Compute diversity and class balance scores
        all_scores['S_D'] = self.compute_diversity_score(features, current_coreset_features)
        all_scores['S_C'] = self.compute_class_balance_score(labels, current_coreset_labels)
        
        # Compute forgetting scores
        all_scores['S_F'] = self.compute_forgetting_score()
        
        # Get strategy weights
        state = self.get_state()
        strategy_weights = self.get_bayesian_weights(state, epoch)
        
        # Combine scores
        combined_scores = np.zeros(n_samples)
        for name in self.strategy_names:
            # Normalize scores
            if np.max(all_scores[name]) > 0:
                all_scores[name] = all_scores[name] / np.max(all_scores[name])
            combined_scores += strategy_weights[name] * all_scores[name]
        
        # Select top-k samples
        selected_indices = np.argsort(combined_scores)[-self.budget:]
        
        # Update memory
        for idx in selected_indices:
            sample_scores = {name: all_scores[name][idx] for name in self.strategy_names}
            self.update_memory(idx, sample_scores, True)
        
        # Update model version (this will invalidate caches for next call)
        self.model_version += 1
        
        print(f"Coreset selection completed in {time.time() - start_time:.2f} seconds")
        return selected_indices.tolist()

    def compute_uncertainty_score(self, model, x):
        """Compute uncertainty scores"""
        with torch.no_grad():
            logits = model(x)
            probabilities = F.softmax(logits, dim=1)
            uncertainty = 1 - torch.max(probabilities, dim=1)[0]
            return uncertainty

    def compute_boundary_score(self, model, x):
        """Compute boundary scores"""
        with torch.no_grad():
            logits = model(x)
            probabilities = F.softmax(logits, dim=1)
            top2 = torch.topk(probabilities, 2, dim=1)[0]
            boundary = 1 - (top2[:, 0] - top2[:, 1])
            return boundary

    def compute_forgetting_score(self):
        """Compute forgetting scores"""
        forgetting_scores = np.zeros(len(self.dataset))
        for idx in range(len(self.dataset)):
            if self.sample_memories[idx] is not None:
                # Count how many times the sample was forgotten (not selected after being selected)
                memory_array = np.array(list(self.sample_memories[idx]))
                if len(memory_array) > 1:
                    selections = memory_array[:, 4]  # Selection history
                    forgetting_events = 0
                    for i in range(1, len(selections)):
                        if selections[i-1] > 0.5 and selections[i] < 0.5:
                            forgetting_events += 1
                    forgetting_scores[idx] = forgetting_events / len(selections) if len(selections) > 0 else 0
        return forgetting_scores

    def get_state(self):
        """Get current state representation"""
        # Simplified state representation
        if len(self.performance_history) == 0:
            return {'accuracy': 0.5, 'loss': 1.0, 'diversity': 0.5}
        
        return {
            'accuracy': np.mean(self.performance_history[-5:]) if len(self.performance_history) >= 5 else np.mean(self.performance_history),
            'loss': 1.0 - np.mean(self.performance_history[-5:]) if len(self.performance_history) >= 5 else 1.0 - np.mean(self.performance_history),
            'diversity': 0.5  # Placeholder - would compute actual diversity metric
        }

# TIMM-based model with feature extractor
class TIMMModel(nn.Module):
    """Wrapper for TIMM models with feature extraction capability"""
    def __init__(self, model_name='resnet18', num_classes=10, pretrained=True):
        super(TIMMModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        
        # Remove the final classification layer for feature extraction
        self.feature_dim = self.model.num_features
        self.classifier = self.model.get_classifier()
        self.model.reset_classifier(0)  # Remove classifier for feature extraction
        
    def forward(self, x):
        features = self.model(x)
        return self.classifier(features)
    
    def feature_extractor(self, x):
        return self.model(x)

# Dataset Manager
class DatasetManager:
    """Manager for handling multiple datasets"""
    def __init__(self):
        self.available_datasets = {
            'cifar10': self.get_cifar10_datasets,
            'cifar100': self.get_cifar100_datasets,
            'imagenet': self.get_imagenet_datasets,
            'mnist': self.get_mnist_datasets,
            'fashionmnist': self.get_fashionmnist_datasets
        }
    
    def get_dataset(self, name, data_dir='./data', **kwargs):
        if name not in self.available_datasets:
            raise ValueError(f"Dataset {name} not supported. Available: {list(self.available_datasets.keys())}")
        
        return self.available_datasets[name](data_dir, **kwargs)
    
    def get_cifar10_datasets(self, data_dir='./data'):
        """Get CIFAR-10 datasets with appropriate transforms"""
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=train_transform
        )
        
        val_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=val_transform
        )
        
        return train_dataset, val_dataset, 10
    
    def get_cifar100_datasets(self, data_dir='./data'):
        """Get CIFAR-100 datasets with appropriate transforms"""
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        
        train_dataset = torchvision.datasets.CIFAR100(
            root=data_dir, train=True, download=True, transform=train_transform
        )
        
        val_dataset = torchvision.datasets.CIFAR100(
            root=data_dir, train=False, download=True, transform=val_transform
        )
        
        return train_dataset, val_dataset, 100
    
    def get_imagenet_datasets(self, data_dir='./data', size=224):
        """Get ImageNet datasets with appropriate transforms"""
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Note: ImageNet requires manual download and setup
        train_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(data_dir, 'train'), transform=train_transform
        )
        
        val_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(data_dir, 'val'), transform=val_transform
        )
        
        return train_dataset, val_dataset, 1000
    
    def get_mnist_datasets(self, data_dir='./data'):
        """Get MNIST datasets with appropriate transforms"""
        train_transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = torchvision.datasets.MNIST(
            root=data_dir, train=True, download=True, transform=train_transform
        )
        
        val_dataset = torchvision.datasets.MNIST(
            root=data_dir, train=False, download=True, transform=val_transform
        )
        
        return train_dataset, val_dataset, 10
    
    def get_fashionmnist_datasets(self, data_dir='./data'):
        """Get FashionMNIST datasets with appropriate transforms"""
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        
        train_dataset = torchvision.datasets.FashionMNIST(
            root=data_dir, train=True, download=True, transform=train_transform
        )
        
        val_dataset = torchvision.datasets.FashionMNIST(
            root=data_dir, train=False, download=True, transform=val_transform
        )
        
        return train_dataset, val_dataset, 10

# Model Manager
class ModelManager:
    """Manager for handling multiple models"""
    def __init__(self):
        self.available_models = {
            'resnet18': lambda num_classes: TIMMModel('resnet18', num_classes),
            'resnet50': lambda num_classes: TIMMModel('resnet50', num_classes),
            'resnet101': lambda num_classes: TIMMModel('resnet101', num_classes),
            'efficientnet_b0': lambda num_classes: TIMMModel('efficientnet_b0', num_classes),
            'efficientnet_b1': lambda num_classes: TIMMModel('efficientnet_b1', num_classes),
            'mobilenetv3_small': lambda num_classes: TIMMModel('mobilenetv3_small_100', num_classes),
            'mobilenetv3_large': lambda num_classes: TIMMModel('mobilenetv3_large_100', num_classes),
            'vit_tiny_patch16_224': lambda num_classes: TIMMModel('vit_tiny_patch16_224', num_classes),
            'vit_small_patch16_224': lambda num_classes: TIMMModel('vit_small_patch16_224', num_classes),
        }
    
    def get_model(self, name, num_classes, device='cuda'):
        if name not in self.available_models:
            raise ValueError(f"Model {name} not supported. Available: {list(self.available_models.keys())}")
        
        model = self.available_models[name](num_classes)
        return model.to(device)

# Experiment Manager
class ExperimentManager:
    """Manager for running experiments with different configurations"""
    def __init__(self, log_dir='./experiments'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        self.dataset_manager = DatasetManager()
        self.model_manager = ModelManager()
        
    def run_experiment(self, config):
        """Run a single experiment with the given configuration"""
        # Create experiment directory
        run_suffix = f"_run{config['sequential_run']}" if 'sequential_run' in config else ""
        exp_name = f"{config['model']}_{config['dataset']}_{config['budget']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}{run_suffix}"
        exp_dir = self.log_dir / exp_name
        exp_dir.mkdir(exist_ok=True)
        
        # Save config
        with open(exp_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=4)
        
        # Setup TensorBoard
        writer = SummaryWriter(log_dir=str(exp_dir / 'tensorboard'))
        
        # Get dataset
        train_dataset, val_dataset, num_classes = self.dataset_manager.get_dataset(
            config['dataset'], config['data_dir']
        )
        
        # Get model
        model = self.model_manager.get_model(
            config['model'], num_classes, config['device']
        )
        
        # Update config with num_classes
        config['num_classes'] = num_classes
        
        # Run training
        best_val_acc = train_with_mascs(
            model, train_dataset, val_dataset, config, writer, exp_dir
        )
        
        # Save final results
        results = {
            'best_val_accuracy': best_val_acc,
            'experiment_name': exp_name,
            'config': config
        }
        
        with open(exp_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        writer.close()
        
        return results
    
    def run_multiple_experiments(self, configs, num_sequential_runs=3):
        """Run multiple experiments with different configurations, each run multiple times sequentially"""
        all_results = []

        for config_idx, config in enumerate(configs):
            config_results = []
            print(f"\n=== Running Configuration {config_idx + 1}/{len(configs)} ===")
            print(f"Config: {config['model']} on {config['dataset']}, budget={config['budget']}")

            for run_idx in range(num_sequential_runs):
                print(f"\n--- Sequential Run {run_idx + 1}/{num_sequential_runs} ---")
                try:
                    # Add run info to config
                    config_with_run = config.copy()
                    config_with_run['sequential_run'] = run_idx + 1

                    result = self.run_experiment(config_with_run)
                    result['config_idx'] = config_idx
                    result['run_idx'] = run_idx + 1

                    config_results.append(result)
                    all_results.append(result)

                    print(f"Run {run_idx + 1} completed: {result['experiment_name']}, Best accuracy: {result['best_val_accuracy']:.2f}%")
                except Exception as e:
                    print(f"Run {run_idx + 1} failed: {e}")
                    error_result = {'error': str(e), 'config': config_with_run, 'config_idx': config_idx, 'run_idx': run_idx + 1}
                    config_results.append(error_result)
                    all_results.append(error_result)

            # Print summary for this configuration
            successful_runs = [r for r in config_results if 'best_val_accuracy' in r]
            if successful_runs:
                accuracies = [r['best_val_accuracy'] for r in successful_runs]
                print(f"\nConfiguration {config_idx + 1} Summary:")
                print(f"  Mean accuracy: {np.mean(accuracies):.2f}% ± {np.std(accuracies):.2f}%")
                print(f"  Best run: {max(accuracies):.2f}%")
                print(f"  Worst run: {min(accuracies):.2f}%")

        # Save all results
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(self.log_dir / 'sequential_results.csv', index=False)

        # Create summary statistics
        self._create_sequential_summary(all_results, configs)

        return all_results

    def _create_sequential_summary(self, results, configs):
        """Create summary statistics for sequential runs"""
        summary_data = []

        for config_idx, config in enumerate(configs):
            config_results = [r for r in results if r.get('config_idx') == config_idx and 'best_val_accuracy' in r]

            if config_results:
                accuracies = [r['best_val_accuracy'] for r in config_results]
                summary_data.append({
                    'config_idx': config_idx,
                    'model': config['model'],
                    'dataset': config['dataset'],
                    'budget': config['budget'],
                    'num_runs': len(config_results),
                    'mean_accuracy': np.mean(accuracies),
                    'std_accuracy': np.std(accuracies),
                    'min_accuracy': min(accuracies),
                    'max_accuracy': max(accuracies),
                    'median_accuracy': np.median(accuracies)
                })

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(self.log_dir / 'sequential_summary.csv', index=False)

        # Print final summary
        print("\n" + "="*50)
        print("SEQUENTIAL RUNS SUMMARY")
        print("="*50)
        for _, row in summary_df.iterrows():
            print(f"{row['model']} on {row['dataset']} (budget={row['budget']}):")
            print(f"  {row['num_runs']} runs: {row['mean_accuracy']:.2f}% ± {row['std_accuracy']:.2f}%")
            print(f"  Range: [{row['min_accuracy']:.2f}%, {row['max_accuracy']:.2f}%]")
            print()

# Training functions
def train_epoch(model, train_loader, optimizer, loss_fn, device, use_amp, scaler, writer=None, epoch=0):
    """Optimized training function with TensorBoard logging"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    for batch_idx, (x, y) in enumerate(progress_bar):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast(device_type='cuda', enabled=use_amp):
            outputs = model(x)
            loss = loss_fn(outputs, y)
        
        # Use GradScaler for mixed precision
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{total_loss/(batch_idx+1):.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
        
        # Log batch metrics to TensorBoard
        if writer and batch_idx % 100 == 0:
            writer.add_scalar('Loss/train_batch', loss.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('Accuracy/train_batch', 100. * correct / total, epoch * len(train_loader) + batch_idx)
    
    avg_loss = total_loss / len(train_loader)
    avg_acc = 100.0 * correct / total
    
    # Log epoch metrics to TensorBoard
    if writer:
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Accuracy/train', avg_acc, epoch)
    
    return avg_loss, avg_acc

def validate(model, val_loader, loss_fn, device, writer=None, epoch=0, class_names=None):
    """Validation function with TensorBoard logging"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(val_loader, desc="Validation")
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(progress_bar):
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = loss_fn(outputs, y)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{total_loss/(batch_idx+1):.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    avg_loss = total_loss / len(val_loader)
    avg_acc = 100.0 * correct / total
    
    # Log validation metrics to TensorBoard
    if writer:
        writer.add_scalar('Loss/val', avg_loss, epoch)
        writer.add_scalar('Accuracy/val', avg_acc, epoch)
        
        # Log confusion matrix every 5 epochs
        if epoch % 5 == 0 and class_names is not None:
            cm = confusion_matrix(all_labels, all_preds)
            fig = plt.figure(figsize=(10, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            writer.add_figure('Confusion Matrix', fig, epoch)
            plt.close(fig)
    
    return avg_loss, avg_acc

def train_with_mascs(model, train_dataset, val_dataset, args, writer, exp_dir):
    """Training loop with optimized MASCS"""
    # Initialize MASCS
    mascs = MASCSOptimized(
        dataset=train_dataset,
        budget=args['budget'],
        feature_extractor=lambda model, x: model.feature_extractor(x),
        num_classes=args['num_classes'],
        device=args['device'],
        state_dim=3,
        action_dim=6,
        use_gradient_cache=args.get('use_gradient_cache', True),
        gradient_batch_accumulation=args.get('gradient_batch_accumulation', 8)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args['batch_size'], shuffle=True, 
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args['batch_size'], shuffle=False, 
        num_workers=4, pin_memory=True
    )
    
    # Initialize optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    
    # Use mixed precision if available
    use_amp = args['device'] == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['epochs'])
    
    # Training loop
    current_coreset = None
    best_val_acc = 0.0
    
    epoch_progress = tqdm(range(args['epochs']), desc="Epochs")
    for epoch in epoch_progress:
        # Select coreset
        current_coreset = mascs.select_coreset(
            model, train_loader, loss_fn, current_coreset, epoch
        )
        
        # Log coreset composition (class distribution)
        coreset_labels = [train_dataset[i][1] for i in current_coreset]
        class_counts = np.bincount(coreset_labels, minlength=args['num_classes'])
        for class_idx, count in enumerate(class_counts):
            writer.add_scalar(f'Coreset/class_{class_idx}', count, epoch)
        
        # Create coreset dataset and loader
        coreset_dataset = Subset(train_dataset, current_coreset)
        coreset_loader = DataLoader(
            coreset_dataset, batch_size=args['batch_size'], shuffle=True, 
            num_workers=4, pin_memory=True
        )
        
        # Train on coreset
        train_loss, train_acc = train_epoch(
            model, coreset_loader, optimizer, loss_fn, args['device'], use_amp, scaler, writer, epoch
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, loss_fn, args['device'], writer, epoch
        )
        mascs.performance_history.append(val_acc)
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar('Learning_rate', current_lr, epoch)
        
        # Update policy based on performance
        state = mascs.get_state()
        state_tensor = torch.tensor([state['accuracy'], state['loss'], state['diversity']], 
                                   device=args['device'], dtype=torch.float32)
        
        # Simple reward: improvement in validation accuracy
        reward = val_acc - best_val_acc if val_acc > best_val_acc else -0.1
        best_val_acc = max(best_val_acc, val_acc)
        writer.add_scalar('Reward', reward, epoch)
        
        next_state = mascs.get_state()
        next_state_tensor = torch.tensor([next_state['accuracy'], next_state['loss'], next_state['diversity']], 
                                        device=args['device'], dtype=torch.float32)
        
        # Update policy (simplified - in practice you'd choose an action)
        policy_loss, value_loss = mascs.update_policy(state_tensor, 'S_U', reward, next_state_tensor)
        writer.add_scalar('Policy_Loss', policy_loss, epoch)
        writer.add_scalar('Value_Loss', value_loss, epoch)
        
        # Log strategy weights
        strategy_weights = mascs.get_bayesian_weights(state, epoch)
        for strategy, weight in strategy_weights.items():
            writer.add_scalar(f'Strategy_weights/{strategy}', weight, epoch)
        
        # Update epoch progress bar
        epoch_progress.set_postfix({
            'Train Loss': f'{train_loss:.4f}',
            'Train Acc': f'{train_acc:.2f}%',
            'Val Acc': f'{val_acc:.2f}%',
            'Best Val Acc': f'{best_val_acc:.2f}%'
        })
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = exp_dir / f'best_model_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), model_path)
            writer.add_text('Best_model', f'Saved at epoch {epoch+1} with accuracy {best_val_acc:.2f}%', epoch)
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                # 'mascs': mascs,  # Skip MASCS due to lambda functions
            }
            torch.save(checkpoint, exp_dir / f'checkpoint_epoch_{epoch}.pth')
    
    return best_val_acc

def main():
    parser = argparse.ArgumentParser(description='MASCS for Multiple Datasets with TIMM')
    parser.add_argument('--budget', type=int, default=5000, help='Coreset budget')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--model', type=str, default='resnet18', help='TIMM model name')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--log_dir', type=str, default='./experiments', help='Log directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--run_multiple', action='store_true', help='Run multiple experiments')
    parser.add_argument('--num_sequential_runs', type=int, default=3, help='Number of sequential runs per configuration')
    parser.add_argument('--use_gradient_cache', action='store_true', default=True, help='Enable gradient score caching')
    parser.add_argument('--no_gradient_cache', dest='use_gradient_cache', action='store_false', help='Disable gradient score caching')
    parser.add_argument('--gradient_batch_accumulation', type=int, default=8, help='Number of batches to accumulate for gradient computation')
    
    args = parser.parse_args()
    
    # Create experiment manager
    exp_manager = ExperimentManager(args.log_dir)
    
    if args.run_multiple:
        # Run multiple experiments with different configurations
        configs = [
            {'model': 'resnet18', 'dataset': 'cifar10', 'budget': 5000, 'batch_size': 128, 'lr': 0.001, 'epochs': 50, 'data_dir': './data', 'device': args.device, 'use_gradient_cache': args.use_gradient_cache, 'gradient_batch_accumulation': args.gradient_batch_accumulation},
            {'model': 'resnet50', 'dataset': 'cifar10', 'budget': 5000, 'batch_size': 128, 'lr': 0.001, 'epochs': 50, 'data_dir': './data', 'device': args.device, 'use_gradient_cache': args.use_gradient_cache, 'gradient_batch_accumulation': args.gradient_batch_accumulation},
            {'model': 'efficientnet_b0', 'dataset': 'cifar10', 'budget': 5000, 'batch_size': 128, 'lr': 0.001, 'epochs': 50, 'data_dir': './data', 'device': args.device, 'use_gradient_cache': args.use_gradient_cache, 'gradient_batch_accumulation': args.gradient_batch_accumulation},
            {'model': 'resnet18', 'dataset': 'cifar100', 'budget': 5000, 'batch_size': 128, 'lr': 0.001, 'epochs': 50, 'data_dir': './data', 'device': args.device, 'use_gradient_cache': args.use_gradient_cache, 'gradient_batch_accumulation': args.gradient_batch_accumulation},
        ]
        
        results = exp_manager.run_multiple_experiments(configs, args.num_sequential_runs)
        print("All experiments completed!")
        
        # Print summary
        for result in results:
            if 'best_val_accuracy' in result:
                print(f"{result['experiment_name']}: {result['best_val_accuracy']:.2f}%")
    
    else:
        # Run single experiment
        config = {
            'model': args.model,
            'dataset': args.dataset,
            'budget': args.budget,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'epochs': args.epochs,
            'data_dir': args.data_dir,
            'device': args.device,
            'use_gradient_cache': args.use_gradient_cache,
            'gradient_batch_accumulation': args.gradient_batch_accumulation
        }
        
        result = exp_manager.run_experiment(config)
        print(f"Experiment completed: {result['experiment_name']}, Best accuracy: {result['best_val_accuracy']:.2f}%")

if __name__ == "__main__":
    main()