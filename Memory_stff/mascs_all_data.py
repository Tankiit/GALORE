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
import os
from torch.utils.data import Dataset, Subset
import scipy.io as sio  # For SVHN dataset

class MultiDatasetMASCS:
    """MASCS implementation supporting multiple datasets"""
    def __init__(self, dataset, budget, feature_extractor, num_classes, 
                 memory_window=100, device='cuda' if torch.cuda.is_available() else 'cpu',
                 state_dim=64, action_dim=5, gamma=0.95):
        self.dataset = dataset
        self.budget = budget
        self.feature_extractor = feature_extractor
        self.num_classes = num_classes
        self.memory_window = memory_window
        self.device = device
        self.model_version = 0
        
        # Memory structures
        self.sample_memories = [None] * len(dataset)
        self.memory_buffer = deque(maxlen=memory_window * 100)
        self.selection_history = defaultdict(int)
        self.validation_improvements = defaultdict(list)
        
        # Caching
        self._cached_features = None
        self._cached_labels = None
        self._cached_model_version = -1
        
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
            for x, y in dataloader:
                x = x.to(self.device)
                feat = self.feature_extractor(model, x)
                features_list.append(feat.cpu())
                labels_list.append(y)
        
        # Cache results
        self._cached_features = torch.cat(features_list, dim=0).numpy()
        self._cached_labels = torch.cat(labels_list, dim=0).numpy()
        self._cached_model_version = self.model_version
        
        return self._cached_features, self._cached_labels

    def compute_gradient_score_batch(self, model, x_batch, y_batch, loss_fn):
        """Compute gradient scores in batches for efficiency"""
        model.zero_grad()
        original_training = model.training
        model.eval()
        
        try:
            with torch.enable_grad():
                # Forward pass
                logits = model(x_batch)
                loss = loss_fn(logits, y_batch)
                loss.backward()
            
            # Compute gradient norms for all parameters
            grad_norms = []
            for param in model.parameters():
                if param.grad is not None:
                    # Compute norm for each sample in batch
                    batch_grad_norms = param.grad.view(param.grad.size(0), -1).norm(dim=1)
                    grad_norms.append(batch_grad_norms.cpu().numpy())
            
            # Average across parameters
            if grad_norms:
                return np.mean(grad_norms, axis=0)
            return np.zeros(len(x_batch))
        finally:
            model.train(original_training)

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
        
        # Compute scores in batches
        model.eval()
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(dataloader):
                x, y = x.to(self.device), y.to(self.device)
                start_idx = batch_idx * dataloader.batch_size
                end_idx = min(start_idx + len(x), n_samples)
                
                # Compute scores in batches
                all_scores['S_U'][start_idx:end_idx] = self.compute_uncertainty_score(model, x).cpu().numpy()
                all_scores['S_B'][start_idx:end_idx] = self.compute_boundary_score(model, x).cpu().numpy()
                all_scores['S_G'][start_idx:end_idx] = self.compute_gradient_score_batch(model, x, y, loss_fn)
        
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
        
        # Update model version
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

# Training functions
def train_epoch(model, train_loader, optimizer, loss_fn, device, use_amp, scaler):
    """Optimized training function"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=use_amp):
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
    
    return total_loss / len(train_loader), 100.0 * correct / total

def validate(model, val_loader, loss_fn, device):
    """Validation function"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = loss_fn(outputs, y)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    return total_loss / len(val_loader), 100.0 * correct / total

def train_with_mascs(model, train_dataset, val_dataset, args):
    """Training loop with optimized MASCS"""
    # Initialize MASCS
    mascs = MultiDatasetMASCS(
        dataset=train_dataset,
        budget=args.budget,
        feature_extractor=lambda model, x: model.feature_extractor(x),
        num_classes=args.num_classes,
        device=args.device
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True
    )
    
    # Initialize optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    
    # Use mixed precision if available
    use_amp = hasattr(torch.cuda, 'amp') and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    current_coreset = None
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # Select coreset
        current_coreset = mascs.select_coreset(
            model, train_loader, loss_fn, current_coreset, epoch
        )
        
        # Create coreset dataset and loader
        coreset_dataset = torch.utils.data.Subset(train_dataset, current_coreset)
        coreset_loader = torch.utils.data.DataLoader(
            coreset_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True
        )
        
        # Train on coreset
        train_loss, train_acc = train_epoch(
            model, coreset_loader, optimizer, loss_fn, args.device, use_amp, scaler
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, loss_fn, args.device)
        mascs.performance_history.append(val_acc)
        
        # Update learning rate
        scheduler.step()
        
        # Update policy based on performance
        state = mascs.get_state()
        state_tensor = torch.tensor([state['accuracy'], state['loss'], state['diversity']], 
                                   device=args.device, dtype=torch.float32)
        
        # Simple reward: improvement in validation accuracy
        reward = val_acc - best_val_acc if val_acc > best_val_acc else -0.1
        best_val_acc = max(best_val_acc, val_acc)
        
        next_state = mascs.get_state()
        next_state_tensor = torch.tensor([next_state['accuracy'], next_state['loss'], next_state['diversity']], 
                                        device=args.device, dtype=torch.float32)
        
        # Update policy (simplified - in practice you'd choose an action)
        mascs.update_policy(state_tensor, 'S_U', reward, next_state_tensor)
        
        print(f'Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Best Val Acc: {best_val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'best_model_{args.dataset}_epoch_{epoch+1}.pth')
    
    return best_val_acc

# Dataset loading functions
def get_cifar10_datasets(data_dir='./data'):
    """Get CIFAR-10 datasets with appropriate transforms"""
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Transformation for validation
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Download and load training dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    
    # Download and load validation dataset
    val_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=val_transform
    )
    
    return train_dataset, val_dataset

def get_cifar100_datasets(data_dir='./data'):
    """Get CIFAR-100 datasets with appropriate transforms"""
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    # Transformation for validation
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    # Download and load training dataset
    train_dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    
    # Download and load validation dataset
    val_dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=val_transform
    )
    
    return train_dataset, val_dataset

def get_svhn_datasets(data_dir='./data'):
    """Get SVHN datasets with appropriate transforms"""
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
    ])
    
    # Transformation for validation
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
    ])
    
    # Download and load training dataset
    train_dataset = torchvision.datasets.SVHN(
        root=data_dir, split='train', download=True, transform=train_transform
    )
    
    # Download and load validation dataset
    val_dataset = torchvision.datasets.SVHN(
        root=data_dir, split='test', download=True, transform=val_transform
    )
    
    return train_dataset, val_dataset

def get_imagenet_datasets(data_dir='./data'):
    """Get ImageNet datasets with appropriate transforms"""
    # Use TIMM's data config for the model
    config = resolve_data_config({}, model='resnet50')
    train_transform = create_transform(**config, is_training=True)
    val_transform = create_transform(**config, is_training=False)
    
    # Load training dataset
    train_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(data_dir, 'train'),
        transform=train_transform
    )
    
    # Load validation dataset
    val_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(data_dir, 'val'),
        transform=val_transform
    )
    
    return train_dataset, val_dataset

def get_dataset(dataset_name, data_dir):
    """Get the appropriate dataset based on name"""
    if dataset_name == 'cifar10':
        return get_cifar10_datasets(data_dir)
    elif dataset_name == 'cifar100':
        return get_cifar100_datasets(data_dir)
    elif dataset_name == 'svhn':
        return get_svhn_datasets(data_dir)
    elif dataset_name == 'imagenet':
        return get_imagenet_datasets(data_dir)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def main():
    parser = argparse.ArgumentParser(description='MASCS for Multiple Datasets')
    parser.add_argument('--dataset', type=str, default='cifar10', 
                        choices=['cifar10', 'cifar100', 'svhn', 'imagenet'],
                        help='Dataset to use')
    parser.add_argument('--budget', type=int, default=5000, help='Coreset budget')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--model', type=str, default='resnet18', help='TIMM model name')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    
    args = parser.parse_args()
    
    # Set number of classes based on dataset
    if args.dataset == 'cifar10':
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        args.num_classes = 100
    elif args.dataset == 'svhn':
        args.num_classes = 10
    elif args.dataset == 'imagenet':
        args.num_classes = 1000
    
    print(f"Using device: {args.device}")
    print(f"Using model: {args.model}")
    print(f"Using dataset: {args.dataset} with {args.num_classes} classes")
    
    # Get datasets
    train_dataset, val_dataset = get_dataset(args.dataset, args.data_dir)
    
    # Initialize model
    model = TIMMModel(model_name=args.model, num_classes=args.num_classes, pretrained=True).to(args.device)
    
    # Adjust budget for ImageNet if needed
    if args.dataset == 'imagenet' and args.budget > 100000:
        args.budget = 100000  # Reasonable limit for ImageNet
        print(f"Adjusting budget to {args.budget} for ImageNet")
    
    # Train with MASCS
    best_val_acc = train_with_mascs(model, train_dataset, val_dataset, args)
    
    print(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    main()