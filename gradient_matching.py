import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
from tqdm import tqdm
import copy


@dataclass
class DirectGradientMatchingConfig:
    # Synthetic dataset size
    num_synthetic: int = 500
    images_per_class: int = 50  # For CIFAR-10: 50 IPC
    
    # Optimization
    num_iterations: int = 2000
    lr_synthetic: float = 0.1
    momentum: float = 0.5
    
    # Gradient computation
    num_gradient_samples: int = 5000  # Use this many real samples
    gradient_batch_size: int = 256
    use_subspace: bool = True
    subspace_dim: int = 50
    
    # Multi-step matching (key innovation!)
    match_at_multiple_steps: bool = True
    training_steps: List[int] = None  # [0, 100, 200, 500]
    
    # Matching loss
    loss_type: str = 'mse'  # 'mse' or 'cosine'
    
    # Regularization
    diversity_weight: float = 0.01
    tv_weight: float = 0.001  # Total variation (smoothness)
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def __post_init__(self):
        if self.training_steps is None:
            self.training_steps = [0, 100, 200, 500]


def flatten_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def load_params(model: nn.Module, flat_params: torch.Tensor):
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat_params[offset:offset+numel].view_as(p.data))
        offset += numel


def get_gradient(model: nn.Module, data: torch.Tensor, labels: torch.Tensor, 
                 criterion: nn.Module) -> torch.Tensor:
    model.zero_grad()
    
    # Forward pass
    output = model(data)
    loss = criterion(output, labels)
    
    # Backward pass
    loss.backward()
    
    # Flatten gradients
    grad = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])
    
    return grad


class GradientSubspaceAnalyzer:
    def __init__(self, subspace_dim: int = 50):
        self.subspace_dim = subspace_dim
        self.basis = None  # [D, k] projection matrix
        self.singular_values = None
        self.mean = None
        self.explained_variance = None
    
    def fit(self, gradients: torch.Tensor, verbose: bool = True):
        # Center
        self.mean = gradients.mean(dim=0, keepdim=True)
        gradients_centered = gradients - self.mean
        
        # SVD (this is expensive for large D!)
        # For very large models, use randomized SVD
        if gradients.shape[1] > 100000:
            # Randomized SVD (approximate but faster)
            from sklearn.utils.extmath import randomized_svd
            U, S, Vt = randomized_svd(
                gradients_centered.cpu().numpy(),
                n_components=min(self.subspace_dim * 2, gradients.shape[0]),
                random_state=42
            )
            S = torch.from_numpy(S).to(gradients.device)
            Vt = torch.from_numpy(Vt).to(gradients.device)
        else:
            U, S, Vt = torch.linalg.svd(gradients_centered, full_matrices=False)
        
        # Keep top-k
        self.basis = Vt[:self.subspace_dim].T  # [D, k]
        self.singular_values = S[:self.subspace_dim]
        
        # Explained variance
        self.explained_variance = (S[:self.subspace_dim]**2).sum() / (S**2).sum()
    
    def project(self, gradients: torch.Tensor) -> torch.Tensor:
        if self.basis is None:
            raise ValueError("Must call fit() first!")
        
        # Handle both single and batch
        if gradients.dim() == 1:
            gradients = gradients.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        
        # Center and project
        centered = gradients - self.mean
        projected = centered @ self.basis
        
        if squeeze:
            projected = projected.squeeze(0)
        
        return projected
    
    def reconstruct(self, projected: torch.Tensor) -> torch.Tensor:
        if self.basis is None:
            raise ValueError("Must call fit() first!")
        
        # Handle both single and batch
        if projected.dim() == 1:
            projected = projected.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        
        # Reconstruct
        reconstructed = projected @ self.basis.T + self.mean
        
        if squeeze:
            reconstructed = reconstructed.squeeze(0)
        
        return reconstructed


class DirectGradientMatcher:
    def __init__(
        self,
        model: nn.Module,
        config: DirectGradientMatchingConfig
    ):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.criterion = nn.CrossEntropyLoss()
        
        self.model.to(self.device)
        
        # Subspace analyzer
        if config.use_subspace:
            self.subspace = GradientSubspaceAnalyzer(config.subspace_dim)
        else:
            self.subspace = None
        
        # Storage
        self.target_gradients = {}  # {step: gradient}
        self.initial_params = None
    
    def compute_target_gradients(
        self,
        real_data_loader: torch.utils.data.DataLoader
    ):
        # Save initial parameters
        self.initial_params = flatten_params(self.model).detach().clone()
        
        # Collect all gradients for subspace fitting
        all_gradients_for_subspace = []
        
        # Train and record gradients at key steps
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config.lr_synthetic,
            momentum=self.config.momentum
        )
        
        step = 0
        gradients_at_steps = {s: [] for s in self.config.training_steps}
        
        max_step = max(self.config.training_steps)
        pbar = tqdm(total=max_step, desc="Training")
        
        for epoch in range(100):  # Enough epochs to reach max_step
            for batch_idx, (data, labels) in enumerate(real_data_loader):
                data, labels = data.to(self.device), labels.to(self.device)
                
                # Compute gradient
                grad = get_gradient(self.model, data, labels, self.criterion)
                
                # Record if at key step
                if step in self.config.training_steps:
                    gradients_at_steps[step].append(grad.detach())
                
                # Always collect for subspace (from early steps)
                if step < 1000:
                    all_gradients_for_subspace.append(grad.detach())
                
                # Update model
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, labels)
                loss.backward()
                optimizer.step()
                
                step += 1
                pbar.update(1)
                
                if step > max_step:
                    break
            
            if step > max_step:
                break
        
        pbar.close()
        
        # Fit subspace if needed
        if self.subspace is not None:
            subspace_grads = torch.stack(all_gradients_for_subspace[:1000])
            self.subspace.fit(subspace_grads, verbose=True)
        
        # Average gradients at each step and project
        for step in self.config.training_steps:
            if len(gradients_at_steps[step]) > 0:
                avg_grad = torch.stack(gradients_at_steps[step]).mean(dim=0)
                
                # Project to subspace
                if self.subspace is not None:
                    avg_grad = self.subspace.project(avg_grad)
                
                self.target_gradients[step] = avg_grad
        
        # Reset model to initial state
        load_params(self.model, self.initial_params)
    
    def initialize_synthetic_data(
        self,
        num_classes: int,
        data_shape: Tuple[int, ...]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Random initialization
        synthetic_data = torch.randn(
            self.config.num_synthetic, *data_shape,
            device=self.device,
            requires_grad=True
        )
        
        # Balanced labels
        synthetic_labels = torch.arange(
            self.config.num_synthetic,
            device=self.device
        ) % num_classes
        
        return synthetic_data, synthetic_labels
    
    def compute_matching_loss(
        self,
        synthetic_data: torch.Tensor,
        synthetic_labels: torch.Tensor
    ) -> torch.Tensor:
        total_loss = 0.0
        
        # Reset model
        load_params(self.model, self.initial_params)
        
        # Match at each training step
        for step_idx, step in enumerate(self.config.training_steps):
            # Compute synthetic gradient at this step
            syn_grad = get_gradient(
                self.model, synthetic_data, synthetic_labels, self.criterion
            )
            
            # Project to subspace
            if self.subspace is not None:
                syn_grad = self.subspace.project(syn_grad)
            
            # Get target
            target_grad = self.target_gradients[step]
            
            # Matching loss
            if self.config.loss_type == 'mse':
                loss = F.mse_loss(syn_grad, target_grad)
            elif self.config.loss_type == 'cosine':
                cos_sim = F.cosine_similarity(
                    syn_grad.unsqueeze(0),
                    target_grad.unsqueeze(0)
                )
                loss = 1 - cos_sim
            else:
                raise ValueError(f"Unknown loss: {self.config.loss_type}")
            
            total_loss += loss
            
            # Update model for next step (if not last)
            if step_idx < len(self.config.training_steps) - 1:
                optimizer = torch.optim.SGD(
                    self.model.parameters(),
                    lr=self.config.lr_synthetic
                )
                
                next_step = self.config.training_steps[step_idx + 1]
                for _ in range(next_step - step):
                    optimizer.zero_grad()
                    output = self.model(synthetic_data.detach())
                    loss_train = self.criterion(output, synthetic_labels)
                    loss_train.backward()
                    optimizer.step()
        
        # Average over steps
        total_loss = total_loss / len(self.config.training_steps)
        
        return total_loss
    
    def compute_regularization(
        self,
        synthetic_data: torch.Tensor
    ) -> torch.Tensor:
        reg_loss = 0.0
        
        # Diversity
        if self.config.diversity_weight > 0:
            flat = synthetic_data.view(self.config.num_synthetic, -1)
            # Pairwise distances
            dists = torch.cdist(flat, flat)
            # Maximize minimum distance
            diversity_loss = -dists.mean()
            reg_loss += self.config.diversity_weight * diversity_loss
        
        # Total variation (smoothness for images)
        if self.config.tv_weight > 0 and len(synthetic_data.shape) == 4:
            # Horizontal differences
            diff_h = synthetic_data[:, :, :, 1:] - synthetic_data[:, :, :, :-1]
            # Vertical differences
            diff_v = synthetic_data[:, :, 1:, :] - synthetic_data[:, :, :-1, :]
            tv_loss = (diff_h.abs().mean() + diff_v.abs().mean())
            reg_loss += self.config.tv_weight * tv_loss
        
        return reg_loss
    
    def distill(
        self,
        real_data_loader: torch.utils.data.DataLoader,
        num_classes: int,
        data_shape: Tuple[int, ...]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Step 1: Compute target gradients
        self.compute_target_gradients(real_data_loader)
        
        # Step 2: Initialize synthetic data
        synthetic_data, synthetic_labels = self.initialize_synthetic_data(
            num_classes, data_shape
        )
        
        # Step 3: Optimization loop
        optimizer = torch.optim.SGD(
            [synthetic_data],
            lr=self.config.lr_synthetic,
            momentum=self.config.momentum
        )
        
        best_loss = float('inf')
        best_data = None
        
        pbar = tqdm(range(self.config.num_iterations), desc="Distilling")
        
        for it in pbar:
            # Compute matching loss
            match_loss = self.compute_matching_loss(
                synthetic_data, synthetic_labels
            )
            
            # Regularization
            reg_loss = self.compute_regularization(synthetic_data)
            
            # Total loss
            total_loss = match_loss + reg_loss
            
            # Update
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Clamp to valid pixel range
            with torch.no_grad():
                synthetic_data.clamp_(0, 1)
            
            # Track best
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_data = synthetic_data.detach().clone()
            
            # Log
            if it % 100 == 0:
                pbar.set_postfix({
                    'match': f'{match_loss.item():.4f}',
                    'reg': f'{reg_loss.item():.4f}',
                    'total': f'{total_loss.item():.4f}'
                })
        
        return best_data, synthetic_labels


def evaluate_distilled_dataset(
    synthetic_data: torch.Tensor,
    synthetic_labels: torch.Tensor,
    test_loader: torch.utils.data.DataLoader,
    model_class: type,
    num_epochs: int = 10,
    device: str = 'cuda'
) -> float:
    # Create fresh model
    model = model_class().to(device)
    
    # Create synthetic dataloader
    syn_dataset = torch.utils.data.TensorDataset(synthetic_data, synthetic_labels)
    syn_loader = torch.utils.data.DataLoader(
        syn_dataset, batch_size=256, shuffle=True
    )
    
    # Train on synthetic data
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for data, labels in syn_loader:
            data, labels = data.to(device), labels.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
    
    # Evaluate on test set
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    
    return accuracy


if __name__ == "__main__":
    # Simple test with dummy data
    pass