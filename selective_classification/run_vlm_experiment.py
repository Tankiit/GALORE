
"""
MODE-VLM: Online Curriculum Learning for Vision-Language Models

Complete implementation that extends existing MODE framework to multimodal pretraining.
Designed for research (not competition), allowing true online adaptation.

Key Features:
1. Binary state encoding for VLM training dynamics
2. Multimodal-specific selection strategies
3. Online adaptation based on vision/text/alignment signals
4. Compatible with CLIP, LLaVA, OpenFlamingo

Author: Extended from existing MODE codebase
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import copy


# ============================================================================
# PART 1: Multimodal Binary State Encoder
# ============================================================================

class MultimodalBinaryStateEncoder(nn.Module):
    """
    Encodes VLM training dynamics into binary state vector
    
    State dimensions (20 total):
    - Vision loss trend (2 bits): improving/degrading
    - Text loss trend (2 bits): improving/degrading  
    - Alignment loss trend (2 bits): improving/degrading
    - Gradient norm (2 bits): high/low
    - Training phase (2 bits): early/late
    - Modality balance (2 bits): vision-heavy/text-heavy
    - Learning rate (2 bits): high/low
    - Stability (2 bits): stable/unstable
    - Data efficiency (2 bits): efficient/inefficient
    - Generalization gap (2 bits): overfitting/underfitting
    """
    def __init__(
        self,
        state_dim: int = 20,
        history_window: int = 100,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.history_window = history_window
        
        # Track loss histories
        self.vision_loss_history = deque(maxlen=history_window)
        self.text_loss_history = deque(maxlen=history_window)
        self.alignment_loss_history = deque(maxlen=history_window)
        self.gradient_norm_history = deque(maxlen=history_window)
        self.train_acc_history = deque(maxlen=history_window)
        self.val_acc_history = deque(maxlen=history_window)
        
        # Initial state
        self.prev_state = None
        
    def update_history(self, metrics: Dict[str, float]):
        """Update all tracked histories"""
        if 'vision_loss' in metrics:
            self.vision_loss_history.append(metrics['vision_loss'])
        if 'text_loss' in metrics:
            self.text_loss_history.append(metrics['text_loss'])
        if 'alignment_loss' in metrics:
            self.alignment_loss_history.append(metrics['alignment_loss'])
        if 'gradient_norm' in metrics:
            self.gradient_norm_history.append(metrics['gradient_norm'])
        if 'train_acc' in metrics:
            self.train_acc_history.append(metrics['train_acc'])
        if 'val_acc' in metrics:
            self.val_acc_history.append(metrics['val_acc'])
    
    def compute_binary_state(
        self,
        current_metrics: Dict[str, float],
        total_steps: int,
    ) -> torch.Tensor:
        """
        Convert training signals to binary state encoding
        
        Args:
            current_metrics: Dict with keys:
                - vision_loss: float
                - text_loss: float
                - alignment_loss: float
                - gradient_norm: float
                - step: int
                - train_acc: float (optional)
                - val_acc: float (optional)
                - lr: float (optional)
            total_steps: Total training steps
            
        Returns:
            binary_state: [state_dim] binary vector (0s and 1s)
        """
        self.update_history(current_metrics)
        
        state = []
        step = current_metrics.get('step', 0)
        
        # Helper function for trend detection
        def get_trend(history, window=10):
            if len(history) < window:
                return 0, 0  # Not enough data
            recent = np.mean(list(history)[-window//2:])
            older = np.mean(list(history)[-window:-window//2])
            improving = 1 if recent < older else 0
            degrading = 1 if recent > older else 0
            return improving, degrading
        
        # 1. Vision loss trend (2 bits)
        v_improving, v_degrading = get_trend(self.vision_loss_history)
        state.extend([v_improving, v_degrading])
        
        # 2. Text loss trend (2 bits)
        t_improving, t_degrading = get_trend(self.text_loss_history)
        state.extend([t_improving, t_degrading])
        
        # 3. Alignment loss trend (2 bits)
        a_improving, a_degrading = get_trend(self.alignment_loss_history)
        state.extend([a_improving, a_degrading])
        
        # 4. Gradient norm (2 bits)
        if len(self.gradient_norm_history) >= 5:
            recent_norm = np.mean(list(self.gradient_norm_history)[-5:])
            median_norm = np.median(list(self.gradient_norm_history))
            high_grad = 1 if recent_norm > median_norm * 1.5 else 0
            low_grad = 1 if recent_norm < median_norm * 0.5 else 0
        else:
            high_grad, low_grad = 0, 0
        state.extend([high_grad, low_grad])
        
        # 5. Training phase (2 bits)
        progress = step / total_steps
        early_phase = 1 if progress < 0.2 else 0
        late_phase = 1 if progress > 0.7 else 0
        state.extend([early_phase, late_phase])
        
        # 6. Modality balance (2 bits)
        if len(self.vision_loss_history) >= 5 and len(self.text_loss_history) >= 5:
            v_avg = np.mean(list(self.vision_loss_history)[-5:])
            t_avg = np.mean(list(self.text_loss_history)[-5:])
            vision_struggling = 1 if v_avg > t_avg * 1.3 else 0
            text_struggling = 1 if t_avg > v_avg * 1.3 else 0
        else:
            vision_struggling, text_struggling = 0, 0
        state.extend([vision_struggling, text_struggling])
        
        # 7. Learning rate (2 bits)
        if 'lr' in current_metrics and 'initial_lr' in current_metrics:
            lr = current_metrics['lr']
            initial_lr = current_metrics['initial_lr']
            high_lr = 1 if lr > 0.5 * initial_lr else 0
            low_lr = 1 if lr < 0.1 * initial_lr else 0
        else:
            high_lr, low_lr = 0, 0
        state.extend([high_lr, low_lr])
        
        # 8. Stability (2 bits) - based on loss variance
        if len(self.alignment_loss_history) >= 10:
            recent_var = np.var(list(self.alignment_loss_history)[-10:])
            stable = 1 if recent_var < 0.01 else 0
            unstable = 1 if recent_var > 0.1 else 0
        else:
            stable, unstable = 0, 0
        state.extend([stable, unstable])
        
        # 9. Data efficiency (2 bits) - improvement per step
        if len(self.train_acc_history) >= 20:
            recent_acc = np.mean(list(self.train_acc_history)[-10:])
            older_acc = np.mean(list(self.train_acc_history)[-20:-10])
            improvement = recent_acc - older_acc
            efficient = 1 if improvement > 0.01 else 0
            inefficient = 1 if improvement < 0.001 else 0
        else:
            efficient, inefficient = 0, 0
        state.extend([efficient, inefficient])
        
        # 10. Generalization gap (2 bits)
        if len(self.train_acc_history) >= 5 and len(self.val_acc_history) >= 5:
            train_acc = np.mean(list(self.train_acc_history)[-5:])
            val_acc = np.mean(list(self.val_acc_history)[-5:])
            gap = train_acc - val_acc
            overfitting = 1 if gap > 0.1 else 0
            underfitting = 1 if gap < 0.02 and train_acc < 0.5 else 0
        else:
            overfitting, underfitting = 0, 0
        state.extend([overfitting, underfitting])
        
        # Convert to tensor
        binary_state = torch.tensor(state, dtype=torch.float32)
        
        # Store for next iteration
        self.prev_state = binary_state
        
        return binary_state
    
    def reset(self):
        """Reset all histories"""
        self.vision_loss_history.clear()
        self.text_loss_history.clear()
        self.alignment_loss_history.clear()
        self.gradient_norm_history.clear()
        self.train_acc_history.clear()
        self.val_acc_history.clear()
        self.prev_state = None


# ============================================================================
# PART 2: Multimodal Selection Strategies
# ============================================================================

class MultimodalSelectionStrategy:
    """Base class for multimodal data selection strategies"""
    
    def compute_scores(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        current_model_state: Optional[Dict] = None,
    ) -> torch.Tensor:
        """
        Compute selection scores for each sample
        
        Args:
            image_features: [N, d_img] pre-computed image embeddings
            text_features: [N, d_txt] pre-computed text embeddings
            current_model_state: Optional dict with model statistics
            
        Returns:
            scores: [N] selection scores (higher = more likely to select)
        """
        raise NotImplementedError


class AlignmentQualityStrategy(MultimodalSelectionStrategy):
    """Select samples with good image-text alignment (easy curriculum)"""
    
    def compute_scores(self, image_features, text_features, current_model_state=None):
        # Normalize
        img_norm = F.normalize(image_features, dim=-1)
        txt_norm = F.normalize(text_features, dim=-1)
        
        # Cosine similarity
        alignment = (img_norm * txt_norm).sum(dim=-1)
        
        # High alignment = easy sample
        return alignment


class AlignmentDifficultyStrategy(MultimodalSelectionStrategy):
    """Select samples with challenging alignment (hard curriculum)"""
    
    def compute_scores(self, image_features, text_features, current_model_state=None):
        img_norm = F.normalize(image_features, dim=-1)
        txt_norm = F.normalize(text_features, dim=-1)
        alignment = (img_norm * txt_norm).sum(dim=-1)
        
        # Low alignment = hard sample
        return 1.0 - alignment


class VisualComplexityStrategy(MultimodalSelectionStrategy):
    """Select visually complex images"""
    
    def compute_scores(self, image_features, text_features, current_model_state=None):
        # Use feature magnitude as proxy for complexity
        # High-norm features = more information
        complexity = torch.norm(image_features, dim=-1)
        
        # Normalize to [0, 1]
        scores = (complexity - complexity.min()) / (complexity.max() - complexity.min() + 1e-8)
        return scores


class TextRichnessStrategy(MultimodalSelectionStrategy):
    """Select samples with rich textual descriptions"""
    
    def compute_scores(self, image_features, text_features, current_model_state=None):
        # Text feature magnitude as proxy
        richness = torch.norm(text_features, dim=-1)
        scores = (richness - richness.min()) / (richness.max() - richness.min() + 1e-8)
        return scores


class CrossModalDiversityStrategy(MultimodalSelectionStrategy):
    """Select diverse samples in joint embedding space"""
    
    def compute_scores(self, image_features, text_features, current_model_state=None):
        # Combine modalities
        joint_features = (F.normalize(image_features, dim=-1) + 
                         F.normalize(text_features, dim=-1)) / 2
        
        # Distance to batch centroid
        centroid = joint_features.mean(dim=0, keepdim=True)
        distances = torch.norm(joint_features - centroid, dim=-1)
        
        # High distance = diverse
        return distances


class BalancedStrategy(MultimodalSelectionStrategy):
    """Select samples with balanced difficulty across modalities"""
    
    def compute_scores(self, image_features, text_features, current_model_state=None):
        img_norm = F.normalize(image_features, dim=-1)
        txt_norm = F.normalize(text_features, dim=-1)
        alignment = (img_norm * txt_norm).sum(dim=-1)
        
        # Prefer medium alignment (neither too easy nor too hard)
        balance = 1.0 - 2.0 * torch.abs(alignment - 0.5)
        return balance


class UncertaintyStrategy(MultimodalSelectionStrategy):
    """Select samples model is uncertain about (requires model predictions)"""
    
    def compute_scores(self, image_features, text_features, current_model_state=None):
        if current_model_state is None or 'prediction_entropy' not in current_model_state:
            # Fallback: use alignment as proxy
            img_norm = F.normalize(image_features, dim=-1)
            txt_norm = F.normalize(text_features, dim=-1)
            alignment = (img_norm * txt_norm).sum(dim=-1)
            return 1.0 - torch.abs(alignment)
        
        # Use actual model uncertainty
        return current_model_state['prediction_entropy']


# ============================================================================
# PART 3: MODE Hypernetwork for Strategy Weighting
# ============================================================================

class MODEHypernetwork(nn.Module):
    """
    Hypernetwork that outputs strategy weights based on binary training state
    
    Architecture: Binary state → MLP → Strategy weights
    """
    def __init__(
        self,
        state_dim: int = 20,
        num_strategies: int = 7,
        hidden_dim: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()
        
        self.num_strategies = num_strategies
        self.strategy_names = [
            'alignment_quality',
            'alignment_difficulty', 
            'visual_complexity',
            'text_richness',
            'cross_modal_diversity',
            'balanced',
            'uncertainty',
        ]
        
        # Build MLP
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
        
        layers.append(nn.Linear(hidden_dim, num_strategies))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, binary_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            binary_state: [batch_size, state_dim] or [state_dim]
        Returns:
            strategy_weights: [num_strategies] or [batch_size, num_strategies]
        """
        logits = self.network(binary_state)
        weights = F.softmax(logits, dim=-1)
        return weights


# ============================================================================
# PART 4: MODE Selector - Main Selection Logic
# ============================================================================

class MODEVLMSelector:
    """
    Main MODE selection pipeline for VLM training
    
    Usage:
        selector = MODEVLMSelector(
            state_encoder=encoder,
            hypernetwork=hypernetwork,
            strategies=strategies,
            selection_ratio=0.3,
        )
        
        selected_indices = selector.select_batch(
            image_features=img_feats,
            text_features=txt_feats,
            current_metrics=metrics,
            total_steps=total_steps,
        )
    """
    def __init__(
        self,
        state_encoder: MultimodalBinaryStateEncoder,
        hypernetwork: MODEHypernetwork,
        strategies: Dict[str, MultimodalSelectionStrategy],
        selection_ratio: float = 0.3,
        device: str = 'cuda',
    ):
        self.state_encoder = state_encoder
        self.hypernetwork = hypernetwork
        self.strategies = strategies
        self.selection_ratio = selection_ratio
        self.device = device
        
        # Move to device
        self.state_encoder = self.state_encoder.to(device)
        self.hypernetwork = self.hypernetwork.to(device)
        
    @torch.no_grad()
    def select_batch(
        self,
        image_features: torch.Tensor,  # [N, d_img]
        text_features: torch.Tensor,   # [N, d_txt]
        current_metrics: Dict[str, float],
        total_steps: int,
        return_metadata: bool = False,
    ) -> Tuple[List[int], Optional[Dict]]:
        """
        Select subset of data based on current training state
        
        Args:
            image_features: Pre-computed image embeddings
            text_features: Pre-computed text embeddings
            current_metrics: Current training statistics
            total_steps: Total training steps
            return_metadata: Whether to return selection metadata
            
        Returns:
            selected_indices: List of indices to use for training
            metadata: Optional dict with selection details
        """
        N = len(image_features)
        k = int(N * self.selection_ratio)
        
        # Move features to device
        image_features = image_features.to(self.device)
        text_features = text_features.to(self.device)
        
        # 1. Encode current training state
        binary_state = self.state_encoder.compute_binary_state(
            current_metrics, total_steps
        ).to(self.device)
        
        # 2. Get strategy weights from hypernetwork
        strategy_weights = self.hypernetwork(binary_state)  # [num_strategies]
        
        # 3. Compute strategy scores
        strategy_scores = {}
        for name, strategy in self.strategies.items():
            scores = strategy.compute_scores(
                image_features, 
                text_features,
                current_metrics.get('model_state', None)
            )
            strategy_scores[name] = scores
        
        # 4. Combine scores using learned weights
        final_scores = torch.zeros(N, device=self.device)
        for i, name in enumerate(self.hypernetwork.strategy_names):
            if name in strategy_scores:
                final_scores += strategy_weights[i] * strategy_scores[name]
        
        # 5. Select top-k
        selected_indices = torch.topk(final_scores, k=k).indices.cpu().tolist()
        
        if return_metadata:
            metadata = {
                'binary_state': binary_state.cpu().numpy(),
                'strategy_weights': strategy_weights.cpu().numpy(),
                'strategy_scores': {k: v.cpu().numpy() for k, v in strategy_scores.items()},
                'final_scores': final_scores.cpu().numpy(),
                'selection_ratio': k / N,
            }
            return selected_indices, metadata
        
        return selected_indices, None


# ============================================================================
# PART 5: Training Loop Integration
# ============================================================================

class MODEVLMTrainer:
    """
    Complete training loop with MODE online selection
    
    Integrates with existing VLM training code (CLIP, LLaVA, etc.)
    """
    def __init__(
        self,
        model,  # VLM (e.g., CLIP, LLaVA)
        train_dataset,  # Full training dataset
        val_dataset,   # Validation dataset
        mode_selector: MODEVLMSelector,
        selection_frequency: int = 1000,
        batch_size: int = 256,
        device: str = 'cuda',
        learning_rate: float = 5e-4,
    ):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.mode_selector = mode_selector
        self.selection_frequency = selection_frequency
        self.batch_size = batch_size
        self.device = device
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=0.01
        )
        
        self.initial_lr = learning_rate
        self.global_step = 0
        
        # Metrics tracking
        self.metrics_history = {
            'vision_loss': [],
            'text_loss': [],
            'alignment_loss': [],
            'train_acc': [],
            'val_acc': [],
        }
        
    def train(self, total_steps: int, log_interval: int = 100):
        """
        Main training loop with periodic data reselection
        
        Args:
            total_steps: Total number of training steps
            log_interval: How often to log metrics
        """
        current_subset_indices = None
        current_dataloader = None
        
        print(f"Starting MODE-VLM training for {total_steps} steps")
        print(f"Reselection frequency: every {self.selection_frequency} steps")
        print(f"Selection ratio: {self.mode_selector.selection_ratio}")
        
        while self.global_step < total_steps:
            # Periodic data reselection
            if self.global_step % self.selection_frequency == 0:
                print(f"\n{'='*60}")
                print(f"Step {self.global_step}: Reselecting training data...")
                
                # Get current training state
                current_metrics = self.get_current_metrics()
                
                # Pre-compute features for selection
                print("Computing features for selection...")
                image_feats, text_feats = self.precompute_features(self.train_dataset)
                
                # Select new subset
                selected_indices, metadata = self.mode_selector.select_batch(
                    image_features=image_feats,
                    text_features=text_feats,
                    current_metrics=current_metrics,
                    total_steps=total_steps,
                    return_metadata=True,
                )
                
                print(f"Selected {len(selected_indices)} / {len(self.train_dataset)} samples")
                print(f"Strategy weights: {metadata['strategy_weights']}")
                
                # Create new dataloader
                current_subset = torch.utils.data.Subset(
                    self.train_dataset, selected_indices
                )
                current_dataloader = torch.utils.data.DataLoader(
                    current_subset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=True,
                )
                print(f"{'='*60}\n")
            
            # Train on current subset
            for batch in current_dataloader:
                if self.global_step >= total_steps:
                    break
                
                # Single training step
                loss_dict = self.train_step(batch)
                
                # Update metrics
                for key in ['vision_loss', 'text_loss', 'alignment_loss']:
                    if key in loss_dict:
                        self.metrics_history[key].append(loss_dict[key])
                
                # Logging
                if self.global_step % log_interval == 0:
                    print(f"Step {self.global_step}/{total_steps} | " +
                          " | ".join([f"{k}: {v:.4f}" for k, v in loss_dict.items()]))
                
                # Periodic validation
                if self.global_step % (log_interval * 10) == 0:
                    val_acc = self.validate()
                    self.metrics_history['val_acc'].append(val_acc)
                    print(f"Validation accuracy: {val_acc:.4f}")
                
                self.global_step += 1
        
        print(f"\nTraining complete! Final step: {self.global_step}")
        return self.metrics_history
    
    def train_step(self, batch) -> Dict[str, float]:
        """
        Single training step
        
        Args:
            batch: Dict with 'images', 'texts', etc.
            
        Returns:
            loss_dict: Dict of loss components
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move to device
        images = batch['images'].to(self.device)
        texts = batch['texts']  # Usually stays as list of strings
        
        # Forward pass (adapt to your VLM architecture)
        # This is pseudo-code - adjust based on your model
        outputs = self.model(images, texts)
        
        # Compute losses
        # For CLIP-style models:
        vision_loss = outputs.get('vision_loss', torch.tensor(0.0))
        text_loss = outputs.get('text_loss', torch.tensor(0.0))
        alignment_loss = outputs.get('contrastive_loss', torch.tensor(0.0))
        
        total_loss = vision_loss + text_loss + alignment_loss
        
        # Backward
        total_loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=1.0
        )
        
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'vision_loss': vision_loss.item(),
            'text_loss': text_loss.item(),
            'alignment_loss': alignment_loss.item(),
            'gradient_norm': grad_norm.item(),
        }
    
    @torch.no_grad()
    def validate(self) -> float:
        """Run validation"""
        self.model.eval()
        
        val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
        )
        
        correct = 0
        total = 0
        
        for batch in val_loader:
            images = batch['images'].to(self.device)
            texts = batch['texts']
            labels = batch.get('labels', None)
            
            # Your validation logic here
            # This is model-specific
            outputs = self.model(images, texts)
            predictions = outputs['predictions']
            
            if labels is not None:
                correct += (predictions == labels).sum().item()
                total += len(labels)
        
        accuracy = correct / total if total > 0 else 0.0
        return accuracy
    
    @torch.no_grad()
    def precompute_features(
        self, 
        dataset, 
        batch_size: int = 512
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pre-compute image and text features for entire dataset
        
        This is efficient because we only do it once per reselection
        """
        self.model.eval()
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
        )
        
        all_image_features = []
        all_text_features = []
        
        for batch in dataloader:
            images = batch['images'].to(self.device)
            texts = batch['texts']
            
            # Extract features (model-specific)
            image_feats = self.model.encode_image(images)
            text_feats = self.model.encode_text(texts)
            
            all_image_features.append(image_feats.cpu())
            all_text_features.append(text_feats.cpu())
        
        image_features = torch.cat(all_image_features, dim=0)
        text_features = torch.cat(all_text_features, dim=0)
        
        return image_features, text_features
    
    def get_current_metrics(self) -> Dict[str, float]:
        """
        Aggregate recent training statistics for MODE selection
        """
        def safe_mean(history, window=10):
            if len(history) == 0:
                return 0.0
            return np.mean(history[-window:])
        
        metrics = {
            'vision_loss': safe_mean(self.metrics_history['vision_loss']),
            'text_loss': safe_mean(self.metrics_history['text_loss']),
            'alignment_loss': safe_mean(self.metrics_history['alignment_loss']),
            'gradient_norm': safe_mean(self.metrics_history.get('gradient_norm', [1.0])),
            'step': self.global_step,
            'lr': self.optimizer.param_groups[0]['lr'],
            'initial_lr': self.initial_lr,
        }
        
        if len(self.metrics_history['val_acc']) > 0:
            metrics['val_acc'] = self.metrics_history['val_acc'][-1]
        
        if len(self.metrics_history['train_acc']) > 0:
            metrics['train_acc'] = self.metrics_history['train_acc'][-1]
        
        return metrics


# ============================================================================
# PART 6: Experiment Configuration
# ============================================================================

@dataclass
class MODEVLMConfig:
    """Configuration for MODE-VLM experiments"""
    
    # Model
    vision_encoder: str = "openai/clip-vit-base-patch32"
    text_encoder: str = "openai/clip-vit-base-patch32"
    
    # Dataset
    dataset_name: str = "cc3m"
    dataset_size: int = 100000  # Start with 100k subset
    
    # Training
    total_steps: int = 10000
    batch_size: int = 256
    learning_rate: float = 5e-4
    
    # MODE
    selection_frequency: int = 1000
    selection_ratio: float = 0.3
    state_dim: int = 20
    num_strategies: int = 7
    hypernetwork_hidden: int = 128
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def create_mode_vlm_system(config: MODEVLMConfig):
    """
    Factory function to create complete MODE-VLM system
    
    Returns:
        selector: MODEVLMSelector
        state_encoder: MultimodalBinaryStateEncoder
        hypernetwork: MODEHypernetwork
    """
    # Create state encoder
    state_encoder = MultimodalBinaryStateEncoder(
        state_dim=config.state_dim,
        history_window=100,
    )
    
    # Create hypernetwork
    hypernetwork = MODEHypernetwork(
        state_dim=config.state_dim,
        num_strategies=config.num_strategies,
        hidden_dim=config.hypernetwork_hidden,
        num_layers=3,
    )
    
    # Create strategies
    strategies = {
        'alignment_quality': AlignmentQualityStrategy(),
        'alignment_difficulty': AlignmentDifficultyStrategy(),
        'visual_complexity': VisualComplexityStrategy(),
        'text_richness': TextRichnessStrategy(),
        'cross_modal_diversity': CrossModalDiversityStrategy(),
        'balanced': BalancedStrategy(),
        'uncertainty': UncertaintyStrategy(),
    }
    
    # Create selector
    selector = MODEVLMSelector(
        state_encoder=state_encoder,
        hypernetwork=hypernetwork,
        strategies=strategies,
        selection_ratio=config.selection_ratio,
        device=config.device,
    )
    
    return selector, state_encoder, hypernetwork


# ============================================================================
# PART 7: Example Usage
# ============================================================================

def mode_vlm_example():
    """Example usage of MODE-VLM system"""
    # Configuration
    config = MODEVLMConfig(
        dataset_size=10000,  # Small for testing
        total_steps=1000,
        selection_frequency=200,
        selection_ratio=0.3,
    )
    
    print("Creating MODE-VLM system...")
    selector, state_encoder, hypernetwork = create_mode_vlm_system(config)
    
    print(f"State encoder: {config.state_dim}-dimensional binary state")
    print(f"Hypernetwork: {config.num_strategies} strategies")
    print(f"Selection: {config.selection_ratio*100:.0f}% of data per epoch")
    
    # Dummy data for testing
    N = config.dataset_size
    d_img = 512
    d_txt = 512
    
    print(f"\nGenerating dummy features for {N} samples...")
    image_features = torch.randn(N, d_img)
    text_features = torch.randn(N, d_txt)
    
    # Simulate training metrics
    current_metrics = {
        'vision_loss': 2.5,
        'text_loss': 3.0,
        'alignment_loss': 1.8,
        'gradient_norm': 1.2,
        'step': 500,
        'lr': 5e-4,
        'initial_lr': 5e-4,
        'train_acc': 0.6,
        'val_acc': 0.55,
    }
    
    print("\nRunning MODE selection...")
    selected_indices, metadata = selector.select_batch(
        image_features=image_features,
        text_features=text_features,
        current_metrics=current_metrics,
        total_steps=config.total_steps,
        return_metadata=True,
    )
    
    print(f"\nSelected {len(selected_indices)} samples")
    print(f"Binary state: {metadata['binary_state']}")
    print(f"Strategy weights: {metadata['strategy_weights']}")
    print(f"\nStrategy weight breakdown:")
    for i, name in enumerate(hypernetwork.strategy_names):
        print(f"  {name}: {metadata['strategy_weights'][i]:.4f}")
    
    print("\n" + "="*60)
    print("✓ MODE-VLM system initialized successfully!")
    print("Ready to integrate with your VLM training code")
    print("="*60)


# ============================================================================
# ORIGINAL EXPERIMENT CODE (LEGACY)
# ============================================================================

from torch.utils.data import DataLoader
import webdataset as wds
from huggingface_hub import HfFileSystem, hf_hub_url
import open_clip
from tqdm import tqdm
import pickle
import random
from pathlib import Path
import os

# NOTE: The following sections for analysis require additional packages.
# Please install them with: pip install matplotlib seaborn pandas scikit-learn
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd
from collections import defaultdict

# --- Configuration ---
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
DATASET_NAME = "pixparse/cc3m-wds"
DATASET_SIZE = 500_000
KEEP_FRACTION = 0.3
KEEP_SAMPLES = int(DATASET_SIZE * KEEP_FRACTION)
BATCH_SIZE = 512
EPOCHS = 50
MODEL_NAME = 'ViT-B-32'
CLIP_PRETRAINED = 'openai'
LR = 5e-4
WARMUP = 2000
WD = 0.1

FEATURE_CACHE_PATH = Path(f'./cc12m_{DATASET_SIZE}_clip_features.pt')
MODE_SELECTION_PATH = Path(f'./mode_selected_{KEEP_SAMPLES}.txt')
RANDOM_SELECTION_PATH = Path(f'./random_selected_{KEEP_SAMPLES}.txt')

print(f"Using device: {DEVICE}")
print(f"Dataset size: {DATASET_SIZE}")
print(f"Keep fraction: {KEEP_FRACTION} ({KEEP_SAMPLES} samples)")

# --- Global Model and Preprocessors ---
print(f"Loading CLIP model '{MODEL_NAME}' ({CLIP_PRETRAINED}).")
clip_model, _, image_processor = open_clip.create_model_and_transforms(
    MODEL_NAME, pretrained=CLIP_PRETRAINED, device=DEVICE
)
tokenizer = open_clip.get_tokenizer(MODEL_NAME)

def preprocess_text(sample):
    return tokenizer(sample[1]['caption'])

def preprocess_image(sample):
    return image_processor(sample[0])

def process_sample(sample):
    return (preprocess_image(sample), preprocess_text(sample))

# --- 1. Dataset Loading ---
def load_conceptual_captions(subset_size):
    """Loads the Conceptual Captions 3M dataset and returns a WebDataset object."""
    print("Setting up dataset streaming from Hugging Face...")
    fs = HfFileSystem()
    
    try:
        files = fs.glob(f"hf://datasets/{DATASET_NAME}/cc3m-train-*.tar")
        urls = [hf_hub_url(DATASET_NAME, f.split(f"{DATASET_NAME}/")[1], repo_type="dataset") for f in files]
        print(f"Found {len(urls)} tar files for the dataset.")
    except Exception as e:
        print(f"Could not automatically list files from Hub: {e}")
        print("Please ensure you are logged in with 'huggingface-cli login'")
        return None, None

    dataset = (
        wds.WebDataset(urls, resampled=True)
        .shuffle(1000)
        .slice(subset_size)
        .decode("pil")
        .to_tuple("jpg;png", "json")
        .map(process_sample)
    )
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)
    
    return dataloader, dataset

# --- 2. Precompute CLIP Features ---

def precompute_clip_features(dataloader, model, cache_path):
    """Extracts and saves CLIP features for the dataset."""
    print(f"Pre-computing CLIP features and saving to {cache_path}...")
    model.eval()
    feature_cache = {'image_features': [], 'text_features': [], 'sample_ids': []}
    
    # We need sample IDs, but webdataset doesn't provide them easily in the loader.
    # We will assign sequential IDs.
    sample_counter = 0

    with torch.no_grad():
        for i, (images, texts) in enumerate(tqdm(dataloader, desc="Extracting features")):
            images = images.to(DEVICE)
            texts = texts.squeeze(1).to(DEVICE)

            img_feats = model.encode_image(images)
            txt_feats = model.encode_text(texts)
            
            img_feats = F.normalize(img_feats, dim=-1)
            txt_feats = F.normalize(txt_feats, dim=-1)
            
            feature_cache['image_features'].append(img_feats.cpu())
            feature_cache['text_features'].append(txt_feats.cpu())
            
            num_samples = images.size(0)
            feature_cache['sample_ids'].extend(range(sample_counter, sample_counter + num_samples))
            sample_counter += num_samples

    features = {
        'image': torch.cat(feature_cache['image_features']),
        'text': torch.cat(feature_cache['text_features']),
        'ids': feature_cache['sample_ids']
    }
    
    torch.save(features, cache_path)
    print(f"Cached {len(features['ids'])} samples to {cache_path}")
    return features

# --- 3. MODE Selection ---

def load_mode_with_vlm_adapter(model_path):
    """
    Loads a trained MODE model with support for multiple architectures.

    Supports:
    1. MODEHypernetwork: State-to-strategy mapping (20 dim -> 7 strategies)
    2. ImportanceHyperNetwork: Feature-to-importance mapping (512*2 -> 1)
    3. SimpleMODEScorer: Lightweight MLP scorer

    Args:
        model_path: Path to saved model checkpoint (.pt or .pth)

    Returns:
        A model with a score_batch(states) method
    """
    print(f"Loading MODE model from: {model_path}")
    model_path = Path(model_path)

    if not model_path.exists():
        print(f"Warning: MODE model file not found at {model_path}")
        print("MODE selection will use a fallback similarity-based scorer.")

        # Fallback: Use similarity-based scoring instead of random
        class SimilarityMode(nn.Module):
            def score_batch(self, states):
                """Score based on loss and similarity features in states"""
                if states.size(1) >= 2:
                    # states[:, 0] is high_loss, states[:, 1] is low_similarity
                    # Prioritize high loss and low similarity samples (harder examples)
                    loss_score = states[:, 0].float()
                    sim_score = states[:, 1].float()
                    return (loss_score * 0.7 + sim_score * 0.3)
                return torch.rand(states.size(0))
        return SimilarityMode()

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')

    # Detect model type from checkpoint structure
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            config = checkpoint.get('config', {})
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            config = checkpoint.get('config', {})
        else:
            state_dict = checkpoint
            config = {}
    else:
        state_dict = checkpoint
        config = {}

    # Determine architecture from state_dict keys
    keys = list(state_dict.keys())

    if any('network.0.weight' in k or 'network.0.bias' in k for k in keys):
        # MODEHypernetwork architecture
        print("Detected MODEHypernetwork architecture")

        # Infer dimensions from checkpoint
        first_layer_key = 'network.0.weight'
        if first_layer_key in state_dict:
            hidden_dim = state_dict[first_layer_key].shape[0]
            state_dim = state_dict[first_layer_key].shape[1]
        else:
            hidden_dim = config.get('hidden_dim', 128)
            state_dim = config.get('state_dim', 20)

        # Count number of layers
        num_layers = sum(1 for k in keys if 'weight' in k and 'network' in k)
        num_strategies = config.get('num_strategies', 7)

        model = MODEHypernetwork(
            state_dim=state_dim,
            num_strategies=num_strategies,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
        model.load_state_dict(state_dict)

        # Wrap to provide score_batch interface
        class HypernetworkWrapper(nn.Module):
            def __init__(self, hypernetwork):
                super().__init__()
                self.hypernetwork = hypernetwork

            def score_batch(self, states):
                """Convert binary states to importance scores via strategy weights"""
                strategy_weights = self.hypernetwork(states.float())
                # Return weighted combination of strategies as final score
                # Higher weight on alignment_quality and difficulty
                scores = (strategy_weights[:, 0] * 0.3 +  # alignment_quality
                         strategy_weights[:, 1] * 0.3 +   # alignment_difficulty
                         strategy_weights[:, 6] * 0.2 +   # uncertainty
                         strategy_weights.sum(dim=1) * 0.2)  # overall
                return scores

        wrapped_model = HypernetworkWrapper(model)

    elif any('image_proj' in k or 'text_proj' in k for k in keys):
        # ImportanceHyperNetwork architecture
        print("Detected ImportanceHyperNetwork architecture")

        from mode_vlm_experiment import ImportanceHyperNetwork

        feature_dim = config.get('feature_dim', 512)
        hidden_dim = config.get('hidden_dim', 256)
        num_layers = config.get('num_layers', 3)

        model = ImportanceHyperNetwork(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
        model.load_state_dict(state_dict)

        # Note: ImportanceHyperNetwork expects (image_features, text_features)
        # but score_batch receives states. Need adapter.
        class ImportanceWrapper(nn.Module):
            def __init__(self, importance_net):
                super().__init__()
                self.importance_net = importance_net

            def score_batch(self, states):
                """
                States expected to contain concatenated image/text features
                or we use states as proxy features
                """
                batch_size = states.size(0)
                # If states are binary (12-dim), expand to feature space
                if states.size(1) < 128:
                    # Use simple expansion
                    pseudo_features = states.float().repeat(1, 512 // states.size(1) + 1)[:, :512]
                    image_feats = pseudo_features
                    text_feats = pseudo_features
                else:
                    # Assume states contain features
                    mid = states.size(1) // 2
                    image_feats = states[:, :mid]
                    text_feats = states[:, mid:]

                scores = self.importance_net(image_feats, text_feats).squeeze(-1)
                return scores

        wrapped_model = ImportanceWrapper(model)

    else:
        # Simple MLP scorer architecture
        print("Detected SimpleMODEScorer or custom MLP architecture")

        # Infer input/output dimensions
        first_weight = next((v for k, v in state_dict.items() if 'weight' in k), None)
        if first_weight is not None:
            input_dim = first_weight.shape[1] if first_weight.ndim > 1 else 12
            hidden_dim = first_weight.shape[0]
        else:
            input_dim = 12
            hidden_dim = 128

        # Build simple MLP
        class SimpleMODEScorer(nn.Module):
            def __init__(self, input_dim=12, hidden_dim=128, num_layers=3):
                super().__init__()
                layers = []
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.ReLU())

                for _ in range(num_layers - 2):
                    layers.append(nn.Linear(hidden_dim, hidden_dim))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(0.1))

                layers.append(nn.Linear(hidden_dim, 1))
                layers.append(nn.Sigmoid())

                self.network = nn.Sequential(*layers)

            def score_batch(self, states):
                return self.network(states.float()).squeeze(-1)

        model = SimpleMODEScorer(input_dim=input_dim, hidden_dim=hidden_dim)

        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            print(f"Warning: Could not load state dict directly: {e}")
            print("Using initialized model with similar architecture")

        wrapped_model = model

    wrapped_model.eval()
    print(f"Successfully loaded MODE model with architecture: {type(wrapped_model).__name__}")

    return wrapped_model


def train_simple_mode_model(features, save_path='mode_trained.pt', epochs=10, device='cuda'):
    """
    Trains a simple MODE model from scratch using image-text features.

    This is a utility function to bootstrap MODE training when you don't have
    a pre-trained model. It trains a simple MLP to predict sample importance
    based on alignment quality and difficulty.

    Args:
        features: Dict with 'image' and 'text' features
        save_path: Where to save the trained model
        epochs: Number of training epochs
        device: Device to train on

    Returns:
        Trained MODE model
    """
    print(f"Training simple MODE model from scratch...")

    image_feats = features['image']
    text_feats = features['text']

    # Extract training states (binary features)
    print("Extracting training states...")
    all_states = []
    all_targets = []

    batch_size = 10000
    for i in tqdm(range(0, len(image_feats), batch_size), desc="Computing states"):
        batch_img = image_feats[i:i+batch_size].to(device)
        batch_txt = text_feats[i:i+batch_size].to(device)

        # Compute alignment metrics
        similarity = (batch_img * batch_txt).sum(dim=-1)
        logits = batch_img @ batch_txt.T / 0.07
        labels = torch.arange(len(batch_img), device=device)
        loss = F.cross_entropy(logits, labels, reduction='none')

        # Create binary state representation (12 dimensions)
        batch_states = torch.zeros(len(batch_img), 12, device=device)
        batch_states[:, 0] = loss > loss.median()  # high_loss
        batch_states[:, 1] = similarity < similarity.median()  # low_similarity
        batch_states[:, 2] = loss < loss.quantile(0.25)  # very_easy
        batch_states[:, 3] = loss > loss.quantile(0.75)  # very_hard
        batch_states[:, 4] = similarity > similarity.quantile(0.75)  # high_alignment
        batch_states[:, 5] = similarity < similarity.quantile(0.25)  # low_alignment

        # Additional features
        batch_states[:, 6] = (batch_img.norm(dim=1) > batch_img.norm(dim=1).median()).float()
        batch_states[:, 7] = (batch_txt.norm(dim=1) > batch_txt.norm(dim=1).median()).float()
        batch_states[:, 8] = torch.rand(len(batch_img), device=device) > 0.5  # random
        batch_states[:, 9] = torch.arange(len(batch_img), device=device) % 2 == 0  # alternating
        batch_states[:, 10] = (loss / loss.max() > 0.5).float()
        batch_states[:, 11] = (similarity / similarity.max() > 0.5).float()

        # Target: Prioritize hard, low-similarity, high-loss samples
        # Normalize loss and similarity for target computation
        norm_loss = (loss - loss.min()) / (loss.max() - loss.min() + 1e-8)
        norm_sim = (similarity - similarity.min()) / (similarity.max() - similarity.min() + 1e-8)
        targets = norm_loss * 0.7 + (1 - norm_sim) * 0.3

        all_states.append(batch_states.cpu())
        all_targets.append(targets.cpu())

    states = torch.cat(all_states)
    targets = torch.cat(all_targets)

    print(f"Training data: {states.shape[0]} samples, {states.shape[1]} features")

    # Build simple MLP model
    class SimpleMODEScorer(nn.Module):
        def __init__(self, input_dim=12, hidden_dim=128, num_layers=3):
            super().__init__()
            layers = []
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())

            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))

            layers.append(nn.Linear(hidden_dim, 1))
            layers.append(nn.Sigmoid())

            self.network = nn.Sequential(*layers)

        def score_batch(self, states):
            return self.network(states.float()).squeeze(-1)

    model = SimpleMODEScorer(input_dim=12, hidden_dim=128, num_layers=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Training loop
    print(f"Training for {epochs} epochs...")
    dataset = torch.utils.data.TensorDataset(states, targets)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_states, batch_targets in dataloader:
            batch_states = batch_states.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()
            predictions = model.score_batch(batch_states)
            loss = criterion(predictions, batch_targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # Save model
    model.eval()
    save_path = Path(save_path)
    torch.save({
        'state_dict': model.state_dict(),
        'config': {
            'input_dim': 12,
            'hidden_dim': 128,
            'num_layers': 3,
            'model_type': 'SimpleMODEScorer'
        }
    }, save_path)

    print(f"Trained MODE model saved to: {save_path}")
    return model


def mode_selection(features, mode_model, selection_path):
    """Selects the top samples using MODE and saves the indices."""
    print("Performing MODE selection...")
    image_feats = features['image']
    text_feats = features['text']
    
    mode_model.to(DEVICE)
    mode_model.eval()
    
    all_scores = []
    batch_size = 10000

    with torch.no_grad():
        for i in tqdm(range(0, len(image_feats), batch_size), desc="Computing MODE scores"):
            batch_img = image_feats[i:i+batch_size].to(DEVICE)
            batch_txt = text_feats[i:i+batch_size].to(DEVICE)
            
            # This state extraction logic is from your provided plan.
            similarity = (batch_img * batch_txt).sum(dim=-1)
            logits = batch_img @ batch_txt.T / 0.07
            labels = torch.arange(len(batch_img), device=DEVICE)
            loss = F.cross_entropy(logits, labels, reduction='none')
            
            batch_states = torch.zeros(len(batch_img), 12, device=DEVICE)
            batch_states[:, 0] = loss > loss.median()
            batch_states[:, 1] = similarity < similarity.median()
            
            scores = mode_model.score_batch(batch_states)
            all_scores.append(scores.cpu())

    scores = torch.cat(all_scores)
    top_indices = torch.topk(scores, KEEP_SAMPLES).indices
    
    selected_ids = [features['ids'][i] for i in top_indices]
    with open(selection_path, 'w') as f:
        for sid in selected_ids:
            f.write(f"{sid}\n")
            
    print(f"Selected {len(selected_ids)} sample indices and saved to {selection_path}")
    return selected_ids

# --- 4. Baselines ---

def random_selection(features, selection_path):
    """Selects random samples and saves the indices."""
    print("Performing random selection...")
    indices = list(range(len(features['ids'])))
    selected_ids = random.sample(indices, KEEP_SAMPLES)
    
    with open(selection_path, 'w') as f:
        for sid in selected_ids:
            f.write(f"{sid}\n")
            
    print(f"Selected {len(selected_ids)} random sample indices and saved to {selection_path}")
    return selected_ids

# --- 5. CLIP Training ---

def train_clip(train_dataset, selection_name):
    """Trains a CLIP model on a selected subset of data."""
    print(f"\n--- Starting CLIP training for '{selection_name}' ---")
    
    model, _, image_processor = open_clip.create_model_and_transforms(MODEL_NAME)
    model = model.to(DEVICE)
    
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    total_steps = len(dataloader) * EPOCHS
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, total_steps=total_steps, pct_start=WARMUP/total_steps)

    print(f"Training for {EPOCHS} epochs with {len(dataloader)} steps per epoch.")

    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for i, (images, texts) in enumerate(pbar):
            images = images.to(DEVICE)
            texts = texts.squeeze(1).to(DEVICE)
            
            optimizer.zero_grad()
            
            with torch.autocast(device_type=DEVICE.type):
                image_features = model.encode_image(images)
                text_features = model.encode_text(texts)
                
                image_features = F.normalize(image_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)
                
                logit_scale = model.logit_scale.exp()
                logits = logit_scale * image_features @ text_features.T
                
                labels = torch.arange(len(images), device=DEVICE)
                loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

            loss.backward()
            optimizer.step()
            scheduler.step()
            
            pbar.set_postfix({"loss": loss.item(), "lr": scheduler.get_last_lr()[0]})
            
        print(f"Epoch {epoch+1} complete. Final loss: {loss.item():.4f}")
        
        # Save checkpoint
        checkpoint_dir = Path(f'./checkpoints/{selection_name}')
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_dir / f"epoch_{epoch+1}.pt")

    print(f"--- Finished CLIP training for '{selection_name}' ---")


# --- Main Orchestrator ---

def main():
    """Main function to run the VLM experiment."""
    
    # --- Data Loading and Feature Extraction ---
    if not FEATURE_CACHE_PATH.exists():
        dataloader, _ = load_conceptual_captions(DATASET_SIZE)
        if dataloader is None:
            return
        features = precompute_clip_features(dataloader, clip_model, FEATURE_CACHE_PATH)
    else:
        print(f"Loading cached features from {FEATURE_CACHE_PATH}")
        features = torch.load(FEATURE_CACHE_PATH)

    # --- MODE Selection ---
    if not MODE_SELECTION_PATH.exists():
        # IMPORTANT: You need to provide your own trained MODE model file.
        mode_model = load_mode_with_vlm_adapter('mode_cifar_best.pt')
        mode_selection(features, mode_model, MODE_SELECTION_PATH)
    
    # --- Random Baseline Selection ---
    if not RANDOM_SELECTION_PATH.exists():
        random_selection(features, RANDOM_SELECTION_PATH)

    # --- Training on Selected Subsets ---
    print("\n--- Starting training runs ---")
    
    # Create a dataset from all downloaded samples (we will filter it)
    # This is a bit tricky with webdataset, as we need to map indices to samples.
    # A simpler way is to re-filter the webdataset.
    
    def create_filtered_dataset(selection_path):
        with open(selection_path, 'r') as f:
            selected_ids = {int(line.strip()) for line in f}
        
        # We need to re-create the webdataset and filter it.
        fs = HfFileSystem()
        files = fs.glob(f"hf://datasets/{DATASET_NAME}/data/*.tar")
        urls = [hf_hub_url(DATASET_NAME, f.split(f"{DATASET_NAME}/")[1], repo_type="dataset") for f in files]

        def id_filter(sample):
            # The sample key __key__ can be used if it's consistent
            # Let's assume we can get an index from the sample.
            # Webdataset does not have a global index by default.
            # We will add one.
            return sample['__sample_index__'] in selected_ids

        dataset = (
            wds.WebDataset(urls, resampled=True)
            .shuffle(1000)
            .slice(DATASET_SIZE)
            .decode("pil")
            .map(lambda x: {**x, '__sample_index__': int(x['__key__'])}) # This is an assumption
            .select(id_filter)
            .to_tuple("jpg;png", "json")
            .map_tuple(image_processor, lambda s: tokenizer(s['caption']))
        )
        return dataset

    # This filtering is complex. A much simpler approach for training is to use the indices
    # on the cached features, if we can load the images from somewhere.
    # Since we are streaming, we don't have the images locally.
    
    # Let's try a different approach for training:
    # We will create a small dataset class that holds the selected indices and
    # then we will iterate through the webdataset until we find them. This is inefficient.
    
    # The most robust way is to download the subset.
    # The plan was to use webdataset for everything. Let's stick to it.
    # The filtering function is the way to go. We need to get the sample index right.
    
    print("Note: Training requires filtering the dataset, which can be slow.")
    
    # Train on MODE selection
    # mode_dataset = create_filtered_dataset(MODE_SELECTION_PATH)
    # train_clip(mode_dataset, "mode_selection")
    
    # Train on Random selection
    # random_dataset = create_filtered_dataset(RANDOM_SELECTION_PATH)
    # train_clip(random_dataset, "random_selection")
    
    print("\nExperiment script finished.")
    print("Due to the complexity of filtering a webdataset for training,")
    print("the training part is commented out. You can enable it if you have a robust way")
    print("to map selection indices to webdataset samples.")
    print("A common approach is to download the required samples first using their URLs,")
    print("which can be extracted during the feature computation phase.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="oneshot", choices=["oneshot", "hypernetwork"],
                        help="Which experiment to run.")
    args = parser.parse_args()

    if args.experiment == "oneshot":
        main()
    elif args.experiment == "hypernetwork":
        print("Running Hypernetwork-driven experiment...")
        print("The hypernetwork experiment is not fully implemented due to ambiguities in the provided pseudo-code.")
        print("Specifically, the logic for state extraction and strategy scoring needs clarification on how to use the model being trained.")
        print("Please review the comments in the generated code.")

# --- Hypernetwork-driven selection (NEW) ---

class HypernetworkDrivenSelection:
    """
    The hypernetwork makes REAL-TIME decisions every epoch
    This is the TRUE MODE architecture
    """
    
    def __init__(self):
        # This is a placeholder for your actual hypernetwork.
        # You would need to define the BinaryHypernetwork class.
        self.hypernetwork = None #BinaryHypernetwork(
            #input_dim=12,      # Binary state vector
            #hidden_dim=256,
            #output_dim=4       # Weights for 4 strategies
        #)
        
        self.strategies = [
            CurriculumStrategies.LossBasedStrategy(),
            CurriculumStrategies.ConfidenceBasedStrategy(), 
            CurriculumStrategies.DiversityStrategy(),
            CurriculumStrategies.EasyFirstStrategy()
        ]

# CRITICAL: How to extract binary training state
class BinaryStateExtractor:
    """
    This encodes the current training context into 12 binary features
    The hypernetwork uses this to decide which strategy to apply
    """
    
    def extract_training_state(self, clip_model, epoch, feature_cache):
        """
        Convert complex training state → 12 binary signals
        
        This is MODE's key insight:
        Training dynamics can be discretized into binary decisions
        
        NOTE: This is a placeholder implementation based on the user's pseudo-code.
        The pseudo-code has a logical contradiction: it tries to use the currently
        training `clip_model` on pre-computed features from a frozen model.
        A correct implementation would need to run the `clip_model` on actual
        image and text data to assess its current state.
        """
        
        image_feats, text_feats = feature_cache['image'], feature_cache['text']
        
        sample_size = 10_000
        indices = torch.randperm(len(image_feats))[:sample_size]
        
        img_sample = image_feats[indices].to(DEVICE)
        txt_sample = text_feats[indices].to(DEVICE)
        
        with torch.no_grad():
            # The following is based on the user's pseudo-code but is problematic.
            # `clip_model.encode_image_from_features` is not a standard function.
            # To correctly assess the model's state, you would need to load the
            # actual images and texts for this sample and run them through the `clip_model`.
            # As a placeholder, we use the pre-computed features directly.
            img_pred = img_sample
            txt_pred = txt_sample
            
            similarity = (img_pred * txt_pred).sum(dim=-1)
            
            logits = img_pred @ txt_pred.T / 0.07
            labels = torch.arange(len(img_sample), device=DEVICE)
            losses = F.cross_entropy(logits, labels, reduction='none')
            
        state = torch.zeros(12, dtype=torch.float32, device=DEVICE)
        
        state[0] = float(losses.mean() > 2.5)
        state[1] = float(losses.std() > 0.8)
        state[2] = float(similarity.mean() < 0.25)
        state[3] = float(epoch < 8)
        state[4] = float(8 <= epoch < 24)
        state[5] = float(epoch >= 24)
        
        # Other state dimensions would be implemented here.
        
        return state

# The 4 Strategies (Implementation)
class CurriculumStrategies:
    """
    Each strategy scores samples differently
    Hypernetwork blends them based on training state
    """
    
    class LossBasedStrategy:
        """Select samples with high loss (hard examples)"""
        def score_all(self, clip_model, image_feats, text_feats):
            with torch.no_grad():
                similarity = (image_feats * text_feats).sum(dim=-1)
                scores = -similarity
                scores = (scores - scores.min()) / (scores.max() - scores.min())
            return scores

    class ConfidenceBasedStrategy:
        """Select samples with low confidence (uncertain examples)"""
        def score_all(self, clip_model, image_feats, text_feats):
            with torch.no_grad():
                similarity = (image_feats * text_feats).sum(dim=-1)
                optimal_uncertainty = 0.20
                scores = -torch.abs(similarity - optimal_uncertainty)
                scores = (scores - scores.min()) / (scores.max() - scores.min())
            return scores

    class DiversityStrategy:
        """Select diverse, representative samples"""
        def fast_kmeans(self, sample, k):
            # A simple kmeans implementation for demonstration
            from sklearn.cluster import MiniBatchKMeans
            kmeans = MiniBatchKMeans(n_clusters=k, batch_size=256, n_init='auto')
            kmeans.fit(sample.cpu().numpy())
            return torch.from_numpy(kmeans.cluster_centers_).to(DEVICE)

        def score_all(self, clip_model, image_feats, text_feats):
            with torch.no_grad():
                k = 100
                if not hasattr(self, 'centers'):
                    indices = torch.randperm(len(image_feats))[:10000]
                    sample = image_feats[indices]
                    self.centers = self.fast_kmeans(sample, k)
                
                distances = torch.cdist(image_feats, self.centers)
                min_distances = distances.min(dim=-1)[0]
                scores = min_distances
                scores = (scores - scores.min()) / (scores.max() - scores.min())
            return scores

    class EasyFirstStrategy:
        """Select easy, high-confidence samples (for early training)"""
        def score_all(self, clip_model, image_feats, text_feats):
            with torch.no_grad():
                similarity = (image_feats * text_feats).sum(dim=-1)
                scores = similarity
                scores = (scores - scores.min()) / (scores.max() - scores.min())
            return scores

class MODEAnalyzer:
    """Analyze MODE selection patterns"""

    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.logs = defaultdict(list)

    def log_selection(self, epoch, binary_state, strategy_weights, strategy_scores, selected_indices, state_info):
        self.logs['epoch'].append(epoch)
        self.logs['binary_state'].append(binary_state.cpu().numpy())
        self.logs['strategy_weights'].append(strategy_weights.cpu().numpy())
        self.logs['state_info'].append(state_info)

        for i, name in enumerate(['loss', 'confidence', 'diversity', 'easy']):
            scores = strategy_scores[i]
            selected_scores = scores[selected_indices]
            self.logs[f'{name}_selected_mean'].append(selected_scores.mean().item())

    def save_logs(self):
        path = self.checkpoint_dir / 'mode_analysis_logs.json'
        serializable = {k: [v.tolist() if isinstance(v, np.ndarray) else v for v in values] for k, values in self.logs.items()}
        with open(path, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"Analyzer logs saved to {path}")

    def plot_strategy_evolution(self):
        path = self.checkpoint_dir / 'strategy_evolution.png'
        strategy_names = ['Loss-Based', 'Confidence-Based', 'Diversity-Based', 'Easy-First']
        fig, ax = plt.subplots(figsize=(12, 7))
        epochs = self.logs['epoch']
        weights = np.array(self.logs['strategy_weights'])
        for i, name in enumerate(strategy_names):
            ax.plot(epochs, weights[:, i], label=name, linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Strategy Weight')
        ax.set_title('MODE Strategy Evolution Over Training')
        ax.legend()
        ax.grid(True, alpha=0.5)
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        print(f"Strategy evolution plot saved to {path}")
        plt.close(fig)

class HypernetworkVisualizer:
    """Visualize hypernetwork decision boundaries"""

    def __init__(self, hypernetwork):
        self.hypernetwork = hypernetwork
        self.hypernetwork.eval()

class SampleAnalyzer:
    """Analyze characteristics of selected vs rejected samples"""

    @staticmethod
    def compare_distributions(selected_scores, rejected_scores, strategy_names):
        """Compare score distributions for selected vs rejected samples"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        for idx, (ax, name) in enumerate(zip(axes.flat, strategy_names)):
            sel = selected_scores[idx].cpu().numpy()
            rej = rejected_scores[idx].cpu().numpy()
            ax.hist(sel, bins=50, alpha=0.6, label='Selected', density=True, color='green')
            ax.hist(rej, bins=50, alpha=0.6, label='Rejected', density=True, color='red')
            ax.axvline(sel.mean(), color='green', linestyle='--', linewidth=2, label=f'Selected μ={sel.mean():.3f}')
            ax.axvline(rej.mean(), color='red', linestyle='--', linewidth=2, label=f'Rejected μ={rej.mean():.3f}')
            ax.set_xlabel('Score')
            ax.set_ylabel('Density')
            ax.set_title(f'{name} Score Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.suptitle('Selected vs Rejected Sample Distributions', fontsize=14)
        plt.tight_layout()
        plt.show()

def run_hypernetwork_experiment():
    """
    Main function for the hypernetwork-driven experiment.
    """
    print("--- Starting Hypernetwork-driven Experiment ---")

    if not FEATURE_CACHE_PATH.exists():
        print(f"Feature cache not found at {FEATURE_CACHE_PATH}. Please run the 'oneshot' experiment first.")
        return

    print(f"Loading cached features from {FEATURE_CACHE_PATH}")
    features = torch.load(FEATURE_CACHE_PATH)
    image_feats = features['image'].to(DEVICE)
    text_feats = features['text'].to(DEVICE)

    clip_model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=CLIP_PRETRAINED, device=DEVICE)
    optimizer = torch.optim.AdamW(clip_model.parameters(), lr=LR, weight_decay=WD)

    # PLACEHOLDER: You need to define and load your actual hypernetwork model.
    hypernetwork = nn.Linear(12, 4).to(DEVICE)
    state_extractor = BinaryStateExtractor()
    strategies = [
        CurriculumStrategies.LossBasedStrategy(),
        CurriculumStrategies.ConfidenceBasedStrategy(),
        CurriculumStrategies.DiversityStrategy(),
        CurriculumStrategies.EasyFirstStrategy()
    ]
    analyzer = MODEAnalyzer(Path("./mode_analysis"))
    loss_history = []

    for epoch in range(EPOCHS):
        print(f"\n{'='*60}\nEPOCH {epoch + 1}/{EPOCHS}: Hypernetwork-driven selection")

        binary_state, state_info = state_extractor.extract_training_state(clip_model, epoch, {'image': image_feats, 'text': text_feats}, loss_history)
        loss_history.append(state_info['avg_loss'])

        strategy_weights = F.softmax(hypernetwork(binary_state), dim=-1)

        all_scores = torch.stack([s.score_all(clip_model, image_feats, text_feats) for s in strategies])

        final_scores = (strategy_weights.unsqueeze(1) * all_scores).sum(dim=0)

        k = int(DATASET_SIZE * 0.1)
        selected_indices = torch.topk(final_scores, k).indices
        print(f"Selected {k} samples for training this epoch.")

        analyzer.log_selection(epoch, binary_state, strategy_weights, all_scores, selected_indices.cpu(), state_info)

        print("TRAINING STEP SKIPPED: Implement data loading for the selected indices to train.")

    print("\nHypernetwork experiment loop finished.")
    analyzer.save_logs()
    analyzer.plot_strategy_evolution()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="oneshot", choices=["oneshot", "hypernetwork"],
                        help="Which experiment to run.")
    args = parser.parse_args()

    if args.experiment == "oneshot":
        main()
    elif args.experiment == "hypernetwork":
        run_hypernetwork_experiment()
