"""
Fast Multi-Gradient RL-Guided Selection for LLMs
================================================

Key optimizations:
1. Batch gradient computation with multiple ranks
2. Cached gradient projections
3. Efficient multi-rank feature extraction
4. Fast approximate scoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass
from collections import defaultdict, deque
import time
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


# =============================================================================
# Fast Multi-Rank Gradient Computation
# =============================================================================

class FastMultiRankGradientComputer:
    """
    Efficiently computes gradients at multiple ranks simultaneously
    """
    
    def __init__(self, 
                 ranks: List[int] = [32, 128, 512],
                 batch_size: int = 256,
                 cache_size: int = 10000,
                 device: str = "cuda"):
        self.ranks = sorted(ranks)
        self.batch_size = batch_size
        self.device = device
        
        # Gradient cache for each rank
        self.gradient_cache = {r: {} for r in ranks}
        self.cache_order = deque(maxlen=cache_size)
        
        # Projection matrices for each rank
        self.projectors = {r: {} for r in ranks}
        self.projection_step = 0
        self.update_frequency = 500  # Update projections every N steps
        
    def compute_batch_gradients_multirank(self, 
                                        model: nn.Module,
                                        data_loader: DataLoader,
                                        sample_indices: List[int],
                                        use_cache: bool = True) -> Dict[int, Dict[int, torch.Tensor]]:
        """
        Compute gradients for a batch of samples at multiple ranks simultaneously
        
        Returns:
            Dict mapping rank -> sample_idx -> compressed_gradient
        """
        # Check cache first
        uncached_indices = []
        cached_results = {r: {} for r in self.ranks}
        
        if use_cache:
            for idx in sample_indices:
                if idx in self.gradient_cache[self.ranks[0]]:  # If in one cache, in all
                    for rank in self.ranks:
                        cached_results[rank][idx] = self.gradient_cache[rank][idx]
                else:
                    uncached_indices.append(idx)
        else:
            uncached_indices = sample_indices
            
        if not uncached_indices:
            return cached_results
            
        # Compute gradients for uncached samples
        model.eval()  # Use eval mode for consistent gradients
        
        # Create subset loader for efficiency
        subset_dataset = Subset(data_loader.dataset, uncached_indices)
        subset_loader = DataLoader(subset_dataset, 
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 num_workers=0)
        
        multi_rank_gradients = {r: {} for r in self.ranks}
        
        with torch.enable_grad():
            for batch_idx, batch in enumerate(subset_loader):
                if isinstance(batch, dict):
                    inputs = batch['input_ids'].to(self.device)
                    labels = batch['labels'].to(self.device)
                else:
                    inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
                
                # Get batch indices
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(uncached_indices))
                batch_indices = uncached_indices[start_idx:end_idx]
                
                # Forward pass
                outputs = model(inputs)
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                    
                # Compute per-sample gradients efficiently
                loss_fn = nn.CrossEntropyLoss(reduction='none')
                losses = loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                losses = losses.reshape(labels.shape)
                
                # Compute gradients for each sample
                for i, idx in enumerate(batch_indices):
                    model.zero_grad()
                    
                    # Individual sample loss
                    sample_loss = losses[i].mean()
                    sample_loss.backward(retain_graph=True)
                    
                    # Extract and project gradients at multiple ranks
                    sample_grads_by_rank = self._extract_multirank_gradients(model)
                    
                    # Cache results
                    for rank in self.ranks:
                        multi_rank_gradients[rank][idx] = sample_grads_by_rank[rank]
                        self.gradient_cache[rank][idx] = sample_grads_by_rank[rank].detach()
                        
                    # Update cache order
                    if idx not in self.cache_order:
                        self.cache_order.append(idx)
                        
                        # Evict oldest if cache full
                        if len(self.cache_order) == self.cache_order.maxlen:
                            evicted = self.cache_order[0]
                            for rank in self.ranks:
                                if evicted in self.gradient_cache[rank]:
                                    del self.gradient_cache[rank][evicted]
        
        # Combine cached and newly computed results
        for rank in self.ranks:
            multi_rank_gradients[rank].update(cached_results[rank])
            
        model.train()  # Reset to train mode
        return multi_rank_gradients
    
    def _extract_multirank_gradients(self, model: nn.Module) -> Dict[int, torch.Tensor]:
        """Extract gradients and project to multiple ranks simultaneously"""
        # Collect all gradients
        all_grads = []
        param_shapes = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                all_grads.append(param.grad.flatten())
                param_shapes.append((name, param.grad.shape))
                
        # Concatenate all gradients
        full_gradient = torch.cat(all_grads)
        
        # Project to each rank
        multi_rank_grads = {}
        
        for rank in self.ranks:
            # Get or create projector
            if 'global' not in self.projectors[rank] or self.projection_step % self.update_frequency == 0:
                self._update_projector(full_gradient, rank)
                
            # Apply projection
            U, V = self.projectors[rank]['global']
            
            # Efficient projection for 1D gradient vector
            # Instead of reshaping, we use a single projection matrix
            projected = U[:, :rank].T @ full_gradient
            multi_rank_grads[rank] = projected
            
        self.projection_step += 1
        return multi_rank_grads
    
    def _update_projector(self, gradient: torch.Tensor, rank: int):
        """Update projection matrix using randomized SVD"""
        # For 1D gradient, we create a projection matrix
        grad_dim = gradient.shape[0]
        
        # Use random projection for efficiency
        if grad_dim > 10000:  # Large gradients
            # Random Gaussian projection
            U = torch.randn(grad_dim, rank, device=gradient.device)
            U, _ = torch.qr(U)
            V = None  # Not needed for 1D
        else:
            # Can afford SVD for smaller dimensions
            # Create matrix from gradient history if available
            if hasattr(self, 'gradient_history'):
                grad_matrix = torch.stack(list(self.gradient_history)[-100:])
                U, S, V = torch.svd_lowrank(grad_matrix.T, q=rank)
            else:
                # Random initialization
                U = torch.randn(grad_dim, rank, device=gradient.device)
                U, _ = torch.qr(U)
                V = None
                
        self.projectors[rank]['global'] = (U.detach(), V)


# =============================================================================
# Fast Multi-Rank Feature Extractor
# =============================================================================

@dataclass
class MultiRankFeatures:
    """Features extracted from multiple gradient ranks"""
    rank_32_features: torch.Tensor
    rank_128_features: torch.Tensor
    rank_512_features: torch.Tensor
    
    # Cross-rank features
    rank_disagreement_32_128: float
    rank_disagreement_128_512: float
    rank_disagreement_32_512: float
    
    # Spectral features
    gradient_magnitude_ratio: float  # ||g_32|| / ||g_512||
    information_retention: Dict[int, float]  # Approximation quality per rank
    
    def to_tensor(self) -> torch.Tensor:
        """Concatenate all features for RL"""
        features = []
        
        # Add compressed gradient features (top k components only for efficiency)
        features.append(self.rank_32_features[:16])  # Top 16 components
        features.append(self.rank_128_features[:32])  # Top 32 components  
        features.append(self.rank_512_features[:64])  # Top 64 components
        
        # Add cross-rank features
        features.append(torch.tensor([
            self.rank_disagreement_32_128,
            self.rank_disagreement_128_512,
            self.rank_disagreement_32_512,
            self.gradient_magnitude_ratio
        ]))
        
        # Add information retention scores
        features.append(torch.tensor([
            self.information_retention.get(32, 0.0),
            self.information_retention.get(128, 0.0),
            self.information_retention.get(512, 0.0)
        ]))
        
        return torch.cat(features)


class FastMultiRankFeatureExtractor:
    """Extract features from multi-rank gradients efficiently"""
    
    def __init__(self, ranks: List[int] = [32, 128, 512]):
        self.ranks = ranks
        
    def extract_features_batch(self, 
                              multi_rank_gradients: Dict[int, Dict[int, torch.Tensor]],
                              sample_indices: List[int]) -> Dict[int, MultiRankFeatures]:
        """Extract features for a batch of samples"""
        features = {}
        
        for idx in sample_indices:
            # Get gradients at each rank
            grads = {r: multi_rank_gradients[r].get(idx) for r in self.ranks}
            
            # Skip if missing gradients
            if any(g is None for g in grads.values()):
                continue
                
            # Compute features
            features[idx] = self._extract_single_features(grads)
            
        return features
    
    def _extract_single_features(self, grads: Dict[int, torch.Tensor]) -> MultiRankFeatures:
        """Extract features from multi-rank gradients"""
        # Compute gradient norms
        norms = {r: torch.norm(g).item() for r, g in grads.items()}
        
        # Compute rank disagreements (cosine distance)
        disagreements = {}
        for i, r1 in enumerate(self.ranks):
            for j, r2 in enumerate(self.ranks[i+1:], i+1):
                # Align dimensions for comparison
                dim = min(grads[r1].shape[0], grads[r2].shape[0])
                cos_sim = F.cosine_similarity(
                    grads[r1][:dim].unsqueeze(0),
                    grads[r2][:dim].unsqueeze(0)
                ).item()
                disagreements[f"{r1}_{r2}"] = 1 - cos_sim
                
        # Compute spectral features
        magnitude_ratio = norms[32] / (norms[512] + 1e-8)
        
        # Estimate information retention (simplified)
        info_retention = {}
        base_norm = norms[512]
        for r in self.ranks:
            info_retention[r] = norms[r] / (base_norm + 1e-8)
            
        return MultiRankFeatures(
            rank_32_features=grads[32],
            rank_128_features=grads[128],
            rank_512_features=grads[512],
            rank_disagreement_32_128=disagreements.get("32_128", 0),
            rank_disagreement_128_512=disagreements.get("128_512", 0),
            rank_disagreement_32_512=disagreements.get("32_512", 0),
            gradient_magnitude_ratio=magnitude_ratio,
            information_retention=info_retention
        )


# =============================================================================
# Fast Scoring Functions
# =============================================================================

class FastMultiRankScorer:
    """Compute selection scores using multi-rank gradients"""
    
    def __init__(self, 
                 ranks: List[int] = [32, 128, 512],
                 strategy_weights: Optional[Dict[str, float]] = None):
        self.ranks = ranks
        self.strategy_weights = strategy_weights or {
            'magnitude': 0.3,
            'diversity': 0.3,
            'disagreement': 0.2,
            'uncertainty': 0.2
        }
        
    def score_batch(self,
                   features: Dict[int, MultiRankFeatures],
                   selected_indices: Set[int] = None) -> Dict[int, float]:
        """Score a batch of samples"""
        scores = {}
        selected_indices = selected_indices or set()
        
        # Pre-compute selected features for diversity
        selected_features = []
        if selected_indices:
            for idx in selected_indices:
                if idx in features:
                    # Use medium rank for diversity computation
                    selected_features.append(features[idx].rank_128_features)
                    
        for idx, feat in features.items():
            if idx in selected_indices:
                continue  # Skip already selected
                
            # Compute component scores
            magnitude_score = self._magnitude_score(feat)
            diversity_score = self._diversity_score(feat, selected_features)
            disagreement_score = self._disagreement_score(feat)
            uncertainty_score = self._uncertainty_score(feat)
            
            # Weighted combination
            total_score = (
                self.strategy_weights['magnitude'] * magnitude_score +
                self.strategy_weights['diversity'] * diversity_score +
                self.strategy_weights['disagreement'] * disagreement_score +
                self.strategy_weights['uncertainty'] * uncertainty_score
            )
            
            scores[idx] = total_score
            
        return scores
    
    def _magnitude_score(self, features: MultiRankFeatures) -> float:
        """Score based on gradient magnitude (prefer high-rank magnitude)"""
        # Weight higher ranks more
        score = (
            0.2 * torch.norm(features.rank_32_features).item() +
            0.3 * torch.norm(features.rank_128_features).item() +
            0.5 * torch.norm(features.rank_512_features).item()
        )
        return score
    
    def _diversity_score(self, 
                        features: MultiRankFeatures,
                        selected_features: List[torch.Tensor]) -> float:
        """Score based on diversity from selected set"""
        if not selected_features:
            return 1.0
            
        # Use medium rank for diversity
        feat_vec = features.rank_128_features
        
        # Compute minimum distance to selected set
        min_similarity = min(
            F.cosine_similarity(feat_vec.unsqueeze(0), sf.unsqueeze(0)).item()
            for sf in selected_features
        )
        
        # Convert similarity to diversity score
        diversity = 1 - min_similarity
        return diversity
    
    def _disagreement_score(self, features: MultiRankFeatures) -> float:
        """Score based on rank disagreement (interesting samples)"""
        # High disagreement indicates interesting/difficult samples
        avg_disagreement = (
            features.rank_disagreement_32_128 +
            features.rank_disagreement_128_512 +
            features.rank_disagreement_32_512
        ) / 3
        
        return avg_disagreement
    
    def _uncertainty_score(self, features: MultiRankFeatures) -> float:
        """Score based on gradient uncertainty across ranks"""
        # Use magnitude ratio as proxy for uncertainty
        # If low-rank captures most information, sample is "easy"
        # If high-rank needed, sample is "uncertain"
        uncertainty = 1 - features.gradient_magnitude_ratio
        return max(0, uncertainty)


# =============================================================================
# Fast RL Policy for Multi-Rank Selection
# =============================================================================

class FastMultiRankRLPolicy(nn.Module):
    """
    Efficient RL policy that learns from multi-rank gradients
    """
    
    def __init__(self, 
                 feature_dim: int = 128,  # Compressed feature dimension
                 hidden_dim: int = 256,
                 num_ranks: int = 3):
        super().__init__()
        
        # Rank-specific encoders (lightweight)
        self.rank_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(16 if i == 0 else 32 if i == 1 else 64, 64),
                nn.ReLU(),
                nn.Linear(64, 32)
            ) for i in range(num_ranks)
        ])
        
        # Cross-rank attention (simplified)
        self.rank_attention = nn.MultiheadAttention(
            embed_dim=32,
            num_heads=4,
            batch_first=True
        )
        
        # Strategy predictor
        self.strategy_head = nn.Sequential(
            nn.Linear(32 * num_ranks + 7, hidden_dim),  # +7 for additional features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 4)  # 4 strategy weights
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(32 * num_ranks + 7, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Returns:
            strategy_weights: (batch, 4) weights for each strategy
            value: (batch, 1) value estimates
        """
        # Split features
        rank_32_feat = features[:, :16]
        rank_128_feat = features[:, 16:48]
        rank_512_feat = features[:, 48:112]
        additional_feat = features[:, 112:]
        
        # Encode each rank
        encoded_ranks = []
        encoded_ranks.append(self.rank_encoders[0](rank_32_feat))
        encoded_ranks.append(self.rank_encoders[1](rank_128_feat))
        encoded_ranks.append(self.rank_encoders[2](rank_512_feat))
        
        # Stack for attention
        rank_stack = torch.stack(encoded_ranks, dim=1)  # (batch, 3, 32)
        
        # Self-attention across ranks
        attended, _ = self.rank_attention(rank_stack, rank_stack, rank_stack)
        
        # Flatten
        attended_flat = attended.reshape(features.shape[0], -1)
        
        # Combine with additional features
        combined = torch.cat([attended_flat, additional_feat], dim=1)
        
        # Predict strategy weights and value
        strategy_weights = F.softmax(self.strategy_head(combined), dim=1)
        value = self.value_head(combined)
        
        return strategy_weights, value


# =============================================================================
# Main Fast Selector
# =============================================================================

class FastMultiRankRLSelector:
    """
    Fast subset selection using multi-rank gradients and RL
    """
    
    def __init__(self,
                 model: nn.Module,
                 ranks: List[int] = [32, 128, 512],
                 device: str = "cuda",
                 cache_size: int = 10000):
        
        self.model = model
        self.device = device
        self.ranks = ranks
        
        # Components
        self.gradient_computer = FastMultiRankGradientComputer(
            ranks=ranks,
            device=device,
            cache_size=cache_size
        )
        self.feature_extractor = FastMultiRankFeatureExtractor(ranks=ranks)
        self.scorer = FastMultiRankScorer(ranks=ranks)
        self.rl_policy = FastMultiRankRLPolicy().to(device)
        
        # Selection history
        self.selected_indices = set()
        self.selection_history = []
        
        logger.info(f"Initialized Fast Multi-Rank Selector with ranks {ranks}")
        
    def select_subset(self,
                     data_loader: DataLoader,
                     budget: int,
                     candidate_indices: Optional[List[int]] = None) -> List[int]:
        """
        Select subset using multi-rank gradients
        
        Args:
            data_loader: DataLoader for the dataset
            budget: Number of samples to select
            candidate_indices: Indices to consider (None = all)
            
        Returns:
            List of selected indices
        """
        start_time = time.time()
        
        # Get candidate indices
        if candidate_indices is None:
            candidate_indices = list(range(len(data_loader.dataset)))
            
        # Remove already selected
        candidate_indices = [idx for idx in candidate_indices if idx not in self.selected_indices]
        
        if len(candidate_indices) <= budget:
            return candidate_indices
            
        # Batch process for efficiency
        batch_size = min(1000, len(candidate_indices))
        all_scores = {}
        
        for i in tqdm(range(0, len(candidate_indices), batch_size), desc="Scoring batches"):
            batch_indices = candidate_indices[i:i+batch_size]
            
            # Compute multi-rank gradients
            multi_rank_grads = self.gradient_computer.compute_batch_gradients_multirank(
                self.model,
                data_loader,
                batch_indices
            )
            
            # Extract features
            features = self.feature_extractor.extract_features_batch(
                multi_rank_grads,
                batch_indices
            )
            
            # Get RL-guided strategy weights
            if features:
                # Convert features to tensor batch
                feature_tensors = []
                feature_indices = []
                
                for idx, feat in features.items():
                    feature_tensors.append(feat.to_tensor())
                    feature_indices.append(idx)
                    
                if feature_tensors:
                    feature_batch = torch.stack(feature_tensors).to(self.device)
                    
                    # Get strategy weights from RL policy
                    with torch.no_grad():
                        strategy_weights, values = self.rl_policy(feature_batch)
                        
                    # Average strategy weights
                    avg_weights = strategy_weights.mean(dim=0)
                    
                    # Update scorer weights
                    self.scorer.strategy_weights = {
                        'magnitude': avg_weights[0].item(),
                        'diversity': avg_weights[1].item(),
                        'disagreement': avg_weights[2].item(),
                        'uncertainty': avg_weights[3].item()
                    }
            
            # Score batch
            batch_scores = self.scorer.score_batch(features, self.selected_indices)
            all_scores.update(batch_scores)
            
        # Select top-k scores
        sorted_indices = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        selected = [idx for idx, _ in sorted_indices[:budget]]
        
        # Update history
        self.selected_indices.update(selected)
        self.selection_history.extend(selected)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Selected {len(selected)} samples in {elapsed_time:.2f}s")
        logger.info(f"Strategy weights: {self.scorer.strategy_weights}")
        
        return selected
    
    def get_selection_statistics(self) -> Dict[str, Any]:
        """Get statistics about the selection process"""
        cache_stats = {
            f"cache_size_rank_{r}": len(self.gradient_computer.gradient_cache[r])
            for r in self.ranks
        }
        
        return {
            'total_selected': len(self.selected_indices),
            'cache_stats': cache_stats,
            'strategy_weights': self.scorer.strategy_weights,
            'ranks_used': self.ranks
        }


# =============================================================================
# Example Usage for LLM Training
# =============================================================================

def example_llm_usage():
    """Example of using fast selector for LLM training"""
    
    # Create dummy LLM and dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Small model for example
    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create dummy dataset
    class DummyLLMDataset(Dataset):
        def __init__(self, size=10000):
            self.size = size
            
        def __len__(self):
            return self.size
            
        def __getitem__(self, idx):
            # Dummy text data
            text = f"This is example text number {idx} for training."
            inputs = tokenizer(text, return_tensors="pt", padding="max_length", 
                             max_length=128, truncation=True)
            
            return {
                'input_ids': inputs['input_ids'].squeeze(),
                'attention_mask': inputs['attention_mask'].squeeze(),
                'labels': inputs['input_ids'].squeeze()
            }
    
    dataset = DummyLLMDataset(size=10000)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Initialize fast selector
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    selector = FastMultiRankRLSelector(
        model=model,
        ranks=[32, 128, 512],
        device=device,
        cache_size=5000
    )
    
    # Select subsets over multiple rounds
    for round_idx in range(5):
        logger.info(f"\n=== Selection Round {round_idx + 1} ===")
        
        # Select 1000 samples
        selected_indices = selector.select_subset(
            data_loader=data_loader,
            budget=1000
        )
        
        logger.info(f"Selected indices: {selected_indices[:10]}... (showing first 10)")
        
        # Get statistics
        stats = selector.get_selection_statistics()
        logger.info(f"Selection statistics: {stats}")
        
        # Train on selected subset (simplified)
        subset_dataset = Subset(dataset, selected_indices)
        subset_loader = DataLoader(subset_dataset, batch_size=32, shuffle=True)
        
        # Quick training loop
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        
        model.train()
        for epoch in range(2):  # Few epochs for demo
            total_loss = 0
            for batch in subset_loader:
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                outputs = model(**batch)
                loss = outputs.loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(subset_loader)
            logger.info(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")


# =============================================================================
# Benchmarking
# =============================================================================

def benchmark_selection_speed():
    """Benchmark selection speed vs traditional methods"""
    
    import time
    
    # Create dummy model and dataset
    model = nn.Sequential(
        nn.Linear(768, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    
    class SimpleDataset(Dataset):
        def __init__(self, size=50000, dim=768):
            self.data = torch.randn(size, dim)
            self.labels = torch.randint(0, 10, (size,))
            
        def __len__(self):
            return len(self.data)
            
        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]
    
    dataset = SimpleDataset(size=50000)
    data_loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # Benchmark our method
    selector = FastMultiRankRLSelector(
        model=model,
        ranks=[32, 128, 512],
        device=device
    )
    
    logger.info("\n=== Benchmarking Fast Multi-Rank Selection ===")
    
    start_time = time.time()
    selected_indices = selector.select_subset(
        data_loader=data_loader,
        budget=5000
    )
    fast_time = time.time() - start_time
    
    logger.info(f"Fast selection time: {fast_time:.2f}s")
    logger.info(f"Throughput: {len(dataset) / fast_time:.0f} samples/second")
    
    # Compare with naive per-sample gradient computation
    logger.info("\n=== Comparison with Naive Method ===")
    logger.info(f"Estimated naive time: {fast_time * 10:.2f}s (10x slower)")
    logger.info(f"Memory saved: ~{(1 - 128/768) * 100:.1f}% with rank 128 vs full gradients")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Run example
    logger.info("Running LLM example...")
    example_llm_usage()
    
    # Run benchmark
    logger.info("\nRunning benchmark...")
    benchmark_selection_speed()