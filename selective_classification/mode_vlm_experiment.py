"""
MODE-VLM: Three-Part Framework

Part 1: MODE with Validation Rewards (as originally proposed)
Part 2: Selective Classification for VLM (Rho-1 inspired)
Part 3: Future Work - Combining MODE + Selective Classification

Author: Research Team
Date: 2025
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import copy


# ============================================================================
# PART 1: MODE with Improved Token Scoring Strategies
# ============================================================================

"""
Improved Token Scoring Strategies for MODE

Key improvements:
1. Remove redundant strategies (entropy vs coherence, loss vs perplexity)
2. Add gradient-based importance (Rho-1 style)
3. Add semantic coherence (contextual flow)
4. Add boundary proximity (decision boundaries)
5. More efficient diversity computation

Based on:
- MODE's multi-objective framework
- Rho-1's selective classification
- Token-level curriculum learning principles
"""

import torch.nn.functional as F
from collections import deque
from dataclasses import dataclass


@dataclass
class ImprovedTokenMODEConfig:
    """Configuration for token-level MODE"""
    
    # Strategy weights (learned by meta-controller)
    n_strategies: int = 6
    
    # Diversity tracking
    diversity_buffer_size: int = 1000
    diversity_sample_size: int = 100  # Sample for efficiency
    
    # Gradient computation
    enable_gradient_scores: bool = True
    gradient_accumulation_steps: int = 4
    
    # Semantic coherence
    coherence_window: int = 5  # Look at ±5 tokens for context
    
    # Boundary proximity
    boundary_temperature: float = 0.1
    
    # Reference model for excess loss (Rho-1 style)
    use_reference_model: bool = True
    reference_model_path: Optional[str] = None


class ImprovedTokenScorer:
    """
    Improved token scorer with 6 non-redundant strategies
    
    Strategy Portfolio:
    1. Excess Loss (Rho-1 inspired) - Training vs reference gap
    2. Gradient Magnitude - Importance for learning
    3. Attention Focus - Information flow quality
    4. Semantic Coherence - Contextual consistency
    5. Boundary Proximity - Near decision boundaries
    6. Diversity - Novelty relative to selected
    """
    
    def __init__(self, 
                 config: ImprovedTokenMODEConfig,
                 reference_model: Optional[nn.Module] = None):
        
        self.config = config
        self.reference_model = reference_model
        
        # For diversity tracking
        self.selected_embeddings = deque(maxlen=config.diversity_buffer_size)
        
        # Strategy names
        self.strategy_names = [
            'excess_loss',      # Rho-1: training loss - reference loss
            'gradient_norm',    # Importance for parameter updates
            'attention_focus',  # Quality of attention (focused vs diffuse)
            'semantic_coherence',  # Contextual consistency
            'boundary_proximity',  # Near decision boundaries
            'diversity'        # Novel representations
        ]
        
        # Statistics for normalization
        self.running_stats = {
            strategy: {'mean': 0.0, 'std': 1.0, 'count': 0}
            for strategy in self.strategy_names
        }
    
    def compute_all_scores(self,
                          input_ids: torch.Tensor,
                          outputs,
                          compute_gradients: bool = False) -> Dict[str, torch.Tensor]:
        """
        Compute all 6 strategy scores
        
        Args:
            input_ids: [batch, seq_len]
            outputs: Model outputs (logits, hidden_states, attentions)
            compute_gradients: Whether to compute gradient-based scores
            
        Returns:
            scores: Dict[strategy_name -> [batch, seq_len-1]]
        """
        
        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1]
        attentions = outputs.attentions
        
        batch_size, seq_len = input_ids.shape
        scores = {}
        
        # Predictions and targets
        pred_logits = logits[:, :-1, :]  # [batch, seq-1, vocab]
        target_ids = input_ids[:, 1:]     # [batch, seq-1]
        
        # ================================================================
        # STRATEGY 1: EXCESS LOSS (Rho-1 inspired)
        # ================================================================
        scores['excess_loss'] = self._compute_excess_loss(
            pred_logits, target_ids, input_ids
        )
        
        # ================================================================
        # STRATEGY 2: GRADIENT MAGNITUDE
        # ================================================================
        if compute_gradients and self.config.enable_gradient_scores:
            scores['gradient_norm'] = self._compute_gradient_importance(
                pred_logits, target_ids
            )
        else:
            # Fallback: use loss as proxy
            scores['gradient_norm'] = self._compute_loss(pred_logits, target_ids)
        
        # ================================================================
        # STRATEGY 3: ATTENTION FOCUS
        # ================================================================
        scores['attention_focus'] = self._compute_attention_focus(
            attentions, batch_size, seq_len
        )
        
        # ================================================================
        # STRATEGY 4: SEMANTIC COHERENCE
        # ================================================================
        scores['semantic_coherence'] = self._compute_semantic_coherence(
            hidden_states, window=self.config.coherence_window
        )
        
        # ================================================================
        # STRATEGY 5: BOUNDARY PROXIMITY
        # ================================================================
        scores['boundary_proximity'] = self._compute_boundary_proximity(
            pred_logits, target_ids
        )
        
        # ================================================================
        # STRATEGY 6: DIVERSITY
        # ================================================================
        scores['diversity'] = self._compute_diversity(
            hidden_states, batch_size, seq_len
        )
        
        # Normalize all scores to [0, 1]
        scores = self._normalize_scores(scores)
        
        return scores
    
    def _compute_excess_loss(self,
                            pred_logits: torch.Tensor,
                            target_ids: torch.Tensor,
                            input_ids: torch.Tensor) -> torch.Tensor:
        """
        Strategy 1: Excess Loss (Rho-1 inspired)
        
        excess_loss(token) = loss_train(token) - loss_ref(token)
        
        High excess loss -> Model struggles -> Informative token
        Low excess loss -> Model already learned -> Skip token
        
        This is MODE's version of Rho-1's selective classification
        """
        
        batch_size, seq_len_minus_1, vocab_size = pred_logits.shape
        
        # Training loss
        train_loss = F.cross_entropy(
            pred_logits.reshape(-1, vocab_size),
            target_ids.reshape(-1),
            reduction='none'
        ).reshape(batch_size, seq_len_minus_1)
        
        # Reference loss (if available)
        if self.reference_model is not None:
            with torch.no_grad():
                ref_outputs = self.reference_model(
                    input_ids,
                    output_hidden_states=True
                )
                ref_logits = ref_outputs.logits[:, :-1, :]
                
                ref_loss = F.cross_entropy(
                    ref_logits.reshape(-1, vocab_size),
                    target_ids.reshape(-1),
                    reduction='none'
                ).reshape(batch_size, seq_len_minus_1)
            
            # Excess loss = training loss - reference loss
            excess = torch.clamp(train_loss - ref_loss, min=0.0)
        else:
            # No reference model: use absolute loss
            # High loss = high importance
            excess = train_loss
        
        return excess
    
    def _compute_gradient_importance(self,
                                    pred_logits: torch.Tensor,
                                    target_ids: torch.Tensor) -> torch.Tensor:
        """
        Strategy 2: Gradient Magnitude
        
        Measures how much this token would change model parameters
        
        High gradient -> Important for learning
        Low gradient -> Model already handles well
        """
        
        batch_size, seq_len_minus_1, vocab_size = pred_logits.shape
        
        # Compute per-token loss (no reduction)
        loss_per_token = F.cross_entropy(
            pred_logits.reshape(-1, vocab_size),
            target_ids.reshape(-1),
            reduction='none'
        ).reshape(batch_size, seq_len_minus_1)
        
        # Compute gradients for each token
        gradient_norms = torch.zeros_like(loss_per_token)
        
        for b in range(batch_size):
            for t in range(seq_len_minus_1):
                # Compute gradient for this specific token
                if pred_logits[b, t].requires_grad:
                    token_loss = loss_per_token[b, t]
                    
                    # Get gradient w.r.t. logits
                    grad = torch.autograd.grad(
                        token_loss, 
                        pred_logits,
                        retain_graph=True,
                        create_graph=False
                    )[0]
                    
                    # Gradient norm for this token
                    gradient_norms[b, t] = grad[b, t].norm().item()
        
        return gradient_norms
    
    def _compute_loss(self,
                     pred_logits: torch.Tensor,
                     target_ids: torch.Tensor) -> torch.Tensor:
        """Fallback: Simple cross-entropy loss"""
        
        batch_size, seq_len_minus_1, vocab_size = pred_logits.shape
        
        loss = F.cross_entropy(
            pred_logits.reshape(-1, vocab_size),
            target_ids.reshape(-1),
            reduction='none'
        ).reshape(batch_size, seq_len_minus_1)
        
        return loss
    
    def _compute_attention_focus(self,
                                attentions: Tuple,
                                batch_size: int,
                                seq_len: int) -> torch.Tensor:
        """
        Strategy 3: Attention Focus
        
        Measures whether attention is focused or diffuse
        
        Focused attention -> Clear dependencies -> High quality
        Diffuse attention -> Uncertain -> May need more training
        
        Uses Gini coefficient instead of entropy for better sensitivity
        """
        
        if attentions is None or len(attentions) == 0:
            # No attention: return zeros
            return torch.zeros(batch_size, seq_len - 1)
        
        # Stack attentions: [layers, batch, heads, seq, seq]
        attn_stack = torch.stack([a for a in attentions])
        
        # Average over layers and heads: [batch, seq, seq]
        attn_mean = attn_stack.mean(dim=(0, 2))[:, :-1, :-1]
        
        # Compute Gini coefficient (focus metric)
        # Gini = 1 - 2 * sum(cumulative_sorted_probs)
        focus_scores = torch.zeros(batch_size, seq_len - 1, device=attn_mean.device)
        
        for b in range(batch_size):
            for t in range(seq_len - 1):
                attn_dist = attn_mean[b, t, :]
                
                # Sort attention weights
                sorted_attn, _ = torch.sort(attn_dist)
                
                # Cumulative sum
                cum_attn = torch.cumsum(sorted_attn, dim=0)
                
                # Gini coefficient
                n = len(sorted_attn)
                gini = 1.0 - 2.0 * cum_attn.sum() / (n * sorted_attn.sum() + 1e-10)
                
                focus_scores[b, t] = gini
        
        return focus_scores
    
    def _compute_semantic_coherence(self,
                                   hidden_states: torch.Tensor,
                                   window: int = 5) -> torch.Tensor:
        """
        Strategy 4: Semantic Coherence
        
        Measures how well token fits with surrounding context
        
        High coherence -> Token consistent with context
        Low coherence -> Token disrupts semantic flow -> May be noise or important boundary
        
        Uses cosine similarity with local context window
        """
        
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden = hidden_states[:, :-1, :]  # [batch, seq-1, hidden]
        
        coherence_scores = torch.zeros(batch_size, seq_len - 1, device=hidden.device)
        
        for b in range(batch_size):
            for t in range(seq_len - 1):
                # Define context window
                start = max(0, t - window)
                end = min(seq_len - 1, t + window + 1)
                
                # Current token embedding
                current = hidden[b, t]
                
                # Context embeddings (excluding current)
                context_indices = list(range(start, t)) + list(range(t + 1, end))
                
                if len(context_indices) > 0:
                    context = hidden[b, context_indices]
                    
                    # Mean context embedding
                    context_mean = context.mean(dim=0)
                    
                    # Cosine similarity
                    coherence = F.cosine_similarity(
                        current.unsqueeze(0),
                        context_mean.unsqueeze(0),
                        dim=1
                    ).item()
                    
                    coherence_scores[b, t] = coherence
                else:
                    # No context: neutral score
                    coherence_scores[b, t] = 0.5
        
        # Convert to [0, 1]: (-1, 1) -> (0, 1)
        coherence_scores = (coherence_scores + 1.0) / 2.0
        
        return coherence_scores
    
    def _compute_boundary_proximity(self,
                                   pred_logits: torch.Tensor,
                                   target_ids: torch.Tensor) -> torch.Tensor:
        """
        Strategy 5: Boundary Proximity
        
        Identifies tokens near decision boundaries
        
        High proximity -> Model uncertain between multiple tokens -> Informative
        Low proximity -> Model confident in single prediction
        
        Measures margin between top-1 and top-2 predictions
        """
        
        batch_size, seq_len_minus_1, vocab_size = pred_logits.shape
        
        # Get top-2 logits
        top2_logits, top2_indices = torch.topk(pred_logits, k=2, dim=-1)
        
        # Margin between top-1 and top-2
        margin = top2_logits[:, :, 0] - top2_logits[:, :, 1]
        
        # Small margin -> Near boundary -> High score
        # Large margin -> Far from boundary -> Low score
        # Use sigmoid to map to [0, 1]
        proximity = torch.sigmoid(-margin / self.config.boundary_temperature)
        
        return proximity
    
    def _compute_diversity(self,
                          hidden_states: torch.Tensor,
                          batch_size: int,
                          seq_len: int) -> torch.Tensor:
        """
        Strategy 6: Diversity (Efficient Version)
        
        Measures novelty relative to previously selected tokens
        
        High diversity -> Novel representation -> Prevents redundancy
        Low diversity -> Similar to selected -> Skip to avoid redundancy
        
        Optimized: Sample from buffer instead of computing all distances
        """
        
        hidden = hidden_states[:, :-1, :]  # [batch, seq-1, hidden]
        
        if len(self.selected_embeddings) < 10:
            # Not enough history: return uniform
            return torch.ones(batch_size, seq_len - 1, device=hidden.device) * 0.5
        
        # Sample from buffer for efficiency
        sample_size = min(
            self.config.diversity_sample_size,
            len(self.selected_embeddings)
        )
        
        # Random sample
        indices = torch.randint(0, len(self.selected_embeddings), (sample_size,))
        sampled_embeddings = [self.selected_embeddings[i] for i in indices]
        selected_stack = torch.stack(sampled_embeddings).to(hidden.device)  # [N, hidden]
        
        # Compute min distance for each token
        diversity_scores = torch.zeros(batch_size, seq_len - 1, device=hidden.device)
        
        for b in range(batch_size):
            for t in range(seq_len - 1):
                current = hidden[b, t]  # [hidden]
                
                # Cosine distances to all sampled
                similarities = F.cosine_similarity(
                    current.unsqueeze(0),
                    selected_stack,
                    dim=1
                )
                
                # Min similarity = max diversity
                diversity_scores[b, t] = 1.0 - similarities.max()
        
        return diversity_scores
    
    def _normalize_scores(self, scores: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Normalize all scores to [0, 1] using running statistics
        
        Uses exponential moving average for stable normalization
        """
        
        normalized = {}
        alpha = 0.1  # EMA coefficient
        
        for strategy, score in scores.items():
            # Update running statistics
            mean = score.mean().item()
            std = score.std().item()
            
            stats = self.running_stats[strategy]
            stats['mean'] = alpha * mean + (1 - alpha) * stats['mean']
            stats['std'] = alpha * std + (1 - alpha) * stats['std']
            stats['count'] += 1
            
            # Standardize
            standardized = (score - stats['mean']) / (stats['std'] + 1e-8)
            
            # Map to [0, 1] using sigmoid
            normalized[strategy] = torch.sigmoid(standardized)
        
        return normalized
    
    def update_selected_embeddings(self, 
                                   hidden_states: torch.Tensor,
                                   mask: torch.Tensor):
        """
        Update diversity buffer with selected token embeddings
        
        Args:
            hidden_states: [batch, seq, hidden]
            mask: [batch, seq-1] binary mask of selected tokens
        """
        
        batch_size = hidden_states.size(0)
        hidden = hidden_states[:, :-1, :]  # Align with mask
        
        for b in range(batch_size):
            selected = hidden[b][mask[b] == 1.0]
            for emb in selected:
                # Store on CPU to save GPU memory
                self.selected_embeddings.append(emb.detach().cpu())


# ================================================================
# STRATEGY USAGE GUIDELINES
# ================================================================

class StrategyUsageGuide:
    """
    When to prioritize each strategy during training
    
    Based on MODE's adaptive framework and token-level curriculum principles
    """
    
    @staticmethod
    def get_strategy_priorities(training_phase: str) -> Dict[str, float]:
        """
        Get recommended strategy weights for different training phases
        
        Args:
            training_phase: 'warmup', 'early', 'middle', 'late'
            
        Returns:
            weights: Recommended weights for each strategy
        """
        
        if training_phase == 'warmup':
            # Focus on easy, coherent tokens
            return {
                'excess_loss': 0.1,        # Low - avoid hard tokens
                'gradient_norm': 0.1,      # Low
                'attention_focus': 0.2,    # High - learn attention
                'semantic_coherence': 0.3, # High - build foundation
                'boundary_proximity': 0.1, # Low
                'diversity': 0.2          # Medium - exploration
            }
        
        elif training_phase == 'early':
            # Balance between learning and exploration
            return {
                'excess_loss': 0.25,       # Growing - learn from errors
                'gradient_norm': 0.2,      # Medium
                'attention_focus': 0.15,   # Medium
                'semantic_coherence': 0.15, # Medium
                'boundary_proximity': 0.15, # Medium - start refining
                'diversity': 0.1          # Lower - less exploration needed
            }
        
        elif training_phase == 'middle':
            # Focus on hard tokens and boundaries
            return {
                'excess_loss': 0.3,        # High - tackle hard cases
                'gradient_norm': 0.25,     # High - maximize learning
                'attention_focus': 0.1,    # Lower
                'semantic_coherence': 0.1, # Lower
                'boundary_proximity': 0.25, # High - decision boundaries
                'diversity': 0.0          # Zero - focus on difficult
            }
        
        elif training_phase == 'late':
            # Polish and refine
            return {
                'excess_loss': 0.2,        # Medium
                'gradient_norm': 0.15,     # Lower
                'attention_focus': 0.2,    # Higher - quality focus
                'semantic_coherence': 0.25, # High - coherent generation
                'boundary_proximity': 0.15, # Medium
                'diversity': 0.05         # Very low - refinement
            }
        
        else:
            # Uniform fallback
            return {s: 1.0/6 for s in [
                'excess_loss', 'gradient_norm', 'attention_focus',
                'semantic_coherence', 'boundary_proximity', 'diversity'
            ]}
    
    @staticmethod
    def explain_strategies():
        """Print explanation of when to use each strategy"""
        
        explanations = {
            'excess_loss': """
            EXCESS LOSS (Rho-1 inspired)
            When: Throughout training, especially middle phase
            Why: Identifies tokens where model struggles most
            High score = Model needs to learn this token
            Low score = Model already knows this token -> Skip
            """,
            
            'gradient_norm': """
            GRADIENT MAGNITUDE  
            When: Early-middle phase for maximum learning
            Why: Tokens with high gradients have most impact on parameters
            High score = Important for parameter updates
            Low score = Minimal learning benefit
            """,
            
            'attention_focus': """
            ATTENTION FOCUS
            When: Warmup and late phase for quality
            Why: Well-focused attention indicates good dependencies
            High score = Clear attention pattern
            Low score = Diffuse/uncertain attention
            """,
            
            'semantic_coherence': """
            SEMANTIC COHERENCE
            When: Warmup for foundation, late for polish
            Why: Ensures tokens fit semantic context
            High score = Contextually consistent
            Low score = Disrupts semantic flow
            """,
            
            'boundary_proximity': """
            BOUNDARY PROXIMITY
            When: Middle-late phase for refinement
            Why: Tokens near decision boundaries need refinement
            High score = Near boundary -> Informative
            Low score = Clear prediction -> Less informative
            """,
            
            'diversity': """
            DIVERSITY
            When: Warmup-early for exploration
            Why: Ensures broad coverage of representation space
            High score = Novel representation
            Low score = Redundant with selected tokens
            """
        }
        
        for strategy, explanation in explanations.items():
            print(f"\n{'='*60}")
            print(explanation.strip())


# ============================================================================
# WEBDATASET INTEGRATION FOR MODE-VLM
# ============================================================================

import webdataset as wds
from huggingface_hub import HfFileSystem, get_token, hf_hub_url
from torchvision import transforms
from PIL import Image
import io
import json


def create_datasets():
    """Create train, test, and validation datasets from CC3M webdataset"""
    
    # Define splits - including test split
    splits = {
        'train': '**/*-train-*.tar', 
        'validation': '**/*-validation-*.tar',
        'test': '**/*-test-*.tar'  # Add test split if available
    }
    
    fs = HfFileSystem()
    datasets = {}
    
    for split_name, pattern in splits.items():
        try:
            # Get file paths for this split
            files = [fs.resolve_path(path) for path in fs.glob(f"hf://datasets/pixparse/cc3m-wds/{pattern}")]
            
            if files:
                # Create URLs with authentication
                urls = [hf_hub_url(file.repo_id, file.path_in_repo, repo_type="dataset") for file in files]
                urls_string = f"pipe: curl -s -L -H 'Authorization:Bearer {get_token()}' {'::'.join(urls)}"
                
                # Create WebDataset for this split
                datasets[split_name] = wds.WebDataset(urls_string).decode()
                print(f"Created {split_name} dataset with {len(files)} tar files")
            else:
                print(f"No files found for {split_name} split")
                
        except Exception as e:
            print(f"Error creating {split_name} dataset: {e}")
    
    return datasets


def create_train_val_test_from_train_only():
    """
    Alternative approach: Create train/val/test splits from training data only
    This is useful if only training data is available in the dataset
    """
    
    fs = HfFileSystem()
    
    # Get all training files
    train_pattern = "**/*-train-*.tar"
    files = [fs.resolve_path(path) for path in fs.glob(f"hf://datasets/pixparse/cc3m-wds/{train_pattern}")]
    
    if not files:
        print("No training files found")
        return {}
    
    # Split files into train/val/test (70/15/15 split)
    total_files = len(files)
    train_end = int(0.7 * total_files)
    val_end = int(0.85 * total_files)
    
    file_splits = {
        'train': files[:train_end],
        'validation': files[train_end:val_end],
        'test': files[val_end:]
    }
    
    datasets = {}
    
    for split_name, split_files in file_splits.items():
        if split_files:
            # Create URLs with authentication
            urls = [hf_hub_url(file.repo_id, file.path_in_repo, repo_type="dataset") for file in split_files]
            urls_string = f"pipe: curl -s -L -H 'Authorization:Bearer {get_token()}' {'::'.join(urls)}"
            
            # Create WebDataset for this split
            datasets[split_name] = wds.WebDataset(urls_string).decode()
            print(f"Created {split_name} dataset with {len(split_files)} tar files")
    
    return datasets


def add_preprocessing_pipeline(dataset, split_name="train"):
    """
    Add preprocessing pipeline to the dataset
    Customize this based on your specific needs
    """
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    def preprocess_sample(sample):
        # Add any preprocessing logic here
        # For example, image resizing, normalization, etc.
        if 'jpg' in sample:
            try:
                image_data = sample['jpg']
                image = Image.open(io.BytesIO(image_data)).convert("RGB")
                sample['jpg'] = transform(image)
            except Exception as e:
                # Handle cases where the image data is corrupt
                # You might want to return None or a placeholder tensor
                return None
        if 'txt' in sample:
            # The text is already a string, so no further decoding is needed
            pass
        if 'json' in sample:
            try:
                sample['json'] = json.loads(sample['json'])
            except (json.JSONDecodeError, TypeError):
                # Handle cases where the json data is corrupt or not a string
                return None
        return sample

    # Add common webdataset operations
    processed_dataset = (dataset
                        .map(preprocess_sample)
                        .shuffle(1000 if split_name == "train" else 100)  # Larger shuffle for training
                        .batched(32)  # Adjust batch size as needed
                        )
    
    return processed_dataset


class VLMDatasetWrapper:
    """
    Wrapper to integrate WebDataset with MODE-VLM framework
    
    Converts WebDataset samples to format expected by MODE token scoring
    """
    
    def __init__(self, webdataset, tokenizer, device='cuda'):
        self.webdataset = webdataset
        self.tokenizer = tokenizer
        self.device = device
    
    def __iter__(self):
        for batch in self.webdataset:
            # Convert WebDataset batch to MODE format
            processed_batch = self._process_batch(batch)
            yield processed_batch
    
    def _process_batch(self, batch):
        """
        Convert WebDataset batch to MODE-compatible format
        
        Args:
            batch: Dict with 'jpg' (images) and 'txt' (text)
            
        Returns:
            Dict with processed images, text, and token ids
        """
        
        processed = {
            'images': [],
            'texts': [],
            'input_ids': [],
            'attention_mask': []
        }
        
        # Handle batch structure
        if isinstance(batch, dict):
            # Single sample
            batch = [batch]
        
        for sample in batch:
            if sample is None:
                continue
                
            # Extract image
            if 'jpg' in sample and sample['jpg'] is not None:
                processed['images'].append(sample['jpg'])
            
            # Extract text
            if 'txt' in sample and sample['txt'] is not None:
                text = sample['txt']
                processed['texts'].append(text)
                
                # Tokenize text for MODE token scoring
                tokens = self.tokenizer(
                    text,
                    padding='max_length',
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )
                
                processed['input_ids'].append(tokens['input_ids'].squeeze(0))
                processed['attention_mask'].append(tokens['attention_mask'].squeeze(0))
        
        # Convert to tensors
        if processed['images']:
            processed['images'] = torch.stack(processed['images']).to(self.device)
        
        if processed['input_ids']:
            processed['input_ids'] = torch.stack(processed['input_ids']).to(self.device)
            processed['attention_mask'] = torch.stack(processed['attention_mask']).to(self.device)
        
        return processed


class MODEMetaController(nn.Module):
    """
    Meta-controller for adaptive strategy weighting
    
    As per MODE paper:
    - Input: 5D training state [epoch, accuracy, grad_norm, budget, perf]
    - Output: Strategy weights via learned MLP
    - Training: Validation-based rewards
    """
    
    def __init__(self, num_strategies: int = 6, hidden_dim: int = 64):
        super().__init__()
        
        self.num_strategies = num_strategies
        
        # MLP as per paper
        self.network = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_strategies)
        )
        
        # Strategy effectiveness tracking
        self.alpha = nn.Parameter(torch.ones(num_strategies))
        
    def encode_training_state(self, metrics: Dict) -> torch.Tensor:
        """
        Encode training state as 5D vector
        
        s_t = [e_t, a_t, g_t, b_t, v_t]
        
        where:
        - e_t: current epoch / max_epochs (progress)
        - a_t: recent validation accuracy (performance)
        - g_t: log gradient magnitude (learning dynamics)
        - b_t: remaining budget fraction (resource constraint)
        - v_t: average strategy performance (meta-feedback)
        """
        state = torch.zeros(5)
        
        # Epoch progress
        state[0] = metrics.get('epoch', 0) / metrics.get('total_epochs', 32)
        
        # Validation accuracy (moving average)
        state[1] = metrics.get('val_accuracy', 0.0)
        
        # Gradient magnitude (log scale)
        grad_norm = metrics.get('grad_norm', 1.0)
        state[2] = np.log(grad_norm + 1e-8) / 10.0  # Normalize
        
        # Remaining budget
        state[3] = metrics.get('remaining_budget', 1.0)
        
        # Average strategy performance
        state[4] = metrics.get('avg_strategy_perf', 0.0)
        
        return state
    
    def forward(self, training_state: Dict, temperature: float = 1.0):
        """
        Compute strategy weights from training state
        
        w_t = softmax(h_φ(s_t) / τ_t)
        
        Returns: [num_strategies] normalized weights
        """
        state = self.encode_training_state(training_state)
        
        # Get logits from network
        logits = self.network(state)
        
        # Add learned strategy effectiveness (alpha)
        logits = logits + self.alpha
        
        # Temperature-controlled softmax
        weights = torch.softmax(logits / temperature, dim=0)
        
        return weights


class MODESelector:
    """
    Complete MODE selector with improved token scoring and validation rewards
    
    Key innovation: Uses improved token scoring strategies with validation-based rewards
    
    Algorithm:
    1. Compute scores from all token-level strategies
    2. Combine using learned weights
    3. Select top-k tokens/samples
    4. Train on selected samples
    5. Measure validation improvement -> REWARD
    6. Update strategy weights based on reward
    """
    
    def __init__(self, 
                 device='cuda',
                 learning_rate: float = 0.001,
                 temperature_init: float = 1.0,
                 reference_model: Optional[nn.Module] = None):
        
        self.device = device
        self.lr = learning_rate
        self.temperature_init = temperature_init
        
        # Initialize improved token scorer
        config = ImprovedTokenMODEConfig(
            n_strategies=6,
            diversity_buffer_size=1000,
            enable_gradient_scores=True,
            use_reference_model=reference_model is not None
        )
        self.token_scorer = ImprovedTokenScorer(config, reference_model)
        
        # Meta-controller (updated for 6 strategies)
        self.meta_controller = MODEMetaController(
            num_strategies=config.n_strategies
        ).to(device)
        
        # Optimizer for meta-controller
        self.optimizer = torch.optim.Adam(
            self.meta_controller.parameters(), 
            lr=learning_rate
        )
        
        # Tracking
        self.selection_history = []
        self.reward_history = []
        
    def compute_temperature(self, epoch: int, total_epochs: int, 
                           remaining_budget: float) -> float:
        """
        Compute annealing temperature
        
        τ_t = τ_0 · exp(-α(1 - b_t)) · exp(-β · e_t/E_max)
        
        Dual decay:
        - Budget-driven: explore when budget abundant
        - Epoch-driven: exploit as training progresses
        """
        tau_0 = self.temperature_init
        alpha = 0.5  # Budget decay rate
        beta = 1.5   # Epoch decay rate
        tau_min = 0.1
        
        progress = epoch / total_epochs
        
        temp = tau_0 * np.exp(-alpha * (1 - remaining_budget)) * \
               np.exp(-beta * progress)
        
        return max(temp, tau_min)
    
    def select_batch(self,
                    input_ids: torch.Tensor,
                    model_outputs,
                    model,
                    val_loader,
                    budget: int,
                    training_state: Dict,
                    compute_gradients: bool = False) -> Tuple[List[int], Dict]:
        """
        Select batch using MODE with improved token scoring and validation rewards
        
        Args:
            input_ids: [batch, seq_len] token ids
            model_outputs: Model outputs with logits, hidden_states, attentions
            model: Training model for validation evaluation
            val_loader: Validation data loader
            budget: Number of tokens/samples to select
            training_state: Dict with training metrics
            compute_gradients: Whether to compute gradient-based scores
        
        Returns:
            selected_indices: List of selected token indices (flattened)
            metadata: Dict with scores, weights, rewards
        """
        
        batch_size, seq_len = input_ids.shape
        
        # Step 1: Get current temperature
        temperature = self.compute_temperature(
            training_state['epoch'],
            training_state['total_epochs'],
            training_state['remaining_budget']
        )
        
        # Step 2: Get strategy weights from meta-controller
        with torch.no_grad():
            weights = self.meta_controller(training_state, temperature)
        
        print(f"\nStrategy weights: {dict(zip(self.token_scorer.strategy_names, weights.tolist()))}")
        
        # Step 3: Compute scores from all token-level strategies
        strategy_scores = self.token_scorer.compute_all_scores(
            input_ids, model_outputs, compute_gradients
        )
        
        # Step 4: Combine scores using learned weights
        # S_MODE(token, t) = Σ w_t,i · S_i(token, t)
        
        # All scores are [batch, seq_len-1], flatten for selection
        combined_scores = torch.zeros(batch_size, seq_len - 1, device=input_ids.device)
        
        for i, strategy_name in enumerate(self.token_scorer.strategy_names):
            score = strategy_scores[strategy_name]
            combined_scores += weights[i].item() * score
        
        # Flatten for selection
        flat_scores = combined_scores.view(-1)
        
        # Step 5: Select top-k tokens
        budget = min(budget, len(flat_scores))
        selected_indices = torch.topk(flat_scores, k=budget).indices.tolist()
        
        # Convert flat indices back to (batch, token) pairs for tracking
        selected_tokens = []
        for idx in selected_indices:
            batch_idx = idx // (seq_len - 1)
            token_idx = idx % (seq_len - 1)
            selected_tokens.append((batch_idx, token_idx))
        
        # Step 6: Train on selected batch and measure validation reward
        model_before = copy.deepcopy(model)
        
        # This would be your actual training step on selected tokens
        # model.train_on_selected_tokens(input_ids, selected_tokens, ...)
        
        # Measure validation performance
        val_acc_before = self._evaluate(model_before, val_loader)
        val_acc_after = self._evaluate(model, val_loader)
        
        # Compute reward (validation improvement)
        delta_val = val_acc_after - val_acc_before
        
        # Step 7: Compute per-strategy rewards
        # r_j^(t) = Δ_val^(t) · w_t,j · 1[Δ_val > 0]
        
        strategy_rewards = {}
        for i, strategy_name in enumerate(self.token_scorer.strategy_names):
            reward = delta_val * weights[i].item() * float(delta_val > 0)
            strategy_rewards[strategy_name] = reward
        
        # Step 8: Update meta-controller using rewards
        # α_j^(t+1) = α_j^(t) + η · r_j^(t)
        
        with torch.no_grad():
            for i, strategy_name in enumerate(self.token_scorer.strategy_names):
                self.meta_controller.alpha[i] += self.lr * strategy_rewards[strategy_name]
        
        # Step 9: Update diversity buffer with selected embeddings
        if hasattr(model_outputs, 'hidden_states') and model_outputs.hidden_states:
            # Create mask for selected tokens
            selection_mask = torch.zeros(batch_size, seq_len - 1, device=input_ids.device)
            for batch_idx, token_idx in selected_tokens:
                selection_mask[batch_idx, token_idx] = 1.0
            
            self.token_scorer.update_selected_embeddings(
                model_outputs.hidden_states[-1], selection_mask
            )
        
        # Metadata
        metadata = {
            'weights': weights.cpu().numpy(),
            'strategy_scores': {k: v.cpu().numpy() for k, v in strategy_scores.items()},
            'combined_scores': combined_scores.cpu().numpy(),
            'selected_tokens': selected_tokens,
            'delta_val': delta_val,
            'strategy_rewards': strategy_rewards,
            'temperature': temperature
        }
        
        print(f"Validation Δ: {delta_val:+.4f}")
        print(f"Strategy rewards: {strategy_rewards}")
        
        return selected_indices, metadata
    
    def _evaluate(self, model, val_loader):
        """Evaluate model on validation set"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Your evaluation logic
                pass
        
        return correct / total if total > 0 else 0.0



# ============================================================================
# PART 2: Selective Classification for VLM (Rho-1 Inspired)
# ============================================================================

class SelectiveVLMClassifier:
    """
    Selective classification for vision-language models
    
    Key idea from Rho-1: Use excess loss to select informative samples
    
    For VLM:
    - Reference model: Pretrained CLIP (represents "clean" distribution)
    - Training model: Current model being trained
    - Excess loss: How much more does current model struggle?
    
    Selective rule: Include sample if excess_loss > threshold
    """
    
    def __init__(self, reference_model, threshold_percentile: float = 0.75):
        """
        Args:
            reference_model: Pretrained CLIP (reference distribution)
            threshold_percentile: Percentile for adaptive thresholding
        """
        self.reference_model = reference_model
        self.reference_model.eval()
        self.threshold_percentile = threshold_percentile
        
    def compute_excess_loss(self,
                           image_features: torch.Tensor,
                           text_features: torch.Tensor,
                           training_model) -> torch.Tensor:
        """
        Compute excess loss for each sample
        
        excess_loss(x) = L_train(x) - L_ref(x)
        
        High excess loss → model struggles → informative sample
        Low excess loss → model already knows → redundant sample
        
        Returns: [N] excess loss per sample
        """
        N = len(image_features)
        excess_losses = torch.zeros(N)
        
        training_model.eval()
        
        with torch.no_grad():
            for i in range(N):
                img_feat = image_features[i:i+1]
                txt_feat = text_features[i:i+1]
                
                # Reference loss (what a good model should get)
                ref_logits = self.reference_model.compute_similarity(img_feat, txt_feat)
                ref_loss = -torch.log_softmax(ref_logits, dim=-1)[0, 0]
                
                # Training loss (what current model gets)
                train_logits = training_model.compute_similarity(img_feat, txt_feat)
                train_loss = -torch.log_softmax(train_logits, dim=-1)[0, 0]
                
                # Excess loss
                excess = train_loss - ref_loss
                excess_losses[i] = max(0, excess.item())  # Clamp negatives
        
        return excess_losses
    
    def select_with_confidence(self,
                               image_features: torch.Tensor,
                               text_features: torch.Tensor,
                               training_model,
                               budget: float = 0.5) -> Tuple[torch.Tensor, float]:
        """
        Select samples using selective classification principle
        
        Algorithm:
        1. Compute excess loss for all samples
        2. Set threshold at percentile (adaptive)
        3. Select samples where excess_loss > threshold
        4. If selected < budget, lower threshold
        
        Returns:
            selected_mask: [N] boolean mask of selected samples
            threshold: Adaptive threshold used
        """
        
        # Compute excess losses
        excess_losses = self.compute_excess_loss(
            image_features,
            text_features,
            training_model
        )
        
        # Adaptive threshold based on distribution
        threshold = torch.quantile(excess_losses, self.threshold_percentile)
        
        # Select samples above threshold
        selected_mask = excess_losses > threshold
        
        # Ensure we meet budget constraint
        n_selected = selected_mask.sum().item()
        target = int(len(excess_losses) * budget)
        
        if n_selected < target:
            # Relax threshold to meet budget
            sorted_losses, sorted_indices = torch.sort(excess_losses, descending=True)
            selected_indices = sorted_indices[:target]
            selected_mask = torch.zeros_like(excess_losses, dtype=torch.bool)
            selected_mask[selected_indices] = True
            threshold = sorted_losses[target-1].item()
        
        print(f"\nSelective Classification:")
        print(f"  Threshold: {threshold:.4f}")
        print(f"  Selected: {selected_mask.sum()}/{len(selected_mask)}")
        print(f"  Mean excess loss (selected): {excess_losses[selected_mask].mean():.4f}")
        print(f"  Mean excess loss (rejected): {excess_losses[~selected_mask].mean():.4f}")
        
        return selected_mask, threshold
    
    def get_coverage_risk_curve(self,
                                image_features: torch.Tensor,
                                text_features: torch.Tensor,
                                training_model,
                                coverages: List[float] = [0.1, 0.3, 0.5, 0.7, 0.9]) -> Dict:
        """
        Compute risk-coverage curve for selective classification
        
        For each coverage level:
        - Select top-coverage samples by excess loss
        - Compute average excess loss (risk)
        - Show tradeoff
        
        Returns: Dict with coverage -> (risk, threshold)
        """
        
        excess_losses = self.compute_excess_loss(
            image_features,
            text_features,
            training_model
        )
        
        curve = {}
        
        for coverage in coverages:
            k = int(len(excess_losses) * coverage)
            
            # Select top-k by excess loss
            topk_losses, _ = torch.topk(excess_losses, k=k)
            
            # Risk = average excess loss
            risk = topk_losses.mean().item()
            
            # Threshold
            threshold = topk_losses.min().item()
            
            curve[coverage] = {
                'risk': risk,
                'threshold': threshold
            }
        
        return curve



# ============================================================================
# PART 3: FUTURE WORK - Combining MODE + Selective Classification
# ============================================================================

class MODESelectiveCombined:
    """
    FUTURE WORK: Combine MODE's multi-strategy learning with selective classification
    
    Two approaches:
    
    Approach 1: Excess loss as MODE strategy
    - Add SelectiveClassifier as 6th MODE strategy
    - Let meta-controller learn when to use it
    
    Approach 2: Two-stage filtering
    - Stage 1: Selective classification (coarse filter)
    - Stage 2: MODE on filtered set (fine-grained selection)
    
    Approach 3: Joint optimization
    - MODE learns strategy weights
    - Selective classification provides confidence bounds
    - Combine: S_final = S_MODE * confidence(excess_loss)
    """
    
    def __init__(self, reference_model, device='cuda'):
        self.device = device
        
        # MODE components
        self.mode_selector = MODESelector(device=device)
        
        # Selective classification components
        self.selective_classifier = SelectiveVLMClassifier(reference_model)
        
    def approach_1_strategy_integration(self,
                                       image_features,
                                       text_features,
                                       training_model,
                                       budget):
        """
        Approach 1: Add excess loss as MODE strategy
        
        Advantages:
        - Learns when excess loss is useful
        - Unified framework
        - Automatic weight learning
        """
        
        # Compute excess loss scores
        excess_losses = self.selective_classifier.compute_excess_loss(
            image_features,
            text_features,
            training_model
        )
        
        # Normalize to [0, 1]
        excess_scores = (excess_losses - excess_losses.min()) / \
                       (excess_losses.max() - excess_losses.min() + 1e-8)
        
        # Add as 6th strategy to MODE
        # self.mode_selector.strategies['excess_loss'] = ExcessLossStrategy()
        
        # MODE will learn when to use this strategy via validation rewards
        
        pass
    
    def approach_2_two_stage(self,
                            image_features,
                            text_features,
                            training_model,
                            model,
                            val_loader,
                            budget,
                            training_state):
        """
        Approach 2: Two-stage filtering
        
        Stage 1 (Selective): Coarse filter via excess loss
        Stage 2 (MODE): Fine-grained selection from filtered set
        
        Advantages:
        - Removes clearly redundant samples first
        - MODE focuses on nuanced decisions
        - Computationally efficient
        
        Example:
        - Stage 1: Filter 1M → 100K (10× reduction) using excess loss
        - Stage 2: Select 10K from 100K using MODE
        """
        
        print("\n" + "="*80)
        print("Two-Stage Selection: Selective + MODE")
        print("="*80)
        
        # Stage 1: Selective classification
        print("\nStage 1: Selective Classification (Coarse Filter)")
        coarse_budget = min(1.0, budget * 10)  # Keep 10× target for MODE
        
        selected_mask, threshold = self.selective_classifier.select_with_confidence(
            image_features,
            text_features,
            training_model,
            budget=coarse_budget
        )
        
        # Filter features
        filtered_img = image_features[selected_mask]
        filtered_txt = text_features[selected_mask]
        filtered_indices = torch.where(selected_mask)[0]
        
        print(f"  Filtered: {len(filtered_img)}/{len(image_features)} samples")
        
        # Stage 2: MODE selection
        print("\nStage 2: MODE Selection (Fine-Grained)")
        fine_budget = int(len(filtered_img) * (budget / coarse_budget))
        
        selected_from_filtered, metadata = self.mode_selector.select_batch(
            filtered_img,
            filtered_txt,
            model,
            val_loader,
            budget=fine_budget,
            training_state=training_state
        )
        
        # Map back to original indices
        final_selected = filtered_indices[selected_from_filtered].tolist()
        
        print(f"  Final selected: {len(final_selected)} samples")
        
        return final_selected, metadata
    
    def approach_3_joint_optimization(self,
                                     image_features,
                                     text_features,
                                     training_model,
                                     model,
                                     val_loader,
                                     budget,
                                     training_state):
        """
        Approach 3: Joint optimization
        
        Combine MODE scores with selective classification confidence
        
        S_final(x) = S_MODE(x) × confidence(excess_loss(x))
        
        where confidence = sigmoid(excess_loss - threshold)
        
        Advantages:
        - Soft weighting (not hard filtering)
        - MODE and selective inform each other
        - Preserves MODE's learned dynamics
        
        Challenges:
        - How to set threshold?
        - How to balance the two signals?
        - Need careful tuning
        """
        
        print("\n" + "="*80)
        print("Joint Optimization: MODE × Selective Confidence")
        print("="*80)
        
        # Get MODE scores
        selected_mode, metadata = self.mode_selector.select_batch(
            image_features,
            text_features,
            model,
            val_loader,
            budget=len(image_features),  # Score all samples
            training_state=training_state
        )
        
        mode_scores = torch.tensor(metadata['combined_scores'])
        
        # Get excess loss confidence
        excess_losses = self.selective_classifier.compute_excess_loss(
            image_features,
            text_features,
            training_model
        )
        
        # Convert excess loss to confidence
        # High excess loss → high confidence (informative)
        threshold = excess_losses.median()
        confidence = torch.sigmoid((excess_losses - threshold) / excess_losses.std())
        
        # Joint score
        joint_scores = mode_scores * confidence
        
        print(f"MODE contribution: {mode_scores.mean():.4f} ± {mode_scores.std():.4f}")
        print(f"Confidence: {confidence.mean():.4f} ± {confidence.std():.4f}")
        print(f"Joint score: {joint_scores.mean():.4f} ± {joint_scores.std():.4f}")
        
        # Select top-k
        selected_indices = torch.topk(joint_scores, k=budget).indices.tolist()
        
        return selected_indices, {
            'mode_scores': mode_scores.numpy(),
            'confidence': confidence.numpy(),
            'joint_scores': joint_scores.numpy(),
            **metadata
        }



# ============================================================================
# Example Usage
# ============================================================================

def example_complete_workflow():
    """
    Complete example showing:
    1. WebDataset loading for VLM data
    2. MODE with improved token scoring strategies
    3. Selective classification
    4. Combined approaches
    """
    
    # Setup (dummy)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # ========================================================================
    # WEBDATASET INTEGRATION
    # ========================================================================
    
    print("="*80)
    print("WEBDATASET INTEGRATION")
    print("="*80)
    
    print("Creating datasets...")
    
    # Option 1: Use predefined splits (if available)
    datasets = create_datasets()
    
    # Option 2: If only training data is available, split it manually
    if len(datasets) <= 1:  # Only train or no datasets found
        print("Using train-only approach to create splits...")
        datasets = create_train_val_test_from_train_only()
    
    # Apply preprocessing if datasets were created
    processed_datasets = {}
    if datasets:
        for split_name, dataset in datasets.items():
            processed_datasets[split_name] = add_preprocessing_pipeline(dataset, split_name)
            print(f"Applied preprocessing to {split_name} dataset")
        
        print("\nDatasets ready for MODE-VLM:")
        for split_name in processed_datasets:
            print(f"- {split_name}: {processed_datasets[split_name]}")
    else:
        print("No datasets available - using dummy data for demonstration")
        processed_datasets = None
    
    # ========================================================================
    # PART 1: MODE with Improved Token Scoring
    # ========================================================================
    
    print("\n" + "="*80)
    print("PART 1: MODE with Improved Token Scoring")
    print("="*80)
    
    # Initialize MODE selector with improved strategies
    mode_selector = MODESelector(device=device, reference_model=None)
    
    training_state = {
        'epoch': 10,
        'total_epochs': 32,
        'val_accuracy': 0.75,
        'grad_norm': 2.5,
        'remaining_budget': 0.5,
        'avg_strategy_perf': 0.02
    }
    
    # Show strategy usage guide
    print("\nStrategy Usage Guide:")
    StrategyUsageGuide.explain_strategies()
    
    # Show recommended weights for each phase
    print("\n" + "="*60)
    print("RECOMMENDED WEIGHTS BY TRAINING PHASE")
    print("="*60)
    
    for phase in ['warmup', 'early', 'middle', 'late']:
        weights = StrategyUsageGuide.get_strategy_priorities(phase)
        print(f"\n{phase.upper()} Phase:")
        for strategy, weight in weights.items():
            bar = '█' * int(weight * 40)
            print(f"  {strategy:20s} {weight:.2f} {bar}")
    
    print("\nMODE uses improved token scoring with validation rewards")
    
    # ========================================================================
    # PART 2: Selective Classification
    # ========================================================================
    
    print("\n" + "="*80)
    print("PART 2: Selective Classification (Rho-1 inspired)")
    print("="*80)
    
    print("Selective classification uses excess loss as confidence")
    
    # ========================================================================
    # PART 3: Future Work - Combined Approaches
    # ========================================================================
    
    print("\n" + "="*80)
    print("PART 3: Future Work - Combining MODE + Selective")
    print("="*80)
    
    # Approach 1: Integration
    print("\nApproach 1: Excess loss as MODE strategy")
    print("  - Add selective classification as 6th strategy")
    print("  - Meta-controller learns when to use it")
    print("  - Unified validation reward framework")
    
    # Approach 2: Two-stage
    print("\nApproach 2: Two-stage filtering")
    print("  - Stage 1: Coarse filter with excess loss")
    print("  - Stage 2: Fine selection with MODE")
    print("  - Computationally efficient")
    
    # Approach 3: Joint
    print("\nApproach 3: Joint optimization")
    print("  - S_final = S_MODE × confidence(excess_loss)")
    print("  - Soft combination")
    print("  - Preserves MODE dynamics + selective confidence")
    
    # ========================================================================
    # EXAMPLE USAGE WITH REAL DATA (if available)
    # ========================================================================
    
    if processed_datasets and 'train' in processed_datasets:
        print("\n" + "="*80)
        print("EXAMPLE WITH REAL WEBDATASET DATA")
        print("="*80)
        
        try:
            # Get a batch from training data
            train_data = processed_datasets['train']
            
            # Note: You would need to set up a tokenizer here
            # from transformers import AutoTokenizer
            # tokenizer = AutoTokenizer.from_pretrained("your-model-name")
            # wrapped_dataset = VLMDatasetWrapper(train_data, tokenizer, device)
            
            print("To use with real data:")
            print("1. Set up a tokenizer (e.g., from transformers)")
            print("2. Use VLMDatasetWrapper to convert WebDataset batches")
            print("3. Pass processed batches to MODE selector")
            print("4. Train on selected tokens/samples")
            
            print("\nExample code:")
            print("""
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
# wrapped_dataset = VLMDatasetWrapper(train_data, tokenizer, device)
# 
# for batch in wrapped_dataset:
#     if batch['input_ids'].size(0) > 0:
#         # Get model outputs (logits, hidden_states, attentions)
#         outputs = model(batch['input_ids'], output_hidden_states=True, output_attentions=True)
#         
#         # Select important tokens using MODE
#         selected_indices, metadata = mode_selector.select_batch(
#             batch['input_ids'],
#             outputs,
#             model,
#             val_loader,
#             budget=100,
#             training_state=training_state,
#             compute_gradients=True
#         )
#         
#         # Train on selected tokens
#         # ... your training logic here
#         
#         break  # Just show first batch
            """)
            
        except Exception as e:
            print(f"Error processing real data: {e}")
            print("Using dummy data instead")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("\n" + "="*80)
    print("COMPLETE FRAMEWORK SUMMARY")
    print("="*80)
    print("Key Improvements Over Original:")
    print("  - Integrated WebDataset for VLM data loading")
    print("  - Improved token scoring strategies (6 non-redundant)")
    print("  - Rho-1 style excess loss for selective classification")
    print("  - Gradient-based importance scoring")
    print("  - Semantic coherence measurement")
    print("  - Efficient diversity computation")
    print("  - Better attention focus measurement (Gini vs entropy)")
    print("  - Training phase-aware strategy recommendations")
    print("  - WebDataset wrapper for seamless integration")
    print("\nFramework ready for VLM training with MODE selection!")
    print("="*80)


def example_improved_token_scoring():
    """Example showing improved token scoring"""
    
    print("="*80)
    print("Improved Token Scoring for MODE")
    print("="*80)
    
    # Configuration
    config = ImprovedTokenMODEConfig(
        n_strategies=6,
        diversity_buffer_size=1000,
        enable_gradient_scores=True,
        use_reference_model=True
    )
    
    # Initialize scorer
    scorer = ImprovedTokenScorer(config, reference_model=None)
    
    # Print strategy explanations
    print("\n" + "="*80)
    print("STRATEGY USAGE GUIDE")
    print("="*80)
    StrategyUsageGuide.explain_strategies()
    
    # Show recommended weights for each phase
    print("\n" + "="*80)
    print("RECOMMENDED WEIGHTS BY TRAINING PHASE")
    print("="*80)
    
    for phase in ['warmup', 'early', 'middle', 'late']:
        weights = StrategyUsageGuide.get_strategy_priorities(phase)
        print(f"\n{phase.upper()} Phase:")
        for strategy, weight in weights.items():
            bar = '█' * int(weight * 40)
            print(f"  {strategy:20s} {weight:.2f} {bar}")
    
    print("\n" + "="*80)
    print("Key Improvements Over Original:")
    print("="*80)
    print("- Removed redundancy (entropy vs coherence, loss vs perplexity)")
    print("- Added Rho-1 style excess loss")
    print("- Added gradient-based importance")
    print("- Added semantic coherence (contextual)")
    print("- More efficient diversity computation")
    print("- Better attention measurement (Gini vs entropy)")
    print("="*80)


if __name__ == '__main__':
    # Run both examples
    example_improved_token_scoring()
    print("\n\n")
    example_complete_workflow()
