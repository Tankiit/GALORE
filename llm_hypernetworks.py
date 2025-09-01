"""
LLM-Specific Hypernetwork Framework for Coreset Selection
=========================================================

This module implements specialized scoring functions and hypernetwork architectures
optimized for Large Language Model training with efficient data selection.

Key Features:
1. Token-level and sequence-level scoring functions
2. Attention-aware diversity metrics
3. Perplexity-based uncertainty scoring
4. Gradient accumulation for large models
5. Memory-efficient implementations for LLMs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
import logging
from abc import ABC, abstractmethod
import math

logger = logging.getLogger(__name__)

# =============================================================================
# LLM-Specific Scoring Functions
# =============================================================================

class LLMScoringFunction(ABC):
    """Abstract base class for LLM-specific scoring functions"""
    
    def __init__(self, name: str, model: nn.Module, tokenizer=None, device='cuda'):
        self.name = name
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.cache = {}
        self.call_count = 0
        
    @abstractmethod
    def score(self, idx: int, text_data: Any, selected_set: List[int], 
              context: Dict[str, Any]) -> float:
        """Compute score for text data given selected set and context"""
        pass
    
    def reset_cache(self):
        """Clear scoring cache"""
        self.cache.clear()
        self.call_count = 0


class PerplexityScoring(LLMScoringFunction):
    """Score based on model perplexity - higher perplexity indicates more learning potential"""
    
    def __init__(self, model: nn.Module, tokenizer, device='cuda'):
        super().__init__("perplexity", model, tokenizer, device)
        
    def score(self, idx: int, text_data: Any, selected_set: List[int], 
              context: Dict[str, Any]) -> float:
        self.call_count += 1
        
        # Cache check
        cache_key = (idx, len(selected_set))
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Tokenize text
        if isinstance(text_data, str):
            inputs = self.tokenizer(text_data, return_tensors="pt", 
                                   truncation=True, max_length=512)
        else:
            inputs = text_data  # Assume pre-tokenized
            
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Compute perplexity
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs['input_ids'])
            perplexity = torch.exp(outputs.loss).item()
        
        # Normalize and apply diminishing returns
        normalized_perplexity = min(perplexity / 100.0, 10.0)  # Cap at 10
        diminishing_factor = 1.0 / (1.0 + 0.05 * len(selected_set))
        score = normalized_perplexity * diminishing_factor
        
        self.cache[cache_key] = score
        return score


class TokenDiversityScoring(LLMScoringFunction):
    """Score based on token-level diversity using embedding space"""
    
    def __init__(self, model: nn.Module, tokenizer, device='cuda', embedding_dim=768):
        super().__init__("token_diversity", model, tokenizer, device)
        self.embedding_dim = embedding_dim
        
    def score(self, idx: int, text_data: Any, selected_set: List[int], 
              context: Dict[str, Any]) -> float:
        self.call_count += 1
        
        if len(selected_set) == 0:
            return 1.0
        
        # Get embeddings for current text
        if isinstance(text_data, str):
            inputs = self.tokenizer(text_data, return_tensors="pt", 
                                   truncation=True, max_length=512)
        else:
            inputs = text_data
            
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Get hidden states from model
            outputs = self.model(**inputs, output_hidden_states=True)
            # Use mean pooling of last hidden states
            hidden_states = outputs.hidden_states[-1]
            text_embedding = hidden_states.mean(dim=1)  # [batch_size, hidden_dim]
        
        # Compute minimum distance to selected set embeddings
        selected_embeddings = context.get('selected_embeddings', {})
        min_dist = float('inf')
        
        for j in selected_set[-100:]:  # Check last 100 for efficiency
            if j in selected_embeddings:
                emb_j = selected_embeddings[j]
                dist = torch.norm(text_embedding - emb_j).item()
                min_dist = min(min_dist, dist)
        
        # Normalize distance
        normalized_dist = min_dist / (self.embedding_dim ** 0.5)
        return min(normalized_dist, 1.0)


class AttentionCoverageScoring(LLMScoringFunction):
    """Score based on attention pattern coverage - prioritize diverse attention patterns"""
    
    def __init__(self, model: nn.Module, tokenizer, device='cuda'):
        super().__init__("attention_coverage", model, tokenizer, device)
        
    def score(self, idx: int, text_data: Any, selected_set: List[int], 
              context: Dict[str, Any]) -> float:
        self.call_count += 1
        
        # Tokenize text
        if isinstance(text_data, str):
            inputs = self.tokenizer(text_data, return_tensors="pt", 
                                   truncation=True, max_length=512)
        else:
            inputs = text_data
            
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Get attention weights
            outputs = self.model(**inputs, output_attentions=True)
            attentions = outputs.attentions  # List of attention tensors
            
            # Average attention across all layers and heads
            avg_attention = torch.stack(attentions).mean(dim=(0, 1, 2))  # [seq_len, seq_len]
            
            # Compute attention entropy as diversity measure
            attention_probs = F.softmax(avg_attention.flatten(), dim=0)
            entropy = -(attention_probs * torch.log(attention_probs + 1e-8)).sum().item()
        
        # Higher entropy = more diverse attention = higher score
        normalized_entropy = entropy / np.log(avg_attention.numel())
        return normalized_entropy


class GradientAlignmentScoring(LLMScoringFunction):
    """Score based on gradient alignment with current mini-batch"""
    
    def __init__(self, model: nn.Module, tokenizer, device='cuda'):
        super().__init__("gradient_alignment", model, tokenizer, device)
        
    def score(self, idx: int, text_data: Any, selected_set: List[int], 
              context: Dict[str, Any]) -> float:
        self.call_count += 1
        
        # Get current batch gradient if available
        batch_grad = context.get('batch_gradient', None)
        if batch_grad is None:
            return 0.5  # Neutral score if no batch gradient
        
        # Tokenize text
        if isinstance(text_data, str):
            inputs = self.tokenizer(text_data, return_tensors="pt", 
                                   truncation=True, max_length=512)
        else:
            inputs = text_data
            
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Compute gradient for this sample
        self.model.zero_grad()
        outputs = self.model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss
        loss.backward()
        
        # Compute gradient alignment
        alignment = 0.0
        num_params = 0
        
        for (name, param), batch_g in zip(self.model.named_parameters(), batch_grad):
            if param.grad is not None:
                # Cosine similarity between gradients
                cos_sim = F.cosine_similarity(
                    param.grad.flatten().unsqueeze(0),
                    batch_g.flatten().unsqueeze(0)
                ).item()
                alignment += cos_sim
                num_params += 1
        
        if num_params > 0:
            alignment /= num_params
        
        # Convert alignment to score (higher alignment = higher score)
        score = (alignment + 1.0) / 2.0  # Normalize to [0, 1]
        return score


class TokenImportanceScoring(LLMScoringFunction):
    """Score based on token-level importance using gradient norms"""
    
    def __init__(self, model: nn.Module, tokenizer, device='cuda'):
        super().__init__("token_importance", model, tokenizer, device)
        
    def score(self, idx: int, text_data: Any, selected_set: List[int], 
              context: Dict[str, Any]) -> float:
        self.call_count += 1
        
        # Tokenize text
        if isinstance(text_data, str):
            inputs = self.tokenizer(text_data, return_tensors="pt", 
                                   truncation=True, max_length=512)
        else:
            inputs = text_data
            
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Enable gradient computation for embeddings
        self.model.zero_grad()
        inputs_embeds = self.model.get_input_embeddings()(inputs['input_ids'])
        inputs_embeds.requires_grad = True
        
        # Forward pass with embedded inputs
        outputs = self.model(inputs_embeds=inputs_embeds, labels=inputs['input_ids'])
        loss = outputs.loss
        
        # Compute gradient w.r.t. input embeddings
        grad_embeds = torch.autograd.grad(loss, inputs_embeds, retain_graph=False)[0]
        
        # Compute importance as L2 norm of gradients per token
        token_importance = grad_embeds.norm(dim=-1).mean().item()
        
        # Normalize and return
        normalized_importance = min(token_importance / 10.0, 1.0)
        return normalized_importance


class RepetitionPenaltyScoring(LLMScoringFunction):
    """Score that penalizes repetitive patterns in selected data"""
    
    def __init__(self, model: nn.Module, tokenizer, device='cuda'):
        super().__init__("repetition_penalty", model, tokenizer, device)
        self.selected_ngrams = defaultdict(int)
        
    def score(self, idx: int, text_data: Any, selected_set: List[int], 
              context: Dict[str, Any]) -> float:
        self.call_count += 1
        
        # Tokenize text
        if isinstance(text_data, str):
            tokens = self.tokenizer.encode(text_data, truncation=True, max_length=512)
        else:
            tokens = text_data.get('input_ids', [])[0].tolist()
        
        # Extract n-grams (3-grams)
        n = 3
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        
        # Count overlap with previously selected n-grams
        overlap_count = 0
        for ngram in ngrams:
            overlap_count += self.selected_ngrams[ngram]
        
        # Compute novelty score (inverse of overlap)
        if len(ngrams) > 0:
            novelty = 1.0 - min(overlap_count / len(ngrams), 1.0)
        else:
            novelty = 0.5
        
        # Update n-gram counts if this sample is selected
        if context.get('update_ngrams', False):
            for ngram in ngrams:
                self.selected_ngrams[ngram] += 1
        
        return novelty


# =============================================================================
# LLM-Optimized Hypernetwork Architecture
# =============================================================================

@dataclass
class LLMTrainingState:
    """Training state specifically for LLM training"""
    epoch: int
    loss: float
    perplexity: float
    gradient_norm: float
    learning_rate: float
    tokens_seen: int
    total_tokens: int
    avg_sequence_length: float
    vocab_coverage: float  # Percentage of vocabulary seen
    performance_history: List[float]
    attention_entropy: float  # Diversity of attention patterns
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to tensor for hypernetwork input"""
        features = [
            self.epoch / 100.0,  # Normalized epoch
            np.log(self.loss + 1e-8),  # Log loss
            np.log(self.perplexity + 1e-8) / 10.0,  # Normalized log perplexity
            np.log(self.gradient_norm + 1e-8),
            np.log10(self.learning_rate + 1e-10),
            self.tokens_seen / self.total_tokens,  # Progress ratio
            self.avg_sequence_length / 512.0,  # Normalized by max length
            self.vocab_coverage,
            self.attention_entropy
        ]
        
        # Performance trend (last 5 epochs)
        if len(self.performance_history) >= 5:
            recent = self.performance_history[-5:]
            for val in recent:
                features.append(val)
            trend = (recent[-1] - recent[0]) / (abs(recent[0]) + 1e-8)
            features.append(trend)
        else:
            # Pad with zeros if not enough history
            features.extend([0.0] * 6)
        
        return torch.tensor(features[:15], dtype=torch.float32)  # Ensure exactly 15 features


class LLMMultiScoringHypernetwork(nn.Module):
    """
    Hypernetwork optimized for LLM training dynamics
    """
    
    def __init__(self,
                 scoring_functions: List[LLMScoringFunction],
                 state_dim: int = 15,
                 hidden_dim: int = 128,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        self.scoring_functions = scoring_functions
        self.num_functions = len(scoring_functions)
        self.function_names = [sf.name for sf in scoring_functions]
        
        # State encoder with layer normalization for stability
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Multi-head self-attention for capturing dependencies
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Function-specific encoders
        self.function_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(self.num_functions)
        ])
        
        # Global weight generator
        self.weight_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, self.num_functions)
        )
        
        # Temperature controller for exploration
        self.temperature_controller = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Value function for RL
        self.value_function = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/He initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)
                nn.init.xavier_uniform_(module.out_proj.weight)
    
    def forward(self, state: LLMTrainingState) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate scoring function weights based on LLM training state
        
        Returns:
            weights: Weights for each scoring function
            temperature: Temperature for exploration
            value: State value estimate
        """
        state_tensor = state.to_tensor().unsqueeze(0)  # [1, state_dim]
        
        # Encode state
        encoded = self.state_encoder(state_tensor)  # [1, hidden_dim]
        
        # Self-attention (treating state as sequence of length 1)
        attended, _ = self.self_attention(encoded, encoded, encoded)
        
        # Residual connection
        encoded = encoded + attended
        
        # Generate function-specific scores
        function_scores = []
        for encoder in self.function_encoders:
            score = encoder(encoded)
            function_scores.append(score)
        
        function_scores = torch.cat(function_scores, dim=-1)  # [1, num_functions]
        
        # Generate global weights
        global_weights = self.weight_generator(encoded)  # [1, num_functions]
        
        # Combine function-specific and global weights
        combined_weights = 0.5 * F.softmax(function_scores, dim=-1) + \
                          0.5 * F.softmax(global_weights, dim=-1)
        
        # Generate temperature
        temperature = self.temperature_controller(encoded)
        temperature = 0.1 + 1.9 * temperature  # Scale to [0.1, 2.0]
        
        # Compute value
        value = self.value_function(encoded)
        
        return combined_weights.squeeze(0), temperature.squeeze(), value.squeeze()


# =============================================================================
# LLM-Specific Coreset Selector
# =============================================================================

class LLMCoresetSelector:
    """
    Coreset selector optimized for LLM training with large-scale text data
    """
    
    def __init__(self,
                 hypernetwork: LLMMultiScoringHypernetwork,
                 scoring_functions: List[LLMScoringFunction],
                 batch_size: int = 8,
                 gradient_accumulation_steps: int = 4,
                 max_sequence_length: int = 512):
        
        self.hypernetwork = hypernetwork
        self.scoring_functions = scoring_functions
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_sequence_length = max_sequence_length
        
        # Performance tracking
        self.selection_history = []
        self.weight_history = []
        self.perplexity_history = []
        
    def select_coreset(self,
                      dataset: Any,
                      budget: int,
                      training_state: LLMTrainingState,
                      tokenizer: Any,
                      verbose: bool = True) -> Tuple[List[int], Dict[str, Any]]:
        """
        Select coreset for LLM training
        """
        # Get weights from hypernetwork
        with torch.no_grad():
            weights, temperature, value = self.hypernetwork(training_state)
        
        if verbose:
            logger.info(f"LLM Hypernetwork weights: {dict(zip(self.hypernetwork.function_names, weights.numpy()))}")
            logger.info(f"Temperature: {temperature.item():.3f}, Value: {value.item():.3f}")
        
        # Store weights for analysis
        self.weight_history.append({
            name: weights[i].item() 
            for i, name in enumerate(self.hypernetwork.function_names)
        })
        
        # Greedy selection with lazy evaluation
        selected_indices = []
        n = len(dataset)
        context = {
            'selected_embeddings': {},
            'update_ngrams': True
        }
        
        # Use priority queue for efficient selection
        from heapq import heappush, heappop
        candidate_heap = []
        
        # Initial scoring of random candidates
        initial_candidates = np.random.choice(n, min(n, budget * 5), replace=False)
        
        for idx in initial_candidates:
            if len(selected_indices) >= budget:
                break
                
            # Compute weighted score
            text_data = dataset[idx]
            combined_score = 0.0
            
            for i, scoring_fn in enumerate(self.scoring_functions):
                score = scoring_fn.score(idx, text_data, selected_indices, context)
                combined_score += weights[i].item() * score
            
            heappush(candidate_heap, (-combined_score, idx))
        
        # Select top candidates
        while len(selected_indices) < budget and candidate_heap:
            neg_score, idx = heappop(candidate_heap)
            if idx not in selected_indices:
                selected_indices.append(idx)
                
                if verbose and len(selected_indices) % 100 == 0:
                    logger.info(f"Selected {len(selected_indices)}/{budget} samples")
        
        # Compile selection info
        selection_info = {
            'weights': self.weight_history[-1],
            'temperature': temperature.item(),
            'value': value.item(),
            'num_selected': len(selected_indices),
            'avg_score': -neg_score if candidate_heap else 0.0
        }
        
        return selected_indices, selection_info
    
    def update_with_feedback(self,
                            previous_state: LLMTrainingState,
                            current_state: LLMTrainingState,
                            performance_delta: float,
                            optimizer: torch.optim.Optimizer) -> float:
        """
        Update hypernetwork based on training feedback
        """
        # Compute reward based on perplexity improvement
        perplexity_improvement = (previous_state.perplexity - current_state.perplexity) / \
                                 (previous_state.perplexity + 1e-8)
        reward = performance_delta + 0.5 * perplexity_improvement
        
        # Forward pass for both states
        prev_weights, prev_temp, prev_value = self.hypernetwork(previous_state)
        curr_weights, curr_temp, curr_value = self.hypernetwork(current_state)
        
        # Compute TD error
        target_value = reward + 0.95 * curr_value.detach()
        value_loss = F.mse_loss(prev_value, target_value)
        
        # Policy gradient loss
        log_probs = torch.log(prev_weights + 1e-8)
        policy_loss = -torch.sum(log_probs * prev_weights.detach()) * reward
        
        # Total loss with entropy regularization
        entropy = -(prev_weights * torch.log(prev_weights + 1e-8)).sum()
        total_loss = value_loss + 0.1 * policy_loss - 0.01 * entropy
        
        # Update
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.hypernetwork.parameters(), 1.0)
        optimizer.step()
        
        return total_loss.item()


# =============================================================================
# Helper Functions for LLM Integration
# =============================================================================

def create_llm_scoring_functions(model: nn.Module, tokenizer: Any, device='cuda') -> List[LLMScoringFunction]:
    """Create comprehensive set of LLM scoring functions"""
    return [
        PerplexityScoring(model, tokenizer, device),
        TokenDiversityScoring(model, tokenizer, device),
        AttentionCoverageScoring(model, tokenizer, device),
        GradientAlignmentScoring(model, tokenizer, device),
        TokenImportanceScoring(model, tokenizer, device),
        RepetitionPenaltyScoring(model, tokenizer, device)
    ]


def create_llm_hypernetwork(model: nn.Module, tokenizer: Any, device='cuda') -> Tuple[LLMMultiScoringHypernetwork, LLMCoresetSelector]:
    """Create and initialize LLM hypernetwork and selector"""
    
    # Create scoring functions
    scoring_functions = create_llm_scoring_functions(model, tokenizer, device)
    
    # Create hypernetwork
    hypernetwork = LLMMultiScoringHypernetwork(
        scoring_functions=scoring_functions,
        state_dim=15,
        hidden_dim=128,
        num_heads=4,
        dropout=0.1
    ).to(device)
    
    # Create selector
    selector = LLMCoresetSelector(
        hypernetwork=hypernetwork,
        scoring_functions=scoring_functions,
        batch_size=8,
        gradient_accumulation_steps=4,
        max_sequence_length=512
    )
    
    return hypernetwork, selector


if __name__ == "__main__":
    print("LLM Hypernetwork Framework for Coreset Selection")
    print("=" * 60)
    print("This module provides:")
    print("- 6 LLM-specific scoring functions")
    print("- Optimized hypernetwork architecture for LLMs")
    print("- Efficient coreset selection for large-scale text data")
    print("- Integration with gradient accumulation for large models")