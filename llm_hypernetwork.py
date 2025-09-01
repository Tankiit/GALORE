"""
Hypernetwork-Based Multi-Scoring Function Framework for LLM Data Selection
==========================================================================

This framework adapts the multi-scoring hypernetwork approach for Large Language Models,
integrating GaLore gradient compression with intelligent text data selection.

Key innovations:
1. LLM-specific scoring functions (perplexity, semantic diversity, quality, difficulty)
2. Hypernetwork generates dynamic weights based on training state and data characteristics
3. GaLore integration for memory-efficient gradient processing
4. Token-level and document-level selection strategies
5. Domain-aware and curriculum learning capabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
from collections import deque, defaultdict
from dataclasses import dataclass
from enum import Enum
import logging
from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModel
import json
import re
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# =============================================================================
# LLM-Specific Scoring Functions
# =============================================================================

class LLMScoringFunction(ABC):
    """Abstract base class for LLM-specific scoring functions"""
    
    def __init__(self, name: str, submodular: bool = True, granularity: str = "document"):
        self.name = name
        self.submodular = submodular
        self.granularity = granularity  # "document", "paragraph", "token"
        self.call_count = 0
        self.cache = {}
        
    @abstractmethod
    def score(self, idx: int, text_data: Dict[str, Any], selected_set: List[int], 
              context: Dict[str, Any]) -> float:
        """Compute score for text data given selected set and context"""
        pass
    
    def reset_cache(self):
        """Clear scoring cache"""
        self.cache.clear()
        self.call_count = 0


class PerplexityScoring(LLMScoringFunction):
    """Score based on model perplexity - measures learning difficulty"""
    
    def __init__(self, model: nn.Module, tokenizer, device='cuda', max_length=512):
        super().__init__("perplexity", submodular=True)
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        
    def score(self, idx: int, text_data: Dict[str, Any], selected_set: List[int], 
              context: Dict[str, Any]) -> float:
        self.call_count += 1
        
        # Cache check
        cache_key = (idx, len(selected_set))
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        text = text_data.get('text', '')
        if not text:
            return 0.0
        
        # Tokenize
        tokens = self.tokenizer.encode(text, max_length=self.max_length, truncation=True, return_tensors='pt')
        tokens = tokens.to(self.device)
        
        # Compute perplexity
        with torch.no_grad():
            outputs = self.model(tokens, labels=tokens)
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
        
        # Submodular diminishing returns
        diminishing_factor = 1.0 / (1.0 + 0.01 * len(selected_set))
        score = perplexity * diminishing_factor
        
        self.cache[cache_key] = score
        return score


class SemanticDiversityScoring(LLMScoringFunction):
    """Score based on semantic diversity using embeddings"""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        super().__init__("semantic_diversity", submodular=True)
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_cache = {}
        
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get cached embedding for text"""
        if text not in self.embedding_cache:
            embedding = self.embedding_model.encode([text])[0]
            self.embedding_cache[text] = embedding
        return self.embedding_cache[text]
    
    def score(self, idx: int, text_data: Dict[str, Any], selected_set: List[int], 
              context: Dict[str, Any]) -> float:
        self.call_count += 1
        
        if len(selected_set) == 0:
            return 1.0  # First document has maximum diversity
        
        text = text_data.get('text', '')
        if not text:
            return 0.0
        
        # Get embedding for current text
        current_embedding = self._get_embedding(text)
        
        # Compute minimum distance to selected set
        min_distance = float('inf')
        selected_embeddings = context.get('selected_embeddings', {})
        
        for j in selected_set:
            if j in selected_embeddings:
                other_embedding = selected_embeddings[j]
                # Cosine distance
                similarity = np.dot(current_embedding, other_embedding) / (
                    np.linalg.norm(current_embedding) * np.linalg.norm(other_embedding)
                )
                distance = 1.0 - similarity
                min_distance = min(min_distance, distance)
        
        return min_distance


class QualityScoring(LLMScoringFunction):
    """Score based on text quality metrics"""
    
    def __init__(self):
        super().__init__("quality", submodular=False)
        
    def _compute_quality_metrics(self, text: str) -> Dict[str, float]:
        """Compute various text quality metrics"""
        if not text:
            return {'readability': 0.0, 'coherence': 0.0, 'informativeness': 0.0}
        
        # Simple quality metrics (can be enhanced with more sophisticated measures)
        words = text.split()
        sentences = text.split('.')
        
        # Readability (average sentence length)
        avg_sentence_length = len(words) / max(len(sentences), 1)
        readability = min(avg_sentence_length / 20.0, 1.0)  # Normalize
        
        # Coherence (simple measure based on word repetition)
        unique_words = len(set(words))
        coherence = unique_words / max(len(words), 1)
        
        # Informativeness (entropy of word distribution)
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        total_words = len(words)
        entropy = 0.0
        for count in word_counts.values():
            prob = count / total_words
            entropy -= prob * np.log(prob + 1e-8)
        
        informativeness = min(entropy / 10.0, 1.0)  # Normalize
        
        return {
            'readability': readability,
            'coherence': coherence,
            'informativeness': informativeness
        }
    
    def score(self, idx: int, text_data: Dict[str, Any], selected_set: List[int], 
              context: Dict[str, Any]) -> float:
        self.call_count += 1
        
        text = text_data.get('text', '')
        quality_metrics = self._compute_quality_metrics(text)
        
        # Weighted combination of quality metrics
        quality_score = (
            0.3 * quality_metrics['readability'] +
            0.3 * quality_metrics['coherence'] +
            0.4 * quality_metrics['informativeness']
        )
        
        return quality_score


class DifficultyScoring(LLMScoringFunction):
    """Score based on learning difficulty using gradient norms"""
    
    def __init__(self, model: nn.Module, tokenizer, device='cuda', max_length=512):
        super().__init__("difficulty", submodular=True)
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        
    def score(self, idx: int, text_data: Dict[str, Any], selected_set: List[int], 
              context: Dict[str, Any]) -> float:
        self.call_count += 1
        
        text = text_data.get('text', '')
        if not text:
            return 0.0
        
        # Tokenize
        tokens = self.tokenizer.encode(text, max_length=self.max_length, truncation=True, return_tensors='pt')
        tokens = tokens.to(self.device)
        
        # Compute gradient norm
        self.model.zero_grad()
        outputs = self.model(tokens, labels=tokens)
        loss = outputs.loss
        loss.backward()
        
        # Aggregate gradient norms
        grad_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.norm().item() ** 2
        grad_norm = grad_norm ** 0.5
        
        # Submodular diminishing returns
        diminishing_factor = 1.0 / (1.0 + 0.01 * len(selected_set))
        score = grad_norm * diminishing_factor
        
        return score


class DomainRelevanceScoring(LLMScoringFunction):
    """Score based on domain relevance for specialized tasks"""
    
    def __init__(self, target_domains: List[str] = None, domain_keywords: Dict[str, List[str]] = None):
        super().__init__("domain_relevance", submodular=False)
        self.target_domains = target_domains or ['code', 'science', 'dialogue']
        self.domain_keywords = domain_keywords or {
            'code': ['function', 'class', 'import', 'def', 'return', 'if', 'for', 'while'],
            'science': ['research', 'study', 'analysis', 'hypothesis', 'experiment', 'data'],
            'dialogue': ['said', 'asked', 'replied', 'conversation', 'discuss', 'talk']
        }
        
    def _detect_domain(self, text: str) -> Dict[str, float]:
        """Detect domain relevance scores"""
        text_lower = text.lower()
        domain_scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            keyword_count = sum(1 for keyword in keywords if keyword in text_lower)
            domain_scores[domain] = keyword_count / len(keywords)
        
        return domain_scores
    
    def score(self, idx: int, text_data: Dict[str, Any], selected_set: List[int], 
              context: Dict[str, Any]) -> float:
        self.call_count += 1
        
        text = text_data.get('text', '')
        domain_scores = self._detect_domain(text)
        
        # Get target domain preference from context
        target_domain = context.get('target_domain', 'code')
        
        return domain_scores.get(target_domain, 0.0)


class CurriculumScoring(LLMScoringFunction):
    """Score based on curriculum learning principles"""
    
    def __init__(self):
        super().__init__("curriculum", submodular=False)
        
    def score(self, idx: int, text_data: Dict[str, Any], selected_set: List[int], 
              context: Dict[str, Any]) -> float:
        self.call_count += 1
        
        # Get training progress from context
        training_progress = context.get('training_progress', 0.0)  # 0.0 to 1.0
        
        text = text_data.get('text', '')
        if not text:
            return 0.0
        
        # Simple curriculum: start with shorter, simpler texts
        text_length = len(text.split())
        
        if training_progress < 0.3:
            # Early training: prefer shorter texts
            length_score = max(0, 1.0 - text_length / 200)
        elif training_progress < 0.7:
            # Mid training: prefer medium-length texts
            optimal_length = 300
            length_score = 1.0 - abs(text_length - optimal_length) / optimal_length
        else:
            # Late training: prefer longer, complex texts
            length_score = min(text_length / 500, 1.0)
        
        return max(0, length_score)


# =============================================================================
# LLM Training State Representation
# =============================================================================

@dataclass
class LLMTrainingState:
    """Encodes current LLM training state for hypernetwork input"""
    epoch: int
    train_loss: float
    val_loss: float
    perplexity: float
    learning_rate: float
    tokens_seen: int
    total_tokens: int
    gradient_norm: float
    domain_distribution: Dict[str, float]
    length_distribution: Dict[str, float]  # Short, medium, long text ratios
    performance_history: List[float]
    selection_diversity: float
    memory_usage: float
    
    def to_tensor(self) -> torch.Tensor:
        """Convert state to tensor for hypernetwork input"""
        features = [
            self.epoch / 1000.0,  # Normalized epoch
            self.train_loss,
            self.val_loss,
            np.log(self.perplexity + 1e-8),
            np.log10(self.learning_rate + 1e-10),
            self.tokens_seen / max(self.total_tokens, 1),  # Progress ratio
            self.gradient_norm,
            self.selection_diversity,
            self.memory_usage
        ]
        
        # Domain distribution features
        domains = ['code', 'science', 'dialogue', 'general']
        for domain in domains:
            features.append(self.domain_distribution.get(domain, 0.0))
        
        # Length distribution features
        length_categories = ['short', 'medium', 'long']
        for category in length_categories:
            features.append(self.length_distribution.get(category, 0.0))
        
        # Performance trend
        if len(self.performance_history) >= 3:
            recent = self.performance_history[-3:]
            trend = (recent[-1] - recent[0]) / 3.0
            features.append(trend)
        else:
            features.append(0.0)
            
        return torch.tensor(features, dtype=torch.float32)


# =============================================================================
# LLM Hypernetwork Architecture
# =============================================================================

class LLMMultiScoringHypernetwork(nn.Module):
    """
    Hypernetwork for LLM data selection with domain-aware and curriculum learning
    """
    
    def __init__(self, 
                 scoring_functions: List[LLMScoringFunction],
                 state_dim: int = 17,  # Adjusted for LLMTrainingState
                 hidden_dim: int = 256,
                 attention_heads: int = 8,
                 num_domains: int = 4):
        super().__init__()
        
        self.scoring_functions = scoring_functions
        self.num_functions = len(scoring_functions)
        self.function_names = [sf.name for sf in scoring_functions]
        self.num_domains = num_domains
        
        # Enhanced state encoder for LLM context
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Multi-head attention with larger capacity for LLM complexity
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=attention_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Domain-aware weight generator
        self.domain_encoder = nn.Sequential(
            nn.Linear(num_domains, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )
        
        # Combined weight generator
        self.weight_generator = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, self.num_functions),
            nn.Softmax(dim=-1)
        )
        
        # Curriculum learning controller
        self.curriculum_controller = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 3),  # Easy, medium, hard difficulty preferences
            nn.Softmax(dim=-1)
        )
        
        # Enhanced temperature controller
        self.temperature_controller = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Value function for RL
        self.value_function = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Performance predictor for each function
        self.performance_predictor = nn.Linear(hidden_dim, self.num_functions)
        
        # Domain adaptation predictor
        self.domain_adapter = nn.Linear(hidden_dim, num_domains)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with careful scaling for LLM context"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)
                nn.init.xavier_uniform_(module.out_proj.weight)
    
    def forward(self, state: LLMTrainingState, domain_context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for LLM hypernetwork
        
        Returns:
            weights: Weights for each scoring function
            curriculum_prefs: Curriculum difficulty preferences
            temperature: Temperature for exploration
            value: State value estimate
            performance_pred: Predicted performance for each function
            domain_prefs: Domain adaptation preferences
        """
        state_tensor = state.to_tensor().unsqueeze(0)  # Add batch dimension
        
        # Encode state
        encoded_state = self.state_encoder(state_tensor)
        
        # Self-attention for context awareness
        attended_state, attention_weights = self.attention(encoded_state, encoded_state, encoded_state)
        attended_state = attended_state.squeeze(0)  # Remove batch dimension
        
        # Domain encoding
        if domain_context is not None:
            domain_encoded = self.domain_encoder(domain_context)
            # Combine state and domain information
            combined_state = torch.cat([attended_state, domain_encoded], dim=-1)
        else:
            # Default domain context (uniform distribution)
            default_domain = torch.ones(self.num_domains) / self.num_domains
            domain_encoded = self.domain_encoder(default_domain)
            combined_state = torch.cat([attended_state, domain_encoded], dim=-1)
        
        # Generate outputs
        weights = self.weight_generator(combined_state)
        curriculum_prefs = self.curriculum_controller(attended_state)
        
        temperature = self.temperature_controller(attended_state)
        temperature = 0.05 + 1.95 * temperature  # Scale to [0.05, 2.0] for LLM context
        
        value = self.value_function(attended_state)
        performance_pred = self.performance_predictor(attended_state)
        domain_prefs = F.softmax(self.domain_adapter(attended_state), dim=-1)
        
        return (weights.squeeze(), curriculum_prefs.squeeze(), temperature.squeeze(), 
                value.squeeze(), performance_pred.squeeze(), domain_prefs.squeeze())


# =============================================================================
# GaLore Integration for Memory-Efficient LLM Training
# =============================================================================

class GaLoreOptimizer:
    """GaLore optimizer integration for memory-efficient LLM training"""
    
    def __init__(self, model_parameters, rank: int = 256, update_proj_gap: int = 200, lr: float = 1e-4):
        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.step_count = 0
        self.projectors = {}
        
        # Base optimizer
        self.base_optimizer = torch.optim.AdamW(model_parameters, lr=lr, weight_decay=0.01)
        
    def project_gradients(self, model: nn.Module):
        """Apply GaLore projection to gradients"""
        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.dim() >= 2:
                # Apply low-rank projection
                grad = param.grad.data
                
                # Update projector periodically
                if name not in self.projectors or self.step_count % self.update_proj_gap == 0:
                    if grad.numel() >= self.rank * 2:
                        U, S, V = torch.svd_lowrank(grad, q=min(self.rank, min(grad.shape)))
                        self.projectors[name] = (U.detach(), V.detach())
                
                # Project gradient if projector exists
                if name in self.projectors:
                    U, V = self.projectors[name]
                    # Project gradient: G_projected = U @ U^T @ G @ V @ V^T
                    projected_grad = U @ (U.T @ grad @ V) @ V.T
                    param.grad.data = projected_grad
    
    def step(self, model: nn.Module):
        """Optimizer step with GaLore projection"""
        self.project_gradients(model)
        self.base_optimizer.step()
        self.step_count += 1
    
    def zero_grad(self):
        """Zero gradients"""
        self.base_optimizer.zero_grad()


# =============================================================================
# LLM Multi-Scoring Selector
# =============================================================================

class LLMMultiScoringSelector:
    """
    LLM-specific multi-scoring selector with domain awareness and curriculum learning
    """
    
    def __init__(self, 
                 hypernetwork: LLMMultiScoringHypernetwork,
                 scoring_functions: List[LLMScoringFunction],
                 tokenizer,
                 max_sequence_length: int = 512,
                 lazy_evaluation: bool = True):
        
        self.hypernetwork = hypernetwork
        self.scoring_functions = scoring_functions
        self.function_dict = {sf.name: sf for sf in scoring_functions}
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.lazy_evaluation = lazy_evaluation
        
        # Performance tracking
        self.selection_history = []
        self.weight_history = []
        self.domain_history = []
        self.curriculum_history = []
        
        # Domain and curriculum state
        self.current_domain_focus = None
        self.current_difficulty_level = 'medium'
        
    def compute_combined_score(self, 
                             idx: int, 
                             text_data: Dict[str, Any], 
                             selected_set: List[int],
                             context: Dict[str, Any],
                             weights: torch.Tensor,
                             curriculum_prefs: torch.Tensor) -> Tuple[float, Dict[str, float]]:
        """
        Compute weighted combination of scoring functions with curriculum weighting
        """
        individual_scores = {}
        
        # Compute individual scores
        for i, scoring_fn in enumerate(self.scoring_functions):
            score = scoring_fn.score(idx, text_data, selected_set, context)
            individual_scores[scoring_fn.name] = score
        
        # Apply hypernetwork weights
        combined_score = 0.0
        for i, (name, score) in enumerate(individual_scores.items()):
            weight = weights[i].item()
            
            # Apply curriculum weighting for difficulty-based functions
            if name == 'difficulty':
                # Get text difficulty level
                text_length = len(text_data.get('text', '').split())
                if text_length < 100:
                    difficulty_level = 0  # Easy
                elif text_length < 300:
                    difficulty_level = 1  # Medium
                else:
                    difficulty_level = 2  # Hard
                
                # Apply curriculum preference
                curriculum_weight = curriculum_prefs[difficulty_level].item()
                weight *= curriculum_weight
            
            combined_score += weight * score
        
        return combined_score, individual_scores
    
    def select_coreset_adaptive(self,
                              dataset: List[Dict[str, Any]],
                              budget: int,
                              training_state: LLMTrainingState,
                              target_domain: str = 'general',
                              context: Dict[str, Any] = None,
                              verbose: bool = True) -> Tuple[List[int], Dict[str, Any]]:
        """
        Adaptive coreset selection for LLM training data
        """
        if context is None:
            context = {'target_domain': target_domain}
        
        # Domain context tensor
        domains = ['code', 'science', 'dialogue', 'general']
        domain_context = torch.zeros(len(domains))
        if target_domain in domains:
            domain_context[domains.index(target_domain)] = 1.0
        else:
            domain_context[-1] = 1.0  # Default to 'general'
        
        # Get hypernetwork outputs
        with torch.no_grad():
            weights, curriculum_prefs, temperature, value, performance_pred, domain_prefs = \
                self.hypernetwork(training_state, domain_context)
        
        # Store for analysis
        self.weight_history.append({
            name: weights[i].item() 
            for i, name in enumerate(self.hypernetwork.function_names)
        })
        self.domain_history.append({
            domains[i]: domain_prefs[i].item() 
            for i in range(len(domains))
        })
        self.curriculum_history.append({
            'easy': curriculum_prefs[0].item(),
            'medium': curriculum_prefs[1].item(),
            'hard': curriculum_prefs[2].item()
        })
        
        if verbose:
            logger.info(f"Hypernetwork weights: {self.weight_history[-1]}")
            logger.info(f"Domain preferences: {self.domain_history[-1]}")
            logger.info(f"Curriculum preferences: {self.curriculum_history[-1]}")
            logger.info(f"Temperature: {temperature.item():.3f}")
        
        # Initialize selection with embedding context
        selected_indices = []
        context['selected_embeddings'] = {}
        context['training_progress'] = training_state.tokens_seen / max(training_state.total_tokens, 1)
        
        # Greedy selection with curriculum and domain awareness
        n = len(dataset)
        
        # Pre-filter by domain if specified
        if target_domain != 'general':
            domain_scores = {}
            domain_scorer = next((sf for sf in self.scoring_functions if sf.name == 'domain_relevance'), None)
            if domain_scorer:
                for idx in range(n):
                    domain_score = domain_scorer.score(idx, dataset[idx], [], context)
                    domain_scores[idx] = domain_score
                
                # Sort by domain relevance and consider top candidates
                domain_candidates = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
                candidate_pool = [idx for idx, score in domain_candidates[:min(budget * 5, n)]]
            else:
                candidate_pool = list(range(n))
        else:
            candidate_pool = list(range(n))
        
        # Main selection loop
        for step in range(budget):
            best_idx = -1
            best_score = -float('inf')
            
            for idx in candidate_pool:
                if idx not in selected_indices:
                    text_data = dataset[idx]
                    
                    # Update context with semantic embeddings for diversity scoring
                    diversity_scorer = next((sf for sf in self.scoring_functions if sf.name == 'semantic_diversity'), None)
                    if diversity_scorer and idx not in context['selected_embeddings']:
                        text = text_data.get('text', '')
                        if text:
                            embedding = diversity_scorer._get_embedding(text)
                            context['selected_embeddings'][idx] = embedding
                    
                    # Compute combined score
                    score, individual = self.compute_combined_score(
                        idx, text_data, selected_indices, context, weights, curriculum_prefs
                    )
                    
                    if score > best_score:
                        best_score = score
                        best_idx = idx
            
            # Add best sample
            if best_idx >= 0:
                selected_indices.append(best_idx)
                
                # Update selected embeddings context
                if best_idx in context['selected_embeddings']:
                    # Keep embedding for future diversity calculations
                    pass
            
            if verbose and step % (budget // 10) == 0:
                logger.info(f"Selected {step+1}/{budget} samples, current score: {best_score:.4f}")
        
        # Compile selection info
        selection_info = {
            'weights': self.weight_history[-1],
            'domain_preferences': self.domain_history[-1],
            'curriculum_preferences': self.curriculum_history[-1],
            'temperature': temperature.item(),
            'value': value.item(),
            'performance_predictions': {
                name: performance_pred[i].item() 
                for i, name in enumerate(self.hypernetwork.function_names)
            },
            'target_domain': target_domain,
            'final_score': best_score,
            'total_candidates': len(candidate_pool)
        }
        
        return selected_indices, selection_info
    
    def update_hypernetwork_llm(self,
                               previous_state: LLMTrainingState,
                               current_state: LLMTrainingState,
                               reward_metrics: Dict[str, float],
                               optimizer: torch.optim.Optimizer) -> float:
        """
        Update hypernetwork using multi-objective reward for LLM training
        """
        # Compute multi-objective reward
        perplexity_improvement = previous_state.perplexity - current_state.perplexity
        loss_improvement = previous_state.train_loss - current_state.train_loss
        
        # Combine different reward signals
        reward = (
            0.4 * perplexity_improvement +
            0.3 * loss_improvement +
            0.2 * reward_metrics.get('downstream_performance', 0.0) +
            0.1 * reward_metrics.get('efficiency_gain', 0.0)
        )
        
        # Forward passes
        prev_outputs = self.hypernetwork(previous_state)
        curr_outputs = self.hypernetwork(current_state)
        
        prev_weights, prev_curriculum, prev_temp, prev_value, prev_perf, prev_domain = prev_outputs
        curr_weights, curr_curriculum, curr_temp, curr_value, curr_perf, curr_domain = curr_outputs
        
        # Multi-component loss
        # Value loss
        target_value = reward + 0.99 * curr_value.detach()
        value_loss = F.mse_loss(prev_value, target_value)
        
        # Policy loss
        log_probs = torch.log(prev_weights + 1e-8)
        policy_loss = -torch.sum(log_probs) * reward
        
        # Curriculum consistency loss
        curriculum_loss = F.mse_loss(prev_curriculum, curr_curriculum.detach())
        
        # Domain adaptation loss
        domain_loss = F.mse_loss(prev_domain, curr_domain.detach())
        
        # Total loss
        total_loss = (
            value_loss + 
            0.1 * policy_loss + 
            0.05 * curriculum_loss + 
            0.05 * domain_loss
        )
        
        # Update
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.hypernetwork.parameters(), 1.0)
        optimizer.step()
        
        return total_loss.item()


# =============================================================================
# Complete LLM Experiment Framework
# =============================================================================

def create_llm_scoring_functions(model: nn.Module, tokenizer, device='cuda') -> List[LLMScoringFunction]:
    """Create comprehensive set of LLM-specific scoring functions"""
    return [
        PerplexityScoring(model, tokenizer, device),
        SemanticDiversityScoring(),
        QualityScoring(),
        DifficultyScoring(model, tokenizer, device),
        DomainRelevanceScoring(),
        CurriculumScoring()
    ]


def run_llm_multi_scoring_experiment(
    model: nn.Module,
    tokenizer,
    dataset: List[Dict[str, Any]],
    val_dataset: List[Dict[str, Any]],
    epochs: int = 100,
    budget_ratio: float = 0.1,
    target_domain: str = 'general',
    use_galore: bool = True,
    device: str = 'cuda'
) -> Tuple[Dict[str, Any], LLMMultiScoringSelector]:
    """
    Complete LLM experiment using hypernetwork-based multi-scoring selection
    """
    
    # Create scoring functions
    scoring_functions = create_llm_scoring_functions(model, tokenizer, device)
    
    # Create hypernetwork
    hypernetwork = LLMMultiScoringHypernetwork(scoring_functions)
    hypernetwork.to(device)
    
    # Create selector
    selector = LLMMultiScoringSelector(hypernetwork, scoring_functions, tokenizer)
    
    # Setup optimizers
    if use_galore:
        optimizer = GaLoreOptimizer(model.parameters(), rank=256, lr=1e-4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    rl_optimizer = torch.optim.Adam(hypernetwork.parameters(), lr=3e-4)
    
    # Results tracking
    results = {
        'train_losses': [],
        'val_losses': [],
        'perplexities': [],
        'selection_info_history': [],
        'weight_evolution': [],
        'domain_evolution': [],
        'curriculum_evolution': []
    }
    
    budget = int(len(dataset) * budget_ratio)
    selected_indices = []
    
    for epoch in range(epochs):
        # Create current training state
        # These would be computed from actual training metrics
        training_state = LLMTrainingState(
            epoch=epoch,
            train_loss=2.5 - 0.01 * epoch,  # Placeholder
            val_loss=2.6 - 0.01 * epoch,    # Placeholder
            perplexity=12.0 - 0.05 * epoch, # Placeholder
            learning_rate=1e-4 * (0.95 ** epoch),
            tokens_seen=epoch * 1000000,
            total_tokens=epochs * 1000000,
            gradient_norm=1.0,              # Placeholder
            domain_distribution={'general': 1.0},
            length_distribution={'medium': 1.0},
            performance_history=results['val_losses'][-10:],
            selection_diversity=0.5,
            memory_usage=0.7
        )
        
        # Adaptive coreset selection
        if epoch % 10 == 0 or epoch == 0:  # Reselect every 10 epochs
            selected_indices, selection_info = selector.select_coreset_adaptive(
                dataset, budget, training_state, target_domain, verbose=True
            )
            results['selection_info_history'].append(selection_info)
        
        # Training step (placeholder)
        train_loss = training_state.train_loss
        val_loss = training_state.val_loss
        perplexity = training_state.perplexity
        
        # Update hypernetwork
        if epoch > 0:
            reward_metrics = {
                'downstream_performance': 0.01,  # Placeholder
                'efficiency_gain': 0.02          # Placeholder
            }
            rl_loss = selector.update_hypernetwork_llm(
                prev_training_state, training_state, reward_metrics, rl_optimizer
            )
        
        # Store results
        results['train_losses'].append(train_loss)
        results['val_losses'].append(val_loss)
        results['perplexities'].append(perplexity)
        results['weight_evolution'].append(selector.weight_history[-1] if selector.weight_history else {})
        results['domain_evolution'].append(selector.domain_history[-1] if selector.domain_history else {})
        results['curriculum_evolution'].append(selector.curriculum_history[-1] if selector.curriculum_history else {})
        
        # Update for next iteration
        prev_training_state = training_state
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: Loss={train_loss:.4f}, Perplexity={perplexity:.2f}")
    
    return results, selector


