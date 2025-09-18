"""
Exhaustive Coreset Selection for Large Language Models
Complete implementation with all advanced techniques for LLM dataset distillation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForCausalLM, 
    AutoModelForSequenceClassification, BertModel, GPT2Model
)

import numpy as np
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import os
import json
from typing import List, Dict, Tuple, Optional, Callable, Union, Any
import logging
import argparse
from heapq import heappush, heappop
import hashlib
import pickle
import random
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import scipy.stats
from itertools import combinations
import re
import math

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# LLM-Specific Dataset Classes
# =============================================================================

class LLMDataset(Dataset):
    """Base class for LLM datasets"""
    
    def __init__(self, texts: List[str], labels: Optional[List] = None, 
                 tokenizer_name: str = "bert-base-uncased", max_length: int = 512):
        self.texts = texts
        self.labels = labels if labels is not None else [0] * len(texts)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Precompute features for efficiency
        self._precompute_features()
    
    def _precompute_features(self):
        """Precompute text features for coreset selection"""
        logger.info("Precomputing text features...")
        
        self.text_lengths = [len(text.split()) for text in self.texts]
        self.char_counts = [len(text) for text in self.texts]
        self.sentence_counts = [len(re.split(r'[.!?]+', text)) for text in self.texts]
        self.complexity_scores = self._compute_complexity_scores()
        self.diversity_hashes = [self._text_hash(text) for text in self.texts]
        
    def _compute_complexity_scores(self) -> List[float]:
        """Compute text complexity using various heuristics"""
        scores = []
        for text in self.texts:
            words = text.split()
            if len(words) == 0:
                scores.append(0.0)
                continue
                
            # Average word length
            avg_word_len = np.mean([len(word) for word in words])
            
            # Unique word ratio
            unique_ratio = len(set(words)) / len(words)
            
            # Punctuation density
            punct_count = sum(1 for char in text if char in '.,!?;:')
            punct_density = punct_count / len(text) if len(text) > 0 else 0
            
            # Combined complexity score
            complexity = avg_word_len * unique_ratio + punct_density * 10
            scores.append(complexity)
            
        return scores
    
    def _text_hash(self, text: str) -> str:
        """Create hash for text similarity computation"""
        # Normalize text and create hash
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        return hashlib.md5(normalized.encode()).hexdigest()[:16]
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long),
            'text': text,
            'idx': idx
        }

class CausalLMDataset(LLMDataset):
    """Dataset for causal language modeling tasks"""
    
    def __init__(self, texts: List[str], tokenizer_name: str = "gpt2", max_length: int = 512):
        super().__init__(texts, None, tokenizer_name, max_length)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # For causal LM, input and labels are the same (shifted)
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        
        return {
            'input_ids': input_ids,
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': input_ids.clone(),  # For causal LM, labels are shifted input_ids
            'text': text,
            'idx': idx
        }

class ClassificationDataset(LLMDataset):
    """Dataset for text classification tasks"""
    
    def __init__(self, texts: List[str], labels: List[int], 
                 tokenizer_name: str = "bert-base-uncased", max_length: int = 512):
        super().__init__(texts, labels, tokenizer_name, max_length)

class QADataset(LLMDataset):
    """Dataset for question-answering tasks"""
    
    def __init__(self, questions: List[str], contexts: List[str], answers: List[str],
                 tokenizer_name: str = "bert-base-uncased", max_length: int = 512):
        # Combine question and context
        texts = [f"Question: {q} Context: {c}" for q, c in zip(questions, contexts)]
        super().__init__(texts, answers, tokenizer_name, max_length)
        self.questions = questions
        self.contexts = contexts
        self.answers = answers


# =============================================================================
# Advanced LLM Coreset Selection Strategies
# =============================================================================

class LLMCoresetStrategies:
    """Comprehensive collection of LLM-specific coreset selection strategies"""
    
    def __init__(self, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        
        # Caches for efficiency
        self.embedding_cache = {}
        self.gradient_cache = {}
        self.perplexity_cache = {}
        self.attention_cache = {}
        self.influence_cache = {}
        
        # Feature extractors
        self.pca = None
        self.cluster_centers = None
        
    def clear_cache(self):
        """Clear all caches to free memory"""
        for cache in [self.embedding_cache, self.gradient_cache, 
                     self.perplexity_cache, self.attention_cache, self.influence_cache]:
            cache.clear()
    
    # =============================================================================
    # Gradient-Based Strategies
    # =============================================================================
    
    def gradient_magnitude_score(self, idx: int, batch: Dict, selected_indices: List[int]) -> float:
        """Score based on gradient magnitude for this sample"""
        cache_key = f"grad_mag_{idx}"
        if cache_key in self.gradient_cache:
            return self.gradient_cache[cache_key]
        
        self.model.zero_grad()
        
        # Forward pass
        input_ids = batch['input_ids'].unsqueeze(0).to(self.device)
        attention_mask = batch['attention_mask'].unsqueeze(0).to(self.device)
        labels = batch['labels'].unsqueeze(0).to(self.device)
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        
        # Compute gradient norm
        grad_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm().item() ** 2
        grad_norm = grad_norm ** 0.5
        
        self.gradient_cache[cache_key] = grad_norm
        return grad_norm
    
    def gradient_diversity_score(self, idx: int, batch: Dict, selected_indices: List[int]) -> float:
        """Score based on gradient diversity with selected samples"""
        if len(selected_indices) == 0:
            return 1.0
        
        # Get gradient for current sample
        current_grad = self._get_gradient_vector(idx, batch)
        
        # Compute similarity with selected samples
        min_similarity = float('inf')
        for sel_idx in selected_indices[-10:]:  # Only consider recent selections
            if f"grad_vec_{sel_idx}" in self.gradient_cache:
                sel_grad = self.gradient_cache[f"grad_vec_{sel_idx}"]
                similarity = F.cosine_similarity(current_grad, sel_grad, dim=0).item()
                min_similarity = min(min_similarity, similarity)
        
        # Lower similarity = higher diversity score
        diversity_score = 1.0 - max(0, min_similarity)
        return diversity_score
    
    def gradient_alignment_score(self, idx: int, batch: Dict, selected_indices: List[int]) -> float:
        """Score based on alignment with average gradient direction"""
        if len(selected_indices) < 5:
            return 0.5  # Neutral score for early selections
        
        current_grad = self._get_gradient_vector(idx, batch)
        
        # Compute average gradient from recent selections
        recent_grads = []
        for sel_idx in selected_indices[-20:]:
            if f"grad_vec_{sel_idx}" in self.gradient_cache:
                recent_grads.append(self.gradient_cache[f"grad_vec_{sel_idx}"])
        
        if recent_grads:
            avg_grad = torch.stack(recent_grads).mean(dim=0)
            alignment = F.cosine_similarity(current_grad, avg_grad, dim=0).item()
            # Return alignment score (higher = better aligned)
            return (alignment + 1) / 2  # Normalize to [0, 1]
        
        return 0.5
    
    def _get_gradient_vector(self, idx: int, batch: Dict) -> torch.Tensor:
        """Get gradient vector for a sample"""
        cache_key = f"grad_vec_{idx}"
        if cache_key in self.gradient_cache:
            return self.gradient_cache[cache_key]
        
        self.model.zero_grad()
        
        input_ids = batch['input_ids'].unsqueeze(0).to(self.device)
        attention_mask = batch['attention_mask'].unsqueeze(0).to(self.device)
        labels = batch['labels'].unsqueeze(0).to(self.device)
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        
        # Flatten gradients
        grad_vector = []
        for param in self.model.parameters():
            if param.grad is not None:
                grad_vector.append(param.grad.view(-1))
        
        if grad_vector:
            grad_tensor = torch.cat(grad_vector)
        else:
            grad_tensor = torch.zeros(1).to(self.device)
        
        self.gradient_cache[cache_key] = grad_tensor
        return grad_tensor
    
    # =============================================================================
    # Embedding-Based Strategies
    # =============================================================================
    
    def embedding_diversity_score(self, idx: int, batch: Dict, selected_indices: List[int]) -> float:
        """Score based on embedding diversity (facility location)"""
        current_emb = self._get_embedding(idx, batch)
        
        if len(selected_indices) == 0:
            return 1.0
        
        # Find minimum distance to selected samples
        min_distance = float('inf')
        for sel_idx in selected_indices:
            if f"emb_{sel_idx}" in self.embedding_cache:
                sel_emb = self.embedding_cache[f"emb_{sel_idx}"]
                distance = torch.norm(current_emb - sel_emb).item()
                min_distance = min(min_distance, distance)
        
        return min_distance
    
    def embedding_coverage_score(self, idx: int, batch: Dict, selected_indices: List[int]) -> float:
        """Score based on coverage of embedding space"""
        current_emb = self._get_embedding(idx, batch)
        
        # If we have cluster centers, compute distance to nearest center
        if self.cluster_centers is not None:
            distances = []
            for center in self.cluster_centers:
                dist = torch.norm(current_emb - torch.tensor(center).to(self.device)).item()
                distances.append(dist)
            
            # Score is inverse of distance to nearest cluster center
            return 1.0 / (min(distances) + 1e-8)
        
        return self.embedding_diversity_score(idx, batch, selected_indices)
    
    def embedding_centroid_score(self, idx: int, batch: Dict, selected_indices: List[int]) -> float:
        """Score based on distance from current selection centroid"""
        if len(selected_indices) < 2:
            return 0.5
        
        current_emb = self._get_embedding(idx, batch)
        
        # Compute centroid of selected embeddings
        selected_embs = []
        for sel_idx in selected_indices:
            if f"emb_{sel_idx}" in self.embedding_cache:
                selected_embs.append(self.embedding_cache[f"emb_{sel_idx}"])
        
        if selected_embs:
            centroid = torch.stack(selected_embs).mean(dim=0)
            distance = torch.norm(current_emb - centroid).item()
            return distance
        
        return 0.5
    
    def _get_embedding(self, idx: int, batch: Dict) -> torch.Tensor:
        """Get embedding for a sample"""
        cache_key = f"emb_{idx}"
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        with torch.no_grad():
            input_ids = batch['input_ids'].unsqueeze(0).to(self.device)
            attention_mask = batch['attention_mask'].unsqueeze(0).to(self.device)
            
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, 
                               output_hidden_states=True)
            
            # Use mean of last hidden state as embedding
            last_hidden_state = outputs.hidden_states[-1]
            embedding = last_hidden_state.mean(dim=1).squeeze()
            
            self.embedding_cache[cache_key] = embedding
            return embedding
    
    # =============================================================================
    # Uncertainty-Based Strategies
    # =============================================================================
    
    def perplexity_score(self, idx: int, batch: Dict, selected_indices: List[int]) -> float:
        """Score based on perplexity (uncertainty)"""
        cache_key = f"perplexity_{idx}"
        if cache_key in self.perplexity_cache:
            return self.perplexity_cache[cache_key]
        
        with torch.no_grad():
            input_ids = batch['input_ids'].unsqueeze(0).to(self.device)
            attention_mask = batch['attention_mask'].unsqueeze(0).to(self.device)
            labels = batch['labels'].unsqueeze(0).to(self.device)
            
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            # Compute perplexity
            loss = outputs.loss.item()
            perplexity = math.exp(loss)
            
            self.perplexity_cache[cache_key] = perplexity
            return perplexity
    
    def entropy_score(self, idx: int, batch: Dict, selected_indices: List[int]) -> float:
        """Score based on prediction entropy"""
        with torch.no_grad():
            input_ids = batch['input_ids'].unsqueeze(0).to(self.device)
            attention_mask = batch['attention_mask'].unsqueeze(0).to(self.device)
            
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Compute entropy over vocabulary
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1).mean().item()
            
            return entropy
    
    def confidence_score(self, idx: int, batch: Dict, selected_indices: List[int]) -> float:
        """Score based on prediction confidence (lower confidence = higher score)"""
        with torch.no_grad():
            input_ids = batch['input_ids'].unsqueeze(0).to(self.device)
            attention_mask = batch['attention_mask'].unsqueeze(0).to(self.device)
            
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Get maximum probability
            probs = F.softmax(logits, dim=-1)
            max_prob = probs.max(dim=-1).values.mean().item()
            
            # Return inverse of confidence
            return 1.0 - max_prob
    
    # =============================================================================
    # Attention-Based Strategies
    # =============================================================================
    
    def attention_diversity_score(self, idx: int, batch: Dict, selected_indices: List[int]) -> float:
        """Score based on attention pattern diversity"""
        attention_pattern = self._get_attention_pattern(idx, batch)
        
        if len(selected_indices) == 0:
            return 1.0
        
        # Compare with selected samples
        max_similarity = 0.0
        for sel_idx in selected_indices[-10:]:
            if f"attn_{sel_idx}" in self.attention_cache:
                sel_pattern = self.attention_cache[f"attn_{sel_idx}"]
                similarity = F.cosine_similarity(attention_pattern, sel_pattern, dim=0).item()
                max_similarity = max(max_similarity, similarity)
        
        # Return diversity score
        return 1.0 - max_similarity
    
    def attention_complexity_score(self, idx: int, batch: Dict, selected_indices: List[int]) -> float:
        """Score based on attention pattern complexity"""
        attention_pattern = self._get_attention_pattern(idx, batch)
        
        # Compute entropy of attention weights
        entropy = -(attention_pattern * torch.log(attention_pattern + 1e-8)).sum().item()
        
        return entropy
    
    def _get_attention_pattern(self, idx: int, batch: Dict) -> torch.Tensor:
        """Get attention pattern for a sample"""
        cache_key = f"attn_{idx}"
        if cache_key in self.attention_cache:
            return self.attention_cache[cache_key]
        
        with torch.no_grad():
            input_ids = batch['input_ids'].unsqueeze(0).to(self.device)
            attention_mask = batch['attention_mask'].unsqueeze(0).to(self.device)
            
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, 
                               output_attentions=True)
            
            # Average attention across layers and heads
            attentions = outputs.attentions
            avg_attention = torch.stack(attentions).mean(dim=0).mean(dim=1).squeeze()
            
            # Flatten attention pattern
            attention_pattern = avg_attention.view(-1)
            
            self.attention_cache[cache_key] = attention_pattern
            return attention_pattern
    
    # =============================================================================
    # Influence-Based Strategies
    # =============================================================================
    
    def influence_score(self, idx: int, batch: Dict, selected_indices: List[int]) -> float:
        """Score based on influence function approximation"""
        cache_key = f"influence_{idx}"
        if cache_key in self.influence_cache:
            return self.influence_cache[cache_key]
        
        # Approximate influence using gradient dot product
        sample_grad = self._get_gradient_vector(idx, batch)
        
        # Compute influence as gradient norm (simplified)
        influence = sample_grad.norm().item()
        
        self.influence_cache[cache_key] = influence
        return influence
    
    def forgetting_score(self, idx: int, batch: Dict, selected_indices: List[int]) -> float:
        """Score based on forgetting events during training"""
        # This would require tracking loss history during training
        # For now, use a proxy based on sample difficulty
        return self.perplexity_score(idx, batch, selected_indices)
    
    # =============================================================================
    # Text-Specific Strategies
    # =============================================================================
    
    def length_diversity_score(self, idx: int, batch: Dict, selected_indices: List[int]) -> float:
        """Score based on text length diversity"""
        current_length = len(batch['text'].split())
        
        if len(selected_indices) == 0:
            return 1.0
        
        # Get lengths of selected samples (would need to store this)
        # For now, return normalized length
        return min(current_length / 100.0, 1.0)
    
    def complexity_score(self, idx: int, batch: Dict, selected_indices: List[int]) -> float:
        """Score based on text complexity"""
        text = batch['text']
        
        # Compute various complexity measures
        words = text.split()
        if len(words) == 0:
            return 0.0
        
        # Average word length
        avg_word_len = np.mean([len(word) for word in words])
        
        # Vocabulary richness
        unique_ratio = len(set(words)) / len(words)
        
        # Sentence complexity
        sentences = re.split(r'[.!?]+', text)
        avg_sent_len = np.mean([len(sent.split()) for sent in sentences if sent.strip()])
        
        # Combined complexity score
        complexity = (avg_word_len / 10.0) + unique_ratio + (avg_sent_len / 50.0)
        return min(complexity, 1.0)
    
    def semantic_diversity_score(self, idx: int, batch: Dict, selected_indices: List[int]) -> float:
        """Score based on semantic diversity using embeddings"""
        return self.embedding_diversity_score(idx, batch, selected_indices)
    
    # =============================================================================
    # Domain-Specific Strategies
    # =============================================================================
    
    def domain_coverage_score(self, idx: int, batch: Dict, selected_indices: List[int]) -> float:
        """Score based on domain coverage (requires domain labels)"""
        # This would require domain classification
        # For now, use text complexity as proxy
        return self.complexity_score(idx, batch, selected_indices)
    
    def task_relevance_score(self, idx: int, batch: Dict, selected_indices: List[int]) -> float:
        """Score based on task relevance"""
        # This would be task-specific
        # For now, use perplexity as proxy
        return self.perplexity_score(idx, batch, selected_indices)


# =============================================================================
# Multi-Objective Coreset Optimization
# =============================================================================

class MultiObjectiveCoresetSelector:
    """Multi-objective coreset selection with Pareto optimization"""
    
    def __init__(self, strategies: Dict[str, Callable], objectives: List[str], 
                 weights: Optional[List[float]] = None):
        self.strategies = strategies
        self.objectives = objectives
        self.num_objectives = len(objectives)
        
        if weights is None:
            self.weights = [1.0] * self.num_objectives
        else:
            self.weights = weights
            
        # Pareto frontier tracking
        self.pareto_front = []
        self.objective_history = defaultdict(list)
        
    def compute_objective_scores(self, idx: int, batch: Dict, selected_indices: List[int]) -> Dict[str, float]:
        """Compute scores for all objectives"""
        scores = {}
        
        for obj_name in self.objectives:
            if obj_name in self.strategies:
                score = self.strategies[obj_name](idx, batch, selected_indices)
                scores[obj_name] = score
                self.objective_history[obj_name].append(score)
        
        return scores
    
    def scalarize_objectives(self, objective_scores: Dict[str, float]) -> float:
        """Convert multi-objective scores to single score"""
        # Weighted sum scalarization
        total_score = 0.0
        
        for i, obj_name in enumerate(self.objectives):
            if obj_name in objective_scores:
                # Normalize using running statistics
                score = objective_scores[obj_name]
                if len(self.objective_history[obj_name]) > 10:
                    history = self.objective_history[obj_name][-100:]
                    mean_score = np.mean(history)
                    std_score = np.std(history) + 1e-8
                    normalized_score = (score - mean_score) / std_score
                else:
                    normalized_score = score
                
                total_score += self.weights[i] * normalized_score
        
        return total_score
    
    def is_pareto_optimal(self, scores: Dict[str, float]) -> bool:
        """Check if a solution is Pareto optimal"""
        score_vector = [scores.get(obj, 0) for obj in self.objectives]
        
        for front_scores in self.pareto_front:
            # Check if current solution is dominated
            dominated = True
            for i in range(self.num_objectives):
                if score_vector[i] >= front_scores[i]:
                    dominated = False
                    break
            
            if dominated:
                return False
        
        return True
    
    def update_pareto_front(self, scores: Dict[str, float], idx: int):
        """Update Pareto frontier"""
        score_vector = [scores.get(obj, 0) for obj in self.objectives]
        
        if self.is_pareto_optimal(scores):
            # Remove dominated solutions
            new_front = []
            for i, front_scores in enumerate(self.pareto_front):
                # Check if this solution dominates the front solution
                dominates = True
                for j in range(self.num_objectives):
                    if score_vector[j] <= front_scores[j]:
                        dominates = False
                        break
                
                if not dominates:
                    new_front.append(front_scores)
            
            # Add new solution
            new_front.append(score_vector)
            self.pareto_front = new_front
    
    def select_coreset_pareto(self, dataset, budget: int, batch_size: int = 50) -> List[int]:
        """Select coreset using Pareto optimization"""
        selected_indices = []
        n = len(dataset)
        available_indices = set(range(n))
        
        pbar = tqdm(total=budget, desc="Multi-objective coreset selection")
        
        while len(selected_indices) < budget and available_indices:
            best_score = float('-inf')
            best_idx = None
            best_objectives = None
            
            # Sample a batch of candidates
            candidates = list(available_indices)[:batch_size]
            
            for idx in candidates:
                batch = dataset[idx]
                objective_scores = self.compute_objective_scores(idx, batch, selected_indices)
                
                # Scalarize for selection
                total_score = self.scalarize_objectives(objective_scores)
                
                if total_score > best_score:
                    best_score = total_score
                    best_idx = idx
                    best_objectives = objective_scores
            
            if best_idx is not None:
                selected_indices.append(best_idx)
                available_indices.remove(best_idx)
                self.update_pareto_front(best_objectives, best_idx)
                pbar.update(1)
        
        pbar.close()
        return selected_indices


# =============================================================================
# Advanced Evaluation Metrics
# =============================================================================

class LLMCoresetEvaluator:
    """Comprehensive evaluation suite for LLM coresets"""
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def evaluate_coreset_quality(self, original_dataset, coreset_indices: List[int], 
                                test_dataset, num_training_steps: int = 1000) -> Dict[str, float]:
        """Comprehensive evaluation of coreset quality"""
        results = {}
        
        # Create coreset
        coreset = Subset(original_dataset, coreset_indices)
        
        # 1. Training Performance
        logger.info("Evaluating training performance...")
        train_metrics = self._evaluate_training_performance(coreset, test_dataset, num_training_steps)
        results.update(train_metrics)
        
        # 2. Representation Quality
        logger.info("Evaluating representation quality...")
        repr_metrics = self._evaluate_representation_quality(original_dataset, coreset_indices)
        results.update(repr_metrics)
        
        # 3. Diversity Metrics
        logger.info("Evaluating diversity...")
        diversity_metrics = self._evaluate_diversity(original_dataset, coreset_indices)
        results.update(diversity_metrics)
        
        # 4. Coverage Metrics
        logger.info("Evaluating coverage...")
        coverage_metrics = self._evaluate_coverage(original_dataset, coreset_indices)
        results.update(coverage_metrics)
        
        # 5. Task-Specific Metrics
        logger.info("Evaluating task-specific metrics...")
        task_metrics = self._evaluate_task_specific_metrics(original_dataset, coreset_indices, test_dataset)
        results.update(task_metrics)
        
        return results
    
    def _evaluate_training_performance(self, coreset, test_dataset, num_steps: int) -> Dict[str, float]:
        """Evaluate training performance on coreset"""
        # Simple training loop
        train_loader = DataLoader(coreset, batch_size=8, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        
        # Training
        self.model.train()
        total_loss = 0
        step = 0
        
        for epoch in range((num_steps // len(train_loader)) + 1):
            for batch in train_loader:
                if step >= num_steps:
                    break
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                step += 1
        
        avg_train_loss = total_loss / num_steps
        
        # Evaluation
        self.model.eval()
        eval_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                eval_loss += outputs.loss.item()
                
                # For classification tasks
                if hasattr(outputs, 'logits'):
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)
        
        results = {
            'train_loss': avg_train_loss,
            'eval_loss': eval_loss / len(test_loader),
            'accuracy': correct / total if total > 0 else 0.0
        }
        
        return results
    
    def _evaluate_representation_quality(self, dataset, coreset_indices: List[int]) -> Dict[str, float]:
        """Evaluate how well coreset represents the full dataset"""
        
        # Get embeddings for full dataset and coreset
        full_embeddings = []
        coreset_embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(len(dataset)), desc="Computing embeddings"):
                batch = dataset[i]
                input_ids = batch['input_ids'].unsqueeze(0).to(self.device)
                attention_mask = batch['attention_mask'].unsqueeze(0).to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, 
                                   output_hidden_states=True)
                
                embedding = outputs.hidden_states[-1].mean(dim=1).squeeze().cpu().numpy()
                full_embeddings.append(embedding)
                
                if i in coreset_indices:
                    coreset_embeddings.append(embedding)
        
        full_embeddings = np.array(full_embeddings)
        coreset_embeddings = np.array(coreset_embeddings)
        
        # Compute representation metrics
        results = {}
        
        # 1. Mean distance between centroids
        full_centroid = full_embeddings.mean(axis=0)
        coreset_centroid = coreset_embeddings.mean(axis=0)
        centroid_distance = np.linalg.norm(full_centroid - coreset_centroid)
        results['centroid_distance'] = centroid_distance
        
        # 2. Covariance matrix difference
        full_cov = np.cov(full_embeddings.T)
        coreset_cov = np.cov(coreset_embeddings.T)
        cov_frobenius = np.linalg.norm(full_cov - coreset_cov, 'fro')
        results['covariance_difference'] = cov_frobenius
        
        # 3. Principal component alignment
        pca_full = PCA(n_components=10).fit(full_embeddings)
        pca_coreset = PCA(n_components=10).fit(coreset_embeddings)
        
        # Compute subspace angle between principal components
        pc_similarity = np.mean([
            np.abs(np.dot(pca_full.components_[i], pca_coreset.components_[i]))
            for i in range(min(10, len(pca_full.components_), len(pca_coreset.components_)))
        ])
        results['pc_alignment'] = pc_similarity
        
        return results
    
    def _evaluate_diversity(self, dataset, coreset_indices: List[int]) -> Dict[str, float]:
        """Evaluate diversity metrics"""
        results = {}
        
        # Text-based diversity
        coreset_texts = [dataset[i]['text'] for i in coreset_indices]
        
        # 1. Vocabulary diversity
        vocab_full = set()
        vocab_coreset = set()
        
        for i in range(len(dataset)):
            words = dataset[i]['text'].lower().split()
            vocab_full.update(words)
            
            if i in coreset_indices:
                vocab_coreset.update(words)
        
        vocab_coverage = len(vocab_coreset) / len(vocab_full)
        results['vocab_coverage'] = vocab_coverage
        
        # 2. Length diversity
        lengths_full = [len(dataset[i]['text'].split()) for i in range(len(dataset))]
        lengths_coreset = [len(dataset[i]['text'].split()) for i in coreset_indices]
        
        length_diversity = np.std(lengths_coreset) / np.std(lengths_full)
        results['length_diversity'] = length_diversity
        
        # 3. Semantic diversity (using text hashes as proxy)
        hashes_full = set()
        hashes_coreset = set()
        
        for i in range(len(dataset)):
            text_hash = hashlib.md5(dataset[i]['text'].lower().encode()).hexdigest()[:8]
            hashes_full.add(text_hash)
            
            if i in coreset_indices:
                hashes_coreset.add(text_hash)
        
        semantic_coverage = len(hashes_coreset) / len(hashes_full)
        results['semantic_coverage'] = semantic_coverage
        
        return results
    
    def _evaluate_coverage(self, dataset, coreset_indices: List[int]) -> Dict[str, float]:
        """Evaluate coverage metrics"""
        results = {}
        
        # 1. Label coverage (for classification tasks)
        if hasattr(dataset[0], 'labels'):
            labels_full = set()
            labels_coreset = set()
            
            for i in range(len(dataset)):
                label = dataset[i]['labels'].item() if torch.is_tensor(dataset[i]['labels']) else dataset[i]['labels']
                labels_full.add(label)
                
                if i in coreset_indices:
                    labels_coreset.add(label)
            
            label_coverage = len(labels_coreset) / len(labels_full) if len(labels_full) > 0 else 1.0
            results['label_coverage'] = label_coverage
        
        # 2. Difficulty coverage
        difficulties = []
        coreset_difficulties = []
        
        with torch.no_grad():
            for i in range(min(1000, len(dataset))):  # Sample for efficiency
                batch = dataset[i]
                input_ids = batch['input_ids'].unsqueeze(0).to(self.device)
                attention_mask = batch['attention_mask'].unsqueeze(0).to(self.device)
                labels = batch['labels'].unsqueeze(0).to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                difficulty = outputs.loss.item()
                difficulties.append(difficulty)
                
                if i in coreset_indices:
                    coreset_difficulties.append(difficulty)
        
        if coreset_difficulties and difficulties:
            # KL divergence between difficulty distributions
            hist_full, bins = np.histogram(difficulties, bins=20, density=True)
            hist_coreset, _ = np.histogram(coreset_difficulties, bins=bins, density=True)
            
            # Add small epsilon to avoid log(0)
            hist_full = hist_full + 1e-8
            hist_coreset = hist_coreset + 1e-8
            
            kl_div = scipy.stats.entropy(hist_coreset, hist_full)
            results['difficulty_kl_divergence'] = kl_div
        
        return results
    
    def _evaluate_task_specific_metrics(self, dataset, coreset_indices: List[int], test_dataset) -> Dict[str, float]:
        """Evaluate task-specific metrics"""
        results = {}
        
        # This would be customized based on the specific task
        # For now, compute general language modeling metrics
        
        coreset = Subset(dataset, coreset_indices)
        
        # 1. Perplexity on test set after training on coreset
        test_perplexity = self._compute_perplexity(test_dataset)
        results['test_perplexity'] = test_perplexity
        
        # 2. Transfer learning performance
        # (Would require training on coreset and evaluating on various downstream tasks)
        
        return results
    
    def _compute_perplexity(self, dataset) -> float:
        """Compute perplexity on dataset"""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        loader = DataLoader(dataset, batch_size=16, shuffle=False)
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                # Count non-padding tokens
                num_tokens = attention_mask.sum().item()
                
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
        
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        
        return perplexity


# =============================================================================
# Main LLM Coreset Framework
# =============================================================================

class LLMCoresetFramework:
    """Complete framework for LLM coreset selection and evaluation"""
    
    def __init__(self, model_name: str = "bert-base-uncased", device: str = 'auto'):
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        logger.info(f"Initializing LLM Coreset Framework with model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        
        # Initialize components
        self.strategies = LLMCoresetStrategies(self.model, self.tokenizer, self.device)
        self.evaluator = LLMCoresetEvaluator(self.model, self.tokenizer, self.device)
        
        # Available strategies
        self.available_strategies = {
            # Gradient-based
            'gradient_magnitude': self.strategies.gradient_magnitude_score,
            'gradient_diversity': self.strategies.gradient_diversity_score,
            'gradient_alignment': self.strategies.gradient_alignment_score,
            
            # Embedding-based
            'embedding_diversity': self.strategies.embedding_diversity_score,
            'embedding_coverage': self.strategies.embedding_coverage_score,
            'embedding_centroid': self.strategies.embedding_centroid_score,
            
            # Uncertainty-based
            'perplexity': self.strategies.perplexity_score,
            'entropy': self.strategies.entropy_score,
            'confidence': self.strategies.confidence_score,
            
            # Attention-based
            'attention_diversity': self.strategies.attention_diversity_score,
            'attention_complexity': self.strategies.attention_complexity_score,
            
            # Influence-based
            'influence': self.strategies.influence_score,
            'forgetting': self.strategies.forgetting_score,
            
            # Text-specific
            'length_diversity': self.strategies.length_diversity_score,
            'complexity': self.strategies.complexity_score,
            'semantic_diversity': self.strategies.semantic_diversity_score,
            
            # Domain-specific
            'domain_coverage': self.strategies.domain_coverage_score,
            'task_relevance': self.strategies.task_relevance_score
        }
        
    def select_coreset(self, dataset, budget: int, strategy_names: List[str], 
                      strategy_weights: Optional[List[float]] = None, 
                      selection_method: str = 'greedy') -> List[int]:
        """Select coreset using specified strategies"""
        
        logger.info(f"Selecting coreset with budget {budget} using strategies: {strategy_names}")
        
        # Prepare strategies
        selected_strategies = {name: self.available_strategies[name] for name in strategy_names}
        
        if selection_method == 'greedy':
            return self._greedy_selection(dataset, budget, selected_strategies, strategy_weights)
        elif selection_method == 'pareto':
            selector = MultiObjectiveCoresetSelector(selected_strategies, strategy_names, strategy_weights)
            return selector.select_coreset_pareto(dataset, budget)
        else:
            raise ValueError(f"Unknown selection method: {selection_method}")
    
    def _greedy_selection(self, dataset, budget: int, strategies: Dict[str, Callable], 
                         weights: Optional[List[float]] = None) -> List[int]:
        """Greedy coreset selection"""
        
        if weights is None:
            weights = [1.0] * len(strategies)
        
        selected_indices = []
        available_indices = set(range(len(dataset)))
        strategy_names = list(strategies.keys())
        
        pbar = tqdm(total=budget, desc="Greedy coreset selection")
        
        while len(selected_indices) < budget and available_indices:
            best_score = float('-inf')
            best_idx = None
            
            # Sample candidates for efficiency
            candidates = list(available_indices)[:100]
            
            for idx in candidates:
                batch = dataset[idx]
                
                # Compute weighted score
                total_score = 0.0
                for i, (strategy_name, strategy_fn) in enumerate(strategies.items()):
                    score = strategy_fn(idx, batch, selected_indices)
                    total_score += weights[i] * score
                
                if total_score > best_score:
                    best_score = total_score
                    best_idx = idx
            
            if best_idx is not None:
                selected_indices.append(best_idx)
                available_indices.remove(best_idx)
                pbar.update(1)
                
                # Periodic cache cleanup
                if len(selected_indices) % 100 == 0:
                    self.strategies.clear_cache()
        
        pbar.close()
        return selected_indices
    
    def evaluate_coreset(self, original_dataset, coreset_indices: List[int], 
                        test_dataset, evaluation_steps: int = 1000) -> Dict[str, Any]:
        """Comprehensive coreset evaluation"""
        
        logger.info("Starting comprehensive coreset evaluation...")
        
        results = self.evaluator.evaluate_coreset_quality(
            original_dataset, coreset_indices, test_dataset, evaluation_steps
        )
        
        # Add basic statistics
        results['coreset_size'] = len(coreset_indices)
        results['compression_ratio'] = len(coreset_indices) / len(original_dataset)
        results['dataset_size'] = len(original_dataset)
        
        return results
    
    def run_coreset_experiments(self, datasets: Dict[str, Any], strategies_config: Dict[str, Any], 
                               output_dir: str = './llm_coreset_results') -> Dict[str, Any]:
        """Run comprehensive coreset experiments"""
        
        os.makedirs(output_dir, exist_ok=True)
        all_results = {}
        
        for dataset_name, dataset_info in datasets.items():
            logger.info(f"Running experiments on dataset: {dataset_name}")
            
            train_dataset = dataset_info['train']
            test_dataset = dataset_info['test']
            
            dataset_results = {}
            
            for strategy_config in strategies_config['configurations']:
                config_name = strategy_config['name']
                strategies = strategy_config['strategies']
                weights = strategy_config.get('weights')
                budget_ratio = strategy_config.get('budget_ratio', 0.1)
                
                logger.info(f"Testing configuration: {config_name}")
                
                # Calculate budget
                budget = int(len(train_dataset) * budget_ratio)
                
                # Select coreset
                start_time = time.time()
                coreset_indices = self.select_coreset(
                    train_dataset, budget, strategies, weights
                )
                selection_time = time.time() - start_time
                
                # Evaluate coreset
                evaluation_results = self.evaluate_coreset(
                    train_dataset, coreset_indices, test_dataset
                )
                evaluation_results['selection_time'] = selection_time
                
                dataset_results[config_name] = evaluation_results
                
                # Clear caches
                self.strategies.clear_cache()
            
            all_results[dataset_name] = dataset_results
            
            # Save intermediate results
            with open(os.path.join(output_dir, f'{dataset_name}_results.json'), 'w') as f:
                json.dump(dataset_results, f, indent=2)
        
        # Save all results
        with open(os.path.join(output_dir, 'all_results.json'), 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Generate visualizations
        self._generate_visualizations(all_results, output_dir)
        
        return all_results
    
    def _generate_visualizations(self, results: Dict[str, Any], output_dir: str):
        """Generate comprehensive visualizations"""
        
        logger.info("Generating visualizations...")
        
        # 1. Performance comparison across strategies
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract performance metrics
        datasets = list(results.keys())
        strategies = list(next(iter(results.values())).keys())
        
        # Training performance
        ax = axes[0, 0]
        train_losses = []
        for dataset in datasets:
            dataset_losses = []
            for strategy in strategies:
                loss = results[dataset][strategy].get('train_loss', 0)
                dataset_losses.append(loss)
            train_losses.append(dataset_losses)
        
        x = np.arange(len(strategies))
        width = 0.8 / len(datasets)
        
        for i, (dataset, losses) in enumerate(zip(datasets, train_losses)):
            ax.bar(x + i * width, losses, width, label=dataset, alpha=0.8)
        
        ax.set_xlabel('Strategy')
        ax.set_ylabel('Training Loss')
        ax.set_title('Training Performance Comparison')
        ax.set_xticks(x + width * (len(datasets) - 1) / 2)
        ax.set_xticklabels(strategies, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Representation quality
        ax = axes[0, 1]
        centroid_distances = []
        for dataset in datasets:
            dataset_distances = []
            for strategy in strategies:
                distance = results[dataset][strategy].get('centroid_distance', 0)
                dataset_distances.append(distance)
            centroid_distances.append(dataset_distances)
        
        for i, (dataset, distances) in enumerate(zip(datasets, centroid_distances)):
            ax.bar(x + i * width, distances, width, label=dataset, alpha=0.8)
        
        ax.set_xlabel('Strategy')
        ax.set_ylabel('Centroid Distance')
        ax.set_title('Representation Quality')
        ax.set_xticks(x + width * (len(datasets) - 1) / 2)
        ax.set_xticklabels(strategies, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Diversity metrics
        ax = axes[1, 0]
        vocab_coverages = []
        for dataset in datasets:
            dataset_coverages = []
            for strategy in strategies:
                coverage = results[dataset][strategy].get('vocab_coverage', 0)
                dataset_coverages.append(coverage)
            vocab_coverages.append(dataset_coverages)
        
        for i, (dataset, coverages) in enumerate(zip(datasets, vocab_coverages)):
            ax.bar(x + i * width, coverages, width, label=dataset, alpha=0.8)
        
        ax.set_xlabel('Strategy')
        ax.set_ylabel('Vocabulary Coverage')
        ax.set_title('Diversity Metrics')
        ax.set_xticks(x + width * (len(datasets) - 1) / 2)
        ax.set_xticklabels(strategies, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Selection efficiency
        ax = axes[1, 1]
        selection_times = []
        for dataset in datasets:
            dataset_times = []
            for strategy in strategies:
                time_taken = results[dataset][strategy].get('selection_time', 0)
                dataset_times.append(time_taken)
            selection_times.append(dataset_times)
        
        for i, (dataset, times) in enumerate(zip(datasets, selection_times)):
            ax.bar(x + i * width, times, width, label=dataset, alpha=0.8)
        
        ax.set_xlabel('Strategy')
        ax.set_ylabel('Selection Time (seconds)')
        ax.set_title('Selection Efficiency')
        ax.set_xticks(x + width * (len(datasets) - 1) / 2)
        ax.set_xticklabels(strategies, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'comprehensive_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Detailed analysis per dataset
        for dataset_name, dataset_results in results.items():
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            
            strategies = list(dataset_results.keys())
            
            # Multiple metrics visualization
            metrics = ['train_loss', 'eval_loss', 'accuracy', 'centroid_distance', 'vocab_coverage', 'selection_time']
            
            for i, metric in enumerate(metrics):
                ax = axes[i // 3, i % 3]
                values = [dataset_results[strategy].get(metric, 0) for strategy in strategies]
                
                bars = ax.bar(strategies, values, alpha=0.7)
                ax.set_title(f'{metric.replace("_", " ").title()}')
                ax.set_ylabel(metric)
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
                
                # Color code bars
                colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
                for bar, color in zip(bars, colors):
                    bar.set_color(color)
            
            plt.suptitle(f'Detailed Analysis - {dataset_name}', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{dataset_name}_detailed_analysis.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()


# =============================================================================
# Example Usage and Test Cases
# =============================================================================

def create_sample_datasets() -> Dict[str, Dict]:
    """Create sample datasets for testing"""
    
    # Sample text data
    sample_texts = [
        "The quick brown fox jumps over the lazy dog. This is a simple sentence.",
        "Machine learning is transforming how we process and understand data.",
        "Natural language processing involves understanding human language computationally.",
        "Deep learning models require large amounts of training data to perform well.",
        "Transformers have revolutionized the field of natural language processing.",
        "BERT and GPT are popular transformer-based language models.",
        "Coreset selection aims to find representative subsets of data.",
        "Dataset distillation is an important technique for efficient training.",
        "Large language models can perform various downstream tasks effectively.",
        "Text classification is one of the fundamental NLP tasks."
    ]
    
    # Extend with variations
    extended_texts = sample_texts * 10  # Create larger dataset
    for i in range(len(extended_texts)):
        extended_texts[i] = f"Sample {i}: {extended_texts[i]}"
    
    # Create labels for classification
    labels = [i % 3 for i in range(len(extended_texts))]  # 3-class classification
    
    # Create datasets
    datasets = {
        'text_classification': {
            'train': ClassificationDataset(extended_texts[:80], labels[:80]),
            'test': ClassificationDataset(extended_texts[80:], labels[80:])
        },
        'causal_lm': {
            'train': CausalLMDataset(extended_texts[:80], tokenizer_name='gpt2'),
            'test': CausalLMDataset(extended_texts[80:], tokenizer_name='gpt2')
        }
    }
    
    return datasets

def create_strategies_config() -> Dict[str, Any]:
    """Create comprehensive strategies configuration"""
    
    return {
        'configurations': [
            {
                'name': 'gradient_based',
                'strategies': ['gradient_magnitude', 'gradient_diversity'],
                'weights': [0.6, 0.4],
                'budget_ratio': 0.3
            },
            {
                'name': 'embedding_based',
                'strategies': ['embedding_diversity', 'embedding_coverage'],
                'weights': [0.5, 0.5],
                'budget_ratio': 0.3
            },
            {
                'name': 'uncertainty_based',
                'strategies': ['perplexity', 'entropy', 'confidence'],
                'weights': [0.4, 0.3, 0.3],
                'budget_ratio': 0.3
            },
            {
                'name': 'comprehensive',
                'strategies': ['gradient_magnitude', 'embedding_diversity', 'perplexity', 'complexity'],
                'weights': [0.3, 0.3, 0.2, 0.2],
                'budget_ratio': 0.3
            },
            {
                'name': 'text_specific',
                'strategies': ['complexity', 'length_diversity', 'semantic_diversity'],
                'weights': [0.4, 0.3, 0.3],
                'budget_ratio': 0.3
            }
        ]
    }

def run_llm_coreset_demo():
    """Run demonstration of LLM coreset framework"""
    
    logger.info("Starting LLM Coreset Demo...")
    
    # Create framework
    framework = LLMCoresetFramework(model_name='bert-base-uncased', device='auto')
    
    # Create sample datasets
    datasets = create_sample_datasets()
    
    # Create strategies configuration
    strategies_config = create_strategies_config()
    
    # Run experiments
    results = framework.run_coreset_experiments(
        datasets=datasets,
        strategies_config=strategies_config,
        output_dir='./llm_coreset_demo_results'
    )
    
    # Print summary
    logger.info("Demo completed successfully!")
    logger.info("Results summary:")
    
    for dataset_name, dataset_results in results.items():
        logger.info(f"\nDataset: {dataset_name}")
        for strategy_name, strategy_results in dataset_results.items():
            logger.info(f"  {strategy_name}:")
            logger.info(f"    Train Loss: {strategy_results.get('train_loss', 'N/A'):.4f}")
            logger.info(f"    Accuracy: {strategy_results.get('accuracy', 'N/A'):.4f}")
            logger.info(f"    Selection Time: {strategy_results.get('selection_time', 'N/A'):.2f}s")


# =============================================================================
# Command Line Interface
# =============================================================================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='LLM Coreset Selection Framework')
    
    parser.add_argument('--model', type=str, default='bert-base-uncased',
                       help='Model name for coreset selection')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--output_dir', type=str, default='./llm_coreset_results',
                       help='Output directory for results')
    parser.add_argument('--demo', action='store_true',
                       help='Run demo with sample data')
    parser.add_argument('--strategies', type=str, nargs='+',
                       default=['gradient_magnitude', 'embedding_diversity', 'perplexity'],
                       help='Strategies to use for coreset selection')
    parser.add_argument('--budget_ratio', type=float, default=0.3,
                       help='Coreset budget as fraction of dataset size')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    if args.demo:
        run_llm_coreset_demo()
    else:
        logger.info("Use --demo flag to run the demonstration")
        logger.info("For custom datasets, use the LLMCoresetFramework class directly")