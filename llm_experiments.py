"""
LLM Training Experiments with Hypernetwork-Based Coreset Selection
==================================================================

This module provides comprehensive experiments for training language models
with efficient data selection using LLM-specific hypernetworks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import time
import os
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    get_linear_schedule_with_warmup
)
import datasets
from collections import defaultdict

# Import dataset utilities
from llm_datasets import (
    create_wikitext_datasets,
    create_llm_dataloaders,
    WikiText103Dataset,
    StratifiedDataset,
    stratify_by_quality,
    stratify_by_length,
    stratify_by_topic,
    stratify_by_complexity
)

# Import components from main framework
from simple_expts import (
    RLGuidedGaLoreSelector,
    PerformanceProfiler,
    PhaseTransitionDetector,
    GaLore,
    SelectionStrategy
)

# Import LLM hypernetwork components
from llm_hypernetworks import (
    LLMMultiScoringHypernetwork,
    LLMCoresetSelector,
    LLMTrainingState,
    create_llm_hypernetwork,
    create_llm_scoring_functions
)

from config import ExperimentConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# LLM Dataset Handling
# =============================================================================

class TextDataset(Dataset):
    """Simple text dataset wrapper for LLM training"""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        # Tokenize with padding and truncation
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        # Remove batch dimension
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # For causal LM, labels are the same as input_ids
        labels = input_ids.clone()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class LLMGaLoreSelector(RLGuidedGaLoreSelector):
    """
    Extended selector for LLM training with specialized hypernetwork support
    """
    
    def __init__(self,
                 model: nn.Module,
                 tokenizer: Any,
                 train_dataset: Dataset,
                 val_dataset: Dataset,
                 memory_budget_mb: int = 2000,
                 rank: int = 512,
                 use_llm_hypernetwork: bool = True):
        
        # Initialize base selector
        super().__init__(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            memory_budget_mb=memory_budget_mb,
            rank=rank,
            use_hypernetwork=False  # We'll use LLM-specific hypernetwork
        )
        
        self.tokenizer = tokenizer
        self.use_llm_hypernetwork = use_llm_hypernetwork
        
        # Initialize LLM hypernetwork if enabled
        if use_llm_hypernetwork:
            self.init_llm_hypernetwork()
            
        # LLM-specific tracking
        self.perplexity_history = []
        self.token_count = 0
        self.vocab_seen = set()
        
    def init_llm_hypernetwork(self):
        """Initialize LLM-specific hypernetwork components"""
        try:
            # Create LLM hypernetwork and selector
            self.llm_hypernetwork, self.llm_selector = create_llm_hypernetwork(
                self.model, self.tokenizer, self.device
            )
            
            # Optimizer for hypernetwork
            self.llm_hypernet_optimizer = torch.optim.AdamW(
                self.llm_hypernetwork.parameters(),
                lr=0.001,
                weight_decay=0.01
            )
            
            logger.info("Initialized LLM hypernetwork with 6 specialized scoring functions")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM hypernetwork: {e}")
            self.use_llm_hypernetwork = False
            
    def compute_llm_metrics(self, loader: DataLoader) -> Dict[str, float]:
        """Compute LLM-specific metrics"""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                total_loss += loss.item() * batch['input_ids'].size(0)
                total_tokens += batch['attention_mask'].sum().item()
                
        avg_loss = total_loss / len(loader.dataset)
        perplexity = np.exp(avg_loss)
        
        return {
            'loss': avg_loss,
            'perplexity': perplexity,
            'tokens_per_sample': total_tokens / len(loader.dataset)
        }
    
    def select_coreset_llm(self,
                           budget: int,
                           current_performance: float) -> Tuple[List[int], Dict[str, Any]]:
        """
        Select coreset using LLM-specific hypernetwork
        """
        if not self.use_llm_hypernetwork:
            # Fall back to standard selection
            return self.select_coreset(budget, current_performance)
            
        # Compute LLM metrics
        val_loader = DataLoader(self.val_dataset, batch_size=8, shuffle=False)
        metrics = self.compute_llm_metrics(val_loader)
        
        # Create LLM training state
        avg_seq_length = metrics['tokens_per_sample']
        vocab_coverage = len(self.vocab_seen) / self.tokenizer.vocab_size
        
        llm_state = LLMTrainingState(
            epoch=self.epoch,
            loss=metrics['loss'],
            perplexity=metrics['perplexity'],
            gradient_norm=self._compute_gradient_norm(),
            learning_rate=0.001,  # Should be passed from optimizer
            tokens_seen=self.token_count,
            total_tokens=len(self.train_dataset) * int(avg_seq_length),
            avg_sequence_length=avg_seq_length,
            vocab_coverage=vocab_coverage,
            performance_history=self.perplexity_history[-10:] if len(self.perplexity_history) >= 10 else [metrics['perplexity']] * 10,
            attention_entropy=0.5  # Placeholder
        )
        
        # Select using LLM hypernetwork
        selected_indices, selection_info = self.llm_selector.select_coreset(
            dataset=self.train_dataset,
            budget=budget,
            training_state=llm_state,
            tokenizer=self.tokenizer,
            verbose=True
        )
        
        # Update histories
        self.selection_history.extend(selected_indices)
        self.perplexity_history.append(metrics['perplexity'])
        self.epoch += 1
        
        # Enhanced selection info
        selection_info.update({
            'perplexity': metrics['perplexity'],
            'avg_sequence_length': avg_seq_length,
            'vocab_coverage': vocab_coverage,
            'strategy': 'llm_hypernetwork'
        })
        
        return selected_indices, selection_info
    
    def _compute_gradient_norm(self) -> float:
        """Compute total gradient norm"""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5
    
    def update_llm_hypernetwork(self,
                                prev_metrics: Dict[str, float],
                                curr_metrics: Dict[str, float],
                                budget: int = 1000) -> float:
        """Update LLM hypernetwork based on performance"""
        if not self.use_llm_hypernetwork:
            return 0.0
            
        # Compute performance delta
        perplexity_improvement = (prev_metrics['perplexity'] - curr_metrics['perplexity']) / \
                                (prev_metrics['perplexity'] + 1e-8)
        loss_improvement = (prev_metrics['loss'] - curr_metrics['loss']) / \
                          (prev_metrics['loss'] + 1e-8)
        
        performance_delta = 0.5 * perplexity_improvement + 0.5 * loss_improvement
        
        # Create states
        prev_state = LLMTrainingState(
            epoch=self.epoch - 1,
            loss=prev_metrics['loss'],
            perplexity=prev_metrics['perplexity'],
            gradient_norm=self._compute_gradient_norm(),
            learning_rate=0.001,
            tokens_seen=self.token_count - budget,
            total_tokens=len(self.train_dataset) * 256,
            avg_sequence_length=256,
            vocab_coverage=len(self.vocab_seen) / self.tokenizer.vocab_size,
            performance_history=self.perplexity_history[-10:],
            attention_entropy=0.5
        )
        
        curr_state = LLMTrainingState(
            epoch=self.epoch,
            loss=curr_metrics['loss'],
            perplexity=curr_metrics['perplexity'],
            gradient_norm=self._compute_gradient_norm(),
            learning_rate=0.001,
            tokens_seen=self.token_count,
            total_tokens=len(self.train_dataset) * 256,
            avg_sequence_length=256,
            vocab_coverage=len(self.vocab_seen) / self.tokenizer.vocab_size,
            performance_history=self.perplexity_history[-10:],
            attention_entropy=0.5
        )
        
        # Update hypernetwork
        loss = self.llm_selector.update_with_feedback(
            prev_state, curr_state, performance_delta, self.llm_hypernet_optimizer
        )
        
        return loss


# =============================================================================
# LLM Training Functions
# =============================================================================

def train_llm_epoch(model: nn.Module,
                   train_loader: DataLoader,
                   optimizer: torch.optim.Optimizer,
                   scheduler: Any,
                   device: str,
                   gradient_accumulation_steps: int = 4) -> Dict[str, float]:
    """Train LLM for one epoch"""
    model.train()
    total_loss = 0
    total_tokens = 0
    step = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for batch_idx, batch in enumerate(progress_bar):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss / gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update weights
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
        
        # Track metrics
        total_loss += loss.item() * gradient_accumulation_steps
        total_tokens += batch['attention_mask'].sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': loss.item() * gradient_accumulation_steps,
            'ppl': np.exp(loss.item() * gradient_accumulation_steps)
        })
    
    return {
        'avg_loss': total_loss / len(train_loader),
        'perplexity': np.exp(total_loss / len(train_loader)),
        'total_tokens': total_tokens
    }


def evaluate_llm(model: nn.Module,
                eval_loader: DataLoader,
                device: str) -> Dict[str, float]:
    """Evaluate LLM on validation set"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            
            total_loss += loss.item() * batch['input_ids'].size(0)
            total_tokens += batch['attention_mask'].sum().item()
    
    avg_loss = total_loss / len(eval_loader.dataset)
    perplexity = np.exp(avg_loss)
    
    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'tokens_per_sample': total_tokens / len(eval_loader.dataset)
    }


# =============================================================================
# Main LLM Experiment Runner
# =============================================================================

def run_llm_experiment(model_name: str = "gpt2",
                      dataset_name: str = "wikitext103_full",
                      data_subset: str = None,
                      max_samples: int = 10000,
                      epochs: int = 5,
                      batch_size: int = 8,
                      coreset_budget: int = 1000,
                      use_llm_hypernetwork: bool = True,
                      use_stratified_dataset: bool = True,
                      stratification_type: str = "quality",
                      device: str = None,
                      config: ExperimentConfig = None):
    """
    Run LLM training experiment with hypernetwork-based coreset selection
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 
                            'mps' if torch.backends.mps.is_available() else 'cpu')
    
    logger.info(f"Running LLM experiment on {device}")
    logger.info(f"Model: {model_name}, Dataset: {dataset_name}")
    
    # Initialize profiler
    profiler = PerformanceProfiler(log_dir="./logs", experiment_name=f"llm_{model_name}")
    
    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    with profiler.timer("model_loading"):
        if model_name == "gpt2":
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            model = GPT2LMHeadModel.from_pretrained(model_name)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = model.to(device)
    
    # Load dataset
    logger.info(f"Loading {dataset_name} dataset...")
    with profiler.timer("dataset_loading"):
        if use_stratified_dataset or dataset_name.startswith("wikitext103"):
            # Use our stratified WikiText-103 datasets
            logger.info(f"Using stratified dataset: {dataset_name}")
            
            # Create or load stratified datasets
            all_datasets = create_wikitext_datasets(max_samples=max_samples)
            
            # Select the specific dataset variant
            if dataset_name in all_datasets:
                base_dataset = all_datasets[dataset_name]
            else:
                # Create base dataset and stratify on the fly
                base_dataset = WikiText103Dataset(split="train", max_samples=max_samples)
                
                if stratification_type == "quality":
                    base_dataset = stratify_by_quality(base_dataset, bins=10)
                elif stratification_type == "length":
                    base_dataset = stratify_by_length(base_dataset, 
                                                     bins=['short', 'medium', 'long'])
                elif stratification_type == "topic":
                    base_dataset = stratify_by_topic(base_dataset,
                                                    domains=['science', 'history', 'culture'])
                elif stratification_type == "complexity":
                    base_dataset = stratify_by_complexity(base_dataset, bins=5)
            
            # Get validation dataset
            val_dataset = all_datasets.get('wikitext103_val', 
                                          WikiText103Dataset(split="validation", 
                                                           max_samples=max_samples//10))
            
            # Convert to text lists for compatibility
            train_texts = [sample.text for sample in base_dataset.samples 
                          if hasattr(base_dataset, 'samples')]
            if not train_texts:
                train_texts = [base_dataset[i].text for i in range(len(base_dataset))]
            
            val_texts = [sample.text for sample in val_dataset.samples 
                        if hasattr(val_dataset, 'samples')]
            if not val_texts:
                val_texts = [val_dataset[i].text for i in range(len(val_dataset))]
            
            # Log dataset statistics
            if isinstance(base_dataset, StratifiedDataset):
                stats = base_dataset.get_stratum_stats()
                logger.info(f"Stratification stats: {stats}")
                
        elif dataset_name == "wikitext" and data_subset:
            # Fallback to standard wikitext loading
            dataset = datasets.load_dataset(dataset_name, data_subset)
            train_texts = dataset['train']['text'][:max_samples]
            val_texts = dataset['validation']['text'][:max_samples//10]
        else:
            # Custom dataset loading
            train_texts = ["Sample text " * 10] * max_samples
            val_texts = ["Validation text " * 10] * (max_samples // 10)
        
        # Create datasets
        train_dataset = TextDataset(train_texts, tokenizer, max_length=256)
        val_dataset = TextDataset(val_texts, tokenizer, max_length=256)
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Initialize selector
    logger.info("Initializing LLM GaLore selector...")
    selector = LLMGaLoreSelector(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        memory_budget_mb=config.memory_budget_mb if config else 2000,
        rank=config.rank if config else 512,
        use_llm_hypernetwork=use_llm_hypernetwork
    )
    
    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    total_steps = (coreset_budget // batch_size) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )
    
    # Training loop
    results = {
        'train_losses': [],
        'val_perplexities': [],
        'selection_info': [],
        'training_times': []
    }
    
    best_perplexity = float('inf')
    prev_metrics = None
    
    for epoch in range(epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        logger.info(f"{'='*60}")
        
        # Select coreset
        with profiler.timer("coreset_selection"):
            if use_llm_hypernetwork:
                selected_indices, selection_info = selector.select_coreset_llm(
                    budget=coreset_budget,
                    current_performance=results['val_perplexities'][-1] if results['val_perplexities'] else float('inf')
                )
            else:
                selected_indices, selection_info = selector.select_coreset(
                    budget=coreset_budget,
                    current_performance=results['val_perplexities'][-1] if results['val_perplexities'] else 0.0
                )
        
        results['selection_info'].append(selection_info)
        logger.info(f"Selected {len(selected_indices)} samples using {selection_info.get('strategy', 'unknown')} strategy")
        
        # Create coreset dataloader
        coreset = Subset(train_dataset, selected_indices)
        train_loader = DataLoader(coreset, batch_size=batch_size, shuffle=True)
        
        # Train on coreset
        with profiler.timer("training"):
            train_metrics = train_llm_epoch(
                model, train_loader, optimizer, scheduler, device,
                gradient_accumulation_steps=4
            )
        
        results['train_losses'].append(train_metrics['avg_loss'])
        
        # Evaluate
        val_loader = DataLoader(val_dataset, batch_size=batch_size * 2)
        with profiler.timer("evaluation"):
            val_metrics = evaluate_llm(model, val_loader, device)
        
        results['val_perplexities'].append(val_metrics['perplexity'])
        
        # Update hypernetwork
        if use_llm_hypernetwork and prev_metrics is not None:
            with profiler.timer("hypernetwork_update"):
                hypernet_loss = selector.update_llm_hypernetwork(prev_metrics, val_metrics, coreset_budget)
                logger.info(f"Hypernetwork loss: {hypernet_loss:.4f}")
        
        prev_metrics = val_metrics
        
        # Log results
        logger.info(f"Train Loss: {train_metrics['avg_loss']:.4f}")
        logger.info(f"Val Perplexity: {val_metrics['perplexity']:.2f}")
        
        # Save best model
        if val_metrics['perplexity'] < best_perplexity:
            best_perplexity = val_metrics['perplexity']
            logger.info(f"New best perplexity: {best_perplexity:.2f}")
            # Save model checkpoint
            if config and config.save_models:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_perplexity': best_perplexity,
                }, f"./checkpoints/llm_{model_name}_best.pt")
    
    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info("Experiment Summary")
    logger.info(f"{'='*60}")
    logger.info(f"Best Perplexity: {best_perplexity:.2f}")
    logger.info(f"Final Perplexity: {results['val_perplexities'][-1]:.2f}")
    logger.info(f"Average Training Time: {np.mean(profiler.timings['training']):.2f}s")
    
    # Save results
    import json
    with open(f"llm_{model_name}_results.json", 'w') as f:
        json.dump({
            'model': model_name,
            'dataset': dataset_name,
            'config': {
                'epochs': epochs,
                'batch_size': batch_size,
                'coreset_budget': coreset_budget,
                'use_llm_hypernetwork': use_llm_hypernetwork
            },
            'results': {
                'best_perplexity': float(best_perplexity),
                'final_perplexity': float(results['val_perplexities'][-1]),
                'train_losses': [float(x) for x in results['train_losses']],
                'val_perplexities': [float(x) for x in results['val_perplexities']]
            }
        }, f, indent=2)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run LLM experiments with hypernetwork")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name")
    parser.add_argument("--dataset", type=str, default="wikitext103_quality_stratified", 
                       choices=['wikitext103_full', 'wikitext103_quality_stratified',
                               'wikitext103_length_stratified', 'wikitext103_domain_stratified',
                               'wikitext103_complexity_stratified', 'wikitext'],
                       help="Dataset name")
    parser.add_argument("--stratification", type=str, default="quality",
                       choices=['quality', 'length', 'topic', 'complexity'],
                       help="Stratification type for custom datasets")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--coreset_budget", type=int, default=1000, help="Coreset budget")
    parser.add_argument("--max_samples", type=int, default=10000, help="Max training samples")
    parser.add_argument("--use_hypernetwork", action="store_true", help="Use LLM hypernetwork")
    parser.add_argument("--use_stratified", action="store_true", default=True,
                       help="Use stratified datasets")
    parser.add_argument("--device", type=str, default=None, help="Device to use")
    
    args = parser.parse_args()
    
    # Run experiment
    results = run_llm_experiment(
        model_name=args.model,
        dataset_name=args.dataset,
        max_samples=args.max_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        coreset_budget=args.coreset_budget,
        use_llm_hypernetwork=args.use_hypernetwork,
        use_stratified_dataset=args.use_stratified,
        stratification_type=args.stratification,
        device=args.device
    )
    
    print(f"\nExperiment completed successfully!")
    print(f"Best perplexity: {min(results['val_perplexities']):.2f}")