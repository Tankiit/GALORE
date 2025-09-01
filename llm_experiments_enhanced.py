"""
Enhanced LLM Training Experiments with Multiple Hypernetwork Configurations
===========================================================================

This module provides comprehensive experiments comparing different hypernetwork
architectures and selection strategies for LLM training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import time
import json
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    get_linear_schedule_with_warmup
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import existing modules
from llm_experiments import (
    LLMGaLoreSelector,
    TextDataset,
    train_llm_epoch,
    evaluate_llm
)
from llm_hypernetworks import (
    LLMMultiScoringHypernetwork,
    LLMCoresetSelector,
    LLMTrainingState,
    create_llm_hypernetwork,
    create_llm_scoring_functions
)
from llm_datasets import (
    create_wikitext_datasets,
    WikiText103Dataset,
    StratifiedDataset
)
from simple_expts import PerformanceProfiler
from config import ExperimentConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HypernetworkConfig:
    """Configuration for hypernetwork experiments"""
    name: str
    state_dim: int
    hidden_dim: int
    num_heads: int
    dropout: float
    scoring_functions: List[str]
    use_attention: bool = True
    use_value_network: bool = True
    temperature_range: Tuple[float, float] = (0.1, 2.0)


class EnhancedLLMSelector(LLMGaLoreSelector):
    """Enhanced selector with multiple hypernetwork support"""
    
    def __init__(self,
                 model: nn.Module,
                 tokenizer: Any,
                 train_dataset: Any,
                 val_dataset: Any,
                 hypernetwork_configs: List[HypernetworkConfig],
                 memory_budget_mb: int = 2000,
                 rank: int = 512,
                 device: str = None):
        
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            memory_budget_mb=memory_budget_mb,
            rank=rank,
            use_llm_hypernetwork=False
        )
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 
                                            'mps' if torch.backends.mps.is_available() else 'cpu')
        
        # Initialize multiple hypernetworks
        self.hypernetworks = {}
        self.selectors = {}
        self.optimizers = {}
        
        for config in hypernetwork_configs:
            self.init_hypernetwork_from_config(config)
            
        # Track performance for each hypernetwork
        self.performance_history = {config.name: [] for config in hypernetwork_configs}
        self.selection_stats = {config.name: {} for config in hypernetwork_configs}
        
    def init_hypernetwork_from_config(self, config: HypernetworkConfig):
        """Initialize a hypernetwork from configuration"""
        
        # Create scoring functions based on config
        all_scoring_functions = create_llm_scoring_functions(
            self.model, self.tokenizer, self.device
        )
        
        # Filter scoring functions based on config
        selected_functions = [
            sf for sf in all_scoring_functions 
            if sf.name in config.scoring_functions
        ]
        
        # Create hypernetwork
        hypernetwork = LLMMultiScoringHypernetwork(
            scoring_functions=selected_functions,
            state_dim=config.state_dim,
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout
        ).to(self.device)
        
        # Create selector
        selector = LLMCoresetSelector(
            hypernetwork=hypernetwork,
            scoring_functions=selected_functions,
            batch_size=8,
            gradient_accumulation_steps=4
        )
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            hypernetwork.parameters(),
            lr=0.001,
            weight_decay=0.01
        )
        
        # Store
        self.hypernetworks[config.name] = hypernetwork
        self.selectors[config.name] = selector
        self.optimizers[config.name] = optimizer
        
        logger.info(f"Initialized hypernetwork '{config.name}' with {len(selected_functions)} scoring functions")
        
    def select_with_hypernetwork(self,
                                 hypernetwork_name: str,
                                 budget: int,
                                 current_performance: float) -> Tuple[List[int], Dict[str, Any]]:
        """Select coreset using a specific hypernetwork"""
        
        if hypernetwork_name not in self.selectors:
            raise ValueError(f"Unknown hypernetwork: {hypernetwork_name}")
            
        # Compute metrics
        val_loader = DataLoader(self.val_dataset, batch_size=8, shuffle=False)
        metrics = self.compute_llm_metrics(val_loader)
        
        # Create training state
        llm_state = LLMTrainingState(
            epoch=self.epoch,
            loss=metrics['loss'],
            perplexity=metrics['perplexity'],
            gradient_norm=self._compute_gradient_norm(),
            learning_rate=0.001,
            tokens_seen=self.token_count,
            total_tokens=len(self.train_dataset) * 256,
            avg_sequence_length=metrics['tokens_per_sample'],
            vocab_coverage=len(self.vocab_seen) / self.tokenizer.vocab_size,
            performance_history=self.performance_history[hypernetwork_name][-10:] 
                              if len(self.performance_history[hypernetwork_name]) >= 10 
                              else [metrics['perplexity']] * 10,
            attention_entropy=0.5
        )
        
        # Select using hypernetwork
        selected_indices, selection_info = self.selectors[hypernetwork_name].select_coreset(
            dataset=self.train_dataset,
            budget=budget,
            training_state=llm_state,
            tokenizer=self.tokenizer,
            verbose=False
        )
        
        # Update statistics
        selection_info['hypernetwork'] = hypernetwork_name
        selection_info['perplexity'] = metrics['perplexity']
        
        # Track selection diversity
        if hypernetwork_name not in self.selection_stats or \
           'total_selected' not in self.selection_stats[hypernetwork_name]:
            self.selection_stats[hypernetwork_name] = {
                'total_selected': set(),
                'selection_counts': {}
            }
            
        for idx in selected_indices:
            self.selection_stats[hypernetwork_name]['total_selected'].add(idx)
            if idx not in self.selection_stats[hypernetwork_name]['selection_counts']:
                self.selection_stats[hypernetwork_name]['selection_counts'][idx] = 0
            self.selection_stats[hypernetwork_name]['selection_counts'][idx] += 1
            
        selection_info['diversity'] = len(self.selection_stats[hypernetwork_name]['total_selected']) / len(self.train_dataset)
        
        return selected_indices, selection_info


def compare_hypernetwork_strategies(
    model_name: str = "gpt2",
    dataset_name: str = "wikitext103_quality_stratified",
    max_samples: int = 2000,
    epochs: int = 5,
    batch_size: int = 8,
    coreset_budget: int = 500,
    device: str = None):
    """
    Compare different hypernetwork configurations
    """
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 
                            'mps' if torch.backends.mps.is_available() else 'cpu')
    
    logger.info(f"Comparing hypernetwork strategies on {device}")
    
    # Define hypernetwork configurations
    configs = [
        HypernetworkConfig(
            name="full",
            state_dim=15,
            hidden_dim=128,
            num_heads=4,
            dropout=0.1,
            scoring_functions=[
                "perplexity", "token_diversity", "attention_coverage",
                "gradient_alignment", "token_importance", "repetition_penalty"
            ]
        ),
        HypernetworkConfig(
            name="perplexity_focused",
            state_dim=15,
            hidden_dim=64,
            num_heads=2,
            dropout=0.1,
            scoring_functions=["perplexity", "token_diversity", "repetition_penalty"]
        ),
        HypernetworkConfig(
            name="gradient_focused",
            state_dim=15,
            hidden_dim=64,
            num_heads=2,
            dropout=0.1,
            scoring_functions=["gradient_alignment", "token_importance", "attention_coverage"]
        ),
        HypernetworkConfig(
            name="lightweight",
            state_dim=10,
            hidden_dim=32,
            num_heads=1,
            dropout=0.05,
            scoring_functions=["perplexity", "gradient_alignment"]
        )
    ]
    
    # Initialize profiler
    profiler = PerformanceProfiler(log_dir="./logs", experiment_name="llm_hypernetwork_comparison")
    
    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    if model_name == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = model.to(device)
    
    # Load dataset
    logger.info("Loading dataset...")
    all_datasets = create_wikitext_datasets(max_samples=max_samples)
    base_dataset = all_datasets.get(dataset_name, 
                                   WikiText103Dataset(split="train", max_samples=max_samples))
    val_dataset = all_datasets.get('wikitext103_val',
                                  WikiText103Dataset(split="validation", max_samples=max_samples//10))
    
    # Convert to text lists
    train_texts = [sample.text for sample in base_dataset.samples] if hasattr(base_dataset, 'samples') else \
                  [base_dataset[i].text for i in range(len(base_dataset))]
    val_texts = [sample.text for sample in val_dataset.samples] if hasattr(val_dataset, 'samples') else \
                [val_dataset[i].text for i in range(len(val_dataset))]
    
    # Create datasets
    train_dataset = TextDataset(train_texts[:max_samples], tokenizer, max_length=256)
    val_dataset = TextDataset(val_texts[:max_samples//10], tokenizer, max_length=256)
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Initialize enhanced selector
    selector = EnhancedLLMSelector(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        hypernetwork_configs=configs,
        memory_budget_mb=2000,
        rank=512,
        device=device
    )
    
    # Results tracking
    results = {config.name: {
        'train_losses': [],
        'val_perplexities': [],
        'selection_times': [],
        'training_times': [],
        'selection_info': []
    } for config in configs}
    
    # Add baseline (no hypernetwork)
    results['baseline'] = {
        'train_losses': [],
        'val_perplexities': [],
        'selection_times': [],
        'training_times': [],
        'selection_info': []
    }
    
    # Training loop
    for epoch in range(epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        logger.info(f"{'='*60}")
        
        for config in configs + [None]:  # None for baseline
            config_name = config.name if config else 'baseline'
            logger.info(f"\nTesting configuration: {config_name}")
            
            # Select coreset
            start_time = time.time()
            
            if config:
                # Use hypernetwork
                selected_indices, selection_info = selector.select_with_hypernetwork(
                    config.name,
                    budget=coreset_budget,
                    current_performance=results[config_name]['val_perplexities'][-1] 
                                      if results[config_name]['val_perplexities'] else float('inf')
                )
            else:
                # Baseline: random selection
                selected_indices = np.random.choice(
                    len(train_dataset), 
                    size=min(coreset_budget, len(train_dataset)),
                    replace=False
                ).tolist()
                selection_info = {'strategy': 'random', 'size': len(selected_indices)}
            
            selection_time = time.time() - start_time
            results[config_name]['selection_times'].append(selection_time)
            results[config_name]['selection_info'].append(selection_info)
            
            logger.info(f"Selected {len(selected_indices)} samples in {selection_time:.2f}s")
            
            # Create coreset dataloader
            coreset = Subset(train_dataset, selected_indices)
            train_loader = DataLoader(coreset, batch_size=batch_size, shuffle=True)
            
            # Setup optimizer and scheduler for this configuration
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
            total_steps = (len(train_loader) // 4) * epochs  # Accounting for gradient accumulation
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=total_steps // 10,
                num_training_steps=total_steps
            )
            
            # Train
            start_time = time.time()
            train_metrics = train_llm_epoch(
                model, train_loader, optimizer, scheduler, device,
                gradient_accumulation_steps=4
            )
            training_time = time.time() - start_time
            
            results[config_name]['train_losses'].append(train_metrics['avg_loss'])
            results[config_name]['training_times'].append(training_time)
            
            # Evaluate
            val_loader = DataLoader(val_dataset, batch_size=batch_size * 2)
            val_metrics = evaluate_llm(model, val_loader, device)
            results[config_name]['val_perplexities'].append(val_metrics['perplexity'])
            
            logger.info(f"Train Loss: {train_metrics['avg_loss']:.4f}")
            logger.info(f"Val Perplexity: {val_metrics['perplexity']:.2f}")
            logger.info(f"Training Time: {training_time:.2f}s")
            
            # Update hypernetwork if applicable
            if config and epoch > 0:
                prev_perplexity = results[config_name]['val_perplexities'][-2]
                curr_perplexity = val_metrics['perplexity']
                improvement = (prev_perplexity - curr_perplexity) / prev_perplexity
                
                if improvement > 0:
                    logger.info(f"Perplexity improved by {improvement*100:.2f}%")
    
    # Analysis and visualization
    logger.info(f"\n{'='*60}")
    logger.info("Experiment Summary")
    logger.info(f"{'='*60}")
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Perplexity over epochs
    ax1 = axes[0, 0]
    for config_name, config_results in results.items():
        ax1.plot(config_results['val_perplexities'], label=config_name, marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Perplexity')
    ax1.set_title('Perplexity Comparison')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Training loss
    ax2 = axes[0, 1]
    for config_name, config_results in results.items():
        ax2.plot(config_results['train_losses'], label=config_name, marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Training Loss')
    ax2.set_title('Training Loss Comparison')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Selection time
    ax3 = axes[1, 0]
    avg_selection_times = {name: np.mean(res['selection_times']) 
                          for name, res in results.items()}
    ax3.bar(avg_selection_times.keys(), avg_selection_times.values())
    ax3.set_ylabel('Average Selection Time (s)')
    ax3.set_title('Selection Efficiency')
    ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: Final perplexity comparison
    ax4 = axes[1, 1]
    final_perplexities = {name: res['val_perplexities'][-1] 
                         for name, res in results.items()}
    colors = ['green' if name != 'baseline' else 'gray' for name in final_perplexities.keys()]
    ax4.bar(final_perplexities.keys(), final_perplexities.values(), color=colors)
    ax4.set_ylabel('Final Perplexity')
    ax4.set_title('Final Performance Comparison')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('llm_hypernetwork_comparison.png', dpi=150)
    logger.info("Saved comparison plots to llm_hypernetwork_comparison.png")
    
    # Print summary statistics
    for config_name, config_results in results.items():
        logger.info(f"\n{config_name}:")
        logger.info(f"  Best Perplexity: {min(config_results['val_perplexities']):.2f}")
        logger.info(f"  Final Perplexity: {config_results['val_perplexities'][-1]:.2f}")
        logger.info(f"  Avg Selection Time: {np.mean(config_results['selection_times']):.2f}s")
        logger.info(f"  Avg Training Time: {np.mean(config_results['training_times']):.2f}s")
    
    # Save detailed results
    with open('llm_hypernetwork_comparison_results.json', 'w') as f:
        # Convert numpy values to Python types for JSON serialization
        json_results = {}
        for config_name, config_results in results.items():
            json_results[config_name] = {
                'train_losses': [float(x) for x in config_results['train_losses']],
                'val_perplexities': [float(x) for x in config_results['val_perplexities']],
                'selection_times': [float(x) for x in config_results['selection_times']],
                'training_times': [float(x) for x in config_results['training_times']],
                'best_perplexity': float(min(config_results['val_perplexities'])),
                'final_perplexity': float(config_results['val_perplexities'][-1])
            }
        
        json.dump({
            'model': model_name,
            'dataset': dataset_name,
            'config': {
                'epochs': epochs,
                'batch_size': batch_size,
                'coreset_budget': coreset_budget,
                'max_samples': max_samples
            },
            'results': json_results
        }, f, indent=2)
    
    logger.info("Saved detailed results to llm_hypernetwork_comparison_results.json")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced LLM experiments with hypernetwork comparison")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name")
    parser.add_argument("--dataset", type=str, default="wikitext103_quality_stratified",
                       help="Dataset name")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--coreset_budget", type=int, default=200, help="Coreset budget")
    parser.add_argument("--max_samples", type=int, default=1000, help="Max training samples")
    parser.add_argument("--device", type=str, default=None, help="Device to use")
    
    args = parser.parse_args()
    
    # Run comparison experiment
    results = compare_hypernetwork_strategies(
        model_name=args.model,
        dataset_name=args.dataset,
        max_samples=args.max_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        coreset_budget=args.coreset_budget,
        device=args.device
    )
    
    print("\nExperiment completed successfully!")
    print("Check llm_hypernetwork_comparison.png for visualizations")
    print("Check llm_hypernetwork_comparison_results.json for detailed results")