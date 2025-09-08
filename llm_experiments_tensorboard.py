"""
LLM Training Experiments with TensorBoard Logging
=================================================

Enhanced LLM experiments with comprehensive TensorBoard visualization support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import time
import json
from pathlib import Path
from datetime import datetime
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
from collections import defaultdict

# Import existing modules
from llm_experiments import (
    LLMGaLoreSelector,
    TextDataset,
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


class TensorBoardLogger:
    """Enhanced TensorBoard logger for LLM experiments"""
    
    def __init__(self, log_dir: str, experiment_name: str):
        self.log_dir = Path(log_dir) / experiment_name / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writers = {}
        self.global_steps = defaultdict(int)
        self.experiment_name = experiment_name
        
    def get_writer(self, config_name: str) -> SummaryWriter:
        """Get or create a writer for a specific configuration"""
        if config_name not in self.writers:
            writer_dir = self.log_dir / config_name
            self.writers[config_name] = SummaryWriter(writer_dir)
            logger.info(f"Created TensorBoard writer at {writer_dir}")
        return self.writers[config_name]
    
    def log_scalars(self, config_name: str, scalars: Dict[str, float], step: int = None):
        """Log multiple scalar values"""
        writer = self.get_writer(config_name)
        if step is None:
            step = self.global_steps[config_name]
            self.global_steps[config_name] += 1
        
        for name, value in scalars.items():
            writer.add_scalar(name, value, step)
    
    def log_histogram(self, config_name: str, name: str, values: torch.Tensor, step: int = None):
        """Log histogram of values"""
        writer = self.get_writer(config_name)
        if step is None:
            step = self.global_steps[config_name]
        writer.add_histogram(name, values, step)
    
    def log_hypernetwork_weights(self, config_name: str, weights: Dict[str, float], step: int):
        """Log hypernetwork scoring function weights"""
        writer = self.get_writer(config_name)
        for func_name, weight in weights.items():
            writer.add_scalar(f'hypernetwork/weights/{func_name}', weight, step)
    
    def log_selection_metrics(self, config_name: str, metrics: Dict[str, Any], step: int):
        """Log coreset selection metrics"""
        writer = self.get_writer(config_name)
        
        # Log selection statistics
        if 'diversity' in metrics:
            writer.add_scalar('selection/diversity', metrics['diversity'], step)
        if 'selection_time' in metrics:
            writer.add_scalar('selection/time_seconds', metrics['selection_time'], step)
        if 'strategy' in metrics:
            writer.add_text('selection/strategy', metrics['strategy'], step)
        if 'temperature' in metrics:
            writer.add_scalar('selection/temperature', metrics['temperature'], step)
        if 'value' in metrics:
            writer.add_scalar('selection/value_estimate', metrics['value'], step)
    
    def log_training_metrics(self, config_name: str, epoch: int, batch_idx: int, 
                            loss: float, perplexity: float, learning_rate: float,
                            gradient_norm: float = None):
        """Log detailed training metrics"""
        writer = self.get_writer(config_name)
        global_step = epoch * 1000 + batch_idx  # Adjust based on your needs
        
        writer.add_scalar('train/loss', loss, global_step)
        writer.add_scalar('train/perplexity', perplexity, global_step)
        writer.add_scalar('train/learning_rate', learning_rate, global_step)
        
        if gradient_norm is not None:
            writer.add_scalar('train/gradient_norm', gradient_norm, global_step)
    
    def log_validation_metrics(self, config_name: str, epoch: int, metrics: Dict[str, float]):
        """Log validation metrics"""
        writer = self.get_writer(config_name)
        
        writer.add_scalar('val/loss', metrics['loss'], epoch)
        writer.add_scalar('val/perplexity', metrics['perplexity'], epoch)
        writer.add_scalar('val/tokens_per_sample', metrics.get('tokens_per_sample', 0), epoch)
    
    def log_comparison_chart(self, results: Dict[str, Dict], epoch: int):
        """Create and log comparison charts"""
        # Create comparison figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Perplexity comparison
        ax1 = axes[0, 0]
        for config_name, config_results in results.items():
            if 'val_perplexities' in config_results and config_results['val_perplexities']:
                ax1.bar(config_name, config_results['val_perplexities'][-1])
        ax1.set_ylabel('Validation Perplexity')
        ax1.set_title(f'Perplexity at Epoch {epoch}')
        ax1.tick_params(axis='x', rotation=45)
        
        # Loss comparison
        ax2 = axes[0, 1]
        for config_name, config_results in results.items():
            if 'train_losses' in config_results and config_results['train_losses']:
                ax2.bar(config_name, config_results['train_losses'][-1])
        ax2.set_ylabel('Training Loss')
        ax2.set_title(f'Training Loss at Epoch {epoch}')
        ax2.tick_params(axis='x', rotation=45)
        
        # Selection time comparison
        ax3 = axes[1, 0]
        for config_name, config_results in results.items():
            if 'selection_times' in config_results and config_results['selection_times']:
                ax3.bar(config_name, np.mean(config_results['selection_times']))
        ax3.set_ylabel('Avg Selection Time (s)')
        ax3.set_title('Selection Efficiency')
        ax3.tick_params(axis='x', rotation=45)
        
        # Training time comparison
        ax4 = axes[1, 1]
        for config_name, config_results in results.items():
            if 'training_times' in config_results and config_results['training_times']:
                ax4.bar(config_name, np.mean(config_results['training_times']))
        ax4.set_ylabel('Avg Training Time (s)')
        ax4.set_title('Training Efficiency')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Convert to tensor and log
        fig.canvas.draw()
        # Use buffer_rgba() for compatibility
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)
        # Convert RGBA to RGB
        img = img[:, :, :3]
        img_tensor = torch.tensor(img).permute(2, 0, 1)
        
        writer = self.get_writer('comparison')
        writer.add_image('comparison/metrics', img_tensor, epoch)
        plt.close()
    
    def log_hyperparameters(self, config_name: str, hparams: Dict, metrics: Dict):
        """Log hyperparameters and their corresponding metrics"""
        # Note: Skipping add_hparams due to NumPy 2.0 compatibility issues
        # Log as text and scalars instead
        writer = self.get_writer(config_name)
        writer.add_text('hyperparameters', str(hparams), 0)
        for key, value in metrics.items():
            writer.add_scalar(f'hparam_metrics/{key}', value, 0)
    
    def log_model_gradients(self, config_name: str, model: nn.Module, step: int):
        """Log gradient statistics for model parameters"""
        writer = self.get_writer(config_name)
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                writer.add_histogram(f'gradients/{name}', param.grad, step)
                writer.add_scalar(f'gradients/{name}_norm', param.grad.norm().item(), step)
    
    def log_attention_weights(self, config_name: str, attention_weights: torch.Tensor, step: int):
        """Log attention weight visualizations"""
        writer = self.get_writer(config_name)
        
        if attention_weights is not None and len(attention_weights.shape) >= 2:
            # Create heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(attention_weights.cpu().numpy(), cmap='Blues', ax=ax)
            ax.set_title('Attention Weights')
            
            # Convert to tensor
            fig.canvas.draw()
            # Use buffer_rgba() for compatibility
            buf = fig.canvas.buffer_rgba()
            img = np.asarray(buf)
            # Convert RGBA to RGB
            img = img[:, :, :3]
            img_tensor = torch.tensor(img).permute(2, 0, 1)
            
            writer.add_image(f'attention/weights', img_tensor, step)
            plt.close()
    
    def close(self):
        """Close all writers"""
        for writer in self.writers.values():
            writer.close()


def train_llm_epoch_with_logging(model: nn.Module,
                                 train_loader: DataLoader,
                                 optimizer: torch.optim.Optimizer,
                                 scheduler: Any,
                                 device: str,
                                 tb_logger: TensorBoardLogger,
                                 config_name: str,
                                 epoch: int,
                                 gradient_accumulation_steps: int = 4) -> Dict[str, float]:
    """Train LLM for one epoch with TensorBoard logging"""
    model.train()
    total_loss = 0
    total_tokens = 0
    batch_losses = []
    
    progress_bar = tqdm(train_loader, desc=f"Training {config_name}")
    
    for batch_idx, batch in enumerate(progress_bar):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss / gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Calculate gradient norm before clipping
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        
        # Update weights
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Log training metrics
            current_loss = loss.item() * gradient_accumulation_steps
            current_ppl = np.exp(current_loss)
            current_lr = scheduler.get_last_lr()[0]
            
            tb_logger.log_training_metrics(
                config_name=config_name,
                epoch=epoch,
                batch_idx=batch_idx,
                loss=current_loss,
                perplexity=current_ppl,
                learning_rate=current_lr,
                gradient_norm=grad_norm
            )
            
            # Log gradient histograms periodically
            if batch_idx % 50 == 0:
                tb_logger.log_model_gradients(config_name, model, epoch * len(train_loader) + batch_idx)
        
        # Track metrics
        batch_loss = loss.item() * gradient_accumulation_steps
        total_loss += batch_loss
        batch_losses.append(batch_loss)
        total_tokens += batch['attention_mask'].sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': batch_loss,
            'ppl': np.exp(batch_loss),
            'grad_norm': f'{grad_norm:.2f}'
        })
    
    avg_loss = total_loss / len(train_loader)
    
    # Log epoch summary
    tb_logger.log_scalars(config_name, {
        'epoch_summary/avg_loss': avg_loss,
        'epoch_summary/avg_perplexity': np.exp(avg_loss),
        'epoch_summary/total_tokens': total_tokens,
        'epoch_summary/loss_std': np.std(batch_losses)
    }, epoch)
    
    return {
        'avg_loss': avg_loss,
        'perplexity': np.exp(avg_loss),
        'total_tokens': total_tokens,
        'loss_std': np.std(batch_losses)
    }


class EnhancedLLMSelectorWithLogging(LLMGaLoreSelector):
    """LLM selector with comprehensive TensorBoard logging"""
    
    def __init__(self,
                 model: nn.Module,
                 tokenizer: Any,
                 train_dataset: Any,
                 val_dataset: Any,
                 tb_logger: TensorBoardLogger,
                 memory_budget_mb: int = 2000,
                 rank: int = 512,
                 use_llm_hypernetwork: bool = True):
        
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            memory_budget_mb=memory_budget_mb,
            rank=rank,
            use_llm_hypernetwork=use_llm_hypernetwork
        )
        
        self.tb_logger = tb_logger
        self.selection_step = 0
        
    def select_coreset_llm_with_logging(self,
                                        budget: int,
                                        current_performance: float,
                                        config_name: str) -> Tuple[List[int], Dict[str, Any]]:
        """Select coreset with detailed logging"""
        
        start_time = time.time()
        
        # Call parent method
        selected_indices, selection_info = self.select_coreset_llm(
            budget=budget,
            current_performance=current_performance
        )
        
        selection_time = time.time() - start_time
        selection_info['selection_time'] = selection_time
        
        # Log selection metrics
        self.tb_logger.log_selection_metrics(
            config_name=config_name,
            metrics=selection_info,
            step=self.selection_step
        )
        
        # Log hypernetwork weights if available
        if 'weights' in selection_info:
            self.tb_logger.log_hypernetwork_weights(
                config_name=config_name,
                weights=selection_info['weights'],
                step=self.selection_step
            )
        
        # Log selection distribution
        selection_counts = defaultdict(int)
        for idx in selected_indices:
            selection_counts[idx % 10] += 1  # Group by last digit for visualization
        
        self.tb_logger.log_scalars(
            config_name=f"{config_name}/selection_distribution",
            scalars={f'bin_{i}': count for i, count in selection_counts.items()},
            step=self.selection_step
        )
        
        self.selection_step += 1
        
        return selected_indices, selection_info


def run_llm_experiment_with_tensorboard(
    model_name: str = "gpt2",
    dataset_name: str = "wikitext103_quality_stratified",
    max_samples: int = 1000,
    epochs: int = 3,
    batch_size: int = 8,
    coreset_budget: int = 200,
    use_hypernetwork: bool = True,
    experiment_name: str = None,
    device: str = None):
    """
    Run LLM experiment with comprehensive TensorBoard logging
    """
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 
                            'mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Initialize TensorBoard logger
    if experiment_name is None:
        experiment_name = f"llm_{model_name}_{dataset_name}"
    
    tb_logger = TensorBoardLogger(log_dir="./tensorboard_logs", experiment_name=experiment_name)
    
    logger.info(f"Running LLM experiment with TensorBoard logging")
    logger.info(f"TensorBoard logs will be saved to: {tb_logger.log_dir}")
    logger.info(f"To view: tensorboard --logdir={tb_logger.log_dir.parent}")
    
    # Log hyperparameters
    hparams = {
        'model': model_name,
        'dataset': dataset_name,
        'max_samples': max_samples,
        'epochs': epochs,
        'batch_size': batch_size,
        'coreset_budget': coreset_budget,
        'use_hypernetwork': use_hypernetwork
    }
    
    # Initialize profiler
    profiler = PerformanceProfiler(log_dir="./logs", experiment_name=experiment_name)
    
    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    with profiler.timer("model_loading"):
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
    with profiler.timer("dataset_loading"):
        all_datasets = create_wikitext_datasets(max_samples=max_samples)
        
        if dataset_name in all_datasets:
            base_dataset = all_datasets[dataset_name]
        else:
            base_dataset = WikiText103Dataset(split="train", max_samples=max_samples)
        
        val_dataset = all_datasets.get('wikitext103_val',
                                      WikiText103Dataset(split="validation", 
                                                       max_samples=max_samples//10))
        
        # Convert to text lists
        train_texts = [sample.text for sample in base_dataset.samples] \
                     if hasattr(base_dataset, 'samples') else \
                     [base_dataset[i].text for i in range(len(base_dataset))]
        
        val_texts = [sample.text for sample in val_dataset.samples] \
                   if hasattr(val_dataset, 'samples') else \
                   [val_dataset[i].text for i in range(len(val_dataset))]
        
        # Create datasets
        train_dataset = TextDataset(train_texts[:max_samples], tokenizer, max_length=256)
        val_dataset = TextDataset(val_texts[:max_samples//10], tokenizer, max_length=256)
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Initialize selector with logging
    selector = EnhancedLLMSelectorWithLogging(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tb_logger=tb_logger,
        memory_budget_mb=2000,
        rank=512,
        use_llm_hypernetwork=use_hypernetwork
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
    config_name = "hypernetwork" if use_hypernetwork else "baseline"
    
    for epoch in range(epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        logger.info(f"{'='*60}")
        
        # Select coreset
        with profiler.timer("coreset_selection"):
            if use_hypernetwork:
                selected_indices, selection_info = selector.select_coreset_llm_with_logging(
                    budget=coreset_budget,
                    current_performance=results['val_perplexities'][-1] if results['val_perplexities'] else float('inf'),
                    config_name=config_name
                )
            else:
                # Baseline: random selection
                selected_indices = np.random.choice(
                    len(train_dataset),
                    size=min(coreset_budget, len(train_dataset)),
                    replace=False
                ).tolist()
                selection_info = {'strategy': 'random', 'size': len(selected_indices)}
                
                tb_logger.log_selection_metrics(
                    config_name=config_name,
                    metrics=selection_info,
                    step=epoch
                )
        
        results['selection_info'].append(selection_info)
        logger.info(f"Selected {len(selected_indices)} samples using {selection_info.get('strategy', 'unknown')} strategy")
        
        # Create coreset dataloader
        coreset = Subset(train_dataset, selected_indices)
        train_loader = DataLoader(coreset, batch_size=batch_size, shuffle=True)
        
        # Train on coreset
        start_time = time.time()
        with profiler.timer("training"):
            train_metrics = train_llm_epoch_with_logging(
                model, train_loader, optimizer, scheduler, device,
                tb_logger, config_name, epoch,
                gradient_accumulation_steps=4
            )
        training_time = time.time() - start_time
        
        results['train_losses'].append(train_metrics['avg_loss'])
        results['training_times'].append(training_time)
        
        # Evaluate
        val_loader = DataLoader(val_dataset, batch_size=batch_size * 2)
        with profiler.timer("evaluation"):
            val_metrics = evaluate_llm(model, val_loader, device)
        
        results['val_perplexities'].append(val_metrics['perplexity'])
        
        # Log validation metrics
        tb_logger.log_validation_metrics(config_name, epoch, val_metrics)
        
        # Log epoch summary
        epoch_summary = {
            'train_loss': train_metrics['avg_loss'],
            'val_perplexity': val_metrics['perplexity'],
            'training_time': training_time,
            'selection_time': selection_info.get('selection_time', 0)
        }
        
        tb_logger.log_scalars(f"{config_name}/epoch_summary", epoch_summary, epoch)
        
        logger.info(f"Train Loss: {train_metrics['avg_loss']:.4f}")
        logger.info(f"Val Perplexity: {val_metrics['perplexity']:.2f}")
        logger.info(f"Training Time: {training_time:.2f}s")
        
        # Save best model
        if val_metrics['perplexity'] < best_perplexity:
            best_perplexity = val_metrics['perplexity']
            logger.info(f"New best perplexity: {best_perplexity:.2f}")
            
            # Log best metrics
            tb_logger.log_scalars(f"{config_name}/best", {
                'perplexity': best_perplexity,
                'epoch': epoch
            }, 0)
    
    # Log final metrics (skip hyperparameters due to NumPy 2.0 compatibility issue)
    final_metrics = {
        'best_perplexity': best_perplexity,
        'final_perplexity': results['val_perplexities'][-1],
        'avg_training_time': np.mean(results['training_times'])
    }
    
    # Log final metrics as scalars instead
    tb_logger.log_scalars(f"{config_name}/final_metrics", final_metrics, epochs)
    
    # Create final comparison chart
    tb_logger.log_comparison_chart({'experiment': results}, epochs-1)
    
    # Save results
    results_path = Path(f"tensorboard_{experiment_name}_results.json")
    with open(results_path, 'w') as f:
        json.dump({
            'model': model_name,
            'dataset': dataset_name,
            'config': hparams,
            'results': {
                'best_perplexity': float(best_perplexity),
                'final_perplexity': float(results['val_perplexities'][-1]),
                'train_losses': [float(x) for x in results['train_losses']],
                'val_perplexities': [float(x) for x in results['val_perplexities']],
                'training_times': results['training_times']
            },
            'tensorboard_log_dir': str(tb_logger.log_dir)
        }, f, indent=2)
    
    logger.info(f"\nExperiment completed!")
    logger.info(f"Results saved to: {results_path}")
    logger.info(f"TensorBoard logs: {tb_logger.log_dir}")
    logger.info(f"Run: tensorboard --logdir={tb_logger.log_dir.parent}")
    
    tb_logger.close()
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM experiments with TensorBoard logging")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name")
    parser.add_argument("--dataset", type=str, default="wikitext103_quality_stratified",
                       help="Dataset name")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--coreset_budget", type=int, default=200, help="Coreset budget")
    parser.add_argument("--max_samples", type=int, default=1000, help="Max training samples")
    parser.add_argument("--use_hypernetwork", action="store_true", help="Use LLM hypernetwork")
    parser.add_argument("--experiment_name", type=str, default=None, help="Experiment name for logging")
    parser.add_argument("--device", type=str, default=None, help="Device to use")
    
    args = parser.parse_args()
    
    # Run experiment with TensorBoard logging
    results = run_llm_experiment_with_tensorboard(
        model_name=args.model,
        dataset_name=args.dataset,
        max_samples=args.max_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        coreset_budget=args.coreset_budget,
        use_hypernetwork=args.use_hypernetwork,
        experiment_name=args.experiment_name,
        device=args.device
    )
    
    print(f"\nTo view TensorBoard logs, run:")
    print(f"tensorboard --logdir=./tensorboard_logs")