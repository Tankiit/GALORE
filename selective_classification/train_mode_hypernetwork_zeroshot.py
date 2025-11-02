#!/usr/bin/env python3
"""
MODE Hypernetwork Training with Zero-Shot Evaluation

Trains VLM (CLIP) with MODE-selected data and evaluates on:
- Zero-shot ImageNet classification
- COCO image-text retrieval (I2T and T2I)

Usage:
    python train_mode_hypernetwork_zeroshot.py --config datacomp_config.yaml
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from pathlib import Path
import argparse
import yaml
from dataclasses import dataclass
from typing import Dict, List, Optional
import json
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import time

# Import your MODE components
from run_vlm_experiment import (
    MODEHypernetwork,
    MultimodalBinaryStateEncoder,
    load_mode_with_vlm_adapter
)

# Try importing transformers for CLIP
try:
    from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    print("Warning: transformers not installed. Install with: pip install transformers")
    HAS_TRANSFORMERS = False


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class MODETrainingConfig:
    """Configuration for MODE hypernetwork training"""

    # Data
    data_dir: str = './datacomp_mode_cache'
    selected_indices_path: str = './datacomp_mode_cache/selected_indices.pt'

    # Model
    clip_model_name: str = 'openai/clip-vit-base-patch32'
    hypernetwork_state_dim: int = 20
    hypernetwork_hidden_dim: int = 128
    hypernetwork_num_layers: int = 3

    # Training
    num_epochs: int = 10
    batch_size: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100

    # MODE settings
    use_mode: bool = True
    reselection_frequency: int = 2  # Re-select every N epochs
    selection_ratio: float = 0.3

    # Evaluation
    eval_every_n_epochs: int = 1
    eval_batch_size: int = 64
    imagenet_val_path: Optional[str] = None
    coco_val_path: Optional[str] = None

    # Output
    output_dir: str = './mode_hypernetwork_output'
    save_every_n_epochs: int = 2
    log_strategy_weights: bool = True

    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# Zero-Shot Evaluation
# ============================================================================

class ZeroShotImageNetEvaluator:
    """
    Zero-shot classification on ImageNet.

    Given a trained CLIP model:
    1. Encode all 1000 ImageNet class names as text embeddings
    2. For each test image, find the closest text embedding
    3. Report top-1 and top-5 accuracy
    """

    def __init__(self, clip_model, processor, device='cuda'):
        self.clip_model = clip_model
        self.processor = processor
        self.device = device

        # ImageNet class names (simplified - use full list in production)
        self.imagenet_classes = self._load_imagenet_classes()

        # Pre-compute text embeddings
        self.text_embeddings = self._encode_class_names()

    def _load_imagenet_classes(self):
        """Load ImageNet class names"""
        # Simplified list - replace with actual ImageNet classes
        # In production, load from imagenet_classes.txt
        return [
            "tench", "goldfish", "great white shark", "tiger shark",
            "hammerhead", "electric ray", "stingray", "cock",
            # ... (1000 classes total)
            # For demo, we'll use a subset
        ] * 10  # Pad to ~1000

    def _encode_class_names(self):
        """Encode all class names into text embeddings"""
        print("Encoding ImageNet class names...")

        # Create prompts: "a photo of a {class}"
        prompts = [f"a photo of a {cls}" for cls in self.imagenet_classes]

        text_embeddings = []
        batch_size = 256

        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i+batch_size]
                inputs = self.processor(
                    text=batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)

                outputs = self.clip_model.get_text_features(**inputs)
                outputs = F.normalize(outputs, dim=-1)
                text_embeddings.append(outputs.cpu())

        return torch.cat(text_embeddings, dim=0).to(self.device)

    def evaluate(self, dataloader, top_k=(1, 5)):
        """
        Evaluate zero-shot classification accuracy.

        Args:
            dataloader: DataLoader yielding (images, labels)
            top_k: Tuple of K values for top-K accuracy

        Returns:
            Dict with accuracy metrics
        """
        self.clip_model.eval()

        correct_at_k = {k: 0 for k in top_k}
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Zero-shot eval"):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Encode images
                image_features = self.clip_model.get_image_features(pixel_values=images)
                image_features = F.normalize(image_features, dim=-1)

                # Compute similarity to all class text embeddings
                logits = image_features @ self.text_embeddings.T  # [batch, num_classes]

                # Get top-k predictions
                for k in top_k:
                    _, pred_indices = torch.topk(logits, k, dim=1)

                    # Check if true label is in top-k
                    correct = torch.any(pred_indices == labels.unsqueeze(1), dim=1)
                    correct_at_k[k] += correct.sum().item()

                total += labels.size(0)

        # Compute accuracies
        results = {
            f'top{k}_accuracy': 100.0 * correct_at_k[k] / total
            for k in top_k
        }

        return results


class COCORetrievalEvaluator:
    """
    Image-text retrieval on COCO.

    Two tasks:
    - Image-to-Text (I2T): Given image, retrieve matching caption
    - Text-to-Image (T2I): Given caption, retrieve matching image

    Metrics: Recall@1, Recall@5, Recall@10
    """

    def __init__(self, clip_model, processor, device='cuda'):
        self.clip_model = clip_model
        self.processor = processor
        self.device = device

    def evaluate(self, dataloader, recall_at=(1, 5, 10)):
        """
        Evaluate retrieval performance.

        Args:
            dataloader: DataLoader yielding (images, captions)
            recall_at: Tuple of K values for Recall@K

        Returns:
            Dict with I2T and T2I metrics
        """
        self.clip_model.eval()

        # Collect all embeddings
        print("Encoding images and captions...")
        image_embeds = []
        text_embeds = []

        with torch.no_grad():
            for images, captions in tqdm(dataloader, desc="Encoding"):
                # Encode images
                img_inputs = self.processor(
                    images=images,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                img_feats = self.clip_model.get_image_features(**img_inputs)
                img_feats = F.normalize(img_feats, dim=-1)
                image_embeds.append(img_feats.cpu())

                # Encode captions
                txt_inputs = self.processor(
                    text=captions,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)
                txt_feats = self.clip_model.get_text_features(**txt_inputs)
                txt_feats = F.normalize(txt_feats, dim=-1)
                text_embeds.append(txt_feats.cpu())

        image_embeds = torch.cat(image_embeds, dim=0).to(self.device)  # [N, D]
        text_embeds = torch.cat(text_embeds, dim=0).to(self.device)    # [N, D]

        # Compute similarity matrix
        print("Computing similarity matrix...")
        similarity = image_embeds @ text_embeds.T  # [N, N]

        # Image-to-Text retrieval
        i2t_results = self._compute_recall(similarity, recall_at, "I2T")

        # Text-to-Image retrieval
        t2i_results = self._compute_recall(similarity.T, recall_at, "T2I")

        results = {**i2t_results, **t2i_results}
        return results

    def _compute_recall(self, similarity, recall_at, prefix):
        """Compute Recall@K for a similarity matrix"""
        N = similarity.size(0)
        results = {}

        for k in recall_at:
            # Get top-k predictions for each query
            _, top_k_indices = torch.topk(similarity, k, dim=1)

            # Ground truth: diagonal (index i matches index i)
            gt_indices = torch.arange(N, device=similarity.device).unsqueeze(1)

            # Check if ground truth is in top-k
            correct = torch.any(top_k_indices == gt_indices, dim=1)
            recall = 100.0 * correct.sum().item() / N

            results[f'{prefix}_R@{k}'] = recall

        return results


# ============================================================================
# MODE Training Pipeline
# ============================================================================

class MODETrainer:
    """
    Main training pipeline with MODE hypernetwork selection.

    Workflow:
    1. Load selected indices from one-shot selection
    2. Train CLIP on selected subset
    3. Track binary state and strategy weights
    4. Periodically re-select data based on hypernetwork
    5. Evaluate zero-shot performance
    """

    def __init__(self, config: MODETrainingConfig):
        self.config = config
        self.device = config.device

        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

        # Load CLIP model
        print(f"Loading CLIP model: {config.clip_model_name}")
        if HAS_TRANSFORMERS:
            self.clip_model = CLIPModel.from_pretrained(config.clip_model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(config.clip_model_name)
        else:
            raise ImportError("transformers not installed")

        # Initialize MODE hypernetwork
        if config.use_mode:
            print("Initializing MODE hypernetwork...")
            self.hypernetwork = MODEHypernetwork(
                state_dim=config.hypernetwork_state_dim,
                num_strategies=7,
                hidden_dim=config.hypernetwork_hidden_dim,
                num_layers=config.hypernetwork_num_layers
            ).to(self.device)

            self.state_encoder = MultimodalBinaryStateEncoder()
        else:
            self.hypernetwork = None
            self.state_encoder = None

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.training_history = defaultdict(list)

        # Load selected indices
        self.selected_indices = self._load_selected_indices()

    def _load_selected_indices(self):
        """Load one-shot selected indices"""
        indices_path = Path(self.config.selected_indices_path)

        if not indices_path.exists():
            print(f"Warning: Selected indices not found at {indices_path}")
            print("Using all data (no MODE selection)")
            return None

        data = torch.load(indices_path)
        indices = data['indices'] if isinstance(data, dict) else data

        print(f"Loaded {len(indices)} selected indices")
        return indices

    def train(self, train_dataset, val_dataset=None):
        """
        Main training loop.

        Args:
            train_dataset: Full training dataset
            val_dataset: Optional validation dataset for evaluation
        """

        # Create subset from selected indices
        if self.selected_indices is not None:
            train_dataset = Subset(train_dataset, self.selected_indices)
            print(f"Training on {len(train_dataset)} selected samples")

        # Create dataloader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        # Optimizer
        optimizer = torch.optim.AdamW(
            self.clip_model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Training loop
        print(f"\n{'='*80}")
        print(f"Starting training for {self.config.num_epochs} epochs")
        print(f"{'='*80}\n")

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch

            # Train one epoch
            train_metrics = self._train_epoch(train_loader, optimizer)

            # Log metrics
            print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")

            # Track strategy weights if using MODE
            if self.config.use_mode and self.config.log_strategy_weights:
                strategy_weights = self._get_current_strategy_weights(train_loader)
                train_metrics['strategy_weights'] = strategy_weights
                self._log_strategy_weights(strategy_weights, epoch)

            # Evaluate
            if val_dataset is not None and (epoch + 1) % self.config.eval_every_n_epochs == 0:
                eval_metrics = self._evaluate(val_dataset)
                print(f"  Evaluation metrics:")
                for k, v in eval_metrics.items():
                    print(f"    {k}: {v:.2f}")
                train_metrics.update(eval_metrics)

            # Save checkpoint
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self._save_checkpoint(epoch)

            # Store history
            for k, v in train_metrics.items():
                if not isinstance(v, (list, dict)):
                    self.training_history[k].append(v)

        # Save final model and results
        self._save_final_results()

    def _train_epoch(self, dataloader, optimizer):
        """Train for one epoch"""
        self.clip_model.train()

        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {self.current_epoch+1}")

        for batch in progress_bar:
            images, captions = batch
            images = images.to(self.device)

            # Process inputs
            inputs = self.processor(
                text=captions,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)

            # Forward pass
            outputs = self.clip_model(**inputs, return_loss=True)
            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}

    def _get_current_strategy_weights(self, dataloader):
        """Get current strategy weights from hypernetwork"""
        if self.hypernetwork is None:
            return None

        self.hypernetwork.eval()

        # Compute binary state for current training state
        metrics = {
            'vision_loss': self.training_history.get('loss', [0])[-1],
            'text_loss': self.training_history.get('loss', [0])[-1],
            'alignment_loss': self.training_history.get('loss', [0])[-1],
            'gradient_norm': 1.0,
            'learning_rate': self.config.learning_rate,
        }

        state = self.state_encoder.encode_state(
            metrics,
            epoch=self.current_epoch,
            total_epochs=self.config.num_epochs
        )

        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            weights = self.hypernetwork(state_tensor).squeeze(0)

        # Convert to dict
        strategy_names = self.hypernetwork.strategy_names
        return {name: weights[i].item() for i, name in enumerate(strategy_names)}

    def _log_strategy_weights(self, weights, epoch):
        """Log strategy weights"""
        print(f"\n  Strategy Weights (Epoch {epoch+1}):")
        for name, weight in weights.items():
            print(f"    {name}: {weight:.3f}")

    def _evaluate(self, val_dataset):
        """Run zero-shot evaluation"""
        metrics = {}

        # ImageNet evaluation
        if self.config.imagenet_val_path:
            print("\n  Running ImageNet zero-shot evaluation...")
            imagenet_eval = ZeroShotImageNetEvaluator(
                self.clip_model,
                self.processor,
                self.device
            )
            imagenet_loader = DataLoader(val_dataset, batch_size=self.config.eval_batch_size)
            imagenet_results = imagenet_eval.evaluate(imagenet_loader)
            metrics.update(imagenet_results)

        # COCO retrieval evaluation
        if self.config.coco_val_path:
            print("\n  Running COCO retrieval evaluation...")
            coco_eval = COCORetrievalEvaluator(
                self.clip_model,
                self.processor,
                self.device
            )
            coco_loader = DataLoader(val_dataset, batch_size=self.config.eval_batch_size)
            coco_results = coco_eval.evaluate(coco_loader)
            metrics.update(coco_results)

        return metrics

    def _save_checkpoint(self, epoch):
        """Save training checkpoint"""
        checkpoint_path = Path(self.config.output_dir) / f'checkpoint_epoch_{epoch+1}.pt'

        checkpoint = {
            'epoch': epoch,
            'clip_model_state': self.clip_model.state_dict(),
            'training_history': dict(self.training_history),
            'config': self.config
        }

        if self.hypernetwork is not None:
            checkpoint['hypernetwork_state'] = self.hypernetwork.state_dict()

        torch.save(checkpoint, checkpoint_path)
        print(f"\n  Checkpoint saved: {checkpoint_path}")

    def _save_final_results(self):
        """Save final training results and plots"""
        results_path = Path(self.config.output_dir) / 'training_results.json'

        # Convert history to JSON-serializable format
        history_json = {
            k: [float(v) if isinstance(v, (int, float, np.number)) else v
                for v in vals]
            for k, vals in self.training_history.items()
        }

        with open(results_path, 'w') as f:
            json.dump(history_json, f, indent=2)

        print(f"\nTraining complete! Results saved to {results_path}")

        # Save final model
        final_model_path = Path(self.config.output_dir) / 'final_clip_model.pt'
        torch.save(self.clip_model.state_dict(), final_model_path)
        print(f"Final model saved to {final_model_path}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train MODE with zero-shot evaluation")
    parser.add_argument('--config', type=str, default='datacomp_config.yaml',
                       help='Path to config file')
    parser.add_argument('--use_mode', action='store_true', default=True,
                       help='Use MODE hypernetwork selection')
    parser.add_argument('--output_dir', type=str, default='./mode_hypernetwork_output',
                       help='Output directory')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')

    args = parser.parse_args()

    # Create config
    config = MODETrainingConfig(
        use_mode=args.use_mode,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs
    )

    # TODO: Load your actual dataset here
    # For now, using placeholder
    print("\n" + "="*80)
    print("NOTE: You need to implement dataset loading for your specific data")
    print("Replace the placeholder dataset with your actual DataComp/CC3M data")
    print("="*80 + "\n")

    # Placeholder dataset (replace with real data)
    class DummyDataset(Dataset):
        def __len__(self):
            return 1000

        def __getitem__(self, idx):
            # Return (image, caption) pair
            image = torch.randn(3, 224, 224)
            caption = f"a photo of an object {idx}"
            return image, caption

    train_dataset = DummyDataset()
    val_dataset = DummyDataset()

    # Create trainer
    trainer = MODETrainer(config)

    # Train
    trainer.train(train_dataset, val_dataset)


if __name__ == '__main__':
    main()
