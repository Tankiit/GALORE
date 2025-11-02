"""
MODE for DataComp Small (12.8M samples)
========================================

Production implementation for DataComp benchmark with MODE-VLM selection.

Target: Select 3.84M samples (30%) from 12.8M pool for CLIP training.
Expected: ~500 GPU hours for selection + 400 GPU hours for training on 4x A100.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import time
from tqdm import tqdm
import argparse
import json

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    raise ImportError("Please install: pip install datasets")

try:
    import open_clip
except ImportError:
    raise ImportError("Please install: pip install open-clip-torch")

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class DataCompSmallConfig:
    """Configuration for DataComp Small with MODE selection"""

    # DataComp Small specifications
    total_samples: int = 12_800_000
    selection_budget: float = 0.30  # Must keep 30% = 3.84M samples
    target_selected: int = 3_840_000

    # Models
    clip_model: str = 'ViT-B-32'  # Standard for DataComp
    proxy_model: str = 'ViT-B-32'

    # Selection (MODE scoring phase)
    selection_batch_size: int = 2048  # Large batches for throughput
    num_selection_workers: int = 8
    selection_device: str = 'cuda'

    # Scoring strategy weights
    nuclear_norm_weight: float = 0.3
    clip_similarity_weight: float = 0.7

    # Training (on selected subset)
    training_epochs: int = 32  # DataComp standard
    training_batch_size: int = 1024  # Per GPU
    learning_rate: float = 5e-4
    warmup_steps: int = 2000
    weight_decay: float = 0.2

    # Distributed training
    world_size: int = 4  # 4x A100 GPUs
    distributed: bool = True

    # Efficiency
    mixed_precision: bool = True
    gradient_checkpointing: bool = False

    # Caching
    cache_dir: str = "./datacomp_small_cache"
    save_scores: bool = True  # Save scores for analysis

    # Logging
    log_dir: str = "./runs/datacomp_small"
    checkpoint_freq: int = 10000  # Save every N steps
    eval_freq: int = 5000

    # DataComp dataset
    datacomp_scale: str = "small"
    datacomp_split: str = "train"
    streaming: bool = True  # Must be True for large datasets

    @classmethod
    def from_json(cls, path: str):
        """Load config from JSON file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def save(self, path: str):
        """Save config to JSON file"""
        import json
        from dataclasses import asdict
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)


# ============================================================================
# DataComp Small Dataset Loader
# ============================================================================

class DataCompSmallDataset:
    """
    Wrapper for DataComp Small dataset (12.8M samples).

    Uses HuggingFace datasets library with streaming for memory efficiency.
    """

    def __init__(self, config: DataCompSmallConfig, subset: Optional[List[int]] = None):
        self.config = config
        self.subset = subset

        print(f"Loading DataComp {config.datacomp_scale}...")
        print(f"  Total samples: {config.total_samples:,}")
        print(f"  Target selection: {config.target_selected:,} ({config.selection_budget*100:.0f}%)")

        # Load dataset
        # Note: DataComp uses custom dataset format, adjust as needed
        try:
            self.dataset = load_dataset(
                "mlfoundations/datacomp_small",  # Adjust to actual dataset path
                split=config.datacomp_split,
                streaming=config.streaming
            )
        except:
            print("Warning: Could not load official DataComp dataset.")
            print("Using local path fallback...")
            # Fallback to local path
            self.dataset = load_dataset(
                "webdataset",
                data_dir=config.cache_dir,
                split="train",
                streaming=True
            )

        # Initialize CLIP preprocessor
        _, _, self.preprocess = open_clip.create_model_and_transforms(
            config.clip_model, pretrained='openai'
        )

        print(f"Dataset loaded successfully!")

    def __len__(self):
        if self.subset is not None:
            return len(self.subset)
        return self.config.total_samples

    def __iter__(self):
        """Iterate over dataset (for streaming)"""
        if self.subset is not None:
            # Return only selected samples
            for idx in self.subset:
                yield self._get_sample(idx)
        else:
            # Return all samples
            for i, sample in enumerate(self.dataset):
                if i >= self.config.total_samples:
                    break
                yield self._process_sample(sample)

    def _process_sample(self, sample):
        """Process a single sample from dataset"""
        # DataComp format: {'image': PIL.Image, 'text': str, ...}
        image = self.preprocess(sample['image'])
        text = sample['text']

        return {
            'image': image,
            'text': text,
            'metadata': sample.get('metadata', {})
        }


# ============================================================================
# MODE Scorer for DataComp Small
# ============================================================================

class DataCompMODEScorer:
    """
    MODE-based scorer for DataComp Small.

    Computes hybrid scores (nuclear norm + CLIP similarity) for 12.8M samples.
    """

    def __init__(self, config: DataCompSmallConfig):
        self.config = config
        self.device = torch.device(config.selection_device if torch.cuda.is_available() else 'cpu')

        print(f"\nInitializing MODE scorer...")
        print(f"  Device: {self.device}")
        print(f"  Batch size: {config.selection_batch_size}")

        # Load proxy model for scoring
        print(f"  Loading proxy model: {config.proxy_model}")
        self.proxy_model, _, self.preprocess = open_clip.create_model_and_transforms(
            config.proxy_model, pretrained='openai'
        )
        self.proxy_model = self.proxy_model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(config.proxy_model)

        # Load checkpoint model (could be different or EMA)
        print(f"  Loading checkpoint model: {config.clip_model}")
        self.checkpoint_model, _, _ = open_clip.create_model_and_transforms(
            config.clip_model, pretrained='openai'
        )
        self.checkpoint_model = self.checkpoint_model.to(self.device).eval()

        # TensorBoard
        if TENSORBOARD_AVAILABLE:
            log_dir = Path(config.log_dir) / "scoring"
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(str(log_dir))
        else:
            self.writer = None

        self.global_step = 0
        print("MODE scorer ready!\n")

    @torch.no_grad()
    def compute_nuclear_norm_score(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        Compute nuclear norm importance score.

        Higher nuclear norm = more diverse/informative sample
        """
        # Nuclear norm = sum of singular values
        # For efficiency, approximate with Frobenius norm for large batches
        batch_size = image_features.shape[0]

        if batch_size <= 256:
            # Exact nuclear norm for small batches
            singular_values = torch.linalg.svdvals(image_features)
            scores = singular_values.sum(dim=-1)
        else:
            # Approximate with Frobenius norm for large batches
            scores = torch.norm(image_features, p='fro', dim=-1)

        # Normalize to [0, 1]
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        return scores

    @torch.no_grad()
    def compute_clip_similarity_score(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute CLIP image-text similarity score.

        Higher similarity = better aligned image-text pair
        """
        # Normalize features
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # Cosine similarity
        similarity = (image_features * text_features).sum(dim=-1)

        # Convert to [0, 1] range
        scores = (similarity + 1) / 2
        return scores

    @torch.no_grad()
    def score_batch(self, images: torch.Tensor, texts: List[str]) -> torch.Tensor:
        """Score a batch of image-text pairs"""
        images = images.to(self.device)

        # Encode with proxy model
        with autocast(enabled=self.config.mixed_precision):
            image_features = self.proxy_model.encode_image(images)
            text_tokens = self.tokenizer(texts).to(self.device)
            text_features = self.proxy_model.encode_text(text_tokens)

        # Compute component scores
        nuclear_scores = self.compute_nuclear_norm_score(image_features)
        similarity_scores = self.compute_clip_similarity_score(image_features, text_features)

        # Combine scores
        combined_scores = (
            self.config.nuclear_norm_weight * nuclear_scores +
            self.config.clip_similarity_weight * similarity_scores
        )

        return combined_scores.cpu(), {
            'nuclear_norm': nuclear_scores.cpu(),
            'clip_similarity': similarity_scores.cpu()
        }

    def score_dataset(
        self,
        dataset,
        save_path: Optional[str] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Score entire DataComp Small dataset (12.8M samples).

        Returns:
            scores: Array of shape [N] with importance scores
            metadata: Dict with component scores and statistics
        """
        print(f"\nScoring {self.config.total_samples:,} samples...")
        print(f"This will take approximately 500 GPU hours on 4x A100")
        print(f"ETA: ~1 week")
        print()

        all_scores = []
        all_nuclear = []
        all_similarity = []

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.selection_batch_size,
            num_workers=self.config.num_selection_workers,
            pin_memory=True,
            drop_last=False
        )

        # Score all samples
        start_time = time.time()
        num_batches = self.config.total_samples // self.config.selection_batch_size

        for batch_idx, batch in enumerate(tqdm(
            dataloader,
            total=num_batches,
            desc="Scoring DataComp Small",
            unit="batch"
        )):
            images = torch.stack([b['image'] for b in batch])
            texts = [b['text'] for b in batch]

            # Score batch
            scores, components = self.score_batch(images, texts)

            all_scores.append(scores)
            all_nuclear.append(components['nuclear_norm'])
            all_similarity.append(components['clip_similarity'])

            # Log progress
            if self.writer and batch_idx % 100 == 0:
                self.writer.add_scalar('scoring/mean_score', scores.mean().item(), batch_idx)
                self.writer.add_scalar('scoring/std_score', scores.std().item(), batch_idx)

            # Save checkpoint periodically
            if save_path and batch_idx % 1000 == 0 and batch_idx > 0:
                checkpoint = {
                    'scores': torch.cat(all_scores).numpy(),
                    'batch_idx': batch_idx,
                    'config': self.config
                }
                torch.save(checkpoint, f"{save_path}.checkpoint_{batch_idx}")
                print(f"Checkpoint saved at batch {batch_idx}")

            self.global_step += 1

        # Concatenate all scores
        all_scores = torch.cat(all_scores).numpy()
        all_nuclear = torch.cat(all_nuclear).numpy()
        all_similarity = torch.cat(all_similarity).numpy()

        elapsed = time.time() - start_time
        print(f"\nScoring complete!")
        print(f"  Time: {elapsed/3600:.1f} hours")
        print(f"  Mean score: {all_scores.mean():.4f}")
        print(f"  Std score: {all_scores.std():.4f}")

        # Save final scores
        if save_path:
            results = {
                'scores': all_scores,
                'nuclear_norm': all_nuclear,
                'clip_similarity': all_similarity,
                'config': self.config,
                'elapsed_hours': elapsed / 3600
            }
            np.savez(save_path, **results)
            print(f"Scores saved to: {save_path}")

        metadata = {
            'nuclear_norm': all_nuclear,
            'clip_similarity': all_similarity,
            'elapsed_hours': elapsed / 3600
        }

        return all_scores, metadata


# ============================================================================
# Sample Selection
# ============================================================================

def select_top_k(
    scores: np.ndarray,
    k: int,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Select top-k samples by score.

    Args:
        scores: Array of shape [N] with importance scores
        k: Number of samples to select
        save_path: Optional path to save selected indices

    Returns:
        selected_indices: Array of shape [k] with indices of selected samples
    """
    print(f"\nSelecting top {k:,} / {len(scores):,} samples ({k/len(scores)*100:.1f}%)")

    # Get top-k indices
    selected_indices = np.argpartition(scores, -k)[-k:]
    selected_indices = selected_indices[np.argsort(scores[selected_indices])][::-1]

    print(f"Selection complete!")
    print(f"  Selected score range: [{scores[selected_indices].min():.4f}, {scores[selected_indices].max():.4f}]")
    print(f"  Rejected score range: [{scores[~np.isin(np.arange(len(scores)), selected_indices)].min():.4f}, "
          f"{scores[~np.isin(np.arange(len(scores)), selected_indices)].max():.4f}]")

    # Save selected indices
    if save_path:
        np.save(save_path, selected_indices)
        print(f"Selected indices saved to: {save_path}")

    return selected_indices


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="MODE for DataComp Small")
    parser.add_argument('--config', type=str, default=None, help='Path to config JSON')
    parser.add_argument('--stage', type=str, default='all',
                       choices=['score', 'select', 'all'],
                       help='Pipeline stage to run')
    parser.add_argument('--scores_path', type=str, default=None,
                       help='Path to load/save scores')
    parser.add_argument('--output_dir', type=str, default='./datacomp_small_output',
                       help='Output directory')
    args = parser.parse_args()

    # Load config
    if args.config:
        config = DataCompSmallConfig.from_json(args.config)
    else:
        config = DataCompSmallConfig()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config.save(str(output_dir / 'config.json'))

    print("="*80)
    print("MODE for DataComp Small")
    print("="*80)
    print(f"Total samples: {config.total_samples:,}")
    print(f"Target selection: {config.target_selected:,} ({config.selection_budget*100:.0f}%)")
    print(f"Output directory: {output_dir}")
    print("="*80)
    print()

    # Stage 1: Score all samples
    if args.stage in ['score', 'all']:
        print("\n" + "="*80)
        print("STAGE 1: SCORING")
        print("="*80)

        # Load dataset
        dataset = DataCompSmallDataset(config)

        # Initialize scorer
        scorer = DataCompMODEScorer(config)

        # Score dataset
        scores_path = args.scores_path or str(output_dir / 'scores.npz')
        scores, metadata = scorer.score_dataset(dataset, save_path=scores_path)
    else:
        # Load pre-computed scores
        if not args.scores_path:
            raise ValueError("Must provide --scores_path when skipping scoring stage")

        print(f"Loading scores from: {args.scores_path}")
        scores_data = np.load(args.scores_path)
        scores = scores_data['scores']
        print(f"Loaded {len(scores):,} scores")

    # Stage 2: Select top-k samples
    if args.stage in ['select', 'all']:
        print("\n" + "="*80)
        print("STAGE 2: SELECTION")
        print("="*80)

        selected_indices = select_top_k(
            scores,
            k=config.target_selected,
            save_path=str(output_dir / 'selected_indices.npy')
        )

        print(f"\nSelected {len(selected_indices):,} samples")
        print(f"Use these indices to filter your DataComp training set!")

    print("\n" + "="*80)
    print("Pipeline complete!")
    print("="*80)
    print(f"\nNext steps:")
    print(f"1. Use selected indices for CLIP training")
    print(f"2. Train CLIP model on selected {config.target_selected:,} samples")
    print(f"3. Evaluate on DataComp benchmark tasks")
    print()


if __name__ == '__main__':
    main()
