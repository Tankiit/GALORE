"""
Pretraining Feature Extractor with Config Support

Extracts features from pretrained models and saves them to disk for later use.
Supports both vision and language models with configurable extraction strategies.

Usage:
    python pretrain_feature_extractor.py --config feature_config.yaml --dataset cifar10
    python pretrain_feature_extractor.py --config feature_config.yaml --mode vision --models vit_base_patch16_224 resnet50
"""

import gc
import os
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision.models.feature_extraction import create_feature_extractor

import torchvision
import torchvision.transforms as transforms
import numpy as np
from sklearn.model_selection import train_test_split


# =============================================================================
# Configuration Management
# =============================================================================

class FeatureExtractionConfig:
    """Configuration manager for feature extraction"""

    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def get_vision_models(self, enabled_only: bool = True) -> List[Dict]:
        """Get list of vision models from config"""
        models = self.config.get('vision_models', [])
        if enabled_only:
            models = [m for m in models if m.get('enabled', False)]
        return models

    def get_language_models(self, enabled_only: bool = True) -> List[Dict]:
        """Get list of language models from config"""
        models = self.config.get('language_models', [])
        if enabled_only:
            models = [m for m in models if m.get('enabled', False)]
        return models

    def get_dataset_config(self, dataset_name: str) -> Dict:
        """Get dataset configuration"""
        return self.config.get('datasets', {}).get(dataset_name, {})

    def get_feature_config(self) -> Dict:
        """Get feature extraction configuration"""
        return self.config.get('feature_extraction', {})

    def get_storage_config(self) -> Dict:
        """Get storage configuration"""
        return self.config.get('storage', {})

    def get_active_models(self, mode: str = 'pretraining') -> Dict:
        """Get active models for given mode"""
        return self.config.get('active_config', {}).get(f'{mode}_models', {})


# =============================================================================
# Vision Feature Extraction
# =============================================================================

class VisionFeatureExtractor:
    """Extract features from pretrained vision models"""

    def __init__(
        self,
        model_name: str,
        model_config: Dict,
        device: str = 'cuda',
        extract_layers: bool = True,
        layer_indices: Optional[List[int]] = None
    ):
        self.model_name = model_name
        self.model_config = model_config
        self.device = torch.device(device)
        self.extract_layers = extract_layers
        self.layer_indices = layer_indices or [3, 6, 9, 12]

        # Load model
        print(f"Loading model: {model_name}")
        self.model = timm.create_model(
            model_name,
            pretrained=model_config.get('pretrained', True)
        ).to(self.device).eval()

        # Get transform
        self.transform = create_transform(
            **resolve_data_config(self.model.pretrained_cfg, model=self.model)
        )

        # Setup feature extraction
        self._setup_feature_extraction()

    def _setup_feature_extraction(self):
        """Setup feature extraction nodes"""
        model_type = self.model_config.get('type', '').lower()

        if 'vit' in model_type or 'vit' in self.model_name:
            # ViT: extract from transformer blocks
            if self.extract_layers:
                self.return_nodes = [
                    f"blocks.{i}.add_1" for i in self.layer_indices
                    if i < len(self.model.blocks)
                ]
            else:
                # Just extract final layer
                self.return_nodes = [f"blocks.{len(self.model.blocks)-1}.add_1"]

            self.model = create_feature_extractor(self.model, return_nodes=self.return_nodes)
            self.pool_method = self.model_config.get('pool', 'cls')

        elif 'resnet' in model_type or 'resnet' in self.model_name:
            # ResNet: extract from layer outputs
            if self.extract_layers:
                self.return_nodes = [f"layer{i}" for i in [1, 2, 3, 4]]
            else:
                self.return_nodes = ["layer4"]

            self.model = create_feature_extractor(self.model, return_nodes=self.return_nodes)
            self.pool_method = self.model_config.get('pool', 'avg')

        else:
            # Generic model: try to extract from common layers
            self.return_nodes = None
            self.pool_method = self.model_config.get('pool', 'avg')

        print(f"  Model type: {model_type}")
        print(f"  Pool method: {self.pool_method}")
        print(f"  Return nodes: {self.return_nodes}")

    @torch.no_grad()
    def extract_features(
        self,
        dataloader: DataLoader,
        compute_loss: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features from entire dataloader

        Returns:
            Dictionary containing:
            - 'feats': [N, L, D] features (N=samples, L=layers, D=dim)
            - 'loss': [N] per-sample loss (if compute_loss=True)
            - 'num_params': int, number of model parameters
        """
        all_features = []
        all_losses = []

        print(f"Extracting features from {len(dataloader.dataset)} samples...")

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting")):
            if isinstance(batch, (tuple, list)):
                images, labels = batch
            else:
                images = batch['image']
                labels = batch['label']

            images = images.to(self.device)
            labels = labels.to(self.device)

            # Extract features
            if self.return_nodes is not None:
                # Multi-layer extraction
                outputs = self.model(images)

                if self.pool_method == 'cls':
                    # ViT: extract CLS token
                    feats = [v[:, 0, :] for v in outputs.values()]
                    feats = torch.stack(feats).permute(1, 0, 2)  # [B, L, D]
                elif self.pool_method == 'avg':
                    # Global average pooling
                    feats = []
                    for v in outputs.values():
                        if len(v.shape) == 4:  # [B, C, H, W]
                            feat = F.adaptive_avg_pool2d(v, 1).squeeze(-1).squeeze(-1)
                        elif len(v.shape) == 3:  # [B, N, D]
                            feat = v.mean(dim=1)
                        else:
                            feat = v
                        feats.append(feat)
                    feats = torch.stack(feats).permute(1, 0, 2)  # [B, L, D]
                else:
                    raise ValueError(f"Unknown pooling: {self.pool_method}")
            else:
                # Single output
                outputs = self.model(images)
                feats = outputs.unsqueeze(1)  # [B, 1, D]

            all_features.append(feats.cpu())

            # Compute loss if needed
            if compute_loss:
                # Need to get logits for loss computation
                # For feature extraction models, we need the classifier head
                if hasattr(self.model, 'head'):
                    logits = self.model.head(feats[:, -1, :])  # Use last layer
                elif hasattr(self.model, 'fc'):
                    logits = self.model.fc(feats[:, -1, :])
                else:
                    # Create a simple linear head for loss computation
                    logits = torch.randn(feats.size(0), labels.max().item() + 1, device=self.device)

                loss = F.cross_entropy(logits, labels, reduction='none')
                all_losses.append(loss.cpu())

        # Concatenate all batches
        features = torch.cat(all_features, dim=0)  # [N, L, D]

        result = {
            'feats': features,
            'num_params': sum(p.numel() for p in self.model.parameters()),
        }

        if compute_loss and len(all_losses) > 0:
            result['loss'] = torch.cat(all_losses, dim=0)  # [N]
            result['avg_loss'] = result['loss'].mean().item()

        print(f"  Features shape: {features.shape}")
        print(f"  Feature dimension: {features.size(-1)}")
        if 'avg_loss' in result:
            print(f"  Average loss: {result['avg_loss']:.4f}")

        return result

    def cleanup(self):
        """Clean up GPU memory"""
        del self.model
        torch.cuda.empty_cache()
        gc.collect()


# =============================================================================
# Dataset Loading
# =============================================================================

def load_vision_dataset(dataset_name: str, dataset_config: Dict, split: str = 'train') -> DataLoader:
    """Load vision dataset with config"""

    img_size = dataset_config.get('img_size', 224)
    mean = dataset_config.get('normalize_mean', [0.485, 0.456, 0.406])
    std = dataset_config.get('normalize_std', [0.229, 0.224, 0.225])

    # Create transform
    if split == 'train':
        transform = transforms.Compose([
            transforms.Resize(img_size) if img_size != 32 else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(img_size) if img_size != 32 else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    # Load dataset
    if dataset_name == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=(split == 'train'),
            download=True,
            transform=transform
        )
    elif dataset_name == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(
            root='./data',
            train=(split == 'train'),
            download=True,
            transform=transform
        )
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

    return dataset


# =============================================================================
# Feature Saving/Loading
# =============================================================================

def save_features(
    features: Dict[str, torch.Tensor],
    save_path: str,
    compress: bool = True
):
    """Save features to disk"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if compress:
        # Save with compression
        torch.save(features, save_path, _use_new_zipfile_serialization=True)
    else:
        torch.save(features, save_path)

    file_size = os.path.getsize(save_path) / (1024 * 1024)  # MB
    print(f"  Saved to: {save_path}")
    print(f"  File size: {file_size:.2f} MB")


def load_features(load_path: str) -> Dict[str, torch.Tensor]:
    """Load features from disk"""
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Feature file not found: {load_path}")

    features = torch.load(load_path)
    print(f"  Loaded from: {load_path}")
    print(f"  Features shape: {features['feats'].shape}")

    return features


def get_feature_save_path(
    output_dir: str,
    dataset_name: str,
    model_name: str,
    split: str = 'train'
) -> str:
    """Generate feature save path"""
    # Sanitize model name for filename
    safe_model_name = model_name.replace('/', '_').replace(':', '_')

    filename = f"{dataset_name}_{split}_{safe_model_name}.pt"
    return os.path.join(output_dir, dataset_name, filename)


# =============================================================================
# Main Extraction Pipeline
# =============================================================================

def extract_all_features(
    config_path: str,
    dataset_name: str,
    models: Optional[List[str]] = None,
    mode: str = 'vision',
    batch_size: int = 64,
    num_workers: int = 4,
    device: str = 'cuda',
    force_remake: bool = False
):
    """
    Extract features from all configured models

    Args:
        config_path: Path to YAML config file
        dataset_name: Dataset to extract features from
        models: List of model names to use (None = use config defaults)
        mode: 'vision' or 'language'
        batch_size: Batch size for extraction
        num_workers: Number of data loading workers
        device: Device to use
        force_remake: Force remake even if features exist
    """
    # Load config
    config = FeatureExtractionConfig(config_path)
    dataset_config = config.get_dataset_config(dataset_name)
    feature_config = config.get_feature_config()
    storage_config = config.get_storage_config()

    # Get models to process
    if models is None:
        if mode == 'vision':
            model_list = config.get_vision_models(enabled_only=True)
        elif mode == 'language':
            model_list = config.get_language_models(enabled_only=True)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    else:
        # Use specified models
        if mode == 'vision':
            all_models = config.get_vision_models(enabled_only=False)
        else:
            all_models = config.get_language_models(enabled_only=False)

        model_list = [m for m in all_models if m['name'] in models]

    print(f"\n{'='*80}")
    print(f"Feature Extraction Pipeline")
    print(f"{'='*80}")
    print(f"Dataset: {dataset_name}")
    print(f"Mode: {mode}")
    print(f"Models: {[m['name'] for m in model_list]}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")

    # Load dataset
    print("Loading dataset...")
    dataset = load_vision_dataset(dataset_name, dataset_config, split='train')
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == 'cuda')
    )
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Batch size: {batch_size}")
    print()

    # Extract features from each model
    output_dir = storage_config.get('feature_dir', './features/precomputed')
    compress = storage_config.get('compress', True)
    extract_layers = feature_config.get('pretraining', {}).get('extract_layers', True)
    layer_indices = feature_config.get('pretraining', {}).get('layer_indices', [3, 6, 9, 12])

    for model_cfg in model_list:
        model_name = model_cfg['name']
        print(f"\n{'-'*80}")
        print(f"Processing: {model_name}")
        print(f"{'-'*80}")

        # Check if features already exist
        save_path = get_feature_save_path(output_dir, dataset_name, model_name, split='train')

        if os.path.exists(save_path) and not force_remake:
            print(f"  Features already exist: {save_path}")
            print("  Skipping (use --force_remake to regenerate)")
            continue

        try:
            # Create extractor
            extractor = VisionFeatureExtractor(
                model_name=model_name,
                model_config=model_cfg,
                device=device,
                extract_layers=extract_layers,
                layer_indices=layer_indices
            )

            # Extract features
            features = extractor.extract_features(
                dataloader,
                compute_loss=True
            )

            # Save features
            save_features(features, save_path, compress=compress)

            # Cleanup
            extractor.cleanup()

        except Exception as e:
            print(f"  Error processing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*80}")
    print("Feature extraction complete!")
    print(f"{'='*80}")


# =============================================================================
# Command Line Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pretraining Feature Extractor with Config Support"
    )

    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        default='feature_config.yaml',
        help='Path to configuration YAML file'
    )

    # Dataset
    parser.add_argument(
        '--dataset',
        type=str,
        default='cifar10',
        help='Dataset name (cifar10, cifar100, imagenet)'
    )

    # Models
    parser.add_argument(
        '--mode',
        type=str,
        default='vision',
        choices=['vision', 'language'],
        help='Feature extraction mode'
    )

    parser.add_argument(
        '--models',
        nargs='+',
        default=None,
        help='Specific models to extract features from (space-separated)'
    )

    # Extraction settings
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for extraction'
    )

    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'mps', 'cpu'],
        help='Device to use for extraction'
    )

    # Flags
    parser.add_argument(
        '--force_remake',
        action='store_true',
        help='Force remake features even if they exist'
    )

    args = parser.parse_args()

    # Run extraction
    extract_all_features(
        config_path=args.config,
        dataset_name=args.dataset,
        models=args.models,
        mode=args.mode,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        force_remake=args.force_remake
    )


if __name__ == "__main__":
    main()
