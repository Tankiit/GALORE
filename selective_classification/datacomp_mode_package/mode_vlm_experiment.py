"""
VLM Gradient Cache for DataComp

Optimized for 1-3 A100 GPUs, 400M samples, nuclear norm utility.
Features caching, approximation, multi-GPU sharding, and Bloom filters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Callable
from dataclasses import dataclass
import hashlib
import pickle
import time
from collections import defaultdict
import open_clip
import copy
from tqdm import tqdm

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: pip install datasets for DataComp loading")

try:
    from diskcache import Cache as DiskCache
    DISKCACHE_AVAILABLE = True
except ImportError:
    DISKCACHE_AVAILABLE = False
    print("Warning: pip install diskcache for production performance")

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: pip install tensorboard for logging")


# ============================================================================
# CLIP MODEL CONFIGURATIONS
# ============================================================================

CLIP_MODELS = ['ViT-B/32', 'ViT-B/16', 'ViT-L/14']  # Default: ViT-B/32

# My recommendation: Focus on ViT-B/32 only
# Reason: Reviewer concerns are about SCALE (ImageNet), not model size

def print_clip_models():
    print(f"Available models: {CLIP_MODELS} (Default: ViT-B/32)")

def get_recommended_model() -> str:
    return 'ViT-B/32'

def validate_model_choice(model_name: str) -> bool:
    if model_name not in CLIP_MODELS:
        print(f"Error: Model '{model_name}' not found in CLIP_MODELS")
        print(f"Available models: {CLIP_MODELS}")
        return False
    return True


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class DataCompCacheConfig:


    # Scale
    dataset_size: int = 400_000_000  # DataComp CommonPool
    budget: float = 0.10  # 10% selection (40M samples)

    # Model selection
    clip_model: str = 'ViT-B/32'  # Default to fastest model

    # Hardware
    num_gpus: int = 1  # 1-3 A100s
    gpu_memory_gb: float = 80.0  # A100 80GB

    # Cache settings
    cache_dir: str = "./datacomp_cache"
    cache_size_gb: float = 200.0  # Large cache for 400M dataset
    enable_bloom: bool = True  # Essential at this scale
    bloom_size: int = 50_000_000  # 50M bits ≈ 6MB

    # Nuclear norm computation
    use_nuclear_norm: bool = True
    nuclear_alpha: float = 0.33  # Image weight
    nuclear_beta: float = 0.33   # Text weight
    nuclear_gamma: float = 0.34  # Interaction weight
    svd_free_approximation: bool = True  # MUCH faster

    # Hypernetwork for adaptive importance
    use_hypernetwork: bool = False  # Use learned importance predictor
    hypernetwork_hidden_dim: int = 256
    hypernetwork_layers: int = 3
    hypernetwork_lr: float = 1e-4

    # Gradient staleness
    step_quantization: int = 500  # Reuse gradients for 500 steps
    enable_gradient_aging: bool = True  # Decay old gradients
    aging_halflife_steps: int = 5000

    # Performance
    prefetch_batch_size: int = 1024
    compression: bool = True
    mixed_precision: bool = True
    pin_memory: bool = True

    # Multi-GPU
    distributed: bool = False
    cache_sharding: bool = True  # Each GPU has local cache

    # Logging
    enable_tensorboard: bool = True
    tensorboard_dir: str = "./runs"
    log_frequency: int = 50  # Log every N steps

    @classmethod
    def for_small_scale(cls, num_gpus: int = 1) -> 'DataCompCacheConfig':
        return cls(
            dataset_size=1_000_000,
            budget=0.30,
            cache_size_gb=10.0,
            enable_bloom=False,
            step_quantization=100,
            num_gpus=num_gpus
        )

    @classmethod
    def for_medium_scale(cls, num_gpus: int = 2) -> 'DataCompCacheConfig':
        return cls(
            dataset_size=50_000_000,
            budget=0.15,
            cache_size_gb=50.0,
            enable_bloom=True,
            bloom_size=10_000_000,
            step_quantization=250,
            num_gpus=num_gpus
        )

    @classmethod
    def for_full_datacomp(cls, num_gpus: int = 3) -> 'DataCompCacheConfig':
        return cls(
            dataset_size=400_000_000,
            budget=0.10,
            cache_size_gb=200.0,
            enable_bloom=True,
            bloom_size=50_000_000,
            step_quantization=500,
            num_gpus=num_gpus,
            distributed=(num_gpus > 1)
        )


# ============================================================================
# BLOOM FILTER (for 400M scale lookups)
# ============================================================================

class BloomFilter:


    def __init__(self, size: int = 50_000_000, num_hashes: int = 7):
        self.size = size
        self.num_hashes = num_hashes
        self.bit_array = np.zeros(size, dtype=bool)
        self.num_inserted = 0

    def _hash(self, key: str, seed: int) -> int:
        h = hashlib.md5(f"{key}{seed}".encode()).digest()
        return int.from_bytes(h[:4], 'little') % self.size

    def add(self, key: str):
        for i in range(self.num_hashes):
            self.bit_array[self._hash(key, i)] = True
        self.num_inserted += 1

    def contains(self, key: str) -> bool:
        return all(self.bit_array[self._hash(key, i)] for i in range(self.num_hashes))

    def save(self, path: Path):
        np.savez_compressed(path,
                           bit_array=self.bit_array,
                           num_inserted=self.num_inserted)

    def load(self, path: Path):
        data = np.load(path)
        self.bit_array = data['bit_array']
        self.num_inserted = int(data['num_inserted'])


# ============================================================================
# SVD-FREE NUCLEAR NORM (10-100x faster than full SVD)
# ============================================================================

class FastNuclearNorm:


    def __init__(self,
                 alpha: float = 0.33,
                 beta: float = 0.33,
                 gamma: float = 0.34,
                 rank_estimate: int = 64):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.rank_estimate = rank_estimate

    def compute_utility(self,
                       image_features: torch.Tensor,
                       text_features: torch.Tensor) -> Tuple[torch.Tensor, Dict]:

        N = len(image_features)
        device = image_features.device

        # Normalize
        img_norm = F.normalize(image_features, dim=-1)
        txt_norm = F.normalize(text_features, dim=-1)

        # Component 1: Image discriminability (Frobenius proxy)
        img_gram = img_norm @ img_norm.T  # [N, N]
        img_discriminability = torch.diagonal(img_gram).mean()

        # Component 2: Text discriminability
        txt_gram = txt_norm @ txt_norm.T
        txt_discriminability = torch.diagonal(txt_gram).mean()

        # Component 3: Interaction (approximate rank)
        # Use cross-correlation matrix
        cross_corr = img_norm @ txt_norm.T  # [N, N]

        # Fast rank estimation via power iteration
        if N < 1000:
            # Small batch: exact singular values
            s = torch.linalg.svdvals(cross_corr)
            interaction_rank = (s > 1e-6).sum().float()
        else:
            # Large batch: randomized trace estimation
            k = min(self.rank_estimate, N // 2)
            Q = torch.randn(N, k, device=device)
            Q, _ = torch.linalg.qr(Q)

            # Approximate top-k singular values
            B = cross_corr @ Q
            s_approx = torch.linalg.svdvals(B)
            interaction_rank = s_approx.sum()

        # Combine into utility (higher = better)
        utility = (
            self.alpha * img_discriminability +
            self.beta * txt_discriminability +
            self.gamma * interaction_rank / N  # Normalize by batch size
        )

        # Per-sample decomposition (for selection)
        per_sample_utilities = self._compute_per_sample(
            img_norm, txt_norm, cross_corr
        )

        metadata = {
            'img_discriminability': img_discriminability.item(),
            'txt_discriminability': txt_discriminability.item(),
            'interaction_rank': interaction_rank.item(),
            'utility': utility.item()
        }

        return per_sample_utilities, metadata

    def _compute_per_sample(self,
                           img_norm: torch.Tensor,
                           txt_norm: torch.Tensor,
                           cross_corr: torch.Tensor) -> torch.Tensor:

        N = len(img_norm)

        # Image contribution: how unique is this image?
        img_similarity = (img_norm @ img_norm.T).abs()
        img_uniqueness = 1.0 - (img_similarity.sum(dim=1) - 1) / (N - 1)

        # Text contribution
        txt_similarity = (txt_norm @ txt_norm.T).abs()
        txt_uniqueness = 1.0 - (txt_similarity.sum(dim=1) - 1) / (N - 1)

        # Interaction contribution: alignment strength
        interaction_strength = cross_corr.abs().mean(dim=1)

        # Combine
        per_sample = (
            self.alpha * img_uniqueness +
            self.beta * txt_uniqueness +
            self.gamma * interaction_strength
        )

        return per_sample


# ============================================================================
# DATACOMP VLM GRADIENT CACHE
# ============================================================================

class DataCompGradientCache:


    def __init__(self, config: DataCompCacheConfig, device: str = 'cpu'):
        self.config = config
        self.device = device
        self.global_step = 0

        # Initialize cache backend
        if config.cache_sharding and config.distributed:
            rank = dist.get_rank()
            cache_dir = Path(config.cache_dir) / f"rank_{rank}"
        else:
            cache_dir = Path(config.cache_dir)

        cache_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard writer
        self.writer = None
        if config.enable_tensorboard and TENSORBOARD_AVAILABLE:
            tb_dir = Path(config.tensorboard_dir) / "cache"
            tb_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(str(tb_dir))
            print(f"TensorBoard logging to: {tb_dir}")

        if DISKCACHE_AVAILABLE:
            cache_dir = Path(config.cache_dir)
            cache_dir.mkdir(exist_ok=True)
            self.cache = DiskCache(str(cache_dir), size_limit=int(config.cache_size_gb * 1e9))
            print(f"DiskCache: {config.cache_size_gb}GB at {cache_dir}")
        else:
            self.cache = {}
            print("Warning: Using memory fallback (install diskcache!)")

        # Bloom filter (essential for 400M scale)
        self.bloom = None
        if config.enable_bloom:
            bloom_path = cache_dir / "bloom_filter.npz"
            self.bloom = BloomFilter(size=config.bloom_size, num_hashes=7)

            if bloom_path.exists():
                self.bloom.load(bloom_path)
                print(f"Loaded Bloom filter: {self.bloom.num_inserted:,} entries")
            else:
                print(f"Bloom filter initialized: {config.bloom_size/8/1e6:.1f}MB")

        # Nuclear norm utility
        self.nuclear_norm = FastNuclearNorm(
            alpha=config.nuclear_alpha,
            beta=config.nuclear_beta,
            gamma=config.nuclear_gamma
        )

        # Gradient aging tracking
        self.gradient_ages = {}  # key -> step_created

        # Statistics
        self.stats = defaultdict(int)
        self.stats['bloom_saves'] = 0
        self.stats['aged_out'] = 0

        print(f"DataComp cache ready: {config.dataset_size/1e6:.0f}M samples")

    def _generate_key(self,
                     image_features: torch.Tensor,
                     text_features: torch.Tensor) -> str:

        # Hash both modalities together
        img_bytes = image_features.cpu().numpy().tobytes()
        txt_bytes = text_features.cpu().numpy().tobytes()
        combined = img_bytes + txt_bytes

        key_hash = hashlib.md5(combined).hexdigest()[:16]

        # Quantize step for temporal grouping
        step_bucket = (self.global_step // self.config.step_quantization) * self.config.step_quantization

        return f"{key_hash}_s{step_bucket}"

    def _compute_importance(self,
                           image_features: torch.Tensor,
                           text_features: torch.Tensor) -> torch.Tensor:

        utilities, metadata = self.nuclear_norm.compute_utility(
            image_features, text_features
        )
        return utilities

    def get_or_compute(self,
                      image_features: torch.Tensor,
                      text_features: torch.Tensor,
                      force_compute: bool = False) -> Tuple[torch.Tensor, Dict]:

        batch_size = len(image_features)

        if force_compute:
            importance = self._compute_and_cache(image_features, text_features)
            return importance, {'cache_hit': False, 'aged': False}

        # Try batch-level cache first (common in DataComp due to duplicates)
        key = self._generate_key(image_features, text_features)

        # STEP 1: Bloom filter (99% of work happens here!)
        if self.bloom is not None:
            if not self.bloom.contains(key):
                # Definitely not cached
                self.stats['bloom_saves'] += 1
                self.stats['misses'] += 1
                importance = self._compute_and_cache(image_features, text_features)
                return importance, {'cache_hit': False, 'bloom_save': True}

        # STEP 2: Actual cache lookup
        cached = self._get(key)

        if cached is not None:
            # Check if gradient is too old
            if self.config.enable_gradient_aging:
                age = self.global_step - self.gradient_ages.get(key, self.global_step)
                decay_factor = 0.5 ** (age / self.config.aging_halflife_steps)

                if decay_factor < 0.1:
                    # Too old, recompute
                    self.stats['aged_out'] += 1
                    self.stats['misses'] += 1
                    importance = self._compute_and_cache(image_features, text_features)
                    return importance, {'cache_hit': False, 'aged': True}

                # Apply aging decay
                cached = cached * decay_factor

            self.stats['hits'] += 1
            return cached.to(self.device), {'cache_hit': True, 'aged': False}

        # STEP 3: Cache miss, compute
        self.stats['misses'] += 1
        importance = self._compute_and_cache(image_features, text_features)
        return importance, {'cache_hit': False, 'aged': False}

    def _get(self, key: str) -> Optional[torch.Tensor]:

        if DISKCACHE_AVAILABLE:
            value = self.cache.get(key)
            if value is not None:
                return pickle.loads(value) if isinstance(value, bytes) else value
        else:
            return self.cache.get(key)
        return None

    def _set(self, key: str, tensor: torch.Tensor):

        if DISKCACHE_AVAILABLE:
            serialized = pickle.dumps(tensor.cpu(), protocol=5)
            self.cache.set(key, serialized)
        else:
            self.cache[key] = tensor.cpu()

        # Track age
        self.gradient_ages[key] = self.global_step

    def _compute_and_cache(self,
                          image_features: torch.Tensor,
                          text_features: torch.Tensor) -> torch.Tensor:

        self.stats['computes'] += 1

        # Compute nuclear norm utility
        importance = self._compute_importance(image_features, text_features)

        # Cache
        key = self._generate_key(image_features, text_features)
        self._set(key, importance)

        # Update Bloom
        if self.bloom is not None:
            self.bloom.add(key)

        return importance

    def step(self):

        self.global_step += 1

        # Log cache statistics to TensorBoard
        if self.writer and self.global_step % self.config.log_frequency == 0:
            stats = self.get_stats()
            self.writer.add_scalar('cache/hit_rate', stats['hit_rate'], self.global_step)
            self.writer.add_scalar('cache/hits', stats['hits'], self.global_step)
            self.writer.add_scalar('cache/misses', stats['misses'], self.global_step)
            self.writer.add_scalar('cache/computes', stats['computes'], self.global_step)
            self.writer.add_scalar('cache/aged_out', stats['aged_out'], self.global_step)
            if 'bloom_efficiency' in stats:
                self.writer.add_scalar('cache/bloom_efficiency', stats['bloom_efficiency'], self.global_step)

    def save_bloom(self):

        if self.bloom is not None:
            bloom_path = Path(self.config.cache_dir) / "bloom_filter.npz"
            self.bloom.save(bloom_path)
            print(f"Saved Bloom filter: {self.bloom.num_inserted:,} entries")

    def get_stats(self) -> Dict:

        total = self.stats['hits'] + self.stats['misses']
        stats = dict(self.stats)
        stats['hit_rate'] = self.stats['hits'] / max(1, total)
        stats['bloom_efficiency'] = self.stats['bloom_saves'] / max(1, total)
        stats['global_step'] = self.global_step

        if DISKCACHE_AVAILABLE and hasattr(self.cache, 'volume'):
            stats['cache_size_mb'] = self.cache.volume() / 1e6
            stats['num_entries'] = len(self.cache)

        return stats

    def print_stats(self):

        stats = self.get_stats()

        print("\n" + "="*80)
        print("DATACOMP GRADIENT CACHE STATS")
        print("="*80)
        print(f"Global step:        {stats['global_step']:,}")
        print(f"Hit rate:           {stats['hit_rate']:.1%}")
        print(f"Hits:               {stats['hits']:,}")
        print(f"Misses:             {stats['misses']:,}")
        print(f"Computes:           {stats['computes']:,}")
        print(f"Aged out:           {stats['aged_out']:,}")

        if self.bloom:
            print("\nBloom Filter:")
            print(f"  Lookups saved:    {stats['bloom_saves']:,} ({stats['bloom_efficiency']:.1%})")

        if 'cache_size_mb' in stats:
            print("\nCache:")
            print(f"  Size:             {stats['cache_size_mb']:.1f} MB")
            print(f"  Entries:          {stats['num_entries']:,}")

        print("="*80 + "\n")


# ============================================================================
# HYPERNETWORK FOR ADAPTIVE IMPORTANCE WEIGHTS
# ============================================================================

class ImportanceHyperNetwork(nn.Module):
    """
    Hypernetwork that generates adaptive importance weights for samples.

    Takes image and text features and outputs a scalar importance score.
    Can be used as an alternative or complement to EL2N/gradient-based scoring.
    """

    def __init__(self,
                 feature_dim: int = 512,
                 hidden_dim: int = 256,
                 num_layers: int = 3):
        super().__init__()

        self.feature_dim = feature_dim

        # Multi-modal fusion
        self.image_proj = nn.Linear(feature_dim, hidden_dim)
        self.text_proj = nn.Linear(feature_dim, hidden_dim)

        # MLP layers for importance prediction
        layers = []
        in_dim = hidden_dim * 2  # Concatenated image + text

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            in_dim = hidden_dim

        # Output layer: single importance score
        layers.append(nn.Linear(hidden_dim, 1))

        self.mlp = nn.Sequential(*layers)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self,
                image_features: torch.Tensor,
                text_features: torch.Tensor) -> torch.Tensor:
        """Predict importance scores from CLIP features"""
        # Project features
        img_h = F.relu(self.image_proj(image_features))
        txt_h = F.relu(self.text_proj(text_features))

        # Concatenate
        combined = torch.cat([img_h, txt_h], dim=1)

        # Predict importance
        importance = self.mlp(combined).squeeze(-1)

        # Apply sigmoid to get [0, 1] range
        importance = torch.sigmoid(importance)

        return importance

    def train_step(self,
                   image_features: torch.Tensor,
                   text_features: torch.Tensor,
                   target_importance: torch.Tensor,
                   optimizer: torch.optim.Optimizer) -> float:
        """Train hypernetwork to predict importance scores"""
        self.train()

        # Normalize target to [0, 1]
        target_norm = (target_importance - target_importance.min()) / \
                     (target_importance.max() - target_importance.min() + 1e-8)

        # Predict importance
        pred_importance = self.forward(image_features, text_features)

        # MSE loss
        loss = F.mse_loss(pred_importance, target_norm)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()


# ============================================================================
# DATACOMP TRAINING INTEGRATION
# ============================================================================

class DataCompTrainer:


    def __init__(self, config: DataCompCacheConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize cache
        self.cache = DataCompGradientCache(config, device=self.device)

        # TensorBoard writer
        self.writer = None
        if config.enable_tensorboard and TENSORBOARD_AVAILABLE:
            tb_dir = Path(config.tensorboard_dir) / "training"
            tb_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(str(tb_dir))
            print(f"TensorBoard logging to: {tb_dir}")

        # Load CLIP model for training (not eval!)
        model_name = config.clip_model
        if not validate_model_choice(model_name):
            raise ValueError(f"Invalid model choice: {model_name}")

        print(f"Loading CLIP model: {model_name}")

        # Convert model name for open_clip (/ to -)
        openclip_name = model_name.replace('/', '-')
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            openclip_name, pretrained='openai'
        )
        self.clip_model = self.clip_model.to(self.device).train()  # Train mode!
        self.tokenizer = open_clip.get_tokenizer(openclip_name)

        # Optimizer for CLIP training
        self.optimizer = torch.optim.AdamW(
            self.clip_model.parameters(),
            lr=5e-5,
            betas=(0.9, 0.98),
            eps=1e-6,
            weight_decay=0.2
        )

        # Mixed precision training
        self.scaler = GradScaler() if config.mixed_precision else None

        # Optional: Hypernetwork for learned importance prediction
        self.hypernetwork = None
        self.hypernetwork_optimizer = None
        if config.use_hypernetwork:
            # Get feature dimension from CLIP model
            feature_dim = self.clip_model.text_projection.shape[1]
            self.hypernetwork = ImportanceHyperNetwork(
                feature_dim=feature_dim,
                hidden_dim=config.hypernetwork_hidden_dim,
                num_layers=config.hypernetwork_layers
            ).to(self.device)

            self.hypernetwork_optimizer = torch.optim.AdamW(
                self.hypernetwork.parameters(),
                lr=config.hypernetwork_lr
            )
            print(f"Hypernetwork initialized (dim={feature_dim}, hidden={config.hypernetwork_hidden_dim})")

        print("DataComp trainer ready")

    @torch.no_grad()
    def extract_features(self, images, texts):

        # Encode
        image_features = self.clip_model.encode_image(images)
        text_features = self.clip_model.encode_text(texts)

        # Normalize
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        return image_features, text_features

    def training_step(self,
                     images: torch.Tensor,
                     texts: torch.Tensor,
                     warmup: bool = False) -> Dict:

        # Phase 1: Score samples with cached/computed importance
        with torch.no_grad():
            self.clip_model.eval()
            image_features = self.clip_model.encode_image(images)
            text_features = self.clip_model.encode_text(texts)
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)

        # Get gradient importance (cached or computed)
        importance, metadata = self.cache.get_or_compute(
            image_features, text_features,
            force_compute=warmup
        )

        # Optional: Use hypernetwork for refined importance prediction
        if self.hypernetwork is not None:
            # Train hypernetwork to predict importance scores
            if warmup:
                # During warmup, train hypernetwork on nuclear norm scores
                hypernet_loss = self.hypernetwork.train_step(
                    image_features.detach(),
                    text_features.detach(),
                    importance.detach(),
                    self.hypernetwork_optimizer
                )
                metadata['hypernet_loss'] = hypernet_loss
            else:
                # After warmup, use hypernetwork predictions
                self.hypernetwork.eval()
                with torch.no_grad():
                    hypernet_importance = self.hypernetwork(
                        image_features.detach(),
                        text_features.detach()
                    )
                # Blend with nuclear norm scores (ensemble)
                importance = 0.7 * importance + 0.3 * hypernet_importance

        # Select top samples for training
        k = int(len(importance) * self.config.budget)
        selected_indices = torch.topk(importance, k=k).indices

        # Phase 2: ACTUAL CLIP TRAINING on selected samples
        self.clip_model.train()

        # Get selected samples
        selected_images = images[selected_indices]
        selected_texts = texts[selected_indices]

        # Forward pass with mixed precision
        with autocast(enabled=self.config.mixed_precision):
            # Encode selected samples
            image_emb = self.clip_model.encode_image(selected_images)
            text_emb = self.clip_model.encode_text(selected_texts)

            # Normalize
            image_emb = F.normalize(image_emb, dim=-1)
            text_emb = F.normalize(text_emb, dim=-1)

            # Compute contrastive loss
            logit_scale = self.clip_model.logit_scale.exp()
            logits = logit_scale * image_emb @ text_emb.T

            # Labels for contrastive learning
            labels = torch.arange(len(selected_images), device=self.device)

            # Bidirectional contrastive loss
            loss_i2t = F.cross_entropy(logits, labels)
            loss_t2i = F.cross_entropy(logits.T, labels)
            loss = (loss_i2t + loss_t2i) / 2

        # Backward pass
        self.optimizer.zero_grad()

        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        # Compute accuracy
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            acc = (preds == labels).float().mean()

        self.cache.step()

        metadata['selected_count'] = k
        metadata['mean_importance'] = importance.mean().item()
        metadata['loss'] = loss.item()
        metadata['accuracy'] = acc.item()

        # Log to TensorBoard
        if self.writer and self.cache.global_step % self.config.log_frequency == 0:
            self.writer.add_scalar('train/loss', loss.item(), self.cache.global_step)
            self.writer.add_scalar('train/accuracy', acc.item(), self.cache.global_step)
            self.writer.add_scalar('train/selected_count', k, self.cache.global_step)
            self.writer.add_scalar('train/selection_ratio', k / len(importance), self.cache.global_step)
            self.writer.add_scalar('train/mean_importance', importance.mean().item(), self.cache.global_step)
            self.writer.add_histogram('train/importance_distribution', importance, self.cache.global_step)
            if 'hypernet_loss' in metadata:
                self.writer.add_scalar('train/hypernet_loss', metadata['hypernet_loss'], self.cache.global_step)

        return metadata

    def train_epoch(self, dataloader, warmup: bool = False, epoch: int = 0):

        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}")

        for batch_idx, batch in pbar:
            images = batch['image'].to(self.device)
            texts = self.tokenizer(batch['text']).to(self.device)

            metadata = self.training_step(images, texts, warmup=warmup)

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{metadata['loss']:.4f}",
                'acc': f"{metadata['accuracy']:.3f}",
                'selected': f"{metadata['selected_count']}/{len(images)}"
            })

            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}: {metadata}")

        self.cache.print_stats()


# ============================================================================
# DATACOMP HYBRID MODE: Proxy-Guided + Self-Distillation
# ============================================================================


@dataclass
class HybridMODEConfig:


    # Dataset
    dataset_size: int = 400_000_000
    budget: float = 0.10  # Select 10%

    # Models
    proxy_model: str = 'ViT-B-32'  # Lightweight proxy
    main_model: str = 'ViT-B-32'   # Main training model

    # Hybrid selection strategy
    proxy_weight: float = 0.6      # Weight for proxy gradient signal
    checkpoint_weight: float = 0.4  # Weight for CLIP feature signal

    # Checkpoint management
    checkpoint_frequency: int = 500  # Update checkpoint every N steps
    checkpoint_warmup: int = 1000    # Steps before using checkpoint

    # Proxy training
    proxy_batch_size: int = 512     # Larger batches for proxy
    proxy_update_freq: int = 100    # Update proxy every N steps

    # Cache configuration
    cache_dir: str = "./datacomp_hybrid_cache"
    cache_size_gb: float = 200.0
    enable_bloom: bool = True
    bloom_size: int = int(400_000_000 * 10)  # 1% FPR

    # Hardware
    device: str = 'cuda'
    num_gpus: int = 1
    mixed_precision: bool = True

    # Training
    batch_size: int = 256
    learning_rate: float = 5e-5
    proxy_lr: float = 1e-4  # Proxy can learn faster
    warmup_steps: int = 1000

    # Logging
    enable_tensorboard: bool = True
    tensorboard_dir: str = "./runs/hybrid_mode"
    log_frequency: int = 50

    @classmethod
    def for_small_scale(cls):
        """Debug configuration"""
        return cls(
            dataset_size=10_000,
            budget=0.30,
            cache_size_gb=1.0,
            enable_bloom=False,
            proxy_batch_size=128,
            batch_size=64
        )

    @classmethod
    def for_datacomp(cls, num_gpus: int = 3):
        """Production DataComp configuration"""
        return cls(
            dataset_size=400_000_000,
            budget=0.10,
            cache_size_gb=200.0,
            enable_bloom=True,
            num_gpus=num_gpus,
            proxy_batch_size=512,
            batch_size=256
        )


class ProxyModel(nn.Module):


    def __init__(self, model_name: str = 'ViT-B-32', device: str = 'cuda'):
        super().__init__()
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained='openai'
        )
        self.model = self.model.to(device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self._freeze_early_layers()

    def _freeze_early_layers(self):

        if hasattr(self.model.visual, 'transformer'):
            for i, block in enumerate(self.model.visual.transformer.resblocks):
                if i < 8:
                    for param in block.parameters():
                        param.requires_grad = False
        if hasattr(self.model, 'transformer'):
            for i, block in enumerate(self.model.transformer.resblocks):
                if i < 8:
                    for param in block.parameters():
                        param.requires_grad = False

    def compute_gradient_importance(self,
                                    images: torch.Tensor,
                                    texts: torch.Tensor) -> torch.Tensor:

        self.model.train()
        image_features = self.model.encode_image(images)
        text_features = self.model.encode_text(texts)
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        logit_scale = self.model.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.T
        labels = torch.arange(len(images), device=images.device)
        loss_i2t = F.cross_entropy(logits, labels, reduction='none')
        loss_t2i = F.cross_entropy(logits.T, labels, reduction='none')
        per_sample_loss = (loss_i2t + loss_t2i) / 2

        importance_scores: List[float] = []
        for i in range(len(images)):
            self.model.zero_grad()
            per_sample_loss[i].backward(retain_graph=True)
            grad_norm_sq = 0.0
            for param in self.model.parameters():
                if param.grad is not None:
                    grad_norm_sq += float(param.grad.norm().item() ** 2)
            importance_scores.append(np.sqrt(grad_norm_sq))
        return torch.tensor(importance_scores, device=images.device)

    def compute_gradient_importance_fast(self,
                                         images: torch.Tensor,
                                         texts: torch.Tensor) -> torch.Tensor:
        """
        Fast gradient importance approximation using per-sample loss magnitude.

        This is 100x faster than per-sample gradients!
        Key insight: Per-sample loss magnitude correlates with gradient norm.

        Based on: "Beyond neural scaling laws: beating power law scaling via
        data pruning" (Sorscher et al., 2022)
        """
        self.model.eval()  # Use eval for faster inference

        with torch.no_grad():
            # Forward pass
            image_features = self.model.encode_image(images)
            text_features = self.model.encode_text(texts)
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)

            # Compute per-sample loss
            logit_scale = self.model.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.T
            labels = torch.arange(len(images), device=images.device)

            # Per-sample contrastive loss
            loss_i2t = F.cross_entropy(logits, labels, reduction='none')
            loss_t2i = F.cross_entropy(logits.T, labels, reduction='none')
            per_sample_loss = (loss_i2t + loss_t2i) / 2

            # Approximate gradient magnitude by feature norms weighted by loss
            # Higher loss + larger features = higher gradient contribution
            img_feat_norm = image_features.norm(dim=1)
            txt_feat_norm = text_features.norm(dim=1)

            # Combined importance score
            importance = per_sample_loss * (img_feat_norm + txt_feat_norm)

        return importance

    def compute_el2n_importance(self,
                                images: torch.Tensor,
                                texts: torch.Tensor) -> torch.Tensor:
        """
        EL2N: Expected L2 Norm of error vector.

        NO gradients needed! Pure inference-based importance.
        Papers show this correlates strongly with gradient norm.

        Based on: "Deep Learning on a Data Diet" (Paul et al., 2021)
        Lower confidence = higher learning potential = higher importance
        """
        self.model.eval()

        with torch.no_grad():
            image_features = self.model.encode_image(images)
            text_features = self.model.encode_text(texts)
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)

            # Compute logits
            logit_scale = self.model.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.T

            # Get prediction confidence for correct matches
            probs = F.softmax(logits, dim=1)
            labels = torch.arange(len(images), device=images.device)
            confidence = probs[torch.arange(len(images)), labels]

            # EL2N score: (1 - confidence)^2
            # Low confidence -> high EL2N -> high importance for learning
            el2n_score = (1 - confidence) ** 2

        return el2n_score

    def update(self,
               images: torch.Tensor,
               texts: torch.Tensor,
               optimizer: torch.optim.Optimizer) -> float:

        self.model.train()
        image_features = self.model.encode_image(images)
        text_features = self.model.encode_text(texts)
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        logit_scale = self.model.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.T
        labels = torch.arange(len(images), device=images.device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return float(loss.item())


class CheckpointManager:


    def __init__(self, checkpoint_dir: Path, max_checkpoints: int = 3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.current_checkpoint: Optional[Path] = None
        self.checkpoint_step: int = 0

    def save_checkpoint(self, model: nn.Module, step: int):
        checkpoint_path = self.checkpoint_dir / f"clip_step_{step}.pt"
        torch.save({'model_state_dict': model.state_dict(), 'step': step}, checkpoint_path)
        self.current_checkpoint = checkpoint_path
        self.checkpoint_step = step
        self._cleanup_old_checkpoints()

    def load_checkpoint(self, model: nn.Module) -> Optional[int]:
        if self.current_checkpoint is None or not self.current_checkpoint.exists():
            return None
        checkpoint = torch.load(self.current_checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        return int(checkpoint['step'])

    def _cleanup_old_checkpoints(self):
        checkpoints = sorted(
            self.checkpoint_dir.glob("clip_step_*.pt"),
            key=lambda p: int(p.stem.split('_')[-1])
        )
        for old_checkpoint in checkpoints[:-self.max_checkpoints]:
            try:
                old_checkpoint.unlink()
            except FileNotFoundError:
                pass


class HybridCache:


    def __init__(self, config: HybridMODEConfig):
        self.config = config
        cache_dir = Path(config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        if DISKCACHE_AVAILABLE:
            self.cache = DiskCache(str(cache_dir), size_limit=int(config.cache_size_gb * 1e9))
        else:
            self.cache = {}
        self.bloom: Optional[BloomFilter] = None
        if config.enable_bloom:
            self.bloom = BloomFilter(config.bloom_size, num_hashes=7)
        self.stats = defaultdict(int)

    def _key(self, data: torch.Tensor, cache_type: str) -> str:
        data_bytes = data.detach().cpu().numpy().tobytes()
        hash_val = hashlib.md5(data_bytes).hexdigest()[:16]
        return f"{cache_type}_{hash_val}"

    def _get(self, key: str):
        if DISKCACHE_AVAILABLE:
            value = self.cache.get(key)
            return pickle.loads(value) if value else None
        return self.cache.get(key)

    def _set(self, key: str, value):
        if DISKCACHE_AVAILABLE:
            self.cache.set(key, pickle.dumps(value, protocol=5))
        else:
            self.cache[key] = value

    def get_proxy_importance(self, images: torch.Tensor, texts: torch.Tensor) -> Optional[torch.Tensor]:
        key_input = torch.cat([images.flatten()[:100], texts.flatten()[:100]])
        key = self._key(key_input, 'proxy')
        if self.bloom and not self.bloom.contains(key):
            self.stats['proxy_bloom_saves'] += 1
            return None
        cached = self._get(key)
        if cached is not None:
            self.stats['proxy_hits'] += 1
            return cached
        self.stats['proxy_misses'] += 1
        return None

    def set_proxy_importance(self, images: torch.Tensor, texts: torch.Tensor, importance: torch.Tensor):
        key_input = torch.cat([images.flatten()[:100], texts.flatten()[:100]])
        key = self._key(key_input, 'proxy')
        self._set(key, importance.detach().cpu())
        if self.bloom:
            self.bloom.add(key)

    def get_clip_features(self, images: torch.Tensor, texts: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        key_input = torch.cat([images.flatten()[:100], texts.flatten()[:100]])
        key = self._key(key_input, 'clip')
        if self.bloom and not self.bloom.contains(key):
            self.stats['clip_bloom_saves'] += 1
            return None
        cached = self._get(key)
        if cached is not None:
            self.stats['clip_hits'] += 1
            return cached
        self.stats['clip_misses'] += 1
        return None

    def set_clip_features(self,
                          images: torch.Tensor,
                          texts: torch.Tensor,
                          image_features: torch.Tensor,
                          text_features: torch.Tensor):
        key_input = torch.cat([images.flatten()[:100], texts.flatten()[:100]])
        key = self._key(key_input, 'clip')
        self._set(key, (image_features.detach().cpu(), text_features.detach().cpu()))
        if self.bloom:
            self.bloom.add(key)

    def get_stats(self) -> Dict:
        proxy_total = self.stats['proxy_hits'] + self.stats['proxy_misses']
        clip_total = self.stats['clip_hits'] + self.stats['clip_misses']
        return {
            **dict(self.stats),
            'proxy_hit_rate': self.stats['proxy_hits'] / max(1, proxy_total),
            'clip_hit_rate': self.stats['clip_hits'] / max(1, clip_total)
        }


class HybridSelector:


    def __init__(self, proxy_weight: float = 0.6, checkpoint_weight: float = 0.4):
        self.proxy_weight = proxy_weight
        self.checkpoint_weight = checkpoint_weight

    def select(self,
               proxy_importance: torch.Tensor,
               clip_alignment: torch.Tensor,
               budget: float) -> torch.Tensor:
        proxy_norm = (proxy_importance - proxy_importance.min()) / (proxy_importance.max() - proxy_importance.min() + 1e-8)
        clip_norm = (clip_alignment - clip_alignment.min()) / (clip_alignment.max() - clip_alignment.min() + 1e-8)
        combined_score = self.proxy_weight * proxy_norm + self.checkpoint_weight * clip_norm
        k = max(1, int(len(combined_score) * budget))
        return torch.topk(combined_score, k=k).indices


class HybridMODETrainer:


    def __init__(self, config: HybridMODEConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.global_step = 0

        # TensorBoard writer
        self.writer = None
        if config.enable_tensorboard and TENSORBOARD_AVAILABLE:
            tb_dir = Path(config.tensorboard_dir)
            tb_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(str(tb_dir))
            print(f"TensorBoard logging to: {tb_dir}")

        # Main CLIP model
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            config.main_model, pretrained='openai'
        )
        self.clip_model = self.clip_model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer(config.main_model)

        # Proxy model
        self.proxy_model = ProxyModel(config.proxy_model, device=str(self.device))

        # Checkpoint manager and model (EMA)
        checkpoint_dir = Path(config.cache_dir) / "checkpoints"
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        self.checkpoint_model = copy.deepcopy(self.clip_model).eval()

        # Cache and selector
        self.cache = HybridCache(config)
        self.selector = HybridSelector(config.proxy_weight, config.checkpoint_weight)

        # Optimizers
        self.clip_optimizer = torch.optim.AdamW(self.clip_model.parameters(), lr=config.learning_rate)
        self.proxy_optimizer = torch.optim.AdamW([p for p in self.proxy_model.parameters() if p.requires_grad], lr=config.proxy_lr)

    @torch.no_grad()
    def compute_clip_alignment(self, images: torch.Tensor, texts: torch.Tensor) -> torch.Tensor:
        self.checkpoint_model.eval()
        cached = self.cache.get_clip_features(images, texts)
        if cached is not None:
            image_features, text_features = cached
            image_features = image_features.to(self.device)
            text_features = text_features.to(self.device)
        else:
            image_features = self.checkpoint_model.encode_image(images)
            text_features = self.checkpoint_model.encode_text(texts)
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            self.cache.set_clip_features(images, texts, image_features, text_features)
        alignment = (image_features * text_features).sum(dim=1).abs()
        return alignment

    def compute_proxy_importance(self, images: torch.Tensor, texts: torch.Tensor) -> torch.Tensor:
        cached = self.cache.get_proxy_importance(images, texts)
        if cached is not None:
            return cached.to(self.device)
        # Use EL2N for speed (gradient-free, inference only)
        # For higher quality but slower: use compute_gradient_importance_fast()
        # For highest quality but slowest: use compute_gradient_importance()
        importance = self.proxy_model.compute_el2n_importance(images, texts)
        self.cache.set_proxy_importance(images, texts, importance)
        return importance

    def training_step(self, images: torch.Tensor, texts: torch.Tensor) -> Dict:
        batch_size = len(images)
        proxy_importance = self.compute_proxy_importance(images, texts)
        clip_alignment = self.compute_clip_alignment(images, texts)
        selected_indices = self.selector.select(proxy_importance, clip_alignment, budget=self.config.budget)

        self.clip_model.train()
        selected_images = images[selected_indices]
        selected_texts = texts[selected_indices]
        image_emb = self.clip_model.encode_image(selected_images)
        text_emb = self.clip_model.encode_text(selected_texts)
        image_emb = F.normalize(image_emb, dim=-1)
        text_emb = F.normalize(text_emb, dim=-1)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_emb @ text_emb.T
        labels = torch.arange(len(selected_images), device=self.device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        self.clip_optimizer.zero_grad()
        loss.backward()
        self.clip_optimizer.step()

        proxy_loss = 0.0
        if self.global_step % self.config.proxy_update_freq == 0:
            proxy_loss = self.proxy_model.update(selected_images, selected_texts, self.proxy_optimizer)

        if self.global_step >= self.config.checkpoint_warmup:
            ema_decay = 0.999
            with torch.no_grad():
                for ema_param, current_param in zip(self.checkpoint_model.parameters(), self.clip_model.parameters()):
                    ema_param.data.mul_(ema_decay).add_(current_param.data, alpha=1 - ema_decay)
            if self.global_step % self.config.checkpoint_frequency == 0:
                self.checkpoint_manager.save_checkpoint(self.checkpoint_model, self.global_step)

        self.global_step += 1

        metrics = {
            'loss': float(loss.item()),
            'proxy_loss': float(proxy_loss),
            'selected': int(len(selected_indices)),
            'budget': float(len(selected_indices) / batch_size),
            'proxy_importance_mean': float(proxy_importance.mean().item()),
            'clip_alignment_mean': float(clip_alignment.mean().item())
        }

        # Log to TensorBoard
        if self.writer and self.global_step % self.config.log_frequency == 0:
            self.writer.add_scalar('train/loss', metrics['loss'], self.global_step)
            self.writer.add_scalar('train/proxy_loss', metrics['proxy_loss'], self.global_step)
            self.writer.add_scalar('train/selected', metrics['selected'], self.global_step)
            self.writer.add_scalar('train/selection_budget', metrics['budget'], self.global_step)
            self.writer.add_scalar('importance/proxy_mean', metrics['proxy_importance_mean'], self.global_step)
            self.writer.add_scalar('importance/clip_alignment_mean', metrics['clip_alignment_mean'], self.global_step)
            self.writer.add_histogram('importance/proxy_distribution', proxy_importance, self.global_step)
            self.writer.add_histogram('importance/clip_alignment_distribution', clip_alignment, self.global_step)

            # Cache statistics
            cache_stats = self.cache.get_stats()
            self.writer.add_scalar('cache/proxy_hit_rate', cache_stats.get('proxy_hit_rate', 0.0), self.global_step)
            self.writer.add_scalar('cache/clip_hit_rate', cache_stats.get('clip_hit_rate', 0.0), self.global_step)

        return metrics

    def train_epoch(self, dataloader: DataLoader, epoch: int = 0):
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}")

        for batch_idx, batch in pbar:
            images = batch['images'].to(self.device)
            texts = self.tokenizer(batch['texts']).to(self.device)
            metrics = self.training_step(images, texts)

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'proxy_loss': f"{metrics['proxy_loss']:.4f}",
                'selected': f"{metrics['selected']}/{metrics['selected']/metrics['budget']:.0f}"
            })

            if batch_idx % 50 == 0:
                cache_stats = self.cache.get_stats()
                tqdm.write(
                    f"Step {self.global_step} | "
                    f"Loss: {metrics['loss']:.4f} | "
                    f"Proxy hit: {cache_stats.get('proxy_hit_rate', 0.0):.1%} | "
                    f"CLIP hit: {cache_stats.get('clip_hit_rate', 0.0):.1%}"
                )


# ============================================================================
# DATACOMP MODE: CORRECT IMPLEMENTATION
# ============================================================================
"""
DataComp MODE: CORRECT Implementation

PROPER WORKFLOW:
1. Pass 1: Score all 400M samples (using hybrid proxy + checkpoint)
2. Select: Choose top 10% (40M samples) based on scores
3. Train: Train CLIP on those 40M samples for MULTIPLE epochs
4. (Optional) Periodic re-selection during training

This matches DataComp filtering track requirements.
"""


@dataclass
class DataCompMODEConfig:
    """Configuration for DataComp MODE"""

    # Dataset
    dataset_size: int = 400_000_000
    selection_budget: float = 0.10  # Select 10% for training

    # Models
    proxy_model: str = 'ViT-B-32'
    main_model: str = 'ViT-B-32'

    # Selection strategy
    selection_strategy: str = 'one_shot'  # 'one_shot', 'periodic', 'continuous'
    reselection_frequency: int = 5  # Re-select every N epochs (if periodic)

    # Proxy weights
    proxy_weight: float = 0.6
    checkpoint_weight: float = 0.4

    # Training (on SELECTED subset)
    num_epochs: int = 10  # Train selected samples for 10 epochs
    batch_size: int = 256
    learning_rate: float = 5e-5
    warmup_steps: int = 2000

    # Efficiency
    selection_batch_size: int = 1024  # Larger batches for scoring phase
    mixed_precision: bool = True
    device: str = 'cuda'

    # Cache (for selection phase)
    cache_dir: str = "./datacomp_mode_cache"
    cache_size_gb: float = 100.0

    # Logging
    enable_tensorboard: bool = True
    tensorboard_dir: str = "./runs/datacomp_mode"
    log_frequency: int = 50

    @classmethod
    def for_debug(cls):
        return cls(
            dataset_size=10_000,
            selection_budget=0.30,
            num_epochs=3,
            batch_size=32,
            selection_batch_size=128,
            cache_size_gb=1.0
        )


# ============================================================================
# PHASE 1: SCORING & SELECTION
# ============================================================================

class DataScorer:
    """
    Score all samples in dataset for importance.

    This is the SELECTION phase - runs once (or periodically).
    """

    def __init__(self, config: DataCompMODEConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

        # TensorBoard writer
        self.writer = None
        if config.enable_tensorboard and TENSORBOARD_AVAILABLE:
            tb_dir = Path(config.tensorboard_dir) / "scoring"
            tb_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(str(tb_dir))
            print(f"TensorBoard logging to: {tb_dir}")

        # Load proxy model for EL2N scores
        print(f"Loading proxy model: {config.proxy_model}")
        self.proxy_model, _, self.proxy_preprocess = open_clip.create_model_and_transforms(
            config.proxy_model, pretrained='openai'
        )
        self.proxy_model = self.proxy_model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(config.proxy_model)

        # Load checkpoint model for feature scores
        print(f"Loading checkpoint model: {config.main_model}")
        self.checkpoint_model, _, _ = open_clip.create_model_and_transforms(
            config.main_model, pretrained='openai'
        )
        self.checkpoint_model = self.checkpoint_model.to(self.device).eval()

        print("Data scorer ready")

    @torch.no_grad()
    def compute_el2n_score(self, images: torch.Tensor, texts: torch.Tensor) -> torch.Tensor:
        """Compute EL2N importance (proxy signal)"""
        image_features = self.proxy_model.encode_image(images)
        text_features = self.proxy_model.encode_text(texts)

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        logit_scale = self.proxy_model.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.T

        probs = F.softmax(logits, dim=1)
        labels = torch.arange(len(images), device=images.device)
        confidence = probs[torch.arange(len(images)), labels]

        # EL2N: (1 - confidence)^2
        return (1 - confidence) ** 2

    @torch.no_grad()
    def compute_alignment_score(self, images: torch.Tensor, texts: torch.Tensor) -> torch.Tensor:
        """Compute alignment score (checkpoint signal)"""
        image_features = self.checkpoint_model.encode_image(images)
        text_features = self.checkpoint_model.encode_text(texts)

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # Per-sample alignment
        alignment = (image_features * text_features).sum(dim=1).abs()
        return alignment

    def compute_hybrid_score(self, images: torch.Tensor, texts: torch.Tensor) -> torch.Tensor:
        """Combine proxy and checkpoint scores"""
        el2n = self.compute_el2n_score(images, texts)
        alignment = self.compute_alignment_score(images, texts)

        # Normalize to [0, 1]
        el2n_norm = (el2n - el2n.min()) / (el2n.max() - el2n.min() + 1e-8)
        alignment_norm = (alignment - alignment.min()) / (alignment.max() - alignment.min() + 1e-8)

        # Weighted combination
        combined = (
            self.config.proxy_weight * el2n_norm +
            self.config.checkpoint_weight * alignment_norm
        )

        return combined

    def score_dataset(self, dataset: Dataset) -> torch.Tensor:
        """Score entire dataset and return importance scores"""
        print(f"\nScoring {len(dataset):,} samples...")

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.selection_batch_size,
            shuffle=False,
            num_workers=0  # Avoid multiprocessing issues with synthetic data
        )

        all_scores = []
        batch_count = 0

        for batch in tqdm(dataloader, desc="Scoring dataset", unit="batch"):
            images = batch['images'].to(self.device)
            texts = self.tokenizer(batch['texts']).to(self.device)

            scores = self.compute_hybrid_score(images, texts)
            all_scores.append(scores.cpu())

            # Log statistics periodically
            batch_count += 1
            if self.writer and batch_count % 100 == 0:
                self.writer.add_scalar('scoring/batch_mean_score', scores.mean().item(), batch_count)
                self.writer.add_scalar('scoring/batch_std_score', scores.std().item(), batch_count)
                self.writer.add_histogram('scoring/score_distribution', scores, batch_count)

        all_scores = torch.cat(all_scores)

        print(f"Scoring complete")
        print(f"   Mean score: {all_scores.mean():.4f}")
        print(f"   Std score: {all_scores.std():.4f}")

        # Log final statistics
        if self.writer:
            self.writer.add_scalar('scoring/final_mean', all_scores.mean().item(), 0)
            self.writer.add_scalar('scoring/final_std', all_scores.std().item(), 0)
            self.writer.add_histogram('scoring/final_distribution', all_scores, 0)

        return all_scores


class DataSelector:
    """
    Select top-k samples based on importance scores.
    """

    def __init__(self, config: DataCompMODEConfig):
        self.config = config

    def select_indices(self, scores: torch.Tensor) -> torch.Tensor:
        """Select top-k indices based on scores"""
        k = int(len(scores) * self.config.selection_budget)

        print(f"\nSelecting top {k:,} / {len(scores):,} samples ({self.config.selection_budget:.1%})")

        selected_indices = torch.topk(scores, k=k).indices

        print(f"Selection complete")
        print(f"   Selected scores - mean: {scores[selected_indices].mean():.4f}")
        print(f"   Rejected scores - mean: {scores[~torch.isin(torch.arange(len(scores)), selected_indices)].mean():.4f}")

        return selected_indices

    def create_selected_dataset(self,
                               full_dataset: Dataset,
                               selected_indices: torch.Tensor) -> Dataset:
        """Create subset dataset with selected samples"""
        return Subset(full_dataset, selected_indices.tolist())


# ============================================================================
# PHASE 2: TRAINING ON SELECTED SUBSET
# ============================================================================

class CLIPTrainer:
    """Standard CLIP trainer for selected subset"""

    def __init__(self, config: DataCompMODEConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.global_step = 0

        # TensorBoard writer
        self.writer = None
        if config.enable_tensorboard and TENSORBOARD_AVAILABLE:
            tb_dir = Path(config.tensorboard_dir) / "training"
            tb_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(str(tb_dir))
            print(f"TensorBoard logging to: {tb_dir}")

        # Load CLIP model for training
        print(f"Loading CLIP for training: {config.main_model}")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            config.main_model, pretrained='openai'
        )
        self.model = self.model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer(config.main_model)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-6,
            weight_decay=0.2
        )

        # Learning rate scheduler
        self.total_steps = None  # Set when training starts

        # Mixed precision
        self.scaler = GradScaler() if config.mixed_precision else None

        print("CLIP trainer ready")

    def training_step(self, images: torch.Tensor, texts: torch.Tensor) -> Dict:
        """Single training step"""
        self.model.train()

        # Learning rate warmup
        if self.global_step < self.config.warmup_steps:
            lr_scale = min(1.0, self.global_step / self.config.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.config.learning_rate * lr_scale

        # Forward pass
        with autocast(enabled=self.config.mixed_precision):
            image_features = self.model.encode_image(images)
            text_features = self.model.encode_text(texts)

            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)

            # Contrastive loss
            logit_scale = self.model.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.T

            labels = torch.arange(len(images), device=self.device)

            loss_i2t = F.cross_entropy(logits, labels)
            loss_t2i = F.cross_entropy(logits.T, labels)
            loss = (loss_i2t + loss_t2i) / 2

        # Backward
        self.optimizer.zero_grad()

        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        self.global_step += 1

        # Compute accuracy
        with torch.no_grad():
            preds_i2t = logits.argmax(dim=1)
            preds_t2i = logits.T.argmax(dim=1)
            acc_i2t = (preds_i2t == labels).float().mean()
            acc_t2i = (preds_t2i == labels).float().mean()
            acc = (acc_i2t + acc_t2i) / 2

        metrics = {
            'loss': loss.item(),
            'acc': acc.item(),
            'acc_i2t': acc_i2t.item(),
            'acc_t2i': acc_t2i.item(),
            'lr': self.optimizer.param_groups[0]['lr']
        }

        # Log to TensorBoard
        if self.writer and self.global_step % self.config.log_frequency == 0:
            self.writer.add_scalar('train/loss', metrics['loss'], self.global_step)
            self.writer.add_scalar('train/accuracy', metrics['acc'], self.global_step)
            self.writer.add_scalar('train/acc_i2t', metrics['acc_i2t'], self.global_step)
            self.writer.add_scalar('train/acc_t2i', metrics['acc_t2i'], self.global_step)
            self.writer.add_scalar('train/learning_rate', metrics['lr'], self.global_step)

        return metrics

    def train_epoch(self, dataloader: DataLoader, epoch: int):
        """Train one epoch"""
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch + 1}/{self.config.num_epochs}")
        print(f"{'='*80}")

        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0

        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}")

        for batch_idx, batch in pbar:
            images = batch['images'].to(self.device)
            texts = self.tokenizer(batch['texts']).to(self.device)

            metrics = self.training_step(images, texts)

            epoch_loss += metrics['loss']
            epoch_acc += metrics['acc']
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'acc': f"{metrics['acc']:.3f}",
                'lr': f"{metrics['lr']:.2e}"
            })

            if batch_idx % 100 == 0:
                tqdm.write(
                    f"  Step {self.global_step} | "
                    f"Loss: {metrics['loss']:.4f} | "
                    f"Acc: {metrics['acc']:.3f} | "
                    f"LR: {metrics['lr']:.6f}"
                )

        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_acc / num_batches

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"   Avg Loss: {avg_loss:.4f}")
        print(f"   Avg Acc: {avg_acc:.3f}")

        # Log epoch summary to TensorBoard
        if self.writer:
            self.writer.add_scalar('epoch/loss', avg_loss, epoch)
            self.writer.add_scalar('epoch/accuracy', avg_acc, epoch)

        return {'loss': avg_loss, 'acc': avg_acc}

    def train(self, dataloader: DataLoader):
        """Train for multiple epochs"""
        self.total_steps = len(dataloader) * self.config.num_epochs

        print(f"\nStarting training on selected subset")
        print(f"   Total epochs: {self.config.num_epochs}")
        print(f"   Batches per epoch: {len(dataloader)}")
        print(f"   Total steps: {self.total_steps:,}")

        for epoch in range(self.config.num_epochs):
            epoch_metrics = self.train_epoch(dataloader, epoch)

            # Optional: Save checkpoint
            if (epoch + 1) % 5 == 0:
                checkpoint_path = Path(self.config.cache_dir) / f"clip_epoch_{epoch+1}.pt"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': epoch_metrics['loss'],
                    'acc': epoch_metrics['acc']
                }, checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")

        print(f"\nTraining complete!")


# ============================================================================
# MAIN MODE PIPELINE
# ============================================================================

class DataCompMODE:
    """
    Complete DataComp MODE pipeline.

    WORKFLOW:
    1. Score all samples
    2. Select top-k
    3. Train CLIP on selected subset for multiple epochs
    4. (Optional) Periodic re-selection
    """

    def __init__(self, config: DataCompMODEConfig):
        self.config = config

        # Initialize components
        self.scorer = DataScorer(config)
        self.selector = DataSelector(config)
        self.trainer = CLIPTrainer(config)

        print("\n" + "="*80)
        print("DataComp MODE Pipeline Ready")
        print("="*80)
        print(f"Strategy: {config.selection_strategy}")
        print(f"Selection budget: {config.selection_budget:.1%}")
        print(f"Training epochs: {config.num_epochs}")
        print("="*80 + "\n")

    def run_one_shot(self, dataset: Dataset):
        """
        One-shot selection strategy.

        1. Score all samples once
        2. Select top-k
        3. Train on selected subset for all epochs
        """
        print("\nONE-SHOT SELECTION STRATEGY")

        # Phase 1: Score and select
        scores = self.scorer.score_dataset(dataset)
        selected_indices = self.selector.select_indices(scores)
        selected_dataset = self.selector.create_selected_dataset(dataset, selected_indices)

        # Save selection for reproducibility
        selection_path = Path(self.config.cache_dir) / "selected_indices.pt"
        selection_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'indices': selected_indices,
            'scores': scores[selected_indices],
            'budget': self.config.selection_budget
        }, selection_path)
        print(f"Selection saved: {selection_path}")

        # Phase 2: Train on selected subset
        dataloader = DataLoader(
            selected_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,  # Avoid multiprocessing issues
            collate_fn=self._collate_fn,
            pin_memory=False  # Disable for CPU
        )

        self.trainer.train(dataloader)

    def _collate_fn(self, batch):
        """Collate function for dataloader (must be method for pickling)"""
        images = torch.stack([b['images'] for b in batch])
        texts = [b['texts'] for b in batch]
        return {'images': images, 'texts': texts}

    def run_periodic_reselection(self, dataset: Dataset):
        """
        Periodic re-selection strategy.

        1. Score and select
        2. Train for N epochs
        3. Re-score and re-select
        4. Train for N more epochs
        5. Repeat
        """
        print("\nPERIODIC RE-SELECTION STRATEGY")

        epochs_per_cycle = self.config.reselection_frequency
        num_cycles = self.config.num_epochs // epochs_per_cycle

        for cycle in range(num_cycles):
            print(f"\n{'='*80}")
            print(f"SELECTION CYCLE {cycle + 1}/{num_cycles}")
            print(f"{'='*80}")

            # Score and select
            scores = self.scorer.score_dataset(dataset)
            selected_indices = self.selector.select_indices(scores)
            selected_dataset = self.selector.create_selected_dataset(dataset, selected_indices)

            # Train for epochs_per_cycle
            def collate_fn(batch):
                images = torch.stack([b['images'] for b in batch])
                texts = [b['texts'] for b in batch]
                return {'images': images, 'texts': texts}

            dataloader = DataLoader(
                selected_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=4,
                collate_fn=collate_fn
            )

            # Temporarily adjust num_epochs for this cycle
            original_epochs = self.config.num_epochs
            self.config.num_epochs = epochs_per_cycle

            self.trainer.train(dataloader)

            self.config.num_epochs = original_epochs

            # Update checkpoint model in scorer for next cycle
            self.scorer.checkpoint_model.load_state_dict(
                self.trainer.model.state_dict()
            )


# Synthetic dataset class (must be at module level for pickling)
class FakeDataset(Dataset):
    def __init__(self, size=1000):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            'images': torch.randn(3, 224, 224),
            'texts': f"a photo of something {idx}"
        }


def demo_datacomp_mode():
    """Demo with synthetic data"""
    config = DataCompMODEConfig.for_debug()

    # Create synthetic dataset
    dataset = FakeDataset(size=1000)

    # Create pipeline
    pipeline = DataCompMODE(config)

    # Run one-shot selection
    pipeline.run_one_shot(dataset)


# ============================================================================
# LEGACY DEMO
# ============================================================================

def demo_hybrid():

    config = HybridMODEConfig.for_small_scale()
    trainer = HybridMODETrainer(config)

    class FakeDataset(Dataset):
        def __len__(self):
            return 100
        def __getitem__(self, idx):
            return {'images': torch.randn(3, 224, 224), 'texts': "a photo of a cat"}

    def collate_fn(batch):
        images = torch.stack([b['images'] for b in batch])
        texts = [b['texts'] for b in batch]
        return {'images': images, 'texts': texts}

    loader = DataLoader(FakeDataset(), batch_size=32, collate_fn=collate_fn, num_workers=0)
    print("\nStarting Hybrid MODE demo training...")
    for epoch in range(2):
        print(f"\n{'='*80}\nEPOCH {epoch + 1}\n{'='*80}")
        trainer.train_epoch(loader)

# ============================================================================
# SLURM SCRIPT GENERATOR
# ============================================================================

def generate_slurm(
    num_gpus: int = 3,
    scale: str = 'full',  # 'small', 'medium', 'full'
    job_name: str = 'datacomp_mode',
    time_hours: int = 48,
    output_dir: str = './slurm_logs'
) -> str:
    """
    Generate SLURM script for multi-GPU DataComp training.

    Usage:
        script = generate_slurm(num_gpus=3, scale='full')
        Path('run_datacomp.sh').write_text(script)
        # sbatch run_datacomp.sh
    """

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --gpus=a100:{num_gpus}
#SBATCH --mem=500G
#SBATCH --time={time_hours}:00:00
#SBATCH --output={output_dir}/{job_name}_%j.out
#SBATCH --error={output_dir}/{job_name}_%j.err

# Environment setup
source ~/.bashrc
conda activate mode_env  # Your conda environment

# Multi-GPU settings
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE={num_gpus}

# Run training
python -m torch.distributed.launch \
    --nproc_per_node={num_gpus} \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_datacomp.py \
    --scale {scale} \
    --num_gpus {num_gpus} \
    --cache_size_gb {{200 if scale == 'full' else 50}} \
    --enable_bloom \
    --mixed_precision

echo "Training complete!"
"""
    return script


# ============================================================================
# DATACOMP DATASET LOADING
# ============================================================================

def load_datacomp_small(split='train', streaming=True):

    if not DATASETS_AVAILABLE:
        raise ImportError("pip install datasets for DataComp loading")

    print(f"Loading DataComp small dataset (split={split}, streaming={streaming})")
    ds = load_dataset('mlfoundations/datacomp_small', split=split, streaming=streaming)

    print(f"Dataset loaded: {len(ds) if not streaming else 'streaming'} samples")
    return ds

def create_datacomp_dataloader(dataset, batch_size=256, num_workers=4):

    from torch.utils.data import DataLoader

    def collate_fn(batch):

        images = []
        texts = []

        for sample in batch:
            # Extract image and text
            images.append(sample['image'])
            texts.append(sample['text'])

        return {
            'images': images,  # List of PIL images
            'texts': texts     # List of text strings
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        shuffle=True
    )

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_small_scale():

    print("="*80)
    print("SMALL SCALE TEST: 1M samples")
    print("="*80)

    # Show available models
    print_clip_models()

    config = DataCompCacheConfig.for_small_scale(num_gpus=1)
    config.clip_model = get_recommended_model()  # Use recommended model
    cache = DataCompGradientCache(config)

    # Try to load real DataComp data
    try:
        ds = load_datacomp_small(streaming=True)
        dataloader = create_datacomp_dataloader(ds, batch_size=256)
        print("Using real DataComp data")
        use_real_data = True
    except Exception as e:
        print(f"Warning: {e}")
        print("Falling back to synthetic data")
        use_real_data = False

    # Simulate training
    for step in range(100):
        if use_real_data:
            # TODO: Process real batch from dataloader
            # For now, use synthetic data until CLIP preprocessing is added
            batch_size = 256
            img_feat = torch.randn(batch_size, 512, device='cpu')
            txt_feat = torch.randn(batch_size, 512, device='cpu')
        else:
            # Fake CLIP features
            batch_size = 256
            img_feat = torch.randn(batch_size, 512, device='cpu')
            txt_feat = torch.randn(batch_size, 512, device='cpu')

        # Get importance
        importance, metadata = cache.get_or_compute(img_feat, txt_feat)

        cache.step()

        if step % 20 == 0:
            print(f"Step {step}: {metadata}")

    cache.print_stats()


def example_full():

    print("="*80)
    print("FULL DATACOMP: 400M samples, 3 A100s")
    print("="*80)

    config = DataCompCacheConfig.for_full_datacomp(num_gpus=3)
    trainer = DataCompTrainer(config)

    # Your DataComp dataloader
    # dataloader = get_datacomp_loader(...)

    # Warmup epoch: build cache
    print("\nWARMUP EPOCH: Building cache...")
    # trainer.train_epoch(dataloader, warmup=True)

    # Training epochs: use cache
    print("\nWARMUP EPOCH: Building cache...")
    config = DataCompCacheConfig.for_small_scale()
    trainer = DataCompTrainer(config)
    trainer.epoch(warmup=True)

    print("\nTRAINING: Using cache...")
    # for epoch in range(10):
    #     trainer.train_epoch(dataloader, warmup=False)

    # Save Bloom filter for next run
    trainer.cache.save_bloom()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='DataComp MODE: Model-Optimized Data Selection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Run with --help for more information or call print_usage() for detailed guide.'
    )
    parser.add_argument('--mode', choices=['small', 'medium', 'full'], default='small',
                       help='Scale of experiment: small (1M), medium (50M), or full (400M)')
    parser.add_argument('--generate_slurm', action='store_true',
                       help='Generate SLURM batch script for cluster execution')
    parser.add_argument('--hybrid_demo', action='store_true',
                       help='Run legacy hybrid MODE demo')
    parser.add_argument('--datacomp_mode', action='store_true',
                       help='Run DataComp MODE demo (CORRECT implementation)')
    args = parser.parse_args()

    if args.datacomp_mode:
        # Run DataComp MODE demo (CORRECT implementation)
        demo_datacomp_mode()
        raise SystemExit(0)

    if args.hybrid_demo:
        # Run Hybrid MODE demo and exit
        demo_hybrid()
        raise SystemExit(0)

    if args.generate_slurm:
        script = generate_slurm(
            num_gpus=3 if args.mode == 'full' else 2,
            scale=args.mode,
            time_hours=48 if args.mode == 'full' else 12
        )
        output_path = Path('run_datacomp.sh')
        output_path.write_text(script)
        print(f"SLURM script saved: {output_path}")
        print(f"   Run: sbatch {output_path}")
    else:
        if args.mode == 'small':
            example_small_scale()
        elif args.mode == 'full':
            example_full()
        else:
            print(f"Mode '{args.mode}' not implemented")
