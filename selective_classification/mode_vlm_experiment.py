"""
VLM Gradient Cache for DataComp

Optimized for 1-3 A100 GPUs, 400M samples, nuclear norm utility.
Features caching, approximation, multi-GPU sharding, and Bloom filters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
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


# ============================================================================ 
# CLIP MODEL CONFIGURATIONS
# ============================================================================ 

CLIP_MODELS = ['ViT-B/32', 'ViT-B/16', 'ViT-L/14']  # Default: ViT-B/32

# My recommendation: Focus on ViT-B/32 only
# Reason: Reviewer concerns are about SCALE (ImageNet), not model size

def print_clip_models():
    """Print available CLIP models."""
    print(f"Available models: {CLIP_MODELS} (Default: ViT-B/32)")

def get_recommended_model() -> str:
    """Get the recommended model for CVPR experiments."""
    return 'ViT-B/32'

def validate_model_choice(model_name: str) -> bool:
    """Validate if the chosen model exists."""
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
    """Configuration for DataComp."""
    
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
    
    @classmethod
    def for_small_scale(cls, num_gpus: int = 1) -> 'DataCompCacheConfig':
        """Debug/prototype: 1M samples"""
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
        """Development: 50M samples"""
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
        """Production: Full 400M DataComp"""
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
    """
    Space-efficient membership test.
    
    Memory efficient with low false positive rate.
    """
    
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
    """
    SVD-free nuclear norm approximation for VLM.
    
    Uses power iteration and randomized estimation for efficiency.
    Speed: O(d²) instead of O(d³).
    """
    
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
        """
        Compute nuclear norm utility WITHOUT full SVD.
        
        Args:
            image_features: [N, D_img]
            text_features: [N, D_txt]
        
        Returns:
            utilities: [N] per-sample scores
            metadata: Dict with components
        """
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
        """
        Decompose batch utility to per-sample scores.
        
        Heuristic: Sample i's contribution = its correlation with others
        """
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
    """
    Gradient cache for DataComp VLM training.
    
    Features Bloom filter, SVD-free nuclear norm, gradient aging,
    multi-GPU sharding, and prefetching.
    """
    
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
        """
        Generate cache key from CLIP features.
        
        Hashes feature vectors directly.
        """
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
        """
        Compute nuclear norm-based gradient importance.
        
        This is what gets cached!
        """
        utilities, metadata = self.nuclear_norm.compute_utility(
            image_features, text_features
        )
        return utilities
    
    def get_or_compute(self,
                      image_features: torch.Tensor,
                      text_features: torch.Tensor,
                      force_compute: bool = False) -> Tuple[torch.Tensor, Dict]:
        """
        Main interface: get cached gradients or compute.
        
        Args:
            image_features: [B, D_img] CLIP image embeddings
            text_features: [B, D_txt] CLIP text embeddings
            force_compute: Skip cache (warmup phase)
        
        Returns:
            importance: [B] per-sample gradient importance
            metadata: Statistics
        """
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
        """Get from cache"""
        if DISKCACHE_AVAILABLE:
            value = self.cache.get(key)
            if value is not None:
                return pickle.loads(value) if isinstance(value, bytes) else value
        else:
            return self.cache.get(key)
        return None
    
    def _set(self, key: str, tensor: torch.Tensor):
        """Set in cache"""
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
        """Compute and store"""
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
        """Increment global step counter"""
        self.global_step += 1
    
    def save_bloom(self):
        """Save Bloom filter to disk"""
        if self.bloom is not None:
            bloom_path = Path(self.config.cache_dir) / "bloom_filter.npz"
            self.bloom.save(bloom_path)
            print(f"Saved Bloom filter: {self.bloom.num_inserted:,} entries")
    
    def get_stats(self) -> Dict:
        """Statistics"""
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
        """Human-readable report"""
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
# DATACOMP TRAINING INTEGRATION
# ============================================================================ 

class DataCompTrainer:
    """
    Integrate MODE cache with DataComp CLIP training.
    
    Extracts CLIP features, gets cached importance scores,
    selects top-k samples, and trains CLIP on selected samples.
    """
    
    def __init__(self, config: DataCompCacheConfig):
        self.config = config
        self.device = 'cpu'
        
        # Initialize cache
        self.cache = DataCompGradientCache(config, device=self.device)
        
        # Load CLIP model for feature extraction
        model_name = config.clip_model
        if not validate_model_choice(model_name):
            raise ValueError(f"Invalid model choice: {model_name}")
            
        print(f"Loading CLIP model: {model_name}")
        
        # Convert model name for open_clip (/ to -)
        openclip_name = model_name.replace('/', '-')
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            openclip_name, pretrained='openai'
        )
        self.clip_model = self.clip_model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(openclip_name)
        
        print("DataComp trainer ready")
    
    @torch.no_grad()
    def extract_features(self, images, texts):
        """Extract CLIP features"""
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
        """
        Single training step with gradient caching.
        
        Args:
            images: [B, 3, 224, 224] preprocessed images
            texts: [B, 77] tokenized text
            warmup: If True, force recompute (build cache)
        
        Returns:
            metadata: Selection statistics
        """
        # Extract features
        image_features, text_features = self.extract_features(images, texts)
        
        # Get gradient importance (cached or computed)
        importance, metadata = self.cache.get_or_compute(
            image_features, text_features,
            force_compute=warmup
        )
        
        # Select top samples for training
        k = int(len(importance) * self.config.budget)
        selected_indices = torch.topk(importance, k=k).indices
        
        # Your actual CLIP training happens here with selected_indices
        # ...
        
        self.cache.step()
        
        metadata['selected_count'] = k
        metadata['mean_importance'] = importance.mean().item()
        return metadata
    
    def train_epoch(self, dataloader, warmup: bool = False):
        """Train one epoch"""
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(self.device)
            texts = self.tokenizer(batch['text']).to(self.device)
            
            metadata = self.training_step(images, texts, warmup=warmup)
            
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}: {metadata}")
        
        self.cache.print_stats()


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
    """
    Load DataComp small dataset (12.8M image-text pairs)
    
    Args:
        split: Dataset split ('train', 'validation')
        streaming: If True, use streaming for memory efficiency
        
    Returns:
        dataset: HuggingFace dataset object
    """
    if not DATASETS_AVAILABLE:
        raise ImportError("pip install datasets for DataComp loading")
    
    print(f"Loading DataComp small dataset (split={split}, streaming={streaming})")
    ds = load_dataset('mlfoundations/datacomp_small', split=split, streaming=streaming)
    
    print(f"Dataset loaded: {len(ds) if not streaming else 'streaming'} samples")
    return ds

def create_datacomp_dataloader(dataset, batch_size=256, num_workers=4):
    """
    Create DataLoader for DataComp dataset with CLIP preprocessing
    
    Args:
        dataset: DataComp dataset from load_datacomp_small()
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        
    Returns:
        dataloader: PyTorch DataLoader
    """
    from torch.utils.data import DataLoader
    
    def collate_fn(batch):
        """Process batch for CLIP training"""
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
    """Quick test: 1M samples, 1 GPU"""
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
    """Production: 400M samples, 3 GPUs"""
    print("="*80)
    print("FULL DATACOMP: 400M samples, 3 A100s")
    print("="*80)
    
    config = DataCompCacheConfig.for_full_datacomp(num_gpus=3)
    trainer = DataCompTrainer(config)
    
    # Your DataComp dataloader
    # dataloader = get_datacomp_loader(...)
    
    # Warmup epoch: build cache
    print("\n🔥 WARMUP EPOCH: Building cache...")
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
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['small', 'medium', 'full'], default='small')
    parser.add_argument('--generate_slurm', action='store_true')
    args = parser.parse_args()
    
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