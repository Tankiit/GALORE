"""
LLM Dataset Utilities with Stratification
=========================================

This module provides comprehensive dataset handling for LLM training,
including WikiText-103 with various stratification methods.
"""

import os
import json
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from tqdm import tqdm
import hashlib
from collections import defaultdict
import re
from datasets import load_dataset
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Try to download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass

logger = logging.getLogger(__name__)

# Data directory
DATA_DIR = "/Users/mukher74/research/data"
os.makedirs(DATA_DIR, exist_ok=True)

# =============================================================================
# Dataset Classes
# =============================================================================

@dataclass
class TextSample:
    """Container for text samples with metadata"""
    text: str
    idx: int
    length: int
    quality_score: float = 0.0
    topic: str = "unknown"
    complexity: float = 0.0
    metadata: Dict[str, Any] = None


class WikiText103Dataset(Dataset):
    """WikiText-103 dataset wrapper with stratification support"""
    
    def __init__(self, 
                 split: str = "train",
                 max_samples: Optional[int] = None,
                 cache_dir: str = None):
        
        self.split = split
        self.max_samples = max_samples
        self.cache_dir = cache_dir or os.path.join(DATA_DIR, "wikitext103_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load dataset
        self.samples = self._load_dataset()
        
        # Compute metadata for all samples
        self.metadata = self._compute_metadata()
        
    def _load_dataset(self) -> List[TextSample]:
        """Load WikiText-103 dataset"""
        cache_file = os.path.join(self.cache_dir, f"wikitext103_{self.split}.pkl")
        
        if os.path.exists(cache_file):
            logger.info(f"Loading cached WikiText-103 {self.split} from {cache_file}")
            with open(cache_file, 'rb') as f:
                samples = pickle.load(f)
        else:
            logger.info(f"Loading WikiText-103 {self.split} from HuggingFace")
            dataset = load_dataset("wikitext", "wikitext-103-v1", split=self.split)
            
            samples = []
            for idx, item in enumerate(tqdm(dataset, desc="Processing samples")):
                text = item['text'].strip()
                if text:  # Skip empty texts
                    samples.append(TextSample(
                        text=text,
                        idx=idx,
                        length=len(text.split())
                    ))
            
            # Save cache
            with open(cache_file, 'wb') as f:
                pickle.dump(samples, f)
            logger.info(f"Cached {len(samples)} samples to {cache_file}")
        
        # Limit samples if specified
        if self.max_samples:
            samples = samples[:self.max_samples]
        
        return samples
    
    def _compute_metadata(self) -> Dict[int, Dict[str, Any]]:
        """Compute metadata for all samples"""
        metadata_file = os.path.join(self.cache_dir, f"metadata_{self.split}.pkl")
        
        if os.path.exists(metadata_file):
            logger.info(f"Loading cached metadata from {metadata_file}")
            with open(metadata_file, 'rb') as f:
                return pickle.load(f)
        
        logger.info("Computing metadata for samples...")
        metadata = {}
        
        for sample in tqdm(self.samples, desc="Computing metadata"):
            # Compute quality score
            quality = self._compute_quality_score(sample.text)
            
            # Compute complexity
            complexity = self._compute_complexity(sample.text)
            
            # Update sample
            sample.quality_score = quality
            sample.complexity = complexity
            
            metadata[sample.idx] = {
                'quality': quality,
                'complexity': complexity,
                'length': sample.length
            }
        
        # Save metadata cache
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        
        return metadata
    
    def _compute_quality_score(self, text: str) -> float:
        """Compute quality score based on various text metrics"""
        score = 0.0
        
        # Length score (prefer medium-length texts)
        words = text.split()
        word_count = len(words)
        if 50 <= word_count <= 500:
            score += 0.3
        elif 20 <= word_count < 50 or 500 < word_count <= 1000:
            score += 0.2
        else:
            score += 0.1
        
        # Sentence structure score
        try:
            sentences = sent_tokenize(text)
            if len(sentences) > 1:
                score += 0.2
            # Average sentence length
            avg_sent_len = word_count / max(len(sentences), 1)
            if 10 <= avg_sent_len <= 25:
                score += 0.2
        except:
            pass
        
        # Vocabulary diversity
        unique_words = len(set(words))
        diversity = unique_words / max(word_count, 1)
        score += min(diversity, 0.3)
        
        return min(score, 1.0)
    
    def _compute_complexity(self, text: str) -> float:
        """Compute text complexity score"""
        words = text.split()
        
        # Average word length
        avg_word_len = np.mean([len(w) for w in words]) if words else 0
        
        # Sentence complexity
        try:
            sentences = sent_tokenize(text)
            avg_sent_len = len(words) / max(len(sentences), 1)
        except:
            avg_sent_len = len(words)
        
        # Normalize and combine
        complexity = (min(avg_word_len / 10, 1.0) * 0.5 + 
                     min(avg_sent_len / 30, 1.0) * 0.5)
        
        return complexity
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


class StratifiedDataset(Dataset):
    """Stratified version of a dataset"""
    
    def __init__(self, 
                 base_dataset: WikiText103Dataset,
                 stratification_type: str,
                 bins: Union[int, List[str]],
                 balanced: bool = True):
        
        self.base_dataset = base_dataset
        self.stratification_type = stratification_type
        self.bins = bins
        self.balanced = balanced
        
        # Stratify samples
        self.strata = self._stratify_samples()
        
        # Create balanced or unbalanced sample list
        self.samples = self._create_sample_list()
    
    def _stratify_samples(self) -> Dict[Any, List[TextSample]]:
        """Stratify samples based on type"""
        strata = defaultdict(list)
        
        if self.stratification_type == "quality":
            for sample in self.base_dataset.samples:
                bin_idx = min(int(sample.quality_score * self.bins), self.bins - 1)
                strata[bin_idx].append(sample)
                
        elif self.stratification_type == "length":
            # Define length bins
            if isinstance(self.bins, list):
                length_ranges = {
                    'short': (0, 50),
                    'medium': (50, 200),
                    'long': (200, float('inf'))
                }
            else:
                # Create equal-sized bins
                lengths = [s.length for s in self.base_dataset.samples]
                percentiles = np.percentile(lengths, 
                                          np.linspace(0, 100, self.bins + 1))
                length_ranges = {i: (percentiles[i], percentiles[i+1]) 
                               for i in range(self.bins)}
            
            for sample in self.base_dataset.samples:
                for bin_name, (min_len, max_len) in length_ranges.items():
                    if min_len <= sample.length < max_len:
                        strata[bin_name].append(sample)
                        break
                        
        elif self.stratification_type == "topic":
            # Use topic modeling for stratification
            self._assign_topics()
            for sample in self.base_dataset.samples:
                strata[sample.topic].append(sample)
        
        elif self.stratification_type == "complexity":
            for sample in self.base_dataset.samples:
                bin_idx = min(int(sample.complexity * self.bins), self.bins - 1)
                strata[bin_idx].append(sample)
        
        return dict(strata)
    
    def _assign_topics(self):
        """Assign topics to samples using LDA or keyword matching"""
        if isinstance(self.bins, list):
            # Use keyword-based topic assignment
            topic_keywords = {
                'science': ['research', 'study', 'experiment', 'theory', 'scientific',
                           'biology', 'physics', 'chemistry', 'mathematics'],
                'history': ['century', 'year', 'historical', 'war', 'empire', 
                           'ancient', 'medieval', 'revolution', 'dynasty'],
                'culture': ['art', 'music', 'literature', 'film', 'culture',
                           'tradition', 'festival', 'language', 'society'],
                'technology': ['computer', 'software', 'internet', 'digital',
                              'technology', 'innovation', 'data', 'system'],
                'politics': ['government', 'election', 'policy', 'political',
                            'democracy', 'parliament', 'president', 'law']
            }
            
            for sample in self.base_dataset.samples:
                text_lower = sample.text.lower()
                topic_scores = {}
                
                for topic, keywords in topic_keywords.items():
                    score = sum(1 for kw in keywords if kw in text_lower)
                    topic_scores[topic] = score
                
                # Assign topic with highest score
                if topic_scores:
                    sample.topic = max(topic_scores, key=topic_scores.get)
                else:
                    sample.topic = 'general'
        else:
            # Use LDA for automatic topic discovery
            logger.info("Performing topic modeling with LDA...")
            texts = [s.text for s in self.base_dataset.samples[:1000]]  # Sample for speed
            
            # Vectorize texts
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            doc_term_matrix = vectorizer.fit_transform(texts)
            
            # Fit LDA
            lda = LatentDirichletAllocation(n_components=self.bins, random_state=42)
            lda.fit(doc_term_matrix)
            
            # Assign topics to all samples
            for sample in self.base_dataset.samples:
                try:
                    vec = vectorizer.transform([sample.text])
                    topic_dist = lda.transform(vec)[0]
                    sample.topic = int(np.argmax(topic_dist))
                except:
                    sample.topic = 0
    
    def _create_sample_list(self) -> List[TextSample]:
        """Create balanced or unbalanced sample list"""
        samples = []
        
        if self.balanced:
            # Equal samples from each stratum
            min_stratum_size = min(len(s) for s in self.strata.values())
            for stratum_samples in self.strata.values():
                # Randomly sample from stratum
                indices = np.random.choice(len(stratum_samples), 
                                         min_stratum_size, 
                                         replace=False)
                samples.extend([stratum_samples[i] for i in indices])
        else:
            # Use all samples
            for stratum_samples in self.strata.values():
                samples.extend(stratum_samples)
        
        return samples
    
    def get_stratum_stats(self) -> Dict[str, Any]:
        """Get statistics about stratification"""
        stats = {
            'type': self.stratification_type,
            'num_strata': len(self.strata),
            'total_samples': len(self.samples),
            'strata_sizes': {k: len(v) for k, v in self.strata.items()},
            'balanced': self.balanced
        }
        return stats
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


# =============================================================================
# Stratification Functions
# =============================================================================

def stratify_by_quality(dataset: WikiText103Dataset, 
                        bins: int = 10,
                        balanced: bool = True) -> StratifiedDataset:
    """Stratify dataset by quality scores"""
    logger.info(f"Stratifying by quality into {bins} bins")
    return StratifiedDataset(dataset, "quality", bins, balanced)


def stratify_by_length(dataset: WikiText103Dataset,
                       bins: Union[int, List[str]] = ['short', 'medium', 'long'],
                       balanced: bool = True) -> StratifiedDataset:
    """Stratify dataset by text length"""
    logger.info(f"Stratifying by length into {bins} bins")
    return StratifiedDataset(dataset, "length", bins, balanced)


def stratify_by_topic(dataset: WikiText103Dataset,
                      domains: List[str] = ['science', 'history', 'culture'],
                      balanced: bool = True) -> StratifiedDataset:
    """Stratify dataset by topic/domain"""
    logger.info(f"Stratifying by topic into domains: {domains}")
    return StratifiedDataset(dataset, "topic", domains, balanced)


def stratify_by_complexity(dataset: WikiText103Dataset,
                          bins: int = 5,
                          balanced: bool = True) -> StratifiedDataset:
    """Stratify dataset by text complexity"""
    logger.info(f"Stratifying by complexity into {bins} bins")
    return StratifiedDataset(dataset, "complexity", bins, balanced)


# =============================================================================
# Dataset Creation and Caching
# =============================================================================

def create_wikitext_datasets(max_samples: Optional[int] = None,
                            cache: bool = True) -> Dict[str, Dataset]:
    """Create all WikiText-103 dataset variations"""
    
    cache_file = os.path.join(DATA_DIR, "wikitext103_datasets.pkl")
    
    if cache and os.path.exists(cache_file):
        logger.info(f"Loading cached datasets from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    logger.info("Creating WikiText-103 dataset variations...")
    
    # Create base dataset
    base_dataset = WikiText103Dataset(split="train", max_samples=max_samples)
    
    # Create all variations
    datasets = {
        'wikitext103_full': base_dataset,
        'wikitext103_quality_stratified': stratify_by_quality(base_dataset, bins=10),
        'wikitext103_length_stratified': stratify_by_length(base_dataset, 
                                         bins=['short', 'medium', 'long']),
        'wikitext103_domain_stratified': stratify_by_topic(base_dataset, 
                                       domains=['science', 'history', 'culture', 
                                               'technology', 'politics']),
        'wikitext103_complexity_stratified': stratify_by_complexity(base_dataset, bins=5)
    }
    
    # Add validation splits
    val_dataset = WikiText103Dataset(split="validation", max_samples=max_samples//10 if max_samples else None)
    datasets['wikitext103_val'] = val_dataset
    
    # Print statistics
    for name, dataset in datasets.items():
        logger.info(f"{name}: {len(dataset)} samples")
        if isinstance(dataset, StratifiedDataset):
            stats = dataset.get_stratum_stats()
            logger.info(f"  Strata: {stats['strata_sizes']}")
    
    # Cache datasets
    if cache:
        logger.info(f"Caching datasets to {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump(datasets, f)
    
    return datasets


def save_dataset_metadata(datasets: Dict[str, Dataset], 
                          output_dir: str = None) -> None:
    """Save dataset metadata to JSON"""
    output_dir = output_dir or DATA_DIR
    
    metadata = {}
    for name, dataset in datasets.items():
        meta = {
            'name': name,
            'size': len(dataset),
            'type': type(dataset).__name__
        }
        
        if isinstance(dataset, StratifiedDataset):
            meta.update(dataset.get_stratum_stats())
        
        metadata[name] = meta
    
    # Save to JSON
    output_file = os.path.join(output_dir, "wikitext103_metadata.json")
    with open(output_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved metadata to {output_file}")


# =============================================================================
# Integration with LLM Training
# =============================================================================

class WikiTextDataset(Dataset):
    """PyTorch dataset wrapper for WikiText samples"""
    
    def __init__(self, 
                 samples: List[TextSample],
                 tokenizer: Any,
                 max_length: int = 512):
        
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            sample.text,
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
            'labels': labels,
            'quality_score': sample.quality_score,
            'complexity': sample.complexity,
            'length': sample.length
        }


def create_llm_dataloaders(dataset_name: str,
                          tokenizer: Any,
                          batch_size: int = 8,
                          max_length: int = 512) -> Tuple[DataLoader, DataLoader]:
    """Create dataloaders for LLM training"""
    
    # Load or create datasets
    datasets = create_wikitext_datasets()
    
    if dataset_name not in datasets:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                        f"Available: {list(datasets.keys())}")
    
    # Get train and val datasets
    train_dataset = datasets[dataset_name]
    val_dataset = datasets.get('wikitext103_val', datasets[dataset_name])
    
    # Convert to PyTorch datasets
    if isinstance(train_dataset, (WikiText103Dataset, StratifiedDataset)):
        train_samples = train_dataset.samples if hasattr(train_dataset, 'samples') else list(train_dataset)
        train_dataset = WikiTextDataset(train_samples, tokenizer, max_length)
    
    if isinstance(val_dataset, (WikiText103Dataset, StratifiedDataset)):
        val_samples = val_dataset.samples if hasattr(val_dataset, 'samples') else list(val_dataset)
        val_dataset = WikiTextDataset(val_samples, tokenizer, max_length)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)
    
    return train_loader, val_loader


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create WikiText-103 datasets")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum samples to use")
    parser.add_argument("--output_dir", type=str, default=DATA_DIR,
                       help="Output directory for datasets")
    parser.add_argument("--no_cache", action="store_true",
                       help="Don't use cached datasets")
    
    args = parser.parse_args()
    
    # Create all datasets
    print("Creating WikiText-103 dataset variations...")
    datasets = create_wikitext_datasets(
        max_samples=args.max_samples,
        cache=not args.no_cache
    )
    
    # Save metadata
    save_dataset_metadata(datasets, args.output_dir)
    
    print(f"\nCreated {len(datasets)} dataset variations:")
    for name, dataset in datasets.items():
        print(f"  {name}: {len(dataset)} samples")
    
    print(f"\nDatasets saved to: {args.output_dir}")