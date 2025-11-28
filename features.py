import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import time
import os
import json
import argparse
from tqdm import tqdm
import hashlib
import pickle
from pathlib import Path

from torch_uncertainty.metrics.classification import Entropy

# Additional imports for enhanced functionality
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import faiss


# ============================================================================
# UNCERTAINTY-BASED IMAGE SELECTION
# ============================================================================


def data_uncertainty(model, data_loader, device):
    model.to(device).eval()
    uncertainty_scores = []
    indices = []
    metric = Entropy(reduction='none')  # Get per-sample entropy

    with torch.no_grad():
        batch_idx = 0
        for images, _ in tqdm(data_loader, desc="Computing model uncertainty"):
            images = images.to(device)
            probs = F.softmax(model(images), dim=-1)

            # Compute per-sample entropy
            entropies = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)

            uncertainty_scores.extend(entropies.cpu().numpy())
            batch_size = len(images)
            indices.extend(range(batch_idx, batch_idx + batch_size))
            batch_idx += batch_size

    return np.array(uncertainty_scores), np.array(indices)


def compute_diversity_features(model, data_loader, device, sample_size=1000):
    """Compute diversity features using k-means clustering on embeddings."""
    model.to(device).eval()
    all_features = []
    indices = []

    with torch.no_grad():
        batch_idx = 0
        for images, _ in tqdm(data_loader, desc="Extracting features for diversity"):
            images = images.to(device)

            # Get intermediate features (before final classification layer)
            if hasattr(model, 'fc'):
                # ResNet-like architecture
                feats = torch.nn.functional.adaptive_avg_pool2d(
                    model._forward_impl(images)[:model.layer4[-1].out_channels], (1, 1)
                ).view(images.size(0), -1)
            else:
                # Fallback: use penultimate layer
                feats = model(images)

            all_features.append(feats.cpu().numpy())
            batch_size = len(images)
            indices.extend(range(batch_idx, batch_idx + batch_size))
            batch_idx += batch_size

    all_features = np.vstack(all_features)

    # Sample subset for k-means if dataset is large
    if len(all_features) > sample_size:
        sample_indices = np.random.choice(len(all_features), sample_size, replace=False)
        features_sample = all_features[sample_indices]
    else:
        sample_indices = np.arange(len(all_features))
        features_sample = all_features

    # K-means clustering for diversity
    n_clusters = min(10, len(features_sample) // 10)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(features_sample)

    # Compute diversity scores based on distance to cluster centroids
    diversity_scores = []
    for feat in tqdm(all_features, desc="Computing diversity scores"):
        distances = np.linalg.norm(feat - kmeans.cluster_centers_, axis=1)
        min_distance = np.min(distances)
        diversity_scores.append(min_distance)

    return np.array(diversity_scores), np.array(indices)


def compute_boundary_samples(model, data_loader, device, margin_threshold=0.1):
    """Identify samples near decision boundaries using prediction margins."""
    model.to(device).eval()
    boundary_scores = []
    indices = []

    with torch.no_grad():
        batch_idx = 0
        for images, _ in tqdm(data_loader, desc="Finding boundary samples"):
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=-1)

            # Compute prediction margin (difference between top 2 probabilities)
            top2_probs, _ = torch.topk(probs, 2, dim=-1)
            margins = top2_probs[:, 0] - top2_probs[:, 1]

            # Lower margin = closer to decision boundary
            batch_scores = (1 - margins).cpu().numpy()
            boundary_scores.extend(batch_scores)

            batch_size = len(images)
            indices.extend(range(batch_idx, batch_idx + batch_size))
            batch_idx += batch_size

    return np.array(boundary_scores), np.array(indices)


def extract_comprehensive_features(model, data_loader, device,
                                include_uncertainty=True,
                                include_diversity=True,
                                include_boundary=True):
    """Extract all three types of features: uncertainty, diversity, and boundary."""
    features = {}

    if include_uncertainty:
        print("Extracting uncertainty features...")
        scores, idxs = data_uncertainty(model, data_loader, device)
        features['uncertainty'] = {'scores': scores, 'indices': idxs}

    if include_diversity:
        print("Extracting diversity features...")
        scores, idxs = compute_diversity_features(model, data_loader, device)
        features['diversity'] = {'scores': scores, 'indices': idxs}

    if include_boundary:
        print("Extracting boundary features...")
        scores, idxs = compute_boundary_samples(model, data_loader, device)
        features['boundary'] = {'scores': scores, 'indices': idxs}

    return features


def select_top_k_samples(features_dict, k, feature_type='uncertainty'):
    """Select top k samples based on specified feature type."""
    if feature_type not in features_dict:
        raise ValueError(f"Feature type '{feature_type}' not found in features_dict")

    scores = features_dict[feature_type]['scores']
    indices = features_dict[feature_type]['indices']

    # Get top k indices based on scores (descending for uncertainty/boundary, ascending for diversity)
    if feature_type == 'diversity':
        top_k_global_indices = np.argsort(scores)[-k:]  # Highest diversity
    else:
        top_k_global_indices = np.argsort(scores)[-k:]  # Highest uncertainty/boundary

    # Get corresponding dataset indices
    selected_indices = indices[top_k_global_indices]
    selected_scores = scores[top_k_global_indices]

    return selected_indices, selected_scores 
            

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced PyTorch Script")
    parser.add_argument('--data-path', type=str, default='/Users/tanmoy/research/data', help='Path to the dataset')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset to use')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate for optimizer')
    args = parser.parse_args()

    # Create proper CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    if args.dataset.lower() == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(
            root=args.data_path, 
            train=True, 
            download=True, 
            transform=transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    data_train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    uncertainty = data_uncertainty(model, data_train_loader, 'mps')
    print(uncertainty)
    
