"""
GaLore + RL-Guided Selection with Phase Transition Detection
===========================================================

Implements:
1. Phase transition detection in training dynamics
2. RL-guided adaptive strategy selection
3. Compositional strategy discovery
4. Multi-objective optimization with constraints
5. CIFAR10/100 dataset variations and corrupted versions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import heapq
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
import time
import random
import os
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms.functional as TF

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# CIFAR Dataset Variations and Corruptions
# =============================================================================

class CIFARVariations:
    """Collection of CIFAR dataset variations and corruptions"""
    
    @staticmethod
    def get_cifar10_variations(data_dir: str = "./data", download: bool = True):
        """Get various CIFAR10 dataset variations"""
        variations = {}
        
        # Standard CIFAR10
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        variations['cifar10_standard'] = {
            'train': CIFAR10(data_dir, train=True, download=download, transform=transform_train),
            'test': CIFAR10(data_dir, train=False, download=download, transform=transform_test),
            'name': 'CIFAR10 Standard'
        }
        
        # CIFAR10 with stronger augmentation
        transform_strong = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        variations['cifar10_strong_aug'] = {
            'train': CIFAR10(data_dir, train=True, download=download, transform=transform_strong),
            'test': CIFAR10(data_dir, train=False, download=download, transform=transform_test),
            'name': 'CIFAR10 Strong Augmentation'
        }
        
        # CIFAR10 with cutout
        transform_cutout = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            Cutout(16)
        ])
        
        variations['cifar10_cutout'] = {
            'train': CIFAR10(data_dir, train=True, download=download, transform=transform_cutout),
            'test': CIFAR10(data_dir, train=False, download=download, transform=transform_test),
            'name': 'CIFAR10 Cutout'
        }
        
        return variations
    
    @staticmethod
    def get_cifar100_variations(data_dir: str = "./data", download: bool = True):
        """Get various CIFAR100 dataset variations"""
        variations = {}
        
        # Standard CIFAR100
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        
        variations['cifar100_standard'] = {
            'train': CIFAR100(data_dir, train=True, download=download, transform=transform_train),
            'test': CIFAR100(data_dir, train=False, download=download, transform=transform_test),
            'name': 'CIFAR100 Standard'
        }
        
        # CIFAR100 with stronger augmentation
        transform_strong = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        
        variations['cifar100_strong_aug'] = {
            'train': CIFAR100(data_dir, train=True, download=download, transform=transform_strong),
            'test': CIFAR100(data_dir, train=False, download=download, transform=transform_test),
            'name': 'CIFAR100 Strong Augmentation'
        }
        
        return variations
    
    @staticmethod
    def create_corrupted_cifar10(clean_dataset, corruption_type: str, severity: int = 1):
        """Create corrupted version of CIFAR10 dataset"""
        if corruption_type == 'gaussian_noise':
            return CorruptedCIFAR10(clean_dataset, corruption_type, severity)
        elif corruption_type == 'shot_noise':
            return CorruptedCIFAR10(clean_dataset, corruption_type, severity)
        elif corruption_type == 'impulse_noise':
            return CorruptedCIFAR10(clean_dataset, corruption_type, severity)
        elif corruption_type == 'defocus_blur':
            return CorruptedCIFAR10(clean_dataset, corruption_type, severity)
        elif corruption_type == 'glass_blur':
            return CorruptedCIFAR10(clean_dataset, corruption_type, severity)
        elif corruption_type == 'motion_blur':
            return CorruptedCIFAR10(clean_dataset, corruption_type, severity)
        elif corruption_type == 'zoom_blur':
            return CorruptedCIFAR10(clean_dataset, corruption_type, severity)
        elif corruption_type == 'snow':
            return CorruptedCIFAR10(clean_dataset, corruption_type, severity)
        elif corruption_type == 'frost':
            return CorruptedCIFAR10(clean_dataset, corruption_type, severity)
        elif corruption_type == 'fog':
            return CorruptedCIFAR10(clean_dataset, corruption_type, severity)
        elif corruption_type == 'brightness':
            return CorruptedCIFAR10(clean_dataset, corruption_type, severity)
        elif corruption_type == 'contrast':
            return CorruptedCIFAR10(clean_dataset, corruption_type, severity)
        elif corruption_type == 'elastic_transform':
            return CorruptedCIFAR10(clean_dataset, corruption_type, severity)
        elif corruption_type == 'pixelate':
            return CorruptedCIFAR10(clean_dataset, corruption_type, severity)
        elif corruption_type == 'jpeg_compression':
            return CorruptedCIFAR10(clean_dataset, corruption_type, severity)
        else:
            raise ValueError(f"Unknown corruption type: {corruption_type}")
    
    @staticmethod
    def create_corrupted_cifar100(clean_dataset, corruption_type: str, severity: int = 1):
        """Create corrupted version of CIFAR100 dataset"""
        return CorruptedCIFAR100(clean_dataset, corruption_type, severity)
    
    @staticmethod
    def get_all_corruption_types():
        """Get all available corruption types"""
        return [
            'gaussian_noise', 'shot_noise', 'impulse_noise',
            'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
            'snow', 'frost', 'fog', 'brightness', 'contrast',
            'elastic_transform', 'pixelate', 'jpeg_compression'
        ]


class CorruptedCIFAR10(Dataset):
    """Corrupted CIFAR10 dataset with various corruption types"""
    
    def __init__(self, clean_dataset, corruption_type: str, severity: int = 1):
        self.clean_dataset = clean_dataset
        self.corruption_type = corruption_type
        self.severity = severity
        
    def __len__(self):
        return len(self.clean_dataset)
    
    def __getitem__(self, idx):
        data, label = self.clean_dataset[idx]
        corrupted_data = self._apply_corruption(data)
        return corrupted_data, label
    
    def _apply_corruption(self, data):
        """Apply corruption to data"""
        if self.corruption_type == 'gaussian_noise':
            return self._add_gaussian_noise(data)
        elif self.corruption_type == 'shot_noise':
            return self._add_shot_noise(data)
        elif self.corruption_type == 'impulse_noise':
            return self._add_impulse_noise(data)
        elif self.corruption_type == 'defocus_blur':
            return self._apply_defocus_blur(data)
        elif self.corruption_type == 'glass_blur':
            return self._apply_glass_blur(data)
        elif self.corruption_type == 'motion_blur':
            return self._apply_motion_blur(data)
        elif self.corruption_type == 'zoom_blur':
            return self._apply_zoom_blur(data)
        elif self.corruption_type == 'snow':
            return self._apply_snow(data)
        elif self.corruption_type == 'frost':
            return self._apply_frost(data)
        elif self.corruption_type == 'fog':
            return self._apply_fog(data)
        elif self.corruption_type == 'brightness':
            return self._adjust_brightness(data)
        elif self.corruption_type == 'contrast':
            return self._adjust_contrast(data)
        elif self.corruption_type == 'elastic_transform':
            return self._apply_elastic_transform(data)
        elif self.corruption_type == 'pixelate':
            return self._apply_pixelate(data)
        elif self.corruption_type == 'jpeg_compression':
            return self._apply_jpeg_compression(data)
        else:
            return data
    
    def _add_gaussian_noise(self, data):
        """Add Gaussian noise"""
        noise_level = self.severity * 0.1
        noise = torch.randn_like(data) * noise_level
        corrupted = torch.clamp(data + noise, 0, 1)
        return corrupted
    
    def _add_shot_noise(self, data):
        """Add shot noise (Poisson noise)"""
        noise_level = self.severity * 0.1
        noise = torch.poisson(data * noise_level) / noise_level
        corrupted = torch.clamp(noise, 0, 1)
        return corrupted
    
    def _add_impulse_noise(self, data):
        """Add impulse noise (salt and pepper)"""
        noise_level = self.severity * 0.1
        mask = torch.rand_like(data) < noise_level
        
        # Salt noise
        salt_mask = mask & (torch.rand_like(data) < 0.5)
        data = torch.where(salt_mask, torch.ones_like(data), data)
        
        # Pepper noise
        pepper_mask = mask & (torch.rand_like(data) >= 0.5)
        data = torch.where(pepper_mask, torch.zeros_like(data), data)
        
        return data
    
    def _apply_defocus_blur(self, data):
        """Apply defocus blur"""
        kernel_size = 3 + self.severity * 2
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Create Gaussian kernel
        sigma = self.severity * 0.5
        kernel = self._create_gaussian_kernel(kernel_size, sigma)
        
        # Apply convolution to each channel
        blurred = torch.zeros_like(data)
        for c in range(data.shape[0]):
            blurred[c] = F.conv2d(
                data[c:c+1].unsqueeze(0), 
                kernel.unsqueeze(0).unsqueeze(0),
                padding=kernel_size//2
            ).squeeze()
        
        return blurred
    
    def _create_gaussian_kernel(self, size, sigma):
        """Create Gaussian kernel for blurring"""
        coords = torch.arange(size, dtype=torch.float32)
        coords -= size // 2
        
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        
        return g
    
    def _apply_glass_blur(self, data):
        """Apply glass blur (simplified)"""
        # Simplified glass blur using random displacement
        displacement = self.severity * 0.1
        h, w = data.shape[1], data.shape[2]
        
        # Create random displacement field
        dx = torch.randn(h, w) * displacement
        dy = torch.randn(h, w) * displacement
        
        # Apply displacement
        y_coords, x_coords = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        new_y = torch.clamp(y_coords + dy, 0, h-1).long()
        new_x = torch.clamp(x_coords + dx, 0, w-1).long()
        
        blurred = torch.zeros_like(data)
        for c in range(data.shape[0]):
            blurred[c] = data[c, new_y, new_x]
        
        return blurred
    
    def _apply_motion_blur(self, data):
        """Apply motion blur"""
        # Simplified motion blur
        kernel_size = 3 + self.severity * 2
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Create motion blur kernel
        kernel = torch.zeros(kernel_size, kernel_size)
        kernel[kernel_size//2, :] = 1.0 / kernel_size
        
        # Apply convolution
        blurred = torch.zeros_like(data)
        for c in range(data.shape[0]):
            blurred[c] = F.conv2d(
                data[c:c+1].unsqueeze(0), 
                kernel.unsqueeze(0).unsqueeze(0),
                padding=kernel_size//2
            ).squeeze()
        
        return blurred
    
    def _apply_zoom_blur(self, data):
        """Apply zoom blur (simplified)"""
        # Simplified zoom blur using scaling
        scale_factor = 1.0 + self.severity * 0.1
        
        # Scale up then crop
        h, w = data.shape[1], data.shape[2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        
        # Use interpolation to scale
        scaled = F.interpolate(data.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False)
        
        # Crop back to original size
        start_h = (new_h - h) // 2
        start_w = (new_w - w) // 2
        cropped = scaled[:, :, start_h:start_h+h, start_w:start_w+w]
        
        return cropped.squeeze(0)
    
    def _apply_snow(self, data):
        """Apply snow effect"""
        # Simplified snow effect
        snow_intensity = self.severity * 0.3
        
        # Create snow mask
        snow_mask = torch.rand_like(data) < snow_intensity
        
        # Add white snow
        data = torch.where(snow_mask, torch.ones_like(data), data)
        
        return data
    
    def _apply_frost(self, data):
        """Apply frost effect"""
        # Simplified frost effect
        frost_intensity = self.severity * 0.2
        
        # Reduce brightness and add blue tint
        data = data * (1 - frost_intensity)
        data[2] = torch.clamp(data[2] + frost_intensity * 0.3, 0, 1)  # Increase blue channel
        
        return data
    
    def _apply_fog(self, data):
        """Apply fog effect"""
        # Simplified fog effect
        fog_intensity = self.severity * 0.3
        
        # Add white fog
        fog = torch.ones_like(data) * fog_intensity
        data = data * (1 - fog_intensity) + fog
        
        return data
    
    def _adjust_brightness(self, data):
        """Adjust brightness"""
        factor = 1.0 + (self.severity - 3) * 0.2  # Severity 1-5
        data = torch.clamp(data * factor, 0, 1)
        return data
    
    def _adjust_contrast(self, data):
        """Adjust contrast"""
        factor = 1.0 + (self.severity - 3) * 0.2  # Severity 1-5
        mean = data.mean()
        data = (data - mean) * factor + mean
        data = torch.clamp(data, 0, 1)
        return data
    
    def _apply_elastic_transform(self, data):
        """Apply elastic transform (simplified)"""
        # Simplified elastic transform
        displacement = self.severity * 0.05
        h, w = data.shape[1], data.shape[2]
        
        # Create displacement field
        dx = torch.randn(h, w) * displacement
        dy = torch.randn(h, w) * displacement
        
        # Apply displacement
        y_coords, x_coords = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        new_y = torch.clamp(y_coords + dy, 0, h-1).long()
        new_x = torch.clamp(x_coords + dx, 0, w-1).long()
        
        transformed = torch.zeros_like(data)
        for c in range(data.shape[0]):
            transformed[c] = data[c, new_y, new_x]
        
        return transformed
    
    def _apply_pixelate(self, data):
        """Apply pixelation"""
        factor = max(1, self.severity)
        h, w = data.shape[1], data.shape[2]
        
        # Downsample
        new_h, new_w = h // factor, w // factor
        if new_h < 1 or new_w < 1:
            return data
        
        downsampled = F.interpolate(data.unsqueeze(0), size=(new_h, new_w), mode='nearest')
        
        # Upsample back
        upsampled = F.interpolate(downsampled, size=(h, w), mode='nearest')
        
        return upsampled.squeeze(0)
    
    def _apply_jpeg_compression(self, data):
        """Apply JPEG compression (simplified)"""
        # Simplified JPEG compression using quantization
        quality = max(1, 10 - self.severity * 2)  # Lower quality for higher severity
        
        # Convert to 0-255 range
        data_255 = (data * 255).long()
        
        # Quantize
        quantization_step = 256 // quality
        data_quantized = (data_255 // quantization_step) * quantization_step
        
        # Convert back to 0-1 range
        data_normalized = data_quantized.float() / 255.0
        
        return data_normalized


class CorruptedCIFAR100(CorruptedCIFAR10):
    """Corrupted CIFAR100 dataset"""
    pass


class Cutout:
    """Cutout augmentation"""
    
    def __init__(self, size: int):
        self.size = size
    
    def __call__(self, img):
        h, w = img.shape[1], img.shape[2]
        
        # Random position for cutout
        y = np.random.randint(0, h)
        x = np.random.randint(0, w)
        
        # Apply cutout
        y1 = np.clip(y - self.size // 2, 0, h)
        y2 = np.clip(y + self.size // 2, 0, h)
        x1 = np.clip(x - self.size // 2, 0, w)
        x2 = np.clip(x + self.size // 2, 0, w)
        
        img[:, y1:y2, x1:x2] = 0
        return img


# =============================================================================
# CIFAR Model Architectures
# =============================================================================

class CIFARResNet(nn.Module):
    """ResNet architecture for CIFAR datasets"""
    
    def __init__(self, num_classes: int = 10, depth: int = 20):
        super().__init__()
        self.depth = depth
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Residual blocks
        self.layer1 = self._make_layer(16, 16, depth // 3)
        self.layer2 = self._make_layer(16, 32, depth // 3, stride=2)
        self.layer3 = self._make_layer(32, 64, depth // 3, stride=2)
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        
        # First block with potential downsampling
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


class ResidualBlock(nn.Module):
    """Residual block for ResNet"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += residual
        out = F.relu(out)
        
        return out


class CIFARVGG(nn.Module):
    """VGG-style architecture for CIFAR datasets"""
    
    def __init__(self, num_classes: int = 10, depth: int = 16):
        super().__init__()
        
        # Feature extraction layers
        self.features = self._make_features(depth)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_features(self, depth):
        layers = []
        in_channels = 3
        
        # VGG-like architecture
        if depth == 16:
            cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        else:  # depth == 19
            cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                    nn.BatchNorm2d(v),
                    nn.ReLU(inplace=True)
                ]
                in_channels = v
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# =============================================================================
# Phase Transition Detection
# =============================================================================

@dataclass
class TrainingPhase:
    """Represents a training phase with specific characteristics"""
    start_epoch: int
    end_epoch: Optional[int]
    dominant_strategy: str
    gradient_properties: Dict[str, float]
    loss_landscape: str  # 'chaotic', 'plateau', 'steep', 'converging'
    

class PhaseTransitionDetector:
    """
    Detects phase transitions in training dynamics using multiple signals:
    - Gradient norm trajectories
    - Loss curvature changes
    - Gradient alignment patterns
    - Data utility decay rates
    """
    
    def __init__(self, 
                 window_size: int = 50,
                 sensitivity: float = 2.0,
                 min_phase_length: int = 10):
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.min_phase_length = min_phase_length
        
        # History buffers
        self.gradient_norms = deque(maxlen=window_size)
        self.gradient_alignments = deque(maxlen=window_size)
        self.loss_values = deque(maxlen=window_size)
        self.hessian_traces = deque(maxlen=window_size)
        self.selection_entropies = deque(maxlen=window_size)
        
        # Phase tracking
        self.current_phase = TrainingPhase(
            start_epoch=0,
            end_epoch=None,
            dominant_strategy='uncertainty',
            gradient_properties={},
            loss_landscape='chaotic'
        )
        self.phase_history = []
        self.transition_scores = deque(maxlen=window_size)
        
    def update(self, 
               gradients: Dict[str, torch.Tensor],
               loss: float,
               selected_indices: List[int],
               epoch: int) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Update detector state and check for phase transition
        
        Returns:
            is_transition: Whether a phase transition is detected
            confidence: Confidence score of the transition (0-1)
            indicators: Dict of transition indicators
        """
        # Compute gradient statistics
        grad_norm = self._compute_gradient_norm(gradients)
        grad_alignment = self._compute_gradient_alignment(gradients)
        hessian_trace = self._estimate_hessian_trace(gradients, loss)
        selection_entropy = self._compute_selection_entropy(selected_indices)
        
        # Update histories
        self.gradient_norms.append(grad_norm)
        self.gradient_alignments.append(grad_alignment)
        self.loss_values.append(loss)
        self.hessian_traces.append(hessian_trace)
        self.selection_entropies.append(selection_entropy)
        
        # Compute transition indicators
        indicators = self._compute_transition_indicators()
        
        # Compute overall transition score
        transition_score = self._compute_transition_score(indicators)
        self.transition_scores.append(transition_score)
        
        # Detect transition
        is_transition = False
        confidence = 0.0
        
        if len(self.transition_scores) >= self.min_phase_length:
            # Check if transition score exceeds threshold
            recent_scores = list(self.transition_scores)[-5:]
            avg_score = np.mean(recent_scores)
            
            if avg_score > self.sensitivity:
                is_transition = True
                confidence = min(avg_score / (self.sensitivity * 2), 1.0)
                
                # End current phase and start new one
                self.current_phase.end_epoch = epoch
                self.phase_history.append(self.current_phase)
                
                # Characterize new phase
                new_phase_props = self._characterize_phase(indicators)
                self.current_phase = TrainingPhase(
                    start_epoch=epoch,
                    end_epoch=None,
                    dominant_strategy=new_phase_props['strategy'],
                    gradient_properties=new_phase_props['properties'],
                    loss_landscape=new_phase_props['landscape']
                )
                
        return is_transition, confidence, indicators
    
    def _compute_gradient_norm(self, gradients: Dict[str, torch.Tensor]) -> float:
        """Compute total gradient norm"""
        total_norm = 0.0
        for grad in gradients.values():
            total_norm += torch.norm(grad).item() ** 2
        return np.sqrt(total_norm)
    
    def _compute_gradient_alignment(self, gradients: Dict[str, torch.Tensor]) -> float:
        """Compute alignment with previous gradient"""
        if len(self.gradient_norms) == 0:
            return 1.0
            
        # Flatten current and previous gradients
        current_flat = torch.cat([g.flatten() for g in gradients.values()])
        
        # Compare with running average (more stable than single previous)
        if hasattr(self, '_gradient_avg'):
            prev_flat = self._gradient_avg
            alignment = F.cosine_similarity(current_flat, prev_flat, dim=0).item()
            # Update running average
            self._gradient_avg = 0.9 * prev_flat + 0.1 * current_flat
        else:
            self._gradient_avg = current_flat
            alignment = 1.0
            
        return alignment
    
    def _estimate_hessian_trace(self, gradients: Dict[str, torch.Tensor], loss: float) -> float:
        """Estimate trace of Hessian (curvature indicator)"""
        if len(self.loss_values) < 2:
            return 0.0
            
        # Finite difference approximation
        if len(self.loss_values) >= 3:
            # Second-order finite difference
            trace_estimate = self.loss_values[-1] - 2 * self.loss_values[-2] + self.loss_values[-3]
        else:
            trace_estimate = self.loss_values[-1] - self.loss_values[-2]
            
        return abs(trace_estimate)
    
    def _compute_selection_entropy(self, selected_indices: List[int]) -> float:
        """Compute entropy of selection distribution"""
        if len(selected_indices) == 0:
            return 0.0
            
        # Create histogram of selections
        unique, counts = np.unique(selected_indices, return_counts=True)
        probs = counts / counts.sum()
        
        # Compute entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        return entropy
    
    def _compute_transition_indicators(self) -> Dict[str, float]:
        """Compute various transition indicators"""
        indicators = {}
        
        if len(self.gradient_norms) < 5:
            return {k: 0.0 for k in ['norm_change', 'alignment_drop', 'curvature_shift', 
                                     'entropy_change', 'landscape_shift']}
        
        # 1. Gradient norm derivative
        norms = np.array(list(self.gradient_norms))
        norm_derivative = np.gradient(norms)
        indicators['norm_change'] = abs(norm_derivative[-1]) / (np.std(norm_derivative) + 1e-8)
        
        # 2. Alignment drop
        alignments = np.array(list(self.gradient_alignments))
        alignment_drop = max(0, alignments[-5:].mean() - alignments[-1])
        indicators['alignment_drop'] = alignment_drop
        
        # 3. Curvature shift
        traces = np.array(list(self.hessian_traces))
        if len(traces) >= 10:
            recent_curvature = traces[-5:].mean()
            past_curvature = traces[-10:-5].mean()
            indicators['curvature_shift'] = abs(recent_curvature - past_curvature) / (past_curvature + 1e-8)
        else:
            indicators['curvature_shift'] = 0.0
            
        # 4. Selection entropy change
        entropies = np.array(list(self.selection_entropies))
        entropy_derivative = np.gradient(entropies)
        indicators['entropy_change'] = abs(entropy_derivative[-1]) / (np.std(entropy_derivative) + 1e-8)
        
        # 5. Loss landscape shift (based on loss variance)
        losses = np.array(list(self.loss_values))
        recent_var = np.var(losses[-10:])
        past_var = np.var(losses[-20:-10]) if len(losses) >= 20 else recent_var
        indicators['landscape_shift'] = abs(recent_var - past_var) / (past_var + 1e-8)
        
        return indicators
    
    def _compute_transition_score(self, indicators: Dict[str, float]) -> float:
        """Combine indicators into overall transition score"""
        weights = {
            'norm_change': 0.3,
            'alignment_drop': 0.25,
            'curvature_shift': 0.2,
            'entropy_change': 0.15,
            'landscape_shift': 0.1
        }
        
        score = sum(weights[k] * indicators[k] for k in weights)
        return score
    
    def _characterize_phase(self, indicators: Dict[str, float]) -> Dict[str, Any]:
        """Characterize the new phase based on indicators"""
        # Determine loss landscape type
        if indicators['curvature_shift'] > 2.0:
            landscape = 'chaotic'
        elif indicators['landscape_shift'] < 0.1:
            landscape = 'plateau'
        elif indicators['norm_change'] > 3.0:
            landscape = 'steep'
        else:
            landscape = 'converging'
            
        # Determine optimal strategy
        if landscape == 'chaotic':
            strategy = 'diversity'  # Explore broadly in chaotic phase
        elif landscape == 'plateau':
            strategy = 'uncertainty'  # Focus on uncertain samples
        elif landscape == 'steep':
            strategy = 'gradient_magnitude'  # Follow steep directions
        else:
            strategy = 'hybrid'  # Balanced approach
            
        # Compute phase properties
        properties = {
            'avg_gradient_norm': np.mean(list(self.gradient_norms)),
            'gradient_stability': np.mean(list(self.gradient_alignments)),
            'loss_variance': np.var(list(self.loss_values)),
            'selection_diversity': np.mean(list(self.selection_entropies))
        }
        
        return {
            'strategy': strategy,
            'landscape': landscape,
            'properties': properties
        }
    
    def get_phase_summary(self) -> Dict[str, Any]:
        """Get summary of detected phases"""
        return {
            'current_phase': self.current_phase,
            'phase_history': self.phase_history,
            'num_transitions': len(self.phase_history),
            'avg_phase_length': np.mean([p.end_epoch - p.start_epoch 
                                        for p in self.phase_history 
                                        if p.end_epoch is not None]) if self.phase_history else 0
        }


# =============================================================================
# RL Components for Adaptive Selection
# =============================================================================

class SelectionStrategy(Enum):
    """Available selection strategies"""
    GRADIENT_MAGNITUDE = "grad_mag"
    GRADIENT_VARIANCE = "grad_var"
    INFLUENCE_SCORE = "influence"
    DIVERSITY = "diversity"
    UNCERTAINTY = "uncertainty"
    GRADIENT_CONFLICT = "grad_conflict"
    FORGETTING = "forgetting"
    HYBRID = "hybrid"


@dataclass
class RLState:
    """State representation for RL policy"""
    # Phase indicators
    phase_indicators: Dict[str, float]
    current_phase_type: str
    epochs_in_phase: int
    
    # Performance metrics
    recent_performance: List[float]
    performance_trend: float
    
    # Resource usage
    memory_usage: float
    compute_budget_used: float
    
    # Strategy effectiveness
    strategy_rewards: Dict[str, float]
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to tensor for neural network input"""
        features = []
        
        # Phase indicators (5 features)
        for key in ['norm_change', 'alignment_drop', 'curvature_shift', 
                   'entropy_change', 'landscape_shift']:
            features.append(self.phase_indicators.get(key, 0.0))
            
        # Phase type one-hot (4 features)
        phase_types = ['chaotic', 'plateau', 'steep', 'converging']
        phase_one_hot = [1.0 if self.current_phase_type == pt else 0.0 for pt in phase_types]
        features.extend(phase_one_hot)
        
        # Other features
        features.extend([
            self.epochs_in_phase / 100.0,  # Normalized
            self.performance_trend,
            self.memory_usage,
            self.compute_budget_used
        ])
        
        # Strategy rewards (8 features)
        for strategy in SelectionStrategy:
            features.append(self.strategy_rewards.get(strategy.value, 0.0))
            
        return torch.tensor(features, dtype=torch.float32)


class AdaptiveSelectionPolicy(nn.Module):
    """
    RL policy that learns to select data selection strategies
    based on training phase and performance feedback
    """
    
    def __init__(self, state_dim: int = 21, hidden_dim: int = 256, num_strategies: int = 8):
        super().__init__()
        
        # Actor network (outputs strategy weights)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_strategies),
            nn.Softmax(dim=-1)
        )
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )
        
        # Phase prediction head
        self.phase_predictor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # Predict next k epochs transition probability
        )
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            strategy_weights: Weights for each selection strategy
            value: Estimated value of current state
            phase_predictions: Transition probabilities for next k epochs
        """
        strategy_weights = self.actor(state)
        value = self.critic(state)
        phase_predictions = torch.sigmoid(self.phase_predictor(state))
        
        return strategy_weights, value, phase_predictions
    
    def select_action(self, state: torch.Tensor, epsilon: float = 0.1) -> Tuple[int, torch.Tensor]:
        """Select strategy with epsilon-greedy exploration"""
        num_strategies = len(list(SelectionStrategy))
        
        if np.random.random() < epsilon:
            # Random exploration
            action = np.random.randint(0, num_strategies)
            return action, torch.tensor([1.0 / num_strategies] * num_strategies)
        else:
            # Exploit learned policy
            with torch.no_grad():
                strategy_weights, _, _ = self.forward(state)
                action = torch.multinomial(strategy_weights, 1).item()
                return action, strategy_weights


# =============================================================================
# Compositional Strategy Discovery
# =============================================================================

class StrategyPrimitive:
    """Base class for strategy primitives"""
    
    def __init__(self, name: str):
        self.name = name
        
    def score(self, data_point: Any, context: Dict[str, Any]) -> float:
        """Compute score for data point given context"""
        raise NotImplementedError


class GradientMagnitudePrimitive(StrategyPrimitive):
    """Score based on gradient magnitude"""
    
    def __init__(self):
        super().__init__("gradient_magnitude")
        
    def score(self, data_point: Any, context: Dict[str, Any]) -> float:
        gradients = context.get('gradients', {})
        total_norm = 0.0
        for grad in gradients.values():
            total_norm += torch.norm(grad).item() ** 2
        return np.sqrt(total_norm)


class GradientVariancePrimitive(StrategyPrimitive):
    """Score based on temporal gradient variance"""
    
    def __init__(self, window_size: int = 10):
        super().__init__("gradient_variance")
        self.window_size = window_size
        
    def score(self, data_point: Any, context: Dict[str, Any]) -> float:
        grad_history = context.get('gradient_history', [])
        if len(grad_history) < 2:
            return 0.0
            
        # Compute variance over recent history
        recent_grads = grad_history[-self.window_size:]
        variances = []
        
        for param_name in recent_grads[0]:
            param_grads = [g[param_name] for g in recent_grads if param_name in g]
            if param_grads:
                stacked = torch.stack(param_grads)
                variances.append(torch.var(stacked).item())
                
        return np.mean(variances) if variances else 0.0


class DiversityPrimitive(StrategyPrimitive):
    """Score based on diversity from selected set"""
    
    def __init__(self):
        super().__init__("diversity")
        
    def score(self, data_point: Any, context: Dict[str, Any]) -> float:
        selected_features = context.get('selected_features', [])
        if not selected_features:
            return 1.0
            
        # Compute minimum distance to selected set
        point_features = context.get('point_features', torch.randn(128))
        min_distance = float('inf')
        
        for selected in selected_features:
            distance = torch.norm(point_features - selected).item()
            min_distance = min(min_distance, distance)
            
        return min_distance


class ComposedStrategy:
    """Strategy composed from primitives"""
    
    def __init__(self, 
                 primitives: List[StrategyPrimitive],
                 operators: List[str],
                 weights: List[float]):
        self.primitives = primitives
        self.operators = operators
        self.weights = weights
        self.performance_history = []
        
    def evaluate(self, data_point: Any, context: Dict[str, Any]) -> float:
        """Evaluate composed strategy"""
        scores = [p.score(data_point, context) for p in self.primitives]
        
        # Apply operators
        result = scores[0] * self.weights[0]
        for i, op in enumerate(self.operators):
            if i + 1 < len(scores):
                if op == 'add':
                    result += scores[i + 1] * self.weights[i + 1]
                elif op == 'multiply':
                    result *= scores[i + 1] * self.weights[i + 1]
                elif op == 'max':
                    result = max(result, scores[i + 1] * self.weights[i + 1])
                elif op == 'min':
                    result = min(result, scores[i + 1] * self.weights[i + 1])
                    
        return result
    
    def mutate(self, mutation_rate: float = 0.1) -> 'ComposedStrategy':
        """Create mutated version of strategy"""
        new_weights = []
        for w in self.weights:
            if np.random.random() < mutation_rate:
                # Mutate weight
                new_w = w + np.random.normal(0, 0.1)
                new_w = max(0.0, min(1.0, new_w))  # Clip to [0, 1]
            else:
                new_w = w
            new_weights.append(new_w)
            
        # Potentially change an operator
        new_operators = self.operators.copy()
        if np.random.random() < mutation_rate and new_operators:
            idx = np.random.randint(len(new_operators))
            new_operators[idx] = np.random.choice(['add', 'multiply', 'max', 'min'])
            
        return ComposedStrategy(self.primitives, new_operators, new_weights)


class StrategyDiscoveryEngine:
    """Discovers new selection strategies through compositional learning"""
    
    def __init__(self, 
                 population_size: int = 50,
                 tournament_size: int = 5,
                 mutation_rate: float = 0.1):
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        
        # Available primitives
        self.primitives = [
            GradientMagnitudePrimitive(),
            GradientVariancePrimitive(),
            DiversityPrimitive()
        ]
        
        # Population of strategies
        self.population = self._initialize_population()
        self.generation = 0
        self.best_strategies = []
        
    def _initialize_population(self) -> List[ComposedStrategy]:
        """Initialize random population"""
        population = []
        
        for _ in range(self.population_size):
            # Random number of primitives (2-4)
            num_primitives = np.random.randint(2, min(5, len(self.primitives) + 1))
            selected_primitives = np.random.choice(self.primitives, num_primitives, replace=True)
            
            # Random operators
            operators = [np.random.choice(['add', 'multiply', 'max', 'min']) 
                        for _ in range(num_primitives - 1)]
            
            # Random weights
            weights = np.random.uniform(0.1, 1.0, num_primitives)
            
            strategy = ComposedStrategy(list(selected_primitives), operators, list(weights))
            population.append(strategy)
            
        return population
    
    def evolve(self, fitness_scores: List[float]) -> List[ComposedStrategy]:
        """Evolve population based on fitness scores"""
        # Tournament selection
        new_population = []
        
        for _ in range(self.population_size):
            # Select tournament participants
            tournament_indices = np.random.choice(self.population_size, 
                                                self.tournament_size, 
                                                replace=False)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            
            # Winner
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            winner = self.population[winner_idx]
            
            # Create offspring (with mutation)
            offspring = winner.mutate(self.mutation_rate)
            new_population.append(offspring)
            
        # Keep best strategy (elitism)
        best_idx = np.argmax(fitness_scores)
        new_population[0] = self.population[best_idx]
        self.best_strategies.append((self.population[best_idx], fitness_scores[best_idx]))
        
        self.population = new_population
        self.generation += 1
        
        return new_population


# =============================================================================
# GaLore Integration
# =============================================================================

class GaLore:
    """Gradient Low-Rank Projection"""
    
    def __init__(self, rank: int = 256, update_proj_gap: int = 200):
        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.step = 0
        self.projectors = {}
        
    def project_gradient(self, grad: torch.Tensor, name: str) -> torch.Tensor:
        """Project gradient to low-rank subspace"""
        if grad.dim() < 2 or grad.numel() < self.rank * 2:
            return grad
            
        grad_2d = grad.view(grad.shape[0], -1)
        
        # Update projector periodically
        if name not in self.projectors or self.step % self.update_proj_gap == 0:
            try:
                # Try SVD with better error handling
                if hasattr(torch, 'svd_lowrank'):
                    U, _, V = torch.svd_lowrank(grad_2d, q=self.rank)
                    self.projectors[name] = (U.detach(), V.detach())
                else:
                    # Fallback for older PyTorch versions
                    raise RuntimeError("svd_lowrank not available")
            except Exception as e:
                logger.warning(f"SVD failed for {name}, using random projection: {e}")
                # Random projection fallback
                m, n = grad_2d.shape
                U = torch.randn(m, self.rank, device=grad.device)
                V = torch.randn(n, self.rank, device=grad.device)
                
                # Try to use QR decomposition with fallback
                try:
                    if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'qr'):
                        U, _ = torch.linalg.qr(U)
                        V, _ = torch.linalg.qr(V)
                    else:
                        # Fallback for older PyTorch versions
                        U = U / torch.norm(U, dim=0, keepdim=True)
                        V = V / torch.norm(V, dim=0, keepdim=True)
                except:
                    # Final fallback - normalize columns
                    U = U / torch.norm(U, dim=0, keepdim=True)
                    V = V / torch.norm(V, dim=0, keepdim=True)
                
                self.projectors[name] = (U, V)
                
        U, V = self.projectors[name]
        projected = U.T @ grad_2d @ V
        
        self.step += 1
        return projected


# =============================================================================
# Main RL-Guided Selection Framework
# =============================================================================

class RLGuidedGaLoreSelector:
    """
    Main framework combining:
    - Phase transition detection
    - RL-guided strategy selection
    - Compositional strategy discovery
    - GaLore gradient compression
    """
    
    def __init__(self,
                 model: nn.Module,
                 train_dataset: Dataset,
                 val_dataset: Dataset,
                 memory_budget_mb: int = 1000,
                 rank: int = 256):
        
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        # Get device safely
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            # Model has no parameters, use CPU as fallback
            self.device = torch.device('cpu')
            logger.warning("Model has no parameters, using CPU device")
        
        # Core components
        self.phase_detector = PhaseTransitionDetector()
        self.galore = GaLore(rank=rank)
        self.rl_policy = AdaptiveSelectionPolicy()
        self.strategy_discovery = StrategyDiscoveryEngine()
        
        # Resource management
        self.memory_budget = memory_budget_mb * 1024 * 1024
        self.current_memory = 0
        
        # History tracking
        self.selection_history = []
        self.performance_history = []
        self.gradient_history = deque(maxlen=100)
        self.strategy_rewards = defaultdict(lambda: deque(maxlen=100))
        
        # Training state
        self.epoch = 0
        self.total_selections = 0
        
        logger.info(f"Initialized RL-Guided GaLore Selector with rank={rank}")
        
    def select_coreset(self, 
                      budget: int,
                      current_performance: float) -> Tuple[List[int], Dict[str, Any]]:
        """
        Main selection method using RL policy and phase detection
        
        Returns:
            selected_indices: Indices of selected data points
            selection_info: Dictionary with selection metadata
        """
        # Get current gradients for phase detection
        sample_gradients = self._compute_sample_gradients(min(100, len(self.train_dataset)))
        
        # Check for phase transition
        is_transition, confidence, indicators = self.phase_detector.update(
            sample_gradients,
            current_performance,
            self.selection_history[-budget:] if len(self.selection_history) > budget else [],
            self.epoch
        )
        
        # Log phase transition
        if is_transition:
            logger.info(f"Phase transition detected at epoch {self.epoch} with confidence {confidence:.2f}")
            logger.info(f"New phase: {self.phase_detector.current_phase.loss_landscape}")
            
        # Prepare RL state
        rl_state = self._prepare_rl_state(indicators, current_performance)
        state_tensor = rl_state.to_tensor().unsqueeze(0)
        
        # Get strategy weights from RL policy
        with torch.no_grad():
            strategy_weights, value, phase_predictions = self.rl_policy(state_tensor)
            
        # Log phase predictions
        logger.info(f"Phase transition predictions for next 4 epochs: {phase_predictions.squeeze().tolist()}")
        
        # Select strategy based on weights
        strategy_idx, _ = self.rl_policy.select_action(state_tensor, epsilon=0.1 if self.epoch < 100 else 0.05)
        selected_strategy = list(SelectionStrategy)[strategy_idx]
        
        logger.info(f"Selected strategy: {selected_strategy.value}")
        
        # Perform selection using chosen strategy
        if selected_strategy == SelectionStrategy.HYBRID:
            # Use compositional strategy from discovery engine
            selected_indices = self._select_with_composed_strategy(budget)
        else:
            selected_indices = self._select_with_strategy(selected_strategy, budget)
            
        # Update histories
        self.selection_history.extend(selected_indices)
        self.epoch += 1
        self.total_selections += len(selected_indices)
        
        # Compute selection info
        selection_info = {
            'phase_transition': is_transition,
            'transition_confidence': confidence,
            'phase_indicators': indicators,
            'current_phase': self.phase_detector.current_phase.loss_landscape,
            'selected_strategy': selected_strategy.value,
            'strategy_weights': strategy_weights.squeeze().tolist(),
            'predicted_value': value.item(),
            'phase_predictions': phase_predictions.squeeze().tolist(),
            'compression_ratio': self._compute_compression_ratio()
        }
        
        return selected_indices, selection_info
    
    def _compute_sample_gradients(self, n_samples: int) -> Dict[str, torch.Tensor]:
        """Compute gradients on a sample of data"""
        indices = np.random.choice(len(self.train_dataset), n_samples, replace=False)
        subset = Subset(self.train_dataset, indices)
        loader = DataLoader(subset, batch_size=32)
        
        # Accumulate gradients
        self.model.zero_grad()
        total_loss = 0
        
        for batch in loader:
            data, labels = batch[0].to(self.device), batch[1].to(self.device)
            outputs = self.model(data)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            total_loss += loss.item()
            
        # Extract and compress gradients
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                compressed = self.galore.project_gradient(param.grad, name)
                gradients[name] = compressed
                
        # Update gradient history
        self.gradient_history.append(gradients)
        
        return gradients
    
    def _prepare_rl_state(self, indicators: Dict[str, float], current_performance: float) -> RLState:
        """Prepare state for RL policy"""
        # Performance trend
        self.performance_history.append(current_performance)
        if len(self.performance_history) >= 5:
            recent_perf = self.performance_history[-5:]
            performance_trend = (recent_perf[-1] - recent_perf[0]) / (abs(recent_perf[0]) + 1e-8)
        else:
            performance_trend = 0.0
            
        # Compute average strategy rewards
        avg_rewards = {}
        for strategy in SelectionStrategy:
            if strategy.value in self.strategy_rewards:
                avg_rewards[strategy.value] = np.mean(list(self.strategy_rewards[strategy.value]))
            else:
                avg_rewards[strategy.value] = 0.0
                
        return RLState(
            phase_indicators=indicators,
            current_phase_type=self.phase_detector.current_phase.loss_landscape,
            epochs_in_phase=self.epoch - self.phase_detector.current_phase.start_epoch,
            recent_performance=self.performance_history[-5:] if len(self.performance_history) >= 5 else [0.0] * 5,
            performance_trend=performance_trend,
            memory_usage=self.current_memory / self.memory_budget,
            compute_budget_used=self.total_selections / (len(self.train_dataset) * 10),  # Assume 10 epoch budget
            strategy_rewards=avg_rewards
        )
    
    def _select_with_strategy(self, strategy: SelectionStrategy, budget: int) -> List[int]:
        """Select data points using specified strategy"""
        try:
            n = len(self.train_dataset)
            scores = np.zeros(n)
            
            # Compute scores based on strategy
            if strategy == SelectionStrategy.GRADIENT_MAGNITUDE:
                # Score based on gradient magnitude
                for i in range(min(n, 1000)):  # Sample for efficiency
                    try:
                        grad = self._compute_single_gradient(i)
                        if grad:  # Check if gradients were computed successfully
                            scores[i] = sum(torch.norm(g).item() for g in grad.values())
                        else:
                            scores[i] = 0.0
                    except Exception as e:
                        logger.warning(f"Gradient computation failed for index {i}: {e}")
                        scores[i] = 0.0
                        
            elif strategy == SelectionStrategy.DIVERSITY:
                # Score based on diversity
                selected_features = []
                for i in range(n):
                    if i in self.selection_history[-1000:]:  # Recent selections
                        try:
                            selected_features.append(self._get_features(i))
                        except Exception as e:
                            logger.warning(f"Feature extraction failed for index {i}: {e}")
                            continue
                        
                for i in range(n):
                    if selected_features:
                        try:
                            features = self._get_features(i)
                            min_dist = min(torch.norm(features - sf).item() for sf in selected_features)
                            scores[i] = min_dist
                        except Exception as e:
                            logger.warning(f"Distance computation failed for index {i}: {e}")
                            scores[i] = 1.0
                    else:
                        scores[i] = 1.0
                        
            elif strategy == SelectionStrategy.UNCERTAINTY:
                # Score based on model uncertainty
                try:
                    self.model.eval()
                    with torch.no_grad():
                        for i in range(min(n, 1000)):  # Sample for efficiency
                            try:
                                data, _ = self.train_dataset[i]
                                data = data.unsqueeze(0).to(self.device)
                                outputs = self.model(data)
                                probs = F.softmax(outputs, dim=1)
                                entropy = -torch.sum(probs * torch.log(probs + 1e-8))
                                scores[i] = entropy.item()
                            except Exception as e:
                                logger.warning(f"Uncertainty computation failed for index {i}: {e}")
                                scores[i] = 0.0
                    self.model.train()
                except Exception as e:
                    logger.warning(f"Uncertainty strategy failed: {e}")
                    # Fallback to random scores
                    scores = np.random.random(n)
                    
            elif strategy == SelectionStrategy.GRADIENT_VARIANCE:
                # Score based on gradient variance (simplified)
                for i in range(min(n, 1000)):  # Sample for efficiency
                    try:
                        grad = self._compute_single_gradient(i)
                        if grad:
                            scores[i] = sum(torch.norm(g).item() for g in grad.values())
                        else:
                            scores[i] = 0.0
                    except Exception as e:
                        logger.warning(f"Gradient variance computation failed for index {i}: {e}")
                        scores[i] = 0.0
                        
            elif strategy == SelectionStrategy.INFLUENCE_SCORE:
                # Score based on influence (simplified - use gradient magnitude)
                for i in range(min(n, 1000)):  # Sample for efficiency
                    try:
                        grad = self._compute_single_gradient(i)
                        if grad:
                            scores[i] = sum(torch.norm(g).item() for g in grad.values())
                        else:
                            scores[i] = 0.0
                    except Exception as e:
                        logger.warning(f"Influence score computation failed for index {i}: {e}")
                        scores[i] = 0.0
                        
            elif strategy == SelectionStrategy.GRADIENT_CONFLICT:
                # Score based on gradient conflict (simplified)
                for i in range(min(n, 1000)):  # Sample for efficiency
                    try:
                        grad = self._compute_single_gradient(i)
                        if grad:
                            scores[i] = sum(torch.norm(g).item() for g in grad.values())
                        else:
                            scores[i] = 0.0
                    except Exception as e:
                        logger.warning(f"Gradient conflict computation failed for index {i}: {e}")
                        scores[i] = 0.0
                        
            elif strategy == SelectionStrategy.FORGETTING:
                # Score based on forgetting events (simplified - use uncertainty)
                try:
                    self.model.eval()
                    with torch.no_grad():
                        for i in range(min(n, 1000)):  # Sample for efficiency
                            try:
                                data, _ = self.train_dataset[i]
                                data = data.unsqueeze(0).to(self.device)
                                outputs = self.model(data)
                                probs = F.softmax(outputs, dim=1)
                                entropy = -torch.sum(probs * torch.log(probs + 1e-8))
                                scores[i] = entropy.item()
                            except Exception as e:
                                logger.warning(f"Forgetting score computation failed for index {i}: {e}")
                                scores[i] = 0.0
                    self.model.train()
                except Exception as e:
                    logger.warning(f"Forgetting strategy failed: {e}")
                    # Fallback to random scores
                    scores = np.random.random(n)
                    
            elif strategy == SelectionStrategy.HYBRID:
                # Hybrid strategy - combine multiple approaches
                for i in range(min(n, 1000)):  # Sample for efficiency
                    try:
                        # Combine gradient magnitude and uncertainty
                        grad = self._compute_single_gradient(i)
                        if grad:
                            grad_score = sum(torch.norm(g).item() for g in grad.values())
                        else:
                            grad_score = 0.0
                        
                        # Get uncertainty score
                        data, _ = self.train_dataset[i]
                        data = data.unsqueeze(0).to(self.device)
                        self.model.eval()
                        with torch.no_grad():
                            outputs = self.model(data)
                            probs = F.softmax(outputs, dim=1)
                            uncertainty_score = -torch.sum(probs * torch.log(probs + 1e-8)).item()
                        self.model.train()
                        
                        # Combine scores
                        scores[i] = 0.5 * grad_score + 0.5 * uncertainty_score
                    except Exception as e:
                        logger.warning(f"Hybrid strategy computation failed for index {i}: {e}")
                        scores[i] = 0.0
            
            # Select top-k based on scores
            selected_indices = np.argsort(scores)[-budget:].tolist()
            
            # Update strategy reward (will be computed after training)
            self.strategy_rewards[strategy.value].append(0.0)  # Placeholder
            
            return selected_indices
            
        except Exception as e:
            logger.error(f"Strategy selection failed for {strategy.value}: {e}")
            # Fallback to random selection
            n = len(self.train_dataset)
            selected_indices = np.random.choice(n, min(budget, n), replace=False).tolist()
            return selected_indices
    
    def _select_with_composed_strategy(self, budget: int) -> List[int]:
        """Select using best discovered compositional strategy"""
        if not self.strategy_discovery.best_strategies:
            # Fallback to gradient magnitude if no discovered strategies yet
            return self._select_with_strategy(SelectionStrategy.GRADIENT_MAGNITUDE, budget)
            
        # Use best discovered strategy
        best_strategy, _ = self.strategy_discovery.best_strategies[-1]
        n = len(self.train_dataset)
        scores = []
        
        for i in range(n):
            context = {
                'gradients': self._compute_single_gradient(i),
                'gradient_history': list(self.gradient_history),
                'selected_features': [self._get_features(j) for j in self.selection_history[-100:]],
                'point_features': self._get_features(i)
            }
            score = best_strategy.evaluate(i, context)
            scores.append(score)
            
        # Select top-k
        selected_indices = np.argsort(scores)[-budget:].tolist()
        return selected_indices
    
    def _compute_single_gradient(self, idx: int) -> Dict[str, torch.Tensor]:
        """Compute gradient for single data point"""
        try:
            data, label = self.train_dataset[idx]
            data = data.unsqueeze(0).to(self.device)
            label = torch.tensor([label]).to(self.device)
            
            self.model.zero_grad()
            output = self.model(data)
            loss = F.cross_entropy(output, label)
            loss.backward()
            
            gradients = {}
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    gradients[name] = param.grad.clone()
                    
            return gradients
        except Exception as e:
            logger.warning(f"Gradient computation failed for index {idx}: {e}")
            # Return empty gradients as fallback
            return {}
    
    def _get_features(self, idx: int) -> torch.Tensor:
        """Get feature representation for data point"""
        try:
            data, _ = self.train_dataset[idx]
            return data.flatten()
        except Exception as e:
            logger.warning(f"Feature extraction failed for index {idx}: {e}")
            # Return random features as fallback
            return torch.randn(128)
    
    def _compute_compression_ratio(self) -> float:
        """Compute average compression ratio from GaLore"""
        # Simplified - would track actual compression in production
        return self.galore.rank / 1000.0  # Approximate
    
    def update_strategy_rewards(self, strategy: SelectionStrategy, reward: float):
        """Update reward for a strategy based on performance"""
        self.strategy_rewards[strategy.value].append(reward)
        
    def train_rl_policy(self, 
                       replay_buffer: List[Tuple],
                       optimizer: torch.optim.Optimizer,
                       gamma: float = 0.99):
        """Train RL policy using PPO"""
        if len(replay_buffer) < 32:
            return
            
        # Sample batch
        batch = random.sample(replay_buffer, 32)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.stack(states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.float32)
        
        # Compute returns
        with torch.no_grad():
            _, next_values, _ = self.rl_policy(next_states)
            returns = rewards + gamma * next_values.squeeze() * (1 - dones)
            
        # Compute loss
        strategy_weights, values, phase_predictions = self.rl_policy(states)
        
        # Policy loss (simplified PPO)
        action_probs = strategy_weights.gather(1, actions.unsqueeze(1))
        policy_loss = -torch.mean(torch.log(action_probs + 1e-8) * (returns - values.squeeze()).detach())
        
        # Value loss
        value_loss = F.mse_loss(values.squeeze(), returns)
        
        # Phase prediction loss (if we have ground truth)
        phase_loss = torch.tensor(0.0)  # Placeholder
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss + 0.1 * phase_loss
        
        # Update
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return total_loss.item()


# =============================================================================
# Visualization and Analysis
# =============================================================================

def plot_phase_transitions(selector: RLGuidedGaLoreSelector):
    """Visualize detected phase transitions"""
    phase_summary = selector.phase_detector.get_phase_summary()
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # Phase timeline
    ax = axes[0, 0]
    phases = phase_summary['phase_history'] + [phase_summary['current_phase']]
    
    colors = {'chaotic': 'red', 'plateau': 'yellow', 'steep': 'blue', 'converging': 'green'}
    
    for i, phase in enumerate(phases):
        start = phase.start_epoch
        end = phase.end_epoch if phase.end_epoch else selector.epoch
        ax.barh(0, end - start, left=start, color=colors[phase.loss_landscape], 
                label=phase.loss_landscape if i == 0 else '')
        
    ax.set_xlabel('Epoch')
    ax.set_title('Training Phases')
    ax.legend()
    
    # Gradient norm trajectory
    ax = axes[0, 1]
    grad_norms = list(selector.phase_detector.gradient_norms)
    ax.plot(grad_norms)
    ax.set_xlabel('Step')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Norm Evolution')
    
    # Strategy usage over time
    ax = axes[1, 0]
    # This would show strategy weights evolution
    ax.set_title('Strategy Weights Over Time')
    
    # Performance trajectory
    ax = axes[1, 1]
    ax.plot(selector.performance_history)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Performance')
    ax.set_title('Performance Evolution')
    
    # Phase prediction accuracy
    ax = axes[2, 0]
    # This would show prediction vs actual transitions
    ax.set_title('Phase Prediction Accuracy')
    
    # Compression ratio over time
    ax = axes[2, 1]
    # This would show GaLore compression evolution
    ax.set_title('Compression Ratio Evolution')
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# CIFAR Experiments with Various Variations
# =============================================================================

def run_cifar_experiments(experiment_type: str = "all", 
                          data_dir: str = "/Users/tanmoy/research/data",
                          device: str = "cuda" if torch.cuda.is_available() else "mps",
                          config_args=None):
    """
    Run comprehensive CIFAR experiments with various variations
    
    Args:
        experiment_type: Type of experiment to run
            - "cifar10": Only CIFAR10 experiments
            - "cifar100": Only CIFAR100 experiments  
            - "corruptions": Only corruption experiments
            - "all": All experiments
        data_dir: Directory to store/load CIFAR datasets
        device: Device to run experiments on
    """
    
    logger.info(f"Starting CIFAR experiments on {device}")
    
    if experiment_type in ["cifar10", "all"]:
        run_cifar10_experiments(data_dir, device, config_args)
    
    if experiment_type in ["cifar100", "all"]:
        run_cifar100_experiments(data_dir, device, config_args)
    
    if experiment_type in ["corruptions", "all"]:
        run_corruption_experiments(data_dir, device, config_args)
    
    logger.info("CIFAR experiments completed!")


def run_cifar10_experiments(data_dir: str, device: str, config_args=None):
    """Run experiments on CIFAR10 variations"""
    logger.info("Running CIFAR10 experiments...")
    
    # Get CIFAR10 variations
    cifar10_variations = CIFARVariations.get_cifar10_variations(data_dir)
    
    results = {}
    
    for variation_name, variation_data in cifar10_variations.items():
        logger.info(f"Testing {variation_name}: {variation_data['name']}")
        
        # Create model
        if config_args and config_args.model_type == 'vgg':
            model = CIFARVGG(num_classes=10, depth=config_args.model_depth).to(device)
        else:
            model = CIFARResNet(num_classes=10, depth=config_args.model_depth if config_args else 20).to(device)
        
        # Initialize selector
        selector = RLGuidedGaLoreSelector(
            model=model,
            train_dataset=variation_data['train'],
            val_dataset=variation_data['test'],
            memory_budget_mb=config_args.memory_budget_mb if config_args else 1000,
            rank=config_args.rank if config_args else 256
        )
        
        # Run experiment
        result = run_single_cifar_experiment(
            selector=selector,
            model=model,
            train_dataset=variation_data['train'],
            val_dataset=variation_data['test'],
            dataset_name=variation_name,
            device=device
        )
        
        results[variation_name] = result
        
        # Clean up
        del model, selector
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif device == "mps":
            # MPS doesn't have empty_cache, but we can force garbage collection
            import gc
            gc.collect()
    
    # Save results
    save_experiment_results(results, "cifar10_results.json")
    
    # Plot results
    plot_cifar_results(results, "CIFAR10 Experiments")
    
    return results


def run_cifar100_experiments(data_dir: str, device: str, config_args=None):
    """Run experiments on CIFAR100 variations"""
    logger.info("Running CIFAR100 experiments...")
    
    # Get CIFAR100 variations
    cifar100_variations = CIFARVariations.get_cifar100_variations(data_dir)
    
    results = {}
    
    for variation_name, variation_data in cifar100_variations.items():
        logger.info(f"Testing {variation_name}: {variation_data['name']}")
        
        # Create model
        if config_args and config_args.model_type == 'vgg':
            model = CIFARVGG(num_classes=100, depth=config_args.model_depth).to(device)
        else:
            model = CIFARResNet(num_classes=100, depth=config_args.model_depth if config_args else 32).to(device)
        
        # Initialize selector
        selector = RLGuidedGaLoreSelector(
            model=model,
            train_dataset=variation_data['train'],
            val_dataset=variation_data['test'],
            memory_budget_mb=config_args.memory_budget_mb if config_args else 1000,
            rank=config_args.rank if config_args else 256
        )
        
        # Run experiment
        result = run_single_cifar_experiment(
            selector=selector,
            model=model,
            train_dataset=variation_data['train'],
            val_dataset=variation_data['test'],
            dataset_name=variation_name,
            device=device,
            config_args=config_args
        )
        
        results[variation_name] = result
        
        # Clean up
        del model, selector
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif device == "mps":
            # MPS doesn't have empty_cache, but we can force garbage collection
            import gc
            gc.collect()
    
    # Save results
    save_experiment_results(results, "cifar100_results.json")
    
    # Plot results
    plot_cifar_results(results, "CIFAR100 Experiments")
    
    return results


def run_corruption_experiments(data_dir: str, device: str, config_args=None):
    """Run experiments on corrupted CIFAR datasets"""
    logger.info("Running corruption experiments...")
    
    # Get base datasets
    cifar10_train = CIFAR10(data_dir, train=True, download=True, transform=transforms.ToTensor())
    cifar100_train = CIFAR100(data_dir, train=True, download=True, transform=transforms.ToTensor())
    
    # Test transforms for evaluation
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    cifar10_test = CIFAR10(data_dir, train=False, download=True, transform=test_transform)
    cifar100_test = CIFAR100(data_dir, train=False, download=True, transform=test_transform)
    
    corruption_types = config_args.corruption_types if config_args else CIFARVariations.get_all_corruption_types()
    severities = config_args.corruption_severities if config_args else [1, 3, 5]  # Test different corruption levels
    
    results = {}
    
    # Test CIFAR10 corruptions
    for corruption_type in corruption_types:
        for severity in severities:
            logger.info(f"Testing CIFAR10 {corruption_type} severity {severity}")
            
            # Create corrupted dataset
            corrupted_train = CIFARVariations.create_corrupted_cifar10(
                cifar10_train, corruption_type, severity
            )
            
            # Create model
            if config_args and config_args.model_type == 'vgg':
                model = CIFARVGG(num_classes=10, depth=config_args.model_depth).to(device)
            else:
                model = CIFARResNet(num_classes=10, depth=config_args.model_depth if config_args else 20).to(device)
            
            # Initialize selector
            selector = RLGuidedGaLoreSelector(
                model=model,
                train_dataset=corrupted_train,
                val_dataset=cifar10_test,
                memory_budget_mb=config_args.memory_budget_mb if config_args else 1000,
                rank=config_args.rank if config_args else 256
            )
            
            # Run experiment
            result = run_single_cifar_experiment(
                selector=selector,
                model=model,
                train_dataset=corrupted_train,
                val_dataset=cifar10_test,
                dataset_name=f"cifar10_{corruption_type}_sev{severity}",
                device=device,
                config_args=config_args
            )
            
            results[f"cifar10_{corruption_type}_sev{severity}"] = result
            
            # Clean up
            del model, selector
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif device == "mps":
                # MPS doesn't have empty_cache, but we can force garbage collection
                import gc
                gc.collect()
    
    # Test CIFAR100 corruptions (subset for efficiency)
    corruption_subset = ['gaussian_noise', 'defocus_blur', 'brightness', 'contrast']
    for corruption_type in corruption_subset:
        for severity in severities:
            logger.info(f"Testing CIFAR100 {corruption_type} severity {severity}")
            
            # Create corrupted dataset
            corrupted_train = CIFARVariations.create_corrupted_cifar100(
                cifar100_train, corruption_type, severity
            )
            
            # Create model
            if config_args and config_args.model_type == 'vgg':
                model = CIFARVGG(num_classes=100, depth=config_args.model_depth).to(device)
            else:
                model = CIFARResNet(num_classes=100, depth=config_args.model_depth if config_args else 32).to(device)
            
            # Initialize selector
            selector = RLGuidedGaLoreSelector(
                model=model,
                train_dataset=corrupted_train,
                val_dataset=cifar100_test,
                memory_budget_mb=config_args.memory_budget_mb if config_args else 1000,
                rank=config_args.rank if config_args else 256
            )
            
            # Run experiment
            result = run_single_cifar_experiment(
                selector=selector,
                model=model,
                train_dataset=corrupted_train,
                val_dataset=cifar100_test,
                dataset_name=f"cifar100_{corruption_type}_sev{severity}",
                device=device,
                config_args=config_args
            )
            
            results[f"cifar100_{corruption_type}_sev{severity}"] = result
            
            # Clean up
            del model, selector
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif device == "mps":
                # MPS doesn't have empty_cache, but we can force garbage collection
                import gc
                gc.collect()
    
    # Save results
    save_experiment_results(results, "corruption_results.json")
    
    # Plot results
    plot_corruption_results(results)
    
    return results


def run_single_cifar_experiment(selector: RLGuidedGaLoreSelector,
                               model: nn.Module,
                               train_dataset: Dataset,
                               val_dataset: Dataset,
                               dataset_name: str,
                               device: str,
                               epochs: int = 50,
                               coreset_budget: int = 1000,
                               config_args=None) -> Dict[str, Any]:
    """
    Run a single CIFAR experiment
    
    Args:
        selector: RL-guided selector
        model: Model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        dataset_name: Name of the dataset variation
        device: Device to run on
        epochs: Number of training epochs
        coreset_budget: Size of coreset to select each epoch
    
    Returns:
        Dictionary with experiment results
    """
    
    logger.info(f"Running experiment on {dataset_name} for {epochs} epochs")
    
    # Training setup
    lr = config_args.learning_rate if config_args else 0.001
    weight_decay = config_args.weight_decay if config_args else 1e-4
    batch_size = config_args.batch_size if config_args else 64
    scheduler_step = config_args.scheduler_step_size if config_args else 20
    scheduler_gamma = config_args.scheduler_gamma if config_args else 0.5
    rl_lr = config_args.rl_lr if config_args else 0.0003
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    rl_optimizer = torch.optim.Adam(selector.rl_policy.parameters(), lr=rl_lr)
    
    # History tracking
    train_losses = []
    val_losses = []
    val_accuracies = []
    selection_info_history = []
    phase_transitions = []
    
    # Replay buffer for RL
    replay_buffer = []
    
    # Training loop
    for epoch in range(epochs):
        # Evaluate current performance
        val_loss, val_acc = evaluate_cifar_model(model, val_dataset, device)
        current_performance = val_acc  # Higher is better
        
        # Select coreset
        selected_indices, selection_info = selector.select_coreset(
            coreset_budget, current_performance
        )
        
        # Log selection info
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: Phase={selection_info['current_phase']}, "
                       f"Strategy={selection_info['selected_strategy']}, "
                       f"Val Acc={val_acc:.3f}")
        
        # Train on selected coreset
        coreset = Subset(train_dataset, selected_indices)
        coreset_loader = DataLoader(coreset, batch_size=batch_size, shuffle=True)
        
        model.train()
        train_loss = 0
        for batch_idx, (data, labels) in enumerate(coreset_loader):
            data, labels = data.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Progress bar for large datasets
            if len(coreset_loader) > 100 and batch_idx % 50 == 0:
                logger.info(f"  Batch {batch_idx}/{len(coreset_loader)}")
        
        scheduler.step()
        avg_train_loss = train_loss / len(coreset_loader)
        
        # Evaluate new performance
        new_val_loss, new_val_acc = evaluate_cifar_model(model, val_dataset, device)
        
        # Compute reward for RL
        reward = new_val_acc - val_acc  # Improvement in accuracy
        
        # Update strategy rewards
        # Map strategy value back to enum member
        strategy_value = selection_info['selected_strategy']
        strategy_used = None
        for strategy in SelectionStrategy:
            if strategy.value == strategy_value:
                strategy_used = strategy
                break
        if strategy_used is None:
            logger.warning(f"Could not map strategy value '{strategy_value}' to enum member, using GRADIENT_MAGNITUDE as fallback")
            strategy_used = SelectionStrategy.GRADIENT_MAGNITUDE
        selector.update_strategy_rewards(strategy_used, reward)
        
        # Add to replay buffer
        if epoch > 0:
            state = selector._prepare_rl_state(selection_info['phase_indicators'], current_performance)
            next_state = selector._prepare_rl_state(selection_info['phase_indicators'], new_val_acc)
            
            replay_buffer.append((
                state.to_tensor(),
                list(SelectionStrategy).index(strategy_used),
                reward,
                next_state.to_tensor(),
                False
            ))
            
            # Train RL policy
            if len(replay_buffer) >= 32:
                rl_loss = selector.train_rl_policy(replay_buffer, rl_optimizer)
                
                if epoch % 10 == 0:
                    logger.info(f"  RL Loss: {rl_loss:.4f}")
        
        # Track history
        train_losses.append(avg_train_loss)
        val_losses.append(new_val_loss)
        val_accuracies.append(new_val_acc)
        selection_info_history.append(selection_info)
        
        # Check for phase transition
        if selection_info['phase_transition']:
            phase_transitions.append({
                'epoch': epoch,
                'confidence': selection_info['transition_confidence'],
                'new_phase': selection_info['current_phase']
            })
    
    # Compile results
    results = {
        'dataset_name': dataset_name,
        'final_val_accuracy': val_accuracies[-1],
        'best_val_accuracy': max(val_accuracies),
        'final_val_loss': val_losses[-1],
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'phase_transitions': phase_transitions,
        'num_phase_transitions': len(phase_transitions),
        'selection_strategies_used': [info['selected_strategy'] for info in selection_info_history],
        'compression_ratios': [info['compression_ratio'] for info in selection_info_history],
        'training_epochs': epochs,
        'coreset_budget': coreset_budget
    }
    
    logger.info(f"Experiment completed. Final accuracy: {val_accuracies[-1]:.3f}")
    
    return results


def evaluate_cifar_model(model: nn.Module, dataset: Dataset, device: str) -> Tuple[float, float]:
    """Evaluate CIFAR model on dataset"""
    loader = DataLoader(dataset, batch_size=128)
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = F.cross_entropy(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def save_experiment_results(results: Dict[str, Any], filename: str):
    """Save experiment results to JSON file"""
    import json
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            serializable_results[key] = {}
            for subkey, subvalue in value.items():
                if isinstance(subvalue, np.ndarray):
                    serializable_results[key][subkey] = subvalue.tolist()
                else:
                    serializable_results[key][subkey] = subvalue
        else:
            serializable_results[key] = value
    
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Results saved to {filename}")


def plot_cifar_results(results: Dict[str, Any], title: str):
    """Plot CIFAR experiment results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(title, fontsize=16)
    
    # Accuracy comparison
    ax = axes[0, 0]
    datasets = list(results.keys())
    final_accuracies = [results[d]['final_val_accuracy'] for d in datasets]
    
    bars = ax.bar(range(len(datasets)), final_accuracies)
    ax.set_xlabel('Dataset Variation')
    ax.set_ylabel('Final Validation Accuracy (%)')
    ax.set_title('Final Accuracy Comparison')
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels([d.replace('_', '\n') for d in datasets], rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, acc in zip(bars, final_accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom')
    
    # Training curves for best performing dataset
    ax = axes[0, 1]
    best_dataset = max(results.keys(), key=lambda k: results[k]['final_val_accuracy'])
    best_result = results[best_dataset]
    
    epochs = range(1, len(best_result['val_accuracies']) + 1)
    ax.plot(epochs, best_result['val_accuracies'], 'b-', label='Validation Accuracy')
    ax.plot(epochs, best_result['train_losses'], 'r--', label='Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%) / Loss')
    ax.set_title(f'Training Curves - {best_dataset}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Phase transitions
    ax = axes[1, 0]
    all_transitions = []
    for dataset, result in results.items():
        for transition in result['phase_transitions']:
            all_transitions.append({
                'dataset': dataset,
                'epoch': transition['epoch'],
                'confidence': transition['confidence'],
                'phase': transition['new_phase']
            })
    
    if all_transitions:
        # Group by dataset
        datasets_with_transitions = list(set(t['dataset'] for t in all_transitions))
        transition_counts = [len([t for t in all_transitions if t['dataset'] == d]) 
                           for d in datasets_with_transitions]
        
        bars = ax.bar(range(len(datasets_with_transitions)), transition_counts)
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Number of Phase Transitions')
        ax.set_title('Phase Transitions per Dataset')
        ax.set_xticks(range(len(datasets_with_transitions)))
        ax.set_xticklabels([d.replace('_', '\n') for d in datasets_with_transitions], 
                          rotation=45, ha='right')
    
    # Strategy usage
    ax = axes[1, 1]
    all_strategies = set()
    for result in results.values():
        all_strategies.update(result['selection_strategies_used'])
    
    strategy_counts = defaultdict(int)
    for result in results.values():
        for strategy in result['selection_strategies_used']:
            strategy_counts[strategy] += 1
    
    if strategy_counts:
        strategies = list(strategy_counts.keys())
        counts = list(strategy_counts.values())
        
        bars = ax.bar(range(len(strategies)), counts)
        ax.set_xlabel('Selection Strategy')
        ax.set_ylabel('Usage Count')
        ax.set_title('Strategy Usage Across Experiments')
        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels(strategies, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()


def plot_corruption_results(results: Dict[str, Any]):
    """Plot corruption experiment results"""
    # Filter corruption results
    corruption_results = {k: v for k, v in results.items() if 'sev' in k}
    
    if not corruption_results:
        logger.warning("No corruption results to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Corruption Experiment Results', fontsize=16)
    
    # Extract corruption types and severities
    corruption_types = set()
    severities = set()
    
    for key in corruption_results.keys():
        if 'cifar10_' in key:
            parts = key.replace('cifar10_', '').split('_sev')
            if len(parts) == 2:
                corruption_types.add(parts[0])
                severities.add(int(parts[1]))
        elif 'cifar100_' in key:
            parts = key.replace('cifar100_', '').split('_sev')
            if len(parts) == 2:
                corruption_types.add(parts[0])
                severities.add(int(parts[1]))
    
    corruption_types = sorted(list(corruption_types))
    severities = sorted(list(severities))
    
    # Accuracy vs corruption severity
    ax = axes[0, 0]
    for corruption_type in corruption_types[:5]:  # Plot first 5 for clarity
        accuracies = []
        for severity in severities:
            key = f"cifar10_{corruption_type}_sev{severity}"
            if key in corruption_results:
                accuracies.append(corruption_results[key]['final_val_accuracy'])
            else:
                accuracies.append(0)
        
        ax.plot(severities, accuracies, 'o-', label=corruption_type, alpha=0.8)
    
    ax.set_xlabel('Corruption Severity')
    ax.set_ylabel('Final Validation Accuracy (%)')
    ax.set_title('Accuracy vs Corruption Severity (CIFAR10)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # CIFAR10 vs CIFAR100 comparison
    ax = axes[0, 1]
    cifar10_acc = []
    cifar100_acc = []
    
    for corruption_type in corruption_types[:5]:
        cifar10_avg = np.mean([corruption_results.get(f"cifar10_{corruption_type}_sev{s}", {}).get('final_val_accuracy', 0) 
                              for s in severities])
        cifar100_avg = np.mean([corruption_results.get(f"cifar100_{corruption_type}_sev{s}", {}).get('final_val_accuracy', 0) 
                               for s in severities])
        
        cifar10_acc.append(cifar10_avg)
        cifar100_acc.append(cifar100_avg)
    
    x = np.arange(len(corruption_types[:5]))
    width = 0.35
    
    ax.bar(x - width/2, cifar10_acc, width, label='CIFAR10', alpha=0.8)
    ax.bar(x + width/2, cifar100_acc, width, label='CIFAR100', alpha=0.8)
    
    ax.set_xlabel('Corruption Type')
    ax.set_ylabel('Average Accuracy (%)')
    ax.set_title('CIFAR10 vs CIFAR100 Corruption Robustness')
    ax.set_xticks(x)
    ax.set_xticklabels(corruption_types[:5], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Phase transition analysis
    ax = axes[1, 0]
    transition_counts = defaultdict(int)
    for result in corruption_results.values():
        transition_counts[result['num_phase_transitions']] += 1
    
    if transition_counts:
        counts = list(transition_counts.keys())
        frequencies = list(transition_counts.values())
        
        bars = ax.bar(counts, frequencies)
        ax.set_xlabel('Number of Phase Transitions')
        ax.set_ylabel('Frequency')
        ax.set_title('Phase Transition Distribution')
        ax.grid(True, alpha=0.3)
    
    # Strategy effectiveness
    ax = axes[1, 1]
    strategy_performance = defaultdict(list)
    
    for result in corruption_results.values():
        for i, strategy in enumerate(result['selection_strategies_used']):
            if i < len(result['val_accuracies']):
                strategy_performance[strategy].append(result['val_accuracies'][i])
    
    if strategy_performance:
        strategies = list(strategy_performance.keys())
        avg_performance = [np.mean(strategy_performance[s]) for s in strategies]
        
        bars = ax.bar(range(len(strategies)), avg_performance)
        ax.set_xlabel('Selection Strategy')
        ax.set_ylabel('Average Accuracy (%)')
        ax.set_title('Strategy Performance on Corrupted Data')
        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels(strategies, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run CIFAR experiments with RL-guided selection and GaLore integration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments
  python simple_expts.py --experiment all
  
  # Run only CIFAR10 experiments
  python simple_expts.py --experiment cifar10 --epochs 100
  
  # Run corruption experiments with custom settings
  python simple_expts.py --experiment corruptions --epochs 50 --coreset_budget 2000
  
  # Quick test on CPU
  python simple_expts.py --experiment cifar10 --epochs 10 --device cpu --coreset_budget 500
  
  # Custom data directory and device
  python simple_expts.py --experiment all --data_dir /path/to/data --device cuda
        """
    )
    
    # Experiment type
    parser.add_argument('--experiment', type=str, default='all', 
                       choices=['cifar10', 'cifar100', 'corruptions', 'all'],
                       help='Type of experiment to run (default: all)')
    
    # Dataset and data settings
    parser.add_argument('--data_dir', type=str, default='/Users/tanmoy/research/data',
                       help='Directory for CIFAR datasets (default: ./data)')
    parser.add_argument('--download', action='store_true', default=True,
                       help='Download datasets if not present (default: True)')
    
    # Device settings
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to run experiments on (default: auto)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Training batch size (default: 64)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay (default: 1e-4)')
    
    # Coreset selection parameters
    parser.add_argument('--coreset_budget', type=int, default=1000,
                       help='Size of coreset to select each epoch (default: 1000)')
    parser.add_argument('--memory_budget_mb', type=int, default=1000,
                       help='Memory budget in MB (default: 1000)')
    
    # GaLore parameters
    parser.add_argument('--rank', type=int, default=256,
                       help='GaLore rank for gradient compression (default: 256)')
    parser.add_argument('--update_proj_gap', type=int, default=200,
                       help='GaLore projection update frequency (default: 200)')
    
    # RL parameters
    parser.add_argument('--rl_lr', type=float, default=0.0003,
                       help='RL policy learning rate (default: 0.0003)')
    parser.add_argument('--epsilon', type=float, default=0.1,
                       help='RL exploration epsilon (default: 0.1)')
    
    # Phase detection parameters
    parser.add_argument('--window_size', type=int, default=50,
                       help='Phase detection window size (default: 50)')
    parser.add_argument('--sensitivity', type=float, default=2.0,
                       help='Phase transition sensitivity (default: 2.0)')
    
    # Output and logging
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    parser.add_argument('--save_results', action='store_true', default=True,
                       help='Save results to JSON files (default: True)')
    parser.add_argument('--plot_results', action='store_true', default=True,
                       help='Generate result plots (default: True)')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for results (default: ./results)')
    
    # Quick test mode
    parser.add_argument('--quick', action='store_true',
                       help='Quick test mode (fewer epochs, smaller models)')
    
    # Corruption experiment specific
    parser.add_argument('--corruption_severities', type=int, nargs='+', default=[1, 3, 5],
                       help='Corruption severity levels to test (default: 1 3 5)')
    parser.add_argument('--corruption_types', type=str, nargs='+', 
                       default=['gaussian_noise', 'defocus_blur', 'brightness', 'contrast'],
                       help='Corruption types to test (default: gaussian_noise defocus_blur brightness contrast)')
    
    # Model architecture
    parser.add_argument('--model_type', type=str, default='resnet',
                       choices=['resnet', 'vgg'],
                       help='Model architecture to use (default: resnet)')
    parser.add_argument('--model_depth', type=int, default=20,
                       help='Model depth (default: 20 for ResNet, 16 for VGG)')
    
    args = parser.parse_args()
    
    # Handle quick mode
    if args.quick:
        args.epochs = min(args.epochs, 10)
        args.coreset_budget = min(args.coreset_budget, 500)
        args.memory_budget_mb = min(args.memory_budget_mb, 500)
        args.rank = min(args.rank, 128)
        logger.info("Quick mode enabled - reduced parameters for faster execution")
    
    # Set device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Create output directory
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    elif device == "mps":
        # MPS doesn't have manual seed functions, but we can set the device seed
        pass
    
    # Log configuration
    logger.info("=" * 60)
    logger.info("CIFAR EXPERIMENT CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Experiment type: {args.experiment}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Training epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Coreset budget: {args.coreset_budget}")
    logger.info(f"Memory budget: {args.memory_budget_mb} MB")
    logger.info(f"GaLore rank: {args.rank}")
    logger.info(f"Model type: {args.model_type} (depth: {args.model_depth})")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 60)
    
    # Run experiments
    try:
        results = run_cifar_experiments(
            experiment_type=args.experiment,
            data_dir=args.data_dir,
            device=device,
            config_args=args
        )
        
        logger.info("All experiments completed successfully!")
        
        # Save final results summary
        if args.save_results:
            import json
            summary_file = os.path.join(args.output_dir, "experiment_summary.json")
            
            # Create summary
            summary = {
                "experiment_type": args.experiment,
                "device": device,
                "parameters": {
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "coreset_budget": args.coreset_budget,
                    "memory_budget_mb": args.memory_budget_mb,
                    "rank": args.rank,
                    "model_type": args.model_type,
                    "model_depth": args.model_depth
                },
                "results_summary": {}
            }
            
            # Add result summaries
            for exp_name, result in results.items():
                if isinstance(result, dict) and 'final_val_accuracy' in result:
                    summary["results_summary"][exp_name] = {
                        "final_accuracy": result['final_val_accuracy'],
                        "best_accuracy": result['best_val_accuracy'],
                        "phase_transitions": result['num_phase_transitions']
                    }
            
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Experiment summary saved to {summary_file}")
        
    except Exception as e:
        logger.error(f"Experiment failed with error: {e}")
        import traceback
        traceback.print_exc()