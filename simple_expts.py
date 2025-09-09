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
from torch.utils.tensorboard import SummaryWriter
import functools
import contextlib
from threading import Lock
from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms.functional as TF

# Import hypernetwork components
from hypernetwork import (
    MultiScoringHypernetwork,
    SubmodularMultiScoringSelector,
    TrainingState as HypernetTrainingState,
    create_scoring_functions,
    GradientMagnitudeScoring,
    DiversityScoring,
    UncertaintyScoring,
    BoundaryScoring,
    InfluenceScoring,
    ForgetScoring
)

# Import LLM hypernetwork components
try:
    from llm_hypernetworks import (
        LLMMultiScoringHypernetwork,
        LLMCoresetSelector,
        LLMTrainingState,
        create_llm_scoring_functions,
        create_llm_hypernetwork
    )
    LLM_HYPERNETWORK_AVAILABLE = True
except ImportError:
    logger.warning("LLM hypernetwork module not available")
    LLM_HYPERNETWORK_AVAILABLE = False

# Import GP-based strategy optimizer
try:
    from galore_gp_optimizer import (
        GALOREStrategyOptimizer,
        EnhancedGALOREFramework
    )
    GP_OPTIMIZER_AVAILABLE = True
except ImportError:
    logger.warning("GP optimizer module not available (install scikit-optimize)")
    GP_OPTIMIZER_AVAILABLE = False

# Import MDP-based dataset selector with Bloom filtering
try:
    from bloom_mdp_selector import (
        DatasetSelectorMDP,
        GALOREMDPIntegration,
        SelectionStrategy as MDPStrategy
    )
    MDP_SELECTOR_AVAILABLE = True
except ImportError:
    logger.warning("MDP selector module not available")
    MDP_SELECTOR_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# PROFILING AND TIMING INFRASTRUCTURE
# =============================================================================

class PerformanceProfiler:
    """Comprehensive performance profiler with timing, memory, and TensorBoard integration."""
    
    def __init__(self, log_dir: str = "./logs", experiment_name: str = "experiment"):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.timings = defaultdict(list)
        self.memory_usage = defaultdict(list)
        self.counters = defaultdict(int)
        self.lock = Lock()
        
        # Create TensorBoard writer
        self.writer = SummaryWriter(log_dir=f"{log_dir}/{experiment_name}")
        
        # Timing context managers
        self.active_timers = {}
        
        logger.info(f"Performance profiler initialized. Logs: {log_dir}/{experiment_name}")
    
    def start_timer(self, name: str):
        """Start timing a named operation."""
        with self.lock:
            self.active_timers[name] = time.perf_counter()
    
    def end_timer(self, name: str) -> float:
        """End timing and record the duration."""
        with self.lock:
            if name not in self.active_timers:
                logger.warning(f"Timer '{name}' was not started")
                return 0.0
            
            duration = time.perf_counter() - self.active_timers[name]
            self.timings[name].append(duration)
            del self.active_timers[name]
            return duration
    
    @contextlib.contextmanager
    def timer(self, name: str):
        """Context manager for timing operations."""
        self.start_timer(name)
        try:
            yield
        finally:
            duration = self.end_timer(name)
            logger.debug(f"{name}: {duration:.4f}s")
    
    def record_memory(self, name: str, device: str = "mps"):
        """Record current memory usage."""
        if device == "mps" and torch.backends.mps.is_available():
            # MPS memory tracking
            allocated = torch.mps.current_allocated_memory() / 1024**2  # MB
            reserved = torch.mps.driver_allocated_memory() / 1024**2  # MB
        elif device == "cuda" and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2  # MB
            reserved = torch.cuda.memory_reserved() / 1024**2  # MB
        else:
            # CPU - use process memory
            import psutil
            process = psutil.Process()
            allocated = process.memory_info().rss / 1024**2  # MB
            reserved = allocated
        
        with self.lock:
            self.memory_usage[name].append({
                'allocated': allocated,
                'reserved': reserved,
                'timestamp': time.time()
            })
    
    def increment_counter(self, name: str, value: int = 1):
        """Increment a named counter."""
        with self.lock:
            self.counters[name] += value
    
    def log_metrics(self, step: int, metrics: Dict[str, float]):
        """Log metrics to TensorBoard."""
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, step)
    
    def log_timing_summary(self, step: int):
        """Log timing summary to TensorBoard."""
        with self.lock:
            for name, times in self.timings.items():
                if times:
                    avg_time = np.mean(times)
                    total_time = np.sum(times)
                    self.writer.add_scalar(f"timing/{name}_avg", avg_time, step)
                    self.writer.add_scalar(f"timing/{name}_total", total_time, step)
    
    def log_memory_summary(self, step: int):
        """Log memory usage summary to TensorBoard."""
        with self.lock:
            for name, memory_records in self.memory_usage.items():
                if memory_records:
                    latest = memory_records[-1]
                    self.writer.add_scalar(f"memory/{name}_allocated", latest['allocated'], step)
                    self.writer.add_scalar(f"memory/{name}_reserved", latest['reserved'], step)
    
    def get_timing_stats(self) -> Dict[str, Dict[str, float]]:
        """Get comprehensive timing statistics."""
        with self.lock:
            stats = {}
            for name, times in self.timings.items():
                if times:
                    stats[name] = {
                        'count': len(times),
                        'total': np.sum(times),
                        'mean': np.mean(times),
                        'std': np.std(times),
                        'min': np.min(times),
                        'max': np.max(times),
                        'median': np.median(times)
                    }
            return stats
    
    def print_summary(self):
        """Print a comprehensive performance summary."""
        logger.info("=" * 80)
        logger.info("PERFORMANCE PROFILING SUMMARY")
        logger.info("=" * 80)
        
        # Timing statistics
        timing_stats = self.get_timing_stats()
        if timing_stats:
            logger.info("\nTIMING STATISTICS:")
            logger.info("-" * 50)
            for name, stats in sorted(timing_stats.items(), key=lambda x: x[1]['total'], reverse=True):
                logger.info(f"{name:30s}: {stats['total']:8.3f}s total, "
                          f"{stats['mean']:6.3f}s avg, {stats['count']:4d} calls")
        
        # Memory usage
        with self.lock:
            if self.memory_usage:
                logger.info("\nMEMORY USAGE:")
                logger.info("-" * 50)
                for name, records in self.memory_usage.items():
                    if records:
                        latest = records[-1]
                        logger.info(f"{name:30s}: {latest['allocated']:8.1f}MB allocated, "
                                  f"{latest['reserved']:8.1f}MB reserved")
        
        # Counters
        with self.lock:
            if self.counters:
                logger.info("\nCOUNTERS:")
                logger.info("-" * 50)
                for name, count in sorted(self.counters.items()):
                    logger.info(f"{name:30s}: {count:8d}")
        
        logger.info("=" * 80)
    
    def close(self):
        """Close the profiler and TensorBoard writer."""
        self.print_summary()
        print_performance_analysis(self)
        self.writer.close()
        logger.info("Performance profiler closed.")

# Global profiler instance
_profiler = None

def get_profiler() -> PerformanceProfiler:
    """Get the global profiler instance."""
    global _profiler
    if _profiler is None:
        _profiler = PerformanceProfiler()
    return _profiler

def init_profiler(log_dir: str = "./logs", experiment_name: str = "experiment") -> PerformanceProfiler:
    """Initialize the global profiler."""
    global _profiler
    _profiler = PerformanceProfiler(log_dir, experiment_name)
    return _profiler

def timing_decorator(name: str = None):
    """Decorator to automatically time function execution."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            timer_name = name or f"{func.__module__}.{func.__name__}"
            profiler = get_profiler()
            with profiler.timer(timer_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator

def memory_tracking_decorator(name: str = None, device: str = "mps"):
    """Decorator to track memory usage before and after function execution."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            timer_name = name or f"{func.__module__}.{func.__name__}"
            profiler = get_profiler()
            
            profiler.record_memory(f"{timer_name}_before", device)
            result = func(*args, **kwargs)
            profiler.record_memory(f"{timer_name}_after", device)
            
            return result
        return wrapper
    return decorator

# =============================================================================
# PERFORMANCE ANALYSIS AND REPORTING
# =============================================================================

def analyze_performance_bottlenecks(profiler: PerformanceProfiler) -> Dict[str, Any]:
    """Analyze performance bottlenecks and provide recommendations."""
    timing_stats = profiler.get_timing_stats()
    
    if not timing_stats:
        return {"message": "No timing data available"}
    
    # Find bottlenecks
    total_time = sum(stats['total'] for stats in timing_stats.values())
    bottlenecks = []
    
    for name, stats in timing_stats.items():
        percentage = (stats['total'] / total_time) * 100
        if percentage > 10:  # More than 10% of total time
            bottlenecks.append({
                'operation': name,
                'total_time': stats['total'],
                'percentage': percentage,
                'avg_time': stats['mean'],
                'call_count': stats['count']
            })
    
    # Sort by total time
    bottlenecks.sort(key=lambda x: x['total_time'], reverse=True)
    
    # Generate recommendations
    recommendations = []
    for bottleneck in bottlenecks:
        if 'coreset_selection' in bottleneck['operation']:
            recommendations.append("Consider optimizing coreset selection algorithm or reducing selection frequency")
        elif 'evaluation' in bottleneck['operation']:
            recommendations.append("Consider reducing evaluation frequency or using smaller validation sets")
        elif 'training' in bottleneck['operation']:
            recommendations.append("Consider optimizing batch size or model architecture")
        elif 'svd' in bottleneck['operation'].lower():
            recommendations.append("SVD operations are expensive - consider reducing rank or using approximations")
    
    return {
        'total_time': total_time,
        'bottlenecks': bottlenecks,
        'recommendations': recommendations,
        'summary': f"Found {len(bottlenecks)} major bottlenecks consuming {sum(b['percentage'] for b in bottlenecks):.1f}% of total time"
    }

def print_performance_analysis(profiler: PerformanceProfiler):
    """Print detailed performance analysis."""
    analysis = analyze_performance_bottlenecks(profiler)
    
    logger.info("\n" + "=" * 80)
    logger.info("PERFORMANCE ANALYSIS")
    logger.info("=" * 80)
    
    if 'message' in analysis:
        logger.info(analysis['message'])
        return
    
    logger.info(f"Total execution time: {analysis['total_time']:.2f} seconds")
    logger.info(f"Analysis: {analysis['summary']}")
    
    if analysis['bottlenecks']:
        logger.info("\nMAJOR BOTTLENECKS:")
        logger.info("-" * 50)
        for i, bottleneck in enumerate(analysis['bottlenecks'], 1):
            logger.info(f"{i}. {bottleneck['operation']}")
            logger.info(f"   Time: {bottleneck['total_time']:.3f}s ({bottleneck['percentage']:.1f}%)")
            logger.info(f"   Avg per call: {bottleneck['avg_time']:.3f}s ({bottleneck['call_count']} calls)")
    
    if analysis['recommendations']:
        logger.info("\nRECOMMENDATIONS:")
        logger.info("-" * 50)
        for i, rec in enumerate(analysis['recommendations'], 1):
            logger.info(f"{i}. {rec}")
    
    logger.info("=" * 80)

# =============================================================================
# INTERMEDIATE RESULTS MANAGER
# =============================================================================

class IntermediateResultsManager:
    """Manages saving and loading of intermediate results and checkpoints."""
    
    def __init__(self, output_dir: str, experiment_name: str, config_args=None):
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.config_args = config_args
        
        # Create directories
        self.checkpoint_dir = os.path.join(output_dir, "checkpoints", experiment_name)
        self.intermediate_dir = os.path.join(output_dir, "intermediate", experiment_name)
        self.strategy_dir = os.path.join(output_dir, "strategy_data", experiment_name)
        self.timing_dir = os.path.join(output_dir, "timing_data", experiment_name)
        self.coreset_dir = os.path.join(output_dir, "coreset_data", experiment_name)
        
        for dir_path in [self.checkpoint_dir, self.intermediate_dir, self.strategy_dir, 
                        self.timing_dir, self.coreset_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Data storage
        self.epoch_data = []
        self.strategy_history = []
        self.timing_history = []
        self.coreset_history = []
        self.performance_history = []
        
        logger.info(f"Intermediate results manager initialized: {output_dir}/{experiment_name}")
    
    def save_epoch_data(self, epoch: int, data: Dict[str, Any]):
        """Save epoch-level data."""
        epoch_data = {
            'epoch': epoch,
            'timestamp': time.time(),
            'data': data
        }
        self.epoch_data.append(epoch_data)
        
        # Save individual epoch file
        epoch_file = os.path.join(self.intermediate_dir, f"epoch_{epoch:04d}.json")
        with open(epoch_file, 'w') as f:
            json.dump(epoch_data, f, indent=2, default=str)
    
    def save_strategy_data(self, epoch: int, strategy_info: Dict[str, Any]):
        """Save strategy selection data."""
        strategy_data = {
            'epoch': epoch,
            'timestamp': time.time(),
            'strategy_info': strategy_info
        }
        self.strategy_history.append(strategy_data)
        
        # Save individual strategy file
        strategy_file = os.path.join(self.strategy_dir, f"strategy_{epoch:04d}.json")
        with open(strategy_file, 'w') as f:
            json.dump(strategy_data, f, indent=2, default=str)
    
    def save_timing_data(self, epoch: int, timing_data: Dict[str, Any]):
        """Save timing and performance data."""
        timing_entry = {
            'epoch': epoch,
            'timestamp': time.time(),
            'timing_data': timing_data
        }
        self.timing_history.append(timing_entry)
        
        # Save individual timing file
        timing_file = os.path.join(self.timing_dir, f"timing_{epoch:04d}.json")
        with open(timing_file, 'w') as f:
            json.dump(timing_entry, f, indent=2, default=str)
    
    def save_coreset_data(self, epoch: int, coreset_info: Dict[str, Any]):
        """Save coreset selection data."""
        coreset_data = {
            'epoch': epoch,
            'timestamp': time.time(),
            'coreset_info': coreset_info
        }
        self.coreset_history.append(coreset_data)
        
        # Save individual coreset file
        coreset_file = os.path.join(self.coreset_dir, f"coreset_{epoch:04d}.json")
        with open(coreset_file, 'w') as f:
            json.dump(coreset_data, f, indent=2, default=str)
    
    def save_checkpoint(self, epoch: int, model: nn.Module, optimizer, scheduler, 
                       selector: 'RLGuidedGaLoreSelector', additional_data: Dict[str, Any] = None):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'timestamp': time.time(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'selector_state': {
                'strategy_rewards': selector.strategy_rewards,
                'phase_detector_state': selector.phase_detector.get_state() if hasattr(selector.phase_detector, 'get_state') else None,
                'rl_agent_state': selector.rl_agent.get_state() if hasattr(selector.rl_agent, 'get_state') else None,
                'step': selector.step,
                'epoch': selector.epoch
            },
            'additional_data': additional_data or {}
        }
        
        checkpoint_file = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch:04d}.pt")
        torch.save(checkpoint, checkpoint_file)
        
        # Also save latest checkpoint
        latest_file = os.path.join(self.checkpoint_dir, "latest_checkpoint.pt")
        torch.save(checkpoint, latest_file)
        
        logger.info(f"Checkpoint saved: {checkpoint_file}")
    
    def load_checkpoint(self, checkpoint_path: str, model: nn.Module, optimizer, scheduler, 
                       selector: 'RLGuidedGaLoreSelector'):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore selector state
        if 'selector_state' in checkpoint:
            selector_state = checkpoint['selector_state']
            selector.strategy_rewards = selector_state.get('strategy_rewards', {})
            selector.step = selector_state.get('step', 0)
            selector.epoch = selector_state.get('epoch', 0)
            
            if hasattr(selector.phase_detector, 'set_state') and selector_state.get('phase_detector_state'):
                selector.phase_detector.set_state(selector_state['phase_detector_state'])
            
            if hasattr(selector.rl_agent, 'set_state') and selector_state.get('rl_agent_state'):
                selector.rl_agent.set_state(selector_state['rl_agent_state'])
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint['epoch'], checkpoint.get('additional_data', {})
    
    def save_final_summary(self, final_results: Dict[str, Any]):
        """Save final comprehensive summary."""
        summary = {
            'experiment_name': self.experiment_name,
            'config': vars(self.config_args) if self.config_args else {},
            'final_results': final_results,
            'epoch_data': self.epoch_data,
            'strategy_history': self.strategy_history,
            'timing_history': self.timing_history,
            'coreset_history': self.coreset_history,
            'performance_history': self.performance_history,
            'summary_timestamp': time.time()
        }
        
        summary_file = os.path.join(self.output_dir, f"{self.experiment_name}_complete_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Complete summary saved: {summary_file}")
        return summary_file
    
    def save_analysis_data(self, profiler: PerformanceProfiler):
        """Save detailed analysis data from profiler."""
        analysis_data = {
            'timing_stats': profiler.get_timing_stats(),
            'memory_usage': dict(profiler.memory_usage),
            'counters': dict(profiler.counters),
            'performance_analysis': analyze_performance_bottlenecks(profiler),
            'timestamp': time.time()
        }
        
        analysis_file = os.path.join(self.output_dir, f"{self.experiment_name}_analysis.json")
        with open(analysis_file, 'w') as f:
            json.dump(analysis_data, f, indent=2, default=str)
        
        logger.info(f"Analysis data saved: {analysis_file}")
        return analysis_file

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
    
    def extract_features(self, x):
        """Extract features before the final classification layer"""
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        return x  # Return features before FC layer


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
    
    def analyze_phase(self, gradient_norm: float, loss_curvature: float, 
                     gradient_alignment: float) -> Dict[str, Any]:
        """
        Analyze current training phase for GP optimizer integration
        
        Args:
            gradient_norm: Current gradient norm
            loss_curvature: Current loss curvature estimate
            gradient_alignment: Current gradient alignment score
            
        Returns:
            Dict with phase information
        """
        # Determine current phase based on metrics
        if len(self.gradient_norms) < 10:
            phase = 'early'
        elif len(self.loss_values) > 0:
            recent_loss_variance = np.var(list(self.loss_values)[-10:]) if len(self.loss_values) >= 10 else 1.0
            recent_grad_norm = np.mean(list(self.gradient_norms)[-10:]) if len(self.gradient_norms) >= 10 else gradient_norm
            
            if recent_loss_variance > 0.1 or recent_grad_norm > 1.0:
                phase = 'early'
            elif recent_loss_variance < 0.01 and recent_grad_norm < 0.1:
                phase = 'late'
            elif abs(gradient_alignment) > 0.8:
                phase = 'transition'
            else:
                phase = 'middle'
        else:
            phase = 'middle'
            
        return {
            'phase': phase,
            'gradient_norm': gradient_norm,
            'loss_curvature': loss_curvature,
            'gradient_alignment': gradient_alignment,
            'current_landscape': getattr(self.current_phase, 'loss_landscape', 'unknown'),
            'dominant_strategy': getattr(self.current_phase, 'dominant_strategy', 'hybrid')
        }
    
    def detected_transition(self) -> bool:
        """Check if a transition was recently detected"""
        if len(self.transition_scores) < 2:
            return False
        return self.transition_scores[-1] > self.sensitivity


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
    HYPERNETWORK = "hypernetwork"  # New hypernetwork-based multi-scoring strategy
    LLM_HYPERNETWORK = "llm_hypernetwork"  # LLM-specific hypernetwork strategy


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
    
    def __init__(self, state_dim: int = 22, hidden_dim: int = 256, num_strategies: int = 9):
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
                    # For MPS device, move to CPU for SVD then back
                    if grad_2d.device.type == 'mps':
                        grad_2d_cpu = grad_2d.cpu()
                        U, _, V = torch.svd_lowrank(grad_2d_cpu, q=self.rank)
                        self.projectors[name] = (U.detach().to(grad.device), V.detach().to(grad.device))
                    else:
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
                        # For MPS device, move to CPU for QR decomposition then back
                        if U.device.type == 'mps':
                            U_cpu, _ = torch.linalg.qr(U.cpu())
                            V_cpu, _ = torch.linalg.qr(V.cpu())
                            U = U_cpu.to(U.device)
                            V = V_cpu.to(V.device)
                        else:
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
                 rank: int = 256,
                 use_hypernetwork: bool = True,
                 use_mdp_selector: bool = False):
        
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.use_hypernetwork = use_hypernetwork
        self.use_mdp_selector = use_mdp_selector
        
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
        
        # Initialize hypernetwork components if enabled
        if use_hypernetwork:
            # Feature extractor for diversity scoring
            def feature_extractor(x):
                with torch.no_grad():
                    if hasattr(self.model, 'get_features'):
                        return self.model.get_features(x)
                    else:
                        # Use model output as features
                        return self.model(x)
            
            # Create scoring functions
            self.scoring_functions = [
                GradientMagnitudeScoring(self.model, self.device),
                DiversityScoring(feature_extractor),
                UncertaintyScoring(self.model, self.device),
                BoundaryScoring(self.model, self.device),
                InfluenceScoring(self.model, self.device),
                ForgetScoring()
            ]
            
            # Create hypernetwork with optimized settings for CIFAR10
            self.hypernetwork = MultiScoringHypernetwork(
                scoring_functions=self.scoring_functions,
                state_dim=19,  # Matches HypernetTrainingState
                hidden_dim=64,  # Reduced for faster training
                attention_heads=2  # Reduced for efficiency
            ).to(self.device)
            
            # Create selector
            self.hypernet_selector = SubmodularMultiScoringSelector(
                hypernetwork=self.hypernetwork,
                scoring_functions=self.scoring_functions,
                lazy_evaluation=True,  # Enable lazy evaluation for speed
                cache_size=5000  # Smaller cache for CIFAR10
            )
            
            # Hypernetwork optimizer
            self.hypernet_optimizer = torch.optim.Adam(
                self.hypernetwork.parameters(), 
                lr=0.001  # Higher LR for faster convergence
            )
            
            logger.info("Initialized hypernetwork-based multi-scoring selection")
        
        # Initialize MDP selector if enabled
        self.mdp_integration = None
        if use_mdp_selector and MDP_SELECTOR_AVAILABLE:
            # Determine feature dimension based on model
            if hasattr(model, 'fc'):
                feature_dim = model.fc.in_features
            elif hasattr(model, 'classifier'):
                if isinstance(model.classifier, nn.Sequential):
                    feature_dim = model.classifier[0].in_features
                else:
                    feature_dim = model.classifier.in_features
            else:
                feature_dim = 512  # Default
            
            self.mdp_integration = GALOREMDPIntegration(
                dataset_size=len(train_dataset),
                feature_dim=feature_dim
            )
            logger.info(f"Initialized MDP selector with feature_dim={feature_dim}, dataset_size={len(train_dataset)}")
        
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
        
        logger.info(f"Initialized RL-Guided GaLore Selector with rank={rank}, hypernetwork={use_hypernetwork}")
        
    @timing_decorator("select_coreset")
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
        
        # Use MDP selector if enabled
        if self.use_mdp_selector and self.mdp_integration is not None:
            # Extract features for MDP selection
            performance_metrics = {
                'accuracy': current_performance,
                'loss': 2.0 - current_performance  # Rough inverse relationship
            }
            
            # Select using MDP
            selected_indices, mdp_strategy = self.mdp_integration.select_coreset(
                self.model, self.train_dataset, budget, performance_metrics
            )
            
            # Map MDP strategy to our SelectionStrategy enum
            mdp_strategy_map = {
                'π_explore': SelectionStrategy.DIVERSITY,
                'π_exploit': SelectionStrategy.GRADIENT_MAGNITUDE,
                'π_refresh': SelectionStrategy.RANDOM,
                'π_balance': SelectionStrategy.HYBRID,
                'π_focus': SelectionStrategy.UNCERTAINTY
            }
            selected_strategy = mdp_strategy_map.get(mdp_strategy.value, SelectionStrategy.HYBRID)
            
            logger.info(f"MDP selected strategy: {mdp_strategy.value} -> {selected_strategy.value}")
            
        else:
            # Select strategy based on weights
            # For CIFAR10, prioritize hypernetwork strategy for faster convergence
            if self.use_hypernetwork and hasattr(self, 'hypernet_selector'):
                # Use hypernetwork for first 30 epochs or every 5 epochs for exploration
                if self.epoch < 30 or self.epoch % 5 == 0:
                    selected_strategy = SelectionStrategy.HYPERNETWORK
                    logger.info(f"Using hypernetwork strategy for epoch {self.epoch}")
                else:
                    strategy_idx, _ = self.rl_policy.select_action(state_tensor, epsilon=0.1 if self.epoch < 100 else 0.05)
                    selected_strategy = list(SelectionStrategy)[strategy_idx]
            else:
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
                        
            elif strategy == SelectionStrategy.HYPERNETWORK:
                # Use hypernetwork-based multi-scoring selection
                if self.use_hypernetwork and hasattr(self, 'hypernet_selector'):
                    # Create training state for hypernetwork
                    try:
                        # Get current performance metrics
                        current_loss = self.performance_history[-1] if self.performance_history else 1.0
                        current_acc = max(0.0, 1.0 - current_loss)  # Simple conversion
                        
                        # Compute gradient norm
                        grad_norm = 0.0
                        sample_grads = self._compute_sample_gradients(min(10, len(self.train_dataset)))
                        if sample_grads:
                            # Handle both tensor and numpy array cases
                            grad_norms = []
                            for g in sample_grads.values() if isinstance(sample_grads, dict) else sample_grads:
                                if isinstance(g, torch.Tensor):
                                    grad_norms.append(g.flatten().norm().item())
                                elif hasattr(g, 'flatten'):
                                    grad_norms.append(np.linalg.norm(g.flatten()))
                            if grad_norms:
                                grad_norm = np.mean(grad_norms)
                        
                        # Get class distribution (simplified for CIFAR10)
                        class_dist = np.ones(10) / 10  # Uniform for simplicity
                        
                        # Create hypernetwork training state
                        hypernet_state = HypernetTrainingState(
                            epoch=self.epoch,
                            loss=current_loss,
                            accuracy=current_acc,
                            gradient_norm=grad_norm,
                            learning_rate=0.001,  # Default LR
                            data_seen_ratio=self.epoch / 100.0,  # Assume 100 epochs max
                            class_distribution=class_dist,
                            performance_history=self.performance_history[-10:] if len(self.performance_history) >= 10 else [0.0] * 10,
                            selection_diversity=0.5  # Placeholder
                        )
                        
                        # Use hypernetwork selector
                        selected_indices, selection_info = self.hypernet_selector.select_coreset_greedy(
                            dataset=self.train_dataset,
                            budget=budget,
                            training_state=hypernet_state,
                            context={'sample_losses': {}},  # Can add sample losses if available
                            verbose=False  # Less verbose for integration
                        )
                        
                        logger.info(f"Hypernetwork selection completed with weights: {selection_info['weights']}")
                        return selected_indices
                        
                    except Exception as e:
                        logger.warning(f"Hypernetwork selection failed: {e}, falling back to gradient magnitude")
                        return self._select_with_strategy(SelectionStrategy.GRADIENT_MAGNITUDE, budget)
                else:
                    logger.warning("Hypernetwork not initialized, falling back to gradient magnitude")
                    return self._select_with_strategy(SelectionStrategy.GRADIENT_MAGNITUDE, budget)
            
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

@timing_decorator("run_cifar_experiments")
def run_cifar_experiments(experiment_type: str = "all", 
                          data_dir: str = "/Users/mukher74/research/data",
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
    
    # Initialize profiler for the entire experiment suite
    experiment_name = f"cifar_experiments_{experiment_type}_{device}"
    profiler = init_profiler(log_dir="./logs", experiment_name=experiment_name)
    profiler.record_memory("experiment_suite_start", device)
    
    logger.info("=" * 80)
    logger.info("STARTING CIFAR EXPERIMENT SUITE")
    logger.info("=" * 80)
    logger.info(f"Experiment type: {experiment_type}")
    logger.info(f"Device: {device}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"TensorBoard logs: ./logs/{experiment_name}")
    logger.info("=" * 80)
    
    try:
        if experiment_type in ["cifar10", "all"]:
            with profiler.timer("cifar10_experiments"):
                run_cifar10_experiments(data_dir, device, config_args)
        
        if experiment_type in ["cifar100", "all"]:
            with profiler.timer("cifar100_experiments"):
                run_cifar100_experiments(data_dir, device, config_args)
        
        if experiment_type in ["corruptions", "all"]:
            with profiler.timer("corruption_experiments"):
                run_corruption_experiments(data_dir, device, config_args)
        
        logger.info("CIFAR experiments completed!")
        
    finally:
        # Always close profiler and print summary
        profiler.record_memory("experiment_suite_end", device)
        profiler.close()


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
            rank=config_args.rank if config_args else 256,
            use_hypernetwork=config_args.use_hypernetwork if config_args and hasattr(config_args, 'use_hypernetwork') else True,
            use_mdp_selector=config_args.use_mdp_selector if config_args and hasattr(config_args, 'use_mdp_selector') else False
        )
        
        # Run experiment
        result = run_single_cifar_experiment(
            selector=selector,
            model=model,
            train_dataset=variation_data['train'],
            val_dataset=variation_data['test'],
            dataset_name=variation_name,
            device=device,
            epochs=config_args.epochs if config_args else 50,
            coreset_budget=config_args.coreset_budget if config_args else 1000,
            config_args=config_args,
            use_gp_optimizer=config_args.use_gp_optimizer if config_args else False
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
            rank=config_args.rank if config_args else 256,
            use_hypernetwork=config_args.use_hypernetwork if config_args and hasattr(config_args, 'use_hypernetwork') else True,
            use_mdp_selector=config_args.use_mdp_selector if config_args and hasattr(config_args, 'use_mdp_selector') else False
        )
        
        # Run experiment
        result = run_single_cifar_experiment(
            selector=selector,
            model=model,
            train_dataset=variation_data['train'],
            val_dataset=variation_data['test'],
            dataset_name=variation_name,
            device=device,
            epochs=config_args.epochs if config_args else 50,
            coreset_budget=config_args.coreset_budget if config_args else 1000,
            config_args=config_args,
            use_gp_optimizer=config_args.use_gp_optimizer if config_args else False
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
                rank=config_args.rank if config_args else 256,
                use_hypernetwork=config_args.use_hypernetwork if config_args and hasattr(config_args, 'use_hypernetwork') else True,
                use_mdp_selector=config_args.use_mdp_selector if config_args and hasattr(config_args, 'use_mdp_selector') else False
            )
            
            # Run experiment
            result = run_single_cifar_experiment(
                selector=selector,
                model=model,
                train_dataset=corrupted_train,
                val_dataset=cifar10_test,
                dataset_name=f"cifar10_{corruption_type}_sev{severity}",
                device=device,
                epochs=config_args.epochs if config_args else 50,
                coreset_budget=config_args.coreset_budget if config_args else 1000,
                config_args=config_args,
                use_gp_optimizer=config_args.use_gp_optimizer if config_args else False
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
                rank=config_args.rank if config_args else 256,
                use_hypernetwork=config_args.use_hypernetwork if config_args and hasattr(config_args, 'use_hypernetwork') else True,
                use_mdp_selector=config_args.use_mdp_selector if config_args and hasattr(config_args, 'use_mdp_selector') else False
            )
            
            # Run experiment
            result = run_single_cifar_experiment(
                selector=selector,
                model=model,
                train_dataset=corrupted_train,
                val_dataset=cifar100_test,
                dataset_name=f"cifar100_{corruption_type}_sev{severity}",
                device=device,
                epochs=config_args.epochs if config_args else 50,
                coreset_budget=config_args.coreset_budget if config_args else 1000,
                config_args=config_args,
                use_gp_optimizer=config_args.use_gp_optimizer if config_args else False
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


@timing_decorator("run_single_cifar_experiment")
@memory_tracking_decorator("run_single_cifar_experiment")
def run_single_cifar_experiment(selector: RLGuidedGaLoreSelector,
                               model: nn.Module,
                               train_dataset: Dataset,
                               val_dataset: Dataset,
                               dataset_name: str,
                               device: str,
                               epochs: int = 50,
                               coreset_budget: int = 1000,
                               config_args=None,
                               use_gp_optimizer: bool = False) -> Dict[str, Any]:
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
        use_gp_optimizer: Whether to use GP-based weight optimization
    
    Returns:
        Dictionary with experiment results
    """
    
    logger.info(f"Running experiment on {dataset_name} for {epochs} epochs")
    
    # Initialize GP optimizer if requested
    gp_framework = None
    if use_gp_optimizer and GP_OPTIMIZER_AVAILABLE:
        logger.info("Initializing GP-based strategy weight optimizer...")
        gp_framework = EnhancedGALOREFramework(
            phase_detector=selector.phase_detector,
            selection_policy=selector.rl_policy,
            galore_config=selector.galore,
            num_epochs=epochs,
            coreset_size=coreset_budget
        )
        # Run initial optimization on validation set
        gp_framework.initialize_optimal_weights(val_dataset)
    
    # Initialize intermediate results manager
    output_dir = config_args.output_dir if config_args else "./results"
    experiment_name = f"{dataset_name}_{device}_{epochs}epochs"
    results_manager = IntermediateResultsManager(output_dir, experiment_name, config_args)
    
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
    
    # Initialize profiler for this experiment
    profiler = get_profiler()
    profiler.record_memory("experiment_start", device)
    
    # Training loop with progress bar
    epoch_pbar = tqdm(range(epochs), desc=f"Training {dataset_name}", 
                     unit="epoch", position=0, leave=True)
    
    for epoch in epoch_pbar:
        # Time the entire epoch
        with profiler.timer(f"epoch_{epoch}_total"):
            # Evaluate current performance
            with profiler.timer(f"epoch_{epoch}_evaluation"):
                val_loss, val_acc = evaluate_cifar_model(model, val_dataset, device)
                current_performance = val_acc  # Higher is better
            
            # Update GP optimizer weights if enabled
            if gp_framework is not None:
                # Compute current training metrics
                current_metrics = {
                    'gradient_norm': selector.phase_detector.gradient_norms[-1] if selector.phase_detector.gradient_norms else 1.0,
                    'loss_curvature': selector.phase_detector.hessian_traces[-1] if selector.phase_detector.hessian_traces else 0.0,
                    'gradient_alignment': selector.phase_detector.gradient_alignments[-1] if selector.phase_detector.gradient_alignments else 0.0,
                    'gradient_variance': np.var(list(selector.phase_detector.gradient_norms)) if len(selector.phase_detector.gradient_norms) > 1 else 0.0,
                    'gradient_variance_avg': np.mean(list(selector.phase_detector.gradient_norms)) if selector.phase_detector.gradient_norms else 1.0,
                    'forgetting_rate': 0.0,  # Would need to track this
                    'class_imbalance': 0.0   # Would need to compute this
                }
                
                # Get optimized weights
                optimized_weights = gp_framework.gp_optimizer.adaptive_weight_update(current_metrics)
                
                # Apply weights to selection (this would need to be implemented in your selector)
                # For now, we'll just log them
                if epoch % 10 == 0:
                    logger.info(f"GP-optimized weights: {optimized_weights}")
            
            # Select coreset
            with profiler.timer(f"epoch_{epoch}_coreset_selection"):
                selected_indices, selection_info = selector.select_coreset(
                    coreset_budget, current_performance
                )
            
            # Save intermediate results if enabled
            if config_args and config_args.save_intermediate:
                # Save strategy data
                if config_args.save_strategy_data:
                    results_manager.save_strategy_data(epoch, selection_info)
                
                # Save coreset data
                if config_args.save_coreset_data:
                    coreset_info = {
                        'selected_indices': selected_indices,
                        'coreset_size': len(selected_indices),
                        'budget': coreset_budget,
                        'selection_time': profiler.timings.get(f"epoch_{epoch}_coreset_selection", [0])[-1] if f"epoch_{epoch}_coreset_selection" in profiler.timings else 0
                    }
                    results_manager.save_coreset_data(epoch, coreset_info)
            
            # Log selection info and update progress bar
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Phase={selection_info['current_phase']}, "
                           f"Strategy={selection_info['selected_strategy']}, "
                           f"Val Acc={val_acc:.3f}")
            
            # Update progress bar with current metrics
            epoch_pbar.set_postfix({
                'val_acc': f'{val_acc:.3f}',
                'phase': selection_info['current_phase'],
                'strategy': selection_info['selected_strategy']
            })
            
            # Train on selected coreset
            with profiler.timer(f"epoch_{epoch}_training"):
                coreset = Subset(train_dataset, selected_indices)
                coreset_loader = DataLoader(coreset, batch_size=batch_size, shuffle=True)
                
                model.train()
                train_loss = 0
                
                # Add batch-level progress bar
                batch_pbar = tqdm(coreset_loader, desc=f"Epoch {epoch} batches", 
                                unit="batch", position=1, leave=False)
                
                for batch_idx, (data, labels) in enumerate(batch_pbar):
                    with profiler.timer(f"epoch_{epoch}_batch_{batch_idx}"):
                        data, labels = data.to(device), labels.to(device)
                        
                        optimizer.zero_grad()
                        outputs = model(data)
                        loss = F.cross_entropy(outputs, labels)
                        loss.backward()
                        optimizer.step()
            
                        train_loss += loss.item()
                        
                        # Update batch progress bar
                        batch_pbar.set_postfix({
                            'loss': f'{loss.item():.4f}',
                            'avg_loss': f'{train_loss/(batch_idx+1):.4f}'
                        })
                
                # Close batch progress bar
                batch_pbar.close()
            
            scheduler.step()
            avg_train_loss = train_loss / len(coreset_loader)
            
            # Evaluate new performance
            with profiler.timer(f"epoch_{epoch}_final_evaluation"):
                new_val_loss, new_val_acc = evaluate_cifar_model(model, val_dataset, device)
            
            # Compute reward for RL
            reward = new_val_acc - val_acc  # Improvement in accuracy
            
            # Log metrics to TensorBoard
            profiler.log_metrics(epoch, {
                'train/loss': avg_train_loss,
                'val/loss': new_val_loss,
                'val/accuracy': new_val_acc,
                'val/accuracy_improvement': reward,
                'coreset/budget': coreset_budget,
                'coreset/selected_size': len(selected_indices),
                'strategy/selected': selection_info['selected_strategy'],
                'phase/current': selection_info['current_phase']
            })
            
            # Save timing data if enabled
            if config_args and config_args.save_intermediate and config_args.save_timing_data:
                timing_data = {
                    'epoch_timing': {
                        'total': profiler.timings.get(f"epoch_{epoch}_total", [0])[-1] if f"epoch_{epoch}_total" in profiler.timings else 0,
                        'evaluation': profiler.timings.get(f"epoch_{epoch}_evaluation", [0])[-1] if f"epoch_{epoch}_evaluation" in profiler.timings else 0,
                        'coreset_selection': profiler.timings.get(f"epoch_{epoch}_coreset_selection", [0])[-1] if f"epoch_{epoch}_coreset_selection" in profiler.timings else 0,
                        'training': profiler.timings.get(f"epoch_{epoch}_training", [0])[-1] if f"epoch_{epoch}_training" in profiler.timings else 0,
                        'final_evaluation': profiler.timings.get(f"epoch_{epoch}_final_evaluation", [0])[-1] if f"epoch_{epoch}_final_evaluation" in profiler.timings else 0
                    },
                    'performance_metrics': {
                        'train_loss': avg_train_loss,
                        'val_loss': new_val_loss,
                        'val_acc': new_val_acc,
                        'accuracy_improvement': reward
                    }
                }
                results_manager.save_timing_data(epoch, timing_data)
            
            # Save epoch data
            if config_args and config_args.save_intermediate:
                epoch_data = {
                    'train_loss': avg_train_loss,
                    'val_loss': new_val_loss,
                    'val_acc': new_val_acc,
                    'accuracy_improvement': reward,
                    'learning_rate': scheduler.get_last_lr()[0] if scheduler else lr,
                    'coreset_size': len(selected_indices)
                }
                results_manager.save_epoch_data(epoch, epoch_data)
            
            # Save checkpoint if enabled
            if config_args and config_args.save_intermediate and epoch % config_args.checkpoint_freq == 0:
                additional_data = {
                    'train_loss': avg_train_loss,
                    'val_loss': new_val_loss,
                    'val_acc': new_val_acc,
                    'accuracy_improvement': reward,
                    'strategy_used': selection_info['selected_strategy'],
                    'phase': selection_info['current_phase']
                }
                results_manager.save_checkpoint(epoch, model, optimizer, scheduler, selector, additional_data)
            
            # Log timing and memory every 10 epochs
            if epoch % 10 == 0:
                profiler.log_timing_summary(epoch)
                profiler.log_memory_summary(epoch)
                profiler.record_memory(f"epoch_{epoch}_end", device)
        
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
    
    # Save final comprehensive results
    if config_args and config_args.save_intermediate:
        # Save final summary
        results_manager.save_final_summary(results)
        
        # Save analysis data
        results_manager.save_analysis_data(profiler)
        
        logger.info(f"All intermediate results saved to: {output_dir}/{experiment_name}")
    
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
                       choices=['auto', 'cpu', 'cuda', 'mps'],
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
    parser.add_argument('--scheduler_step_size', type=int, default=20,
                       help='Step size for learning rate scheduler (default: 20)')
    parser.add_argument('--scheduler_gamma', type=float, default=0.5,
                       help='Gamma for learning rate scheduler (default: 0.5)')
    parser.add_argument('--use_gp_optimizer', action='store_true',
                       help='Use GP-based strategy weight optimization')
    parser.add_argument('--use_mdp_selector', action='store_true',
                       help='Use MDP-based dataset selection with Bloom filtering')
    
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
    
    # Profiling and logging options
    parser.add_argument('--enable_profiling', action='store_true', default=True,
                       help='Enable performance profiling (default: True)')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='Directory for TensorBoard logs (default: ./logs)')
    parser.add_argument('--profile_memory', action='store_true', default=True,
                       help='Enable memory profiling (default: True)')
    parser.add_argument('--profile_timing', action='store_true', default=True,
                       help='Enable detailed timing profiling (default: True)')
    
    # Intermediate results saving
    parser.add_argument('--save_intermediate', action='store_true', default=True,
                       help='Save intermediate results and checkpoints (default: True)')
    parser.add_argument('--checkpoint_freq', type=int, default=10,
                       help='Save checkpoint every N epochs (default: 10)')
    parser.add_argument('--save_strategy_data', action='store_true', default=True,
                       help='Save detailed strategy selection data (default: True)')
    parser.add_argument('--save_timing_data', action='store_true', default=True,
                       help='Save detailed timing and performance data (default: True)')
    parser.add_argument('--save_coreset_data', action='store_true', default=True,
                       help='Save coreset selection data (default: True)')
    
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
        
        # Print TensorBoard instructions
        if args.enable_profiling:
            log_dir = args.log_dir
            experiment_name = f"cifar_experiments_{args.experiment}_{device}"
            logger.info("\n" + "=" * 80)
            logger.info("TENSORBOARD VISUALIZATION")
            logger.info("=" * 80)
            logger.info(f"To view detailed metrics and timing profiles, run:")
            logger.info(f"tensorboard --logdir {log_dir}/{experiment_name}")
            logger.info(f"Then open http://localhost:6006 in your browser")
            logger.info("=" * 80)
        
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