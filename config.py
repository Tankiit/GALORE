"""
Configuration file for CIFAR experiments
"""

from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class ExperimentConfig:
    """Configuration for CIFAR experiments"""
    
    # Dataset settings
    data_dir: str = "./data"
    download_datasets: bool = True
    
    # Model settings
    cifar10_model_depth: int = 20
    cifar100_model_depth: int = 32
    use_vgg: bool = False
    vgg_depth: int = 16
    
    # Training settings
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    scheduler_step_size: int = 20
    scheduler_gamma: float = 0.5
    
    # Coreset selection settings
    coreset_budget: int = 1000
    memory_budget_mb: int = 1000
    
    # GaLore settings
    rank: int = 256
    update_proj_gap: int = 200
    
    # RL settings
    rl_learning_rate: float = 0.0003
    rl_hidden_dim: int = 256
    epsilon: float = 0.1
    epsilon_decay: float = 0.95
    replay_buffer_size: int = 1000
    batch_size_rl: int = 32
    gamma: float = 0.99
    
    # Phase detection settings
    window_size: int = 50
    sensitivity: float = 2.0
    min_phase_length: int = 10
    
    # Strategy discovery settings
    population_size: int = 50
    tournament_size: int = 5
    mutation_rate: float = 0.1
    
    # Hypernetwork settings
    use_hypernetwork: bool = True
    hypernet_hidden_dim: int = 64  # Reduced for faster training
    hypernet_attention_heads: int = 2  # Reduced for efficiency
    hypernet_learning_rate: float = 0.001  # Higher for faster convergence
    hypernet_lazy_eval: bool = True  # Enable lazy evaluation for speed
    hypernet_cache_size: int = 5000  # Smaller cache for CIFAR10
    hypernet_update_freq: int = 5  # Update hypernetwork every N epochs
    
    # LLM Hypernetwork settings
    use_llm_hypernetwork: bool = True
    llm_hypernet_hidden_dim: int = 128  # Larger for LLM complexity
    llm_hypernet_num_heads: int = 4  # More attention heads for text
    llm_hypernet_dropout: float = 0.1  # Dropout for regularization
    llm_hypernet_learning_rate: float = 0.001
    llm_max_sequence_length: int = 512
    llm_gradient_accumulation_steps: int = 4  # For handling large models
    llm_batch_size: int = 8
    llm_coreset_budget: int = 2000  # Larger budget for text data
    llm_warmup_steps_ratio: float = 0.1  # Warmup for LLM training
    
    # Corruption experiment settings
    corruption_severities: List[int] = None
    corruption_types_subset: List[str] = None
    
    # Logging and output
    log_level: str = "INFO"
    save_results: bool = True
    plot_results: bool = True
    save_models: bool = False
    
    def __post_init__(self):
        if self.corruption_severities is None:
            self.corruption_severities = [1, 3, 5]
        
        if self.corruption_types_subset is None:
            self.corruption_types_subset = [
                'gaussian_noise', 'defocus_blur', 'brightness', 'contrast'
            ]


@dataclass
class CIFAR10Config(ExperimentConfig):
    """CIFAR10 specific configuration"""
    
    # CIFAR10 specific settings
    num_classes: int = 10
    model_depth: int = 20
    
    # CIFAR10 variations to test
    variations: List[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.variations is None:
            self.variations = ['cifar10_standard', 'cifar10_strong_aug', 'cifar10_cutout']


@dataclass
class CIFAR100Config(ExperimentConfig):
    """CIFAR100 specific configuration"""
    
    # CIFAR100 specific settings
    num_classes: int = 100
    model_depth: int = 32
    
    # CIFAR100 variations to test
    variations: List[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.variations is None:
            self.variations = ['cifar100_standard', 'cifar100_strong_aug']


@dataclass
class CorruptionConfig(ExperimentConfig):
    """Corruption experiment configuration"""
    
    # Corruption specific settings
    test_all_corruptions: bool = True
    test_cifar100_corruptions: bool = True
    
    # Corruption types to test
    corruption_types: List[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.corruption_types is None:
            self.corruption_types = [
                'gaussian_noise', 'shot_noise', 'impulse_noise',
                'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                'snow', 'frost', 'fog', 'brightness', 'contrast',
                'elastic_transform', 'pixelate', 'jpeg_compression'
            ]


@dataclass
class LLMConfig(ExperimentConfig):
    """LLM-specific configuration"""
    
    # LLM model settings
    model_name: str = "gpt2"
    tokenizer_name: str = "gpt2"
    max_sequence_length: int = 512
    
    # LLM training settings
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    
    # LLM dataset settings
    dataset_name: str = "wikitext"
    dataset_subset: str = "wikitext-2-raw-v1"
    max_train_samples: int = 10000
    max_val_samples: int = 1000
    
    # LLM coreset settings
    use_llm_hypernetwork: bool = True
    llm_coreset_budget: int = 2000
    llm_selection_frequency: int = 1  # Select new coreset every N epochs
    
    # LLM hypernetwork specific
    llm_scoring_functions: List[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.llm_scoring_functions is None:
            self.llm_scoring_functions = [
                'perplexity', 'token_diversity', 'attention_coverage',
                'gradient_alignment', 'token_importance', 'repetition_penalty'
            ]


# Default configurations
DEFAULT_CONFIG = ExperimentConfig()

# Quick test configuration (fewer epochs, smaller models)
QUICK_TEST_CONFIG = ExperimentConfig(
    epochs=10,
    coreset_budget=500,
    memory_budget_mb=500,
    rank=128,
    corruption_severities=[1, 3],
    corruption_types_subset=['gaussian_noise', 'defocus_blur', 'brightness']
)

# Full experiment configuration
FULL_EXPERIMENT_CONFIG = ExperimentConfig(
    epochs=100,
    coreset_budget=2000,
    memory_budget_mb=2000,
    rank=512,
    population_size=100,
    tournament_size=10
)

# GPU optimized configuration
GPU_CONFIG = ExperimentConfig(
    batch_size=128,
    coreset_budget=2000,
    memory_budget_mb=2000,
    rank=512
)

# CPU optimized configuration
CPU_CONFIG = ExperimentConfig(
    batch_size=32,
    coreset_budget=500,
    memory_budget_mb=500,
    rank=128,
    epochs=30
)

# LLM Quick Test Configuration
LLM_QUICK_CONFIG = LLMConfig(
    epochs=3,
    max_train_samples=1000,
    max_val_samples=100,
    llm_coreset_budget=200,
    batch_size=4,
    gradient_accumulation_steps=2,
    llm_hypernet_hidden_dim=64,
    llm_hypernet_num_heads=2
)

# LLM Full Training Configuration
LLM_FULL_CONFIG = LLMConfig(
    epochs=10,
    max_train_samples=50000,
    max_val_samples=5000,
    llm_coreset_budget=5000,
    batch_size=16,
    gradient_accumulation_steps=8,
    llm_hypernet_hidden_dim=256,
    llm_hypernet_num_heads=8,
    learning_rate=2e-5
)

# LLM GPU Optimized Configuration
LLM_GPU_CONFIG = LLMConfig(
    batch_size=32,
    gradient_accumulation_steps=2,
    llm_coreset_budget=8000,
    llm_hypernet_hidden_dim=256,
    memory_budget_mb=4000,
    rank=1024
)


def get_config(config_name: str = "default") -> ExperimentConfig:
    """Get configuration by name"""
    configs = {
        "default": DEFAULT_CONFIG,
        "quick": QUICK_TEST_CONFIG,
        "full": FULL_EXPERIMENT_CONFIG,
        "gpu": GPU_CONFIG,
        "cpu": CPU_CONFIG,
        "llm_quick": LLM_QUICK_CONFIG,
        "llm_full": LLM_FULL_CONFIG,
        "llm_gpu": LLM_GPU_CONFIG
    }
    
    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(configs.keys())}")
    
    return configs[config_name]


def create_custom_config(**kwargs) -> ExperimentConfig:
    """Create custom configuration from keyword arguments"""
    base_config = DEFAULT_CONFIG
    
    # Update with custom values
    for key, value in kwargs.items():
        if hasattr(base_config, key):
            setattr(base_config, key, value)
        else:
            raise ValueError(f"Unknown config parameter: {key}")
    
    return base_config


# Example usage:
if __name__ == "__main__":
    # Get default config
    config = get_config("default")
    print(f"Default epochs: {config.epochs}")
    
    # Get quick test config
    quick_config = get_config("quick")
    print(f"Quick test epochs: {quick_config.epochs}")
    
    # Create custom config
    custom_config = create_custom_config(epochs=75, batch_size=128)
    print(f"Custom epochs: {custom_config.epochs}")
    print(f"Custom batch size: {custom_config.batch_size}")
