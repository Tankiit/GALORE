# GaLore + RL-Guided Selection with CIFAR Dataset Variations

This repository implements a comprehensive framework for RL-guided data selection with GaLore integration, specifically designed to work with various CIFAR10 and CIFAR100 dataset variations and corruptions.

## Features

- **Phase Transition Detection**: Automatically detects training phase changes using multiple signals
- **RL-Guided Strategy Selection**: Uses reinforcement learning to adaptively choose data selection strategies
- **Compositional Strategy Discovery**: Discovers new strategies through evolutionary composition
- **GaLore Integration**: Gradient Low-Rank Projection for efficient gradient compression

- **CIFAR10 Variations**:
  - Standard CIFAR10
  - Strong augmentation (ColorJitter, RandomRotation)
  - Cutout augmentation
  
- **CIFAR100 Variations**:
  - Standard CIFAR100
  - Strong augmentation
  
- **Corruption Types** (15 different types):
  - Noise: Gaussian, Shot, Impulse
  - Blur: Defocus, Glass, Motion, Zoom
  - Weather: Snow, Frost, Fog
  - Digital: Brightness, Contrast, Elastic Transform, Pixelate, JPEG Compression

- **CIFARResNet**: ResNet variants (depth 20 for CIFAR10, 32 for CIFAR100)
- **CIFARVGG**: VGG-style architectures (depth 16/19)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd GALORE
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run all experiments:
```bash
python simple_expts.py --experiment all --data_dir ./data
```

Run specific experiment types:
```bash
# Only CIFAR10 experiments
python simple_expts.py --experiment cifar10 --data_dir ./data

# Only CIFAR100 experiments  
python simple_expts.py --experiment cifar100 --data_dir ./data

# Only corruption experiments
python simple_expts.py --experiment corruptions --data_dir ./data
```

### Command Line Arguments

- `--experiment`: Type of experiment to run
  - `cifar10`: CIFAR10 variations only
  - `cifar100`: CIFAR100 variations only
  - `corruptions`: Corruption experiments only
  - `all`: All experiments (default)
  
- `--data_dir`: Directory for CIFAR datasets (default: `./data`)
- `--device`: Device to run on (default: `auto` - automatically detects CUDA/CPU)

### Example Commands

```bash
# Run on specific device
python simple_expts.py --experiment cifar10 --device cuda

# Custom data directory
python simple_expts.py --experiment all --data_dir /path/to/data

# Run corruption experiments only
python simple_expts.py --experiment corruptions --device cpu
```

## Experiment Details

### CIFAR10 Experiments
- **Standard**: Basic CIFAR10 with standard augmentation
- **Strong Augmentation**: Enhanced with ColorJitter, RandomRotation
- **Cutout**: Standard + Cutout augmentation for regularization

### CIFAR100 Experiments  
- **Standard**: Basic CIFAR100 with standard augmentation
- **Strong Augmentation**: Enhanced with ColorJitter, RandomRotation

### Corruption Experiments
Tests 15 corruption types at 3 severity levels (1, 3, 5):
- **Noise corruptions**: Test robustness to various noise types
- **Blur corruptions**: Test robustness to different blur effects
- **Weather corruptions**: Test robustness to weather-like distortions
- **Digital corruptions**: Test robustness to compression and processing artifacts

## Output

### Results Files
- `cifar10_results.json`: CIFAR10 experiment results
- `cifar100_results.json`: CIFAR100 experiment results  
- `corruption_results.json`: Corruption experiment results

### Visualization
- **Accuracy Comparison**: Bar charts comparing final accuracies across variations
- **Training Curves**: Learning curves for best performing datasets
- **Phase Transitions**: Analysis of detected training phase changes
- **Strategy Usage**: Distribution of selected strategies across experiments
- **Corruption Analysis**: Robustness analysis across corruption types and severities

## Key Components

### 1. Phase Transition Detection
```python
class PhaseTransitionDetector:
    """Detects training phase changes using:
    - Gradient norm trajectories
    - Loss curvature changes  
    - Gradient alignment patterns
    - Data utility decay rates
    """
```

### 2. RL-Guided Strategy Selection
```python
class AdaptiveSelectionPolicy:
    """Neural network policy that learns to select:
    - Gradient magnitude strategy
    - Diversity strategy
    - Uncertainty strategy
    - Hybrid strategies
    """
```

### 3. Compositional Strategy Discovery
```python
class StrategyDiscoveryEngine:
    """Evolutionary algorithm that discovers new strategies by:
    - Combining primitive strategies
    - Using genetic operators (mutation, crossover)
    - Evaluating performance on validation data
    """
```

### 4. GaLore Integration
```python
class GaLore:
    """Gradient Low-Rank Projection for:
    - Efficient gradient compression
    - Memory-efficient training
    - Maintaining training quality
    """
```

## Performance Metrics

The framework tracks:
- **Accuracy**: Final and best validation accuracy
- **Loss**: Training and validation loss curves
- **Phase Transitions**: Number and timing of detected transitions
- **Strategy Usage**: Distribution of selected strategies
- **Compression Ratio**: GaLore compression efficiency
- **Training Dynamics**: Phase characteristics and transitions

## Customization

### Adding New Datasets
```python
# Add new dataset variation
new_variation = {
    'train': your_train_dataset,
    'test': your_test_dataset,
    'name': 'Your Dataset Name'
}
```

### Adding New Corruptions
```python
# Extend CorruptedCIFAR10 class
def _apply_your_corruption(self, data):
    # Your corruption logic
    return corrupted_data
```

### Adding New Models
```python
# Create new model class
class YourModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Your architecture
        
    def forward(self, x):
        # Your forward pass
        return output
```

## Requirements

- Python 3.7+
- PyTorch 1.9+
- CUDA support (optional, for GPU acceleration)
- 8GB+ RAM (16GB+ recommended for large experiments)
- 10GB+ disk space for CIFAR datasets

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{galore_rl_cifar,
  title={GaLore + RL-Guided Selection with CIFAR Dataset Variations},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/GALORE}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For questions and support, please open an issue on GitHub.
