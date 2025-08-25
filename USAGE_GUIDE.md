# CIFAR Experiments Usage Guide

This guide shows you how to use the comprehensive argparse system for running CIFAR experiments with various configurations.

## 🚀 Quick Start

### Basic Usage
```bash
# Run all experiments with default settings
python simple_expts.py

# Run only CIFAR10 experiments
python simple_expts.py --experiment cifar10

# Quick test mode
python simple_expts.py --quick
```

## 📋 All Available Arguments

### Experiment Type
```bash
--experiment {cifar10,cifar100,corruptions,all}
    Type of experiment to run (default: all)
```

### Dataset Settings
```bash
--data_dir PATH
    Directory for CIFAR datasets (default: ./data)

--download
    Download datasets if not present (default: True)
```

### Device Settings
```bash
--device {auto,cpu,cuda}
    Device to run experiments on (default: auto)
```

### Training Parameters
```bash
--epochs INT
    Number of training epochs (default: 50)

--batch_size INT
    Training batch size (default: 64)

--learning_rate FLOAT
    Learning rate (default: 0.001)

--weight_decay FLOAT
    Weight decay (default: 1e-4)
```

### Coreset Selection
```bash
--coreset_budget INT
    Size of coreset to select each epoch (default: 1000)

--memory_budget_mb INT
    Memory budget in MB (default: 1000)
```

### GaLore Parameters
```bash
--rank INT
    GaLore rank for gradient compression (default: 256)

--update_proj_gap INT
    GaLore projection update frequency (default: 200)
```

### RL Parameters
```bash
--rl_lr FLOAT
    RL policy learning rate (default: 0.0003)

--epsilon FLOAT
    RL exploration epsilon (default: 0.1)
```

### Phase Detection
```bash
--window_size INT
    Phase detection window size (default: 50)

--sensitivity FLOAT
    Phase transition sensitivity (default: 2.0)
```

### Output and Logging
```bash
--log_level {DEBUG,INFO,WARNING,ERROR}
    Logging level (default: INFO)

--save_results
    Save results to JSON files (default: True)

--plot_results
    Generate result plots (default: True)

--output_dir PATH
    Output directory for results (default: ./results)
```

### Quick Test Mode
```bash
--quick
    Quick test mode (fewer epochs, smaller models)
```

### Corruption Experiments
```bash
--corruption_severities INT [INT ...]
    Corruption severity levels to test (default: 1 3 5)

--corruption_types STR [STR ...]
    Corruption types to test (default: gaussian_noise defocus_blur brightness contrast)
```

### Model Architecture
```bash
--model_type {resnet,vgg}
    Model architecture to use (default: resnet)

--model_depth INT
    Model depth (default: 20 for ResNet, 16 for VGG)
```

## 🎯 Example Commands

### Basic Experiments
```bash
# Run all experiments with default settings
python simple_expts.py

# Run only CIFAR10 experiments
python simple_expts.py --experiment cifar10

# Run only CIFAR100 experiments
python simple_expts.py --experiment cifar100

# Run only corruption experiments
python simple_expts.py --experiment corruptions
```

### Custom Training Parameters
```bash
# Run with custom epochs and batch size
python simple_expts.py --epochs 100 --batch_size 128

# Custom learning rate and weight decay
python simple_expts.py --learning_rate 0.0005 --weight_decay 1e-3

# Custom coreset budget
python simple_expts.py --coreset_budget 2000 --memory_budget_mb 2000
```

### Model Architecture
```bash
# Use VGG instead of ResNet
python simple_expts.py --model_type vgg --model_depth 19

# Custom ResNet depth
python simple_expts.py --model_type resnet --model_depth 50
```

### Device and Performance
```bash
# Force CPU usage
python simple_expts.py --device cpu

# Force CUDA usage
python simple_expts.py --device cuda

# GPU optimized settings
python simple_expts.py --device cuda --batch_size 128 --coreset_budget 2000
```

### Quick Testing
```bash
# Quick test mode
python simple_expts.py --quick

# Quick test with specific experiment
python simple_expts.py --experiment cifar10 --quick

# Quick test with custom settings
python simple_expts.py --quick --epochs 5 --coreset_budget 200
```

### Corruption Experiments
```bash
# Test specific corruption types
python simple_expts.py --experiment corruptions --corruption_types gaussian_noise defocus_blur

# Test specific severity levels
python simple_expts.py --experiment corruptions --corruption_severities 1 2

# Custom corruption experiment
python simple_expts.py --experiment corruptions --corruption_types snow frost --corruption_severities 1 3 5
```

### Advanced Configuration
```bash
# Custom output directory
python simple_expts.py --output_dir ./my_results

# Debug logging
python simple_expts.py --log_level DEBUG

# Custom GaLore settings
python simple_expts.py --rank 512 --update_proj_gap 100

# Custom RL settings
python simple_expts.py --rl_lr 0.0001 --epsilon 0.05
```

## 🔧 Configuration Files

You can also use the configuration system:

```python
from config import get_config, create_custom_config

# Use pre-built configs
config = get_config("quick")      # quick, full, gpu, cpu
config = get_config("default")

# Create custom config
custom_config = create_custom_config(
    epochs=75, 
    batch_size=128,
    coreset_budget=1500
)
```

## 📊 Output Files

The experiments generate several output files:

- `cifar10_results.json` - CIFAR10 experiment results
- `cifar100_results.json` - CIFAR100 experiment results  
- `corruption_results.json` - Corruption experiment results
- `experiment_summary.json` - Overall experiment summary
- Various plots and visualizations

## 🚨 Important Notes

1. **Quick Mode**: Automatically reduces parameters for faster execution
2. **Device Auto-detection**: Automatically detects CUDA/CPU if not specified
3. **Memory Management**: Automatically cleans up GPU memory between experiments
4. **Download**: Datasets are automatically downloaded if not present
5. **Output Directory**: Automatically created if it doesn't exist

## 🧪 Testing

Test the argparse functionality:

```bash
# Test help
python simple_expts.py --help

# Test argument validation
python test_argparse.py

# Test configuration
python config.py
```

## 💡 Tips

1. **Start with quick mode** to verify everything works
2. **Use specific experiment types** to focus on what you need
3. **Monitor memory usage** with large coreset budgets
4. **Check logs** for detailed progress information
5. **Use custom output directories** to organize results

## 🆘 Troubleshooting

### Common Issues
- **CUDA out of memory**: Reduce `--coreset_budget` or `--memory_budget_mb`
- **Slow execution**: Use `--quick` mode or reduce `--epochs`
- **Import errors**: Check `requirements.txt` and install dependencies
- **Dataset issues**: Verify `--data_dir` and `--download` settings

### Getting Help
```bash
# Show all options
python simple_expts.py --help

# Test imports
python test_argparse.py

# Check configuration
python config.py
```
