# MODE-VLM: Selective Classification for Vision-Language Models

This directory contains the implementation of MODE-VLM, a three-part framework combining MODE (Multi-Objective Data-Efficient learning) with selective classification principles for vision-language models.

## Overview

The framework consists of three main parts:

1. **MODE with Improved Token Scoring Strategies** - Enhanced token-level selection using 6 non-redundant strategies
2. **Selective Classification for VLM** - Rho-1 inspired excess loss for sample selection
3. **Future Work** - Combined approaches integrating both methodologies

## Key Features

### Improved Token Scoring Strategies

- **Excess Loss** (Rho-1 inspired) - Training vs reference gap
- **Gradient Magnitude** - Importance for learning
- **Attention Focus** - Information flow quality using Gini coefficient
- **Semantic Coherence** - Contextual consistency
- **Boundary Proximity** - Near decision boundaries
- **Diversity** - Novelty relative to selected (efficient version)

### WebDataset Integration

- Seamless loading from HuggingFace datasets
- Train/validation/test splitting functionality
- Preprocessing pipeline for vision-language data
- MODE-compatible data wrapper

### Training Phase Awareness

The framework provides different strategy weights for different training phases:

- **Warmup**: Focus on easy, coherent tokens
- **Early**: Balance between learning and exploration
- **Middle**: Focus on hard tokens and boundaries
- **Late**: Polish and refine

## Installation

```bash
pip install torch torchvision numpy webdataset huggingface_hub Pillow transformers
```

## Usage

### Basic Usage

```python
from mode_vlm_experiment import MODESelector, ImprovedTokenMODEConfig

# Initialize MODE selector
mode_selector = MODESelector(device='cuda', reference_model=None)

# Set up training state
training_state = {
    'epoch': 10,
    'total_epochs': 32,
    'val_accuracy': 0.75,
    'grad_norm': 2.5,
    'remaining_budget': 0.5,
    'avg_strategy_perf': 0.02
}

# Select important tokens
selected_indices, metadata = mode_selector.select_batch(
    input_ids,
    model_outputs,
    model,
    val_loader,
    budget=100,
    training_state=training_state,
    compute_gradients=True
)
```

### WebDataset Integration

```python
from mode_vlm_experiment import create_datasets, VLMDatasetWrapper
from transformers import AutoTokenizer

# Create datasets
datasets = create_datasets()
processed_datasets = {}
for split_name, dataset in datasets.items():
    processed_datasets[split_name] = add_preprocessing_pipeline(dataset, split_name)

# Set up tokenizer and wrapper
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
wrapped_dataset = VLMDatasetWrapper(processed_datasets['train'], tokenizer, device)

# Use with MODE selector
for batch in wrapped_dataset:
    if batch['input_ids'].size(0) > 0:
        outputs = model(batch['input_ids'], output_hidden_states=True, output_attentions=True)
        selected_indices, metadata = mode_selector.select_batch(
            batch['input_ids'], outputs, model, val_loader, budget=100, training_state=training_state
        )
        break
```

## Running the Example

```bash
python mode_vlm_experiment.py
```

This will run the complete workflow demonstration including:
- Strategy explanations and phase-specific recommendations
- WebDataset loading from CC3M dataset
- Example integration code

## Key Improvements Over Original MODE

- ✅ Integrated WebDataset for VLM data loading
- ✅ Improved token scoring strategies (6 non-redundant)
- ✅ Rho-1 style excess loss for selective classification
- ✅ Gradient-based importance scoring
- ✅ Semantic coherence measurement
- ✅ Efficient diversity computation
- ✅ Better attention focus measurement (Gini vs entropy)
- ✅ Training phase-aware strategy recommendations
- ✅ WebDataset wrapper for seamless integration

## Files

- `mode_vlm_experiment.py` - Main implementation file
- `requirements.txt` - Python dependencies
- `README.md` - This documentation

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{mode_vlm_2025,
  title={MODE-VLM: Selective Classification for Vision-Language Models},
  author={Research Team},
  year={2025},
  note={Implementation combining MODE and Rho-1 selective classification}
}
```

## License

This project is licensed under the same license as the parent GALORE repository.