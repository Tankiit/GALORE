# WikiText-103 Stratified Datasets

This module provides comprehensive WikiText-103 dataset support with multiple stratification methods for LLM training experiments.

## Features

### Dataset Variations

1. **wikitext103_full**: Complete WikiText-103 dataset without stratification
2. **wikitext103_quality_stratified**: Stratified by text quality scores (10 bins)
3. **wikitext103_length_stratified**: Stratified by text length (short/medium/long)
4. **wikitext103_domain_stratified**: Stratified by topic/domain (science/history/culture/technology/politics)
5. **wikitext103_complexity_stratified**: Stratified by text complexity (5 levels)

### Stratification Methods

- **Quality Stratification**: Based on text coherence, vocabulary diversity, and sentence structure
- **Length Stratification**: Categorizes texts into length-based bins
- **Topic Stratification**: Uses keyword matching or LDA for topic assignment
- **Complexity Stratification**: Based on average word length and sentence complexity

## Quick Start

### 1. Initialize Datasets

First, initialize and cache all dataset variations:

```bash
# Initialize with all samples
python initialize_datasets.py --data_dir /Users/mukher74/research/data

# Initialize with limited samples for testing
python initialize_datasets.py --max_samples 10000 --data_dir /Users/mukher74/research/data

# Create small test datasets (1000 samples)
python initialize_datasets.py --small
```

### 2. Use in LLM Experiments

```bash
# Run experiment with quality-stratified dataset
python llm_experiments.py \
    --model gpt2 \
    --dataset wikitext103_quality_stratified \
    --epochs 5 \
    --use_hypernetwork

# Run with length stratification
python llm_experiments.py \
    --model gpt2 \
    --dataset wikitext103_length_stratified \
    --epochs 5 \
    --use_hypernetwork

# Run with topic stratification
python llm_experiments.py \
    --model gpt2 \
    --dataset wikitext103_domain_stratified \
    --epochs 5 \
    --use_hypernetwork
```

### 3. Use in Python

```python
from llm_datasets import create_wikitext_datasets

# Load all dataset variations
datasets = create_wikitext_datasets()

# Access specific variations
quality_dataset = datasets['wikitext103_quality_stratified']
length_dataset = datasets['wikitext103_length_stratified']
topic_dataset = datasets['wikitext103_domain_stratified']

# Get statistics
if hasattr(quality_dataset, 'get_stratum_stats'):
    stats = quality_dataset.get_stratum_stats()
    print(f"Strata: {stats['strata_sizes']}")
```

## Dataset Structure

Each dataset contains `TextSample` objects with the following attributes:
- `text`: The raw text content
- `idx`: Sample index
- `length`: Word count
- `quality_score`: Computed quality score (0-1)
- `topic`: Assigned topic/domain
- `complexity`: Complexity score (0-1)
- `metadata`: Additional metadata dictionary

## Stratification Benefits

### Quality Stratification
- Ensures balanced representation of high and low-quality texts
- Helps model learn from diverse quality levels
- Useful for curriculum learning approaches

### Length Stratification
- Balances short, medium, and long sequences
- Helps with gradient stability across different sequence lengths
- Useful for studying model behavior on different text lengths

### Topic Stratification
- Ensures diverse domain coverage
- Prevents overfitting to specific topics
- Useful for domain adaptation studies

### Complexity Stratification
- Gradual increase in text complexity
- Supports curriculum learning
- Helps identify model capabilities at different complexity levels

## Integration with Hypernetworks

The stratified datasets work seamlessly with both standard and LLM-specific hypernetworks:

```python
from llm_experiments import run_llm_experiment

# Run with LLM hypernetwork and quality stratification
results = run_llm_experiment(
    model_name="gpt2",
    dataset_name="wikitext103_quality_stratified",
    use_llm_hypernetwork=True,
    use_stratified_dataset=True,
    stratification_type="quality",
    epochs=10,
    coreset_budget=2000
)
```

## Caching

All datasets are automatically cached for faster subsequent loading:
- Base datasets: `{data_dir}/wikitext103_cache/`
- Metadata: `{data_dir}/wikitext103_metadata.json`
- Full datasets: `{data_dir}/wikitext103_datasets.pkl`

To clear cache and regenerate:
```bash
python initialize_datasets.py --no_cache
```

## Performance Optimizations

1. **Lazy Evaluation**: Metadata computed only when needed
2. **Efficient Caching**: Pickle format for fast loading
3. **Batch Processing**: Stratification done in batches
4. **Memory Management**: Stream processing for large datasets

## Custom Stratification

To add custom stratification methods:

```python
from llm_datasets import WikiText103Dataset, StratifiedDataset

# Create custom stratification
class CustomStratifiedDataset(StratifiedDataset):
    def _stratify_samples(self):
        # Your custom stratification logic
        pass

# Use custom stratification
base_dataset = WikiText103Dataset()
custom_dataset = CustomStratifiedDataset(
    base_dataset, 
    "custom", 
    bins=5
)
```

## Requirements

- datasets (HuggingFace)
- nltk
- scikit-learn
- numpy
- torch
- tqdm

## Troubleshooting

### Out of Memory
- Use `--max_samples` to limit dataset size
- Use `--small` flag for testing

### Slow Loading
- Ensure datasets are cached (run `initialize_datasets.py` first)
- Check disk I/O performance

### Missing NLTK Data
The script automatically downloads required NLTK data. If issues persist:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Citation

If you use these stratified datasets in your research, please cite:
```bibtex
@misc{stratified_wikitext,
  title={Stratified WikiText-103 for Efficient LLM Training},
  author={Your Name},
  year={2024}
}
```