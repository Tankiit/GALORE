# DataComp MODE - Server-Ready Configuration Complete!

## What's Been Created

### 1. YAML Configuration System
- **datacomp_config.yaml** - Default balanced config
- **server_configs/quick_test.yaml** - Fast test config (1K samples, 3 epochs)
- **server_configs/large_scale.yaml** - Production config (1M samples, 50 epochs)

### 2. Training Script
- **train_datacomp.py** - Clean, server-ready training script
  - Loads YAML configs
  - Supports command-line overrides
  - Creates experiment directories with timestamped results
  - Saves config and results summary automatically

### 3. Documentation
- **SERVER_DEPLOYMENT.md** - Complete deployment guide
  - Server setup instructions
  - Multiple deployment workflows
  - Hardware-specific configurations
  - Troubleshooting guide

## Quick Start

### Local Testing
```bash
# Test with small dataset
python train_datacomp.py --config server_configs/quick_test.yaml
```

### Server Deployment
```bash
# 1. Transfer files to server
rsync -avz selective_classification/ user@server:/path/to/project/

# 2. On server: Setup environment
conda create -n mode python=3.10 -y
conda activate mode
pip install torch torchvision open_clip_torch tqdm tensorboard pyyaml

# 3. Run training
python train_datacomp.py --config server_configs/large_scale.yaml
```

### Command-Line Overrides
```bash
# Quick test on CPU
python train_datacomp.py --override model.device=cpu

# Longer training
python train_datacomp.py --override training.epochs=50

# Multiple overrides
python train_datacomp.py \
  --config server_configs/large_scale.yaml \
  --override training.epochs=100 model.device=cuda training.batch_size=512
```

## What the Script Does

1. **Loads config** from YAML file
2. **Sets random seed** for reproducibility
3. **Creates experiment directory** with timestamp
4. **Saves config copy** for later reference
5. **Initializes models** (CLIP, proxy, etc.)
6. **Runs data selection** (scoring phase)
7. **Trains on selected data** (training phase)
8. **Logs to TensorBoard** throughout
9. **Saves results summary** as YAML

## Output Structure

After training:
```
results/
└── datacomp_mode_test_20251031_155126/
    ├── config.yaml              # Config used for this run
    └── results_summary.yaml     # Training summary with metrics

checkpoints/
└── selected_indices.pt          # IDs of selected samples

runs/
├── scoring/                     # TensorBoard logs for scoring
└── training/                    # TensorBoard logs for training
```

## Test Results

Successfully tested with `quick_test.yaml`:
- Dataset: 1,000 synthetic samples
- Selected: 300 samples (30% budget)
- Training: 3 epochs completed
- Logs: TensorBoard data saved
- Results: All summaries generated

Training metrics:
- **Epoch 1**: Loss 3.41, Acc 4.1%
- **Epoch 2**: Loss 3.38, Acc 3.8%
- **Epoch 3**: Loss 3.37, Acc 3.6%

## Files to Transfer for Server

Minimal files needed:
```
selective_classification/
├── mode_vlm_experiment.py       # Core implementation (2,210 lines, no emojis)
├── train_datacomp.py            # Training script with YAML support
├── datacomp_config.yaml         # Default config
└── server_configs/              # Pre-made configs
    ├── quick_test.yaml
    └── large_scale.yaml
```

Optional but useful:
```
├── view_tensorboard_stats.py    # CLI tool for monitoring
└── SERVER_DEPLOYMENT.md         # Full deployment guide
```

## Configuration Examples

### For Better Results (Based on Your Comment)

The current test showed poor results. Here are configs to try:

#### 1. More Training Epochs
```yaml
training:
  epochs: 50  # Up from 3
  learning_rate: 0.0001  # Higher LR
```

#### 2. Larger Selection Budget
```yaml
selection:
  budget: 0.5  # Select 50% instead of 30%
```

#### 3. Better Scoring Weights
```yaml
selection:
  nuclear_norm_weight: 0.5  # Balanced
  clip_score_weight: 0.5    # weights
```

#### 4. Larger Batch Sizes (if GPU available)
```yaml
training:
  batch_size: 256  # Up from 32
selection:
  batch_size: 512  # Up from 128
```

### Quick Override Commands

```bash
# Try 20 epochs
python train_datacomp.py --override training.epochs=20

# Select more data
python train_datacomp.py --override selection.budget=0.5

# Higher learning rate
python train_datacomp.py --override training.learning_rate=0.0001

# All together
python train_datacomp.py \
  --override training.epochs=20 \
             selection.budget=0.5 \
             training.learning_rate=0.0001
```

## Next Steps

1. **Test different configs locally** to find best settings
2. **Create custom config** based on what works (copy quick_test.yaml)
3. **Push to server** when ready
4. **Monitor with TensorBoard** during training
5. **Iterate on hyperparameters** based on results

## Key Features

✅ **No emojis in code** - Production ready
✅ **YAML configs** - Easy to edit and version control
✅ **Command-line overrides** - Quick experiments
✅ **Automatic logging** - TensorBoard integration
✅ **Experiment tracking** - Timestamped directories
✅ **Results summary** - YAML output for analysis
✅ **Reproducible** - Random seeds set
✅ **Server-ready** - Tested end-to-end

## Troubleshooting

### Poor Results

Try these adjustments in order:

1. Increase training epochs (50+)
2. Increase selection budget (0.5+)
3. Adjust learning rate (0.0001)
4. Balance scoring weights (0.5/0.5)
5. Use larger batch sizes (if GPU allows)

### Out of Memory

```yaml
model:
  device: cpu  # Use CPU
training:
  batch_size: 16  # Reduce batch size
```

### Slow Training

```yaml
selection:
  batch_size: 512  # Increase if GPU allows
cache:
  enabled: false  # Disable for small datasets
```

## Support

For detailed information, see:
- **SERVER_DEPLOYMENT.md** - Full deployment guide
- **datacomp_config.yaml** - All available options
- **mode_vlm_experiment.py** - Implementation details

## Ready for Server! 🚀

The codebase is now abstracted, configurable, and server-ready!
