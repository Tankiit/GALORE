# DataComp MODE - Server Deployment Guide

This guide shows how to deploy DataComp MODE training on a server with YAML configuration support.

## Quick Start

### 1. Basic Training

```bash
# Run with default config
python train_datacomp.py

# Run with specific config
python train_datacomp.py --config server_configs/large_scale.yaml

# Quick test run
python train_datacomp.py --config server_configs/quick_test.yaml
```

### 2. Override Config from Command Line

```bash
# Override training epochs
python train_datacomp.py --override training.epochs=50

# Override multiple values
python train_datacomp.py --override training.epochs=50 model.device=cuda training.batch_size=256

# Override with config file + command line
python train_datacomp.py --config server_configs/large_scale.yaml --override training.epochs=100
```

### 3. Resume from Checkpoint

```bash
python train_datacomp.py --resume checkpoints/datacomp_mode_20231031/checkpoint_500.pt
```

## Configuration Files

### Available Configs

1. **datacomp_config.yaml** - Default balanced config
2. **server_configs/quick_test.yaml** - Fast test (1K samples, 3 epochs)
3. **server_configs/large_scale.yaml** - Production config (1M samples, 50 epochs)

### Config Structure

```yaml
dataset:
  name: "datacomp"       # Dataset name
  size: 1000000          # Number of samples (-1 for all)

model:
  clip_model: "ViT-B-32" # CLIP architecture
  device: "cuda"         # cuda or cpu

selection:
  strategy: "one_shot"   # one_shot, iterative, hybrid
  budget: 0.1           # Select 10% of data
  batch_size: 512       # Scoring batch size

training:
  epochs: 50
  batch_size: 256
  learning_rate: 0.0001

cache:
  enabled: true
  max_size_mb: 8192     # 8GB cache for gradients

logging:
  tensorboard: true
  log_frequency: 50
```

## Server Deployment

### 1. Transfer Files to Server

```bash
# SCP method
scp -r selective_classification/ user@server:/path/to/project/

# Or use rsync (recommended)
rsync -avz --progress \
  --exclude '*.pyc' \
  --exclude '__pycache__' \
  --exclude '.git' \
  selective_classification/ user@server:/path/to/project/
```

### 2. Setup on Server

```bash
# SSH to server
ssh user@server

# Navigate to project
cd /path/to/project/selective_classification

# Create conda environment
conda create -n mode python=3.10 -y
conda activate mode

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install open_clip_torch tqdm tensorboard pyyaml
```

### 3. Run Training

```bash
# Activate environment
conda activate mode

# Run training
python train_datacomp.py --config server_configs/large_scale.yaml
```

### 4. Run in Background (tmux/screen)

```bash
# Using tmux
tmux new -s datacomp_training
python train_datacomp.py --config server_configs/large_scale.yaml 2>&1 | tee training.log
# Press Ctrl+B then D to detach

# Reattach later
tmux attach -t datacomp_training

# Or using screen
screen -S datacomp_training
python train_datacomp.py --config server_configs/large_scale.yaml 2>&1 | tee training.log
# Press Ctrl+A then D to detach

# Reattach later
screen -r datacomp_training

# Or using nohup
nohup python train_datacomp.py --config server_configs/large_scale.yaml > training.log 2>&1 &
```

## Monitoring Training

### 1. TensorBoard (Remote)

On server:
```bash
tensorboard --logdir=./runs --port=6006 --bind_all
```

On local machine (port forwarding):
```bash
ssh -N -L 6006:localhost:6006 user@server
```

Then open: http://localhost:6006

### 2. Watch Log File

```bash
# Tail the log
tail -f training.log

# Watch with grep for key metrics
tail -f training.log | grep -E "Loss|Accuracy|Epoch"
```

### 3. Check Progress

```bash
python view_tensorboard_stats.py --logdir runs
```

## Example Workflows

### Workflow 1: Quick Test Before Large Run

```bash
# 1. Test with small dataset
python train_datacomp.py --config server_configs/quick_test.yaml

# 2. If successful, run large scale
python train_datacomp.py --config server_configs/large_scale.yaml
```

### Workflow 2: Hyperparameter Sweep

```bash
# Test different learning rates
for lr in 0.0001 0.00005 0.00001; do
  python train_datacomp.py \
    --config server_configs/large_scale.yaml \
    --override training.learning_rate=$lr \
              experiment.name="datacomp_lr${lr}"
done
```

### Workflow 3: Multi-GPU Training (Coming Soon)

```bash
# Single node, multiple GPUs
python -m torch.distributed.launch \
  --nproc_per_node=4 \
  train_datacomp.py \
  --config server_configs/large_scale.yaml \
  --override server.distributed=true server.world_size=4
```

## Output Structure

After training, you'll have:

```
results/
└── datacomp_mode_large_20231031_143022/
    ├── config.yaml              # Copy of config used
    ├── results_summary.yaml     # Training summary
    └── ...

checkpoints/
└── cache/
    └── selected_indices.pt      # IDs of selected samples

runs/
└── datacomp_mode/
    ├── scoring/                 # TensorBoard logs for scoring
    └── training/                # TensorBoard logs for training
```

## Configuration Recommendations

### For Different Hardware

#### Small GPU (8GB VRAM)
```yaml
selection:
  batch_size: 64
training:
  batch_size: 32
cache:
  max_size_mb: 1024
```

#### Medium GPU (16GB VRAM)
```yaml
selection:
  batch_size: 256
training:
  batch_size: 128
cache:
  max_size_mb: 4096
```

#### Large GPU (40GB+ VRAM)
```yaml
selection:
  batch_size: 512
training:
  batch_size: 256
cache:
  max_size_mb: 16384
```

#### CPU Only
```yaml
model:
  device: "cpu"
  mixed_precision: false
selection:
  batch_size: 32
training:
  batch_size: 16
cache:
  max_size_mb: 2048
```

### For Different Dataset Sizes

#### Small (< 10K samples)
```yaml
dataset:
  size: 10000
selection:
  budget: 0.5  # Select 50%
training:
  epochs: 20
cache:
  enabled: false  # Not worth it for small datasets
```

#### Medium (10K - 100K samples)
```yaml
dataset:
  size: 100000
selection:
  budget: 0.3
training:
  epochs: 30
cache:
  enabled: true
  max_size_mb: 2048
```

#### Large (100K - 1M samples)
```yaml
dataset:
  size: 1000000
selection:
  budget: 0.1
training:
  epochs: 50
cache:
  enabled: true
  max_size_mb: 8192
```

#### Very Large (> 1M samples)
```yaml
dataset:
  size: 10000000
selection:
  budget: 0.05
training:
  epochs: 100
cache:
  enabled: true
  max_size_mb: 16384
  bloom_filter_size: 1000000
```

## Troubleshooting

### Out of Memory

```yaml
# Reduce batch sizes
selection:
  batch_size: 64  # Down from 512
training:
  batch_size: 32  # Down from 256

# Reduce cache size
cache:
  max_size_mb: 1024  # Down from 8192
```

### Slow Training

```yaml
# Increase batch sizes (if GPU allows)
training:
  batch_size: 512

# Enable mixed precision
model:
  mixed_precision: true

# Reduce logging frequency
logging:
  log_frequency: 100  # Up from 50
```

### Poor Results

```yaml
# Increase training epochs
training:
  epochs: 100

# Adjust learning rate
training:
  learning_rate: 0.0001  # Try different values

# Adjust selection budget
selection:
  budget: 0.2  # Select more data

# Tune scoring weights
selection:
  nuclear_norm_weight: 0.5
  clip_score_weight: 0.5
```

## Advanced Features

### Custom Datasets

Edit `train_datacomp.py` function `create_dataset()`:

```python
def create_dataset(yaml_config):
    dataset_name = yaml_config['dataset']['name']

    if dataset_name == "my_custom_dataset":
        from my_dataset import MyDataset
        return MyDataset(
            root=yaml_config['dataset']['root'],
            size=yaml_config['dataset']['size']
        )
    # ... existing code
```

### Custom Scoring Functions

Modify `mode_vlm_experiment.py` class `DataScorer`:

```python
def compute_hybrid_score(self, images, texts):
    # Your custom scoring logic
    custom_score = your_scoring_function(images, texts)
    return custom_score
```

### Checkpointing (TODO)

```python
# Will be added in future version
# For now, training runs end-to-end
```

## Performance Tips

1. **Use SSD for data** - Much faster than HDD
2. **Pin memory** - Set `pin_memory=True` in DataLoader
3. **Num workers** - Set to number of CPU cores (e.g., `num_workers=8`)
4. **Mixed precision** - Always enable on modern GPUs
5. **Cache gradients** - Enable for datasets > 10K samples
6. **Batch size** - Maximize GPU utilization without OOM

## Files Needed for Server

Minimal files to transfer:

```
selective_classification/
├── mode_vlm_experiment.py       # Core implementation
├── train_datacomp.py            # Training script
├── datacomp_config.yaml         # Default config
├── server_configs/              # Pre-made configs
│   ├── quick_test.yaml
│   └── large_scale.yaml
└── view_tensorboard_stats.py    # Optional: for monitoring
```

## Support

For issues or questions:
1. Check TensorBoard logs
2. Review training.log file
3. Try quick_test.yaml first
4. Adjust config based on hardware
