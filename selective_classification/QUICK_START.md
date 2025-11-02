# 🚀 MODE VLM Quick Start Guide

Complete guide to running the DataComp MODE implementation with TensorBoard logging and tqdm progress bars.

## ⚡ Quick Start (1 minute)

```bash
# 1. Run experiment
python mode_vlm_experiment.py --mode small

# 2. View TensorBoard (new terminal)
tensorboard --logdir=./runs --port=6006

# 3. Open browser
# Go to: http://localhost:6006
```

## 📦 Installation

```bash
# Install all dependencies
pip install torch torchvision open_clip_torch datasets diskcache tensorboard tqdm numpy

# Or if using conda
conda install pytorch torchvision -c pytorch
pip install open_clip_torch datasets diskcache tensorboard tqdm
```

## 🎯 Available Commands

### Run Experiments

```bash
# Small scale test (1M samples, ~5 min)
python mode_vlm_experiment.py --mode small

# Medium scale (50M samples, ~1 hour)
python mode_vlm_experiment.py --mode medium

# Full DataComp MODE pipeline (correct implementation)
python mode_vlm_experiment.py --datacomp_mode

# Hybrid MODE demo (proxy-guided selection)
python mode_vlm_experiment.py --hybrid_demo

# Generate SLURM script for cluster
python mode_vlm_experiment.py --generate_slurm

# Show detailed usage guide
python mode_vlm_experiment.py --usage
```

### View Results

```bash
# Launch TensorBoard
tensorboard --logdir=./runs --port=6006

# View stats from command line (no UI)
python view_tensorboard_stats.py

# View specific run
python view_tensorboard_stats.py --logdir ./runs/cache

# Show all logged steps
python view_tensorboard_stats.py --all-steps
```

## 📊 What Gets Logged

### Automatic Logging

All experiments automatically log to TensorBoard:

```
./runs/
├── cache/              # Cache statistics
├── training/           # Training metrics
├── hybrid_mode/        # Hybrid MODE specific
└── datacomp_mode/      # DataComp MODE pipeline
    ├── scoring/
    └── training/
```

### Key Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| `cache/hit_rate` | Cache hit percentage | >80% after warmup |
| `train/loss` | Contrastive loss | Decreasing |
| `train/accuracy` | Top-1 accuracy | >70% |
| `train/selection_ratio` | % samples selected | Matches budget |
| `cache/bloom_efficiency` | Bloom filter savings | >90% |

## 🎨 TensorBoard Features

### Scalars Tab
- Line plots of all metrics over time
- Smoothing slider for noisy metrics
- Multi-run comparison

### Histograms Tab
- Importance score distributions
- Gradient distributions
- Feature alignment patterns

### Time Series Tab
- Real-time metric updates
- Relative/absolute time views

## 📈 Example Session

```bash
# Terminal 1: Start training
$ python mode_vlm_experiment.py --mode small

================================================================================
SMALL SCALE TEST: 1M samples
================================================================================
Available models: ['ViT-B/32', 'ViT-B/16', 'ViT-L/14'] (Default: ViT-B/32)
TensorBoard logging to: runs/cache
DiskCache: 10.0GB at datacomp_cache
DataComp cache ready: 1M samples

Step 0: {'cache_hit': False, 'aged': False}
Step 20: {'cache_hit': False, 'aged': False}
...
Step 500: {'cache_hit': True, 'aged': False}  # Cache warming up!
...

================================================================================
DATACOMP GRADIENT CACHE STATS
================================================================================
Global step:        1000
Hit rate:           65.3%  # ✅ Cache is working!
Hits:               653
Misses:             347
Computes:           347
...
```

```bash
# Terminal 2: Launch TensorBoard
$ tensorboard --logdir=./runs --port=6006

TensorBoard 2.x at http://localhost:6006 (Press CTRL+C to quit)
```

```bash
# Terminal 3: Quick stats check
$ python view_tensorboard_stats.py

📊 TensorBoard Statistics Viewer
================================================================================

📁 CACHE
--------------------------------------------------------------------------------
🔢 Scalar Metrics:

  [CACHE]
    hit_rate        Latest: 0.6530  (step 1000)  [20 points]
    computes        Latest: 347     (step 1000)  [20 points]
    bloom_efficiency Latest: 0.9245  (step 1000)  [20 points]

💡 To view in browser:
   tensorboard --logdir=runs --port=6006
```

## 🔬 Advanced Usage

### Custom Configuration

```python
from mode_vlm_experiment import DataCompCacheConfig, DataCompTrainer

# Create custom config
config = DataCompCacheConfig(
    dataset_size=5_000_000,        # 5M samples
    budget=0.15,                   # Select 15%
    cache_size_gb=50.0,
    enable_bloom=True,
    enable_tensorboard=True,
    tensorboard_dir="./my_experiment",
    log_frequency=25               # Log every 25 steps
)

# Initialize trainer
trainer = DataCompTrainer(config)

# Train with custom dataloader
# trainer.train_epoch(my_dataloader, warmup=True, epoch=0)
```

### Multi-Run Comparison

```bash
# Run 1: Budget 10%
python mode_vlm_experiment.py --mode small
# Logs to: ./runs/cache/

# Run 2: Budget 30% (modify code)
# Edit config.budget = 0.30
python mode_vlm_experiment.py --mode small
# Logs to: ./runs/cache/ (different timestamp)

# Compare both in TensorBoard
tensorboard --logdir=./runs --port=6006
# Both runs appear in the same dashboard!
```

### Remote Server

```bash
# On remote server
$ ssh user@remote-server
$ cd /path/to/project
$ python mode_vlm_experiment.py --mode small
$ tensorboard --logdir=./runs --port=6006 --host=0.0.0.0

# On local machine
$ ssh -L 6006:localhost:6006 user@remote-server

# Open browser: http://localhost:6006
```

### Export for Publication

```python
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd

# Load TensorBoard data
event_acc = EventAccumulator('./runs/cache/')
event_acc.Reload()

# Extract metrics
hit_rate_data = event_acc.Scalars('cache/hit_rate')

# Convert to DataFrame
df = pd.DataFrame([
    {'step': e.step, 'wall_time': e.wall_time, 'value': e.value}
    for e in hit_rate_data
])

# Save for paper
df.to_csv('cache_hit_rate.csv', index=False)

# Plot with matplotlib
import matplotlib.pyplot as plt
plt.plot(df['step'], df['value'])
plt.xlabel('Training Steps')
plt.ylabel('Cache Hit Rate')
plt.savefig('cache_performance.pdf')
```

## 🐛 Troubleshooting

### TensorBoard not showing data

```bash
# Check if logs exist
ls -lh runs/cache/

# Verify permissions
chmod -R 755 runs/

# Clear browser cache or use incognito mode

# Try different port
tensorboard --logdir=./runs --port=6007
```

### Port already in use

```bash
# Kill existing TensorBoard
pkill -f tensorboard

# Or use different port
tensorboard --logdir=./runs --port=6007
```

### Out of disk space

```bash
# Check cache size
du -sh datacomp_cache/

# Clear old runs
rm -rf runs/old_experiment_*

# Reduce cache size in config
config.cache_size_gb = 10.0  # Smaller cache
```

### Import errors

```bash
# Install missing packages
pip install tensorboard tqdm

# Verify installation
python -c "from torch.utils.tensorboard import SummaryWriter; print('OK')"
python -c "from tqdm import tqdm; print('OK')"
```

### CUDA out of memory

```python
# Use smaller batch size
config.batch_size = 128  # Instead of 256

# Enable mixed precision
config.mixed_precision = True

# Use CPU if needed (slower)
config.device = 'cpu'
```

## 📚 File Structure

```
selective_classification/
├── mode_vlm_experiment.py           # Main implementation
├── view_tensorboard_stats.py        # CLI stats viewer
├── QUICK_START.md                   # This file
├── TENSORBOARD_LOGGING_GUIDE.md     # Detailed logging guide
│
├── runs/                            # TensorBoard logs
│   ├── cache/
│   ├── training/
│   └── datacomp_mode/
│
├── datacomp_cache/                  # Disk cache
│   ├── bloom_filter.npz
│   └── ...
│
└── slurm_logs/                      # SLURM output (if using cluster)
```

## 🎓 Learning Path

1. **Start Simple** (5 min)
   ```bash
   python mode_vlm_experiment.py --usage
   python mode_vlm_experiment.py --mode small
   ```

2. **Explore TensorBoard** (10 min)
   ```bash
   tensorboard --logdir=./runs --port=6006
   # Open browser, explore different tabs
   ```

3. **Try Different Modes** (30 min)
   ```bash
   python mode_vlm_experiment.py --datacomp_mode
   python mode_vlm_experiment.py --hybrid_demo
   ```

4. **Customize Config** (1 hour)
   - Modify budget, cache size, logging frequency
   - Compare different configurations
   - Analyze results in TensorBoard

5. **Scale Up** (several hours)
   ```bash
   python mode_vlm_experiment.py --mode medium
   python mode_vlm_experiment.py --mode full  # Requires GPUs
   ```

## 💡 Best Practices

1. **Always use TensorBoard** - Real-time monitoring prevents wasted runs
2. **Start small** - Test on `--mode small` before scaling up
3. **Monitor hit_rate** - Should reach >80% after warmup
4. **Compare runs** - Use TensorBoard's multi-run comparison
5. **Save important runs** - Archive logs before cleanup
6. **Use progress bars** - tqdm shows immediate feedback
7. **Check bloom_efficiency** - Should be >90% for 400M scale

## 🚀 Performance Tips

- **Cache warm start**: Cache persists between runs (reuse disk cache)
- **Bloom filter**: Saves 90%+ of expensive lookups at scale
- **Mixed precision**: 2x faster, half memory usage
- **Gradient aging**: Automatically refreshes stale gradients
- **Step quantization**: Reuses gradients across multiple steps

## 📞 Getting Help

```bash
# Built-in help
python mode_vlm_experiment.py --help
python mode_vlm_experiment.py --usage
python view_tensorboard_stats.py --help

# Check logs
cat slurm_logs/datacomp_mode_*.out

# View documentation
cat TENSORBOARD_LOGGING_GUIDE.md
```

## 🎯 Success Metrics

Your experiment is working well if:

- ✅ Cache hit rate > 80% after warmup
- ✅ Training loss decreases steadily
- ✅ Accuracy > 70% on selected samples
- ✅ Bloom efficiency > 90%
- ✅ Selection ratio matches configured budget
- ✅ No OOM errors
- ✅ TensorBoard shows smooth curves

## 📖 Next Steps

1. Read `TENSORBOARD_LOGGING_GUIDE.md` for detailed metrics explanation
2. Try different selection budgets (10%, 20%, 30%)
3. Compare ViT-B/32 vs ViT-B/16 vs ViT-L/14
4. Scale up to full DataComp (400M samples)
5. Export results for your paper/report

---

**Happy Training! 🎉**

For more information, see the full documentation in `mode_vlm_experiment.py`.
