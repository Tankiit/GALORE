# TensorBoard Logging Guide for DataComp MODE

This guide explains how to use TensorBoard logging in the DataComp MODE implementation.

## Quick Start

### 1. Install Dependencies

```bash
pip install torch open_clip_torch datasets diskcache tensorboard tqdm
```

### 2. Run Training with TensorBoard Enabled

```bash
# Small scale demo
python mode_vlm_experiment.py --mode small

# DataComp MODE demo
python mode_vlm_experiment.py --datacomp_mode

# Hybrid MODE demo
python mode_vlm_experiment.py --hybrid_demo
```

### 3. Launch TensorBoard

In a separate terminal:

```bash
tensorboard --logdir=./runs --port=6006
```

Then open in your browser: **http://localhost:6006**

---

## Log Directory Structure

```
./runs/
├── cache/                    # DataCompGradientCache logs
│   └── events.out.tfevents.*
├── training/                 # DataCompTrainer logs
│   └── events.out.tfevents.*
├── hybrid_mode/             # HybridMODETrainer logs
│   └── events.out.tfevents.*
└── datacomp_mode/           # DataCompMODE pipeline logs
    ├── scoring/             # DataScorer logs
    └── training/            # CLIPTrainer logs
```

---

## Logged Metrics

### Cache Statistics (`cache/`)

| Metric | Description | Use Case |
|--------|-------------|----------|
| `cache/hit_rate` | Percentage of cache hits | Monitor cache effectiveness |
| `cache/hits` | Total number of cache hits | Track cache usage |
| `cache/misses` | Total cache misses | Identify cold start periods |
| `cache/computes` | Number of gradient computations | Measure computational savings |
| `cache/aged_out` | Aged-out gradients | Monitor gradient staleness |
| `cache/bloom_efficiency` | Bloom filter effectiveness | Validate bloom filter design |

**Interpretation:**
- **Hit rate > 80%**: Excellent cache performance
- **Hit rate 50-80%**: Good cache performance
- **Hit rate < 50%**: Consider tuning cache parameters

### Training Metrics (`train/`)

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| `train/loss` | Contrastive loss | Should decrease over time |
| `train/accuracy` | Average top-1 accuracy | Should increase (aim for >70%) |
| `train/acc_i2t` | Image-to-text accuracy | Monitor modality-specific performance |
| `train/acc_t2i` | Text-to-image accuracy | Should match acc_i2t closely |
| `train/selected_count` | Number of samples selected | Fixed by budget parameter |
| `train/selection_ratio` | Fraction of samples used | Should match configured budget |
| `train/mean_importance` | Mean importance score | Tracks sample quality |
| `train/learning_rate` | Current learning rate | Verify warmup schedule |
| `train/hypernet_loss` | Hypernetwork training loss | Only if hypernetwork enabled |

**Key Insights:**
- If `acc_i2t` and `acc_t2i` diverge significantly, one modality may be overfitting
- Flat `train/loss` indicates potential learning issues
- `mean_importance` should be stable after warmup

### Importance Scoring (`importance/`)

| Metric | Description | Use Case |
|--------|-------------|----------|
| `importance/proxy_mean` | Mean proxy importance | Compare proxy vs checkpoint signals |
| `importance/clip_alignment_mean` | Mean CLIP alignment | Measure feature quality |
| `importance/proxy_distribution` | Histogram of proxy scores | Visualize score distribution |
| `importance/clip_alignment_distribution` | Histogram of alignments | Check for mode collapse |

**Histogram Analysis:**
- **Uniform distribution**: Poor sample discrimination
- **Bimodal distribution**: Good separation of easy/hard samples
- **Heavy tail**: Some samples are much more important

### Epoch Summaries (`epoch/`)

| Metric | Description | Frequency |
|--------|-------------|-----------|
| `epoch/loss` | Average loss per epoch | Once per epoch |
| `epoch/accuracy` | Average accuracy per epoch | Once per epoch |

### Scoring Phase (`scoring/`)

| Metric | Description | Use Case |
|--------|-------------|----------|
| `scoring/batch_mean_score` | Mean score per batch | Monitor scoring progress |
| `scoring/batch_std_score` | Score variance per batch | Check score stability |
| `scoring/final_mean` | Overall mean score | Compare across runs |
| `scoring/final_std` | Overall score variance | Measure dataset diversity |
| `scoring/score_distribution` | Histogram of all scores | Visualize selection threshold |

---

## Configuration

### Enable/Disable TensorBoard

```python
# DataCompCacheConfig
config = DataCompCacheConfig(
    enable_tensorboard=True,        # Enable logging
    tensorboard_dir="./runs",       # Log directory
    log_frequency=50                # Log every N steps
)

# HybridMODEConfig
config = HybridMODEConfig(
    enable_tensorboard=True,
    tensorboard_dir="./runs/hybrid_mode",
    log_frequency=50
)

# DataCompMODEConfig
config = DataCompMODEConfig(
    enable_tensorboard=True,
    tensorboard_dir="./runs/datacomp_mode",
    log_frequency=50
)
```

### Adjust Logging Frequency

```python
# Log more frequently (every 10 steps)
config.log_frequency = 10

# Log less frequently (every 200 steps)
config.log_frequency = 200
```

---

## Advanced Usage

### Multi-Run Comparison

```bash
# Run 1: Budget 10%
python mode_vlm_experiment.py --mode small  # Logs to ./runs/

# Run 2: Budget 30%
# Modify config.budget = 0.30
python mode_vlm_experiment.py --mode small

# View both runs in TensorBoard
tensorboard --logdir=./runs --port=6006
```

TensorBoard will show both runs side-by-side for comparison.

### Remote Server Access

If running on a remote server:

```bash
# On remote server
tensorboard --logdir=./runs --port=6006 --host=0.0.0.0

# On local machine
ssh -L 6006:localhost:6006 user@remote-server

# Open in browser
http://localhost:6006
```

### Export Data for Analysis

```python
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Load TensorBoard logs
event_acc = EventAccumulator('./runs/cache/')
event_acc.Reload()

# Extract scalar data
hit_rate = event_acc.Scalars('cache/hit_rate')

# Convert to pandas DataFrame
import pandas as pd
df = pd.DataFrame(hit_rate)
```

---

## Progress Bars (tqdm)

All training loops use `tqdm` for real-time progress tracking:

```
Epoch 1: 100%|██████████| 1000/1000 [05:23<00:00, 3.09batch/s, loss=0.5234, acc=0.723]
```

**Progress Bar Fields:**
- `loss`: Current batch loss
- `acc`: Current batch accuracy
- `selected`: Number of samples selected (DataCompTrainer)
- `proxy_loss`: Proxy model loss (HybridMODETrainer)

---

## Best Practices

### 1. Monitor During Training

Keep TensorBoard open during training to:
- Detect training instabilities early
- Verify hyperparameters are working
- Track cache efficiency in real-time

### 2. Compare Runs

Use meaningful run names:

```python
config.tensorboard_dir = f"./runs/budget_{config.budget}_lr_{config.learning_rate}"
```

### 3. Clean Up Old Logs

```bash
# Remove old runs
rm -rf ./runs/old_experiment_*

# Archive important runs
tar -czf important_run.tar.gz ./runs/important_experiment/
```

### 4. Key Metrics to Watch

1. **cache/hit_rate** - Should increase over time
2. **train/loss** - Should decrease steadily
3. **train/accuracy** - Should increase to >70%
4. **train/selection_ratio** - Should match configured budget
5. **importance/proxy_distribution** - Should show clear separation

---

## Troubleshooting

### Issue: TensorBoard shows no data

**Solution:**
1. Check that `enable_tensorboard=True` in config
2. Verify `tensorboard` package is installed: `pip install tensorboard`
3. Ensure training has run for at least `log_frequency` steps

### Issue: Port 6006 already in use

**Solution:**
```bash
# Use a different port
tensorboard --logdir=./runs --port=6007

# Or kill existing TensorBoard
pkill -f tensorboard
```

### Issue: Old data appearing in TensorBoard

**Solution:**
```bash
# Clear browser cache or use incognito mode
# Or delete old logs
rm -rf ./runs/*
```

### Issue: Logs taking too much disk space

**Solution:**
```python
# Reduce logging frequency
config.log_frequency = 200  # Log every 200 steps instead of 50

# Or disable histogram logging (code modification needed)
# Comment out: self.writer.add_histogram(...)
```

---

## Example Workflow

```bash
# Terminal 1: Start training
python mode_vlm_experiment.py --mode small

# Terminal 2: Launch TensorBoard
tensorboard --logdir=./runs --port=6006

# Browser: Open http://localhost:6006

# Monitor:
# - cache/hit_rate should increase
# - train/loss should decrease
# - train/accuracy should increase
# - Histograms should show clear distributions

# After training completes:
# - Review final metrics
# - Compare with baseline runs
# - Export data for paper/reports
```

---

## Visualization Examples

### Cache Hit Rate Over Time
```
100% |████████████████████████████████|
 80% |                    ╱────────────
 60% |              ╱────╱
 40% |        ╱────╱
 20% |  ╱────╱
  0% |╱
     0   1000   2000   3000   4000   5000
              Training Steps
```

### Importance Score Distribution
```
Count
  │     ╭───╮
  │   ╭─╯   ╰─╮
  │  ╭╯       ╰╮
  │ ╭╯         ╰╮
  │╭╯           ╰─
  └──────────────── Score
  0.0          1.0
```

---

## Citation

If you use this implementation, please cite:

```bibtex
@software{datacomp_mode_2025,
  title={DataComp MODE: Model-Optimized Data Selection with TensorBoard Logging},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/datacomp-mode}
}
```

---

## Additional Resources

- **TensorBoard Documentation**: https://www.tensorflow.org/tensorboard
- **PyTorch TensorBoard Tutorial**: https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html
- **DataComp Paper**: https://arxiv.org/abs/2304.14108
- **MODE Paper**: [Your paper link]

---

## Contact

For questions or issues:
- Open an issue on GitHub
- Check the main README.md for more information
- See `python mode_vlm_experiment.py --usage` for command-line help
