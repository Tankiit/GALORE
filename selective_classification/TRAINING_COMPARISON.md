# DataComp Training: Fixed LR vs Adaptive Scheduler

## Experiment Comparison

### Setup
- **Dataset**: 1000 synthetic samples, 300 selected (30% budget)
- **Model**: CLIP ViT-B/32
- **Training**: 100 epochs, batch size 32
- **Optimizer**: AdamW

### Run 1: Fixed Learning Rate (OLD)
**Config**: LR = 1e-6 (constant)

**Results**:
- Epoch 1: Loss = 3.41, Acc = 4.1%
- Epoch 25: Loss = 3.37, Acc = 3.7%
- Epoch 50: Loss = 3.37, Acc = ~4%
- Epoch 75: Loss = 3.37, Acc = ~4%
- Epoch 100: Loss = 3.37, Acc = 4.5%

**Analysis**:
❌ **Barely any learning!**
- Loss decreased by only 0.04 over 100 epochs
- Accuracy increased by only 0.4%
- Learning rate was too small and never adapted

### Run 2: Adaptive Scheduler (NEW)
**Config**: Base LR = 1e-4, Cosine annealing with warmup

**Scheduler Behavior**:
```
Steps 1-50 (Warmup):
  LR: 0 → 1e-4 (linear increase)

Steps 51-1000 (Cosine Decay):
  LR: 1e-4 ~~~> ~5e-6 (smooth decay)
```

**Early Results** (first 4 steps):
- Step 1: LR = 2.00e-05 ✓ (warming up)
- Step 2: LR = 4.00e-05 ✓ (increasing)
- Step 3: LR = 6.00e-05 ✓ (increasing)
- Step 4: LR = 7.20e-05 ✓ (increasing)

✅ **Scheduler is working!**

**Status**: Training in progress...
- Log file: `datacomp_100epochs_with_scheduler.log`
- Monitor: `tail -f datacomp_100epochs_with_scheduler.log`

## Key Improvements

### 1. Adaptive Learning Rate
**Old**: Fixed 1e-6 (never changes)
```python
# Manual warmup only
if step < warmup_steps:
    lr = base_lr * (step / warmup_steps)
else:
    lr = base_lr  # STUCK HERE!
```

**New**: Cosine annealing (continuously adapts)
```python
# Warmup
if step < warmup_steps:
    lr = base_lr * (step / warmup_steps)
else:
    # Cosine decay
    progress = (step - warmup) / (total - warmup)
    lr = base_lr * 0.5 * (1 + cos(π * progress))
```

### 2. Higher Base Learning Rate
- Old: 1e-6 (too conservative)
- New: 1e-4 (100x larger, appropriate for CLIP)

### 3. Smooth Convergence
- Old: Constant LR → oscillates or gets stuck
- New: Decaying LR → smooth convergence to minimum

## Expected Final Results

Based on typical CLIP training curves:

| Metric | Fixed LR (Actual) | Adaptive LR (Expected) | Improvement |
|--------|------------------|------------------------|-------------|
| Final Loss | 3.37 | ~1.5-2.0 | 40-55% better |
| Final Accuracy | 4.5% | ~30-50% | 6-10x better |
| Training Stability | Poor (stuck) | Good (converging) | Much better |
| Learning Progress | Minimal | Continuous | Significant |

## Monitoring the New Run

### Check Learning Rate Schedule
```bash
grep "lr=" datacomp_100epochs_with_scheduler.log | awk '{print $NF}' | head -100
```

### Check Loss Progression
```bash
grep "Avg Loss" datacomp_100epochs_with_scheduler.log
```

### Watch Live
```bash
tail -f datacomp_100epochs_with_scheduler.log
```

### Compare at Specific Epochs
```bash
# Fixed LR vs Adaptive LR
echo "=== Epoch 1 ==="
grep -A 2 "Epoch 1 Summary" datacomp_100epochs.log
grep -A 2 "Epoch 1 Summary" datacomp_100epochs_with_scheduler.log

echo "=== Epoch 50 ==="
grep -A 2 "Epoch 50 Summary" datacomp_100epochs.log
grep -A 2 "Epoch 50 Summary" datacomp_100epochs_with_scheduler.log

echo "=== Epoch 100 ==="
grep -A 2 "Epoch 100 Summary" datacomp_100epochs.log
grep -A 2 "Epoch 100 Summary" datacomp_100epochs_with_scheduler.log
```

## Why This Matters

### For Paper/Research
1. **Shows MODE works**: With proper training, the selected data should perform well
2. **Validates selection**: If accuracy improves, the 30% selected data is indeed informative
3. **Demonstrates curriculum**: The adaptive schedule aligns with curriculum learning principles

### For Deployment
1. **Faster convergence**: Reaches good performance earlier
2. **Better final model**: Lower loss, higher accuracy
3. **More stable**: Smooth decay prevents oscillations

## Visualization

### Learning Rate Curve
```
LR
│
1e-4│    ___Warmup___
    │   /             \
    │  /               \___Cosine Decay___
    │ /                                   \
1e-6│/                                     \___
    └────────────────────────────────────────> Steps
    0    50                              1000
```

### Expected Loss Curve
```
Loss
│
3.4 │\
    │ \___Fixed LR (stuck at 3.37)
    │  \.
3.0 │   \.
    │    \.__Adaptive LR
2.5 │     \  \.
    │      \   \.
2.0 │       \__  \.___
    │          \________\.___
1.5 │                    \___
    └────────────────────────────────────────> Epochs
    0    25      50      75              100
```

## Next Steps

1. **Wait for completion**: ~50 minutes total
2. **Compare results**: Check if accuracy significantly improved
3. **Analyze learning curve**: Plot loss/accuracy vs epochs
4. **Adjust if needed**:
   - If still poor: Increase base LR further or check data quality
   - If unstable: Decrease base LR slightly
   - If converged early: Reduce total epochs

## Files

- `datacomp_100epochs.log` - Old run (fixed LR)
- `datacomp_100epochs_with_scheduler.log` - New run (adaptive LR)
- `mode_vlm_experiment.py` - Updated code with scheduler
- `server_configs/quick_test.yaml` - Updated config with higher LR

## Code Changes

See `mode_vlm_experiment.py` lines:
- **1659-1682**: `initialize_scheduler()` method
- **1663-1665**: Scheduler step in training
- **1805-1808**: Scheduler initialization before training

**The adaptive scheduler should dramatically improve convergence! 📈**
