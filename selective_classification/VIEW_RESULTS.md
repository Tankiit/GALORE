# 📊 How to View Your Results

## ✅ Your Experiment is Running!

Current status after 100 steps:
- **Hit Rate**: 0% (normal during warmup)
- **Computes**: 100 gradient computations
- **Cache Misses**: 100 (building cache)
- **TensorBoard**: Logging active ✓

---

## 🎯 3 Ways to View Results

### Method 1: TensorBoard Web UI (RECOMMENDED)

**Step 1:** TensorBoard is already running! Just open your browser.

**Step 2:** Go to this URL:
```
http://localhost:6006
```

**Step 3:** You'll see the TensorBoard interface:
- Click **"SCALARS"** tab at the top
- Look for these metrics in the left sidebar:
  ```
  cache/
  ├── hit_rate
  ├── computes
  ├── misses
  ├── hits
  ├── aged_out
  └── bloom_efficiency
  ```

**Step 4:** Click on any metric to see the graph!

**Troubleshooting:**
- If you see "No data found", click the refresh button (🔄) in the top right
- Make sure you're on the SCALARS tab
- Wait 5 seconds for auto-refresh
- The data updates every 5 seconds automatically

---

### Method 2: Open the HTML File

**Quick way to open TensorBoard:**

```bash
# On Mac
open open_tensorboard.html

# On Linux
xdg-open open_tensorboard.html

# Or just double-click the file in Finder
```

This will automatically redirect you to http://localhost:6006

---

### Method 3: View PNG Image (FASTEST)

**A static image has been created for you!**

```bash
# On Mac
open tensorboard_stats.png

# On Linux
xdg-open tensorboard_stats.png

# Or just look at the file: ./tensorboard_stats.png
```

You should see 6 graphs showing:
1. **Hit Rate** - Currently 0% (will increase after warmup)
2. **Gradient Computes** - 100 total so far
3. **Cache Misses** - 100 (expected during warmup)
4. **Cache Hits** - 0 (will increase soon)
5. **Aged Out** - 0 (gradients are fresh)
6. **Bloom Efficiency** - 0% (will improve at scale)

---

### Method 4: Command Line Viewer

```bash
python view_tensorboard_stats.py
```

Output:
```
📊 TensorBoard Statistics Viewer
================================================================================

📁 CACHE
--------------------------------------------------------------------------------
  [CACHE]
    hit_rate        Latest: 0.0000  (step 100)  [2 points]
    computes        Latest: 100     (step 100)  [2 points]
    misses          Latest: 100     (step 100)  [2 points]
    ...
```

---

## 🔍 What You Should See

### Current Status (Step 100)

```
┌─────────────────────────────────────────┐
│ Metric            │ Value    │ Status   │
├─────────────────────────────────────────┤
│ Hit Rate          │ 0.0%     │ ⚠️ Warmup│
│ Gradient Computes │ 100      │ ✅ Good  │
│ Cache Misses      │ 100      │ ⚠️ Warmup│
│ Cache Hits        │ 0        │ ⚠️ Warmup│
│ Bloom Efficiency  │ 0.0%     │ ⚠️ Early │
└─────────────────────────────────────────┘

📝 Note: 0% hit rate is NORMAL during initial warmup!
```

### Expected After 500 Steps

```
┌─────────────────────────────────────────┐
│ Metric            │ Value    │ Status   │
├─────────────────────────────────────────┤
│ Hit Rate          │ 65-75%   │ ✅ Good  │
│ Gradient Computes │ 200-250  │ ✅ Good  │
│ Cache Misses      │ 150-200  │ ✅ Good  │
│ Cache Hits        │ 300-350  │ ✅ Good  │
│ Bloom Efficiency  │ 80-90%   │ ✅ Good  │
└─────────────────────────────────────────┘
```

---

## 📈 Live Monitoring

### Watch in Real-Time

**Option A: TensorBoard**
- Open http://localhost:6006
- Graphs update automatically every 5 seconds
- No need to refresh manually

**Option B: Regenerate Image**
```bash
# Re-run this to update the PNG
python3 << 'EOF'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

event_acc = EventAccumulator('./runs/cache')
event_acc.Reload()

# Create plots...
plt.savefig('tensorboard_stats.png', dpi=150)
print("Updated: tensorboard_stats.png")
EOF

open tensorboard_stats.png  # View updated image
```

**Option C: Command Line**
```bash
# Quick stats check
watch -n 5 python view_tensorboard_stats.py
```

---

## 🚀 Interactive TensorBoard Features

### In the Web UI (http://localhost:6006)

1. **Smoothing Slider** (left sidebar)
   - Adjust to smooth noisy curves
   - Useful for loss curves

2. **Download Options**
   - Click on a metric
   - Press the download button
   - Export as CSV or JSON

3. **Compare Runs**
   - Run multiple experiments
   - They all appear in the same dashboard
   - Toggle runs on/off in the left sidebar

4. **Zoom**
   - Click and drag on a graph to zoom
   - Double-click to reset zoom

5. **Tooltips**
   - Hover over points to see exact values
   - Shows step number and metric value

---

## 📂 File Locations

```
selective_classification/
├── runs/                              # TensorBoard logs
│   └── cache/
│       └── events.out.tfevents.*      # Event file
│
├── tensorboard_stats.png              # Static visualization (THIS!)
├── open_tensorboard.html              # Quick browser opener
├── view_tensorboard_stats.py          # CLI viewer
│
└── VIEW_RESULTS.md                    # This guide
```

---

## 💡 Interpreting Results

### Hit Rate

```
  0-20%  → ⚠️  Cache warming up (first ~200 steps)
 20-50%  → 🔄 Cache building  (steps 200-500)
 50-80%  → ✅ Good performance
 80-95%  → 🎉 Excellent! Cache is very effective
 95-100% → ⚠️  Too high - might indicate issues
```

### Gradient Computes

```
Should be much lower than total steps after warmup!

Example:
- Total steps: 1000
- Computes: 250  ✅ Good! (75% cache efficiency)
- Computes: 800  ❌ Bad  (only 20% efficiency)
```

### Bloom Efficiency

```
At large scale (>1M samples):
  0-50%  → ⚠️  Bloom filter not helping much
 50-90%  → ✅ Good
 90-99%  → 🎉 Excellent! Bloom saving lots of work
```

---

## 🔧 Troubleshooting

### Issue: "No data found" in TensorBoard

**Solution:**
```bash
# Check if experiment is running
ps aux | grep "python mode_vlm"

# Check if logs exist
ls -lh runs/cache/

# Restart TensorBoard
pkill -f tensorboard
tensorboard --logdir=./runs --port=6006 &

# Wait 10 seconds, then open http://localhost:6006
```

### Issue: Graphs not updating

**Solution:**
1. Click the refresh button (🔄) in TensorBoard
2. Check that your experiment is still running
3. Verify auto-refresh is enabled (should say "5s" in TensorBoard)
4. Clear browser cache (Cmd+Shift+R on Mac, Ctrl+Shift+R on Linux)

### Issue: Can't access http://localhost:6006

**Solution:**
```bash
# Check if TensorBoard is running
ps aux | grep tensorboard

# If not running, start it
tensorboard --logdir=./runs --port=6006 &

# Try a different port
tensorboard --logdir=./runs --port=6007 &
# Then go to http://localhost:6007
```

### Issue: Port already in use

**Solution:**
```bash
# Kill existing TensorBoard
pkill -f tensorboard

# Or use a different port
tensorboard --logdir=./runs --port=6007
```

---

## 🎯 Quick Commands Reference

```bash
# View in browser (if TensorBoard running)
open http://localhost:6006

# Or open the HTML file
open open_tensorboard.html

# View static image
open tensorboard_stats.png

# Command-line stats
python view_tensorboard_stats.py

# Start TensorBoard (if not running)
tensorboard --logdir=./runs --port=6006 &

# Kill TensorBoard
pkill -f tensorboard

# Check TensorBoard status
ps aux | grep tensorboard
```

---

## 📊 Example: Good Results

After training completes (~1000 steps), you should see:

```
================================================================================
DATACOMP GRADIENT CACHE STATS
================================================================================
Global step:        1,000
Hit rate:           82.5%     ✅ Excellent!
Hits:               825       ✅ Great cache efficiency
Misses:             175       ✅ Minimal recomputation
Computes:           175       ✅ Only 17.5% of samples recomputed
Aged out:           12        ✅ Few stale gradients
Bloom efficiency:   94.3%     ✅ Bloom filter working well

Cache:
  Size:             45.2 MB   ✅ Reasonable size
  Entries:          2,000     ✅ Good coverage
================================================================================
```

---

## 🚀 Next Steps

1. **Wait for experiment to complete** (should show 1000 steps total)

2. **View final results**:
   ```bash
   python view_tensorboard_stats.py
   open tensorboard_stats.png
   ```

3. **Compare different runs**:
   ```bash
   # Run with different settings
   # All runs appear in same TensorBoard!
   ```

4. **Export data for paper**:
   ```bash
   # TensorBoard → Click metric → Download button → CSV
   ```

5. **Try other modes**:
   ```bash
   python mode_vlm_experiment.py --datacomp_mode
   python mode_vlm_experiment.py --hybrid_demo
   ```

---

## 📞 Still Having Issues?

If you can't see results, run this diagnostic:

```bash
echo "=== Diagnostic Check ==="
echo "1. Checking TensorBoard process..."
ps aux | grep tensorboard | grep -v grep

echo -e "\n2. Checking log files..."
ls -lh runs/cache/

echo -e "\n3. Checking experiment process..."
ps aux | grep "python mode_vlm" | grep -v grep

echo -e "\n4. Reading latest stats..."
python view_tensorboard_stats.py

echo -e "\n5. Testing TensorBoard connection..."
curl -s http://localhost:6006 > /dev/null && echo "✅ TensorBoard is accessible" || echo "❌ TensorBoard not responding"
```

---

**🎉 You're all set! Your results are being logged and you have 4 ways to view them!**

Most users prefer the TensorBoard web UI: **http://localhost:6006**
