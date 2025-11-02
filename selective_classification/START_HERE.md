# 🎯 MODE Retrieval Demo - START HERE

## ✨ One-Command Demo Setup

Your MODE demo is ready! Get started with just one command:

```bash
./setup_demo.sh --quick
```

This will:
1. Download 1,000 MODE-selected samples (~5 min)
2. Build demo databases (~5 min)
3. Launch interactive web demo

**Total time: 10 minutes** ⏱️

---

## 🎨 What You'll Get

An interactive web demo that shows:

- **Text-to-Image Retrieval**: Enter text, find matching images
- **Image-to-Text Retrieval**: Upload image, find matching captions
- **MODE vs Random Comparison**: See MODE's advantage clearly
- **Statistics Dashboard**: Quantify MODE's performance gain

Perfect for:
- Paper supplementary materials
- Reviewer demonstrations
- Lab presentations
- Your own experimentation

---

## 📖 Complete Documentation

| File | What It's For |
|------|---------------|
| **START_HERE.md** | This file - quick start guide |
| **COMPLETE_DEMO_GUIDE.md** | Comprehensive setup & usage guide |
| **DEMO_SETUP_STATUS.md** | Current status and technical options |
| **BUILD_LARGE_DEMO_QUICKSTART.txt** | Database builder reference |
| **GRADIO_DEMO_COMPLETE_GUIDE.txt** | Gradio interface details |

Start with `COMPLETE_DEMO_GUIDE.md` for full details!

---

## 🚀 Quick Options

### Option 1: Quick Test (10 minutes)
```bash
./setup_demo.sh --quick
```
Perfect for first-time testing!

### Option 2: Production Demo (5-6 hours)
```bash
./setup_demo.sh
```
Full-scale demo with impressive results!

### Option 3: Manual Control
See `COMPLETE_DEMO_GUIDE.md` for step-by-step instructions.

---

## 💡 What This Demo Shows

MODE achieves **96%+ of full-data quality** using only **30% of the data**!

The demo compares:
- **MODE (30K)**: Your intelligent selection - High quality ✓
- **Random (30K)**: Baseline - Lower quality ✗
- **Full (100K)**: Upper bound - Best, but 3× more data

Example results:
```
Query: "a dog playing in a park"

MODE (30K):   Avg Score: 0.83 ✓
Random (30K): Avg Score: 0.74 ✗ (12% worse)
Full (100K):  Avg Score: 0.86 ✓ (only 3% better!)
```

---

## ✅ Prerequisites Check

Before starting, verify you have:

```bash
# 1. MODE selections (should exist)
ls datacomp_mode_cache/selected_indices.pt

# 2. Python 3.8+
python3 --version

# 3. ~10 GB free disk space
df -h .

# 4. Scripts are ready
ls -l setup_demo.sh
```

All good? Run `./setup_demo.sh --quick`!

---

## 🎓 For Your Paper

After building the demo:

1. **Launch with share link**:
   ```bash
   cd demo_databases
   ./launch_demo.sh --share
   ```

2. **Add to paper**:
   - Include demo URL in supplementary materials
   - Take screenshots of comparison results
   - Record 2-minute video walkthrough

3. **Reviewer response**:
   > "Please try our interactive demo at [LINK] to see MODE's
   > advantages on your own queries!"

---

## 🐛 Having Issues?

### Quick Fixes

1. **"setup_demo.sh not found"**
   ```bash
   chmod +x setup_demo.sh
   ./setup_demo.sh --quick
   ```

2. **"selected_indices.pt not found"**
   - Run MODE selection first, or
   - Check path in your config files

3. **"Out of disk space"**
   - Use `--quick` mode (only 50 MB), or
   - Free up space for full demo (10 GB)

4. **"Download is slow"**
   - Normal! DataComp is large
   - Use `--quick` first to test (faster)

See `COMPLETE_DEMO_GUIDE.md` → Troubleshooting for more!

---

## 📂 What Gets Created

After setup:

```
datacomp_data/           # Your downloaded images
demo_databases/          # Built retrieval databases
  ├── mode_30k/         # MODE-selected samples
  ├── random_30k/       # Random baseline
  ├── full_100k/        # Full dataset
  └── launch_demo.sh    # ← Use this to launch!
```

---

## ⏱️ Time Estimates

| Mode | Samples | Time | Disk | Use Case |
|------|---------|------|------|----------|
| Quick | 1K | 10 min | 50 MB | Testing |
| Medium | 10K | 30 min | 500 MB | Presentations |
| Full | 30K | 5-6 hrs | 10 GB | Papers |

---

## 🎯 Next Steps

1. **Run quick test**:
   ```bash
   ./setup_demo.sh --quick
   ```

2. **Try the demo**:
   - Open browser at http://localhost:7860
   - Enter text queries
   - Upload test images
   - Compare MODE vs Random!

3. **Build full demo** (when ready):
   ```bash
   ./setup_demo.sh
   # Choose option 3 (full databases)
   ```

4. **Share with reviewers**:
   ```bash
   cd demo_databases
   ./launch_demo.sh --share
   ```

---

## 📖 Learn More

- **Complete Guide**: Open `COMPLETE_DEMO_GUIDE.md`
- **Technical Details**: See `DEMO_SETUP_STATUS.md`
- **Database Building**: Read `BUILD_LARGE_DEMO_QUICKSTART.txt`

---

## 🎉 You're Ready!

Everything is set up and ready to go!

**Start your demo now:**

```bash
./setup_demo.sh --quick
```

Then open http://localhost:7860 in your browser!

---

**Questions?** Check `COMPLETE_DEMO_GUIDE.md` for comprehensive documentation.

**Good luck with your MODE demo! 🚀**
