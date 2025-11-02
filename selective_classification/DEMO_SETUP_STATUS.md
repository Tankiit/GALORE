# Demo Setup Status and Next Steps

## Current Setup

### ✅ What's Ready

1. **MODE Selection Complete**
   - Selected indices saved: `datacomp_mode_cache/selected_indices.pt`
   - These indices represent the MODE-selected subset from your training

2. **Demo Infrastructure Built**
   - `build_demo_pipeline.sh` - Automated pipeline with checkpoints
   - `launch_gradio_demo.sh` - Quick demo launcher
   - `gradio_demo_largescale.py` - Professional web interface
   - `build_large_retrieval_database.py` - Database builder
   - Complete documentation files

3. **Data Format**
   - Your data uses **webdataset streaming format** from HuggingFace
   - Cache directory: `./datacomp_cache/`
   - No local image files stored

### ⚠️ What's Needed

The demo pipeline expects **local image files** in a directory structure, but your current setup uses **streaming webdataset**.

## Options to Proceed

### Option 1: Download Sample Images (Recommended for Demo)

Create a local cache of images from the webdataset for fast retrieval:

```bash
# Create a script to download selected samples
python download_mode_samples.py \
    --selected_indices ./datacomp_mode_cache/selected_indices.pt \
    --output_dir ./datacomp_local_images \
    --num_samples 30000
```

This would:
- Download 30K MODE-selected images locally
- Save them with metadata (captions, embeddings)
- Enable fast local retrieval

**Time**: ~2-3 hours depending on download speed
**Disk**: ~1.5 GB

### Option 2: Use Streaming Demo (Slower but No Download)

Modify the demo to work directly with streaming webdataset:
- Slower retrieval (network latency)
- Can't use FAISS indexing efficiently
- Good for quick testing

### Option 3: Use Pre-Encoded Embeddings Only

Build database from pre-computed embeddings without storing images:
- Store only embeddings + image URLs
- Retrieve by showing URLs or downloading on-demand
- Minimal disk space
- Images load from web in demo

## Recommended Workflow

### Quick Test (10 minutes)

Build a small streaming-based demo to test:

```bash
# Build small database (1000 samples, embeddings only)
python build_streaming_retrieval_db.py \
    --selected_indices ./datacomp_mode_cache/selected_indices.pt \
    --output_dir ./demo_db_small \
    --max_samples 1000 \
    --embeddings_only

# Launch demo
python gradio_demo_largescale.py \
    --mode_db ./demo_db_small \
    --streaming_mode
```

### Full Production Demo (3-4 hours)

Download images locally for best performance:

```bash
# Step 1: Download MODE samples
python download_mode_samples.py \
    --selected_indices ./datacomp_mode_cache/selected_indices.pt \
    --output_dir ./datacomp_data \
    --num_samples 30000

# Step 2: Build databases with pipeline
./build_demo_pipeline.sh --data_dir ./datacomp_data

# Step 3: Launch
cd demo_databases
./launch_demo.sh --share
```

## What I Can Create Next

Tell me which option you prefer:

1. **"Create download script"** - I'll build a script to download MODE-selected images locally
2. **"Create streaming demo"** - I'll modify the demo to work with streaming webdataset
3. **"Create embeddings-only demo"** - I'll build a lightweight demo using only embeddings
4. **"Use different dataset"** - If you have local images elsewhere (CIFAR-10, ImageNet subset, etc.)

## Current File Structure

```
selective_classification/
├── datacomp_cache/              # Webdataset cache
├── datacomp_mode_cache/
│   └── selected_indices.pt      # MODE selections (ready!)
├── build_demo_pipeline.sh       # Ready (needs local images)
├── build_large_retrieval_database.py  # Ready (needs local images)
├── gradio_demo_largescale.py    # Ready (needs database)
├── launch_gradio_demo.sh        # Ready (needs database)
└── [Documentation files]        # All ready!
```

## Quick Decision Guide

- **Want impressive demo for paper/reviewers?** → Option 1 (download images)
- **Want quick test right now?** → Option 2 (streaming demo)
- **Limited disk space?** → Option 3 (embeddings only)
- **Have other local dataset?** → Option 4 (use that instead)

Let me know which direction you'd like to go!
