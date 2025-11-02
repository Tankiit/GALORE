# GitHub Push Guide for MODE-VLM

## 📦 What to Push

### ✅ Essential Files (Must Push)

```bash
# Core Training Scripts
datacomp_small_mode.py
train_datacomp.py
run_vlm_experiment.py
train_mode_hypernetwork_zeroshot.py
mode_vlm_experiment.py
mode_selective_classification.py

# Demo Scripts
quick_demo.sh
build_embedding_demo.py
simple_embedding_demo.py
build_demo_pipeline.sh
launch_gradio_demo.sh
gradio_demo_largescale.py
download_mode_samples.py

# Utilities
visualize_mode_training.py
view_tensorboard_stats.py
build_retrieval_database.py
build_large_retrieval_database.py
interactive_retrieval_demo.py
inspect_dataset.py

# Configuration
requirements.txt
datacomp_config.yaml
.gitignore

# Documentation
README.md
START_HERE.md
COMPLETE_DEMO_GUIDE.md
WORKING_DEMO_SOLUTION.md
QUICK_START.md
BUILD_LARGE_DEMO_QUICKSTART.txt
GRADIO_DEMO_COMPLETE_GUIDE.txt
DEMO_SETUP_STATUS.md
GITHUB_PUSH_GUIDE.md  # This file
```

### ❌ Do NOT Push (Too Large / Sensitive)

```bash
# Large Files
*.pt  # Model checkpoints
*.pth  # Model weights
*.pkl  # Pickle files
*.bin  # FAISS indices

# Data
datacomp_data/  # Downloaded images
datacomp_cache/  # HuggingFace cache
data/  # Any local datasets
datasets/  # Dataset directories

# Outputs
checkpoints/  # Training checkpoints
mode_output/  # MODE selections
results/  # Experiment results
runs/  # TensorBoard logs
demo_databases/  # Built databases
demo_embedding_db/  # Demo databases

# Logs
*.log
logs/
```

### ⚠️ Maybe Push (Small Essential Data)

```bash
# Small example files (< 10 MB)
datacomp_mode_cache/selected_indices.pt  # Your MODE selections (5KB - OK to push!)

# Example configs
server_configs/  # Deployment configs (if not sensitive)

# Example results
results_summary.json  # Small result files (< 1MB)
```

## 🚀 Step-by-Step Push Instructions

### 1. Initialize Git (if not already)

```bash
cd selective_classification

# Check if git is initialized
git status

# If not initialized:
git init
```

### 2. Add Remote Repository

```bash
# Replace with your GitHub repo URL
git remote add origin https://github.com/yourusername/mode-vlm.git

# Verify
git remote -v
```

### 3. Stage Files

```bash
# Add essential files
git add README.md
git add requirements.txt
git add .gitignore
git add *.py
git add *.sh
git add *.md
git add *.txt
git add *.yaml

# Add small data files (check size first!)
ls -lh datacomp_mode_cache/selected_indices.pt
git add datacomp_mode_cache/selected_indices.pt

# Check what will be committed
git status
```

### 4. Commit

```bash
git commit -m "Initial commit: MODE-VLM implementation

- Core training scripts for DataComp with MODE selection
- Hypernetwork-driven curriculum learning
- Zero-shot evaluation (ImageNet + COCO)
- Interactive retrieval demos (embedding-based & image-based)
- Comprehensive documentation and quick start guides
- Example MODE selections included"
```

### 5. Push to GitHub

```bash
# First push (creates branch)
git push -u origin main

# Or if using 'master' branch:
# git push -u origin master

# Subsequent pushes:
git push
```

## 📋 Pre-Push Checklist

### Before Pushing, Verify:

```bash
# 1. Check file sizes
find . -type f -size +10M | grep -v ".git"

# 2. Ensure .gitignore is working
git status --ignored

# 3. Test imports locally
python -c "import mode_vlm_experiment; print('✓ Imports work')"

# 4. Check for sensitive info
grep -r "password\|secret\|api_key" . --exclude-dir=.git

# 5. Verify documentation links
grep -r "\[.*\](.*.md)" *.md
```

### Quick Size Check

```bash
# List files to be committed and their sizes
git ls-files -z | xargs -0 du -h | sort -h | tail -20
```

## 🗂️ Recommended GitHub Repository Structure

```
mode-vlm/
├── selective_classification/          # Your code (this directory)
│   ├── README.md                     # Main documentation
│   ├── requirements.txt              # Dependencies
│   ├── .gitignore                    # Git ignore rules
│   ├── [All .py files]               # Python code
│   ├── [All .sh files]               # Shell scripts
│   ├── [All .md files]               # Documentation
│   └── datacomp_mode_cache/
│       └── selected_indices.pt       # Example MODE selections (5KB)
│
├── LICENSE                           # Add license file
└── .github/                          # GitHub-specific (optional)
    ├── workflows/                    # CI/CD (optional)
    └── ISSUE_TEMPLATE/               # Issue templates (optional)
```

## 📝 GitHub Repository Settings

### 1. Add LICENSE

Choose a license (MIT recommended):

```bash
# Create LICENSE file
cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2024 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF

git add LICENSE
git commit -m "Add MIT license"
```

### 2. Repository Description

Add to GitHub repo settings:
```
MODE-VLM: Data-efficient vision-language learning via multi-objective curriculum selection. Achieve 96%+ performance with 70% less data!
```

### 3. Topics/Tags

Add these topics on GitHub:
- `deep-learning`
- `vision-language`
- `clip`
- `curriculum-learning`
- `data-selection`
- `pytorch`
- `datacomp`
- `efficient-training`

### 4. README Badges

Already included in README.md:
- Python version
- PyTorch version
- License

## 🔄 Updating After Push

### For Future Updates

```bash
# 1. Make changes to files
# ... edit code ...

# 2. Stage changes
git add <modified_files>

# 3. Commit with descriptive message
git commit -m "Add feature: <description>"

# 4. Push
git push
```

### Example Update Commits

```bash
# Feature addition
git commit -m "feat: Add distributed training support for multi-GPU setups"

# Bug fix
git commit -m "fix: Resolve CUDA out of memory error in large batch processing"

# Documentation
git commit -m "docs: Update demo guide with troubleshooting section"

# Performance
git commit -m "perf: Optimize embedding computation with caching"
```

## 📦 Git LFS (For Large Files)

If you need to push model checkpoints:

```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.pt"
git lfs track "*.pth"
git lfs track "*.bin"

# Add .gitattributes
git add .gitattributes
git commit -m "Add Git LFS tracking"

# Now add large files
git add model.pt
git commit -m "Add pretrained model"
git push
```

**Note**: GitHub has LFS bandwidth limits. Better to host models elsewhere (HuggingFace, Google Drive, etc.)

## 🌐 Hosting Large Files Externally

### Option 1: HuggingFace Hub

```python
# Upload model to HuggingFace
from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj="./mode_output/final_clip_model.pt",
    path_in_repo="final_clip_model.pt",
    repo_id="yourusername/mode-vlm",
    repo_type="model"
)
```

Then in README:
```markdown
## Download Pretrained Models

```bash
# From HuggingFace Hub
wget https://huggingface.co/yourusername/mode-vlm/resolve/main/final_clip_model.pt
```
```

### Option 2: Google Drive / Dropbox

Upload to cloud storage and add download link in README:
```markdown
## Download Data

- [MODE Selected Indices (5KB)](https://drive.google.com/file/d/XXXXX)
- [Pretrained CLIP Model (1.2GB)](https://drive.google.com/file/d/XXXXX)
```

## ✅ Final Verification

After pushing, verify on GitHub:

```bash
# 1. Clone in a fresh directory to test
cd /tmp
git clone https://github.com/yourusername/mode-vlm.git
cd mode-vlm/selective_classification

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run quick test
./quick_demo.sh

# 4. Verify documentation renders correctly on GitHub
```

## 📊 Recommended Push Order

```bash
# First push: Core functionality
git add README.md requirements.txt .gitignore
git add mode_vlm_experiment.py datacomp_small_mode.py train_datacomp.py
git commit -m "Initial commit: Core MODE-VLM implementation"
git push

# Second push: Demo system
git add quick_demo.sh build_embedding_demo.py simple_embedding_demo.py
git add gradio_demo_largescale.py
git commit -m "Add interactive demo system"
git push

# Third push: Documentation
git add START_HERE.md COMPLETE_DEMO_GUIDE.md *.txt
git commit -m "Add comprehensive documentation"
git push

# Fourth push: Utilities
git add visualize_mode_training.py build_demo_pipeline.sh
git commit -m "Add training visualization and advanced tools"
git push

# Fifth push: Example data
git add datacomp_mode_cache/selected_indices.pt
git commit -m "Add example MODE selections"
git push
```

## 🎉 You're Ready!

Follow these steps and your MODE-VLM code will be on GitHub!

### Quick Commands Summary

```bash
# Setup
git init
git remote add origin https://github.com/yourusername/mode-vlm.git

# Add files
git add README.md requirements.txt .gitignore
git add *.py *.sh *.md *.yaml *.txt
git add datacomp_mode_cache/selected_indices.pt

# Commit & Push
git commit -m "Initial commit: MODE-VLM implementation"
git push -u origin main
```

### After Push

1. ✅ Add repository description on GitHub
2. ✅ Add topics/tags
3. ✅ Enable Issues & Discussions
4. ✅ Add collaborators if needed
5. ✅ Star your own repo 😄

**Done! Your MODE-VLM is now on GitHub! 🚀**
