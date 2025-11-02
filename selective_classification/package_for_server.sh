#!/bin/bash
# Package DataComp MODE for server deployment

echo "=========================================="
echo "Packaging DataComp MODE for Server"
echo "=========================================="

# Create package directory
PACKAGE_DIR="datacomp_mode_package"
rm -rf "$PACKAGE_DIR"
mkdir -p "$PACKAGE_DIR"
mkdir -p "$PACKAGE_DIR/server_configs"

# Copy essential files
echo "Copying core files..."
cp mode_vlm_experiment.py "$PACKAGE_DIR/"
cp train_datacomp.py "$PACKAGE_DIR/"
cp datacomp_config.yaml "$PACKAGE_DIR/"
cp server_configs/*.yaml "$PACKAGE_DIR/server_configs/"

# Copy documentation
echo "Copying documentation..."
cp SERVER_DEPLOYMENT.md "$PACKAGE_DIR/"
cp SETUP_COMPLETE.md "$PACKAGE_DIR/"

# Copy optional monitoring tools
echo "Copying monitoring tools..."
cp view_tensorboard_stats.py "$PACKAGE_DIR/" 2>/dev/null || echo "  (view_tensorboard_stats.py not found, skipping)"

# Make scripts executable
chmod +x "$PACKAGE_DIR/train_datacomp.py"

# Create README in package
cat > "$PACKAGE_DIR/README.md" << 'EOF'
# DataComp MODE - Server Package

## Quick Start

### 1. Setup Environment
```bash
conda create -n mode python=3.10 -y
conda activate mode
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install open_clip_torch tqdm tensorboard pyyaml
```

### 2. Test Locally First
```bash
# Quick test with small dataset
python train_datacomp.py --config server_configs/quick_test.yaml
```

### 3. Run Full Training
```bash
# Large-scale training
python train_datacomp.py --config server_configs/large_scale.yaml

# Or with overrides
python train_datacomp.py --override training.epochs=50 model.device=cuda
```

### 4. Monitor Training
```bash
# Start TensorBoard
tensorboard --logdir=runs --port=6006

# Or use CLI viewer
python view_tensorboard_stats.py
```

## Files Included

- `mode_vlm_experiment.py` - Core implementation
- `train_datacomp.py` - Training script
- `datacomp_config.yaml` - Default config
- `server_configs/` - Pre-made configurations
- `SERVER_DEPLOYMENT.md` - Full deployment guide
- `SETUP_COMPLETE.md` - Quick reference

## Configuration

Edit YAML files in `server_configs/` or use command-line overrides:

```bash
python train_datacomp.py \
  --config server_configs/large_scale.yaml \
  --override training.epochs=100 training.batch_size=512
```

See SERVER_DEPLOYMENT.md for detailed instructions.
EOF

# Create requirements.txt
cat > "$PACKAGE_DIR/requirements.txt" << 'EOF'
torch>=2.0.0
torchvision>=0.15.0
open-clip-torch>=2.20.0
tqdm>=4.65.0
tensorboard>=2.13.0
pyyaml>=6.0
numpy>=1.24.0
EOF

echo ""
echo "Package created successfully!"
echo ""
echo "Contents of $PACKAGE_DIR:"
ls -lh "$PACKAGE_DIR"
echo ""
echo "To transfer to server:"
echo "  rsync -avz $PACKAGE_DIR/ user@server:/path/to/project/"
echo ""
echo "Or create a tarball:"
echo "  tar -czf datacomp_mode.tar.gz $PACKAGE_DIR/"
echo "=========================================="
