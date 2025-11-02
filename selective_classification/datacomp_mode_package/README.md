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
