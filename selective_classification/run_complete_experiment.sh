#!/bin/bash
# Complete MODE Experiment Pipeline
# Runs: One-shot selection → Training → Evaluation → Visualization

set -e  # Exit on error

echo "=========================================="
echo "MODE HYPERNETWORK COMPLETE EXPERIMENT"
echo "=========================================="
echo ""

# Configuration
NUM_EPOCHS=10
BATCH_SIZE=256
OUTPUT_DIR="./mode_experiment_results"
SELECTION_RATIO=0.3

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Check if one-shot selection already done
echo -e "${BLUE}Step 1: Checking one-shot selection...${NC}"
if [ -f "datacomp_mode_cache/selected_indices.pt" ]; then
    echo -e "${GREEN}✓ Found existing selection (300 samples)${NC}"
    echo "  Location: datacomp_mode_cache/selected_indices.pt"
else
    echo -e "${YELLOW}⚠ No selection found. Running one-shot selection...${NC}"
    echo "  Note: This requires your dataset to be set up."
    echo "  If you haven't set up data yet, this will fail."
    echo ""
    read -p "Continue? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python mode_vlm_experiment.py --mode one_shot
    else
        echo "Exiting. Please run one-shot selection first."
        exit 1
    fi
fi
echo ""

# Step 2: Train with MODE
echo -e "${BLUE}Step 2: Training CLIP with MODE selection...${NC}"
echo "  Epochs: $NUM_EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Output: $OUTPUT_DIR"
echo ""

mkdir -p $OUTPUT_DIR

python train_mode_hypernetwork_zeroshot.py \
    --use_mode \
    --num_epochs $NUM_EPOCHS \
    --output_dir $OUTPUT_DIR \
    2>&1 | tee $OUTPUT_DIR/training.log

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Training completed successfully${NC}"
else
    echo -e "${YELLOW}⚠ Training failed or incomplete${NC}"
    echo "  Check logs: $OUTPUT_DIR/training.log"
fi
echo ""

# Step 3: Visualize results
echo -e "${BLUE}Step 3: Generating visualizations...${NC}"

python visualize_mode_training.py \
    --results_dir $OUTPUT_DIR \
    --output_dir $OUTPUT_DIR

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Visualizations generated${NC}"
    echo ""
    echo "  Generated files:"
    echo "    - $OUTPUT_DIR/convergence_curves.png"
    echo "    - $OUTPUT_DIR/strategy_evolution.png"
    echo "    - $OUTPUT_DIR/results_table.tex"
else
    echo -e "${YELLOW}⚠ Visualization failed${NC}"
fi
echo ""

# Step 4: Compare with baseline (optional)
echo -e "${BLUE}Step 4: (Optional) Compare with random baseline?${NC}"
read -p "Run random baseline comparison? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Training random baseline..."

    RANDOM_DIR="${OUTPUT_DIR}_random_baseline"
    mkdir -p $RANDOM_DIR

    python train_mode_hypernetwork_zeroshot.py \
        --use_mode=False \
        --num_epochs $NUM_EPOCHS \
        --output_dir $RANDOM_DIR \
        2>&1 | tee $RANDOM_DIR/training.log

    echo ""
    echo "Generating comparison plots..."
    python visualize_mode_training.py \
        --results_dir $OUTPUT_DIR \
        --compare_methods $RANDOM_DIR

    echo -e "${GREEN}✓ Comparison complete${NC}"
fi
echo ""

# Step 5: Summary
echo "=========================================="
echo "EXPERIMENT COMPLETE!"
echo "=========================================="
echo ""
echo "Results location: $OUTPUT_DIR"
echo ""
echo "Key files:"
echo "  1. Training history:      $OUTPUT_DIR/training_results.json"
echo "  2. Final model:           $OUTPUT_DIR/final_clip_model.pt"
echo "  3. Convergence curves:    $OUTPUT_DIR/convergence_curves.png"
echo "  4. Strategy evolution:    $OUTPUT_DIR/strategy_evolution.png"
echo "  5. Results table (LaTeX): $OUTPUT_DIR/results_table.tex"
echo ""
echo "Next steps:"
echo "  1. View plots: open $OUTPUT_DIR/*.png"
echo "  2. Check metrics: cat $OUTPUT_DIR/training_results.json | jq"
echo "  3. Use in paper: $OUTPUT_DIR/results_table.tex"
echo ""
echo "For paper writing:"
echo "  - Main results: Use convergence_curves.png"
echo "  - Curriculum learning: Use strategy_evolution.png"
echo "  - Tables: Use results_table.tex"
echo ""
echo -e "${GREEN}Good luck with your paper! 🚀${NC}"
echo ""
