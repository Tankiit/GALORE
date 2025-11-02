#!/bin/bash
# Complete Demo Setup Script
# Guides you through downloading samples and building demo

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                            ║"
echo "║              🎯 MODE DEMO - COMPLETE SETUP WIZARD                         ║"
echo "║                                                                            ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Configuration
SELECTED_INDICES="./datacomp_mode_cache/selected_indices.pt"
DATA_DIR="./datacomp_data"
NUM_SAMPLES=30000

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --quick)
            NUM_SAMPLES=1000
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--num_samples N] [--data_dir PATH] [--quick]"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}Setup Configuration:${NC}"
echo "  Selected indices: $SELECTED_INDICES"
echo "  Data directory: $DATA_DIR"
echo "  Number of samples: $NUM_SAMPLES"
echo ""

# Check if indices exist
if [ ! -f "$SELECTED_INDICES" ]; then
    echo -e "${RED}✗ Error: Selected indices not found!${NC}"
    echo "  Expected: $SELECTED_INDICES"
    echo ""
    echo "Please run MODE selection first or specify correct path."
    exit 1
fi

echo -e "${GREEN}✓${NC} Found selected indices"
echo ""

# Check if data already exists
if [ -d "$DATA_DIR/images" ] && [ -f "$DATA_DIR/metadata.json" ]; then
    echo -e "${YELLOW}⚠ Data directory already exists!${NC}"
    echo "  Location: $DATA_DIR"
    echo ""
    echo "Options:"
    echo "  1) Use existing data (skip download)"
    echo "  2) Re-download (will overwrite)"
    echo "  3) Use different directory"
    echo ""
    read -p "Enter choice (1-3): " choice

    case $choice in
        1)
            echo "Using existing data..."
            SKIP_DOWNLOAD=true
            ;;
        2)
            echo "Will re-download..."
            SKIP_DOWNLOAD=false
            ;;
        3)
            read -p "Enter new directory path: " DATA_DIR
            SKIP_DOWNLOAD=false
            ;;
        *)
            echo "Invalid choice. Exiting."
            exit 1
            ;;
    esac
else
    SKIP_DOWNLOAD=false
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                         SETUP STEPS                                        ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Download samples
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo -e "${BLUE}STEP 1: Downloading MODE-Selected Samples${NC}"
    echo "  This will download $NUM_SAMPLES images from DataComp"
    echo "  Expected time: ~10-180 minutes (depends on number and network)"
    echo "  Disk space needed: ~$((NUM_SAMPLES * 50 / 1024)) MB"
    echo ""

    read -p "Continue with download? (y/n): " confirm
    if [ "$confirm" != "y" ]; then
        echo "Setup cancelled."
        exit 0
    fi

    echo ""
    echo "Starting download..."
    python3 download_mode_samples.py \
        --selected_indices "$SELECTED_INDICES" \
        --output_dir "$DATA_DIR" \
        --num_samples "$NUM_SAMPLES" \
        --cache_dir ./datacomp_cache

    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ Download failed!${NC}"
        exit 1
    fi

    echo -e "${GREEN}✓ Download complete!${NC}"
else
    echo -e "${BLUE}STEP 1: Using Existing Data${NC}"
    echo -e "${GREEN}✓ Skipped download${NC}"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                            ║"
echo "║                  📊 DATA READY! NEXT: BUILD DATABASES                     ║"
echo "║                                                                            ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Ask about building databases
echo "Would you like to build the retrieval databases now?"
echo "This will create MODE, Random, and Full databases for comparison demo."
echo ""
echo "Options:"
echo "  1) Build small databases (1K MODE, 1K Random, 3K Full) - Quick test (~5 min)"
echo "  2) Build medium databases (10K MODE, 10K Random, 30K Full) - (~30 min)"
echo "  3) Build full databases (30K MODE, 30K Random, 100K Full) - Full demo (~4 hours)"
echo "  4) Skip for now (build manually later)"
echo ""

read -p "Enter choice (1-4): " build_choice

case $build_choice in
    1)
        echo ""
        echo "Building SMALL databases for quick testing..."
        ./build_demo_pipeline.sh \
            --data_dir "$DATA_DIR" \
            --mode_samples 1000 \
            --random_samples 1000 \
            --full_samples 3000 \
            --output_dir ./demo_databases_small

        DEMO_DIR="./demo_databases_small"
        ;;
    2)
        echo ""
        echo "Building MEDIUM databases..."
        ./build_demo_pipeline.sh \
            --data_dir "$DATA_DIR" \
            --mode_samples 10000 \
            --random_samples 10000 \
            --full_samples 30000 \
            --output_dir ./demo_databases_medium

        DEMO_DIR="./demo_databases_medium"
        ;;
    3)
        echo ""
        echo "Building FULL databases (this will take ~4 hours)..."
        ./build_demo_pipeline.sh \
            --data_dir "$DATA_DIR" \
            --mode_samples 30000 \
            --random_samples 30000 \
            --full_samples 100000 \
            --output_dir ./demo_databases

        DEMO_DIR="./demo_databases"
        ;;
    4)
        echo ""
        echo "Skipping database build."
        echo ""
        echo "To build later, run:"
        echo "  ./build_demo_pipeline.sh --data_dir $DATA_DIR"
        echo ""
        exit 0
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

# Check if build succeeded
if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Database build failed!${NC}"
    exit 1
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                            ║"
echo "║                      🎉 SETUP COMPLETE!                                   ║"
echo "║                                                                            ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Your demo is ready to launch!"
echo ""
echo "Quick Launch:"
echo "  cd $DEMO_DIR"
echo "  ./launch_demo.sh"
echo ""
echo "Or with public share link:"
echo "  cd $DEMO_DIR"
echo "  ./launch_demo.sh --share"
echo ""
echo "Manual launch:"
echo "  python gradio_demo_largescale.py \\"
echo "      --mode_db $DEMO_DIR/mode_* \\"
echo "      --random_db $DEMO_DIR/random_* \\"
echo "      --full_db $DEMO_DIR/full_*"
echo ""
echo "Demo will open at: http://localhost:7860"
echo ""
echo "Enjoy your MODE retrieval demo! 🚀"
echo ""
