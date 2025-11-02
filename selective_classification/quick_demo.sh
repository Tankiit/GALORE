#!/bin/bash
# Quick MODE Demo - Works immediately with your existing selections!
# No image download needed - uses embeddings only

set -e

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                            ║"
echo "║                  🚀 QUICK MODE DEMO (5 MINUTES)                           ║"
echo "║                                                                            ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if selected indices exist
if [ ! -f "datacomp_mode_cache/selected_indices.pt" ]; then
    echo "❌ Error: MODE selected indices not found!"
    echo "   Expected: datacomp_mode_cache/selected_indices.pt"
    exit 1
fi

echo "✓ Found MODE selected indices"
echo ""

# Step 1: Build embedding database
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 1/3: Building embedding database (~2 minutes)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ -f "demo_embedding_db/embedding_database.pt" ]; then
    echo "✓ Database already exists (skipping build)"
else
    python3 build_embedding_demo.py \
        --selected_indices ./datacomp_mode_cache/selected_indices.pt \
        --output_dir ./demo_embedding_db \
        --num_samples 300

    if [ $? -ne 0 ]; then
        echo "❌ Failed to build database"
        exit 1
    fi
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 2/3: Building random baseline for comparison (~2 minutes)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Create random baseline by building with random samples
if [ -f "demo_embedding_db_random/embedding_database.pt" ]; then
    echo "✓ Random database already exists (skipping build)"
else
    echo "Creating random selection..."
    python3 -c "
import torch
import numpy as np

# Create random indices
np.random.seed(123)
random_indices = np.random.choice(10000, 300, replace=False)
random_indices = torch.from_numpy(random_indices)

# Save as same format
data = {
    'indices': random_indices,
    'scores': torch.rand(300),
    'budget': 0.3
}

torch.save(data, 'datacomp_mode_cache/random_indices.pt')
print('✓ Created random indices')
"

    python3 build_embedding_demo.py \
        --selected_indices ./datacomp_mode_cache/random_indices.pt \
        --output_dir ./demo_embedding_db_random \
        --num_samples 300

    if [ $? -ne 0 ]; then
        echo "⚠️  Random database build failed (demo will show MODE only)"
    fi
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 3/3: Launching demo"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                            ║"
echo "║                     ✨ DEMO STARTING!                                     ║"
echo "║                                                                            ║"
echo "║              Opening at: http://localhost:7860                            ║"
echo "║                                                                            ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Build launch command
LAUNCH_CMD="python3 simple_embedding_demo.py --mode_db ./demo_embedding_db/embedding_database.pt"

if [ -f "demo_embedding_db_random/embedding_database.pt" ]; then
    LAUNCH_CMD="$LAUNCH_CMD --random_db ./demo_embedding_db_random/embedding_database.pt"
    echo "✓ Will show MODE vs Random comparison"
else
    echo "ℹ️  Will show MODE only (no comparison)"
fi

# Check for share flag
if [ "$1" == "--share" ]; then
    LAUNCH_CMD="$LAUNCH_CMD --share"
    echo "✓ Will create public share link"
fi

echo ""
echo "Try these queries:"
echo "  • a dog playing in a park"
echo "  • beautiful sunset over mountains"
echo "  • people playing soccer"
echo ""

# Launch
$LAUNCH_CMD
