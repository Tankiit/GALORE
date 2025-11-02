#!/bin/bash
# Quick Launch Script for Gradio Demo

set -e

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                            ║"
echo "║              🚀 GRADIO LARGE-SCALE RETRIEVAL DEMO LAUNCHER                ║"
echo "║                                                                            ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
MODE_DB="./large_db_mode_30k"
RANDOM_DB="./large_db_random_30k"
FULL_DB="./large_db_full_100k"
MODE_MODEL="./mode_output/final_clip_model.pt"
PORT=7860
SHARE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode_db)
            MODE_DB="$2"
            shift 2
            ;;
        --random_db)
            RANDOM_DB="$2"
            shift 2
            ;;
        --full_db)
            FULL_DB="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --share)
            SHARE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if at least one database exists
DB_FOUND=false

if [ -d "$MODE_DB" ]; then
    echo -e "${GREEN}✓${NC} Found MODE database: $MODE_DB"
    DB_FOUND=true
    MODE_ARG="--mode_db $MODE_DB"
else
    echo -e "${YELLOW}⚠${NC} MODE database not found: $MODE_DB"
    MODE_ARG=""
fi

if [ -d "$RANDOM_DB" ]; then
    echo -e "${GREEN}✓${NC} Found Random database: $RANDOM_DB"
    DB_FOUND=true
    RANDOM_ARG="--random_db $RANDOM_DB"
else
    echo -e "${YELLOW}⚠${NC} Random database not found: $RANDOM_DB"
    RANDOM_ARG=""
fi

if [ -d "$FULL_DB" ]; then
    echo -e "${GREEN}✓${NC} Found Full database: $FULL_DB"
    DB_FOUND=true
    FULL_ARG="--full_db $FULL_DB"
else
    echo -e "${YELLOW}⚠${NC} Full database not found: $FULL_DB"
    FULL_ARG=""
fi

if [ "$DB_FOUND" = false ]; then
    echo ""
    echo -e "${YELLOW}Error: No databases found!${NC}"
    echo ""
    echo "Please build databases first:"
    echo "  python build_large_retrieval_database.py --output_dir $MODE_DB --max_samples 30000"
    echo ""
    echo "Or specify custom paths:"
    echo "  ./launch_gradio_demo.sh --mode_db /path/to/db"
    echo ""
    exit 1
fi

# Check if model exists
if [ -f "$MODE_MODEL" ]; then
    echo -e "${GREEN}✓${NC} Found MODE model: $MODE_MODEL"
    MODEL_ARG="--mode_model $MODE_MODEL"
else
    echo -e "${YELLOW}⚠${NC} MODE model not found, using pretrained CLIP"
    MODEL_ARG=""
fi

# Check dependencies
echo ""
echo -e "${BLUE}Checking dependencies...${NC}"

python3 -c "import gradio" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}⚠ gradio not installed${NC}"
    echo "Installing gradio..."
    pip install gradio
fi

python3 -c "import transformers" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}⚠ transformers not installed${NC}"
    echo "Installing transformers..."
    pip install transformers
fi

echo -e "${GREEN}✓${NC} All dependencies ready"

# Build launch command
LAUNCH_CMD="python gradio_demo_largescale.py $MODE_ARG $RANDOM_ARG $FULL_ARG $MODEL_ARG --port $PORT"

if [ "$SHARE" = true ]; then
    LAUNCH_CMD="$LAUNCH_CMD --share"
    SHARE_MSG="Public link will be created (shareable)"
else
    SHARE_MSG="Local only (add --share for public link)"
fi

# Show launch info
echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                         LAUNCHING DEMO                                     ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "  Port:    $PORT"
echo "  Share:   $SHARE_MSG"
echo ""
echo "  Databases loaded:"
[ -n "$MODE_ARG" ] && echo "    • MODE"
[ -n "$RANDOM_ARG" ] && echo "    • Random"
[ -n "$FULL_ARG" ] && echo "    • Full"
echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                            ║"
echo "║  Demo will open at: http://localhost:$PORT                                 ║"
echo "║                                                                            ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Launch
$LAUNCH_CMD
