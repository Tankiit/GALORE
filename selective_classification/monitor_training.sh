#!/bin/bash
# Monitor ongoing DataComp training

LOG_FILE=${1:-datacomp_100epochs.log}

echo "=========================================="
echo "DataComp Training Monitor"
echo "=========================================="
echo "Log file: $LOG_FILE"
echo ""

# Check if training is running
if pgrep -f "train_datacomp.py" > /dev/null; then
    echo "✓ Training process is RUNNING"
else
    echo "✗ Training process NOT found"
fi

echo ""
echo "Latest progress:"
echo "------------------------------------------"

# Show last 30 lines
tail -30 "$LOG_FILE"

echo ""
echo "------------------------------------------"
echo "To watch live:"
echo "  tail -f $LOG_FILE"
echo ""
echo "To see epoch summaries only:"
echo "  grep -E 'EPOCH|Summary|complete' $LOG_FILE"
echo ""
echo "To check specific metrics:"
echo "  grep 'Loss' $LOG_FILE | tail -10"
echo "=========================================="
