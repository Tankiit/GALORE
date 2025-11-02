#!/bin/bash
# Compare Fixed LR vs Adaptive Scheduler runs

echo "========================================================================"
echo "DataComp Training Comparison: Fixed LR vs Adaptive Scheduler"
echo "========================================================================"
echo ""

# Check if new run is still running
if pgrep -f "datacomp_100epochs_with_scheduler" > /dev/null; then
    NEW_STATUS="🟢 RUNNING"
else
    NEW_STATUS="✅ COMPLETE"
fi

echo "Status:"
echo "  Old run (Fixed LR):      ✅ COMPLETE"
echo "  New run (Adaptive LR):   $NEW_STATUS"
echo ""

echo "========================================================================"
echo "📊 Results Comparison"
echo "========================================================================"
echo ""

echo "--- EPOCH 1 ---"
echo "Fixed LR:"
grep -A 2 "Epoch 1 Summary" datacomp_100epochs.log 2>/dev/null || echo "  (not found)"
echo ""
echo "Adaptive LR:"
grep -A 2 "Epoch 1 Summary" datacomp_100epochs_with_scheduler.log 2>/dev/null || echo "  (training not started yet)"
echo ""

echo "--- EPOCH 25 ---"
echo "Fixed LR:"
grep -A 2 "Epoch 25 Summary" datacomp_100epochs.log 2>/dev/null || echo "  (not found)"
echo ""
echo "Adaptive LR:"
grep -A 2 "Epoch 25 Summary" datacomp_100epochs_with_scheduler.log 2>/dev/null || echo "  (not reached yet)"
echo ""

echo "--- EPOCH 50 ---"
echo "Fixed LR:"
grep -A 2 "Epoch 50 Summary" datacomp_100epochs.log 2>/dev/null || echo "  (not found)"
echo ""
echo "Adaptive LR:"
grep -A 2 "Epoch 50 Summary" datacomp_100epochs_with_scheduler.log 2>/dev/null || echo "  (not reached yet)"
echo ""

echo "--- EPOCH 100 (FINAL) ---"
echo "Fixed LR:"
grep -A 2 "Epoch 100 Summary" datacomp_100epochs.log 2>/dev/null || echo "  (not found)"
echo ""
echo "Adaptive LR:"
grep -A 2 "Epoch 100 Summary" datacomp_100epochs_with_scheduler.log 2>/dev/null || echo "  (not reached yet)"
echo ""

echo "========================================================================"
echo "📈 Learning Rate Schedule"
echo "========================================================================"
echo ""
echo "Fixed LR (first 10 steps):"
grep "lr=" datacomp_100epochs.log | head -10 | awk '{print $NF}'
echo ""
echo "Adaptive LR (first 10 steps):"
grep "lr=" datacomp_100epochs_with_scheduler.log | head -10 | awk '{print $NF}'
echo ""

echo "========================================================================"
echo "💡 Quick Commands"
echo "========================================================================"
echo ""
echo "Watch new run live:"
echo "  tail -f datacomp_100epochs_with_scheduler.log"
echo ""
echo "Check current progress:"
echo "  grep 'EPOCH' datacomp_100epochs_with_scheduler.log | tail -1"
echo ""
echo "View this comparison anytime:"
echo "  ./compare_runs.sh"
echo ""
echo "========================================================================"
