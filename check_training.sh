#!/bin/bash
# Simple training status checker for OSCAR terminal
# Usage: ./check_training.sh

echo "=========================================="
echo "🚀 Training Status Check"
echo "=========================================="
echo ""

# Check running jobs
echo "📋 Your Jobs:"
JOBS=$(squeue -u $USER -o "%.10i %.12j %.8T %.10M %.10L" 2>/dev/null)
if [ -z "$JOBS" ] || [ "$(echo "$JOBS" | wc -l)" -eq 1 ]; then
    echo "No jobs currently running"
    echo ""
    echo "Recent jobs:"
    sacct -u $USER --format=JobID%10,JobName%25,State%12,Elapsed%12 -n | head -3
else
    echo "$JOBS"
fi
echo ""

# Find latest training directory
LATEST=$(ls -td ~/urbanize/results/conditional_wealthy_* 2>/dev/null | head -1)

if [ -z "$LATEST" ]; then
    echo "❌ No training results found yet"
    exit 0
fi

echo "📂 Latest Training: $(basename $LATEST)"
echo ""

# Check metrics
if [ -f "$LATEST/metrics.csv" ]; then
    TOTAL=$(tail -n +2 "$LATEST/metrics.csv" | wc -l | tr -d ' ')

    if [ "$TOTAL" -eq 0 ]; then
        echo "⏳ Training starting..."
    else
        # Progress
        PCT=$((TOTAL * 100 / 100))
        FILLED=$((PCT / 5))
        BAR="["
        for i in {1..20}; do
            [ $i -le $FILLED ] && BAR="${BAR}█" || BAR="${BAR}░"
        done
        BAR="${BAR}]"

        echo "📊 Progress: $TOTAL/100 epochs ($PCT%)"
        echo "$BAR"
        echo ""

        # Latest metrics
        echo "📈 Latest Epoch:"
        tail -1 "$LATEST/metrics.csv" | awk -F',' '{printf "   Epoch %d: G_Loss=%.4f  D_Loss=%.4f  Time=%.1fs\n", $1, $2, $3, $4}'
        echo ""

        # ETA
        AVG=$(tail -n +2 "$LATEST/metrics.csv" | awk -F',' '{sum+=$4; n++} END {print sum/n}')
        REMAIN=$((100 - TOTAL))
        if [ "$REMAIN" -gt 0 ]; then
            ETA_MIN=$(echo "$AVG * $REMAIN / 60" | bc 2>/dev/null)
            ETA_HR=$(echo "$ETA_MIN / 60" | bc 2>/dev/null)
            echo "⏱️  ETA: ~${ETA_MIN} min (~${ETA_HR}h remaining)"
        else
            echo "✅ Training complete!"
        fi
    fi
else
    echo "⏳ Waiting for training to start..."
fi

echo ""
echo "=========================================="
echo "💡 Quick commands:"
echo "   Watch live:  tail -f $LATEST/training_log_*.txt"
echo "   Full monitor: ./monitor_job_14758669.sh"
echo "=========================================="

