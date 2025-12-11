#!/bin/bash
# Monitor multi-attribute GAN training on OSCAR
# Usage: bash check_multi_status.sh

cd "$(dirname "$0")"

echo "=========================================="
echo "Multi-Attribute GAN Training Monitor"
echo "=========================================="
echo ""

# Find latest run
LATEST_RUN=$(ls -td results/multi_attr_gan_* 2>/dev/null | head -1)

if [ -z "$LATEST_RUN" ]; then
    echo "❌ No multi-attribute training runs found in results/"
    echo ""
    echo "Checking job queue:"
    squeue -u $USER | grep multi
    echo ""
    echo "Recent SLURM logs:"
    ls -lht logs/multi_train-*.out 2>/dev/null | head -3
    exit 0
fi

echo "📁 Latest run: $LATEST_RUN"
echo ""

# Check job status first
echo "💼 Job Status:"
RUNNING=$(squeue -u $USER -h -o "%j %T %M" 2>/dev/null | grep "multi_attr")
if [ -z "$RUNNING" ]; then
    echo "  ⚠️  No multi-attribute GAN job running"
    echo "  (May have completed, failed, or not started yet)"
else
    echo "  ✅ $RUNNING"
fi
echo ""

# Check if training log exists
LOG_FILE="$LATEST_RUN"/training_log_*.txt
if [ ! -f $LOG_FILE ]; then
    echo "⏳ Training hasn't started yet (no training log)"
    echo ""
    echo "Checking SLURM output:"
    LATEST_LOG=$(ls -t logs/multi_train-*.out 2>/dev/null | head -1)
    if [ -n "$LATEST_LOG" ]; then
        echo "Latest log: $LATEST_LOG"
        echo ""
        tail -30 "$LATEST_LOG"
    fi
    exit 0
fi

# Count epochs
EPOCHS_DONE=$(grep -c "\[train\] epoch=" $LOG_FILE 2>/dev/null || echo "0")
echo "✅ Epochs completed: $EPOCHS_DONE / 100"

# Calculate progress percentage
if [ "$EPOCHS_DONE" -gt 0 ]; then
    PROGRESS=$((EPOCHS_DONE * 100 / 100))
    echo "📊 Progress: $PROGRESS%"
fi
echo ""

# Show latest 5 epochs with metrics
if [ "$EPOCHS_DONE" -gt 0 ]; then
    echo "📈 Latest 5 Epochs:"
    echo "-------------------------------------------------------------------------"
    echo "Epoch | D_adv | D_reg | G_adv | G_reg | D(real) | D(fake) | Status"
    echo "-------------------------------------------------------------------------"
    grep "\[train\] epoch=" $LOG_FILE | tail -5 | while read line; do
        # Extract epoch number
        EPOCH=$(echo "$line" | grep -o "epoch=[0-9]*" | cut -d'=' -f2)
        # Extract metrics
        D_ADV=$(echo "$line" | grep -o "D_adv=[0-9.]*" | cut -d'=' -f2)
        D_REG=$(echo "$line" | grep -o "D_reg=[0-9.]*" | cut -d'=' -f2)
        G_ADV=$(echo "$line" | grep -o "G_adv=[0-9.]*" | cut -d'=' -f2)
        G_REG=$(echo "$line" | grep -o "G_reg=[0-9.]*" | cut -d'=' -f2)
        D_REAL=$(echo "$line" | grep -o "D(real)=[0-9.]*" | cut -d'=' -f2)
        D_FAKE=$(echo "$line" | grep -o "D(fake)=[0-9.]*" | cut -d'=' -f2)

        # Health check
        STATUS="✅"
        if [ -n "$D_REAL" ] && [ -n "$D_FAKE" ]; then
            # Simple integer comparison (multiply by 100)
            D_REAL_INT=$(echo "$D_REAL" | awk '{printf "%.0f", $1*100}')
            D_FAKE_INT=$(echo "$D_FAKE" | awk '{printf "%.0f", $1*100}')

            if [ "$D_REAL_INT" -gt 95 ] && [ "$D_FAKE_INT" -lt 5 ]; then
                STATUS="⚠️"
            fi
        fi

        printf "%5s | %5s | %5s | %5s | %5s | %7s | %7s | %s\n" \
            "$EPOCH" "$D_ADV" "$D_REG" "$G_ADV" "$G_REG" "$D_REAL" "$D_FAKE" "$STATUS"
    done
    echo ""
fi

# Latest epoch details
if [ "$EPOCHS_DONE" -gt 0 ]; then
    LATEST=$(grep "\[train\] epoch=" $LOG_FILE | tail -1)

    echo "🎯 Current Status:"
    echo "$LATEST"
    echo ""
fi

# Check for mode collapse warnings
WARNINGS=$(grep -c "WARNING.*mode collapse" $LOG_FILE 2>/dev/null || echo "0")
if [ "$WARNINGS" -gt 0 ]; then
    echo "⚠️  Mode Collapse Warnings: $WARNINGS"
    echo "Latest warning:"
    grep "WARNING.*mode collapse" $LOG_FILE | tail -1
    echo ""
fi

# Check for evaluation outputs
EVAL_COUNT=$(grep -c "\[eval\]" $LOG_FILE 2>/dev/null || echo "0")
if [ "$EVAL_COUNT" -gt 0 ]; then
    echo "📊 Latest Evaluation:"
    grep "\[eval\]" $LOG_FILE | tail -1
    echo ""
fi

# Show generated files
SAMPLE_COUNT=$(ls -1 "$LATEST_RUN"/samples/*.png 2>/dev/null | wc -l | tr -d ' ')
SWEEP_COUNT=$(ls -1 "$LATEST_RUN"/sweeps/*.png 2>/dev/null | wc -l | tr -d ' ')
CKPT_COUNT=$(ls -1 "$LATEST_RUN"/ckpt/ckpt-*.index 2>/dev/null | wc -l | tr -d ' ')

if [ "$SAMPLE_COUNT" -gt 0 ] || [ "$SWEEP_COUNT" -gt 0 ] || [ "$CKPT_COUNT" -gt 0 ]; then
    echo "🖼️  Generated Files:"
    echo "  Sample images: $SAMPLE_COUNT"
    echo "  Sweep images: $SWEEP_COUNT"
    echo "  Checkpoints: $CKPT_COUNT"
    echo ""

    if [ "$SWEEP_COUNT" -gt 0 ]; then
        echo "  Latest sweeps:"
        ls -t "$LATEST_RUN"/sweeps/*.png 2>/dev/null | head -6 | while read img; do
            echo "    $(basename "$img")"
        done
        echo ""
    fi
fi

# Estimate time remaining
if [ "$EPOCHS_DONE" -gt 0 ] && [ "$EPOCHS_DONE" -lt 100 ]; then
    # Estimate based on recent epoch time (assume ~200s per epoch)
    REMAINING=$((100 - EPOCHS_DONE))
    EST_SECONDS=$((REMAINING * 200))
    EST_MINUTES=$((EST_SECONDS / 60))
    EST_HOURS=$((EST_MINUTES / 60))
    EST_MINS_REMAINING=$((EST_MINUTES % 60))

    echo "⏱️  Estimated time remaining:"
    echo "  $EST_MINUTES minutes (~${EST_HOURS}h ${EST_MINS_REMAINING}m)"
    echo ""
fi

echo "=========================================="
echo "💡 Commands:"
echo "  Watch live:  watch -n 30 bash check_multi_status.sh"
echo "  View log:    tail -f $LATEST_RUN/training_log_*.txt"
echo "  View sweeps: ls $LATEST_RUN/sweeps/"
echo "  Cancel job:  scancel \$(squeue -u \$USER -h -o %i | grep multi | head -1)"
echo "=========================================="

