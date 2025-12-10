#!/bin/bash
# Quick training monitor for conditional GAN
#
# Usage:
#   bash check_status.sh                    # Single check
#   watch -n 10 bash check_status.sh        # Auto-refresh every 10 seconds
#
# Shows: Epochs completed, latest losses, D(real)/D(fake), health warnings, ETA

echo "=========================================="
echo "Conditional GAN Training Monitor"
echo "=========================================="
echo ""

# Find latest run
LATEST_RUN=$(ls -td results/conditional_wealthy_* 2>/dev/null | head -1)

if [ -z "$LATEST_RUN" ]; then
    echo "❌ No training runs found in results/"
    echo ""
    echo "Checking job queue:"
    squeue -u $USER
    echo ""
    echo "Recent SLURM logs:"
    ls -lht logs/train-*.out 2>/dev/null | head -3
    exit 0
fi

echo "📁 Latest run: $LATEST_RUN"
echo ""

# Check job status first
echo "💼 Job Status:"
RUNNING=$(squeue -u $USER -h -o "%j %T %M" 2>/dev/null | grep "cond_gan")
if [ -z "$RUNNING" ]; then
    echo "  ⚠️  No conditional GAN job running"
    echo "  (May have completed, failed, or not started yet)"
else
    echo "  ✅ $RUNNING"
fi
echo ""

# Check if metrics file exists
METRICS_FILE="$LATEST_RUN/metrics.csv"
if [ ! -f "$METRICS_FILE" ]; then
    echo "⏳ Training hasn't started yet (no metrics.csv)"
    echo ""
    echo "Checking SLURM output:"
    LATEST_LOG=$(ls -t logs/train-*.out 2>/dev/null | head -1)
    if [ -n "$LATEST_LOG" ]; then
        echo "Latest log: $LATEST_LOG"
        echo ""
        tail -20 "$LATEST_LOG"
    fi
    exit 0
fi

# Count epochs
EPOCHS_DONE=$(tail -n +2 "$METRICS_FILE" 2>/dev/null | wc -l | tr -d ' ')
echo "✅ Epochs completed: $EPOCHS_DONE / 100"

# Calculate progress percentage
if [ "$EPOCHS_DONE" -gt 0 ]; then
    PROGRESS=$((EPOCHS_DONE * 100 / 100))
    echo "📊 Progress: $PROGRESS%"
fi
echo ""

# Show latest 5 metrics
if [ "$EPOCHS_DONE" -gt 0 ]; then
    echo "📈 Latest 5 Epochs:"
    echo "-----------------------------------------------------"
    echo "Epoch | G Loss | D Loss | D(real) | D(fake) | Time(s)"
    echo "-----------------------------------------------------"
    tail -5 "$METRICS_FILE" | while IFS=',' read -r epoch g_loss d_loss d_real d_fake d_real_logit d_fake_logit time; do
        # Round time to integer
        time_int=$(echo "$time" | cut -d'.' -f1)
        printf "%5s | %6.3f | %6.3f | %7.3f | %7.3f | %7s\n" "$epoch" "$g_loss" "$d_loss" "$d_real" "$d_fake" "$time_int"
    done
    echo ""
fi

# Latest epoch details
if [ "$EPOCHS_DONE" -gt 0 ]; then
    LATEST=$(tail -1 "$METRICS_FILE")
    EPOCH=$(echo "$LATEST" | cut -d',' -f1)
    G_LOSS=$(echo "$LATEST" | cut -d',' -f2 | cut -c1-6)
    D_LOSS=$(echo "$LATEST" | cut -d',' -f3 | cut -c1-6)
    D_REAL=$(echo "$LATEST" | cut -d',' -f4 | cut -c1-5)
    D_FAKE=$(echo "$LATEST" | cut -d',' -f5 | cut -c1-5)
    TIME=$(echo "$LATEST" | cut -d',' -f8 | cut -d'.' -f1)

    echo "🎯 Current Status (Epoch $EPOCH):"
    echo "  Generator Loss:     $G_LOSS"
    echo "  Discriminator Loss: $D_LOSS"
    echo "  D(real):            $D_REAL"
    echo "  D(fake):            $D_FAKE"
    echo "  Epoch Time:         ${TIME}s"
    echo ""

    # Simple health check (works without bc)
    echo "🏥 Health Check:"

    # D(real) check - convert to integer by removing decimal
    D_REAL_INT=$(echo "$D_REAL" | tr -d '.' | cut -c1-2)
    if [ "$D_REAL_INT" -gt 95 ]; then
        echo "  ⚠️  D(real) too high ($D_REAL) - discriminator might be too strong"
    elif [ "$D_REAL_INT" -lt 40 ]; then
        echo "  ⚠️  D(real) too low ($D_REAL) - generator might be dominating"
    else
        echo "  ✅ D(real) looks good ($D_REAL)"
    fi

    # D(fake) check
    D_FAKE_INT=$(echo "$D_FAKE" | tr -d '.' | cut -c1-2)
    if [ "$D_FAKE_INT" -lt 5 ]; then
        echo "  ⚠️  D(fake) too low ($D_FAKE) - possible mode collapse"
    elif [ "$D_FAKE_INT" -gt 60 ]; then
        echo "  ⚠️  D(fake) too high ($D_FAKE) - discriminator might be too weak"
    else
        echo "  ✅ D(fake) looks good ($D_FAKE)"
    fi
    echo ""

    # Estimate time remaining
    REMAINING=$((100 - EPOCHS_DONE))
    if [ "$REMAINING" -gt 0 ]; then
        EST_SECONDS=$((TIME * REMAINING))
        EST_MINUTES=$((EST_SECONDS / 60))
        EST_HOURS=$((EST_MINUTES / 60))
        EST_MINS_REMAINING=$((EST_MINUTES % 60))

        echo "⏱️  Estimated time remaining:"
        echo "  $EST_MINUTES minutes (~${EST_HOURS}h ${EST_MINS_REMAINING}m)"
        echo ""
    fi
fi

# Show preview images
PREVIEW_COUNT=$(ls -1 "$LATEST_RUN"/preview_epoch_*.png 2>/dev/null | wc -l | tr -d ' ')
if [ "$PREVIEW_COUNT" -gt 0 ]; then
    echo "🖼️  Preview images: $PREVIEW_COUNT generated"
    ls -t "$LATEST_RUN"/preview_epoch_*.png 2>/dev/null | head -3 | while read img; do
        echo "  $(basename "$img")"
    done
    echo ""
fi

# Show training log tail
LOG_FILE="$LATEST_RUN"/training_log_*.txt
if [ -f $LOG_FILE ]; then
    echo "📝 Recent training output:"
    echo "-----------------------------------------------------"
    tail -5 $LOG_FILE | grep -v "^$"
    echo "-----------------------------------------------------"
    echo ""
fi

echo "=========================================="
echo "💡 Commands:"
echo "  Watch live:  watch -n 10 bash check_status.sh"
echo "  View log:    tail -f $LATEST_RUN/training_log_*.txt"
echo "  Cancel job:  scancel \$(squeue -u \$USER -h -o %i | head -1)"
echo "=========================================="

