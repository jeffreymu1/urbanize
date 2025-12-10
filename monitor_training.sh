#!/bin/bash
# Quick monitoring script for your conditional GAN training
# Usage: ./monitor_training.sh

echo "=========================================="
echo "Conditional GAN Training Monitor"
echo "=========================================="
echo ""

# Find latest run
LATEST_RUN=$(ls -td results/conditional_wealthy_* 2>/dev/null | head -1)

if [ -z "$LATEST_RUN" ]; then
    echo "❌ No training runs found in results/"
    echo "Have you started training yet?"
    exit 1
fi

echo "📁 Latest run: $LATEST_RUN"
echo ""

# Check if metrics file exists
METRICS_FILE="$LATEST_RUN/metrics.csv"
if [ ! -f "$METRICS_FILE" ]; then
    echo "⏳ Training hasn't started yet (no metrics.csv)"
    echo "Check SLURM logs:"
    ls -lht logs/train-*.out 2>/dev/null | head -5
    exit 0
fi

# Count epochs
EPOCHS_DONE=$(tail -n +2 "$METRICS_FILE" 2>/dev/null | wc -l | tr -d ' ')
echo "✅ Epochs completed: $EPOCHS_DONE / 100"
echo ""

# Show latest metrics
echo "📊 Latest 5 Epochs:"
echo "-------------------------------------------"
echo "Epoch | G Loss | D Loss | D(real) | D(fake)"
echo "-------------------------------------------"
tail -5 "$METRICS_FILE" | while IFS=',' read -r epoch g_loss d_loss d_real d_fake d_real_logit d_fake_logit time; do
    printf "%5s | %6.3f | %6.3f | %7.3f | %7.3f\n" "$epoch" "$g_loss" "$d_loss" "$d_real" "$d_fake"
done
echo ""

# Latest epoch details
LATEST=$(tail -1 "$METRICS_FILE")
EPOCH=$(echo "$LATEST" | cut -d',' -f1)
G_LOSS=$(echo "$LATEST" | cut -d',' -f2)
D_LOSS=$(echo "$LATEST" | cut -d',' -f3)
D_REAL=$(echo "$LATEST" | cut -d',' -f4)
D_FAKE=$(echo "$LATEST" | cut -d',' -f5)
TIME=$(echo "$LATEST" | cut -d',' -f8)

echo "🎯 Current Status (Epoch $EPOCH):"
echo "  Generator Loss:     $G_LOSS"
echo "  Discriminator Loss: $D_LOSS"
echo "  D(real):            $D_REAL"
echo "  D(fake):            $D_FAKE"
echo "  Epoch Time:         ${TIME}s"
echo ""

# Health check
echo "🏥 Health Check:"

# Check D(real)
D_REAL_FLOAT=$(echo "$D_REAL" | bc 2>/dev/null || echo "$D_REAL")
if (( $(echo "$D_REAL > 0.95" | bc -l 2>/dev/null || echo 0) )); then
    echo "  ⚠️  D(real) too high ($D_REAL) - discriminator might be too strong"
elif (( $(echo "$D_REAL < 0.4" | bc -l 2>/dev/null || echo 0) )); then
    echo "  ⚠️  D(real) too low ($D_REAL) - generator might be dominating"
else
    echo "  ✅ D(real) looks good ($D_REAL)"
fi

# Check D(fake)
if (( $(echo "$D_FAKE < 0.05" | bc -l 2>/dev/null || echo 0) )); then
    echo "  ⚠️  D(fake) too low ($D_FAKE) - possible mode collapse"
elif (( $(echo "$D_FAKE > 0.6" | bc -l 2>/dev/null || echo 0) )); then
    echo "  ⚠️  D(fake) too high ($D_FAKE) - discriminator might be too weak"
else
    echo "  ✅ D(fake) looks good ($D_FAKE)"
fi

# Estimate time remaining
if [ "$EPOCHS_DONE" -gt 0 ]; then
    REMAINING=$((100 - EPOCHS_DONE))
    EST_REMAINING=$(echo "$TIME * $REMAINING" | bc)
    EST_MINUTES=$(echo "$EST_REMAINING / 60" | bc)
    EST_HOURS=$(echo "scale=1; $EST_MINUTES / 60" | bc)
    echo ""
    echo "⏱️  Estimated time remaining: ${EST_MINUTES} minutes (~${EST_HOURS} hours)"
fi

# Show preview images
echo ""
echo "🖼️  Preview images:"
ls -t "$LATEST_RUN"/preview_epoch_*.png 2>/dev/null | head -3 | while read img; do
    echo "  $(basename "$img")"
done

# Check if job is still running
echo ""
echo "💼 Job Status:"
RUNNING=$(squeue -u $USER -h -o "%j %T" 2>/dev/null | grep "cond_gan_wealthy" | head -1)
if [ -z "$RUNNING" ]; then
    echo "  ⚠️  No job running (may have completed or failed)"
    echo "  Check logs: ls -lht logs/train-*.out | head -3"
else
    echo "  ✅ Job is running: $RUNNING"
fi

echo ""
echo "=========================================="
echo "💡 Tips:"
echo "  - Watch live: watch -n 10 ./monitor_training.sh"
echo "  - View log: tail -f $LATEST_RUN/training_log_*.txt"
echo "  - Cancel: scancel \$(squeue -u \$USER -h -o %i | head -1)"
echo "=========================================="

