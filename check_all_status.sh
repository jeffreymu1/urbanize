#!/bin/bash
# Universal GAN Training Monitor
# Works for: conditional_wealthy, 2attr_wealthy_lively, and other GAN jobs
#
# Usage:
#   bash check_all_status.sh                # Check all running GANs
#   bash check_all_status.sh wealthy        # Check only wealthy GAN
#   bash check_all_status.sh 2attr          # Check only 2-attr GAN
#   watch -n 10 bash check_all_status.sh    # Auto-refresh

FILTER="${1:-all}"

echo "=========================================="
echo "GAN Training Monitor"
echo "=========================================="
echo ""

# Function to check a specific GAN type
check_gan() {
    local pattern=$1
    local name=$2
    local job_pattern=$3
    local total_epochs=$4

    LATEST_RUN=$(ls -td results/${pattern}* 2>/dev/null | head -1)

    if [ -z "$LATEST_RUN" ]; then
        return
    fi

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 $name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📁 Latest run: $LATEST_RUN"
    echo ""

    # Check job status
    echo "💼 Job Status:"
    RUNNING=$(squeue -u $USER -h -o "%j %T %M" 2>/dev/null | grep "$job_pattern")
    if [ -z "$RUNNING" ]; then
        echo "  ⚠️  No job running (may be completed or failed)"
    else
        echo "  ✅ $RUNNING"
    fi
    echo ""

    # Check metrics
    METRICS_FILE="$LATEST_RUN/metrics.csv"
    if [ ! -f "$METRICS_FILE" ]; then
        echo "⏳ Training hasn't started yet (no metrics.csv)"
        echo ""
        return
    fi

    # Count epochs
    EPOCHS_DONE=$(tail -n +2 "$METRICS_FILE" 2>/dev/null | wc -l | tr -d ' ')
    echo "✅ Epochs completed: $EPOCHS_DONE / $total_epochs"

    if [ "$EPOCHS_DONE" -gt 0 ]; then
        PROGRESS=$((EPOCHS_DONE * 100 / total_epochs))
        echo "📊 Progress: $PROGRESS%"
    fi
    echo ""

    # Show latest 5 epochs
    if [ "$EPOCHS_DONE" -gt 0 ]; then
        echo "📈 Latest 5 Epochs:"
        echo "-----------------------------------------------------"
        echo "Epoch | G Loss | D Loss | D(real) | D(fake) | Time(s)"
        echo "-----------------------------------------------------"
        tail -6 "$METRICS_FILE" | tail -5 | while IFS=',' read -r epoch g_loss d_loss d_real_prob d_fake_prob d_real_logit d_fake_logit epoch_seconds; do
            if [ "$epoch" != "epoch" ]; then
                printf "%5s | %6.3f | %6.3f | %7.3f | %7.3f | %7s\n" \
                    "$epoch" "$g_loss" "$d_loss" "$d_real_prob" "$d_fake_prob" "${epoch_seconds%.*}"
            fi
        done
        echo ""

        # Current status
        LATEST=$(tail -1 "$METRICS_FILE")
        IFS=',' read -r epoch g_loss d_loss d_real_prob d_fake_prob d_real_logit d_fake_logit epoch_seconds <<< "$LATEST"

        echo "🎯 Current Status (Epoch $epoch):"
        echo "  Generator Loss:     $g_loss"
        echo "  Discriminator Loss: $d_loss"
        echo "  D(real):            $d_real_prob"
        echo "  D(fake):            $d_fake_prob"
        echo "  Epoch Time:         ${epoch_seconds%.*}s"
        echo ""

        # Health check
        echo "🏥 Health Check:"
        d_real_int=$(echo "$d_real_prob" | awk '{printf "%.0f", $1*100}')
        d_fake_int=$(echo "$d_fake_prob" | awk '{printf "%.0f", $1*100}')

        if [ "$d_real_int" -lt 60 ] || [ "$d_real_int" -gt 85 ]; then
            echo "  ⚠️  D(real) outside optimal range (0.60-0.85): $d_real_prob"
        else
            echo "  ✅ D(real) looks good ($d_real_prob)"
        fi

        if [ "$d_fake_int" -lt 15 ]; then
            echo "  ⚠️  D(fake) too low ($d_fake_prob) - possible mode collapse"
        elif [ "$d_fake_int" -gt 50 ]; then
            echo "  ⚠️  D(fake) too high ($d_fake_prob) - discriminator too weak"
        else
            echo "  ✅ D(fake) looks good ($d_fake_prob)"
        fi
        echo ""

        # ETA
        if [ "$EPOCHS_DONE" -gt 0 ] && [ "$EPOCHS_DONE" -lt "$total_epochs" ]; then
            TIME="${epoch_seconds%.*}"
            REMAINING=$((total_epochs - EPOCHS_DONE))
            EST_REMAINING=$((TIME * REMAINING))
            EST_MINUTES=$((EST_REMAINING / 60))
            EST_HOURS=$((EST_MINUTES / 60))
            EST_MINS_REM=$((EST_MINUTES % 60))

            echo "⏱️  Estimated time remaining:"
            echo "  $EST_MINUTES minutes (~${EST_HOURS}h ${EST_MINS_REM}m)"
            echo ""
        fi
    fi

    # Preview images
    PREVIEW_COUNT=$(ls -1 "$LATEST_RUN"/preview*.png 2>/dev/null | wc -l | tr -d ' ')
    if [ "$PREVIEW_COUNT" -gt 0 ]; then
        echo "🖼️  Preview images: $PREVIEW_COUNT generated"
        ls -t "$LATEST_RUN"/preview*.png 2>/dev/null | head -3 | while read img; do
            echo "  $(basename "$img")"
        done
        echo ""
    fi
}

# Check GANs based on filter
if [ "$FILTER" = "all" ] || [ "$FILTER" = "wealthy" ]; then
    check_gan "conditional_wealthy_" "Single-Attribute GAN (Wealthy, 128×128)" "cond_gan" 200
fi

if [ "$FILTER" = "all" ] || [ "$FILTER" = "2attr" ]; then
    check_gan "2attr_wealthy_lively_" "2-Attribute GAN (Wealthy+Lively, 64×64)" "2attr_gan" 150
fi

echo "=========================================="
echo "💡 Commands:"
echo "  Wealthy GAN only:    bash check_all_status.sh wealthy"
echo "  2-Attr GAN only:     bash check_all_status.sh 2attr"
echo "  Watch all:           watch -n 10 bash check_all_status.sh"
echo "  View training log:   tail -f results/*/training_log_*.txt"
echo "  Cancel job:          scancel JOBID"
echo "=========================================="

