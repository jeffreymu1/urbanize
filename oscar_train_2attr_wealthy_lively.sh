#!/bin/bash
#SBATCH --job-name=2attr_gan_wl
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/2attr_train-%j.out
#SBATCH --error=logs/2attr_train-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@brown.edu

# OSCAR 2-Attribute GAN Training Script (Wealthy + Lively)
# Uses same container and proven hyperparameters from successful 128x128 wealthy training

echo "=========================================="
echo "OSCAR 2-Attribute GAN Training"
echo "Attributes: Wealthy + Lively"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Clean environment
module purge
unset LD_LIBRARY_PATH

# Container setup
export APPTAINER_BINDPATH="/oscar/home/$USER,/oscar/scratch/$USER,/oscar/data,$PWD"
CONTAINER_PATH="/oscar/runtime/software/external/ngc-containers/tensorflow.d/x86_64.d/tensorflow-24.03-tf2-py3.simg"
EXEC="apptainer exec --nv"

echo "Container Information:"
echo "Path: $CONTAINER_PATH"
echo ""

# Check GPU
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
echo ""

# TensorFlow GPU check
echo "TensorFlow GPU Detection:"
$EXEC $CONTAINER_PATH python -c "
import tensorflow as tf
print('TensorFlow version:', tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print('GPUs detected:', len(gpus))
if len(gpus) > 0:
    print('✅ GPU DETECTED - Ready to train!')
else:
    print('❌ NO GPU DETECTED')
"
echo ""

# Install dependencies
echo "Installing dependencies..."
$EXEC $CONTAINER_PATH pip install --user --no-cache-dir pandas matplotlib scipy 2>&1 | grep -v "already satisfied" || true
echo "✅ Dependencies installed"
echo ""

# Create output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="results/2attr_wealthy_lively_${TIMESTAMP}"
mkdir -p logs
mkdir -p "${OUT_DIR}"

# Check if required files exist
echo "Checking required files..."
if [ ! -f "data/wealthy_lively/wealthy_lively_scores_train.csv" ]; then
    echo "❌ ERROR: data/wealthy_lively/wealthy_lively_scores_train.csv not found"
    echo ""
    echo "You need to upload the wealthy_lively dataset to OSCAR:"
    echo "  scp -r data/wealthy_lively \$USER@ssh.ccv.brown.edu:~/urbanize/data/"
    echo ""
    echo "Or create it on OSCAR:"
    echo "  python src/create_wealthy_lively_dataset.py"
    exit 1
fi

if [ ! -f "src/train_2attr_gan.py" ]; then
    echo "❌ ERROR: src/train_2attr_gan.py not found"
    exit 1
fi

if [ ! -d "data/preprocessed_images" ]; then
    echo "❌ ERROR: data/preprocessed_images/ not found"
    exit 1
fi

echo "✅ All required files found"
echo ""

echo "Configuration:"
echo "  Attributes: wealthy_score + lively_score"
echo "  Training images: 73,988"
echo "  Validation images: 9,248"
echo "  Epochs: 150"
echo "  Batch size: 128"
echo "  Image size: 64x64 (recommended for 2-attr stability)"
echo "  Output directory: ${OUT_DIR}"
echo ""

# Ensure FID stats are saved during training
FID_STATS_PATH="results/fid_real_stats_256.npz"

# Run training
echo "Starting 2-attribute GAN training (150 epochs at 64x64)..."
echo "Expected duration: ~4-5 hours"
echo "Note: 64x64 chosen for stability - can upscale to 128x128 after success"
echo "=========================================="
echo ""

# Set environment variables for better output
export TF_CPP_MIN_LOG_LEVEL=0  # Show all TensorFlow logs
export PYTHONUNBUFFERED=1       # Force unbuffered output

# Background progress monitor
(
  sleep 300
  while kill -0 $$ 2>/dev/null; do
    if [ -f "${OUT_DIR}/metrics.csv" ]; then
      echo ""
      echo "========================================"
      echo "PROGRESS UPDATE - $(date '+%H:%M:%S')"
      echo "========================================"

      EPOCHS_DONE=$(tail -n +2 "${OUT_DIR}/metrics.csv" 2>/dev/null | wc -l | tr -d ' ')
      if [ "$EPOCHS_DONE" -gt 0 ]; then
        echo "Epochs completed: $EPOCHS_DONE/150"
        PROGRESS_PCT=$((EPOCHS_DONE * 100 / 150))
        echo "Progress: $PROGRESS_PCT%"

        echo ""
        echo "Latest epoch metrics:"
        tail -1 "${OUT_DIR}/metrics.csv"

        # Check for mode collapse
        LATEST_FAKE=$(tail -1 "${OUT_DIR}/metrics.csv" | cut -d',' -f5)
        if [ -n "$LATEST_FAKE" ]; then
          FAKE_INT=$(echo "$LATEST_FAKE" | awk '{printf "%.0f", $1*100}')
          if [ "$FAKE_INT" -lt 15 ]; then
            echo ""
            echo "⚠️  WARNING: D(fake) may be too low - check for mode collapse"
          fi
        fi
      fi
      echo "========================================"
      echo ""
    fi
    sleep 600
  done
) &
MONITOR_PID=$!

$EXEC $CONTAINER_PATH python -u src/train_2attr_gan.py \
  --train_csv data/wealthy_lively/wealthy_lively_scores_train.csv \
  --val_csv data/wealthy_lively/wealthy_lively_scores_val.csv \
  --image_dir data/preprocessed_images \
  --image_size 64 \
  --latent_dim 128 \
  --batch_size 128 \
  --epochs 200 \
  --lr 0.0002 \
  --beta1 0.5 \
  --save_every 10 \
  --preview_every 10 \
  --fid_every 5 \
  --fid_num_samples 512 \
  --fid_batch_size 32 \
  --fid_stats_path "$FID_STATS_PATH" \
  --out_dir "$OUT_DIR" \
  2>&1 | tee "$OUT_DIR/training_log_${TIMESTAMP}.txt"

EXIT_CODE=$?

# Kill monitor
kill $MONITOR_PID 2>/dev/null || true

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ TRAINING COMPLETED SUCCESSFULLY!"
    echo ""
    echo "Results saved to: ${OUT_DIR}"
    echo ""
    echo "Files generated:"
    echo "  - checkpoints/ (model weights)"
    echo "  - preview_2attr_epoch_*.png (2D attribute grids)"
    echo "  - metrics.csv (training history)"
    echo "  - training_log_${TIMESTAMP}.txt (full logs)"
    echo ""
    echo "Preview grids show:"
    echo "  - Rows: wealthy (0.0 → 1.0)"
    echo "  - Cols: lively (0.0 → 1.0)"
    echo ""
    echo "To copy results to local machine:"
    echo "  scp -r $USER@ssh.ccv.brown.edu:~/urbanize/${OUT_DIR} ."
else
    echo "❌ TRAINING FAILED"
    echo "Exit code: ${EXIT_CODE}"
    echo "Check logs/2attr_train-${SLURM_JOB_ID}.err for errors"
fi
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE
