#!/bin/bash
#SBATCH --job-name=multi_attr_gan
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/multi_attr-%j.out
#SBATCH --error=logs/multi_attr-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@brown.edu

# OSCAR Training Script for Multi-Attribute GAN Using TensorFlow Container

echo "=========================================="
echo "OSCAR Multi-Attribute GAN Training"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Clean environment for container
module purge
unset LD_LIBRARY_PATH

# Set up container
export APPTAINER_BINDPATH="/oscar/home/$USER,/oscar/scratch/$USER,/oscar/data,$PWD"
CONTAINER_PATH="/oscar/runtime/software/external/ngc-containers/tensorflow.d/x86_64.d/tensorflow-24.03-tf2-py3.simg"
EXEC="apptainer exec --nv"

echo "Container Information:"
echo "Path: $CONTAINER_PATH"
echo "Bind paths: $APPTAINER_BINDPATH"
echo ""

# Check GPU on host
echo "GPU Information (host):"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
echo ""

# Check GPU in container
echo "GPU Information (container):"
$EXEC $CONTAINER_PATH nvidia-smi -L
echo ""

# Check TensorFlow GPU detection
echo "TensorFlow GPU Detection:"
$EXEC $CONTAINER_PATH python -c "
import tensorflow as tf
print('TensorFlow version:', tf.__version__)
print('Built with CUDA:', tf.test.is_built_with_cuda())
gpus = tf.config.list_physical_devices('GPU')
print('GPUs detected:', len(gpus))
if len(gpus) > 0:
    print('✅ GPU DETECTED - Ready to train!')
    for gpu in gpus:
        print(f'  - {gpu}')
else:
    print('❌ NO GPU DETECTED')
"
echo ""

# Install dependencies in container
echo "Installing dependencies..."
$EXEC $CONTAINER_PATH pip install --user --no-cache-dir pandas matplotlib scipy 2>&1 | grep -v "already satisfied" || true
echo "Dependencies installed"
echo ""

# Create output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="results/multi_attribute_gan_${TIMESTAMP}"
mkdir -p logs
mkdir -p "${OUT_DIR}"

echo "Configuration:"
echo "  Attributes: wealthy, depressing, safety, lively, boring, beautiful (6 total)"
echo "  Training images: ~89,000"
echo "  Validation images: ~9,900"
echo "  Epochs: 100"
echo "  Batch size: 128"
echo "  Latent dim: 256 (increased for better diversity)"
echo "  Image size: 64x64"
echo "  Input dimension: 256 (latent) + 6 (attributes) = 262"
echo "  Output directory: ${OUT_DIR}"
echo ""

# Run training
echo "Starting multi-attribute training (100 epochs)..."
echo "Expected duration: ~2-3 hours (similar to single-attribute)"
echo "=========================================="
echo ""

# Create a background process to show periodic progress
(
  sleep 300  # Wait 5 minutes before first update
  while kill -0 $$ 2>/dev/null; do
    if [ -f "${OUT_DIR}/metrics.csv" ]; then
      echo ""
      echo "========================================"
      echo "PROGRESS UPDATE - $(date '+%H:%M:%S')"
      echo "========================================"
      EPOCHS_DONE=$(tail -n +2 "${OUT_DIR}/metrics.csv" 2>/dev/null | wc -l | tr -d ' ')
      if [ "$EPOCHS_DONE" -gt 0 ]; then
        echo "Epochs completed: $EPOCHS_DONE/100"
        LATEST=$(tail -n 1 "${OUT_DIR}/metrics.csv" 2>/dev/null)
        echo "Latest metrics: $LATEST"
        PROGRESS_PCT=$((EPOCHS_DONE * 100 / 100))
        echo "Progress: $PROGRESS_PCT%"

        # Show last few lines of log
        echo ""
        echo "Recent training output:"
        tail -n 8 "${OUT_DIR}/training_log_${TIMESTAMP}.txt" 2>/dev/null | head -n 8
      fi
      echo "========================================"
      echo ""
    fi
    sleep 600  # Update every 10 minutes
  done
) &
MONITOR_PID=$!

$EXEC $CONTAINER_PATH python -u src/train_multi_attribute_gan.py \
  --train_csv data/all_attribute_scores_train.csv \
  --val_csv data/all_attribute_scores_val.csv \
  --image_dir data/preprocessed_images \
  --attribute_names wealthy depressing safety lively boring beautiful \
  --epochs 100 \
  --batch_size 128 \
  --latent_dim 256 \
  --image_size 64 \
  --lr 0.0002 \
  --beta1 0.5 \
  --save_every 10 \
  --preview_every 10 \
  --out_dir "${OUT_DIR}" \
  2>&1 | tee "${OUT_DIR}/training_log_${TIMESTAMP}.txt"

EXIT_CODE=$?

# Kill the monitor background process
kill $MONITOR_PID 2>/dev/null || true

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ MULTI-ATTRIBUTE TRAINING COMPLETED SUCCESSFULLY!"
    echo ""
    echo "Results saved to: ${OUT_DIR}"
    echo ""
    echo "Files generated:"
    echo "  - checkpoints/ckpt-* (model weights)"
    echo "  - preview_epoch_*.png (generated image samples showing attribute variations)"
    echo "  - metrics.csv (loss history)"
    echo "  - training_log_${TIMESTAMP}.txt (full logs)"
    echo ""
    echo "Next steps:"
    echo "1. Copy results to local machine:"
    echo "   scp -r $USER@ssh.ccv.brown.edu:~/urbanize/${OUT_DIR} ."
    echo ""
    echo "2. Generate images with your trained model:"
    echo "   python src/generate_multi_attribute.py --checkpoint_dir ${OUT_DIR}/checkpoints/"
    echo ""
    echo "3. Launch interactive demo with 5 sliders:"
    echo "   python src/interactive_multi_attribute_demo.py --checkpoint_dir ${OUT_DIR}/checkpoints/"
else
    echo "❌ TRAINING FAILED"
    echo "Exit code: ${EXIT_CODE}"
    echo "Check logs/multi_attr-${SLURM_JOB_ID}.err for error details"
    echo "Check ${OUT_DIR}/training_log_${TIMESTAMP}.txt for full output"
fi
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE

