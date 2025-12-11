#!/bin/bash
#SBATCH --job-name=multi_attr_gan
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/multi_train-%j.out
#SBATCH --error=logs/multi_train-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@brown.edu

# OSCAR Multi-Attribute GAN Training Script
# 6 Attributes: wealthy, depressing, safety, lively, boring, beautiful

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
EXIT_CODE=$?

# Kill the monitor background process
kill $MONITOR_PID 2>/dev/null || true

$EXEC $CONTAINER_PATH pip install --user --no-cache-dir pandas matplotlib 2>&1 | grep -v "already satisfied" || true
echo "Dependencies installed"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ TRAINING COMPLETED SUCCESSFULLY!"
    echo ""
    echo "Results saved to: ${OUT_DIR}"
    echo ""
    echo "Files generated:"
    echo "  - ckpt/ (model checkpoints)"
    echo "  - samples/epoch_*.png (generated samples)"
    echo "  - sweeps/epoch_*_sweep_*.png (attribute sweeps)"
    echo "  - cond_meta.json (attribute metadata)"
    echo "  - training_log_${TIMESTAMP}.txt (full logs)"
    echo ""
    echo "Key metrics to check:"
    echo "  - D(real) should be 0.6-0.8"
    echo "  - D(fake) should be 0.3-0.5"
    echo "  - Look for mode collapse warnings"
    echo ""
    echo "View sweep images to verify attribute control:"
    echo "  ls ${OUT_DIR}/sweeps/epoch_0100_*.png"
    echo ""
    echo "To copy results to local machine:"
    echo "  scp -r $USER@ssh.ccv.brown.edu:~/urbanize/${OUT_DIR} ."
else
    echo "❌ TRAINING FAILED"
    echo "Exit code: ${EXIT_CODE}"
    echo "Check logs/multi_train-${SLURM_JOB_ID}.err for error details"
    echo "Check ${OUT_DIR}/training_log_${TIMESTAMP}.txt for full output"
fi
echo "End time: $(date)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="results/multi_attr_gan_${TIMESTAMP}"
exit $EXIT_CODE

mkdir -p logs
mkdir -p "${OUT_DIR}"

echo "Configuration:"
echo "  Attributes: 6 (wealthy, depressing, safety, lively, boring, beautiful)"
echo "  Training images: 89,013"
echo "  Image size: 64x64"
echo "  Batch size: 64"
echo "  Epochs: 100"
echo "  Latent dim: 128"
echo "  Output directory: ${OUT_DIR}"
echo ""

# Run training
echo "Starting full training (100 epochs)..."
echo "Expected duration: ~6-8 hours"
echo "=========================================="
echo ""

# Create a background process to show periodic progress
(
  sleep 300  # Wait 5 minutes before first update
  while kill -0 $$ 2>/dev/null; do
    if [ -f "${OUT_DIR}/training_log_${TIMESTAMP}.txt" ]; then
      echo ""
      echo "========================================"
      echo "PROGRESS UPDATE - $(date '+%H:%M:%S')"
      echo "========================================"

      # Count epochs by looking for "[train] epoch=" lines
      EPOCHS_DONE=$(grep -c "\[train\] epoch=" "${OUT_DIR}/training_log_${TIMESTAMP}.txt" 2>/dev/null || echo "0")
      if [ "$EPOCHS_DONE" -gt 0 ]; then
        echo "Epochs completed: $EPOCHS_DONE/100"
        PROGRESS_PCT=$((EPOCHS_DONE * 100 / 100))
        echo "Progress: $PROGRESS_PCT%"

        # Show last training epoch
        echo ""
        echo "Latest epoch metrics:"
        grep "\[train\] epoch=" "${OUT_DIR}/training_log_${TIMESTAMP}.txt" 2>/dev/null | tail -1

        # Check for mode collapse warnings
        if grep -q "WARNING.*mode collapse" "${OUT_DIR}/training_log_${TIMESTAMP}.txt" 2>/dev/null; then
          echo ""
          echo "⚠️  MODE COLLAPSE WARNING DETECTED - Check training!"
        fi
      fi
      echo "========================================"
      echo ""
    fi
    sleep 600  # Update every 10 minutes
  done
) &
MONITOR_PID=$!

$EXEC $CONTAINER_PATH python -u src/multi_model_files/multi_model.py \
  --csv_path data/all_attribute_scores/all_attribute_scores_train.csv \
  --img_dir data/preprocessed_images \
  --out_dir "${OUT_DIR}" \
  --image_size 64 \
  --batch_size 64 \
  --epochs 100 \
  --latent_dim 128 \
  --lr_g 0.0002 \
  --lr_d 0.0002 \
  --lambda_reg 10.0 \
  --augment \
  --save_every_epochs 10 \
  --num_eval_samples 64 \
  2>&1 | tee "${OUT_DIR}/training_log_${TIMESTAMP}.txt"

echo ""
echo "=========================================="
echo "Training Complete!"
echo "End: $(date)"
echo "Results: ${OUT_DIR}"
echo "=========================================="

