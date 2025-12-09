#!/bin/bash
#SBATCH --job-name=cond_gan_wealthy_container
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/container-slurm-%j.out
#SBATCH --error=logs/container-slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@brown.edu

# OSCAR SLURM script for training Conditional GAN inside a container (optional)
# Usage:
#   sbatch oscar_train_conditional_container.sh
# Environment:
#   If CONTAINER_IMAGE is set to a valid Apptainer/Singularity image path, the job
#   will execute the training inside that container using --nv for GPU support.
#   Otherwise it falls back to running the training directly on the node.

set -euo pipefail

echo "=========================================="
echo "OSCAR Conditional GAN Training (container wrapper)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Print SLURM GPU allocation info
echo "SLURM GPU Allocation:"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_JOB_GPUS: ${SLURM_JOB_GPUS:-unset}"
scontrol show job $SLURM_JOB_ID | grep -i gres || true
echo ""

# Load helpful modules (non-destructive if modules not available)
module load python/3.11.0 2>/dev/null || true
module load cuda/11.8.0 2>/dev/null || true

# Try to load cuDNN - attempt several common names
for m in cudnn/8.6.0 cudnn/8.9.0 cudnn/8.7.0; do
  module load "$m" 2>/dev/null && echo "Loaded $m" && break || true
done

# Ensure logs directory exists
mkdir -p logs

# Prepare output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="results/conditional_wealthy_container_${TIMESTAMP}"
mkdir -p "${OUT_DIR}"

# Default training command (adjust flags to match your needs)
TRAIN_CMD="python src/train_conditional_gan.py \
  --train_csv data/wealthy_scores_train.csv \
  --val_csv data/wealthy_scores_val.csv \
  --image_dir data/preprocessed_images \
  --attribute_name wealthy_score \
  --image_size 64 \
  --batch_size 128 \
  --latent_dim 128 \
  --epochs 100 \
  --lr 0.0002 \
  --beta1 0.5 \
  --save_every 10 \
  --preview_every 10 \
  --out_dir ${OUT_DIR}"

# If CONTAINER_IMAGE is defined, prefer to run inside container using apptainer/singularity
if [ -n "${CONTAINER_IMAGE:-}" ]; then
  # Find an available apptainer or singularity binary
  if command -v apptainer >/dev/null 2>&1; then
    SING_BIN=$(command -v apptainer)
  elif command -v singularity >/dev/null 2>&1; then
    SING_BIN=$(command -v singularity)
  else
    echo "Warning: CONTAINER_IMAGE specified but neither apptainer nor singularity found. Falling back to host python."
    SING_BIN=""
  fi

  if [ -n "$SING_BIN" ]; then
    echo "Running inside container: ${CONTAINER_IMAGE} using $SING_BIN"
    # Bind the current workspace into /workspace inside the container so relative paths work
    # Use --nv (apptainer)/--nv (singularity) to enable GPU inside container
    $SING_BIN exec --nv --bind "${PWD}:/workspace" "${CONTAINER_IMAGE}" /bin/bash -c \
      "cd /workspace && source .venv/bin/activate 2>/dev/null || true && $TRAIN_CMD 2>&1 | tee ${OUT_DIR}/training_log_${TIMESTAMP}.txt"
  fi
else
  # No container - run on host
  echo "No CONTAINER_IMAGE set - running training on the node directly."
  # Activate virtualenv if present
  if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
  fi

  # Ensure CUDA env vars are set for SLURM
  if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    if [ -n "${SLURM_JOB_GPUS:-}" ]; then
      export CUDA_VISIBLE_DEVICES="$SLURM_JOB_GPUS"
    elif [ -n "${SLURM_STEP_GPUS:-}" ]; then
      export CUDA_VISIBLE_DEVICES="$SLURM_STEP_GPUS"
    else
      export CUDA_VISIBLE_DEVICES=0
    fi
  fi

  echo "Checking GPU availability (quick TF probe)..."
  python - <<'PYEOF'
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
try:
    import tensorflow as tf
    print('TensorFlow version:', tf.__version__)
    print('GPUs:', tf.config.list_physical_devices('GPU'))
except Exception as e:
    print('TensorFlow import/probe failed:', e)
PYEOF

  echo "Starting training (host)..."
  echo "Logging to ${OUT_DIR}/training_log_${TIMESTAMP}.txt"
  eval "$TRAIN_CMD" 2>&1 | tee "${OUT_DIR}/training_log_${TIMESTAMP}.txt"
fi

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Training completed with exit code: ${EXIT_CODE}"
echo "End time: $(date)"
echo "Output saved to: ${OUT_DIR}"
echo ""

if [ ${EXIT_CODE} -eq 0 ]; then
  echo "✅ SUCCESS - Training completed successfully!"
else
  echo "❌ FAILED - Training exited with error code ${EXIT_CODE}"
  echo "Check ${OUT_DIR}/training_log_${TIMESTAMP}.txt and logs/container-slurm-${SLURM_JOB_ID}.err for details"
fi

echo "=========================================="

