#!/bin/bash
#SBATCH --job-name=cond_gan_wealthy
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@brown.edu

# OSCAR SLURM script for training Conditional GAN on wealthy attribute
#
# Usage:
#   sbatch oscar_train_conditional.sh
#
# To monitor:
#   squeue -u $USER
#   tail -f logs/slurm-JOBID.out

echo "=========================================="
echo "OSCAR Conditional GAN Training"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# Load modules
module load python/3.11.0
module load cuda/11.8.0

# Try to load cuDNN - try multiple versions
if module load cudnn/8.6.0 2>/dev/null; then
    echo "Loaded cudnn/8.6.0"
elif module load cudnn/8.9.0 2>/dev/null; then
    echo "Loaded cudnn/8.9.0"
elif module load cudnn/8.7.0 2>/dev/null; then
    echo "Loaded cudnn/8.7.0"
else
    echo "Warning: Could not load cuDNN module"
fi

# Activate virtual environment
source .venv/bin/activate

# Set CUDA environment variables explicitly for TensorFlow
CUDA_HOME=$(dirname $(dirname $(which nvcc)))

# Find cuDNN from environment or search common locations
if [ -n "$CUDNN_ROOT" ]; then
    CUDNN_HOME=$CUDNN_ROOT
elif [ -n "$CUDNN_DIR" ]; then
    CUDNN_HOME=$CUDNN_DIR
else
    for dir in /gpfs/runtime/opt/cudnn/8.6.0 /oscar/runtime/opt/cudnn/8.6.0 /oscar/rt/*/software/*/cudnn-8.6.0* $CUDA_HOME; do
        if [ -f "$dir/lib64/libcudnn.so" ] || [ -f "$dir/lib/libcudnn.so" ]; then
            CUDNN_HOME=$dir
            break
        fi
    done
fi

# Find the actual lib directory
if [ -d "$CUDNN_HOME/lib64" ]; then
    CUDNN_LIB=$CUDNN_HOME/lib64
elif [ -d "$CUDNN_HOME/lib" ]; then
    CUDNN_LIB=$CUDNN_HOME/lib
else
    CUDNN_LIB=$CUDNN_HOME
fi

export CUDA_HOME
export CUDNN_HOME
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDNN_LIB:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

# Verify GPU is available
echo "Checking GPU availability..."
python -c "
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import tensorflow as tf
print(f'TensorFlow version: {tf.__version__}')
print(f'GPUs Available: {tf.config.list_physical_devices(\"GPU\")}')
"
echo ""

# Create timestamped output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="results/conditional_wealthy_oscar_${TIMESTAMP}"
mkdir -p "${OUT_DIR}"
mkdir -p logs

echo "Output directory: ${OUT_DIR}"
echo ""
echo "Starting training..."
echo "=========================================="
echo ""

# Run training
python src/train_conditional_gan.py \
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
  --out_dir "${OUT_DIR}" \
  2>&1 | tee "${OUT_DIR}/training_log_${TIMESTAMP}.txt"

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Training completed with exit code: ${EXIT_CODE}"
echo "End time: $(date)"
echo "Output saved to: ${OUT_DIR}"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ SUCCESS - Training completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Copy results back to your local machine:"
    echo "   scp -r YOUR_USERNAME@ssh.ccv.brown.edu:~/urbanize/${OUT_DIR} ~/path/to/local/results/"
    echo ""
    echo "2. View preview images:"
    echo "   ls ${OUT_DIR}/preview_*.png"
    echo ""
    echo "3. Launch interactive demo (on local machine):"
    echo "   python src/interactive_demo.py --checkpoint_dir ${OUT_DIR}/checkpoints/"
else
    echo "❌ FAILED - Training exited with error code ${EXIT_CODE}"
    echo "Check logs/slurm-${SLURM_JOB_ID}.err for details"
fi

echo "=========================================="

