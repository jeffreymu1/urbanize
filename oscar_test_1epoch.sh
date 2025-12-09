#!/bin/bash
#SBATCH --job-name=test_gan_1ep
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/test-%j.out
#SBATCH --error=logs/test-%j.err

# Quick 1-epoch test to verify everything works on OSCAR
# Usage: sbatch oscar_test_1epoch.sh

echo "=========================================="
echo "OSCAR 1-Epoch Test Run"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Load modules
module load python/3.11.0
module load cuda/11.8.0
module load cudnn/8.6.0

# Activate virtual environment
source .venv/bin/activate

# Verify GPU
echo "GPU Check:"
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
echo ""

# Create logs directory
mkdir -p logs
mkdir -p results/oscar_test_1epoch

# Run 1-epoch test
python src/train_conditional_gan.py \
  --train_csv data/wealthy_scores_train.csv \
  --val_csv data/wealthy_scores_val.csv \
  --image_dir data/preprocessed_images \
  --attribute_name wealthy_score \
  --epochs 1 \
  --batch_size 128 \
  --save_every 1 \
  --preview_every 1 \
  --out_dir results/oscar_test_1epoch

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Test PASSED - Ready for full training!"
    echo "Submit full job: sbatch oscar_train_conditional.sh"
else
    echo "❌ Test FAILED - Check logs/test-${SLURM_JOB_ID}.err"
fi
echo "End time: $(date)"
echo "=========================================="

