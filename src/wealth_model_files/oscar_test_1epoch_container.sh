#!/bin/bash
#SBATCH --job-name=test_gan_1ep
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/test-%j.out
#SBATCH --error=logs/test-%j.err

# OSCAR Test Script Using TensorFlow Container (RECOMMENDED APPROACH)
# This uses a pre-built TensorFlow container with GPU support

echo "=========================================="
echo "OSCAR 1-Epoch Test Run (Container-based)"
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
print('GPU devices:', gpus)
if len(gpus) > 0:
    print('✅ GPU DETECTED - Ready to train!')
else:
    print('❌ NO GPU DETECTED')
"
echo ""

# Install dependencies in container
echo "Installing dependencies..."
$EXEC $CONTAINER_PATH pip install --user --no-cache-dir pandas matplotlib scipy 2>&1 | grep -v "already satisfied" || true
echo "Dependencies installed"
echo ""

# Create output directory
mkdir -p logs
mkdir -p results/oscar_test_1epoch

# Run training
echo "Starting 1-epoch test..."
echo "=========================================="
echo ""

$EXEC $CONTAINER_PATH python -u src/train_conditional_gan.py \
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
    echo "Submit full job: sbatch oscar_train_conditional_container.sh"
else
    echo "❌ Test FAILED - Check logs/test-${SLURM_JOB_ID}.err"
fi
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE

