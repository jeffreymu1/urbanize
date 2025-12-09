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
echo "Loading required modules..."
module load python/3.11.0
module load cuda/11.8.0

# Try to load cuDNN - try multiple versions if needed
echo "Attempting to load cuDNN..."
if module load cudnn/8.6.0 2>/dev/null; then
    echo "✅ Loaded cudnn/8.6.0"
elif module load cudnn/8.9.0 2>/dev/null; then
    echo "✅ Loaded cudnn/8.9.0"
elif module load cudnn/8.7.0 2>/dev/null; then
    echo "✅ Loaded cudnn/8.7.0"
else
    echo "⚠️  Could not load cuDNN module, will search for it manually"
    echo "Available cuDNN modules:"
    module avail cudnn 2>&1 | grep cudnn || echo "No cudnn modules found"
fi
echo ""

echo "Modules loaded:"
module list
echo ""

echo "CUDA Environment:"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
which nvcc
nvcc --version
nvidia-smi
echo ""

# Activate virtual environment
source .venv/bin/activate

# Print Python and TensorFlow info
echo "Python environment:"
which python
python --version
echo ""

# Set CUDA environment variables explicitly for TensorFlow
# Find CUDA installation path from loaded module
CUDA_HOME=$(dirname $(dirname $(which nvcc)))

# Find cuDNN from environment variables set by module load
echo "Searching for cuDNN installation..."
echo "Module environment variables:"
env | grep -i cudnn || echo "No CUDNN env vars found"
echo ""

# Try to find cuDNN library location - need the actual archive directory with full libraries
if [ -n "$CUDNN_ROOT" ]; then
    CUDNN_HOME=$CUDNN_ROOT
elif [ -n "$CUDNN_DIR" ]; then
    CUDNN_HOME=$CUDNN_DIR
else
    # Search common locations - prefer the full archive directory
    for dir in \
        /gpfs/runtime/opt/cudnn/8.6.0/src/cudnn-linux-x86_64-8.6.0.163_cuda11-archive \
        /gpfs/runtime/opt/cudnn/8.6.0 \
        /oscar/runtime/opt/cudnn/8.6.0 \
        /oscar/rt/*/software/*/cudnn-8.6.0* \
        $CUDA_HOME; do
        if [ -f "$dir/lib64/libcudnn.so" ] || [ -f "$dir/lib/libcudnn.so" ]; then
            CUDNN_HOME=$dir
            echo "Found cuDNN at: $CUDNN_HOME"
            break
        fi
    done
fi

echo "Setting CUDA paths:"
echo "CUDA_HOME: $CUDA_HOME"
echo "CUDNN_HOME: $CUDNN_HOME"

# Find the actual lib directory (lib64 or lib) and add both possible locations
CUDNN_LIB_PATHS=""
if [ -d "$CUDNN_HOME/lib64" ]; then
    CUDNN_LIB_PATHS="$CUDNN_HOME/lib64"
fi
if [ -d "$CUDNN_HOME/lib" ]; then
    CUDNN_LIB_PATHS="$CUDNN_LIB_PATHS:$CUDNN_HOME/lib"
fi
# Also add the archive subdirectory if it exists
if [ -d "$CUDNN_HOME/src/cudnn-linux-x86_64-8.6.0.163_cuda11-archive/lib" ]; then
    CUDNN_LIB_PATHS="$CUDNN_LIB_PATHS:$CUDNN_HOME/src/cudnn-linux-x86_64-8.6.0.163_cuda11-archive/lib"
fi

export CUDA_HOME
export CUDNN_HOME
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDNN_LIB_PATHS:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

echo "Updated LD_LIBRARY_PATH:"
echo "$LD_LIBRARY_PATH" | tr ':' '\n' | nl
echo ""

# Check CUDA libraries are accessible
echo "Checking CUDA libraries:"
ls -la $CUDA_HOME/lib64/libcudart.so* 2>&1 | head -3
ls -la $CUDA_HOME/lib64/libcublas.so* 2>&1 | head -3
echo ""
echo "Checking cuDNN libraries:"
find $CUDNN_HOME -name "libcudnn.so*" 2>/dev/null | head -5
for libdir in $(echo $CUDNN_LIB_PATHS | tr ':' ' '); do
    if [ -d "$libdir" ]; then
        echo "In $libdir:"
        ls -la $libdir/libcudnn*.so* 2>&1 | head -3
    fi
done
echo ""

# Verify GPU with detailed TensorFlow diagnostics
echo "GPU Check (TensorFlow):"
python -c "
import os
import sys

# Enable all TensorFlow logging to see the actual error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_CPP_VMODULE'] = 'gpu_device=10'

print('Loading TensorFlow...')
import tensorflow as tf
print('TensorFlow version:', tf.__version__)
print('Built with CUDA:', tf.test.is_built_with_cuda())

# Try to list physical devices with error handling
try:
    print('Attempting to list GPU devices...')
    gpu_devices = tf.config.list_physical_devices('GPU')
    print('GPU devices:', gpu_devices)

    if len(gpu_devices) > 0:
        print('✅ GPU DETECTED')
        for gpu in gpu_devices:
            print(f'  - {gpu}')
    else:
        print('❌ NO GPU DETECTED')
except Exception as e:
    print(f'❌ ERROR detecting GPU: {e}')
    import traceback
    traceback.print_exc()

# Check if libraries can be loaded
print('')
print('Checking CUDA libraries...')
import ctypes
libs_to_check = [
    'libcudart.so',
    'libcudart.so.11.0',
    'libcudnn.so',
    'libcudnn.so.8',
    'libcublas.so',
    'libcublasLt.so',
]

for lib in libs_to_check:
    try:
        ctypes.CDLL(lib)
        print(f'  ✅ {lib} found')
    except Exception as e:
        print(f'  ❌ {lib} NOT found: {e}')
"
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

