#!/bin/bash
#SBATCH --job-name=2attr_gan_generate
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/2attr_generate-%j.out
#SBATCH --error=logs/2attr_generate-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@brown.edu

# OSCAR 2-Attribute GAN Image Generation Script
# Generates images using a trained GAN checkpoint

module purge
unset LD_LIBRARY_PATH

export APPTAINER_BINDPATH="/oscar/home/$USER,/oscar/scratch/$USER,/oscar/data,$PWD"
CONTAINER_PATH="/oscar/runtime/software/external/ngc-containers/tensorflow.d/x86_64.d/tensorflow-24.03-tf2-py3.simg"
EXEC="apptainer exec --nv"

# TensorFlow GPU check
$EXEC $CONTAINER_PATH python -c "
import tensorflow as tf
print('TensorFlow version:', tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print('GPUs detected:', len(gpus))
if len(gpus) > 0:
    print('✅ GPU DETECTED - Ready to generate!')
else:
    print('❌ NO GPU DETECTED')
"

# Generation parameters
CHECKPOINT_DIR="results/2attr_wealthy_lively_01/checkpoints/"
CHECKPOINT_NAME="ckpt-16"
IMAGE_SIZE=64
LATENT_DIM=128
OUTPUT_MODE="grid" # or "individual"
WEALTHY_VALUES="0.0 0.5 1.0"
LIVELY_VALUES="0.0 0.5 1.0"
NUM_SAMPLES=1
OUT_PATH="results/generated_2attr_grid_ckpt16.png"
OUT_DIR="results/generated_2attr_samples_ckpt16/"

# Run generation
$EXEC $CONTAINER_PATH python src/generate_2attr_images.py \
  --checkpoint_dir $CHECKPOINT_DIR \
  --checkpoint_name $CHECKPOINT_NAME \
  --image_size $IMAGE_SIZE \
  --latent_dim $LATENT_DIM \
  --output_mode $OUTPUT_MODE \
  --wealthy_values $WEALTHY_VALUES \
  --lively_values $LIVELY_VALUES \
  --num_samples $NUM_SAMPLES \
  --out_path $OUT_PATH \
  --out_dir $OUT_DIR

# For individual samples, change OUTPUT_MODE and provide --wealthy and --lively
# Example:
# OUTPUT_MODE="individual"
# WEALTHY=0.8
# LIVELY=0.2
# NUM_SAMPLES=10
# OUT_DIR="results/wealthy0.8_lively0.2_samples_ckpt16/"
# $EXEC $CONTAINER_PATH python src/generate_2attr_images.py \
#   --checkpoint_dir $CHECKPOINT_DIR \
#   --checkpoint_name $CHECKPOINT_NAME \
#   --image_size $IMAGE_SIZE \
#   --latent_dim $LATENT_DIM \
#   --output_mode $OUTPUT_MODE \
#   --wealthy $WEALTHY \
#   --lively $LIVELY \
#   --num_samples $NUM_SAMPLES \
#   --out_dir $OUT_DIR


