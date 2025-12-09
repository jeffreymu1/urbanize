#!/bin/bash
# Train conditional GAN on "wealthy" attribute

set -e

echo "🏛️  Training Conditional GAN - Wealthy Attribute"
echo "=============================================="
echo ""
echo "This will train an attribute-controllable GAN that can generate"
echo "urban images with varying 'wealthy' appearance based on a 0-1 slider."
echo ""
echo "Estimated training time: ~8-12 hours for 100 epochs"
echo "Start time: $(date)"
echo ""

cd /Users/mwinter/School/School/csci1470-course/urbanize

# Activate virtual environment
source .venv/bin/activate

# Create output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="results/conditional_wealthy_${TIMESTAMP}"

echo "Output directory: ${OUT_DIR}"
echo ""

# Run training with caffeinate to prevent sleep
caffeinate -d -i python src/train_conditional_gan.py \
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

echo ""
echo "✅ Training complete!"
echo "End time: $(date)"
echo ""
echo "Next steps:"
echo "1. Check preview images in ${OUT_DIR}/"
echo "2. Launch interactive demo:"
echo "   python src/interactive_demo.py --checkpoint_dir ${OUT_DIR}/checkpoints/"

