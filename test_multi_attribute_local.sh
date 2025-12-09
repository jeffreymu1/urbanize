#!/bin/bash
# Local test of multi-attribute GAN (1 epoch)
# Run this to verify everything works before submitting to OSCAR

echo "=========================================="
echo "Multi-Attribute GAN - Local Test (1 Epoch)"
echo "=========================================="
echo ""

# Check if data files exist
if [ ! -f "data/all_attribute_scores_train.csv" ]; then
    echo "❌ ERROR: data/all_attribute_scores_train.csv not found!"
    echo "Run: ./step1_compute_all_scores.sh"
    exit 1
fi

if [ ! -f "data/all_attribute_scores_val.csv" ]; then
    echo "❌ ERROR: data/all_attribute_scores_val.csv not found!"
    echo "Run: ./step1_compute_all_scores.sh"
    exit 1
fi

echo "✅ Data files found"
echo ""

# Create test output directory
OUT_DIR="results/multi_attr_test_local"
mkdir -p "$OUT_DIR"

echo "Configuration:"
echo "  Epochs: 1 (test run)"
echo "  Batch size: 64 (smaller for local)"
echo "  Attributes: 6 (wealthy, depressing, safety, lively, boring, beautiful)"
echo "  Output: $OUT_DIR"
echo ""

echo "Starting test training..."
echo "=========================================="
echo ""

python src/train_multi_attribute_gan.py \
  --train_csv data/all_attribute_scores_train.csv \
  --val_csv data/all_attribute_scores_val.csv \
  --image_dir data/preprocessed_images \
  --attribute_names wealthy depressing safety lively boring beautiful \
  --epochs 1 \
  --batch_size 64 \
  --latent_dim 128 \
  --image_size 64 \
  --save_every 1 \
  --preview_every 1 \
  --out_dir "$OUT_DIR"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ TEST PASSED!"
    echo ""
    echo "Generated files:"
    ls -lh "$OUT_DIR"
    echo ""
    echo "Preview image:"
    ls -lh "$OUT_DIR"/preview_*.png
    echo ""
    echo "Ready to submit to OSCAR:"
    echo "  1. Upload data: scp data/all_attribute_scores*.csv YOUR_USER@ssh.ccv.brown.edu:~/urbanize/data/"
    echo "  2. Submit job: sbatch oscar_train_multi_attribute.sh"
else
    echo "❌ TEST FAILED"
    echo "Fix errors before submitting to OSCAR"
fi
echo "=========================================="

exit $EXIT_CODE

