#!/bin/bash
# Step 1: Compute all attribute scores for multi-attribute GAN
# Run this NOW while the wealthy model is training on OSCAR

echo "=========================================="
echo "Computing All Attribute Scores"
echo "=========================================="
echo "This will take ~30 minutes"
echo ""

# Create output directory
mkdir -p data/multi_attribute

# Compute scores for all 5 attributes
python src/compute_attribute_scores.py \
  --data_csv data/PP2/final_data.csv \
  --attribute all \
  --output data/all_attribute_scores.csv \
  --method elo \
  --image_dir data/preprocessed_images \
  --split \
  --train_frac 0.8 \
  --val_frac 0.1

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ SUCCESS!"
    echo "=========================================="
    echo ""
    echo "Created files:"
    ls -lh data/all_attribute_scores*.csv
    echo ""
    echo "Next steps:"
    echo "1. Check the score distribution:"
    echo "   python src/check_distribution.py --scores_csv data/all_attribute_scores.csv"
    echo ""
    echo "2. Upload to OSCAR (if needed):"
    echo "   scp data/all_attribute_scores*.csv YOUR_USER@ssh.ccv.brown.edu:~/urbanize/data/"
    echo ""
    echo "3. After wealthy model completes, start multi-attribute training"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "❌ FAILED"
    echo "=========================================="
    echo "Check the error messages above"
fi

