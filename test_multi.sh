#!/bin/bash
# Quick stability test for multi-attribute GAN

cd "$(dirname "$0")"

echo "==================================="
echo "Multi-Attribute GAN - Quick Test"
echo "==================================="

# Create small test dataset
echo "Creating test dataset (500 samples)..."
python3 -c "
import pandas as pd

df = pd.read_csv('data/all_attribute_scores/all_attribute_scores_train.csv')
df_test = df.head(500)
df_test.to_csv('data/multi_test.csv', index=False)
print(f'Created test dataset: {len(df_test)} samples')
print(f'Attributes: {list(df_test.columns)}')
"

if [ $? -ne 0 ]; then
    echo "Error creating test dataset"
    exit 1
fi

# Run 3 epoch test
echo ""
echo "Running 3-epoch stability test..."
echo "Expected time: ~2-3 minutes"
echo ""

python3 src/multi_model_files/multi_model.py \
    --csv_path data/multi_test.csv \
    --img_dir data/preprocessed_images \
    --out_dir results/multi_quick_test \
    --image_size 64 \
    --batch_size 32 \
    --epochs 3 \
    --latent_dim 128 \
    --lr_g 0.0002 \
    --lr_d 0.0002 \
    --lambda_reg 10.0 \
    --save_every_epochs 3 \
    --num_eval_samples 16

echo ""
echo "==================================="
echo "Test complete!"
echo "==================================="
echo "Results: results/multi_quick_test/"
echo ""
echo "Check:"
echo "  - samples/epoch_0003_samples.png"
echo "  - sweeps/epoch_0003_sweep_*.png"
echo ""
echo "Training should show:"
echo "  D_adv ~1.0-2.0, D_reg decreasing"
echo "  G_adv ~-1.0 to 1.0, G_reg decreasing"
echo ""

# Cleanup
rm -f data/multi_test.csv

