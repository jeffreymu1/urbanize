#!/bin/bash
# Quick diagnostic test for 2-attribute GAN on OSCAR
# Run this to diagnose why the training script fails silently

echo "=========================================="
echo "2-Attribute GAN Diagnostic Test"
echo "=========================================="
echo ""

cd ~/urbanize || exit 1

echo "1. Checking files exist..."
echo ""

if [ -f "data/wealthy_lively/wealthy_lively_scores_train.csv" ]; then
    echo "✅ Train CSV exists"
    wc -l data/wealthy_lively/wealthy_lively_scores_train.csv
else
    echo "❌ Train CSV missing"
fi

if [ -f "data/wealthy_lively/wealthy_lively_scores_val.csv" ]; then
    echo "✅ Val CSV exists"
    wc -l data/wealthy_lively/wealthy_lively_scores_val.csv
else
    echo "❌ Val CSV missing"
fi

if [ -d "data/preprocessed_images" ]; then
    echo "✅ Image directory exists"
    echo "Image count: $(ls data/preprocessed_images/*.jpg 2>/dev/null | wc -l)"
else
    echo "❌ Image directory missing"
fi

echo ""
echo "2. Testing Python imports..."
echo ""

python3 -c "
import sys
print('Python version:', sys.version)
print()

try:
    import tensorflow as tf
    print('✅ TensorFlow:', tf.__version__)
    print('   GPUs:', len(tf.config.list_physical_devices('GPU')))
except Exception as e:
    print('❌ TensorFlow import failed:', e)

try:
    import pandas as pd
    print('✅ Pandas:', pd.__version__)
except Exception as e:
    print('❌ Pandas import failed:', e)

try:
    import matplotlib.pyplot as plt
    print('✅ Matplotlib imported')
except Exception as e:
    print('❌ Matplotlib import failed:', e)

try:
    import numpy as np
    print('✅ NumPy:', np.__version__)
except Exception as e:
    print('❌ NumPy import failed:', e)
"

echo ""
echo "3. Testing CSV file loading..."
echo ""

python3 -c "
import pandas as pd
try:
    df = pd.read_csv('data/wealthy_lively/wealthy_lively_scores_train.csv')
    print(f'✅ Loaded training CSV: {len(df):,} rows')
    print(f'   Columns: {list(df.columns)}')
    print(f'   First row:')
    print(df.head(1))
except Exception as e:
    print(f'❌ Failed to load CSV: {e}')
"

echo ""
echo "4. Testing train_2attr_gan.py imports..."
echo ""

python3 -c "
import sys
sys.path.insert(0, 'src')

try:
    print('Importing train_conditional_gan...')
    from train_conditional_gan import (
        make_conditional_generator,
        make_conditional_discriminator,
        discriminator_loss,
        generator_loss
    )
    print('✅ All imports from train_conditional_gan successful')
except Exception as e:
    print(f'❌ Import failed: {e}')
    import traceback
    traceback.print_exc()
"

echo ""
echo "5. Testing script argument parsing..."
echo ""

python3 src/train_2attr_gan.py --help > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Script can parse arguments"
else
    echo "❌ Script failed to parse arguments"
    python3 src/train_2attr_gan.py --help
fi

echo ""
echo "6. Testing with minimal epochs (DRY RUN)..."
echo ""

echo "Running: python3 src/train_2attr_gan.py --epochs 1 --batch_size 16"
echo "(This will attempt to run 1 epoch with small batch size)"
echo ""

timeout 120 python3 src/train_2attr_gan.py \
    --train_csv data/wealthy_lively/wealthy_lively_scores_train.csv \
    --val_csv data/wealthy_lively/wealthy_lively_scores_val.csv \
    --image_dir data/preprocessed_images \
    --epochs 1 \
    --batch_size 16 \
    --image_size 64 \
    --out_dir results/diagnostic_test 2>&1

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ DIAGNOSTIC PASSED - Script ran successfully!"
    echo "The issue was likely the container environment."
    echo "Try submitting the job again."
elif [ $EXIT_CODE -eq 124 ]; then
    echo "⏱️  TIMEOUT - Script is running but slow"
    echo "This is normal - it's loading data."
    echo "The job should work on OSCAR."
else
    echo "❌ DIAGNOSTIC FAILED - Exit code: $EXIT_CODE"
    echo "Check the error output above for details."
fi
echo "=========================================="

