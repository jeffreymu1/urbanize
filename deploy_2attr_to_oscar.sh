#!/bin/bash
# Deploy 2-attribute GAN files to OSCAR
# Run this from your LOCAL machine

OSCAR_USER="${OSCAR_USER:-$USER}"
OSCAR_HOST="ssh.ccv.brown.edu"

echo "=========================================="
echo "Deploying 2-Attribute GAN to OSCAR"
echo "=========================================="
echo "OSCAR user: $OSCAR_USER"
echo "OSCAR host: $OSCAR_HOST"
echo ""

# Check if files exist locally
if [ ! -d "data/wealthy_lively" ]; then
    echo "❌ ERROR: data/wealthy_lively/ not found locally"
    echo "Run this first: python src/create_wealthy_lively_dataset.py"
    exit 1
fi

if [ ! -f "src/train_2attr_gan.py" ]; then
    echo "❌ ERROR: src/train_2attr_gan.py not found"
    exit 1
fi

if [ ! -f "oscar_train_2attr_wealthy_lively.sh" ]; then
    echo "❌ ERROR: oscar_train_2attr_wealthy_lively.sh not found"
    exit 1
fi

echo "✅ All files found locally"
echo ""

# Upload files
echo "Uploading files to OSCAR..."
echo ""

echo "1/4 Uploading dataset (data/wealthy_lively/)..."
scp -r data/wealthy_lively ${OSCAR_USER}@${OSCAR_HOST}:~/urbanize/data/
if [ $? -eq 0 ]; then
    echo "  ✅ Dataset uploaded"
else
    echo "  ❌ Failed to upload dataset"
    exit 1
fi

echo ""
echo "2/4 Uploading training script (src/train_2attr_gan.py)..."
scp src/train_2attr_gan.py ${OSCAR_USER}@${OSCAR_HOST}:~/urbanize/src/
if [ $? -eq 0 ]; then
    echo "  ✅ Training script uploaded"
else
    echo "  ❌ Failed to upload training script"
    exit 1
fi

echo ""
echo "3/4 Uploading dataset creation script (src/create_wealthy_lively_dataset.py)..."
scp src/create_wealthy_lively_dataset.py ${OSCAR_USER}@${OSCAR_HOST}:~/urbanize/src/
if [ $? -eq 0 ]; then
    echo "  ✅ Dataset creation script uploaded"
else
    echo "  ❌ Failed to upload dataset creation script"
    exit 1
fi

echo ""
echo "4/4 Uploading OSCAR job script (oscar_train_2attr_wealthy_lively.sh)..."
scp oscar_train_2attr_wealthy_lively.sh ${OSCAR_USER}@${OSCAR_HOST}:~/urbanize/
if [ $? -eq 0 ]; then
    echo "  ✅ OSCAR job script uploaded"
else
    echo "  ❌ Failed to upload OSCAR job script"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ Deployment Complete!"
echo "=========================================="
echo ""
echo "Next steps on OSCAR:"
echo "  1. SSH to OSCAR:"
echo "     ssh ${OSCAR_USER}@${OSCAR_HOST}"
echo ""
echo "  2. Verify files:"
echo "     cd ~/urbanize"
echo "     ls -lh data/wealthy_lively/"
echo "     ls -lh src/train_2attr_gan.py"
echo ""
echo "  3. Submit training job:"
echo "     sbatch oscar_train_2attr_wealthy_lively.sh"
echo ""
echo "  4. Monitor:"
echo "     squeue -u \$USER"
echo "     tail -f results/2attr_wealthy_lively_*/training_log_*.txt"
echo ""

