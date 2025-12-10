# Urbanize - Conditional Urban Scene GAN

Generate urban street scenes with controllable **wealthy** attribute.

## Current Model: 128x128 Wealthy Attribute GAN

**Recent Updates (Dec 9, 2024):**
- ✅ Upgraded resolution: 64x64 → 128x128 (4x more pixels)
- ✅ Improved architecture: Added dropout, label smoothing
- ✅ Better monitoring: Added discriminator confidence metrics
- ✅ Fixed mode collapse issues from original model

## Quick Start on OSCAR

### 1. Submit Training Job
```bash
cd ~/urbanize  # or your path
sbatch oscar_train_conditional_container.sh
```

### 2. Monitor Training
```bash
# Easy way (no permissions needed)
watch -n 10 bash check_status.sh

# Or view live log
tail -f results/conditional_wealthy_*/training_log_*.txt

# Check job status
squeue -u $USER
```

### 3. Generate Images (After Training)
```bash
# Generate grid showing wealth 0.0 → 1.0
python src/generate_samples.py \
    --checkpoint results/conditional_wealthy_*/checkpoints/ckpt-10 \
    --mode grid

# Generate everything (grid, samples, interpolation)
python src/generate_samples.py \
    --checkpoint results/conditional_wealthy_*/checkpoints/ckpt-10 \
    --mode all
```

## Training Details

**Architecture:** Conditional DCGAN
- Input: 128-d latent noise + 1-d wealth score
- Output: 128×128×3 RGB images
- Generator: 4×4 start → progressive upsampling → 128×128
- Discriminator: Dropout(0.3) for stability

**Training:**
- Dataset: 89K training images, 10K validation
- Epochs: 100 (~4-5 hours on OSCAR GPU)
- Batch size: 128
- Learning rate: 0.0002 (Adam, β₁=0.5)

**Expected Metrics (Healthy Training):**
- D(real): 0.6-0.8
- D(fake): 0.1-0.4
- G loss: 1.5-2.5 (stable)
- D loss: 0.6-0.8 (stable)

## Generation Script Usage

The `generate_samples.py` script supports 4 modes:

**1. Grid** - Shows wealth progression (0.0 → 1.0)
```bash
python src/generate_samples.py --checkpoint PATH --mode grid
```

**2. Specific** - Generate N samples at specific wealth levels
```bash
python src/generate_samples.py --checkpoint PATH --mode specific \
    --wealth_scores 0.0 0.5 1.0 --num_samples 10
```

**3. Random** - Generate diverse dataset
```bash
python src/generate_samples.py --checkpoint PATH --mode random \
    --num_random 200
```

**4. Interpolation** - Smooth transition visualization
```bash
python src/generate_samples.py --checkpoint PATH --mode interpolation
```

## Key Files
- `src/train_multi_attribute_gan.py` - Multi-attribute GAN training
- `oscar_train_multi_attribute.sh` - OSCAR GPU training script
- `test_multi_attribute_local.sh` - Local 1-epoch test

### Data
- `data/all_attribute_scores_*.csv` - Train/val/test splits with 6 attributes
- `data/preprocessed_images/` - 64×64 preprocessed street view images


### Scripts
- `oscar_train_conditional_container.sh` - OSCAR training job (128x128)
- `src/train_conditional_gan.py` - Training code
- `src/generate_samples.py` - Image generation
- `check_status.sh` - Simple monitoring (no permissions needed)

### Data
- `data/wealthy_scores_train.csv` - Training data (89K images)
- `data/wealthy_scores_val.csv` - Validation data (10K images)
- `data/preprocessed_images/` - 300×300 cropped images

## Troubleshooting

### Permission Denied on ./script.sh
Use `bash check_status.sh` instead of `./check_status.sh`

### Job Not Starting
```bash
squeue -u $USER  # Check if queued
squeue -p gpu    # See GPU availability
```

### Training Looks Bad
Check discriminator balance:
- D(real) > 0.95, D(fake) < 0.05 → Mode collapse
- D(real) < 0.4, D(fake) > 0.6 → Generator dominating
- Both metrics should be in healthy ranges above

### Out of Memory
Reduce batch size in `oscar_train_conditional_container.sh`:
```bash
--batch_size 64  # Instead of 128
```

## Project Structure

```
urbanize/
├── src/
│   ├── train_conditional_gan.py      # Main training (128x128)
│   ├── generate_samples.py           # Image generation
│   ├── baseline_model.py             # Baseline unconditional GAN
│   └── preprocess.py                 # Image preprocessing
├── data/
│   ├── wealthy_scores_train.csv      # Training scores
│   ├── wealthy_scores_val.csv        # Validation scores
│   └── preprocessed_images/          # 300×300 images
├── results/                           # Training outputs
├── oscar_train_conditional_container.sh  # OSCAR job script
├── check_status.sh                    # Simple monitoring
└── README.md                          # This file
```

## Dataset

**Source:** Place Pulse 2.0 - Urban perception dataset  
**Images:** 110K street view images  
**Annotations:** 1.2M pairwise comparisons  
**Attributes:** Wealth scores computed via Elo ranking system  
**Processing:** Center-cropped to 300×300, resized to 128×128 for training

## Citation

If you use this dataset or model:
```
Dubey, A., Naik, N., Parikh, D., Raskar, R., & Hidalgo, C. A. (2016).
Deep Learning the City: Quantifying Urban Perception at a Global Scale.
ECCV 2016.
```

│   └── check_multi_attribute_distribution.py  # Data analysis
├── data/
│   ├── all_attribute_scores_train.csv    # Training data
│   ├── all_attribute_scores_val.csv      # Validation data
│   └── preprocessed_images/              # 110K images
├── oscar_train_multi_attribute.sh        # OSCAR training
├── test_multi_attribute_local.sh         # Local test
├── check_training.sh                     # Monitor (OSCAR)
└── monitor_oscar_training.sh             # Monitor (local)
```

---

**Model Status:** Ready to train on OSCAR  
**Expected Training Time:** ~2 hours (100 epochs on GPU)  
**Model Size:** ~15M parameters  
**Computational Complexity:** O(N) in attributes (negligible overhead)

