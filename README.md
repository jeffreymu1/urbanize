# Urbanize - Multi-Attribute Urban Scene GAN

Generate urban street scenes with independent control over **6 attributes**:
- **Wealthy** (poor → wealthy)
- **Depressing** (uplifting → depressing)  
- **Safety** (unsafe → safe)
- **Lively** (quiet → lively)
- **Boring** (interesting → boring)
- **Beautiful** (ugly → beautiful)

## Quick Start

### Local Testing (1 epoch)
```bash
./test_multi_attribute_local.sh
```

### Training on OSCAR (100 epochs, ~2 hours)
```bash
# Upload data and scripts
scp data/all_attribute_scores*.csv YOUR_USER@ssh.ccv.brown.edu:~/urbanize/data/
scp src/train_multi_attribute_gan.py YOUR_USER@ssh.ccv.brown.edu:~/urbanize/src/
scp oscar_train_multi_attribute.sh YOUR_USER@ssh.ccv.brown.edu:~/urbanize/

# Submit job
ssh YOUR_USER@ssh.ccv.brown.edu
cd ~/urbanize
sbatch oscar_train_multi_attribute.sh
```

### Monitor Training
```bash
# On OSCAR terminal:
./check_training.sh

# From local machine:
./monitor_oscar_training.sh YOUR_USER
```

## Architecture

**Model:** Conditional DCGAN
- **Generator Input:** 128-d noise + 6-d attributes = 134-d total
- **Discriminator Input:** 64×64×3 image + 6-d attributes
- **Output:** 64×64×3 RGB images

**Dataset:** 110K urban street view images with computed attribute scores from 1.2M pairwise comparisons

## Key Files

### Training Scripts
- `src/train_multi_attribute_gan.py` - Multi-attribute GAN training
- `oscar_train_multi_attribute.sh` - OSCAR GPU training script
- `test_multi_attribute_local.sh` - Local 1-epoch test

### Data
- `data/all_attribute_scores_*.csv` - Train/val/test splits with 6 attributes
- `data/preprocessed_images/` - 64×64 preprocessed street view images

### Monitoring
- `check_training.sh` - Quick training status (on OSCAR)
- `monitor_oscar_training.sh` - Full monitoring dashboard (local)

### Documentation (in `info/`)
- `info/MULTI_ATTRIBUTE_PLAN.md` - Implementation roadmap
- `info/MONITORING_GUIDE.md` - Detailed monitoring instructions
- `info/OSCAR_QUICK_REF.md` - Quick OSCAR commands
- `info/OSCAR_QUICK_COMMANDS.txt` - Copy-paste command reference
- `info/OSCAR_SETUP_GUIDE.md` - OSCAR setup (first-time only)
- `info/OSCAR_COMPLETE_GUIDE.md` - Complete OSCAR reference

## Results

After training, results will be in `results/multi_attribute_gan_TIMESTAMP/`:
- `checkpoints/` - Model weights
- `preview_epoch_*.png` - Generated samples (6 attributes × 6 values grid)
- `metrics.csv` - Training loss history
- `training_log_*.txt` - Full training logs

## Generate Images

After training completes:

```python
# Generate with specific attributes
python src/generate_multi_attribute.py \
  --checkpoint_dir results/multi_attribute_gan_*/checkpoints/ \
  --wealthy 0.8 --safety 0.9 --beautiful 0.7

# Interactive demo with sliders
python src/interactive_multi_attribute_demo.py \
  --checkpoint_dir results/multi_attribute_gan_*/checkpoints/
```

## Dependencies

```bash
# Install on local machine
pip install -r requirements.txt

# OSCAR uses pre-built TensorFlow container (no install needed)
```

## Project Structure

```
urbanize/
├── src/
│   ├── train_multi_attribute_gan.py      # Main training script
│   ├── compute_attribute_scores.py       # Score computation
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

