#!/usr/bin/env python3
"""
2-Attribute Conditional DCGAN for Wealthy + Lively Control

Generates images conditioned on TWO attributes: wealthy and lively scores.

Architecture:
- Generator: [latent(128-d), wealthy(1-d), lively(1-d)] → 128x128x3 image
- Discriminator: [128x128x3 image, wealthy(1-d), lively(1-d)] → real/fake

Usage:
    python train_2attr_gan.py \\
        --train_csv data/wealthy_lively/wealthy_lively_scores_train.csv \\
        --val_csv data/wealthy_lively/wealthy_lively_scores_val.csv \\
        --epochs 200
"""

import os
import sys

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

# Set TensorFlow logging before import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # Show all logs

# Add parent directory to path to import from train_conditional_gan
sys.path.insert(0, os.path.dirname(__file__))

import time
import csv
from pathlib import Path
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Import from existing single-attribute script
from train_conditional_gan import (
    make_conditional_generator,
    make_conditional_discriminator,
    discriminator_loss,
    generator_loss
)


# ==================== 2-ATTRIBUTE DATA LOADING ====================

def load_image_with_2attrs(image_path, wealthy_score, lively_score, image_size):
    """Load image with 2 attribute scores."""
    img_bytes = tf.io.read_file(image_path)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [image_size, image_size], antialias=True)
    img = img * 2.0 - 1.0  # [-1, 1]

    # Return image and 2D attribute vector
    attrs = tf.stack([wealthy_score, lively_score])
    return img, attrs


def make_2attr_dataset(scores_csv, image_dir, image_size, batch_size, shuffle=True):
    """
    Create dataset for 2-attribute training.

    Returns dataset yielding (images, [wealthy_score, lively_score])
    """
    df = pd.read_csv(scores_csv)

    image_paths = []
    wealthy_scores = []
    lively_scores = []

    for _, row in df.iterrows():
        img_path = Path(image_dir) / f"{row['image_id']}.jpg"
        if img_path.exists():
            image_paths.append(str(img_path))
            wealthy_scores.append(float(row['wealthy_score']))
            lively_scores.append(float(row['lively_score']))

    print(f"  Loaded {len(image_paths):,} images with 2 attributes")

    # Create dataset
    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    wealthy_ds = tf.data.Dataset.from_tensor_slices(wealthy_scores)
    lively_ds = tf.data.Dataset.from_tensor_slices(lively_scores)
    ds = tf.data.Dataset.zip((path_ds, wealthy_ds, lively_ds))

    if shuffle:
        ds = ds.shuffle(min(10000, len(image_paths)))

    ds = ds.map(
        lambda path, w, l: load_image_with_2attrs(path, w, l, image_size),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds, len(image_paths)


# ==================== TRAINING ====================

@tf.function
def train_step_2attr(generator, discriminator, g_opt, d_opt,
                     real_images, real_attrs, latent_dim, batch_size):
    """Training step for 2-attribute GAN."""

    noise = tf.random.normal([batch_size, latent_dim])

    # Add attribute noise (reduced for better control)
    attr_noise = tf.random.normal(tf.shape(real_attrs), mean=0.0, stddev=0.03)
    noisy_attrs = tf.clip_by_value(real_attrs + attr_noise, 0.0, 1.0)

    # Train discriminator
    with tf.GradientTape() as disc_tape:
        fake_images = generator([noise, noisy_attrs], training=True)
        real_output = discriminator([real_images, real_attrs], training=True)
        fake_output = discriminator([fake_images, noisy_attrs], training=True)
        d_loss = discriminator_loss(real_output, fake_output)

    d_gradients = disc_tape.gradient(d_loss, discriminator.trainable_variables)
    d_opt.apply_gradients(zip(d_gradients, discriminator.trainable_variables))

    # Train generator
    noise = tf.random.normal([batch_size, latent_dim])
    attr_noise = tf.random.normal(tf.shape(real_attrs), mean=0.0, stddev=0.03)
    noisy_attrs = tf.clip_by_value(real_attrs + attr_noise, 0.0, 1.0)

    with tf.GradientTape() as gen_tape:
        fake_images = generator([noise, noisy_attrs], training=True)
        fake_output = discriminator([fake_images, noisy_attrs], training=True)
        g_loss = generator_loss(fake_output)

    g_gradients = gen_tape.gradient(g_loss, generator.trainable_variables)
    g_opt.apply_gradients(zip(g_gradients, generator.trainable_variables))

    return g_loss, d_loss


def generate_2attr_preview(generator, latent_dim, epoch, out_dir):
    """
    Generate preview grid for 2 attributes.

    Layout:
    - Rows: Different wealthy values (0.0, 0.5, 1.0)
    - Cols: Different lively values (0.0, 0.33, 0.67, 1.0)
    """
    wealthy_vals = [0.0, 0.5, 1.0]
    lively_vals = [0.0, 0.33, 0.67, 1.0]

    fig, axes = plt.subplots(len(wealthy_vals), len(lively_vals),
                             figsize=(len(lively_vals)*2.5, len(wealthy_vals)*2.5))

    # Use same noise for all to see attribute effect
    noise = tf.random.normal([1, latent_dim])

    for i, wealthy in enumerate(wealthy_vals):
        for j, lively in enumerate(lively_vals):
            attrs = tf.constant([[wealthy, lively]], dtype=tf.float32)
            img = generator([noise, attrs], training=False)
            img = (img[0].numpy() + 1.0) / 2.0
            img = np.clip(img, 0, 1)

            ax = axes[i, j]
            ax.imshow(img)
            ax.axis('off')

            if i == 0:
                ax.set_title(f'lively={lively:.2f}', fontsize=10)
            if j == 0:
                ax.set_ylabel(f'wealthy={wealthy:.1f}', fontsize=10, rotation=0,
                             ha='right', va='center')

    plt.suptitle(f'Epoch {epoch} - Wealthy × Lively Control', fontsize=14)
    plt.tight_layout()

    out_path = Path(out_dir) / f'preview_2attr_epoch_{epoch:04d}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved 2-attr preview: {out_path.name}")


def train_2attr_gan(train_ds, val_ds, train_size, val_size,
                    generator, discriminator, g_opt, d_opt,
                    latent_dim, batch_size, epochs, out_dir,
                    save_every=10, preview_every=10, fid_every=5,
                    fid_num_samples=512, fid_batch_size=32, fid_stats_path='results/fid_real_stats_512.npz'):
    """Main training loop for 2-attribute GAN."""

    checkpoint_dir = out_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    ckpt = tf.train.Checkpoint(generator=generator, discriminator=discriminator,
                                g_opt=g_opt, d_opt=d_opt)
    ckpt_manager = tf.train.CheckpointManager(ckpt, str(checkpoint_dir), max_to_keep=5)

    # Metrics CSV
    metrics_path = out_dir / 'metrics.csv'
    with open(metrics_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'g_loss', 'd_loss', 'd_real_prob', 'd_fake_prob',
                         'd_real_logit', 'd_fake_logit', 'epoch_seconds'])

    print("\n" + "="*70)
    print("Starting 2-Attribute GAN Training (Wealthy + Lively)")
    print("="*70)

    total_start = time.time()
    num_batches = train_size // batch_size

    for epoch in range(epochs):
        epoch_start = time.time()
        g_losses = []
        d_losses = []

        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*70}")

        for batch_idx, (real_images, real_attrs) in enumerate(train_ds):
            g_loss, d_loss = train_step_2attr(
                generator, discriminator, g_opt, d_opt,
                real_images, real_attrs, latent_dim, batch_size
            )

            g_losses.append(float(g_loss))
            d_losses.append(float(d_loss))

            # Progress updates
            if batch_idx % 50 == 0 or batch_idx == num_batches - 1:
                avg_batch_time = (time.time() - epoch_start) / (batch_idx + 1)
                eta_mins = avg_batch_time * (num_batches - batch_idx - 1) / 60
                progress_pct = (batch_idx + 1) / num_batches * 100

                print(f"  Batch {batch_idx+1:4d}/{num_batches} ({progress_pct:5.1f}%) | "
                      f"G: {float(g_loss):.4f} D: {float(d_loss):.4f} | "
                      f"ETA: {eta_mins:.1f}m", flush=True)

        avg_g_loss = np.mean(g_losses)
        avg_d_loss = np.mean(d_losses)
        epoch_time = time.time() - epoch_start
        total_elapsed = time.time() - total_start

        # Validation metrics
        val_batch_iter = iter(val_ds)
        try:
            val_images, val_attrs = next(val_batch_iter)
            val_noise = tf.random.normal([tf.shape(val_images)[0], latent_dim])
            val_fake_images = generator([val_noise, val_attrs], training=False)

            d_real_logits = discriminator([val_images, val_attrs], training=False)
            d_fake_logits = discriminator([val_fake_images, val_attrs], training=False)

            d_real_prob = float(tf.reduce_mean(tf.sigmoid(d_real_logits)).numpy())
            d_fake_prob = float(tf.reduce_mean(tf.sigmoid(d_fake_logits)).numpy())
            d_real_logit = float(tf.reduce_mean(d_real_logits).numpy())
            d_fake_logit = float(tf.reduce_mean(d_fake_logits).numpy())
        except StopIteration:
            val_images = None  # Ensure val_images is defined
            d_real_prob = d_fake_prob = d_real_logit = d_fake_logit = 0.0

        # ETA
        avg_epoch_time = total_elapsed / (epoch + 1)
        eta_total_mins = (avg_epoch_time * (epochs - epoch - 1)) / 60

        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{epochs} Complete")
        print(f"  G Loss: {avg_g_loss:.4f} | D Loss: {avg_d_loss:.4f}")
        print(f"  D(real): {d_real_prob:.3f} | D(fake): {d_fake_prob:.3f}")
        print(f"  Epoch Time: {epoch_time:.1f}s ({epoch_time/60:.1f}m)")
        print(f"  Total Elapsed: {total_elapsed/60:.1f}m ({total_elapsed/3600:.2f}h)")
        if (epochs - epoch - 1) > 0:
            print(f"  ETA: {eta_total_mins:.1f}m ({eta_total_mins/60:.2f}h)")
        print(f"{'='*70}")

        # Save metrics
        with open(metrics_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg_g_loss, avg_d_loss,
                           d_real_prob, d_fake_prob, d_real_logit, d_fake_logit, epoch_time])

        # Preview and checkpoint
        if (epoch + 1) % preview_every == 0 or epoch == 0:
            generate_2attr_preview(generator, latent_dim, epoch, out_dir)

        if (epoch + 1) % save_every == 0:
            ckpt_path = ckpt_manager.save()
            print(f"  Saved checkpoint: {ckpt_path}")

        # FID computation
        if (epoch + 1) % fid_every == 0:
            # Sample fake images
            num_fake_samples = fid_num_samples // batch_size
            fake_images_ds = (
                tf.data.Dataset
                .from_tensor_slices(tf.random.normal([num_fake_samples, latent_dim]))
                .map(lambda noise: generator([noise, val_attrs], training=False),
                     num_parallel_calls=tf.data.AUTOTUNE)
                .unbatch()
                .batch(fid_batch_size)
            )

            # Calculate FID
            if val_images is not None:  # Ensure val_images is available
                fid_value = calculate_fid(
                    real_images=val_images, fake_images=fake_images_ds,
                    stats_path=fid_stats_path, batch_size=fid_batch_size
                )
                print(f"  FID at epoch {epoch+1}: {fid_value:.4f}")
            else:
                print("  Skipped FID computation: validation images not available")

    # Final save
    final_ckpt = ckpt_manager.save()
    total_time = time.time() - total_start

    print(f"\n{'='*70}")
    print("✅ 2-ATTRIBUTE GAN TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"  Total Epochs: {epochs}")
    print(f"  Total Time: {total_time/60:.1f}m ({total_time/3600:.2f}h)")
    print(f"  Final Checkpoint: {final_ckpt}")
    print(f"{'='*70}\n")


# ==================== FID CALCULATION ====================

from scipy.linalg import sqrtm
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input

def calculate_fid(real_images, fake_images, stats_path, batch_size=32):
    """
    Calculate the Frechet Inception Distance (FID) between real and fake images.

    Args:
        real_images: Dataset of real images.
        fake_images: Dataset of fake images.
        stats_path: Path to precomputed real image statistics (mean, covariance).
        batch_size: Batch size for processing images.

    Returns:
        FID score.
    """
    # Load InceptionV3 model
    model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

    def get_activations(dataset):
        activations = []
        for batch in dataset:
            batch = tf.image.resize(batch, (299, 299))
            batch = preprocess_input(batch.numpy())
            act = model.predict(batch, batch_size=batch_size)
            activations.append(act)
        return np.vstack(activations)

    # Load real image statistics
    with np.load(stats_path) as data:
        mu_real, sigma_real = data['mu'], data['sigma']

    # Compute activations for fake images
    fake_activations = get_activations(fake_images)
    mu_fake = np.mean(fake_activations, axis=0)
    sigma_fake = np.cov(fake_activations, rowvar=False)

    # Calculate FID
    diff = mu_real - mu_fake
    covmean = sqrtm(sigma_real @ sigma_fake)
    # Ensure covmean is complex before accessing the real attribute
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = np.sum(diff**2) + np.trace(sigma_real + sigma_fake - 2 * covmean)
    return fid


# ==================== MAIN ====================

def main():
    parser = argparse.ArgumentParser(description="Train 2-Attribute GAN (Wealthy + Lively)")

    # Data
    parser.add_argument('--train_csv', type=str,
                        default='data/wealthy_lively/wealthy_lively_scores_train.csv')
    parser.add_argument('--val_csv', type=str,
                        default='data/wealthy_lively/wealthy_lively_scores_val.csv')
    parser.add_argument('--image_dir', type=str, default='data/preprocessed_images')

    # Model
    parser.add_argument('--image_size', type=int, default=64)  # 64x64 recommended for 2-attr stability
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=128)

    # Training
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--preview_every', type=int, default=10)

    # FID Tracking
    parser.add_argument('--fid_every', type=int, default=5, help='Frequency of FID computation (epochs)')
    parser.add_argument('--fid_num_samples', type=int, default=512, help='Number of samples for FID computation')
    parser.add_argument('--fid_batch_size', type=int, default=32, help='Batch size for FID computation')
    parser.add_argument('--fid_stats_path', type=str, default='results/fid_real_stats_512.npz', help='Path to FID stats file')

    # Output
    parser.add_argument('--out_dir', type=str,
                        default=f'results/2attr_wealthy_lively_{datetime.now().strftime("%Y%m%d_%H%M%S")}')

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("2-Attribute Conditional GAN - Wealthy + Lively")
    print("="*70)
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Output: {out_dir}")
    print()

    # Validate input files exist
    print("Validating input files...")
    train_csv_path = Path(args.train_csv)
    val_csv_path = Path(args.val_csv)
    img_dir_path = Path(args.image_dir)

    if not train_csv_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {train_csv_path}")
    if not val_csv_path.exists():
        raise FileNotFoundError(f"Validation CSV not found: {val_csv_path}")
    if not img_dir_path.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir_path}")

    print(f"  ✅ Train CSV found: {train_csv_path}")
    print(f"  ✅ Val CSV found: {val_csv_path}")
    print(f"  ✅ Image dir found: {img_dir_path}")
    print()

    # Load datasets
    print("Loading datasets...")
    print(f"  Train CSV: {args.train_csv}")
    print(f"  Val CSV: {args.val_csv}")
    print(f"  Image dir: {args.image_dir}")

    try:
        train_ds, train_size = make_2attr_dataset(
            args.train_csv, args.image_dir, args.image_size, args.batch_size, shuffle=True
        )
        print(f"  ✅ Train dataset loaded: {train_size:,} images")
    except Exception as e:
        print(f"  ❌ ERROR loading training dataset: {e}")
        raise

    try:
        val_ds, val_size = make_2attr_dataset(
            args.val_csv, args.image_dir, args.image_size, args.batch_size, shuffle=False
        )
        print(f"  ✅ Val dataset loaded: {val_size:,} images")
    except Exception as e:
        print(f"  ❌ ERROR loading validation dataset: {e}")
        raise

    # Build models (num_attributes=2 for wealthy + lively)
    print("\nBuilding models...")
    generator = make_conditional_generator(args.image_size, args.latent_dim, num_attributes=2)
    discriminator = make_conditional_discriminator(args.image_size, num_attributes=2)

    print(f"\nGenerator parameters: {generator.count_params():,}")
    print(f"Discriminator parameters: {discriminator.count_params():,}")

    # Optimizers (D learns slower like successful 128x128 wealthy model)
    g_opt = tf.keras.optimizers.Adam(learning_rate=args.lr, beta_1=args.beta1)
    d_opt = tf.keras.optimizers.Adam(learning_rate=args.lr * 0.5, beta_1=args.beta1)

    # Train
    train_2attr_gan(
        train_ds, val_ds, train_size, val_size,
        generator, discriminator, g_opt, d_opt,
        args.latent_dim, args.batch_size, args.epochs,
        out_dir, args.save_every, args.preview_every, args.fid_every,
        args.fid_num_samples, args.fid_batch_size, args.fid_stats_path
    )


if __name__ == '__main__':
    import sys
    import traceback

    print("="*70)
    print("Starting 2-Attribute GAN Training Script")
    print("="*70)
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print()

    try:
        main()
    except Exception as e:
        print("\n" + "="*70)
        print("❌ FATAL ERROR IN TRAINING SCRIPT")
        print("="*70)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print()
        print("Full traceback:")
        print("-"*70)
        traceback.print_exc()
        print("="*70)
        sys.exit(1)

