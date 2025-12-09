#!/usr/bin/env python3
"""
Multi-Attribute Conditional DCGAN for Controllable Image Generation

This model generates images conditioned on MULTIPLE attribute scores simultaneously.

Architecture:
- Generator: Takes [latent_noise (128-d), attributes (5-d)] → 64x64x3 image
- Discriminator: Takes [64x64x3 image, attributes (5-d)] → real/fake prediction

Attributes: wealthy, depressing, safety, lively, boring

Usage:
    python train_multi_attribute_gan.py \\
        --train_csv data/all_attribute_scores_train.csv \\
        --val_csv data/all_attribute_scores_val.csv \\
        --attribute_names wealthy depressing safety lively boring \\
        --epochs 100 \\
        --batch_size 128
"""

import time
import csv
from pathlib import Path
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime


# ==================== DATA LOADING ====================

def load_image_with_attributes(image_path, attributes, image_size):
    """Load and preprocess image, return with its attribute scores."""
    img_bytes = tf.io.read_file(image_path)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [image_size, image_size], antialias=True)
    img = img * 2.0 - 1.0  # Normalize to [-1, 1]
    return img, attributes


def make_multi_attribute_dataset(scores_csv, image_dir, attribute_names, image_size, batch_size, shuffle=True):
    """
    Create dataset that yields (image, attributes) pairs.

    Args:
        scores_csv: Path to CSV with columns [image_id, attr1, attr2, ...]
        image_dir: Directory containing {image_id}.jpg files
        attribute_names: List of attribute column names (e.g., ['wealthy', 'depressing', ...])
        image_size: Image size (64)
        batch_size: Batch size
        shuffle: Whether to shuffle

    Returns:
        tf.data.Dataset yielding (images, attributes) where:
            images: [B, H, W, 3] in [-1, 1]
            attributes: [B, num_attributes] in [0, 1]
    """
    # Load scores CSV
    df = pd.read_csv(scores_csv)

    print(f"  Loaded CSV with {len(df):,} rows")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Requested attributes: {attribute_names}")

    # Check for missing values
    missing_before = df[attribute_names].isnull().sum().sum()
    if missing_before > 0:
        print(f"  ⚠️  Found {missing_before:,} missing values")
        print(f"  Dropping rows with missing attributes...")
        df = df.dropna(subset=attribute_names)
        print(f"  Remaining rows: {len(df):,}")

    # Build paths and attribute vectors
    image_paths = []
    attribute_vectors = []

    for _, row in df.iterrows():
        img_path = Path(image_dir) / f"{row['image_id']}.jpg"
        if img_path.exists():
            image_paths.append(str(img_path))
            # Extract all attributes as a vector
            attr_vec = [float(row[attr]) for attr in attribute_names]
            attribute_vectors.append(attr_vec)

    print(f"  Loaded {len(image_paths):,} images with {len(attribute_names)} attributes")

    # Create dataset
    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    attr_ds = tf.data.Dataset.from_tensor_slices(attribute_vectors)
    ds = tf.data.Dataset.zip((path_ds, attr_ds))

    if shuffle:
        ds = ds.shuffle(min(10000, len(image_paths)))

    ds = ds.map(
        lambda path, attrs: load_image_with_attributes(path, attrs, image_size),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.batch(batch_size, drop_remainder=True)  # drop_remainder for stable training
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds, len(image_paths)


# ==================== MODEL ARCHITECTURE ====================

def make_multi_attribute_generator(image_size, latent_dim, num_attributes):
    """
    Multi-Attribute Conditional Generator: [latent, attributes] → image

    Args:
        image_size: Output image size (must be power of 2, >= 32)
        latent_dim: Latent noise dimension (e.g., 128)
        num_attributes: Number of attribute inputs (e.g., 5)

    Returns:
        Keras Model
    """
    assert image_size >= 32 and (image_size & (image_size - 1) == 0), "Image size must be power of 2"

    up_steps = int(np.log2(image_size) - 2)  # Number of upsampling layers
    start_res = 4
    start_ch = 512

    # Inputs
    latent_input = tf.keras.layers.Input(shape=(latent_dim,), name='latent')
    attr_input = tf.keras.layers.Input(shape=(num_attributes,), name='attributes')

    # Concatenate latent and ALL attributes
    combined = tf.keras.layers.Concatenate()([latent_input, attr_input])
    # Total dimension: latent_dim + num_attributes (e.g., 128 + 5 = 133)

    # Project to spatial
    x = tf.keras.layers.Dense(start_res * start_res * start_ch, use_bias=False)(combined)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Reshape((start_res, start_res, start_ch))(x)

    # Upsample progressively
    ch = start_ch
    for i in range(up_steps - 1):
        ch = max(ch // 2, 64)
        x = tf.keras.layers.Conv2DTranspose(ch, 4, strides=2, padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

    # Final layer to RGB
    outputs = tf.keras.layers.Conv2DTranspose(3, 4, strides=2, padding='same', activation='tanh', use_bias=False)(x)

    return tf.keras.Model([latent_input, attr_input], outputs, name='multi_attribute_generator')


def make_multi_attribute_discriminator(image_size, num_attributes):
    """
    Multi-Attribute Conditional Discriminator: [image, attributes] → real/fake logit

    Uses projection discriminator approach: attributes are projected into intermediate layers.

    Args:
        image_size: Input image size
        num_attributes: Number of attribute inputs (e.g., 5)

    Returns:
        Keras Model
    """
    # Inputs
    image_input = tf.keras.layers.Input(shape=(image_size, image_size, 3), name='image')
    attr_input = tf.keras.layers.Input(shape=(num_attributes,), name='attributes')

    # Image processing path
    x = image_input

    # Downsample
    down_steps = int(np.log2(image_size) - 2)
    ch = 64

    for i in range(down_steps):
        x = tf.keras.layers.Conv2D(ch, 4, strides=2, padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        ch = min(ch * 2, 512)

    # Final conv to spatial features
    x = tf.keras.layers.Conv2D(512, 4, strides=2, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    # Flatten image features
    x = tf.keras.layers.Flatten()(x)
    img_dim = x.shape[-1]  # Get the flattened dimension

    # Project attributes to same dimension as flattened image
    attr_proj = tf.keras.layers.Dense(img_dim)(attr_input)

    # Combine using element-wise product (projection discriminator style)
    combined = tf.keras.layers.Multiply()([x, attr_proj])

    # Additional dense layers
    combined = tf.keras.layers.Dense(512)(combined)
    combined = tf.keras.layers.LeakyReLU(0.2)(combined)

    # Output logit (no activation - we'll use from_logits=True in loss)
    output = tf.keras.layers.Dense(1)(combined)

    return tf.keras.Model([image_input, attr_input], output, name='multi_attribute_discriminator')


# ==================== LOSS FUNCTIONS ====================

bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    """Standard GAN discriminator loss."""
    real_loss = bce(tf.ones_like(real_output), real_output)
    fake_loss = bce(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    """Standard GAN generator loss."""
    return bce(tf.ones_like(fake_output), fake_output)


# ==================== TRAINING ====================

@tf.function
def train_step(generator, discriminator, g_opt, d_opt,
               real_images, real_attributes, latent_dim, batch_size):
    """Single training step."""

    # Sample random latent vectors
    noise = tf.random.normal([batch_size, latent_dim])

    # Add small noise to attribute values to prevent overfitting and mode collapse
    attribute_noise = tf.random.normal(tf.shape(real_attributes), mean=0.0, stddev=0.05)
    noisy_attributes = tf.clip_by_value(real_attributes + attribute_noise, 0.0, 1.0)

    # Train discriminator
    with tf.GradientTape() as disc_tape:
        # Generate fake images with noisy attribute vectors
        fake_images = generator([noise, noisy_attributes], training=True)

        # Discriminator predictions (use original attributes for discriminator)
        real_output = discriminator([real_images, real_attributes], training=True)
        fake_output = discriminator([fake_images, noisy_attributes], training=True)

        d_loss = discriminator_loss(real_output, fake_output)

    d_gradients = disc_tape.gradient(d_loss, discriminator.trainable_variables)
    d_opt.apply_gradients(zip(d_gradients, discriminator.trainable_variables))

    # Train generator
    noise = tf.random.normal([batch_size, latent_dim])
    attribute_noise = tf.random.normal(tf.shape(real_attributes), mean=0.0, stddev=0.05)
    noisy_attributes = tf.clip_by_value(real_attributes + attribute_noise, 0.0, 1.0)

    with tf.GradientTape() as gen_tape:
        fake_images = generator([noise, noisy_attributes], training=True)
        fake_output = discriminator([fake_images, noisy_attributes], training=True)
        g_loss = generator_loss(fake_output)

    g_gradients = gen_tape.gradient(g_loss, generator.trainable_variables)
    g_opt.apply_gradients(zip(g_gradients, generator.trainable_variables))

    return g_loss, d_loss


def generate_multi_attribute_preview(generator, latent_dim, attribute_names, epoch, out_dir, num_samples=6):
    """
    Generate preview showing effect of varying each attribute independently.

    Creates a grid where:
    - Each row shows one attribute varying from 0.0 to 1.0
    - Other attributes held constant at 0.5
    """
    num_attributes = len(attribute_names)
    num_steps = num_samples

    fig, axes = plt.subplots(num_attributes, num_steps, figsize=(num_steps * 2, num_attributes * 2))

    # Use same noise for all generations
    noise = tf.random.normal([1, latent_dim], seed=42)

    for attr_idx, attr_name in enumerate(attribute_names):
        # Vary this attribute, keep others at 0.5
        for step_idx, attr_val in enumerate(np.linspace(0.0, 1.0, num_steps)):
            # Create attribute vector: all 0.5 except current attribute
            attr_vec = np.ones((1, num_attributes)) * 0.5
            attr_vec[0, attr_idx] = attr_val

            attr_tensor = tf.constant(attr_vec, dtype=tf.float32)
            img = generator([noise, attr_tensor], training=False)
            img = (img[0].numpy() + 1.0) / 2.0  # [-1,1] → [0,1]
            img = np.clip(img, 0, 1)

            ax = axes[attr_idx, step_idx] if num_attributes > 1 else axes[step_idx]
            ax.imshow(img)
            ax.axis('off')

            # Labels
            if step_idx == 0:
                ax.set_ylabel(attr_name.capitalize(), fontsize=12, rotation=0, ha='right', va='center')
            if attr_idx == 0:
                ax.set_title(f'{attr_val:.1f}', fontsize=10)

    plt.tight_layout()
    out_path = Path(out_dir) / f'preview_epoch_{epoch:04d}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved preview: {out_path.name}")


def train_multi_attribute_gan(
    train_ds,
    val_ds,
    train_size,
    val_size,
    generator,
    discriminator,
    g_opt,
    d_opt,
    latent_dim,
    batch_size,
    epochs,
    out_dir,
    attribute_names,
    save_every=10,
    preview_every=10
):
    """Main training loop."""

    checkpoint_dir = out_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    ckpt = tf.train.Checkpoint(generator=generator, discriminator=discriminator,
                                g_opt=g_opt, d_opt=d_opt)
    ckpt_manager = tf.train.CheckpointManager(ckpt, directory=str(checkpoint_dir), max_to_keep=5)

    # Metrics CSV
    metrics_path = out_dir / 'metrics.csv'
    with open(metrics_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'g_loss', 'd_loss', 'time_seconds'])

    print("\n" + "=" * 70)
    print("Starting Multi-Attribute Conditional GAN Training")
    print("=" * 70)
    print(f"Attributes ({len(attribute_names)}): {', '.join(attribute_names)}")
    print("=" * 70)

    total_start_time = time.time()
    num_batches = train_size // batch_size

    # Initialize for final summary
    avg_g_loss = 0.0
    avg_d_loss = 0.0

    for epoch in range(epochs):
        epoch_start = time.time()

        # Training
        g_losses = []
        d_losses = []

        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*70}")

        for batch_idx, (real_images, real_attributes) in enumerate(train_ds):
            batch_start = time.time()

            g_loss, d_loss = train_step(
                generator, discriminator, g_opt, d_opt,
                real_images, real_attributes, latent_dim, batch_size
            )

            g_losses.append(float(g_loss))
            d_losses.append(float(d_loss))

            # Progress update every 50 batches or first/last batch
            if batch_idx % 50 == 0 or batch_idx == num_batches - 1:
                batch_time = time.time() - batch_start
                avg_batch_time = (time.time() - epoch_start) / (batch_idx + 1)
                eta_seconds = avg_batch_time * (num_batches - batch_idx - 1)
                eta_mins = eta_seconds / 60

                progress_pct = (batch_idx + 1) / num_batches * 100
                print(f"  Batch {batch_idx+1:4d}/{num_batches} ({progress_pct:5.1f}%) | "
                      f"G: {float(g_loss):.4f} D: {float(d_loss):.4f} | "
                      f"ETA: {eta_mins:.1f}m", flush=True)

        avg_g_loss = np.mean(g_losses)
        avg_d_loss = np.mean(d_losses)
        epoch_time = time.time() - epoch_start
        total_elapsed = time.time() - total_start_time

        # Calculate ETA for remaining epochs
        avg_epoch_time = total_elapsed / (epoch + 1)
        remaining_epochs = epochs - (epoch + 1)
        eta_total_mins = (avg_epoch_time * remaining_epochs) / 60

        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{epochs} Complete")
        print(f"  G Loss: {avg_g_loss:.4f} | D Loss: {avg_d_loss:.4f}")
        print(f"  Epoch Time: {epoch_time:.1f}s ({epoch_time/60:.1f}m)")
        print(f"  Total Elapsed: {total_elapsed/60:.1f}m ({total_elapsed/3600:.2f}h)")
        if remaining_epochs > 0:
            print(f"  ETA for {remaining_epochs} epochs: {eta_total_mins:.1f}m ({eta_total_mins/60:.2f}h)")
        print(f"{'='*70}")

        # Save metrics
        with open(metrics_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_g_loss, avg_d_loss, epoch_time])

        # Generate preview
        if (epoch + 1) % preview_every == 0 or epoch == 0:
            generate_multi_attribute_preview(generator, latent_dim, attribute_names, epoch + 1, out_dir)

        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            ckpt_path = ckpt_manager.save()
            print(f"  Saved checkpoint: {ckpt_path}")

    # Final save
    final_ckpt = ckpt_manager.save()

    # Final summary
    total_training_time = time.time() - total_start_time
    print(f"\n{'='*70}")
    print("✅ MULTI-ATTRIBUTE TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"  Attributes: {', '.join(attribute_names)}")
    print(f"  Total Epochs: {epochs}")
    print(f"  Total Time: {total_training_time/60:.1f}m ({total_training_time/3600:.2f}h)")
    print(f"  Avg Time per Epoch: {total_training_time/epochs:.1f}s")
    print(f"  Final G Loss: {avg_g_loss:.4f}")
    print(f"  Final D Loss: {avg_d_loss:.4f}")
    print(f"  Final Checkpoint: {final_ckpt}")
    print(f"{'='*70}\n")


# ==================== MAIN ====================

def main():
    parser = argparse.ArgumentParser(description="Train Multi-Attribute Conditional GAN")

    # Data
    parser.add_argument('--train_csv', type=str, default='data/all_attribute_scores_train.csv',
                        help='Training CSV with image_id and ALL attribute scores')
    parser.add_argument('--val_csv', type=str, default='data/all_attribute_scores_val.csv',
                        help='Validation CSV')
    parser.add_argument('--image_dir', type=str, default='data/preprocessed_images',
                        help='Directory with preprocessed images')
    parser.add_argument('--attribute_names', type=str, nargs='+',
                        default=['wealthy', 'depressing', 'safety', 'lively', 'boring', 'beautiful'],
                        help='Names of attribute columns in CSV (space-separated)')

    # Model
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=128)

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--preview_every', type=int, default=10)

    # Output
    parser.add_argument('--out_dir', type=str,
                        default=f'results/multi_attribute_gan_{datetime.now().strftime("%Y%m%d_%H%M%S")}')

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    num_attributes = len(args.attribute_names)

    print("=" * 70)
    print("Multi-Attribute Conditional GAN - Urban Scene Generation")
    print("=" * 70)
    print(f"Attributes ({num_attributes}): {', '.join(args.attribute_names)}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Latent dim: {args.latent_dim}")
    print(f"Total input dim: {args.latent_dim + num_attributes}")
    print(f"Epochs: {args.epochs}")
    print(f"Output: {out_dir}")
    print()

    # Load datasets
    print("Loading datasets...")
    train_ds, train_size = make_multi_attribute_dataset(
        args.train_csv, args.image_dir, args.attribute_names,
        args.image_size, args.batch_size, shuffle=True
    )
    val_ds, val_size = make_multi_attribute_dataset(
        args.val_csv, args.image_dir, args.attribute_names,
        args.image_size, args.batch_size, shuffle=False
    )

    print(f"\nDataset Summary:")
    print(f"  Training images: {train_size:,}")
    print(f"  Validation images: {val_size:,}")
    print(f"  Batches per epoch: {train_size // args.batch_size}")
    print()

    # Build models
    print("Building models...")
    generator = make_multi_attribute_generator(
        args.image_size, args.latent_dim, num_attributes
    )
    discriminator = make_multi_attribute_discriminator(
        args.image_size, num_attributes
    )

    # Print model summaries
    print("\nGenerator:")
    generator.summary()
    print("\nDiscriminator:")
    discriminator.summary()

    # Optimizers
    g_opt = tf.keras.optimizers.Adam(learning_rate=args.lr, beta_1=args.beta1)
    d_opt = tf.keras.optimizers.Adam(learning_rate=args.lr, beta_1=args.beta1)

    # Train
    train_multi_attribute_gan(
        train_ds, val_ds, train_size, val_size,
        generator, discriminator,
        g_opt, d_opt,
        args.latent_dim, args.batch_size, args.epochs,
        out_dir, args.attribute_names,
        args.save_every, args.preview_every
    )


if __name__ == '__main__':
    main()

