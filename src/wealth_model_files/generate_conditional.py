#!/usr/bin/env python3
"""
Generate conditional images from command line (no UI needed).

Usage:
    # Generate grid with varying attribute values
    python generate_conditional.py \\
        --checkpoint_dir ../results/conditional_wealthy_*/checkpoints/ \\
        --attribute_values 0.0 0.2 0.4 0.6 0.8 1.0 \\
        --num_samples 5 \\
        --out_path results/wealthy_grid.png

    # Generate many individual samples at specific attribute value
    python generate_conditional.py \\
        --checkpoint_dir ../results/conditional_wealthy_*/checkpoints/ \\
        --attribute_value 0.8 \\
        --num_samples 100 \\
        --output_mode individual \\
        --out_dir results/wealthy_0.8_samples/
"""

import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def make_conditional_generator(image_size, latent_dim, num_attributes=1):
    """Recreate conditional generator architecture."""
    assert image_size >= 32 and (image_size & (image_size - 1) == 0)

    up_steps = int(np.log2(image_size) - 2)
    start_res = 4
    start_ch = 512

    latent_input = tf.keras.layers.Input(shape=(latent_dim,), name='latent')
    attr_input = tf.keras.layers.Input(shape=(num_attributes,), name='attribute')

    combined = tf.keras.layers.Concatenate()([latent_input, attr_input])

    x = tf.keras.layers.Dense(start_res * start_res * start_ch, use_bias=False)(combined)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Reshape((start_res, start_res, start_ch))(x)

    ch = start_ch
    for i in range(up_steps - 1):
        ch = max(ch // 2, 64)
        x = tf.keras.layers.Conv2DTranspose(ch, 4, strides=2, padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

    outputs = tf.keras.layers.Conv2DTranspose(3, 4, strides=2, padding='same', activation='tanh', use_bias=False)(x)

    return tf.keras.Model([latent_input, attr_input], outputs, name='conditional_generator')


def load_generator(checkpoint_dir, image_size, latent_dim, num_attributes=1):
    """Load generator from checkpoint."""
    gen = make_conditional_generator(image_size, latent_dim, num_attributes)

    # Create checkpoint with dummy discriminator
    dummy_disc_input_img = tf.keras.layers.Input(shape=(image_size, image_size, 3))
    dummy_disc_input_attr = tf.keras.layers.Input(shape=(num_attributes,))
    dummy_disc_out = tf.keras.layers.Flatten()(dummy_disc_input_img)
    dummy_disc_out = tf.keras.layers.Dense(1)(dummy_disc_out)
    dummy_disc = tf.keras.Model([dummy_disc_input_img, dummy_disc_input_attr], dummy_disc_out)

    g_opt = tf.keras.optimizers.Adam()
    d_opt = tf.keras.optimizers.Adam()

    ckpt = tf.train.Checkpoint(generator=gen, discriminator=dummy_disc, g_opt=g_opt, d_opt=d_opt)
    ckpt_mgr = tf.train.CheckpointManager(ckpt, directory=str(checkpoint_dir), max_to_keep=None)

    if not ckpt_mgr.latest_checkpoint:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

    print(f"Loading: {ckpt_mgr.latest_checkpoint}")
    ckpt.restore(ckpt_mgr.latest_checkpoint).expect_partial()

    return gen


def generate_grid(gen, latent_dim, attribute_values, num_samples, seed=None):
    """
    Generate grid of images.

    Args:
        gen: Generator model
        latent_dim: Latent dimension
        attribute_values: List of attribute values to test
        num_samples: Number of samples (rows) per attribute value
        seed: Random seed

    Returns:
        Grid of images [num_samples, len(attribute_values), H, W, 3]
    """
    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)

    grid = []

    for sample_idx in range(num_samples):
        # Same noise for this row
        noise = tf.random.normal([1, latent_dim])

        row = []
        for attr_val in attribute_values:
            attr_tensor = tf.constant([[attr_val]], dtype=tf.float32)
            img = gen([noise, attr_tensor], training=False)
            img = (img[0].numpy() + 1.0) / 2.0
            img = np.clip(img, 0, 1)
            row.append(img)

        grid.append(row)

    return np.array(grid)


def generate_samples(gen, latent_dim, attribute_value, num_samples, seed=None):
    """
    Generate multiple samples at a single attribute value.

    Returns:
        Array of images [num_samples, H, W, 3]
    """
    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)

    noise = tf.random.normal([num_samples, latent_dim])
    attr_tensor = tf.constant([[attribute_value]] * num_samples, dtype=tf.float32)

    images = gen([noise, attr_tensor], training=False)
    images = (images.numpy() + 1.0) / 2.0
    images = np.clip(images, 0, 1)

    return images


def save_grid(grid, out_path, attribute_values, attribute_name):
    """Save grid as single image with labels."""
    num_samples, num_values = grid.shape[:2]

    fig, axes = plt.subplots(num_samples, num_values,
                              figsize=(num_values * 2.2, num_samples * 2.2))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        for j, attr_val in enumerate(attribute_values):
            ax = axes[i, j]
            ax.imshow(grid[i, j])
            ax.axis('off')

            if i == 0:
                ax.set_title(f'{attribute_name}={attr_val:.2f}', fontsize=10)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Saved grid: {out_path}")


def save_individual(images, out_dir, prefix='sample', attribute_value=None):
    """Save images individually."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, img in enumerate(images):
        if attribute_value is not None:
            filename = f"{prefix}_attr{attribute_value:.2f}_{i:04d}.png"
        else:
            filename = f"{prefix}_{i:04d}.png"

        out_path = out_dir / filename
        plt.imsave(out_path, img)

    print(f"✅ Saved {len(images)} images to: {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Generate conditional GAN images")

    # Required
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Path to checkpoint directory')

    # Generation mode
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--attribute_values', type=float, nargs='+',
                       help='Multiple attribute values for grid (e.g., 0.0 0.2 0.4 0.6 0.8 1.0)')
    group.add_argument('--attribute_value', type=float,
                       help='Single attribute value for batch generation')

    # Model params
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--attribute_name', type=str, default='wealthy',
                        help='Attribute name for labeling')

    # Generation params
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples (rows in grid mode, total in single mode)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')

    # Output
    parser.add_argument('--output_mode', type=str, default='grid', choices=['grid', 'individual'],
                        help='Output mode')
    parser.add_argument('--out_path', type=str, default=None,
                        help='Output path for grid image')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Output directory for individual images')

    args = parser.parse_args()

    # Validate output paths
    if args.output_mode == 'grid' and args.out_path is None:
        args.out_path = 'results/generated_grid.png'
    if args.output_mode == 'individual' and args.out_dir is None:
        args.out_dir = 'results/generated_samples'

    print("=" * 70)
    print("Conditional GAN Image Generation")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"Mode: {args.output_mode}")
    print(f"Samples: {args.num_samples}")
    if args.seed is not None:
        print(f"Seed: {args.seed}")
    print("=" * 70)

    # Load generator
    print("\nLoading generator...")
    gen = load_generator(Path(args.checkpoint_dir), args.image_size, args.latent_dim)
    print("✅ Loaded successfully!")

    # Generate
    if args.attribute_values is not None:
        # Grid mode
        print(f"\nGenerating grid with attribute values: {args.attribute_values}")
        grid = generate_grid(gen, args.latent_dim, args.attribute_values,
                            args.num_samples, seed=args.seed)

        if args.output_mode == 'grid':
            save_grid(grid, args.out_path, args.attribute_values, args.attribute_name)
        else:
            # Save grid as individual images
            all_images = grid.reshape(-1, *grid.shape[2:])
            save_individual(all_images, args.out_dir, prefix=f'{args.attribute_name}_grid')

    else:
        # Single attribute value mode
        print(f"\nGenerating {args.num_samples} samples at {args.attribute_name}={args.attribute_value}")
        images = generate_samples(gen, args.latent_dim, args.attribute_value,
                                 args.num_samples, seed=args.seed)

        if args.output_mode == 'grid':
            # Arrange in grid
            grid_cols = int(np.ceil(np.sqrt(args.num_samples)))
            grid_rows = int(np.ceil(args.num_samples / grid_cols))

            fig = plt.figure(figsize=(grid_cols * 2.2, grid_rows * 2.2))
            for i, img in enumerate(images):
                ax = plt.subplot(grid_rows, grid_cols, i + 1)
                ax.imshow(img)
                ax.axis('off')

            plt.tight_layout()
            fig.savefig(args.out_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"✅ Saved grid: {args.out_path}")
        else:
            save_individual(images, args.out_dir, prefix=args.attribute_name,
                          attribute_value=args.attribute_value)

    print("\n" + "=" * 70)
    print("✅ Generation complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()

