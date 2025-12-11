#!/usr/bin/env python3
"""
Generate images from 2-Attribute GAN (Wealthy + Lively)

Usage:
    # Generate grid with varying wealthy/lively values
    python generate_2attr_images.py \
        --checkpoint_dir results/2attr_wealthy_lively_01/checkpoints/ \
        --wealthy_values 0.0 0.5 1.0 \
        --lively_values 0.0 0.33 0.67 1.0 \
        --num_samples 1 \
        --out_path results/2attr_grid.png

    # Generate individual samples at specific attribute values
    python generate_2attr_images.py \
        --checkpoint_dir results/2attr_wealthy_lively_01/checkpoints/ \
        --wealthy 0.8 --lively 0.2 \
        --num_samples 10 \
        --output_mode individual \
        --out_dir results/wealthy0.8_lively0.2_samples/
"""
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from train_conditional_gan import make_conditional_generator
from baseline_model import setup_fid, compute_fid
import sys

def load_generator(checkpoint_dir, image_size, latent_dim, checkpoint_name=None):
    gen = make_conditional_generator(image_size, latent_dim, num_attributes=2)
    print("Generator weights before loading checkpoint:")
    print([w.numpy() for w in gen.weights])
    sys.stdout.flush()
    dummy_disc_input_img = tf.keras.layers.Input(shape=(image_size, image_size, 3))
    dummy_disc_input_attr = tf.keras.layers.Input(shape=(2,))
    x = tf.keras.layers.Flatten()(dummy_disc_input_img)
    y = tf.keras.layers.Concatenate()([x, dummy_disc_input_attr])
    out = tf.keras.layers.Dense(1)(y)
    dummy_disc = tf.keras.Model([dummy_disc_input_img, dummy_disc_input_attr], out)
    g_opt = tf.keras.optimizers.Adam()
    d_opt = tf.keras.optimizers.Adam()
    ckpt = tf.train.Checkpoint(generator=gen, discriminator=dummy_disc, g_opt=g_opt, d_opt=d_opt)
    if checkpoint_name:
        checkpoint_path = str(Path(checkpoint_dir) / checkpoint_name)
        print(f"Loading specific checkpoint: {checkpoint_path}")
        ckpt.restore(checkpoint_path).expect_partial()
    else:
        ckpt_mgr = tf.train.CheckpointManager(ckpt, directory=str(checkpoint_dir), max_to_keep=None)
        if not ckpt_mgr.latest_checkpoint:
            print(f"No checkpoint found in {checkpoint_dir}")
            sys.stdout.flush()
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
        print(f"Loading latest checkpoint: {ckpt_mgr.latest_checkpoint}")
        sys.stdout.flush()
        ckpt.restore(ckpt_mgr.latest_checkpoint).expect_partial()
    print("Generator weights after loading checkpoint:")
    print([w.numpy() for w in gen.weights])
    sys.stdout.flush()
    return gen

def generate_grid(gen, latent_dim, wealthy_values, lively_values, num_samples, seed=None):
    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)
    grid = []
    for i, wealthy in enumerate(wealthy_values):
        row = []
        for j, lively in enumerate(lively_values):
            imgs = []
            for _ in range(num_samples):
                noise = tf.random.normal([1, latent_dim])
                attrs = tf.constant([[wealthy, lively]], dtype=tf.float32)
                img = gen([noise, attrs], training=False)
                img = (img[0].numpy() + 1.0) / 2.0
                img = np.clip(img, 0, 1)
                imgs.append(img)
            row.append(np.stack(imgs))
        grid.append(row)
    return np.array(grid)  # shape: [wealthy, lively, num_samples, H, W, 3]

def save_grid(grid, wealthy_values, lively_values, out_path):
    W, L, N, H, Wd, C = grid.shape
    fig, axes = plt.subplots(W, L, figsize=(L*2.5, W*2.5))
    for i in range(W):
        for j in range(L):
            img = grid[i, j, 0]  # show first sample per cell
            ax = axes[i, j]
            ax.imshow(img)
            ax.axis('off')
            if i == 0:
                ax.set_title(f'lively={lively_values[j]:.2f}', fontsize=10)
            if j == 0:
                ax.set_ylabel(f'wealthy={wealthy_values[i]:.2f}', fontsize=10, rotation=0, ha='right', va='center')
    plt.suptitle('Wealthy × Lively Control', fontsize=14)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Saved grid: {out_path}")

def generate_samples(gen, latent_dim, wealthy, lively, num_samples, seed=None):
    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)
    noise = tf.random.normal([num_samples, latent_dim])
    attrs = tf.constant([[wealthy, lively]] * num_samples, dtype=tf.float32)
    print(f"Noise sample: {noise.numpy()[0]}")
    print(f"Attrs sample: {attrs.numpy()[0]}")
    sys.stdout.flush()
    images = gen([noise, attrs], training=False)
    images_np = (images.numpy() + 1.0) / 2.0
    images_np = np.clip(images_np, 0, 1)
    print(f"Sample output stats: min={images_np.min()}, max={images_np.max()}, mean={images_np.mean()}")
    print(f"Sample image tensor: {images_np[0]}")
    sys.stdout.flush()
    return images_np

def save_individual(images, out_dir, wealthy, lively):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        filename = f"sample_wealthy{wealthy:.2f}_lively{lively:.2f}_{i:04d}.png"
        out_path = out_dir / filename
        plt.imsave(out_path, img)
    print(f"✅ Saved {len(images)} images to: {out_dir}/")

def main():
    parser = argparse.ArgumentParser(description="Generate images from 2-Attribute GAN")
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--checkpoint_name', type=str, default=None, help='Specific checkpoint to load (e.g. ckpt-14)')
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--output_mode', type=str, default='grid', choices=['grid', 'individual'])
    parser.add_argument('--wealthy_values', type=float, nargs='+', help='Wealthy values for grid')
    parser.add_argument('--lively_values', type=float, nargs='+', help='Lively values for grid')
    parser.add_argument('--wealthy', type=float, help='Wealthy value for samples')
    parser.add_argument('--lively', type=float, help='Lively value for samples')
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--out_path', type=str, default='results/generated_2attr_grid.png')
    parser.add_argument('--out_dir', type=str, default='results/generated_2attr_samples')
    parser.add_argument('--fid_stats_path', type=str, default=None, help='Path to FID stats file')
    parser.add_argument('--fid_num_samples', type=int, default=512, help='Number of samples for FID computation')
    parser.add_argument('--fid_batch_size', type=int, default=32, help='Batch size for FID computation')
    args = parser.parse_args()
    print("="*70)
    print("2-Attribute GAN Image Generation")
    print("="*70)
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print(f"Checkpoint name: {args.checkpoint_name}")
    print(f"Mode: {args.output_mode}")
    print(f"Samples: {args.num_samples}")
    print("="*70)
    print("\nLoading generator...")
    gen = load_generator(Path(args.checkpoint_dir), args.image_size, args.latent_dim, args.checkpoint_name)
    print("✅ Loaded successfully!")
    if args.output_mode == 'grid':
        if args.wealthy_values is None or args.lively_values is None:
            raise ValueError("wealthy_values and lively_values must be provided for grid mode")
        print(f"\nGenerating grid with wealthy={args.wealthy_values}, lively={args.lively_values}")
        grid = generate_grid(gen, args.latent_dim, args.wealthy_values, args.lively_values, args.num_samples, seed=args.seed)
        save_grid(grid, args.wealthy_values, args.lively_values, args.out_path)
    else:
        if args.wealthy is None or args.lively is None:
            raise ValueError("wealthy and lively must be provided for individual mode")
        print(f"\nGenerating {args.num_samples} samples at wealthy={args.wealthy}, lively={args.lively}")
        images = generate_samples(gen, args.latent_dim, args.wealthy, args.lively, args.num_samples, seed=args.seed)
        save_individual(images, args.out_dir, args.wealthy, args.lively)
    if args.fid_stats_path:
        print("\nSetting up FID computation...")
        fid_inception, fid_real_mu, fid_real_sigma = setup_fid(None, Path(args.fid_stats_path).parent, args.fid_num_samples)
        print("✅ FID setup complete!")
        print("\nComputing FID...")
        fid_value = compute_fid(gen, args.latent_dim, fid_inception, fid_real_mu, fid_real_sigma, args.fid_num_samples, args.fid_batch_size)
        print(f"FID: {fid_value}")
    print("\n" + "="*70)
    print("✅ Generation complete!")
    print("="*70)

if __name__ == '__main__':
    main()
