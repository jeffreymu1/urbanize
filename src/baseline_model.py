import os
import time
import csv
from pathlib import Path
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


#!/usr/bin/env python3
# train a BASELINE gan that gives some interesting images with urban data

## USAGE: 
"""
  python baseline_model.py \
    --data_dir ../data/preprocessed_images/ \
    --out_dir ../results/ \
    --image_size 64 \
    --batch_size 128 \
    --epochs 1
"""


# DATA WRANGLING HELPERS
# gets the images files (think all are jpg but just in case)
def list_image_files(data_dir: Path):
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    files = []

    for p in patterns:
        files.extend(tf.io.gfile.glob(str(data_dir/p)))

    return sorted(files)

# preprocess and create dataset
def make_dataset(files, image_size: int, batch_size: int, shuffle_buffer: int, shuffle: bool):
    autotune = tf.data.AUTOTUNE
    ds = tf.data.Dataset.from_tensor_slices(files)

    # convert to tensor
    def decode_and_preprocess(path):
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])

        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [image_size, image_size], antialias=True)

        # normalize
        img = img * 2.0 - 1.0
        return img
    if shuffle:
        ds = ds.shuffle(min(shuffle_buffer, len(files)))
    ds = ds.map(decode_and_preprocess, num_parallel_calls=autotune)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(autotune)

    return ds

## EVAL METRICS
def disc_confidence(disc, real_images, fake_images):
    real_p = tf.sigmoid(disc(real_images, training=False))
    fake_p = tf.sigmoid(disc(fake_images, training=False))

    return float(tf.reduce_mean(real_p).numpy()), float(tf.reduce_mean(fake_p).numpy())



# MODEL GENERATOR (DCGAN)
# since image sizes are power of 2, upsamples by factor of 2
def make_generator(image_size: int, latent_dim: int) -> tf.keras.Model:
    # check this is true
    assert image_size >= 32 and (image_size & (image_size - 1) == 0)

    up_steps = int(np.log2(image_size)-2)
    start_res = 4
    start_ch = 512

    inputs = tf.keras.layers.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(start_res * start_res * start_ch, use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Reshape((start_res, start_res, start_ch))(x)

    ch = start_ch
    for _ in range(up_steps - 1):
        ch = max(ch //2, 64)
        x = tf.keras.layers.Conv2DTranspose(ch, 4, strides=2, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

    outputs = tf.keras.layers.Conv2DTranspose(3, 4, strides=2, padding="same", use_bias=False, activation="tanh")(x)

    return tf.keras.Model(inputs, outputs, name="generator")

# MODEL DISCRIMINATOR (DCGAN)
def make_discriminator(image_size: int) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(image_size, image_size, 3))
    x = inputs

    chs = [64, 128, 256, 512] if image_size >= 64 else [64, 128, 256]
    for ch in chs:
        x = tf.keras.layers.Conv2D(ch, 4, strides=2, padding="same")(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Flatten()(x)
    logits = tf.keras.layers.Dense(1)(x)

    return tf.keras.Model(inputs, logits, name="discriminator")


# preview grid (chat)
def save_preview_grid(generator: tf.keras.Model, out_path: Path, latent_dim: int, n: int = 16, seed=None):
    if seed is None:
        seed = tf.random.normal([n, latent_dim])
    preds = generator(seed, training=False)
    preds = (preds + 1.0) / 2.0
    preds = tf.clip_by_value(preds, 0.0, 1.0)

    cols = int(np.sqrt(n))
    rows = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(cols * 2.2, rows * 2.2))
    for i in range(n):
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(preds[i].numpy())
        ax.axis("off")

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/AADB/datasetImages_warp256")
    parser.add_argument("--out_dir", type=str, default="../results/baseline_gan/")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--shuffle_buffer", type=int, default=20000)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--num_preview", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir /"checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir.resolve()}")

    print("TensorFlow:", tf.__version__)
    print("GPU devices:", tf.config.list_physical_devices("GPU"))
    print("Data dir:", data_dir.resolve())
    print("Out dir :", out_dir.resolve())

    files = list_image_files(data_dir)
    print("Found images:", len(files))
    if len(files) == 0:
        raise RuntimeError("no images found (SEE SOURCE DIR)")

    ds_train = make_dataset(files, args.image_size, args.batch_size, args.shuffle_buffer, shuffle=True)
    ds_eval = make_dataset(files, args.image_size, args.batch_size, args.shuffle_buffer, shuffle=False)
    
    gen = make_generator(args.image_size, args.latent_dim)
    disc = make_discriminator(args.image_size)
    print(gen.summary())
    print(disc.summary())

    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def g_loss_fn(fake_logits):
        return bce(tf.ones_like(fake_logits), fake_logits)

    def d_loss_fn(real_logits, fake_logits):
        real_loss = bce(tf.ones_like(real_logits), real_logits)
        fake_loss = bce(tf.zeros_like(fake_logits), fake_logits)
        return real_loss + fake_loss

    g_opt = tf.keras.optimizers.Adam(learning_rate=args.lr, beta_1=args.beta1)
    d_opt = tf.keras.optimizers.Adam(learning_rate=args.lr, beta_1=args.beta1)

    ckpt = tf.train.Checkpoint(generator=gen, discriminator=disc, g_opt=g_opt, d_opt=d_opt)
    ckpt_mgr = tf.train.CheckpointManager(ckpt, directory=str(ckpt_dir), max_to_keep=5)

    if args.resume and ckpt_mgr.latest_checkpoint:
        print("resuming from:", ckpt_mgr.latest_checkpoint)
        ckpt.restore(ckpt_mgr.latest_checkpoint).expect_partial()

    preview_seed = tf.random.normal([args.num_preview, args.latent_dim])
    save_preview_grid(gen, out_dir/"preview_epoch_0000.png", args.latent_dim, n=args.num_preview, seed=preview_seed)

    @tf.function
    def train_step(real_images):
        bs = tf.shape(real_images)[0]
        noise = tf.random.normal([bs, args.latent_dim])

        with tf.GradientTape() as gtape, tf.GradientTape() as dtape:
            fake_images = gen(noise, training=True)
            real_logits = disc(real_images, training=True)
            fake_logits = disc(fake_images, training=True)

            gl = g_loss_fn(fake_logits)
            dl = d_loss_fn(real_logits, fake_logits)

        g_grads = gtape.gradient(gl, gen.trainable_variables)
        d_grads = dtape.gradient(dl, disc.trainable_variables)
        g_opt.apply_gradients(zip(g_grads, gen.trainable_variables))
        d_opt.apply_gradients(zip(d_grads, disc.trainable_variables))

        return gl, dl

    history_g, history_d = [], []

    metrics_rows = []
    eval_ds_iter = iter(ds_eval)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        g_vals, d_vals = [], []

        for real_batch in ds_train:
            gl, dl = train_step(real_batch)
            g_vals.append(float(gl))
            d_vals.append(float(dl))

        g_mean = float(np.mean(g_vals))
        d_mean = float(np.mean(d_vals))
        history_g.append(g_mean)
        history_d.append(d_mean)

        print(f"epoch {epoch:03d}/{args.epochs} | g: {g_mean:.4f} | d: {d_mean:.4f} | {time.time()-t0:.1f}s")

        try:
            real_batch_eval = next(eval_ds_iter)
        except StopIteration:
            eval_ds_iter = iter(ds_eval)
            real_batch_eval = next(eval_ds_iter)

        z = tf.random.normal([tf.shape(real_batch_eval)[0], args.latent_dim])
        fake_batch = gen(z, training=False)

        d_real, d_fake = disc_confidence(disc, real_batch_eval, fake_batch)
        d_real_logit = float(tf.reduce_mean(disc(real_batch_eval, training=False)).numpy())
        d_fake_logit = float(tf.reduce_mean(disc(fake_batch, training=False)).numpy())

        print(f"D(real)={d_real:.3f}  D(fake)={d_fake:.3f}")

        metrics_rows.append({
            "epoch": epoch,
            "g_loss": g_mean,
            "d_loss": d_mean,
            "d_real_prob": d_real,
            "d_fake_prob": d_fake,
            "d_real_logit": d_real_logit,
            "d_fake_logit": d_fake_logit,
            "epoch_seconds": time.time() - t0,
        })

        if epoch == 1 or epoch == args.epochs or (epoch % args.save_every == 0):
            prev_path = out_dir / f"preview_epoch_{epoch:04d}.png"
            save_preview_grid(gen, prev_path, args.latent_dim, n=args.num_preview, seed=preview_seed)
            ckpt_path = ckpt_mgr.save()
            print("saved preview:", prev_path)
            print("saved ckpt:", ckpt_path)

    # save loss curve
    plt.figure(figsize=(7, 4))
    plt.plot(history_g, label="gen loss")
    plt.plot(history_d, label="disc loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir/"loss_curve.png", dpi=300, bbox_inches="tight")
    plt.close()


    # save the metrics
    csv_path = out_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metrics_rows)

    print("saved metrics csv:", csv_path)


    print("done, outputs in:", out_dir.resolve())


if __name__ == "__main__":
    main()
