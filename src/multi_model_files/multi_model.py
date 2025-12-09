#!/usr/bin/env python3
"""
conditional gan over a multidim vector of traits

USAGE:
  python multi_model.py \
    --csv_path ../data/all_attribute_scores.csv \
    --img_dir ../data/preprocessed_images \
    --out_dir ../results/multi_gan \
    --image_size 64 \
    --batch_size 64 \
    --epochs 1
"""

import os
import glob
import math
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


# helpers
IMG_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

def find_image_path(img_dir: Path, image_id: str):
    for ext in IMG_EXTS:
        p = img_dir / f"{image_id}{ext}"
        if p.exists():
            return str(p)


def load_metadata(csv_path: Path, img_dir: Path):
    df = pd.read_csv(csv_path)

    if "image_id" not in df.columns:
        raise ValueError()

    # All columns except image_id are treated as conditioning dims
    cond_cols = [c for c in df.columns if c != "image_id"]
    if len(cond_cols) == 0:
        raise ValueError()

    for c in cond_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    col_means = df[cond_cols].mean(skipna=True)
    df[cond_cols] = df[cond_cols].fillna(col_means)

    df[cond_cols] = df[cond_cols].clip(0.0, 1.0)

    paths = []
    conds = []
    missing = 0
    for _, row in df.iterrows():
        image_id = str(row["image_id"])
        p = find_image_path(img_dir, image_id)
        if p is None:
            missing += 1
            continue
        paths.append(p)
        conds.append(row[cond_cols].to_numpy(dtype=np.float32))

    if len(paths) == 0:
        raise ValueError()

    print(f"[load_metadata] matched {len(paths)} images. missing={missing}. cond_dim={len(cond_cols)}")

    return np.array(paths), np.array(conds, dtype=np.float32), cond_cols, col_means.to_dict()


def decode_and_preprocess(path, cond, image_size: int, augment: bool):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    img = tf.image.resize(img, [image_size, image_size], method="bilinear")

    if augment:
        img = tf.image.random_flip_left_right(img)
        # tiny jitter
        img = tf.image.random_brightness(img, 0.05)
        img = tf.image.random_contrast(img, 0.9, 1.1)

    # scale to [-1, 1] for tanh output
    img = img * 2.0 - 1.0
    return img, cond


def make_dataset(paths, conds, image_size: int, batch_size: int, shuffle: bool, augment: bool):
    ds = tf.data.Dataset.from_tensor_slices((paths, conds))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(paths), 8192), reshuffle_each_iteration=True)
    ds = ds.map(lambda p, c: decode_and_preprocess(p, c, image_size, augment),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    return ds


# MODEL comps

class FiLM(tf.keras.layers.Layer):
    def __init__(self, channels: int, name=None):
        super().__init__(name=name)
        self.channels = channels
        self.dense = tf.keras.layers.Dense(channels * 2)

    def call(self, x, c):
        gb = self.dense(c)
        gamma, beta = tf.split(gb, 2, axis=-1)
        gamma = 1.0 + 0.1 * tf.tanh(gamma)
        beta = 0.1 * tf.tanh(beta)
        gamma = gamma[:, None, None, :]
        beta = beta[:, None, None, :]

        return x * gamma + beta


# generator
def build_generator(image_size: int, latent_dim: int, cond_dim: int):
    z_in = tf.keras.Input(shape=(latent_dim,), name="z")
    c_in = tf.keras.Input(shape=(cond_dim,), name="c")

    x = tf.keras.layers.Concatenate()([z_in, c_in])
    x = tf.keras.layers.Dense(4 * 4 * 512, use_bias=False)(x)
    x = tf.keras.layers.Reshape((4, 4, 512))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = FiLM(512, name="film0")(x, c_in)

    def up_block(x, channels, name):
        x = tf.keras.layers.Conv2DTranspose(channels, 4, strides=2, padding="same", use_bias=False, name=name)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = FiLM(channels, name=f"{name}_film")(x, c_in)

        return x
    
    x = up_block(x, 256, "g_up1")
    x = up_block(x, 128, "g_up2")
    x = up_block(x, 64, "g_up3")

    if image_size >= 64:
        x = up_block(x, 32, "g_up4")
    if image_size >= 128:
        x = up_block(x, 16, "g_up5")
    if image_size >= 256:
        x = up_block(x, 8, "g_up6")

    out = tf.keras.layers.Conv2D(3, 3, padding="same", activation="tanh", name="g_out")(x)

    return tf.keras.Model([z_in, c_in], out, name="Generator")

# discriminator
def build_discriminator(image_size: int, cond_dim: int):
    x_in = tf.keras.Input(shape=(image_size, image_size, 3), name="x")
    c_in = tf.keras.Input(shape=(cond_dim,), name="c")

    x = x_in

    def down_block(x, channels, name):
        x = tf.keras.layers.Conv2D(channels, 4, strides=2, padding="same", name=name)(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)

        return x
    
    
    x = down_block(x, 64, "d_down1")  # /2
    x = down_block(x, 128, "d_down2")  # /4
    x = down_block(x, 256, "d_down3")  # /8
    x = down_block(x, 512, "d_down4")  # /16

    if image_size >= 128:
        x = down_block(x, 512, "d_down5")  # /32

    x = tf.keras.layers.GlobalAveragePooling2D()(x)  # h(x)  [B, F]
    feat_dim = x.shape[-1]

    # Unconditional logit
    uncond = tf.keras.layers.Dense(1, name="d_uncond")(x)  # [B,1]

    # Projection term
    c_embed = tf.keras.layers.Dense(feat_dim, use_bias=False, name="d_c_embed")(c_in)  # [B,F]
    proj = tf.reduce_sum(x * c_embed, axis=-1, keepdims=True)  # [B,1]
    logit = uncond + proj

    # Aux regressor head
    c_hat = tf.keras.layers.Dense(cond_dim, activation="sigmoid", name="d_reg")(x)

    return tf.keras.Model([x_in, c_in], [logit, c_hat], name="Discriminator")


# training

def h_d_loss(real_logits, fake_logits):
    loss_real = tf.reduce_mean(tf.nn.relu(1.0 - real_logits))
    loss_fake = tf.reduce_mean(tf.nn.relu(1.0 + fake_logits))
    return loss_real + loss_fake

def g_loss_from_logits(fake_logits):
    return -tf.reduce_mean(fake_logits)

def mae(a, b):
    return tf.reduce_mean(tf.abs(a - b))

def make_grid(images, nrow: int):
    images = (images + 1.0) / 2.0
    images = tf.clip_by_value(images, 0.0, 1.0).numpy()

    N, H, W, C = images.shape
    ncol = int(math.ceil(N / nrow))
    grid = np.zeros((ncol * H, nrow * W, C), dtype=np.float32)

    for idx in range(N):
        r = idx // nrow
        c = idx % nrow
        grid[r*H:(r+1)*H, c*W:(c+1)*W] = images[idx]

    return grid


def save_grid(images, out_path: Path, nrow: int, title: str = None):
    grid = make_grid(images, nrow=nrow)
    plt.figure(figsize=(nrow * 2.2, max(1, (grid.shape[0] / grid.shape[1]) * nrow * 2.2)))
    plt.axis("off")

    if title:
        plt.title(title)

    plt.imshow(grid)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--latent_dim", type=int, default=128)

    parser.add_argument("--lr_g", type=float, default=2e-4)
    parser.add_argument("--lr_d", type=float, default=2e-4)
    parser.add_argument("--lambda_reg", type=float, default=10.0)
    parser.add_argument("--augment", action="store_true")

    parser.add_argument("--save_every_epochs", type=int, default=1)
    parser.add_argument("--num_eval_samples", type=int, default=64)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir/"samples").mkdir(exist_ok=True)
    (out_dir/"sweeps").mkdir(exist_ok=True)
    (out_dir/"ckpt").mkdir(exist_ok=True)

    paths, conds, cond_cols, col_means = load_metadata(Path(args.csv_path), Path(args.img_dir))
    cond_dim = conds.shape[1]


    # save metadata
    with open(out_dir / "cond_meta.json", "w") as f:
        json.dump({"cond_cols": cond_cols, "col_means": col_means}, f, indent=2)

    ds = make_dataset(paths, conds, args.image_size, args.batch_size, shuffle=True, augment=args.augment)



    G = build_generator(args.image_size, args.latent_dim, cond_dim)
    D = build_discriminator(args.image_size, cond_dim)

    opt_g = tf.keras.optimizers.Adam(args.lr_g, beta_1=0.5, beta_2=0.999)
    opt_d = tf.keras.optimizers.Adam(args.lr_d, beta_1=0.5, beta_2=0.999)



    ckpt = tf.train.Checkpoint(G=G, D=D, opt_g=opt_g, opt_d=opt_d)
    manager = tf.train.CheckpointManager(ckpt, str(out_dir/ "ckpt"), max_to_keep=3)

    @tf.function

    def train_step(real_x, c):
        batch_size = tf.shape(real_x)[0]
        z = tf.random.normal([batch_size, args.latent_dim])

        # throw back
        with tf.GradientTape(persistent=True) as tape:
            fake_x = G([z, c], training=True)

            real_logits, real_c_hat = D([real_x, c], training=True)
            fake_logits, fake_c_hat = D([fake_x, c], training=True)

            d_adv = h_d_loss(real_logits, fake_logits)
            g_adv = g_loss_from_logits(fake_logits)

            d_reg = mae(real_c_hat, c)
            g_reg = mae(fake_c_hat, c)

            d_loss = d_adv + args.lambda_reg * d_reg
            g_loss = g_adv + args.lambda_reg * g_reg

        d_grads = tape.gradient(d_loss, D.trainable_variables)
        g_grads = tape.gradient(g_loss, G.trainable_variables)
        opt_d.apply_gradients(zip(d_grads, D.trainable_variables))
        opt_g.apply_gradients(zip(g_grads, G.trainable_variables))

        return d_adv, d_reg, g_adv, g_reg

    def sample_and_save(epoch: int):
        idx = np.random.choice(len(conds), size=args.num_eval_samples, replace=True)
        c = tf.convert_to_tensor(conds[idx], dtype=tf.float32)
        z = tf.random.normal([args.num_eval_samples, args.latent_dim])
        fake = G([z, c], training=False)

        _, c_hat = D([fake, c], training=False)
        cond_mae = tf.reduce_mean(tf.abs(c_hat - c)).numpy().item()

        save_grid(fake,
            out_dir/ "samples" / f"epoch_{epoch:04d}_samples.png",
            nrow=int(math.sqrt(args.num_eval_samples)),
            title=f"epoch {epoch} | cond_MAE={cond_mae:.3f}")
        print(f"[eval] epoch={epoch} cond_MAE(fake)={cond_mae:.3f}")

        return cond_mae

    def sweep_and_save(epoch: int):
        base_c = np.array([col_means.get(k, 0.5) for k in cond_cols], dtype=np.float32)[None, :]
        base_c = np.clip(base_c, 0.0, 1.0)

        z = tf.random.normal([1, args.latent_dim])
        steps = 6
        vals = np.linspace(0.0, 1.0, steps).astype(np.float32)

        for j, name in enumerate(cond_cols):
            c_list = []
            for v in vals:
                c = base_c.copy()
                c[0, j] = v
                c_list.append(c[0])
            c_batch = tf.convert_to_tensor(np.stack(c_list, axis=0), dtype=tf.float32)

            z_batch = tf.repeat(z, repeats=steps, axis=0)
            fake = G([z_batch, c_batch], training=False)

            # annotate 
            _, c_hat = D([fake, c_batch], training=False)
            c_hat_np = c_hat.numpy()

            dim_pred = c_hat_np[:, j]
            title = f"sweep {name} |req {vals[0]:.1f}->{vals[-1]:.1f} |pred {dim_pred[0]:.2f}->{dim_pred[-1]:.2f}"

            save_grid(fake,
                out_dir/"sweeps" /f"epoch_{epoch:04d}_sweep_{j:02d}_{name}.png",
                nrow=steps,
                title=title)

    # training
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        d_adv_m, d_reg_m, g_adv_m, g_reg_m = [], [], [], []

        for real_x, c in ds:
            d_adv, d_reg, g_adv, g_reg = train_step(real_x, c)
            d_adv_m.append(float(d_adv.numpy()))
            d_reg_m.append(float(d_reg.numpy()))
            g_adv_m.append(float(g_adv.numpy()))
            g_reg_m.append(float(g_reg.numpy()))
            global_step += 1

        print(
            f"[train] epoch={epoch} "
            f"D_adv={np.mean(d_adv_m):.3f} D_reg={np.mean(d_reg_m):.3f} "
            f"G_adv={np.mean(g_adv_m):.3f} G_reg={np.mean(g_reg_m):.3f}")

        if epoch % args.save_every_epochs == 0:
            sample_and_save(epoch)
            sweep_and_save(epoch)
            manager.save(checkpoint_number=epoch)

    print("done. results in results multimodal")


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    tf.get_logger().setLevel("ERROR")
    main()
