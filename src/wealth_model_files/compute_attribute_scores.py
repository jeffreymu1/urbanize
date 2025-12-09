#!/usr/bin/env python3
"""
Compute attribute scores from pairwise comparison data.

Uses Bradley-Terry model to convert pairwise comparisons into continuous scores (0-1)
for each image and attribute.

Usage:
    python compute_attribute_scores.py --attribute wealthy --output ../data/wealthy_scores.csv
    python compute_attribute_scores.py --attribute all --output ../data/all_attribute_scores.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import json


def bradley_terry_simple(comparisons, max_iterations=100, epsilon=1e-6):
    """
    Simple Bradley-Terry model implementation using iterative algorithm.

    Args:
        comparisons: List of (winner_id, loser_id) tuples
        max_iterations: Maximum number of iterations
        epsilon: Convergence threshold

    Returns:
        Dictionary mapping item_id -> score (0-1 normalized)
    """
    # Count wins and total comparisons per item
    wins = defaultdict(int)
    total = defaultdict(int)
    items = set()

    for winner, loser in comparisons:
        items.add(winner)
        items.add(loser)
        wins[winner] += 1
        total[winner] += 1
        total[loser] += 1

    # Initialize ratings uniformly
    ratings = {item: 1.0 for item in items}

    # Iteratively update ratings
    for iteration in range(max_iterations):
        new_ratings = {}
        max_change = 0.0

        for item in items:
            if total[item] == 0:
                new_ratings[item] = ratings[item]
                continue

            # Bradley-Terry update formula
            numerator = wins[item]
            denominator = 0.0

            for winner, loser in comparisons:
                if winner == item or loser == item:
                    opponent = loser if winner == item else winner
                    denominator += 1.0 / (ratings[item] + ratings[opponent])

            if denominator > 0:
                new_ratings[item] = numerator / denominator
            else:
                new_ratings[item] = ratings[item]

            max_change = max(max_change, abs(new_ratings[item] - ratings[item]))

        ratings = new_ratings

        if max_change < epsilon:
            print(f"  Converged after {iteration + 1} iterations")
            break

    # Normalize to [0, 1]
    min_rating = min(ratings.values())
    max_rating = max(ratings.values())

    if max_rating > min_rating:
        ratings = {k: (v - min_rating) / (max_rating - min_rating)
                   for k, v in ratings.items()}
    else:
        ratings = {k: 0.5 for k in ratings.keys()}

    return ratings


def elo_rating(comparisons, K=32, initial_rating=1500):
    """
    Alternative: ELO rating system (simpler, faster)

    Args:
        comparisons: List of (winner_id, loser_id) tuples
        K: ELO K-factor (higher = more volatile)
        initial_rating: Starting rating for all items

    Returns:
        Dictionary mapping item_id -> score (0-1 normalized)
    """
    ratings = defaultdict(lambda: initial_rating)

    for winner, loser in comparisons:
        r_winner = ratings[winner]
        r_loser = ratings[loser]

        # Expected scores
        exp_winner = 1.0 / (1.0 + 10**((r_loser - r_winner) / 400))
        exp_loser = 1.0 / (1.0 + 10**((r_winner - r_loser) / 400))

        # Update ratings (winner gets 1, loser gets 0)
        ratings[winner] = r_winner + K * (1.0 - exp_winner)
        ratings[loser] = r_loser + K * (0.0 - exp_loser)

    # Normalize to [0, 1]
    ratings_dict = dict(ratings)
    min_rating = min(ratings_dict.values())
    max_rating = max(ratings_dict.values())

    if max_rating > min_rating:
        ratings_dict = {k: (v - min_rating) / (max_rating - min_rating)
                        for k, v in ratings_dict.items()}
    else:
        ratings_dict = {k: 0.5 for k in ratings_dict.keys()}

    return ratings_dict


def compute_scores_for_attribute(df, attribute, method='elo', verbose=True):
    """
    Compute scores for a specific attribute.

    Args:
        df: Full dataframe with all comparisons
        attribute: Attribute name (e.g., 'wealthy')
        method: 'elo' or 'bradley_terry'
        verbose: Print progress

    Returns:
        Dictionary mapping image_id -> score
    """
    if verbose:
        print(f"\nProcessing attribute: {attribute}")

    # Filter to this attribute
    attr_df = df[df['category'] == attribute].copy()

    if verbose:
        print(f"  Total comparisons: {len(attr_df):,}")
        print(f"  Winner distribution:")
        print(f"    Left wins:  {(attr_df['winner'] == 'left').sum():,}")
        print(f"    Right wins: {(attr_df['winner'] == 'right').sum():,}")
        print(f"    Equal:      {(attr_df['winner'] == 'equal').sum():,}")

    # Create comparison pairs (winner, loser)
    comparisons = []

    for _, row in attr_df.iterrows():
        if row['winner'] == 'left':
            comparisons.append((row['left_id'], row['right_id']))
        elif row['winner'] == 'right':
            comparisons.append((row['right_id'], row['left_id']))
        # Skip 'equal' comparisons - they don't provide directional info

    if verbose:
        print(f"  Usable comparisons (excluding 'equal'): {len(comparisons):,}")
        unique_images = set([img for pair in comparisons for img in pair])
        print(f"  Unique images: {len(unique_images):,}")

    # Compute scores
    if method == 'elo':
        scores = elo_rating(comparisons)
    elif method == 'bradley_terry':
        scores = bradley_terry_simple(comparisons)
    else:
        raise ValueError(f"Unknown method: {method}")

    if verbose:
        score_values = list(scores.values())
        print(f"  Score statistics:")
        print(f"    Min:    {min(score_values):.4f}")
        print(f"    Max:    {max(score_values):.4f}")
        print(f"    Mean:   {np.mean(score_values):.4f}")
        print(f"    Median: {np.median(score_values):.4f}")
        print(f"    Std:    {np.std(score_values):.4f}")

    return scores


def check_image_availability(scores_dict, image_dir):
    """
    Check which scored images actually exist in the preprocessed directory.

    Returns:
        Dictionary with only available images
    """
    image_dir = Path(image_dir)
    available_scores = {}
    missing_count = 0

    for image_id, score in scores_dict.items():
        image_path = image_dir / f"{image_id}.jpg"
        if image_path.exists():
            available_scores[image_id] = score
        else:
            missing_count += 1

    print(f"\n  Image availability:")
    print(f"    Total scored images: {len(scores_dict):,}")
    print(f"    Available images:    {len(available_scores):,}")
    print(f"    Missing images:      {missing_count:,}")

    return available_scores


def main():
    parser = argparse.ArgumentParser(description="Compute attribute scores from pairwise comparisons")
    parser.add_argument("--data_csv", type=str,
                        default="data/PP2/final_data.csv",
                        help="Path to pairwise comparison CSV")
    parser.add_argument("--attribute", type=str, required=True,
                        help="Attribute to process (wealthy, depressing, safety, lively, boring, or 'all')")
    parser.add_argument("--output", type=str, required=True,
                        help="Output CSV path")
    parser.add_argument("--method", type=str, default="elo", choices=["elo", "bradley_terry"],
                        help="Scoring method")
    parser.add_argument("--image_dir", type=str, default="data/preprocessed_images",
                        help="Directory with preprocessed images")
    parser.add_argument("--split", action="store_true",
                        help="Also create train/val/test splits")
    parser.add_argument("--train_frac", type=float, default=0.8,
                        help="Fraction for training set")
    parser.add_argument("--val_frac", type=float, default=0.1,
                        help="Fraction for validation set")

    args = parser.parse_args()

    print("=" * 70)
    print("Attribute Score Computation")
    print("=" * 70)
    print(f"Data CSV: {args.data_csv}")
    print(f"Attribute: {args.attribute}")
    print(f"Method: {args.method}")
    print(f"Output: {args.output}")

    # Load data
    print("\nLoading pairwise comparison data...")
    df = pd.read_csv(args.data_csv)
    print(f"Loaded {len(df):,} comparisons")

    # Compute scores
    if args.attribute.lower() == 'all':
        attributes = df['category'].unique()
        print(f"\nProcessing all attributes: {list(attributes)}")

        all_scores = {}
        for attr in attributes:
            scores = compute_scores_for_attribute(df, attr, method=args.method)
            scores = check_image_availability(scores, args.image_dir)

            for img_id, score in scores.items():
                if img_id not in all_scores:
                    all_scores[img_id] = {}
                all_scores[img_id][attr] = score

        # Create dataframe
        rows = []
        for img_id, attr_scores in all_scores.items():
            row = {'image_id': img_id}
            row.update(attr_scores)
            rows.append(row)

        result_df = pd.DataFrame(rows)

    else:
        # Single attribute
        scores = compute_scores_for_attribute(df, args.attribute, method=args.method)
        scores = check_image_availability(scores, args.image_dir)

        result_df = pd.DataFrame([
            {'image_id': img_id, f'{args.attribute}_score': score}
            for img_id, score in scores.items()
        ])

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print(f"\n✅ Saved scores to: {output_path}")
    print(f"   Total images: {len(result_df):,}")

    # Create splits if requested
    if args.split:
        print("\nCreating train/val/test splits...")

        # Shuffle
        result_df = result_df.sample(frac=1, random_state=42).reset_index(drop=True)

        n = len(result_df)
        n_train = int(n * args.train_frac)
        n_val = int(n * args.val_frac)

        train_df = result_df[:n_train]
        val_df = result_df[n_train:n_train + n_val]
        test_df = result_df[n_train + n_val:]

        # Save splits
        base_path = output_path.parent / output_path.stem
        train_path = base_path.parent / f"{base_path.name}_train.csv"
        val_path = base_path.parent / f"{base_path.name}_val.csv"
        test_path = base_path.parent / f"{base_path.name}_test.csv"

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

        print(f"  Train: {len(train_df):,} images → {train_path}")
        print(f"  Val:   {len(val_df):,} images → {val_path}")
        print(f"  Test:  {len(test_df):,} images → {test_path}")

        # Save split info
        split_info = {
            'total_images': n,
            'train_images': len(train_df),
            'val_images': len(val_df),
            'test_images': len(test_df),
            'train_frac': args.train_frac,
            'val_frac': args.val_frac,
            'test_frac': 1.0 - args.train_frac - args.val_frac,
            'random_seed': 42
        }

        info_path = base_path.parent / f"{base_path.name}_split_info.json"
        with open(info_path, 'w') as f:
            json.dump(split_info, f, indent=2)
        print(f"  Split info: {info_path}")

    print("\n" + "=" * 70)
    print("✅ Done!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Examine the score distribution")
    print("2. Start training conditional GAN:")
    print(f"   python src/train_conditional_gan.py --attribute_csv {args.output}")


if __name__ == "__main__":
    main()

