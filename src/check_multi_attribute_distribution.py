#!/usr/bin/env python3
"""
Analyze multi-attribute score distribution

Usage:
    python check_multi_attribute_distribution.py --scores_csv data/all_attribute_scores.csv
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scores_csv', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='data/multi_attribute_analysis')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Multi-Attribute Score Distribution Analysis")
    print("=" * 70)
    print(f"Input: {args.scores_csv}")
    print()

    # Load data
    df = pd.read_csv(args.scores_csv)
    print(f"Total images: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    print()

    # Identify attribute columns (all except image_id)
    attribute_cols = [col for col in df.columns if col != 'image_id']
    print(f"Attributes found: {attribute_cols}")
    print()

    # Check for missing values
    print("Missing values per attribute:")
    missing = df[attribute_cols].isnull().sum()
    for attr, count in missing.items():
        pct = count / len(df) * 100
        print(f"  {attr}: {count:,} ({pct:.1f}%)")
    print()

    # Decide how to handle missing values
    total_missing = missing.sum()
    if total_missing > 0:
        print(f"⚠️  Total missing values: {total_missing:,}")
        print()
        print("Options:")
        print("  1. Drop images with ANY missing attribute")
        print("  2. Fill missing with median score (0.5)")
        print("  3. Fill missing with attribute mean")
        print()

        # Show impact of dropping
        df_complete = df.dropna()
        print(f"Option 1: Keep {len(df_complete):,} images ({len(df_complete)/len(df)*100:.1f}%)")
        print()

        # For now, let's analyze both scenarios
        print("Analysis will show COMPLETE CASES ONLY")
        df_analysis = df_complete
    else:
        print("✅ No missing values - all images have all attributes!")
        df_analysis = df

    print()
    print("=" * 70)
    print("SCORE STATISTICS")
    print("=" * 70)
    print()

    # Statistics per attribute
    for attr in attribute_cols:
        values = df_analysis[attr].dropna()
        print(f"{attr}:")
        print(f"  Count:  {len(values):,}")
        print(f"  Min:    {values.min():.6f}")
        print(f"  Max:    {values.max():.6f}")
        print(f"  Mean:   {values.mean():.6f}")
        print(f"  Median: {values.median():.6f}")
        print(f"  Std:    {values.std():.6f}")
        print()

    # Correlation analysis
    print("=" * 70)
    print("ATTRIBUTE CORRELATIONS")
    print("=" * 70)
    print()

    corr_matrix = df_analysis[attribute_cols].corr()
    print(corr_matrix.round(3))
    print()

    # Highlight strong correlations
    print("Strong correlations (|r| > 0.5):")
    for i, attr1 in enumerate(attribute_cols):
        for j, attr2 in enumerate(attribute_cols):
            if i < j:  # Upper triangle only
                r = corr_matrix.loc[attr1, attr2]
                if abs(r) > 0.5:
                    print(f"  {attr1} ↔ {attr2}: r={r:.3f}")
    print()

    # Visualization 1: Distribution histograms
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, attr in enumerate(attribute_cols):
        ax = axes[idx]
        values = df_analysis[attr].dropna()
        ax.hist(values, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel(f'{attr} score')
        ax.set_ylabel('Count')
        ax.set_title(f'{attr.capitalize()} Distribution')
        ax.axvline(values.mean(), color='red', linestyle='--', label=f'Mean: {values.mean():.3f}')
        ax.legend()

    # Remove extra subplot if odd number of attributes
    if len(attribute_cols) < len(axes):
        fig.delaxes(axes[-1])

    plt.tight_layout()
    dist_path = output_dir / 'attribute_distributions.png'
    plt.savefig(dist_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {dist_path}")
    plt.close()

    # Visualization 2: Correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, vmin=-1, vmax=1, square=True, ax=ax)
    ax.set_title('Attribute Correlation Matrix')
    plt.tight_layout()
    corr_path = output_dir / 'attribute_correlations.png'
    plt.savefig(corr_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {corr_path}")
    plt.close()

    # Visualization 3: Pairwise scatter plots (subset)
    if len(attribute_cols) >= 2:
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.flatten()

        # Plot most interesting pairs
        pairs = [
            ('wealthy', 'depressing'),
            ('wealthy', 'safety'),
            ('lively', 'boring'),
            ('safety', 'depressing')
        ]

        for idx, (attr1, attr2) in enumerate(pairs):
            if idx >= 4 or attr1 not in attribute_cols or attr2 not in attribute_cols:
                break

            ax = axes[idx]
            x = df_analysis[attr1]
            y = df_analysis[attr2]

            ax.scatter(x, y, alpha=0.3, s=1)
            ax.set_xlabel(f'{attr1} score')
            ax.set_ylabel(f'{attr2} score')
            ax.set_title(f'{attr1.capitalize()} vs {attr2.capitalize()}\n(r={corr_matrix.loc[attr1, attr2]:.3f})')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        scatter_path = output_dir / 'attribute_pairwise_scatter.png'
        plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {scatter_path}")
        plt.close()

    # Save summary statistics
    summary_path = output_dir / 'summary_statistics.txt'
    with open(summary_path, 'w') as f:
        f.write("MULTI-ATTRIBUTE SCORE SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total images: {len(df):,}\n")
        f.write(f"Complete cases: {len(df_analysis):,}\n")
        f.write(f"Attributes: {', '.join(attribute_cols)}\n\n")

        f.write("STATISTICS PER ATTRIBUTE\n")
        f.write("-" * 70 + "\n")
        for attr in attribute_cols:
            values = df_analysis[attr].dropna()
            f.write(f"\n{attr}:\n")
            f.write(f"  Count:  {len(values):,}\n")
            f.write(f"  Min:    {values.min():.6f}\n")
            f.write(f"  Max:    {values.max():.6f}\n")
            f.write(f"  Mean:   {values.mean():.6f}\n")
            f.write(f"  Median: {values.median():.6f}\n")
            f.write(f"  Std:    {values.std():.6f}\n")

        f.write("\n\nCORRELATION MATRIX\n")
        f.write("-" * 70 + "\n")
        f.write(corr_matrix.to_string())

    print(f"✅ Saved: {summary_path}")

    print()
    print("=" * 70)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 70)
    print()
    print(f"Results saved to: {output_dir}/")
    print()
    print("Key Findings:")
    print(f"  • {len(df_analysis):,} images with complete attribute scores")
    print(f"  • {len(attribute_cols)} attributes: {', '.join(attribute_cols)}")

    # Check if any extreme correlations
    max_corr = 0
    for i, attr1 in enumerate(attribute_cols):
        for j, attr2 in enumerate(attribute_cols):
            if i < j:
                r = abs(corr_matrix.loc[attr1, attr2])
                if r > max_corr:
                    max_corr = r

    if max_corr > 0.7:
        print(f"  • ⚠️  Strong correlation detected (max r={max_corr:.3f})")
        print("    This is OK - real-world attributes are correlated")
    else:
        print(f"  • ✅ Attributes are reasonably independent (max r={max_corr:.3f})")

    print()
    print("Next step:")
    print("  Create multi-attribute training script:")
    print("  python src/train_multi_attribute_gan.py --help")


if __name__ == '__main__':
    main()

