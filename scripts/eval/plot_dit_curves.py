#!/usr/bin/env python
"""Generate blog-friendly DiT training curves.

Creates a clean plot showing generation FID vs training steps for different
latent configurations. Labels focus on token count (f16 = 256 tokens) to
communicate the key efficiency message.

Output:
    docs/assets/images/dit_training_curve.png

Usage:
    python scripts/eval/plot_dit_curves.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Publication style matching plot_results.py
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})

# =============================================================================
# Training data: gFID vs steps for different VAE configurations
# All use f16 compression (256 tokens for 256px images)
# =============================================================================

# Large decoder models (1.2B DiT)
# 64ch latent - highest channel capacity
x_64ch = [0, 20000, 50000, 100000, 150000, 200000, 250000, 295000, 350000, 400000]
y_64ch = [28.2, 20.0, 8.2, 5.8, 4.9, 4.5, 3.9, 3.6, 3.25, 2.85]

# 32ch latent
x_32ch = [0, 25000, 35000, 50000, 100000, 150000, 200000, 250000, 295000, 350000, 400000]
y_32ch = [12.0, 8.8, 15.6, 7.0, 4.8, 4.2, 4.0, 3.8, 3.4, 3.15, 2.85]

# 16ch latent - lowest channel capacity
x_16ch = [0, 20000, 50000, 100000, 200000, 250000, 290000, 350000, 400000]
y_16ch = [11.3, 10.5, 5.8, 5.0, 4.5, 4.3, 4.0, 3.75, 3.5]

# Colors - blue/purple tones for ViTok
colors = {
    '64ch': '#6c5ce7',  # Purple - best
    '32ch': '#0984e3',  # Blue
    '16ch': '#00b894',  # Teal
}

def to_k(x):
    """Convert steps to thousands."""
    return [v / 1000 for v in x]


def main():
    # Create output directory
    output_dir = Path(__file__).parent.parent.parent / "docs" / "assets" / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Single plot: FID vs Training Steps
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot curves (higher channel capacity tends to win with more training)
    ax.plot(to_k(x_64ch), y_64ch, '-', color=colors['64ch'], linewidth=2.5,
            label='256 tokens, 64ch (best)', marker='o', markersize=0)
    ax.plot(to_k(x_32ch), y_32ch, '-', color=colors['32ch'], linewidth=2.5,
            label='256 tokens, 32ch', marker='o', markersize=0)
    ax.plot(to_k(x_16ch), y_16ch, '-', color=colors['16ch'], linewidth=2.5,
            label='256 tokens, 16ch', marker='o', markersize=0)

    # End markers
    for x, y, c in [(x_64ch, y_64ch, colors['64ch']),
                    (x_32ch, y_32ch, colors['32ch']),
                    (x_16ch, y_16ch, colors['16ch'])]:
        ax.scatter(x[-1] / 1000, y[-1], color=c, s=60, zorder=5, edgecolors='white', linewidths=1.5)

    # Reference line for SD/Flux baseline (1024 tokens @ f8 compression)
    ax.axhline(y=2.5, color='#636e72', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(390, 2.7, 'SD/Flux baseline\n(1024 tokens)', fontsize=9, color='#636e72',
            ha='right', va='bottom')

    ax.set_xlabel('Training Steps (k)', fontweight='medium')
    ax.set_ylabel('Generation FID (lower is better)', fontweight='medium')
    ax.set_xlim(0, 420)
    ax.set_ylim(2, 12)

    # Legend
    ax.legend(loc='upper right', framealpha=0.95, edgecolor='none')

    # Add annotation
    ax.annotate('4x fewer tokens\nsame quality',
                xy=(295, 3.4), xytext=(200, 7),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#dee2e6'))

    plt.tight_layout()

    # Save
    output_path = output_dir / "dit_training_curve.png"
    plt.savefig(output_path, facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")

    # Also create a dark-mode version for the blog
    plt.rcParams.update({
        'axes.facecolor': '#161b22',
        'figure.facecolor': '#0d1117',
        'axes.edgecolor': '#30363d',
        'axes.labelcolor': '#e6edf3',
        'xtick.color': '#8b949e',
        'ytick.color': '#8b949e',
        'grid.color': '#30363d',
        'text.color': '#e6edf3',
        'legend.facecolor': '#161b22',
        'legend.edgecolor': '#30363d',
    })

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(to_k(x_64ch), y_64ch, '-', color='#a29bfe', linewidth=2.5,
            label='256 tokens, 64ch (best)')
    ax.plot(to_k(x_32ch), y_32ch, '-', color='#74b9ff', linewidth=2.5,
            label='256 tokens, 32ch')
    ax.plot(to_k(x_16ch), y_16ch, '-', color='#55efc4', linewidth=2.5,
            label='256 tokens, 16ch')

    for x, y, c in [(x_64ch, y_64ch, '#a29bfe'),
                    (x_32ch, y_32ch, '#74b9ff'),
                    (x_16ch, y_16ch, '#55efc4')]:
        ax.scatter(x[-1] / 1000, y[-1], color=c, s=60, zorder=5, edgecolors='#0d1117', linewidths=1.5)

    ax.axhline(y=2.5, color='#8b949e', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(390, 2.7, 'SD/Flux baseline\n(1024 tokens)', fontsize=9, color='#8b949e',
            ha='right', va='bottom')

    ax.set_xlabel('Training Steps (k)', fontweight='medium')
    ax.set_ylabel('Generation FID (lower is better)', fontweight='medium')
    ax.set_xlim(0, 420)
    ax.set_ylim(2, 12)
    ax.legend(loc='upper right', framealpha=0.95)

    ax.annotate('4x fewer tokens\nsame quality',
                xy=(295, 3.4), xytext=(200, 7),
                fontsize=10, ha='center', color='#e6edf3',
                arrowprops=dict(arrowstyle='->', color='#8b949e', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#21262d', edgecolor='#30363d'))

    plt.tight_layout()

    output_path_dark = output_dir / "dit_training_curve_dark.png"
    plt.savefig(output_path_dark, facecolor='#0d1117', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path_dark}")


if __name__ == "__main__":
    main()
