"""Generate evaluation report with figures from batch eval results.

This script reads evaluation results from Modal volume and creates:
1. Random 5 samples grid (Original | Reconstruction | Difference)
2. SSIM percentile showcase (P1, P25, P50, P75, P99)
3. Metrics comparison bar chart
4. Markdown report with all findings

Usage:
    # Download results from Modal volume first
    modal volume get vitok-eval-results / ./results/

    # Generate report
    python scripts/create_eval_report.py --results-dir ./results

    # Generate report for specific models
    python scripts/create_eval_report.py --results-dir ./results --models T-32x64,T-32x128
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Optional
import numpy as np

# Optional imports for figure generation
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed, figures will be skipped")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: PIL not installed, sample grids will be skipped")


def load_results(results_dir: Path) -> list[dict]:
    """Load all evaluation results from directory."""
    results = []
    for results_file in results_dir.glob("*/results.json"):
        if "batch_" not in str(results_file.parent):
            with open(results_file) as f:
                data = json.load(f)
                data["_dir"] = results_file.parent
                results.append(data)
    return results


def create_random_samples_figure(
    samples_dir: Path,
    output_path: Path,
    n_samples: int = 5,
    seed: int = 42,
) -> Optional[Path]:
    """Create a figure with random samples showing original, reconstruction, difference."""
    if not HAS_PIL or not HAS_MATPLOTLIB:
        return None

    # Find sample images (original and reconstructed pairs)
    originals = sorted(samples_dir.glob("*_original.png")) + sorted(samples_dir.glob("*_original.jpg"))
    recons = sorted(samples_dir.glob("*_recon.png")) + sorted(samples_dir.glob("*_recon.jpg"))

    if not originals or not recons:
        # Try alternate naming
        originals = sorted(samples_dir.glob("original_*.png"))
        recons = sorted(samples_dir.glob("recon_*.png"))

    if not originals or len(originals) < n_samples:
        print(f"  Not enough samples in {samples_dir}")
        return None

    # Random selection
    np.random.seed(seed)
    indices = np.random.choice(len(originals), min(n_samples, len(originals)), replace=False)

    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i, idx in enumerate(indices):
        orig_img = Image.open(originals[idx]).convert("RGB")
        recon_img = Image.open(recons[idx]).convert("RGB")

        # Compute difference
        orig_arr = np.array(orig_img).astype(float)
        recon_arr = np.array(recon_img).astype(float)
        diff_arr = np.abs(orig_arr - recon_arr)
        # Amplify difference for visibility
        diff_arr = np.clip(diff_arr * 3, 0, 255).astype(np.uint8)

        axes[i, 0].imshow(orig_img)
        axes[i, 0].set_title("Original" if i == 0 else "")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(recon_img)
        axes[i, 1].set_title("Reconstruction" if i == 0 else "")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(diff_arr)
        axes[i, 2].set_title("Difference (3x amplified)" if i == 0 else "")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def create_ssim_percentile_figure(
    samples_dir: Path,
    ssim_scores: list[tuple[str, float]],  # (filename, ssim)
    output_path: Path,
) -> Optional[Path]:
    """Create figure showing samples at different SSIM percentiles."""
    if not HAS_PIL or not HAS_MATPLOTLIB:
        return None

    if not ssim_scores:
        return None

    # Sort by SSIM
    sorted_scores = sorted(ssim_scores, key=lambda x: x[1])
    n = len(sorted_scores)

    # Select percentiles: P1, P25, P50, P75, P99
    percentiles = [
        ("P1 (worst)", int(0.01 * n)),
        ("P25", int(0.25 * n)),
        ("P50 (median)", int(0.50 * n)),
        ("P75", int(0.75 * n)),
        ("P99 (best)", min(int(0.99 * n), n - 1)),
    ]

    fig, axes = plt.subplots(5, 3, figsize=(12, 20))

    for i, (label, idx) in enumerate(percentiles):
        filename, ssim = sorted_scores[idx]

        # Load images
        orig_path = samples_dir / f"{filename}_original.png"
        recon_path = samples_dir / f"{filename}_recon.png"

        if not orig_path.exists():
            orig_path = samples_dir / f"original_{filename}.png"
            recon_path = samples_dir / f"recon_{filename}.png"

        if not orig_path.exists():
            continue

        orig_img = Image.open(orig_path).convert("RGB")
        recon_img = Image.open(recon_path).convert("RGB")

        orig_arr = np.array(orig_img).astype(float)
        recon_arr = np.array(recon_img).astype(float)
        diff_arr = np.clip(np.abs(orig_arr - recon_arr) * 3, 0, 255).astype(np.uint8)

        axes[i, 0].imshow(orig_img)
        axes[i, 0].set_ylabel(f"{label}\nSSIM: {ssim:.4f}", fontsize=10)
        axes[i, 0].set_title("Original" if i == 0 else "")
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

        axes[i, 1].imshow(recon_img)
        axes[i, 1].set_title("Reconstruction" if i == 0 else "")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(diff_arr)
        axes[i, 2].set_title("Difference" if i == 0 else "")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def create_metrics_comparison_figure(
    results: list[dict],
    output_path: Path,
) -> Optional[Path]:
    """Create bar chart comparing metrics across models."""
    if not HAS_MATPLOTLIB:
        return None

    if not results:
        return None

    # Group by model and resolution
    data = {}
    for r in results:
        config = r.get("config_id", "unknown")
        model = r.get("model", config.split("_")[0] if "_" in config else config)
        res = r.get("max_size", 256)

        key = (model, res)
        data[key] = {
            "ssim": r.get("ssim"),
            "psnr": r.get("psnr"),
            "fid": r.get("fid"),
            "fdd": r.get("fdd"),
        }

    if not data:
        return None

    # Sort keys for consistent ordering
    keys = sorted(data.keys())
    labels = [f"{m}\n{r}p" for m, r in keys]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = [
        ("ssim", "SSIM (higher is better)", axes[0, 0], "tab:blue"),
        ("psnr", "PSNR (higher is better)", axes[0, 1], "tab:green"),
        ("fid", "FID (lower is better)", axes[1, 0], "tab:orange"),
        ("fdd", "FDD (lower is better)", axes[1, 1], "tab:red"),
    ]

    x = np.arange(len(keys))
    width = 0.6

    for metric, title, ax, color in metrics:
        values = [data[k].get(metric) or 0 for k in keys]
        bars = ax.bar(x, values, width, color=color, alpha=0.7)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel(metric.upper())

        # Add value labels on bars
        for bar, val in zip(bars, values):
            if val:
                height = bar.get_height()
                ax.annotate(
                    f"{val:.3f}" if metric == "ssim" else f"{val:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def generate_markdown_report(
    results: list[dict],
    figures_dir: Path,
    output_path: Path,
) -> Path:
    """Generate markdown report with results and figure references."""
    lines = [
        "# ViTok Evaluation Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        f"Total evaluations: {len(results)}",
        "",
    ]

    # Results table
    lines.extend([
        "## Metrics",
        "",
        "| Config | SSIM | PSNR | FID | FDD | Samples |",
        "|--------|------|------|-----|-----|---------|",
    ])

    for r in sorted(results, key=lambda x: x.get("config_id", "")):
        config = r.get("config_id", "unknown")
        ssim = f"{r.get('ssim', 0):.4f}" if r.get("ssim") else "N/A"
        psnr = f"{r.get('psnr', 0):.2f}" if r.get("psnr") else "N/A"
        fid = f"{r.get('fid', 0):.2f}" if r.get("fid") else "N/A"
        fdd = f"{r.get('fdd', 0):.2f}" if r.get("fdd") else "N/A"
        samples = r.get("samples", "N/A")
        lines.append(f"| {config} | {ssim} | {psnr} | {fid} | {fdd} | {samples} |")

    lines.append("")

    # Key findings
    if results:
        best_ssim = max(results, key=lambda x: x.get("ssim", 0))
        worst_ssim = min(results, key=lambda x: x.get("ssim", float("inf")) if x.get("ssim") else float("inf"))
        best_fid = min(results, key=lambda x: x.get("fid", float("inf")) if x.get("fid") else float("inf"))

        lines.extend([
            "## Key Findings",
            "",
            f"- **Best SSIM**: {best_ssim.get('config_id')} ({best_ssim.get('ssim', 0):.4f})",
            f"- **Best FID**: {best_fid.get('config_id')} ({best_fid.get('fid', 0):.2f})",
            f"- **Worst SSIM**: {worst_ssim.get('config_id')} ({worst_ssim.get('ssim', 0):.4f})",
            "",
        ])

    # Figure references
    figures = list(figures_dir.glob("*.png"))
    if figures:
        lines.extend([
            "## Figures",
            "",
        ])
        for fig in sorted(figures):
            lines.append(f"![{fig.stem}]({fig.name})")
            lines.append("")

    # Float8 comparison if available
    bf16_results = [r for r in results if not r.get("float8_mode")]
    f8_results = [r for r in results if r.get("float8_mode")]

    if bf16_results and f8_results:
        lines.extend([
            "## Float8 vs BF16 Comparison",
            "",
            "| Config | BF16 SSIM | Float8 SSIM | Diff |",
            "|--------|-----------|-------------|------|",
        ])

        for bf16_r in bf16_results:
            base_config = bf16_r.get("config_id", "").replace("_bf16", "")
            f8_match = next(
                (r for r in f8_results if base_config in r.get("config_id", "")),
                None,
            )
            if f8_match:
                bf16_ssim = bf16_r.get("ssim", 0)
                f8_ssim = f8_match.get("ssim", 0)
                diff = f8_ssim - bf16_ssim
                lines.append(
                    f"| {base_config} | {bf16_ssim:.4f} | {f8_ssim:.4f} | {diff:+.4f} |"
                )

        lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation report with figures")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing evaluation results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for report (default: results/report)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated list of models to include",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir = args.output_dir or results_dir / "report"
    output_dir.mkdir(parents=True, exist_ok=True)

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    print(f"Loading results from: {results_dir}")
    results = load_results(results_dir)

    if not results:
        print("No results found!")
        return

    # Filter by models if specified
    if args.models:
        model_list = [m.strip() for m in args.models.split(",")]
        results = [r for r in results if any(m in r.get("config_id", "") for m in model_list)]
        print(f"Filtered to {len(results)} results for models: {model_list}")

    print(f"Found {len(results)} evaluation results")

    # Generate metrics comparison figure
    print("Generating metrics comparison figure...")
    metrics_fig = create_metrics_comparison_figure(
        results,
        figures_dir / "metrics_comparison.png",
    )
    if metrics_fig:
        print(f"  Saved: {metrics_fig}")

    # Generate sample figures for each result
    for r in results:
        config = r.get("config_id", "unknown")
        samples_dir = r.get("_dir", results_dir / config)

        if samples_dir and (samples_dir / "samples").exists():
            samples_dir = samples_dir / "samples"
        elif samples_dir and samples_dir.exists():
            pass
        else:
            continue

        print(f"Generating sample figures for {config}...")

        # Random samples
        random_fig = create_random_samples_figure(
            samples_dir,
            figures_dir / f"random_samples_{config}.png",
        )
        if random_fig:
            print(f"  Saved: {random_fig}")

    # Generate markdown report
    print("Generating markdown report...")
    report_path = generate_markdown_report(results, figures_dir, output_dir / "EVAL_REPORT.md")
    print(f"  Saved: {report_path}")

    print(f"\nReport complete! See: {output_dir}")


if __name__ == "__main__":
    main()
