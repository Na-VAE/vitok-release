#!/usr/bin/env python
"""Generate plots from evaluation results for blog.

Reads JSON results from eval runs and creates publication-ready plots.

Usage:
    python scripts/eval/plot_results.py --results-dir results/
    python scripts/eval/plot_results.py --results-dir results/ --output-dir results/plots
"""
import argparse
import json
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

# Style settings for publication
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {
    # ViTok models
    "350M-f16x16": "#1f77b4",
    "350M-f16x32": "#2ca02c",
    "350M-f16x64": "#ff7f0e",
    "5B-f32x64": "#9467bd",
    "5B-f32x128": "#8c564b",
    "5B-f32x256": "#e377c2",
    # Baselines
    "flux": "#7f7f7f",
    "sd": "#bcbd22",
    "qwen": "#17becf",
}


def load_results(results_dir: Path) -> list[dict]:
    """Load all JSON result files from directory."""
    results = []
    for json_file in results_dir.glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)
            if isinstance(data, list):
                results.extend(data)
            else:
                results.append(data)
    return results


def plot_metric_comparison(
    results: list[dict],
    metric: str,
    resolution: int,
    output_path: Path,
    title: str = None,
    higher_better: bool = True,
):
    """Create bar chart comparing models on a metric at a resolution."""
    # Filter results
    filtered = [r for r in results if r.get("max_size") == resolution and r.get("status") == "ok"]
    if not filtered:
        print(f"No results for resolution {resolution}")
        return

    # Group by model
    models = []
    values = []
    colors = []

    for r in sorted(filtered, key=lambda x: (x.get("model_type", ""), x.get("model", ""))):
        model = r.get("model", "unknown")
        value = r.get(metric)
        if value is not None:
            models.append(model)
            values.append(value)
            colors.append(COLORS.get(model, "#333333"))

    if not models:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(models)), values, color=colors)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylabel(metric.upper())

    if title:
        ax.set_title(title)
    else:
        direction = "Higher" if higher_better else "Lower"
        ax.set_title(f"{metric.upper()} at {resolution}px ({direction} is better)")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_swa_ablation(
    results: list[dict],
    metric: str,
    output_path: Path,
    title: str = None,
):
    """Create line chart showing metric vs SWA window size."""
    # Group by (model, resolution)
    from collections import defaultdict
    data = defaultdict(list)

    for r in results:
        if r.get("status") != "ok":
            continue
        model = r.get("model", "unknown")
        resolution = r.get("max_size", 0)
        swa = r.get("swa_window")
        value = r.get(metric)

        if value is not None:
            # Convert None to "full" for display
            swa_label = "full" if swa is None else swa
            data[(model, resolution)].append((swa_label, value))

    if not data:
        print("No SWA ablation data found")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for (model, resolution), points in sorted(data.items()):
        # Sort by window size (numeric, with "full" last)
        def sort_key(p):
            return float("inf") if p[0] == "full" else p[0]
        points = sorted(points, key=sort_key)

        x_labels = [str(p[0]) for p in points]
        y_values = [p[1] for p in points]

        label = f"{model} @ {resolution}px"
        ax.plot(range(len(points)), y_values, marker="o", label=label)

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("SWA Window Size")
    ax.set_ylabel(metric.upper())
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"{metric.upper()} vs SWA Window Size")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_memory_vs_quality(
    results: list[dict],
    output_path: Path,
):
    """Create scatter plot of memory usage vs quality (PSNR)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for r in results:
        if r.get("status") != "ok":
            continue

        model = r.get("model", "unknown")
        psnr = r.get("psnr")
        memory = r.get("memory_gb")

        if psnr is not None and memory is not None:
            color = COLORS.get(model, "#333333")
            ax.scatter(memory, psnr, c=color, s=100, label=model, alpha=0.7)

    # Remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc="upper left")

    ax.set_xlabel("Memory (GB)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Memory vs Quality Trade-off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_throughput_comparison(
    results: list[dict],
    output_path: Path,
):
    """Create bar chart of throughput (images/sec)."""
    # Group by model, average across runs
    from collections import defaultdict
    throughputs = defaultdict(list)

    for r in results:
        if r.get("status") != "ok":
            continue
        model = r.get("model", "unknown")
        tp = r.get("throughput")
        if tp is not None:
            throughputs[model].append(tp)

    if not throughputs:
        return

    models = sorted(throughputs.keys())
    avg_throughputs = [sum(throughputs[m]) / len(throughputs[m]) for m in models]
    colors = [COLORS.get(m, "#333333") for m in models]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(models)), avg_throughputs, color=colors)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylabel("Throughput (img/sec)")
    ax.set_title("Average Throughput")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def generate_summary_table(results: list[dict], output_path: Path):
    """Generate markdown summary table."""
    lines = ["# Evaluation Results Summary\n"]
    lines.append("| Model | Resolution | FID | PSNR | SSIM | Memory (GB) | Status |")
    lines.append("|-------|------------|-----|------|------|-------------|--------|")

    for r in sorted(results, key=lambda x: (x.get("model", ""), x.get("max_size", 0))):
        model = r.get("model", "?")
        res = r.get("max_size", "?")
        fid = f"{r.get('fid', 0):.2f}" if r.get("fid") else "-"
        psnr = f"{r.get('psnr', 0):.2f}" if r.get("psnr") else "-"
        ssim = f"{r.get('ssim', 0):.4f}" if r.get("ssim") else "-"
        mem = f"{r.get('memory_gb', 0):.1f}" if r.get("memory_gb") else "-"
        status = r.get("status", "?")

        lines.append(f"| {model} | {res} | {fid} | {psnr} | {ssim} | {mem} | {status} |")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate plots from eval results")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/plots"))
    args = parser.parse_args()

    results = load_results(args.results_dir)
    if not results:
        print(f"No results found in {args.results_dir}")
        return

    print(f"Loaded {len(results)} results")

    # Generate standard comparison plots
    for resolution in [256, 512, 1024, 2048]:
        plot_metric_comparison(
            results, "fid", resolution,
            args.output_dir / f"fid_comparison_{resolution}.png",
            higher_better=False,
        )
        plot_metric_comparison(
            results, "psnr", resolution,
            args.output_dir / f"psnr_comparison_{resolution}.png",
            higher_better=True,
        )

    # Generate SWA ablation plots
    swa_results = [r for r in results if r.get("swa_window") is not None or "swa" in str(r.get("dataset", "")).lower()]
    if swa_results:
        plot_swa_ablation(swa_results, "psnr", args.output_dir / "swa_ablation_psnr.png")
        plot_swa_ablation(swa_results, "ssim", args.output_dir / "swa_ablation_ssim.png")
        plot_swa_ablation(swa_results, "memory_gb", args.output_dir / "swa_ablation_memory.png")

    # Generate summary plots
    plot_memory_vs_quality(results, args.output_dir / "memory_vs_quality.png")
    plot_throughput_comparison(results, args.output_dir / "throughput_comparison.png")

    # Generate summary table
    generate_summary_table(results, args.output_dir / "summary.md")

    print(f"\nAll plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
