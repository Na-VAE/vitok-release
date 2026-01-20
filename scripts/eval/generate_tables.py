#!/usr/bin/env python
"""Generate markdown tables from evaluation JSON results.

Usage:
    python scripts/eval/generate_tables.py

Outputs:
    results/tables/coco_comparison.md  - ViTok vs baselines
    results/tables/swa_ablation.md     - SWA window size ablation
"""
import json
from pathlib import Path


def load_results(results_dir: Path, pattern: str = "*.json") -> list[dict]:
    """Load all JSON results from directory."""
    results = []
    for path in sorted(results_dir.glob(pattern)):
        with open(path) as f:
            data = json.load(f)
            data["_file"] = path.stem
            results.append(data)
    return results


def generate_coco_table(results_256: list[dict], results_512: list[dict]) -> str:
    """Generate COCO comparison markdown table."""
    # Index by model name
    by_model_256 = {r.get("model") or r.get("baseline", f"baseline-{r.get('_file', '').replace('baseline-', '')}"): r for r in results_256}
    by_model_512 = {r.get("model") or r.get("baseline", f"baseline-{r.get('_file', '').replace('baseline-', '')}"): r for r in results_512}

    all_models = sorted(set(by_model_256.keys()) | set(by_model_512.keys()))

    lines = [
        "# COCO Reconstruction Quality",
        "",
        "| Model | 256px FID | 256px PSNR | 512px FID | 512px PSNR | Stride |",
        "|-------|-----------|------------|-----------|------------|--------|",
    ]

    # Organize models by type
    vitok_350m = [m for m in all_models if m.startswith("350M")]
    vitok_5b = [m for m in all_models if m.startswith("5B")]
    baselines = [m for m in all_models if "baseline" in m or m in ("flux", "sd", "qwen")]

    for model in vitok_350m + vitok_5b + baselines:
        r256 = by_model_256.get(model, {})
        r512 = by_model_512.get(model, {})

        fid_256 = f"{r256.get('fid', 'N/A'):.2f}" if isinstance(r256.get('fid'), (int, float)) else "N/A"
        psnr_256 = f"{r256.get('psnr', 'N/A'):.1f}" if isinstance(r256.get('psnr'), (int, float)) else "N/A"
        fid_512 = f"{r512.get('fid', 'N/A'):.2f}" if isinstance(r512.get('fid'), (int, float)) else "N/A"
        psnr_512 = f"{r512.get('psnr', 'N/A'):.1f}" if isinstance(r512.get('psnr'), (int, float)) else "N/A"

        # Determine stride
        if "baseline" in model or model in ("flux", "sd", "qwen"):
            stride = "8x"
        elif "f16" in model:
            stride = "16x"
        elif "f32" in model:
            stride = "32x"
        else:
            stride = "?"

        lines.append(f"| {model} | {fid_256} | {psnr_256} | {fid_512} | {psnr_512} | {stride} |")

    return "\n".join(lines)


def generate_swa_table(results: list[dict]) -> str:
    """Generate SWA ablation markdown table."""
    lines = [
        "# SWA (Sliding Window Attention) Ablation",
        "",
        "Comparison of different window sizes for high-resolution inference.",
        "",
        "| Model | Resolution | Window | PSNR | SSIM | Memory (GB) |",
        "|-------|------------|--------|------|------|-------------|",
    ]

    # Sort by model, resolution, window
    def sort_key(r):
        fname = r["_file"]
        parts = fname.split("-")
        model = parts[0]
        res = int(parts[1]) if len(parts) > 1 else 0
        window = int(parts[2].replace("w", "")) if len(parts) > 2 else 0
        return (model, res, window)

    results = sorted(results, key=sort_key)

    for r in results:
        fname = r["_file"]
        parts = fname.split("-")
        model = parts[0] if parts else "?"
        res = parts[1] if len(parts) > 1 else "?"
        window = parts[2] if len(parts) > 2 else "full"

        psnr = f"{r.get('psnr', 'N/A'):.1f}" if isinstance(r.get('psnr'), (int, float)) else "N/A"
        ssim = f"{r.get('ssim', 'N/A'):.3f}" if isinstance(r.get('ssim'), (int, float)) else "N/A"
        mem = f"{r.get('max_memory_allocated_gb', 'N/A'):.1f}" if isinstance(r.get('max_memory_allocated_gb'), (int, float)) else "N/A"

        lines.append(f"| {model} | {res} | {window} | {psnr} | {ssim} | {mem} |")

    return "\n".join(lines)


def main():
    results_dir = Path("results")
    output_dir = Path("results/tables")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating markdown tables...")

    # COCO comparison table
    coco_256 = load_results(results_dir / "coco-256")
    coco_512 = load_results(results_dir / "coco-512")

    if coco_256 or coco_512:
        table = generate_coco_table(coco_256, coco_512)
        (output_dir / "coco_comparison.md").write_text(table)
        print(f"  Wrote {output_dir / 'coco_comparison.md'}")
    else:
        print("  No COCO results found")

    # SWA ablation table
    swa_results = load_results(results_dir / "swa")
    if swa_results:
        table = generate_swa_table(swa_results)
        (output_dir / "swa_ablation.md").write_text(table)
        print(f"  Wrote {output_dir / 'swa_ablation.md'}")
    else:
        print("  No SWA results found")

    print("Done!")


if __name__ == "__main__":
    main()
