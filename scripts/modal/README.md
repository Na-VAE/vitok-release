# Modal Setup for ViTok

This directory contains scripts for running ViTok on Modal (cloud GPU compute).

## Quick Start

```bash
# 1. Install modal
pip install modal

# 2. Authenticate (one-time)
modal token new

# 3. Set up secrets
modal secret create huggingface-secret HF_TOKEN=<your-hf-token>
modal secret create wandb-secret WANDB_API_KEY=<your-wandb-key>  # optional, for training

# 4. Pre-build environment and set up volumes (optional, speeds up first run)
modal run scripts/modal/setup_env.py
modal run scripts/modal/setup_data.py

# 5. Run evaluation
modal run scripts/modal/eval_vae.py --model L-64
```

## Volumes

Modal volumes persist data across runs. ViTok uses:

| Volume | Purpose | Setup |
|--------|---------|-------|
| `vitok-weights` | Model weights, HF cache | Auto-created |
| `vitok-data` | Evaluation datasets (COCO, DIV8K) | `setup_data.py` |
| `vitok-checkpoints` | Training checkpoints | Auto-created |

### Volume Commands

```bash
# List all volumes
modal volume list

# List contents
modal volume ls vitok-weights
modal volume ls vitok-data

# Download file locally
modal volume get vitok-weights /huggingface/hub/models--philippehansen--ViTok-L-16x64 ./local/

# Delete volume contents
modal volume rm vitok-data /coco --recursive

# Delete entire volume
modal volume delete vitok-data
```

## Scripts

### Setup Scripts

| Script | Purpose |
|--------|---------|
| `setup_env.py` | Pre-build Modal image with dependencies |
| `setup_data.py` | Download and cache evaluation datasets |
| `setup_volume.py` | Create/manage weights volume |

### Evaluation

| Script | Purpose |
|--------|---------|
| `eval_vae.py` | Run VAE evaluation (FID, FDD, SSIM, PSNR) |

### Inference

| Script | Purpose |
|--------|---------|
| `inference.py` | Image/video encoding and reconstruction |

### Utilities

| Script | Purpose |
|--------|---------|
| `split_weights.py` | Split model into encoder/decoder files |
| `upload_to_hf.py` | Upload weights to HuggingFace |
| `clean_weights.py` | Clean/convert weight files |
| `cleanup_volume.py` | Clean up old checkpoints |

## Dataset Setup

Datasets are cached to the `vitok-data` volume for reuse:

```bash
# Download all evaluation datasets (~15 min)
modal run scripts/modal/setup_data.py

# Download specific dataset
modal run scripts/modal/setup_data.py --dataset coco
modal run scripts/modal/setup_data.py --dataset div8k

# Check what's cached
modal run scripts/modal/setup_data.py --check
```

### Supported Datasets

| Dataset | Size | Resolution | Use Case |
|---------|------|------------|----------|
| COCO val2017 | 5K images, ~1GB | Mixed | General eval |
| DIV8K | 1.5K images, ~6GB | 8K | High-res eval (1024p+) |
| ImageNet-val | 50K images | 224-512px | Streamed via WebDataset |

**Note:** ImageNet is streamed via WebDataset from HuggingFace, not downloaded to the volume.

## Running Evaluations

### Basic Usage

```bash
# Evaluate L-64 on COCO (default, 5000 samples)
modal run scripts/modal/eval_vae.py --model L-64

# Quick test (100 samples)
modal run scripts/modal/eval_vae.py --model L-64 --num-samples 100

# Save results to JSON
modal run scripts/modal/eval_vae.py --model L-64 --output-json results.json
```

### Full Evaluation Command

```bash
modal run scripts/modal/eval_vae.py \
  --model L-64 \
  --dataset coco-val \
  --crop-style adm_square \
  --max-size 256 \
  --num-samples 5000 \
  --batch-size 16 \
  --save-visuals 8
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `L-64` | Model alias (L-64, L-32, L-16) |
| `--dataset` | `coco-val` | Dataset preset (coco-val, div8k, imagenet-val) |
| `--crop-style` | `native` | Crop style (adm_square, native) |
| `--max-size` | `512` | Maximum image size in pixels |
| `--num-samples` | `5000` | Number of samples to evaluate |
| `--batch-size` | `16` | Batch size for evaluation |
| `--swa-window` | `None` | Sliding window attention radius |
| `--save-visuals` | `8` | Number of sample images to save (0=none) |
| `--output-json` | `None` | Save results to JSON file |
| `--output-dir` | auto | Directory for visuals (default: `results/<model>_<dataset>_<size>p_<crop>`) |

### Available Models

| Alias | Variant | HF Repo |
|-------|---------|---------|
| `L-64` | Ld4-Ld24/1x16x64 | philippehansen/ViTok-L-16x64 |
| `L-32` | Ld4-Ld24/1x16x32 | philippehansen/ViTok-L-16x32 |
| `L-16` | Ld4-Ld24/1x16x16 | philippehansen/ViTok-L-16x16 |

```bash
# List available models
modal run scripts/modal/eval_vae.py --list-models

# List available datasets
modal run scripts/modal/eval_vae.py --list-datasets
```

### Crop Styles

- `adm_square`: Center crop to square (ADM-style), standard for FID comparison with other VAEs
- `native`: Preserve aspect ratio, resize longest side to max-size

### Metrics

The evaluation computes:

| Metric | Description | Better |
|--------|-------------|--------|
| FID | Fréchet Inception Distance | Lower |
| FDD | Fréchet DINO Distance | Lower |
| SSIM | Structural Similarity Index | Higher |
| PSNR | Peak Signal-to-Noise Ratio | Higher |

### Output

Results are printed to console and optionally saved:

```
==================================================
Evaluation Results: L-64
==================================================
Variant: Ld4-Ld24/1x16x64
Samples: 5000
Crop style: adm_square
SWA window: None

Metrics:
  FID   : 21.7741
  FDD   : 24.3235
  SSIM  : 0.9214
  PSNR  : 32.5742

Saved comparison grid to: results/L-64_coco-val_256p_adm_square/comparison_grid.jpg
```

The comparison grid shows original | reconstruction | diff (5x amplified) for visual inspection.

### Example Evaluation Matrix

```bash
# Standard benchmark at 256p (ADM-style, comparable to other VAEs)
modal run scripts/modal/eval_vae.py --model L-64 --dataset coco-val --crop-style adm_square --max-size 256
modal run scripts/modal/eval_vae.py --model L-32 --dataset coco-val --crop-style adm_square --max-size 256
modal run scripts/modal/eval_vae.py --model L-16 --dataset coco-val --crop-style adm_square --max-size 256

# Higher resolution (512p)
modal run scripts/modal/eval_vae.py --model L-64 --dataset coco-val --crop-style adm_square --max-size 512
```

### High-Resolution with SWA

For 1024p+ images, use Sliding Window Attention to reduce memory:

```bash
# 1024p with SWA window of 8
modal run scripts/modal/eval_vae.py --model L-64 --dataset div8k --max-size 1024 --swa-window 8

# 2048p (requires smaller batch size)
modal run scripts/modal/eval_vae.py --model L-64 --dataset div8k --max-size 2048 --swa-window 4 --batch-size 4
```

### Local Evaluation

You can also run evaluations locally (requires GPU):

```bash
python scripts/eval_vae.py \
  --model L-64 \
  --data /path/to/images \
  --max-size 256 \
  --num-samples 1000
```

## Troubleshooting

### "Volume not found"

Volumes are auto-created on first use. Run `modal run scripts/modal/setup_data.py` to pre-create.

### "Secret not found"

```bash
modal secret create huggingface-secret HF_TOKEN=hf_xxx
```

### "Out of memory"

- Reduce `--batch-size` (default: 16)
- Use `--swa-window` for high-res images
- Use smaller `--max-size`

### Slow first run

The Modal image is built on first run. Pre-build with:
```bash
modal run scripts/modal/setup_env.py
```
