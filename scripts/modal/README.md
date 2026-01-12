# Modal Setup for ViTok

Run ViTok on Modal cloud GPUs using the `--modal` flag on main scripts.

## Quick Start

```bash
# 1. Install modal
pip install modal

# 2. Authenticate (one-time)
modal token new

# 3. Set up secrets
modal secret create huggingface-secret HF_TOKEN=<your-hf-token>

# 4. Run evaluation on Modal
python scripts/eval_vae.py --modal --model L-64 --num-samples 5000
```

## Running Evaluation

Use the `--modal` flag on the main eval script:

```bash
# Basic evaluation
python scripts/eval_vae.py --modal --model L-64

# With dataset preset
python scripts/eval_vae.py --modal --model L-64 --dataset coco-val

# Full options
python scripts/eval_vae.py --modal \
  --model L-64 \
  --dataset coco-val \
  --crop-style adm_square \
  --max-size 256 \
  --num-samples 5000
```

### Available Options

| Option | Default | Description |
|--------|---------|-------------|
| `--modal` | - | Run on Modal cloud GPU |
| `--model` | - | Model alias (L-64, L-32, L-16, T-64, etc.) |
| `--dataset` | `coco-val` | Dataset preset (coco-val, imagenet-val, div8k) |
| `--crop-style` | `native` | Crop style (adm_square, native) |
| `--max-size` | `256` | Maximum image size |
| `--num-samples` | `5000` | Number of samples |
| `--output-json` | - | Save results to JSON |

## Dataset Setup

Pre-download datasets to Modal volume for faster runs:

```bash
# Download COCO val2017 (default, ~1GB)
modal run scripts/modal/setup_data.py

# Download specific dataset
modal run scripts/modal/setup_data.py --dataset coco
modal run scripts/modal/setup_data.py --dataset imagenet  # requires HF auth
modal run scripts/modal/setup_data.py --dataset div8k

# Check what's cached
modal run scripts/modal/setup_data.py --check
```

## Volumes

Modal volumes persist data across runs:

| Volume | Purpose |
|--------|---------|
| `vitok-weights` | Model weights, HF cache |
| `vitok-data` | Evaluation datasets |

```bash
# List volumes
modal volume list

# Check contents
modal volume ls vitok-data
```

## Configuration

All Modal configs are in `scripts/modal_config.py`:
- Shared images, volumes, secrets
- Config dicts: `EVAL_CONFIG`, `INFERENCE_CONFIG`, `TRAINING_CONFIG`

## Troubleshooting

### "Secret not found"
```bash
modal secret create huggingface-secret HF_TOKEN=hf_xxx
```

### "Out of memory"
- Reduce `--batch-size`
- Use `--swa-window` for high-res
- Use smaller `--max-size`
