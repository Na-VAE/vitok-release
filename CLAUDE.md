# Claude Code Instructions for vitok-release

## Project Overview

ViTok is a Vision Transformer tokenizer with autoencoder (AE) and diffusion transformer (DiT) components.

## Environment Setup

### Local Development

```bash
# Create virtual environment
uv venv .venv
source .venv/bin/activate

# Install dependencies
uv pip install -e ".[dev]"
```

### Modal (GPU Training & Testing)

Modal is used for GPU training and testing.

```bash
# Install modal
pip install modal

# Authenticate (one-time)
modal token new

# Set up secrets for dataset access and logging
modal secret create huggingface-secret HF_TOKEN=<your-hf-token>
modal secret create wandb-secret WANDB_API_KEY=<your-wandb-key>
```

## Running Tests

### Local CPU Tests

```bash
pytest tests/ -v                    # All tests
pytest tests/test_ae.py -v          # AE tests
pytest tests/test_dit.py -v         # DiT tests
pytest tests/test_pp.py -v          # Preprocessing tests
```

### Modal GPU Tests

```bash
# Quick smoke tests (~1 min, cheapest)
modal run modal_tests/test_all.py --quick

# Full GPU tests (~3 min)
modal run modal_tests/test_all.py

# Individual component tests
modal run modal_tests/test_ae_gpu.py
modal run modal_tests/test_dit_gpu.py
```

### Testing Data Loading on Modal

For quick data pipeline tests, create a simple test:

```bash
# Test data loading with cheap T4 GPU
modal run scripts/modal_train_vae.py --sync --steps 5

# Or use the debug script for data-only testing (no training)
modal run modal_tests/test_data.py
```

## Training

### Local Training

```bash
# Single GPU
python scripts/train_vae.py \
    --data /path/to/shards/*.tar \
    --output_dir checkpoints/vae

# Multi-GPU with FSDP2
torchrun --nproc_per_node=8 scripts/train_vae.py \
    --data hf://timm/imagenet-22k-wds/imagenet22k-train-{0000..0099}.tar \
    --fsdp \
    --output_dir checkpoints/vae
```

### Modal Training

```bash
# First time: sync code to Modal volume
modal run scripts/modal_train_vae.py --sync-only

# Debug run (1xA100, small batch)
modal run scripts/modal_train_vae.py --sync --debug --steps 100 --wandb-project vitok

# Production run (8xA100, FSDP)
# Use tmux/screen to keep running after disconnect
tmux new -s train
modal run scripts/modal_train_vae.py --sync --batch-size 128 --steps 100000 --wandb-project vitok --wandb-name "run-name"
# Ctrl+B, D to detach. Reconnect: tmux attach -t train

# Default settings:
#   - 8xA100-80GB with FSDP
#   - batch_size=64 per GPU (512 total), use --batch-size 128 for 1024 total
#   - lr=3e-4, warmup=1%, weight_decay=0.01
#   - ImageNet-22k (1024 shards)
```

### Modal Volumes

Checkpoints are saved to Modal Volumes (persistent cloud storage):

```bash
# List volumes
modal volume list

# List checkpoints
modal volume ls vitok-checkpoints

# Download checkpoint locally
modal volume get vitok-checkpoints /vae/step_5000 ./local_checkpoint/

# Delete old checkpoints
modal volume rm vitok-checkpoints /vae-debug --recursive
```

## HuggingFace Data Sources

Use brace expansion syntax to avoid HfFileSystem API stalls:

```python
# GOOD: Brace expansion (no API call, instant)
"hf://timm/imagenet-22k-wds/imagenet22k-train-{0000..0049}.tar"

# SLOWER: Glob pattern (requires HF API call)
"hf://timm/imagenet-22k-wds/imagenet22k-train-00*.tar"
```

## Key Directories

- `vitok/` - Core library (AE, DiT, data loading, preprocessing)
- `scripts/` - Training and utility scripts
- `tests/` - CPU tests (pytest)
- `modal_tests/` - GPU tests (Modal)

## Code Style

- Use type hints
- Keep functions focused and small
- Follow existing patterns in the codebase
