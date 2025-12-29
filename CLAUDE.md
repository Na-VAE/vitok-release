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

### Modal (GPU Testing)

Modal is used for GPU testing. All GPU tests are in `modal_tests/`.

```bash
# Install modal
pip install modal

# Authenticate (one-time)
modal token new

# Set up HuggingFace secret for dataset access
modal secret create huggingface-secret HF_TOKEN=<your-token>
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

# Run training (use --sync after code changes)
modal run scripts/modal_train_vae.py --steps 100
modal run scripts/modal_train_vae.py --sync --steps 1000
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
- `examples/` - Example configs and notebooks

## Code Style

- Use type hints
- Keep functions focused and small
- Follow existing patterns in the codebase
