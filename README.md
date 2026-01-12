# ViTok-v2

**ViTok-v2** (Vision Transformer Tokenizer v2) is a ViT-based image tokenizer designed for generative models. It features native aspect-ratio support via NaFlex patchification, enabling deployment at arbitrary resolutions while maintaining the scaling advantages of Vision Transformer architectures.

[![Try Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue)](https://huggingface.co/spaces/philippehansen/navae)
[![Paper](https://img.shields.io/badge/arXiv-2501.09755-b31b1b)](https://arxiv.org/abs/2501.09755)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/Na-VAE)

> **Note**: This is an independent public reimplementation by Philippe Hansen-Estruch and is not affiliated with Meta. The original research was conducted at Meta using a separate internal codebase.

## Key Features

- **NaFlex Patchification**: Native aspect-ratio training with flexible token budgets
- **Asymmetric Encoder-Decoder**: Shallow encoders paired with deep decoders for optimal reconstruction
- **2D RoPE**: Rotary position embeddings for spatial awareness at any resolution
- **Streaming Data**: WebDataset and HuggingFace Hub support for large-scale training

## Interactive Demo

Try ViTok reconstruction on your own images:

**[Launch Demo on HuggingFace Spaces](https://huggingface.co/spaces/philippehansen/navae)**

The demo shows:
- **Original** vs **Reconstruction** side-by-side
- **Difference heatmap** (blue=low error, red=high error)
- **SSIM/PSNR metrics** for quality assessment

<!-- Uncomment when comparison images are generated:
## Visual Results

<p align="center">
  <img src="assets/comparisons/astronaut_comparison.png" alt="Reconstruction Comparison" width="800"/>
</p>

*Comparison of ViTok-L-64 against SD-VAE and FLUX VAE on standard test images.*
-->

## Installation

```bash
pip install -e .
```

## Quick Start

### Encode and Decode Images

```python
from vitok import AE, decode_variant, preprocess, postprocess, download_pretrained
from safetensors.torch import load_file
from PIL import Image
import torch

# Load pretrained AE
weights_paths = download_pretrained("L-64")  # returns list of paths
weights = {}
for path in weights_paths:
    weights.update(load_file(path))
model = AE(**decode_variant("Ld4-Ld24/1x16x64"))
model.load_state_dict(weights)
model.to(device="cuda", dtype=torch.bfloat16)
model.eval()

# Encode image
image = Image.open("input.jpg")
patch_dict = preprocess(image, device="cuda")
encoded = model.encode(patch_dict)
z = encoded['z']

# Decode back
decoded = model.decode(encoded)
images = postprocess(decoded, output_format="0_255", do_unpack=True)
```

### Encoder-Only or Decoder-Only

```python
# Encoder only
encoder = AE(**decode_variant("Ld4-Ld24/1x16x64"), decoder=False)
encoder.load_state_dict(load_file(weights_path), strict=False)

# Decoder only
decoder = AE(**decode_variant("Ld4-Ld24/1x16x64"), encoder=False)
decoder.load_state_dict(load_file(weights_path), strict=False)
```

## Model Variants

### Pretrained Models

| Alias | Full Variant | Description |
|-------|--------------|-------------|
| `L-64` | `Ld4-Ld24/1x16x64` | Large, stride 16, 64 latent channels |
| `L-32` | `Ld4-Ld24/1x32x64` | Large, stride 32, 64 latent channels |
| `L-16` | `Ld4-Ld24/1x16x16` | Large, stride 16, 16 latent channels |
| `T-64` | `Td2-Td12/1x16x64` | Tiny, stride 16, 64 latent channels |
| `T-128` | `Td2-Td12/1x16x128` | Tiny, stride 16, 128 latent channels |
| `T-256` | `Td2-Td12/1x16x256` | Tiny, stride 16, 256 latent channels |

### Variant Format

Format: `{encoder}[-{decoder}]/{temporal}x{spatial}x{channels}`

- **Encoder/Decoder sizes**: `T` (Tiny), `S` (Small), `B` (Base), `L` (Large), `G` (Giant)
- **Asymmetric notation**: `Ld4-Ld24` = 4-layer Large encoder, 24-layer Large decoder
- **Temporal**: Always `1` for images
- **Spatial**: Patch stride (16 or 32)
- **Channels**: Latent dimension

## Testing

```bash
# CPU tests (fast, local)
pytest tests/cpu/ -v

# GPU tests via Modal
modal run tests/gpu/test_all.py --quick    # Quick tests (~1 min)
modal run tests/gpu/test_all.py            # Full tests (~3 min)

# Individual GPU tests
modal run tests/gpu/test_ae.py
modal run tests/gpu/test_dit.py

# Benchmarks
modal run benchmarks/benchmark_mfu.py
```

## Project Structure

```
vitok/
├── __init__.py          # Public API
├── data.py              # create_dataloader
├── pretrained.py        # download_pretrained, list_pretrained
├── utils.py             # Training utilities
├── models/
│   ├── ae.py            # AE + decode_variant
│   ├── dit.py           # DiT + decode_variant
│   └── modules/         # Attention, MLP, etc.
└── pp/                  # Preprocessing
    ├── ops.py           # patchify, unpatchify, sample_tiles
    ├── io.py            # preprocess, postprocess
    └── registry.py      # DSL parser
```

## Modal Quickstart (GPU Inference)

Run inference on Modal's cloud GPUs without local GPU setup.

### First-time Setup

```bash
# Install Modal CLI
pip install modal

# Authenticate (one-time)
modal token new

# Pre-build the inference environment (optional, speeds up first run)
modal run scripts/modal/setup_env.py

# Create volume for caching model weights
modal run scripts/modal/setup_volume.py
```

### Run Inference

```bash
# Run with default model and astronaut test image
modal run scripts/modal/inference.py

# Specify model variant
modal run scripts/modal/inference.py --model T-32x64

# Use your own image
modal run scripts/modal/inference.py --model L-16x64 --image path/to/image.jpg

# Save output locally
modal run scripts/modal/inference.py --output reconstructed.png

# List available pretrained models
modal run scripts/modal/inference.py --list-models
```

Available models: `L-64`, `L-32`, `L-16`, `T-64`, `T-128`, `T-256`

## Evaluation

Evaluate pretrained models on standard benchmarks using reconstruction metrics (FID, SSIM, PSNR).

### Install Evaluation Datasets

```bash
# Download COCO val2017 (~1GB, 5000 images)
./scripts/install_eval_datasets.sh
```

### Local Evaluation

```bash
# Evaluate a checkpoint on local data
python scripts/eval_vae.py \
    --checkpoint path/to/model.safetensors \
    --variant Ld4-Ld24/1x16x64 \
    --data ./data/coco/val2017 \
    --num-samples 5000 \
    --metrics fid ssim psnr

# With HuggingFace pretrained weights (uses --model for automatic download)
python scripts/eval_vae.py \
    --model L-64 \
    --data ./data/coco/val2017
```

### Modal Evaluation (Recommended)

Run evaluation on Modal's cloud GPUs without local GPU setup.

```bash
# Quick test (100 samples)
modal run scripts/modal/eval_vae.py --model L-64 --num-samples 100

# Full evaluation (1000 samples, default)
modal run scripts/modal/eval_vae.py --model L-64

# Complete benchmark (5000 samples)
modal run scripts/modal/eval_vae.py --model L-64 --num-samples 5000

# Evaluate different models
modal run scripts/modal/eval_vae.py --model L-16
modal run scripts/modal/eval_vae.py --model L-32

# List available models
modal run scripts/modal/eval_vae.py --list-models
```

### Metrics

| Metric | Description | Better |
|--------|-------------|--------|
| FID | Fréchet Inception Distance (Inception-V3 features) | Lower |
| SSIM | Structural Similarity Index | Higher |
| PSNR | Peak Signal-to-Noise Ratio (dB) | Higher |

## License

MIT License

## Citation

If you find this code or work helpful, please cite:

### ViTok-v2 (this work)

```bibtex
@article{hansenestruch2025vitokv2,
  title={ViTok-v2: A Vision Transformer Tokenizer for Generative Models},
  author={Hansen-Estruch, Philippe and Chen, Jiahui and Ramanujan, Vivek and Zohar, Orr and Ping, Yan and Sinha, Animesh and Georgopoulos, Markos and Schoenfeld, Edgar and Hou, Ji and Juefei-Xu, Felix and Vishwanath, Sriram and Thabet, Ali},
  year={2025},
  url={https://github.com/Na-VAE/vitok-release}
}
```

### ViTok-v1

This implementation builds upon ideas from the original ViTok work:

```bibtex
@article{hansenestruch2025vitok,
  title={Learnings from Scaling Visual Tokenizers for Reconstruction and Generation},
  author={Hansen-Estruch, Philippe and Yan, David and Chung, Ching-Yao and Zohar, Orr and Wang, Jialiang and Hou, Tingbo and Xu, Tao and Vishwanath, Sriram and Vajda, Peter and Chen, Xinlei},
  journal={arXiv preprint arXiv:2501.09755},
  year={2025}
}
```

## Acknowledgments

This is a PyTorch reimplementation designed for simplicity and flexibility, supporting single GPU, single node, and multi-node training environments.

## Disclaimer

This repository is an independent public reimplementation of the ViTok-v2 architecture by Philippe Hansen-Estruch. It is not affiliated with, endorsed by, or connected to Meta or Google in any way.
