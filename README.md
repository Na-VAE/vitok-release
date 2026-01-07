# ViTok-v2

**ViTok-v2** (Vision Transformer Tokenizer v2) is a ViT-based image tokenizer designed for generative models. It features native aspect-ratio support via NaFlex patchification, enabling deployment at arbitrary resolutions while maintaining the scaling advantages of Vision Transformer architectures.

> **Note**: This is an independent public reimplementation by Philippe Hansen-Estruch and is not affiliated with Meta. The original research was conducted at Meta using a separate internal codebase.

> **Coming Soon**: ArXiv paper and pretrained model weights.

## Key Features

- **NaFlex Patchification**: Native aspect-ratio training with flexible token budgets
- **Asymmetric Encoder-Decoder**: Shallow encoders paired with deep decoders for optimal reconstruction
- **2D RoPE**: Rotary position embeddings for spatial awareness at any resolution
- **Streaming Data**: WebDataset and HuggingFace Hub support for large-scale training

## Installation

```bash
pip install -e .

# With training dependencies
pip install -e ".[train]"

# With evaluation dependencies
pip install -e ".[eval]"

# Development
pip install -e ".[dev]"
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
# Run with default model (L-64) and astronaut test image
modal run scripts/modal/inference.py

# Specify model variant
modal run scripts/modal/inference.py --model T-64

# Use your own image
modal run scripts/modal/inference.py --model L-64 --image path/to/image.jpg

# Save output locally
modal run scripts/modal/inference.py --output reconstructed.png

# List available pretrained models
modal run scripts/modal/inference.py --list-models
```

Available models: `L-16`, `L-32`, `L-64`, `T-64`, `T-128`, `T-256`

## Quick Start

### Encode and Decode Images

```python
from safetensors.torch import load_file
from vitok import AE, decode_variant, preprocess, postprocess
from PIL import Image
import torch

# Load pretrained AE
model = AE(**decode_variant("Ld2-Ld22/1x16x64"))
model.to(device="cuda", dtype=torch.bfloat16)
model.load_state_dict(load_file("path/to/checkpoint.safetensors"))
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
encoder = AE(**decode_variant("Ld2-Ld22/1x16x64"), decoder=False)
encoder.load_state_dict(load_file("checkpoint.safetensors"), strict=False)

# Decoder only
decoder = AE(**decode_variant("Ld2-Ld22/1x16x64"), encoder=False)
decoder.load_state_dict(load_file("checkpoint.safetensors"), strict=False)
```

## Model Variants

Format: `{encoder}[-{decoder}]/{temporal}x{spatial}x{channels}`

| Variant | Description |
|---------|-------------|
| `B/1x16x64` | Base encoder/decoder, stride 16, 64 latent channels |
| `L/1x16x64` | Large encoder/decoder |
| `Ld2-Ld22/1x16x64` | 2-layer encoder, 22-layer decoder (asymmetric) |
| `Gd4-G/1x16x64` | 4-layer Giant encoder, full Giant decoder |

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
vitok-release/
├── vitok/                    # Core library
│   ├── ae.py                 # AE, decode_variant
│   ├── naflex_io.py          # preprocess, postprocess, unpatchify
│   ├── data.py               # create_dataloader
│   ├── models/               # AE, DiT implementations
│   └── pp/                   # Preprocessing pipeline DSL
├── scripts/                  # Training and utility scripts
│   └── modal/                # Modal inference scripts
├── tests/
│   ├── cpu/                  # CPU tests (pytest)
│   ├── gpu/                  # GPU tests (Modal)
│   ├── visual/               # Visual inspection tests
│   └── utils/                # Debug utilities
└── benchmarks/               # Performance benchmarks
```

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
