# ViTok-v2

**ViTok-v2** (Vision Transformer Tokenizer v2) is a ViT-based image tokenizer designed for generative models. It features native aspect-ratio support via NaFlex patchification, enabling deployment at arbitrary resolutions while maintaining the scaling advantages of Vision Transformer architectures.

> **Note**: This is an independent public reimplementation by Philippe Hansen-Estruch and is not affiliated with Meta. The original research was conducted at Meta using a separate internal codebase.

> **Coming Soon**: ArXiv paper and pretrained model weights.

## Key Features

- **NaFlex Patchification**: Native aspect-ratio training with flexible token budgets
- **Asymmetric Encoder-Decoder**: Shallow encoders paired with deep decoders for optimal reconstruction
- **Flow Matching DiT**: Diffusion Transformer with UniPC sampling for image generation
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

## Quick Start

### Encode and Decode Images

```python
from vitok import AEConfig, load_ae, preprocess_images, postprocess_images
from PIL import Image

# Load pretrained AE
config = AEConfig(variant="Ld2-Ld22/1x16x64", variational=True)
ae = load_ae("path/to/checkpoint.safetensors", config, device="cuda")

# Encode image
image = Image.open("input.jpg")
patch_dict = preprocess_images(image, device="cuda")
encoded = ae.encode(patch_dict)
z = encoded['z']

# Decode back
decode_dict = {
    'z': z,
    'ptype': patch_dict['ptype'],
    'yidx': patch_dict['yidx'],
    'xidx': patch_dict['xidx'],
    'original_height': patch_dict['original_height'],
    'original_width': patch_dict['original_width'],
}
decoded = ae.decode(decode_dict)
images = postprocess_images(decoded, output_format="0_255", unpack=True)
```

### Generate Images with DiT

```python
from vitok import AEConfig, DiTConfig, load_ae, load_dit
from vitok.diffusion.unipc import FlowUniPCMultistepScheduler
import torch

# Load models
ae = load_ae("path/to/ae.safetensors", AEConfig(variant="Ld2-Ld22/1x16x64"), device="cuda")
dit = load_dit("path/to/dit.safetensors", DiTConfig(variant="L/256", code_width=64), device="cuda")

# Setup scheduler
scheduler = FlowUniPCMultistepScheduler(shift=2.0)
scheduler.set_timesteps(num_inference_steps=20)

# Generate
labels = torch.tensor([207, 360, 387, 974], device="cuda")  # ImageNet classes
z = torch.randn(4, 256, 64, device="cuda")

for t in scheduler.timesteps:
    model_output = dit({'z': z, 't': t.expand(4), 'context': labels})
    z = scheduler.step(model_output, t, z).prev_sample

# Decode with ae.decode(...)
```

## Model Variants

### Autoencoder

Format: `{encoder}[-{decoder}]/{temporal}x{spatial}x{channels}`

| Variant | Description |
|---------|-------------|
| `B/1x16x64` | Base encoder/decoder, stride 16, 64 latent channels |
| `L/1x16x64` | Large encoder/decoder |
| `Ld2-Ld22/1x16x64` | 2-layer encoder, 22-layer decoder (asymmetric) |
| `Gd4-G/1x16x64` | 4-layer Giant encoder, full Giant decoder |

### DiT (Diffusion Transformer)

Format: `{model}/{num_tokens}`

| Variant | Description |
|---------|-------------|
| `B/256` | Base DiT, 256 tokens (16x16) |
| `L/256` | Large DiT, 256 tokens |
| `L/1024` | Large DiT, 1024 tokens (32x32) |
| `G/256` | Giant DiT, 256 tokens |

## Training

See `examples/` for training scripts:

```bash
# Train DiT on ImageNet
python examples/train_dit.py \
    --ae_checkpoint path/to/ae.safetensors \
    --hf_repo ILSVRC/imagenet-1k \
    --dit_variant L/256

# Train VAE
python examples/train_vae.py \
    --variant Ld2-Ld22/1x16x64 \
    --data_paths /path/to/shards/
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# GPU tests via Modal
modal run modal/test_all.py
```

See [TESTING.md](TESTING.md) for detailed testing instructions.

## Project Structure

```
vitok/
├── vitok/
│   ├── ae.py                 # AEConfig, create_ae, load_ae
│   ├── dit.py                # DiTConfig, create_dit, load_dit
│   ├── naflex_io.py          # Image preprocessing/postprocessing
│   ├── models/               # AE, DiT implementations
│   ├── diffusion/            # UniPC scheduler
│   └── pp/                   # Preprocessing pipeline DSL
├── examples/                 # Training and inference examples
├── tests/                    # Test suite
└── modal/                    # GPU testing infrastructure
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

This is a PyTorch reimplementation designed for simplicity and flexibility, supporting single GPU, single node, and multi-node training environments. The codebase is based on a refactor of:

- [ViTok](https://github.com/google-research/big_vision) (JAX, part of big_vision)
- [big_vision](https://github.com/google-research/big_vision) by Google Research

## Disclaimer

This repository is an independent public reimplementation of the ViTok-v2 architecture by Philippe Hansen-Estruch. It is not affiliated with, endorsed by, or connected to Meta or Google in any way. The original ViTok-v2 research was conducted at Meta using a separate internal codebase that is not publicly available.
