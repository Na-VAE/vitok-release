# ViTok

Vision Transformer Tokenizer with NaFlex - A flexible resolution image tokenizer for generative models.

## Features

- **NaFlex Patchification**: Variable resolution support with flexible token budgets
- **Vision Transformer Autoencoder**: Encode images to compact latent representations
- **Diffusion Transformer (DiT)**: Class-conditional image generation with flow matching
- **Streaming Data**: HuggingFace Hub and WebDataset support for large-scale training
- **2D RoPE**: Rotary position embeddings for spatial awareness
- **FSDP Training**: Distributed training with PyTorch FSDP2

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
z = encoded['posterior'].mode()

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
from vitok.diffusion.flow_matching import euler_sample, FlowMatchingScheduler
import torch

# Load models
ae = load_ae("path/to/ae.safetensors", AEConfig(variant="Ld2-Ld22/1x16x64"), device="cuda")
dit = load_dit("path/to/dit.safetensors", DiTConfig(variant="L/256", code_width=64), device="cuda")

# Generate
scheduler = FlowMatchingScheduler()
labels = torch.tensor([207, 360, 387, 974], device="cuda")  # ImageNet classes
z = torch.randn(4, 256, 64, device="cuda")

samples = euler_sample(dit, scheduler, z, labels, num_steps=50, cfg_scale=4.0)
# Decode samples with ae.decode(...)
```

## Model Variants

### Autoencoder Variants

Format: `{encoder}[-{decoder}]/{temporal}x{spatial}x{channels}`

| Variant | Description |
|---------|-------------|
| `B/1x16x64` | Base encoder/decoder, stride 16, 64 channels |
| `L/1x16x64` | Large encoder/decoder, stride 16, 64 channels |
| `Ld2-Ld22/1x16x64` | 2-layer encoder, 22-layer decoder |
| `Gd32/1x16x64` | Giant with 32 depth layers |

### DiT Variants

Format: `{model}/{num_tokens}`

| Variant | Description |
|---------|-------------|
| `B/256` | Base DiT, 256 tokens (16x16 grid) |
| `L/256` | Large DiT, 256 tokens |
| `L/1024` | Large DiT, 1024 tokens (32x32 grid) |
| `G/256` | Giant DiT, 256 tokens |

## Training

### Train DiT on ImageNet

```bash
# Single GPU
python scripts/train_dit.py \
    --ae_checkpoint path/to/ae.safetensors \
    --hf_repo ILSVRC/imagenet-1k \
    --dit_variant L/256 \
    --batch_size 64

# Multi-GPU with FSDP
torchrun --nproc_per_node=8 scripts/train_dit.py \
    --ae_checkpoint path/to/ae.safetensors \
    --hf_repo ILSVRC/imagenet-1k \
    --dit_variant L/256 \
    --batch_size 32 \
    --fsdp
```

### Train on Custom Data

```bash
python scripts/train_dit.py \
    --ae_checkpoint path/to/ae.safetensors \
    --data_paths /path/to/shards/ \
    --dit_variant B/256 \
    --num_classes 100 \
    --batch_size 32
```

## Sampling

```bash
python scripts/sample_dit.py \
    --ae_checkpoint path/to/ae.safetensors \
    --dit_checkpoint path/to/dit.safetensors \
    --classes 207 360 387 974 \
    --cfg_scale 4.0 \
    --num_samples 4 \
    --output_dir samples/
```

## Data Format

ViTok uses WebDataset tar files for training. Each sample should contain:

```
sample_key.jpg    # Image (jpg, png, or webp)
sample_key.cls    # Class label (optional, integer)
```

### Create Training Shards

```python
import webdataset as wds
import io
from PIL import Image

with wds.TarWriter("shard-00000.tar") as sink:
    for i, (image, label) in enumerate(dataset):
        # Convert image to bytes
        buf = io.BytesIO()
        image.save(buf, format='JPEG')

        sink.write({
            "__key__": f"{i:06d}",
            "jpg": buf.getvalue(),
            "cls": str(label).encode(),
        })
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Test data streaming
pytest tests/test_data_stream.py -v

# Test AE compatibility (requires vitokv2)
pytest tests/test_ae_compatibility.py -v
```

## Project Structure

```
vitok/
├── vitok/
│   ├── __init__.py           # Public API
│   ├── ae.py                  # AEConfig + create_ae/load_ae
│   ├── dit.py                 # DiTConfig + create_dit/load_dit
│   ├── naflex.py              # NaFlex transform config
│   ├── data.py                # Streaming dataloader config
│   ├── models/                # Core model implementations
│   ├── transforms/            # Image transforms and patching
│   ├── datasets/              # Data loading utilities
│   ├── diffusion/             # Flow matching scheduler
│   └── configs/               # Variant parser
├── scripts/
│   ├── train_dit.py           # DiT training
│   ├── sample_dit.py          # DiT sampling
│   └── install_eval_datasets.sh
├── examples/
│   ├── encode_decode.py
│   ├── dit_generation.py
│   └── configs/
├── tests/
└── pyproject.toml
```

## License

MIT License

## Citation

```bibtex
@article{vitok2024,
  title={ViTok: Vision Transformer Tokenizer},
  author={...},
  year={2024}
}
```
