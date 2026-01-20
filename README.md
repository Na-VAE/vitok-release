# ViTok-v2

**ViTok-v2** (Vision Transformer Tokenizer v2) is a ViT-based image tokenizer designed for generative models. It achieves **4x token reduction** for diffusion training through f16 compression (256 tokens vs 1024), matching SD/Flux generation quality via scaled decoders.

[![Blog](https://img.shields.io/badge/Blog-GitHub%20Pages-blue)](https://na-vae.github.io/vitok-release/)
[![Paper (ViTok)](https://img.shields.io/badge/arXiv-2501.09755-b31b1b)](https://arxiv.org/abs/2501.09755)
[![Paper (ViTok-v2)](https://img.shields.io/badge/arXiv-Coming%20Soon-lightgrey)](https://github.com/Na-VAE/vitok-release)
[![Models](https://img.shields.io/badge/Models-Registry-yellow)](https://github.com/Na-VAE/vitok-release/blob/main/vitok/pretrained.py)

> **Note**: This is an independent public reimplementation by Philippe Hansen-Estruch and is not affiliated with Meta. The original research was conducted at Meta using a separate internal codebase.

## Key Features

- **Configurable Channels**: Choose latent dimension (16, 32, 64, 128, 256 channels)
- **Flexible Compression**: 5B models support f16 (16x compression) or f32 (32x compression) patch sizes
- **NaFlex Format**: Any resolution/aspect ratio - images become patch dictionaries with spatial indices
- **Asymmetric Architecture**: Lightweight encoders (4 layers) with deep decoders (24-40 layers)

### NaFlex Patch Dictionary

Images are converted to a patch dictionary format that supports variable resolutions:

```python
patch_dict = {
    'patches': Tensor,     # [B, N, C*P*P] flattened patches
    'patch_mask': Tensor,  # [B, N] valid patch indicators
    'row_idx': Tensor,     # [B, N] row positions for 2D RoPE
    'col_idx': Tensor,     # [B, N] column positions for 2D RoPE
    'grid_rows': Tensor,   # patch grid height
    'grid_cols': Tensor,   # patch grid width
}
```

## Installation

```bash
pip install git+https://github.com/Na-VAE/vitok-release.git
# or local
pip install -e .
```

## Quick Start

### Encode and Decode Images

```python
from vitok import AE, decode_variant, load_pretrained, preprocess, postprocess
import torch

# Load model (downloads weights automatically)
data = load_pretrained("350M-f16x64")
model = AE(**decode_variant(data['variant']))
model.load_state_dict({**data['encoder'], **data['decoder']})
model.to("cuda", torch.bfloat16).eval()

# Optional: compile for faster inference (requires PyTorch 2.0+)
model.encode = torch.compile(model.encode, fullgraph=True)
model.decode = torch.compile(model.decode, fullgraph=True)

# Encode/decode
image = "input.jpg"  # or PIL Image
patch_dict = preprocess(image, device="cuda")
z = model.encode(patch_dict)['z']
reconstructed = model.decode(model.encode(patch_dict))
images = postprocess(reconstructed, do_unpack=True)
```

### Encoder-Only or Decoder-Only

```python
from vitok import AE, decode_variant, load_pretrained

# Encoder only
data = load_pretrained("350M-f16x64", component="encoder")
encoder = AE(**decode_variant(data['variant']), decoder=False)
encoder.load_state_dict(data['encoder'], strict=False)

# Decoder only
data = load_pretrained("350M-f16x64", component="decoder")
decoder = AE(**decode_variant(data['variant']), encoder=False)
decoder.load_state_dict(data['decoder'], strict=False)
```

## Model Variants

### 350M Family (51M encoder + 303M decoder)

| Model | Variant | Channels | HuggingFace |
|-------|---------|----------|-------------|
| `350M-f16x16` | `Ld4-Ld24/1x16x16` | 16 | [philippehansen/ViTok-v2-350M-f16x16](https://huggingface.co/philippehansen/ViTok-v2-350M-f16x16) |
| `350M-f16x32` | `Ld4-Ld24/1x16x32` | 32 | [philippehansen/ViTok-v2-350M-f16x32](https://huggingface.co/philippehansen/ViTok-v2-350M-f16x32) |
| `350M-f16x64` | `Ld4-Ld24/1x16x64` | 64 | [philippehansen/ViTok-v2-350M-f16x64](https://huggingface.co/philippehansen/ViTok-v2-350M-f16x64) |

### 5B Family (463M encoder + 4.5B decoder)

| Model | Variant | Patch | Channels | HuggingFace |
|-------|---------|-------|----------|-------------|
| `5B-f16x16` | `Td4-T/1x16x16` | 16 | 16 | [philippehansen/ViTok-v2-5B-f16x16](https://huggingface.co/philippehansen/ViTok-v2-5B-f16x16) |
| `5B-f16x32` | `Td4-T/1x16x32` | 16 | 32 | [philippehansen/ViTok-v2-5B-f16x32](https://huggingface.co/philippehansen/ViTok-v2-5B-f16x32) |
| `5B-f16x64` | `Td4-T/1x16x64` | 16 | 64 | [philippehansen/ViTok-v2-5B-f16x64](https://huggingface.co/philippehansen/ViTok-v2-5B-f16x64) |
| `5B-f32x64` | `Td4-T/1x32x64` | 32 | 64 | [philippehansen/ViTok-v2-5B-f32x64](https://huggingface.co/philippehansen/ViTok-v2-5B-f32x64) |
| `5B-f32x128` | `Td4-T/1x32x128` | 32 | 128 | [philippehansen/ViTok-v2-5B-f32x128](https://huggingface.co/philippehansen/ViTok-v2-5B-f32x128) |
| `5B-f32x256` | `Td4-T/1x32x256` | 32 | 256 | [philippehansen/ViTok-v2-5B-f32x256](https://huggingface.co/philippehansen/ViTok-v2-5B-f32x256) |

### Naming Convention

- **Format**: `{size}-f{patch}x{channels}` (e.g., `5B-f32x128` = 5B params, patch size 32, 128 latent channels)
- **Patch size**: f16 = 16×16 patches, f32 = 32×32 patches (more compression)
- **Channels**: Higher = more detail preserved (16→256)

**Choosing a model:**
- More tokens needed? Use f16 models (256 tokens for 256×256 image)
- Fewer tokens needed? Use f32 models (64 tokens for 256×256 image)
- Better reconstruction? Use higher channels (64, 128, 256)

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
├── pretrained.py        # load_pretrained, list_pretrained
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

## Evaluation

Evaluate pretrained models on standard benchmarks using reconstruction metrics (FID, FDD, SSIM, PSNR).

### Modal Cloud Evaluation (Recommended)

Run evaluation on Modal's cloud GPUs - no local GPU needed.

```bash
# First-time setup
pip install modal
modal token new
modal secret create huggingface-secret HF_TOKEN=<your-hf-token>
```

#### Stream from HuggingFace (No Download Required)

Stream datasets directly from HuggingFace - no need to pre-download anything:

```bash
# Stream COCO validation set
modal run scripts/eval_vae.py --model 350M-f16x64 --data coco

# Stream other datasets
modal run scripts/eval_vae.py --model 350M-f16x64 --data div8k      # High-res photos
modal run scripts/eval_vae.py --model 350M-f16x64 --data nature     # Nature/landscapes
modal run scripts/eval_vae.py --model 350M-f16x64 --data portraits  # Human faces
modal run scripts/eval_vae.py --model 350M-f16x64 --data animals    # Cats and dogs

# With options
modal run scripts/eval_vae.py --model 350M-f16x64 --data coco \
    --max-size 512 --num-samples 1000 --batch-size 32
```

Available streaming datasets:
| Dataset | Source | Description |
|---------|--------|-------------|
| `coco` | detection-datasets/coco | COCO val2017 (general objects) |
| `div8k` | Iceclear/DIV8K_TrainingSet | High-resolution photos |
| `nature` | eugenesiow/Div2k | Nature/landscape images |
| `portraits` | jlbaker361/celebrity-100k | Human face portraits |
| `text` | nielsr/funsd | Documents with text |
| `architecture` | GATE-engine/mini-Unsplash | Buildings/architecture |
| `animals` | cats_vs_dogs | Cat and dog images |

#### Compare Against Baseline VAEs

Evaluate Flux, Stable Diffusion, or Qwen VAEs for comparison:

```bash
# Baseline VAEs
modal run scripts/eval_vae.py --baseline flux --data coco
modal run scripts/eval_vae.py --baseline sd --data coco
modal run scripts/eval_vae.py --baseline qwen --data coco
```

#### Save Visual Samples

Save original/reconstruction pairs for qualitative comparison:

```bash
# Save 20 sample images to Modal volume
modal run scripts/eval_vae.py --model 350M-f16x64 --data coco \
    --save-visuals 20 --output-dir /output/samples/coco

# Download from Modal volume
modal volume get vitok-output /samples ./local_samples/
```

#### Pre-downloaded Datasets (Faster)

For repeated evaluations, pre-download datasets to Modal volume:

```bash
# Download datasets to Modal volume (one-time)
modal run scripts/modal/setup_data.py --data coco
modal run scripts/modal/setup_data.py --data div8k

# Evaluate with pre-downloaded data (faster, no streaming overhead)
modal run scripts/eval_vae.py --model 350M-f16x64 --data coco-val
modal run scripts/eval_vae.py --model 350M-f16x64 --data div8k --max-size 1024
```

### Local Evaluation

Evaluate on your local machine with a GPU:

```bash
# With a folder of images
python scripts/eval_vae.py --model 350M-f16x64 --data /path/to/images/

# With COCO val2017
python scripts/eval_vae.py --model 350M-f16x64 --data ./data/coco/val2017

# With a custom checkpoint
python scripts/eval_vae.py \
    --checkpoint path/to/model.safetensors \
    --variant Ld4-Ld24/1x16x64 \
    --data ./data/coco/val2017

# Save results to JSON
python scripts/eval_vae.py --model 350M-f16x64 --data /path/to/images/ \
    --output-json results.json
```

### Metrics

| Metric | Description | Better |
|--------|-------------|--------|
| FID | Fréchet Inception Distance (Inception-V3 features) | Lower |
| FDD | Fréchet DINO Distance (DINO features) | Lower |
| SSIM | Structural Similarity Index | Higher |
| PSNR | Peak Signal-to-Noise Ratio (dB) | Higher |

## Training

Train ViTok VAE from scratch or finetune pretrained models.

### Data Sources

Training supports multiple data sources:

```bash
# WebDataset shards (local)
--data /path/to/shards/*.tar

# HuggingFace Hub (streaming)
--data hf://timm/imagenet-22k-wds/imagenet22k-train-{0000..1023}.tar

# HuggingFace with brace expansion (faster than glob)
--data hf://ILSVRC/imagenet-1k/train-{00000..01023}.tar
```

### Modal Training (Recommended)

Train on Modal's cloud GPUs (8x A100):

```bash
# First-time setup
modal secret create wandb-secret WANDB_API_KEY=<your-key>

# Train with defaults (ImageNet-22k, 8x A100, FSDP)
modal run scripts/train_vae.py --steps 100000 --wandb-project vitok

# Custom training
modal run scripts/train_vae.py \
    --data hf://ILSVRC/imagenet-1k/train-{00000..01023}.tar \
    --variant Ld2-Ld22/1x16x64 \
    --steps 50000

# Finetune from pretrained
modal run scripts/train_vae.py --pretrained 350M-f16x64 --freeze-encoder --steps 10000

# Check training progress
modal volume ls vitok-output /checkpoints
```

### Local Training

```bash
# Single GPU
python scripts/train_vae.py \
    --data /path/to/shards/*.tar \
    --output_dir checkpoints/vae \
    --batch_size 32 \
    --steps 50000

# Multi-GPU with FSDP2 (recommended for large models)
torchrun --nproc_per_node=8 scripts/train_vae.py \
    --data hf://timm/imagenet-22k-wds/imagenet22k-train-{0000..0099}.tar \
    --fsdp \
    --output_dir checkpoints/vae

# Multi-GPU with DDP
torchrun --nproc_per_node=4 scripts/train_vae.py \
    --data hf://ILSVRC/imagenet-1k/train/*.tar \
    --output_dir checkpoints/vae
```

### Training Options

| Option | Default | Description |
|--------|---------|-------------|
| `--variant` | `Ld2-Ld22/1x16x64` | Model architecture |
| `--batch_size` | 32 | Per-GPU batch size |
| `--steps` | 100000 | Total training steps |
| `--lr` | 3e-4 | Learning rate |
| `--max_size` | 256 | Max image size |
| `--fsdp` | False | Use FSDP2 for large models |
| `--pretrained` | None | Pretrained model to finetune |
| `--freeze_encoder` | False | Freeze encoder (decoder-only finetuning) |

### Loss Functions

Training uses a combination of losses:

| Loss | Weight | Description |
|------|--------|-------------|
| Charbonnier | 1.0 | Smooth L1-like reconstruction loss |
| SSIM | 0.1 | Structural similarity |
| DINO Perceptual | 500.0 | DINO-based perceptual loss |

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
