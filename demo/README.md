---
title: ViTok Image Tokenizer
emoji: 🖼️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.9.1"
app_file: app.py
pinned: true
license: mit
---

# ViTok: Vision Transformer Image Tokenizer

Interactive demo for ViTok, a Vision Transformer-based image tokenizer with native aspect-ratio support (NaFlex).

## Features

- **Upload any image** for instant encode/decode reconstruction
- **Compare 9 model variants** - 350M and 5B parameter models with different strides and channels
- **Three-panel view**: Original | Reconstruction | Difference Heatmap
- **Real-time metrics**: SSIM and PSNR quality scores
- **Latent visualization**: See compression ratios and tensor shapes

## Models

| Model | Encoder | Decoder | Stride | Channels | Use Case |
|-------|---------|---------|--------|----------|----------|
| 5B-f16x64 | 463M | 4.5B | 16 | 64 | Best quality (default) |
| 5B-f16x32 | 463M | 4.5B | 16 | 32 | Higher compression |
| 5B-f16x16 | 463M | 4.5B | 16 | 16 | Highest compression |
| 5B-f32x64 | 463M | 4.5B | 32 | 64 | Faster, lower res latents |
| 5B-f32x128 | 463M | 4.5B | 32 | 128 | Balanced |
| 5B-f32x256 | 463M | 4.5B | 32 | 256 | Lower compression |
| 350M-f16x64 | 51M | 303M | 16 | 64 | Fast inference |
| 350M-f16x32 | 51M | 303M | 16 | 32 | Fast + higher compression |
| 350M-f16x16 | 51M | 303M | 16 | 16 | Fastest + highest compression |

## Difference Heatmap

The heatmap visualizes per-pixel reconstruction error:
- **Blue** = Low error (good reconstruction)
- **Red** = High error (detail loss)

This helps identify which image regions are hardest to reconstruct (typically high-frequency details, text, fine textures).

## Links

- [Paper (ArXiv)](https://arxiv.org/abs/2501.09755)
- [GitHub Repository](https://github.com/philippe-eecs/vitok-release)
- [Models on HuggingFace](https://huggingface.co/Na-VAE)

## Citation

```bibtex
@article{vitok2025,
  title={ViTok: A Vision Tokenizer for Native Aspect-Ratio Image Encoding},
  author={Hansen, Philippe and others},
  journal={arXiv preprint arXiv:2501.09755},
  year={2025}
}
```
