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
- **Compare 6 model variants** - Large (L) and Tiny (T) with different latent sizes
- **Three-panel view**: Original | Reconstruction | Difference Heatmap
- **Real-time metrics**: SSIM and PSNR quality scores
- **Latent visualization**: See compression ratios and tensor shapes

## Models

| Model | Parameters | Latent Channels | Stride | Use Case |
|-------|------------|-----------------|--------|----------|
| L-64 | 1.1B | 64 | 16 | Best quality |
| L-32 | 1.1B | 64 | 32 | Faster, lower resolution |
| L-16 | 1.1B | 16 | 16 | High compression |
| T-64 | ~100M | 64 | 16 | Fast inference |
| T-128 | ~100M | 128 | 16 | Balanced |
| T-256 | ~100M | 256 | 16 | Lower compression |

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
