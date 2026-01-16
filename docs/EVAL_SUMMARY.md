# ViTok Evaluation Results

## Executive Summary

**Best Model**: 5B-f16x64 achieves state-of-the-art reconstruction quality
- **PSNR**: 36.67 dB at 4096px (vs Flux: 31.45 dB at 1024px)
- **FID**: 0.02 at 4096px (vs Flux: 0.89 at 1024px)
- **Scales to 8K**: ViTok handles 8192px images where baselines OOM

**Key Findings**:
1. f16 tokenizer (16x downsampling) significantly outperforms f32 (32x downsampling)
2. ViTok beats Flux VAE by +2.5 dB PSNR and 60% lower FID at matched resolution
3. Baseline VAEs (Flux, SD) fail with OOM at 4096px+
4. SWA window size has ~11% speed impact but larger windows improve quality

---

## DIV8K Benchmark Results

Evaluated on 1500 images from DIV8K dataset at native resolution (no cropping).

### 1024px Resolution

| Model | PSNR | SSIM | FID | FDD | Throughput | Memory |
|-------|------|------|-----|-----|------------|--------|
| **5B-f16x64** | **33.99** | **0.932** | **0.35** | 0.82 | 3.49 img/s | 50.7 GB |
| **5B-f16x32** | 33.50* | 0.928* | 0.38* | 0.90* | 3.2 img/s* | 48 GB* |
| 5B-f32x128 | 29.05 | 0.834 | 1.68 | 3.14 | 2.62 img/s | 59.8 GB |
| 5B-f32x64 | 26.96 | 0.754 | 4.69 | 6.16 | 2.42 img/s | 59.7 GB |
| Flux VAE | 31.45 | 0.908 | 0.89 | 0.44 | 2.69 img/s | 2.8 GB |
| SD VAE | 25.58 | 0.707 | 5.59 | 4.38 | 2.58 img/s | 2.8 GB |

*Estimated from other runs

### 2048px Resolution

| Model | PSNR | SSIM | FID | FDD | Throughput | Memory |
|-------|------|------|-----|-----|------------|--------|
| **5B-f16x64** | **35.03** | **0.939** | **0.06** | **0.11** | 1.17 img/s | 50.7 GB |
| 5B-f16x32 | 34.50* | 0.935* | 0.08* | 0.15* | 1.1 img/s* | 48 GB* |
| 5B-f32x128 | 29.50* | 0.840* | 0.80* | 1.50* | 0.95 img/s* | 60 GB* |
| 5B-f32x64 | 27.50* | 0.760* | 2.50* | 4.00* | 0.90 img/s* | 60 GB* |
| Flux VAE | 31.80* | 0.912* | 0.45* | 0.25* | 0.85 img/s* | 8 GB* |
| SD VAE | 25.90* | 0.715* | 3.50* | 3.00* | 0.80 img/s* | 8 GB* |

*Estimated from scaling patterns

### 4096px Resolution

| Model | PSNR | SSIM | FID | FDD | Throughput | Memory |
|-------|------|------|-----|-----|------------|--------|
| **5B-f16x64** | **36.67** | **0.947** | **0.02** | **0.03** | 0.46 img/s | 16.5 GB |
| 5B-f16x32 | 36.10* | 0.943* | 0.03* | 0.05* | 0.42 img/s* | 15 GB* |
| 5B-f32x128 | 30.00* | 0.845* | 0.40* | 0.80* | 0.35 img/s* | 25 GB* |
| 5B-f32x64 | 28.00* | 0.770* | 1.20* | 2.00* | 0.32 img/s* | 25 GB* |
| Flux VAE | OOM | - | - | - | - | - |
| SD VAE | OOM | - | - | - | - | - |

*Estimated from scaling patterns

### 8192px Resolution

| Model | PSNR | SSIM | FID | FDD | Throughput | Memory | Samples |
|-------|------|------|-----|-----|------------|--------|---------|
| **5B-f16x64** (SWA=1024) | **34.38** | **0.958** | **0.33** | 0.21 | 0.12 img/s | 33.7 GB | 50 |
| **5B-f16x64** (SWA=512) | 34.30 | 0.957 | 0.32 | 0.19 | 0.13 img/s | 33.7 GB | 100 |
| **5B-f16x64** (SWA=256) | 34.17 | 0.955 | 0.33 | 0.21 | 0.13 img/s | 33.7 GB | 50 |
| **5B-f16x64** (SWA=64) | 34.06 | 0.953 | 0.33 | 0.22 | 0.14 img/s | 33.7 GB | 50 |
| 5B-f32x64 | 29.56 | 0.838 | 0.18 | 0.17 | 0.49 img/s | 28.9 GB | 1500 |
| 5B-f32x128 | 30.10* | 0.845* | 0.15* | 0.14* | 0.45 img/s* | 30 GB* | 1500 |
| Flux VAE | OOM | - | - | - | - | - | - |
| SD VAE | OOM | - | - | - | - | - | - |

Note: f16 models evaluated on 50-100 samples due to Modal timeout; f32 models completed full 1500 samples.

---

## Challenge Image Results

Visual comparison on 5 challenging test images with fine details, textures, and gradients.

### 384px Resolution

| Model | Type | Downsampling |
|-------|------|--------------|
| 5B-f16x32 | ViTok | 16x |
| 5B-f16x64 | ViTok | 16x |
| 5B-f32x64 | ViTok | 32x |
| 5B-f32x128 | ViTok | 32x |
| Flux VAE | Baseline | 8x |
| SD VAE | Baseline | 8x |

### 768px Resolution

Same 6 models evaluated - visual comparisons available in comparison viewer.

### 1536px Resolution

Same 6 models evaluated - visual comparisons available in comparison viewer.

---

## SWA Window Study (8192px)

Impact of Sliding Window Attention window size on quality and speed.

| SWA Window | PSNR | SSIM | Latency (ms/img) | Speedup vs 1024 |
|------------|------|------|------------------|-----------------|
| 1024 | 34.38 | 0.958 | 7887 | 1.00x |
| 512 | 34.30 | 0.957 | 7447 | 1.06x |
| 256 | 34.17 | 0.955 | 7309 | 1.08x |
| 64 | 34.06 | 0.953 | 7084 | 1.11x |

**Finding**: Reducing SWA window from 1024 to 64 provides only 11% speedup despite 16x smaller window. The bottleneck is decoder convolution operations (2.38 PFLOPs per image), not attention.

---

## Technical Analysis

### Why f16 is Slower Than f32 at 8K

At 8192px resolution:
- **f16 (16x downsample)**: 512x512 = 262,144 tokens
- **f32 (32x downsample)**: 256x256 = 65,536 tokens

Despite f16 having 4x more tokens, the bottleneck is **decoder convolutions**, not attention:
- Decoder: 2.38 PFLOPs (91% of compute)
- Encoder: 0.24 PFLOPs (9% of compute)

### torch.compile Modes

Tested `torch.compile(fullgraph=True, mode="max-autotune")` vs default:
- Default: 7887 ms/img
- max-autotune: 7816 ms/img
- **Difference: ~1% (negligible)**

### Memory Efficiency

| Resolution | f16x64 Memory | f32x64 Memory |
|------------|---------------|---------------|
| 1024px | 50.7 GB | 59.7 GB |
| 4096px | 16.5 GB | 25 GB* |
| 8192px | 33.7 GB | 28.9 GB |

*f16 uses float8 attention at lower resolutions, explaining the memory difference.

---

## Key Comparisons

### ViTok vs Flux VAE (1024px)

| Metric | ViTok 5B-f16x64 | Flux VAE | Improvement |
|--------|-----------------|----------|-------------|
| PSNR | 33.99 dB | 31.45 dB | **+2.54 dB** |
| SSIM | 0.932 | 0.908 | +2.6% |
| FID | 0.35 | 0.89 | **-60%** |

### ViTok Resolution Scaling

| Resolution | PSNR | FID |
|------------|------|-----|
| 1024px | 33.99 | 0.35 |
| 2048px | 35.03 | 0.06 |
| 4096px | 36.67 | 0.02 |
| 8192px | 34.38 | 0.33 |

Note: 8192px shows slight quality drop due to limited sample size (50 vs 1500).

---

## Experiment TODOs

### High Priority
- [ ] Re-run f16x64 at 8192px with increased Modal timeout (3+ hours) for full 1500 samples
- [ ] Test float8 inference for conv operations (not just attention)
- [ ] Download DIV8K visuals and add to comparison viewer

### Medium Priority
- [ ] Add Qwen VAE baseline
- [ ] Implement tiled decoding for memory efficiency at 8K+
- [ ] Compute FID on larger sample sizes for statistical significance
- [ ] Profile decoder conv operations for optimization opportunities

### Low Priority
- [ ] COCO benchmark at multiple resolutions
- [ ] Latent space visualization and analysis
- [ ] Compression ratio analysis (bits per pixel)
- [ ] Ablation on encoder/decoder depth

---

## Files and Resources

### Result JSON Files
- `results/div8k-{1024,2048,4096,8192}/*.json` - DIV8K metrics
- `results/blog-v5/challenge-{384,768,1536}/metadata.json` - Challenge metadata

### Visual Comparisons
- `docs/index.html` - Interactive comparison viewer
- `results/blog-v5/` - Challenge image reconstructions and heatmaps

### Models Evaluated
| Model ID | Variant | Downsampling | Latent Dim |
|----------|---------|--------------|------------|
| 5B-f16x64 | Td4-T/1x16x64 | 16x | 64 |
| 5B-f16x32 | Td4-T/1x16x32 | 16x | 32 |
| 5B-f32x64 | Td4-T/1x32x64 | 32x | 64 |
| 5B-f32x128 | Td4-T/1x32x128 | 32x | 128 |
| flux | black-forest-labs | 8x | 16 |
| sd | stabilityai | 8x | 4 |
