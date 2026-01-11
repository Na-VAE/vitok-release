# torch.compile + INT8 Quantization Research for ViTok VAE

## Executive Summary

| Optimization | Recommendation | Expected Speedup | When to Use |
|--------------|----------------|------------------|-------------|
| `torch.compile` | **Yes, with caveats** | 1.2-1.5x (after warmup) | Single-GPU eval, >1000 samples |
| INT8 (A100) | **Marginal benefit** | 1.0-1.25x | Memory-bound, large batches |
| FP8 (H100+) | **Yes** | 1.3-1.6x | H100+ only |
| bf16 baseline | **Safe default** | Baseline | Always works |

## torch.compile Analysis

### When It Helps
- Large model, fixed batch size, compute-bound workloads
- Long evaluation loops (>1000 samples to amortize compilation)
- Static input shapes (fixed image resolution)

### When It Hurts
- Small models (compilation overhead dominates)
- Dynamic shapes (causes recompilation)
- Frequent graph breaks
- Multi-GPU distributed eval (shape mismatches with flex_attention)

### Warmup Costs
- **Cold compile**: 30-120 seconds
- **Cached warmup**: 5-15 seconds
- **Break-even**: ~1200 samples (60s compile / 50ms saved per batch)

### Recommended Settings
```python
# Use reduce-overhead for faster compilation
encoder = torch.compile(encoder, fullgraph=True, mode='reduce-overhead')
decoder = torch.compile(decoder, fullgraph=True, mode='reduce-overhead')
```

### Current Issue in eval_vae.py
The current code uses `torch.compile(model, fullgraph=True)` without specifying mode. Adding `mode='reduce-overhead'` should improve compilation time.

## INT8 Quantization Analysis

### Key Finding
`torch.ao.quantization.quantize_dynamic` is **primarily for CPU**, not GPU. On GPU, bf16 is typically faster than INT8 dynamic quantization.

### A100 vs H100
- **A100** (compute 8.0): Strong INT8 tensor cores, but bf16/TF32 already fast
- **H100** (compute 9.0): Better INT8 + native FP8 support

### When INT8 Helps on GPU
- Memory-bound workloads (reduces bandwidth pressure)
- Very large batches (32+)
- Large models where weight memory dominates

### ViTok Specifics
- **Encoder**: 463M params (5B model) - moderate benefit possible
- **Decoder**: 4.5B params - larger potential benefit
- **Attention**: Memory-bound, limited INT8 benefit
- **FFN (SwiGLU)**: Compute-bound, good INT8 benefit

## Test Plan

### Metrics to Measure
1. **Throughput**: images/second
2. **Latency**: ms per image (after warmup)
3. **Peak Memory**: GB allocated
4. **Quality**: SSIM between quantized and bf16 outputs

### Test Matrix
| Config | compile | quantization | Expected |
|--------|---------|--------------|----------|
| Baseline | No | bf16 | Reference |
| Compile only | Yes (reduce-overhead) | bf16 | 1.2-1.5x faster |
| INT8 only | No | INT8 | ~1.0-1.1x, less memory |
| Both | Yes | INT8 | Test if additive |

### Test Command
```bash
modal run tests/gpu/test_compile_int8_benchmark.py --model 350M-64 --gpu H100
```

## Recommendations

1. **Keep torch.compile** for single-GPU eval with fixed resolution
2. **Add `mode='reduce-overhead'`** to reduce compilation time
3. **Test INT8 on H100** - may show better results than A100
4. **If no speedup**: Remove INT8 fallback, simplify code
5. **Add proper timing** with `torch.cuda.synchronize()` for accurate measurements

## Sources
- PyTorch torch.compile docs
- TorchAO Quantization Overview
- Codex/Claude research analysis (Jan 2026)
