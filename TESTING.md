# Testing Guide

This document describes how to run tests for vitok-release.

## Local Tests (CPU)

Run tests locally using pytest:

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run specific test files
pytest tests/test_dit.py -v          # DiT model tests
pytest tests/test_unipc.py -v        # UniPC scheduler tests
pytest tests/test_ae_compatibility.py -v  # AE encode/decode tests
pytest tests/test_pp.py -v           # Preprocessing pipeline tests
```

### Test Coverage

| Test File | What it tests |
|-----------|---------------|
| `test_dit.py` | DiT instantiation, forward pass, CFG, special tokens, weight compatibility |
| `test_unipc.py` | UniPC scheduler timesteps, step function, noise, determinism |
| `test_ae_compatibility.py` | AE encode/decode, reconstruction, weight compatibility with vitokv2 |
| `test_pp.py` | Preprocessing DSL, patchify, NaFlex roundtrip, WebDataset integration |

## GPU Tests on Modal

For GPU testing, we use [Modal](https://modal.com/) which provides serverless GPUs.

### Setup

1. Install Modal:
   ```bash
   pip install modal
   ```

2. Authenticate:
   ```bash
   modal token new
   ```

### Running GPU Tests

```bash
# Quick tests - just forward passes (~1 min)
modal run modal/test_all.py --quick

# Full test suite on GPU (~3 min)
modal run modal/test_all.py

# Include vitokv2 compatibility tests (requires ../vitokv2)
modal run modal/test_all.py --compat

# Run individual test files
modal run modal/test_ae_gpu.py
modal run modal/test_dit_gpu.py
```

### What GPU Tests Verify

1. **Forward Pass** - Models run correctly on GPU
2. **torch.compile** - Models work with torch.compile
3. **Weight Compatibility** - Weights transfer correctly from vitokv2
4. **UniPC Sampling** - Full denoising loop works on GPU

### Modal Test Structure

```
modal/
├── env.py           # Modal image config with dependencies
├── test_all.py      # Combined test runner
├── test_ae_gpu.py   # AE-specific GPU tests
└── test_dit_gpu.py  # DiT-specific GPU tests
```

## Compatibility Testing

To test weight compatibility with vitokv2, ensure vitokv2 is at `../vitokv2`:

```
parent/
├── vitok-release/   # This repo
└── vitokv2/         # Original vitokv2 repo
```

Then run:
```bash
# CPU
pytest tests/test_ae_compatibility.py::test_ae_weight_compatibility -v
pytest tests/test_dit.py::test_dit_weight_compatibility -v

# GPU
modal run modal/test_all.py --compat
```
