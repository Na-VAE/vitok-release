# Plan: Fix flex_attention for Sliding Window Attention

## Problem Summary

flex_attention with BlockMask OOMs or crashes at large sequence lengths (16K+ tokens) due to:
1. **PyTorch 2.9.1 bug**: `torch._safe_softmax` not registered for Dynamo tracing
2. **PyTorch 2.8.0**: Works in minimal reproduction but fails in full model context
3. **Compilation conflicts**: flex_attention's internal torch.compile conflicts with outer model compilation

## What Works (Verified)

Minimal test on PyTorch 2.8.0 with `torch.compile(flex_attention, mode="max-autotune")`:
- 1,024 tokens (32x32): 0.07 GB peak memory, SUCCESS
- 16,384 tokens (128x128): 0.29 GB peak memory, SUCCESS

## What Fails

Full ViTok model with same setup:
- Crashes with `_safe_softmax` Dynamo error
- Falls back to dense `math_attention` → OOM

## Root Cause Hypothesis

1. **float8 dispatch mode**: torchao's float8 quantization may interfere with flex_attention's Triton kernel dispatch
2. **Nested compilation**: Model-level torch.compile + flex_attention's internal compile = conflict
3. **dtype issues**: flex_attention may require explicit bfloat16 inputs, not float8

## Proposed Solution

### Option A: Isolate flex_attention from model compilation

```python
# In Attention.forward():
if sliding_window is not None:
    with torch.compiler.disable():  # Disable outer compile for this block
        # Cast to bfloat16, call compiled flex_attention, cast back
        attn = _flex_attn(q.bfloat16(), k.bfloat16(), v.bfloat16(), block_mask=mask)
        attn = attn.to(orig_dtype)
```

### Option B: Use separate compilation for SWA path

```python
# Pre-compile a wrapper function that handles the dtype conversion
@torch.compile(mode="max-autotune")
def _swa_attention(q, k, v, block_mask):
    return flex_attention(q.bfloat16(), k.bfloat16(), v.bfloat16(), block_mask=block_mask)
```

### Option C: Disable float8 for attention layers only

Keep float8 for linear layers but use bfloat16 for attention computation.

## Implementation Steps

1. **Create minimal reproduction** in full model context (not just attention module)
2. **Test each option** with TORCH_LOGS="graph_breaks,recompiles" to understand compilation behavior
3. **Verify float8 interaction** by testing with/without float8 quantization
4. **Profile memory** to ensure Triton kernel is actually being used (not dense fallback)

## Testing Matrix

| PyTorch | float8 | compile | SWA | Expected |
|---------|--------|---------|-----|----------|
| 2.8.0   | off    | off     | off | PASS (baseline) |
| 2.8.0   | off    | off     | on  | Test |
| 2.8.0   | off    | on      | on  | Test |
| 2.8.0   | on     | off     | on  | Test |
| 2.8.0   | on     | on      | on  | Test |

## Files to Modify

- `vitok/models/modules/attention.py` - Add flex_attention path with proper isolation
- `scripts/eval_vae.py` - May need warmup adjustments
- `scripts/modal/modal_config.py` - Pin PyTorch version

## Success Criteria

1. SWA works at 2048px (16K tokens) without OOM
2. Memory usage significantly lower than full attention
3. No graph breaks or compilation warnings
4. Works with and without float8 quantization
