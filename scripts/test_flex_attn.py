#!/usr/bin/env python
"""Minimal reproduction of flex_attention + BlockMask issue at large sequence lengths."""

import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def make_2d_swa_mask(window: int, grid_size: int):
    """2D sliding window mask_mod."""
    def mask_mod(b, h, q_idx, kv_idx):
        q_row = q_idx // grid_size
        q_col = q_idx % grid_size
        kv_row = kv_idx // grid_size
        kv_col = kv_idx % grid_size
        return ((q_row - kv_row).abs() <= window) & ((q_col - kv_col).abs() <= window)
    return mask_mod


def test_flex_attention(seq_len: int, compile_mode: str = None):
    """Test flex_attention with 2D SWA BlockMask."""
    device = torch.device("cuda")

    # Setup
    batch_size = 1
    num_heads = 16
    head_dim = 64
    grid_size = int(seq_len ** 0.5)
    window = 8

    print(f"\n{'='*60}")
    print(f"Testing: seq_len={seq_len} ({grid_size}x{grid_size}), compile_mode={compile_mode}")
    print(f"{'='*60}")

    # Create inputs
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Create block mask
    print("Creating BlockMask...")
    mask_mod = make_2d_swa_mask(window, grid_size)
    block_mask = create_block_mask(mask_mod, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device)
    print(f"BlockMask created: {block_mask}")

    # Setup flex_attention
    if compile_mode:
        print(f"Compiling flex_attention with mode={compile_mode}...")
        flex_fn = torch.compile(flex_attention, mode=compile_mode)
    else:
        print("Using eager flex_attention...")
        flex_fn = flex_attention

    # Run attention
    print("Running flex_attention...")
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        out = flex_fn(q, k, v, block_mask=block_mask)

    torch.cuda.synchronize()
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    print(f"Output shape: {out.shape}")
    print(f"Peak memory: {peak_mem:.2f} GB")
    print("SUCCESS!")

    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length (must be perfect square)")
    parser.add_argument("--compile", type=str, default=None, choices=[None, "default", "reduce-overhead", "max-autotune"])
    args = parser.parse_args()

    test_flex_attention(args.seq_len, args.compile)
