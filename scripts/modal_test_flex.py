#!/usr/bin/env python
"""Modal runner for flex_attention test."""

import modal

app = modal.App("test-flex-attn")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch==2.8.0", "torchvision==0.23.0")
)


@app.function(image=image, gpu="H100", timeout=600)
def test_flex(seq_len: int = 1024, compile_mode: str = None):
    import torch
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    def make_2d_swa_mask(window: int, grid_size: int):
        def mask_mod(b, h, q_idx, kv_idx):
            q_row = q_idx // grid_size
            q_col = q_idx % grid_size
            kv_row = kv_idx // grid_size
            kv_col = kv_idx % grid_size
            return ((q_row - kv_row).abs() <= window) & ((q_col - kv_col).abs() <= window)
        return mask_mod

    device = torch.device("cuda")
    batch_size, num_heads, head_dim = 1, 16, 64
    grid_size = int(seq_len ** 0.5)
    window = 8

    print(f"\nPyTorch: {torch.__version__}")
    print(f"Testing: seq_len={seq_len} ({grid_size}x{grid_size}), compile_mode={compile_mode}")

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    print("Creating BlockMask...")
    mask_mod = make_2d_swa_mask(window, grid_size)
    block_mask = create_block_mask(mask_mod, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device)
    print(f"BlockMask: {block_mask}")

    if compile_mode:
        print(f"Compiling with mode={compile_mode}...")
        flex_fn = torch.compile(flex_attention, mode=compile_mode)
    else:
        flex_fn = flex_attention

    print("Running flex_attention...")
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        out = flex_fn(q, k, v, block_mask=block_mask)

    torch.cuda.synchronize()
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    print(f"Output: {out.shape}, Peak memory: {peak_mem:.2f} GB")
    return {"status": "success", "peak_mem_gb": peak_mem}


@app.local_entrypoint()
def main(seq_len: int = 1024, compile_mode: str = None):
    result = test_flex.remote(seq_len, compile_mode)
    print(f"\nResult: {result}")
