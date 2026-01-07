"""Debug checkpoint structure."""

import modal

app = modal.App("vitok-debug-ckpt")

image = modal.Image.debian_slim(python_version="3.10").pip_install("safetensors", "torch")
checkpoints_volume = modal.Volume.from_name("vitok-checkpoints")


downloads_volume = modal.Volume.from_name("vitok-downloads")


@app.function(image=image, volumes={"/checkpoints": checkpoints_volume, "/downloads": downloads_volume}, timeout=60)
def debug_checkpoint(use_downloads: bool = False):
    from safetensors import safe_open

    if use_downloads:
        ckpt_path = "/downloads/extracted/dir_for_transfer/Ld4-L_x16x64_test/model.safetensors"
    else:
        ckpt_path = "/checkpoints/vae/last/model.safetensors"

    print(f"Checking: {ckpt_path}")

    with safe_open(ckpt_path, framework="pt") as f:
        keys = list(f.keys())

    print(f"Total keys: {len(keys)}")

    # Find encoder blocks
    encoder_keys = sorted([k for k in keys if "encoder" in k])
    decoder_keys = sorted([k for k in keys if "decoder" in k])

    print(f"\nEncoder keys: {len(encoder_keys)}")
    print(f"Sample encoder keys: {encoder_keys[:5]}")
    # Extract block indices
    encoder_blocks = set()
    for k in encoder_keys:
        k = k.replace("_orig_mod.", "")
        if "encoder_blocks." in k:
            parts = k.split(".")
            idx_pos = parts.index("encoder_blocks") + 1
            if idx_pos < len(parts) and parts[idx_pos].isdigit():
                encoder_blocks.add(int(parts[idx_pos]))
    print(f"Encoder block indices: {sorted(encoder_blocks)}")

    print(f"\nDecoder keys: {len(decoder_keys)}")
    print(f"Sample decoder keys: {decoder_keys[:5]}")
    decoder_blocks = set()
    for k in decoder_keys:
        k = k.replace("_orig_mod.", "")
        if "decoder_blocks." in k:
            parts = k.split(".")
            idx_pos = parts.index("decoder_blocks") + 1
            if idx_pos < len(parts) and parts[idx_pos].isdigit():
                decoder_blocks.add(int(parts[idx_pos]))
    print(f"Decoder block indices: {sorted(decoder_blocks)}")

    # Check encoder width from patch_embed
    with safe_open(ckpt_path, framework="pt") as f:
        if "patch_embed.weight" in keys:
            shape = f.get_tensor("patch_embed.weight").shape
            print(f"\npatch_embed.weight shape: {shape} (out_features={shape[0]}, in_features={shape[1]})")
        if "to_code.weight" in keys:
            shape = f.get_tensor("to_code.weight").shape
            print(f"to_code.weight shape: {shape} (latent_channels={shape[0]}, encoder_width={shape[1]})")
        if "decoder_embed.weight" in keys:
            shape = f.get_tensor("decoder_embed.weight").shape
            print(f"decoder_embed.weight shape: {shape} (decoder_width={shape[0]}, latent_channels={shape[1]})")

    # Show all unique key prefixes
    prefixes = set()
    for k in keys:
        k = k.replace("_orig_mod.", "")
        parts = k.split(".")
        if len(parts) >= 2:
            if parts[1].isdigit():
                prefixes.add(f"{parts[0]}.X")
            else:
                prefixes.add(f"{parts[0]}.{parts[1]}")
        else:
            prefixes.add(parts[0])
    print(f"\nKey prefixes: {sorted(prefixes)}")


@app.local_entrypoint()
def main(downloads: bool = False):
    debug_checkpoint.remote(use_downloads=downloads)
