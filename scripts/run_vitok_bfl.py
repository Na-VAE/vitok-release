"""Run ViTok L-64 on BFL images.

Usage: modal run scripts/run_vitok_bfl.py
"""
import modal

app = modal.App("vitok-bfl-eval")

# Simple image with torch and required packages
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.3",
        "torchvision",
        "safetensors",
        "huggingface_hub",
        "Pillow",
        "numpy",
        "einops",
        "webdataset",
    )
)

# Reference the existing volumes
data_volume = modal.Volume.from_name("vitok-data")
code_volume = modal.Volume.from_name("vitok-code")


@app.function(
    image=image,
    gpu="T4",
    timeout=600,
    volumes={"/data": data_volume, "/code": code_volume},
)
def run_vitok_reconstruction():
    """Run ViTok L-64 on BFL original images and save reconstructions."""
    import sys
    import os

    # Add vitok to path (from synced code volume)
    sys.path.insert(0, "/code/vitok-release")

    import torch
    import numpy as np
    from PIL import Image
    from pathlib import Path
    from safetensors.torch import load_file

    # Import vitok
    from vitok import AE, decode_variant
    from vitok.pretrained import download_pretrained, get_pretrained_info
    from vitok.pp.io import preprocess, postprocess

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.get_device_name(0)}")

    # Download L-64 model
    print("\nDownloading L-64 model...")
    _, _, variant = get_pretrained_info("L-64")
    result = download_pretrained("L-64")
    weights_paths = result if isinstance(result, list) else [result]
    config = decode_variant(variant)
    patch_size = config["spatial_stride"]
    print(f"Variant: {variant}, patch_size: {patch_size}")

    # Load weights (merge if split encoder/decoder files)
    print(f"Loading model weights...")
    weights = {}
    for path in weights_paths:
        weights.update(load_file(path))
    dtype = torch.bfloat16

    encoder = AE(**config, decoder=False).to(device="cuda", dtype=dtype)
    encoder.load_state_dict(weights, strict=False)
    encoder.eval()

    decoder = AE(**config, encoder=False).to(device="cuda", dtype=dtype)
    decoder.load_state_dict(weights, strict=False)
    decoder.eval()

    print("Compiling encoder/decoder...")
    encoder = torch.compile(encoder, fullgraph=True)
    decoder = torch.compile(decoder, fullgraph=True)

    # Warmup with proper preprocessing
    print("Warming up compilation...")
    dummy_img = Image.new("RGB", (360, 360), (128, 128, 128))
    pp_str = f"to_tensor|normalize(minus_one_to_one)|patchify({patch_size}, 1024)"
    dummy_batch = preprocess(dummy_img, pp=pp_str, device="cuda")
    dummy_batch["patches"] = dummy_batch["patches"].to(dtype)
    with torch.no_grad():
        _ = decoder.decode(encoder.encode(dummy_batch))
    print("Compilation done.")

    # Process BFL images
    input_dir = Path("/data/bfl_originals/originals_full")
    output_dir = Path("/data/bfl_vitok_recon")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing images from {input_dir}...")

    for img_path in sorted(input_dir.glob("*.png")):
        print(f"\nProcessing {img_path.name}...")

        # Load image as PIL
        img = Image.open(img_path).convert("RGB")
        print(f"  Original size: {img.size}")

        # Preprocess using proper vitok preprocessing
        batch = preprocess(img, pp=pp_str, device="cuda")
        batch["patches"] = batch["patches"].to(dtype)

        # Encode and decode
        with torch.no_grad():
            encoded = encoder.encode(batch)
            output = decoder.decode(encoded)

        # Copy metadata needed for unpacking
        output["orig_height"] = batch["orig_height"]
        output["orig_width"] = batch["orig_width"]

        # Postprocess to get image back
        # Using postprocess with do_unpack=True returns list of tensors cropped to original size
        result = postprocess(
            output,
            output_format="0_255",
            current_format="minus_one_to_one",
            do_unpack=True,
            patch=patch_size,
        )

        # result is a list of tensors, get first one
        recon_tensor = result[0]  # [C, H, W] uint8
        recon_np = recon_tensor.permute(1, 2, 0).cpu().numpy()  # [H, W, C]
        recon_pil = Image.fromarray(recon_np)

        # Save
        out_path = output_dir / img_path.name
        recon_pil.save(out_path)
        print(f"  Saved to {out_path} (size: {recon_pil.size})")

    # Commit changes
    data_volume.commit()

    print("\n\nDone! Reconstructions saved to /data/bfl_vitok_recon/")
    return {"status": "success", "output_dir": str(output_dir)}


@app.local_entrypoint()
def main():
    result = run_vitok_reconstruction.remote()
    print(f"\nResult: {result}")
