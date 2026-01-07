"""Visualize reconstructions from a trained VAE checkpoint.

Run with:
    modal run tests/utils/test_recon_viz.py
"""

import modal

app = modal.App("vitok-recon-viz")

VITOK_PATH = "/root/vitok-release"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.4.0",
        "torchvision>=0.19.0",
        "safetensors>=0.4.0",
        "numpy>=1.24.0",
        "pillow>=10.0.0",
        "requests>=2.31.0",
        "packaging>=21.0",
        "webdataset>=0.2.86",
        "huggingface_hub>=0.23.0,<1.0",
    )
    .add_local_dir("vitok", remote_path=f"{VITOK_PATH}/vitok")
)

downloads_volume = modal.Volume.from_name("vitok-downloads")


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/downloads": downloads_volume},
    timeout=300,
)
def visualize_recon():
    """Download an image and show 256p and 512p reconstructions."""
    import sys
    sys.path.insert(0, VITOK_PATH)

    import io
    import requests
    import torch
    import numpy as np
    from PIL import Image
    from safetensors.torch import load_file

    from vitok.ae import AEConfig, create_ae
    from vitok.naflex_io import preprocess_images, postprocess_images

    # Download a test image
    print("Downloading test image...")
    urls = [
        "https://raw.githubusercontent.com/pytorch/vision/main/gallery/assets/astronaut.jpg",
        "https://picsum.photos/512/512",
    ]

    original_img = None
    for url in urls:
        try:
            response = requests.get(url, timeout=10, allow_redirects=True)
            if response.status_code == 200:
                original_img = Image.open(io.BytesIO(response.content)).convert("RGB")
                print(f"Downloaded from: {url}")
                break
        except Exception as e:
            print(f"Failed to download from {url}: {e}")

    if original_img is None:
        print("Creating synthetic test image...")
        x = torch.linspace(0, 1, 512).unsqueeze(0).expand(512, -1)
        y = torch.linspace(0, 1, 512).unsqueeze(1).expand(-1, 512)
        img_tensor = torch.stack([x, y, (x + y) / 2], dim=0)
        img_np = (img_tensor.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        original_img = Image.fromarray(img_np)

    print(f"Original image size: {original_img.size}")

    # Load Ld4-L checkpoint from downloads (decoder-only, but let's check)
    print("\nLoading checkpoint...")
    ckpt_path = "/downloads/extracted/dir_for_transfer/Ld4-L_x16x64_test/model.safetensors"
    checkpoint_state = load_file(ckpt_path)
    print(f"Loaded {len(checkpoint_state)} keys")

    # Remap keys from vitokv2 naming to vitok-release naming
    # vitokv2: encoder.X, decoder.X
    # vitok-release: encoder_blocks.X, decoder_blocks.X
    remapped_state = {}
    for k, v in checkpoint_state.items():
        new_key = k.replace("_orig_mod.", "")
        # Remap encoder.X -> encoder_blocks.X
        if new_key.startswith("encoder.") and len(new_key) > 8 and new_key[8].isdigit():
            new_key = "encoder_blocks." + new_key[8:]
        # Remap decoder.X -> decoder_blocks.X
        if new_key.startswith("decoder.") and len(new_key) > 8 and new_key[8].isdigit():
            new_key = "decoder_blocks." + new_key[8:]
        remapped_state[new_key] = v

    print(f"Sample keys after remap: {list(remapped_state.keys())[:3]}")

    # Check if this is encoder+decoder or decoder-only
    has_encoder = any(k.startswith("encoder_blocks.") for k in remapped_state)
    has_decoder = any(k.startswith("decoder_blocks.") for k in remapped_state)
    print(f"Has encoder: {has_encoder}, Has decoder: {has_decoder}")

    if not has_encoder:
        print("ERROR: Checkpoint is decoder-only, cannot do encode->decode")
        return None, []

    # Infer depths
    encoder_depth = max(int(k.split(".")[1]) for k in remapped_state if k.startswith("encoder_blocks.")) + 1
    decoder_depth = max(int(k.split(".")[1]) for k in remapped_state if k.startswith("decoder_blocks.")) + 1
    print(f"Encoder depth: {encoder_depth}, Decoder depth: {decoder_depth}")

    # Get latent channels
    latent_channels = remapped_state['to_code.weight'].shape[0]
    print(f"Latent channels: {latent_channels}")

    # Build variant - Ld4-L means encoder=L with d4, decoder=L (default 24 layers)
    variant = f"Ld{encoder_depth}-Ld{decoder_depth}/1x16x{latent_channels}"
    print(f"\nCreating model with variant: {variant}")
    config = AEConfig(variant=variant, encoder=True, decoder=True)
    model = create_ae(config).cuda().eval()
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Load weights
    missing, unexpected = model.load_state_dict(remapped_state, strict=False)
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")
    if missing:
        print(f"  Missing: {missing[:5]}...")
    if unexpected:
        print(f"  Unexpected: {unexpected[:5]}...")

    # Process images at different resolutions
    results = []
    for size in [256, 512]:
        print(f"\n{'='*40}")
        print(f"Processing at {size}x{size}")
        print(f"{'='*40}")

        img = original_img.resize((size, size), Image.LANCZOS)

        pp_str = f"to_tensor|normalize(minus_one_to_one)|patchify({size}, 16, {(size//16)**2})"
        print(f"Preprocessing: {pp_str}")

        patches = preprocess_images(img, pp=pp_str, device="cuda")
        print(f"Patches shape: {patches['patches'].shape}")

        with torch.no_grad():
            encoded = model.encode(patches)
            print(f"Latent z shape: {encoded['z'].shape}")
            decoded = model.decode(encoded)

        recon_images = postprocess_images(
            decoded,
            output_format="zero_to_one",
            current_format="minus_one_to_one",
            unpack=True,
            patch=16,
        )

        recon_np = recon_images[0].permute(1, 2, 0).cpu().numpy()
        recon_np = np.clip(recon_np, 0, 1)
        recon_img = Image.fromarray((recon_np * 255).astype(np.uint8))
        print(f"Reconstruction size: {recon_img.size}")

        results.append({
            'size': size,
            'input': img,
            'recon': recon_img,
            'latent_shape': list(encoded['z'].shape),
        })

    # Create comparison: stack vertically
    # Row 1: 256 input | 256 recon
    # Row 2: 512 input | 512 recon
    print("\nCreating comparison image...")

    gap = 10
    row1_w = 256 + gap + 256
    row2_w = 512 + gap + 512
    total_w = max(row1_w, row2_w)
    total_h = 256 + gap + 512

    comparison = Image.new('RGB', (total_w, total_h), (40, 40, 40))

    r256 = results[0]
    r512 = results[1]

    # Row 1: 256 images, centered horizontally
    x_offset = (total_w - row1_w) // 2
    comparison.paste(r256['input'], (x_offset, 0))
    comparison.paste(r256['recon'], (x_offset + 256 + gap, 0))

    # Row 2: 512 images
    x_offset = (total_w - row2_w) // 2
    comparison.paste(r512['input'], (x_offset, 256 + gap))
    comparison.paste(r512['recon'], (x_offset + 512 + gap, 256 + gap))

    buf = io.BytesIO()
    comparison.save(buf, format='PNG')
    buf.seek(0)

    print("\nDone!")
    return buf.getvalue(), results


@app.local_entrypoint()
def main():
    """Run visualization and save result."""
    print("Running reconstruction visualization...")

    img_bytes, results = visualize_recon.remote()

    if img_bytes is None:
        print("Failed to generate comparison")
        return

    output_path = "recon_comparison.png"
    with open(output_path, "wb") as f:
        f.write(img_bytes)

    print(f"\nSaved comparison to: {output_path}")
    print("\nResults:")
    for r in results:
        print(f"  {r['size']}x{r['size']}: latent shape {r['latent_shape']}")

    print(f"\nOpen with: open {output_path}")
