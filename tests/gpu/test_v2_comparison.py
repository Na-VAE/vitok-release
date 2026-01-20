"""Compare vitokv2 vs vitok-release forward pass outputs.

This test loads the same pretrained weights into both repos and compares
encoder/decoder outputs to diagnose numerical differences.

Run with: modal run tests/gpu/test_v2_comparison.py
"""

import modal

# Paths inside Modal container
VITOK_PATH = "/root/vitok-release"
V2_PATH = "/root/vitokv2"

# Image with both vitok-release and vitokv2 for compatibility testing
compat_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.4.0",
        "torchvision>=0.19.0",
        "safetensors>=0.4.0",
        "numpy>=1.24.0",
        "pillow>=10.0.0",
        "webdataset>=0.2.86",
        "huggingface_hub>=0.23.0,<1.0",
        "pytest>=7.0.0",
        "requests",
        "diffusers>=0.31.0",
        "ml_collections",
        "transformers",  # Required by vitokv2
    )
    .add_local_dir("vitok", remote_path=f"{VITOK_PATH}/vitok")
    .add_local_dir("../vitokv2/vitok", remote_path=f"{V2_PATH}/vitok")
)

app = modal.App("vitok-v2-comparison")


def remap_release_to_v2(state_dict):
    """Remap vitok-release keys to vitokv2 naming."""
    new_dict = {}
    for key, value in state_dict.items():
        new_key = key
        # Strip _orig_mod. prefix if present
        if new_key.startswith("_orig_mod."):
            new_key = new_key[len("_orig_mod."):]
        # Remap encoder_blocks.N -> encoder.N
        if new_key.startswith("encoder_blocks."):
            new_key = new_key.replace("encoder_blocks.", "encoder.", 1)
        # Remap decoder_blocks.N -> decoder.N
        if new_key.startswith("decoder_blocks."):
            new_key = new_key.replace("decoder_blocks.", "decoder.", 1)
        new_dict[new_key] = value
    return new_dict


def remap_v2_to_release(state_dict):
    """Remap vitokv2 keys to vitok-release naming."""
    new_dict = {}
    for key, value in state_dict.items():
        new_key = key
        # Strip _orig_mod. prefix if present
        if new_key.startswith("_orig_mod."):
            new_key = new_key[len("_orig_mod."):]
        # Remap encoder.N -> encoder_blocks.N (only if followed by digit)
        if new_key.startswith("encoder.") and len(new_key) > 8 and new_key[8].isdigit():
            new_key = "encoder_blocks." + new_key[8:]
        # Remap decoder.N -> decoder_blocks.N
        if new_key.startswith("decoder.") and len(new_key) > 8 and new_key[8].isdigit():
            new_key = "decoder_blocks." + new_key[8:]
        new_dict[new_key] = value
    return new_dict


def patches_to_image(patches, row_idx, col_idx, patch_mask, orig_height, orig_width, patch_size=16):
    """Convert patches back to image tensor."""
    import torch
    B, N, C = patches.shape
    # Assuming square grid for simplicity
    H = orig_height[0].item()
    W = orig_width[0].item()
    grid_h = H // patch_size
    grid_w = W // patch_size

    # Create empty image
    img = torch.zeros(B, 3, H, W, device=patches.device, dtype=patches.dtype)

    for b in range(B):
        for i in range(N):
            if patch_mask[b, i]:
                r = row_idx[b, i].item()
                c = col_idx[b, i].item()
                patch = patches[b, i].reshape(3, patch_size, patch_size)
                y_start = r * patch_size
                x_start = c * patch_size
                if y_start + patch_size <= H and x_start + patch_size <= W:
                    img[b, :, y_start:y_start+patch_size, x_start:x_start+patch_size] = patch

    return img


@app.function(
    image=compat_image,
    gpu="A10G",
    timeout=900,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def compare_forward_pass(
    model_name: str = "philippehansen/ViTok-L-16x64",
    variant: str = "Ld4-Ld24/1x16x64",
    save_images: bool = True,
):
    """Compare forward pass between vitokv2 and vitok-release."""
    import os
    import sys

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    import torch
    import numpy as np
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download

    print("=" * 70)
    print(f"Comparing {model_name} ({variant})")
    print("=" * 70)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.get_device_name(0)}")

    # Download weights
    print(f"\nDownloading weights from {model_name}...")
    weights_path = hf_hub_download(repo_id=model_name, filename="model.safetensors")
    weights = load_file(weights_path)
    print(f"Loaded {len(weights)} weight tensors")

    # Sample keys to understand format
    sample_keys = sorted(weights.keys())[:10]
    print(f"Sample keys: {sample_keys}")

    # Determine if weights are in v2 or release format
    has_encoder_blocks = any(k.startswith("encoder_blocks.") for k in weights.keys())
    has_encoder = any(k.startswith("encoder.") and k.split(".")[1].isdigit() for k in weights.keys())
    print(f"Has encoder_blocks.*: {has_encoder_blocks}")
    print(f"Has encoder.N.*: {has_encoder}")

    # === Load vitokv2 model ===
    print("\n" + "-" * 70)
    print("Loading vitokv2 model...")
    print("-" * 70)
    sys.path.insert(0, V2_PATH)
    from vitok.models.ae import AE as V2_AE
    from vitok.configs.vae.base import decode_variant as v2_decode

    v2_params = v2_decode(variant)
    # Match the pretrained model settings
    v2_model = V2_AE(
        **v2_params,
        variational=False,
        drop_path_rate=0.0,
        use_naflex_posemb=False,
        encoder_output_fn='layernorm',
        decoder_output_fn='none',
    ).cuda().eval()
    print(f"V2 model: {sum(p.numel() for p in v2_model.parameters()):,} params")

    # Load weights into v2
    v2_weights = remap_release_to_v2(weights) if has_encoder_blocks else weights
    v2_result = v2_model.load_state_dict(v2_weights, strict=False)
    print(f"V2 missing keys ({len(v2_result.missing_keys)}): {v2_result.missing_keys[:5]}...")
    print(f"V2 unexpected keys ({len(v2_result.unexpected_keys)}): {v2_result.unexpected_keys[:5]}...")

    # === Clear and load vitok-release model ===
    print("\n" + "-" * 70)
    print("Loading vitok-release model...")
    print("-" * 70)
    for mod in [k for k in list(sys.modules.keys()) if k.startswith('vitok')]:
        del sys.modules[mod]
    sys.path.remove(V2_PATH)
    sys.path.insert(0, VITOK_PATH)

    from vitok.models.ae import AE as Release_AE
    from vitok.models.ae import decode_variant as release_decode

    release_params = release_decode(variant)
    release_model = Release_AE(
        **release_params,
        variational=False,
        drop_path_rate=0.0,
    ).cuda().eval()
    print(f"Release model: {sum(p.numel() for p in release_model.parameters()):,} params")

    # Load weights into release
    release_weights = remap_v2_to_release(weights) if has_encoder else weights
    release_result = release_model.load_state_dict(release_weights, strict=False)
    print(f"Release missing keys ({len(release_result.missing_keys)}): {release_result.missing_keys[:5]}...")
    print(f"Release unexpected keys ({len(release_result.unexpected_keys)}): {release_result.unexpected_keys[:5]}...")

    # === Compare weights after loading ===
    print("\n" + "-" * 70)
    print("Comparing loaded weights...")
    print("-" * 70)

    v2_state = v2_model.state_dict()
    release_state = release_model.state_dict()

    # Build mapping from v2 keys to release keys
    v2_to_release = {}
    for v2_key in v2_state.keys():
        release_key = v2_key
        if release_key.startswith("encoder.") and len(release_key) > 8 and release_key[8].isdigit():
            release_key = "encoder_blocks." + release_key[8:]
        if release_key.startswith("decoder.") and len(release_key) > 8 and release_key[8].isdigit():
            release_key = "decoder_blocks." + release_key[8:]
        if release_key in release_state:
            v2_to_release[v2_key] = release_key

    weight_diffs = {}
    for v2_key, release_key in v2_to_release.items():
        v2_w = v2_state[v2_key]
        rel_w = release_state[release_key]
        if v2_w.shape == rel_w.shape:
            diff = (v2_w - rel_w).abs().max().item()
            weight_diffs[v2_key] = diff
            if diff > 1e-6:
                print(f"  Weight diff {v2_key}: {diff:.2e}")

    max_weight_diff = max(weight_diffs.values()) if weight_diffs else 0.0
    print(f"Max weight difference: {max_weight_diff:.2e}")
    if max_weight_diff == 0.0:
        print("All matched weights are IDENTICAL")

    # === Create test input ===
    print("\n" + "-" * 70)
    print("Running forward pass comparison...")
    print("-" * 70)

    torch.manual_seed(42)
    batch_size, patch_size = 1, 16

    # Try to load a real image
    try:
        from PIL import Image
        import requests
        from io import BytesIO

        # Try multiple image sources
        image_urls = [
            "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/cat.png",
        ]

        img = None
        for url in image_urls:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content)).convert('RGB')
                    break
            except Exception:
                continue

        if img is None:
            # Generate a synthetic but visually interesting test image
            print("  Using synthetic test image (gradient + shapes)")
            img = Image.new('RGB', (256, 256))
            pixels = img.load()

            # Create gradient background with shapes
            for y_pix in range(256):
                for x_pix in range(256):
                    # Gradient background
                    r = int(x_pix)
                    g = int(y_pix)
                    b = int((x_pix + y_pix) / 2)

                    # Add some circles/shapes
                    dist_center = ((x_pix - 128)**2 + (y_pix - 128)**2)**0.5
                    if dist_center < 50:
                        r, g, b = 255, 200, 100  # Yellow circle
                    elif 60 < dist_center < 70:
                        r, g, b = 50, 100, 200  # Blue ring

                    # Add checkerboard pattern in corner
                    if x_pix < 64 and y_pix < 64:
                        if ((x_pix // 8) + (y_pix // 8)) % 2 == 0:
                            r, g, b = 200, 200, 200
                        else:
                            r, g, b = 50, 50, 50

                    pixels[x_pix, y_pix] = (r, g, b)
        else:
            # Resize to 256x256 for clean patches
            img = img.resize((256, 256), Image.LANCZOS)
            print(f"  Loaded real image: {img.size}")

        # Convert to tensor and normalize to [-1, 1]
        img_tensor = torch.tensor(np.array(img)).float() / 255.0
        img_tensor = img_tensor * 2 - 1  # [0,1] -> [-1,1]
        img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW

        # Patchify
        H, W = 256, 256
        grid_size = H // patch_size
        seq_len = grid_size * grid_size

        # Extract patches
        patches_list = []
        for r in range(grid_size):
            for c in range(grid_size):
                patch = img_tensor[:, r*patch_size:(r+1)*patch_size, c*patch_size:(c+1)*patch_size]
                patches_list.append(patch.flatten())
        patches = torch.stack(patches_list).unsqueeze(0).cuda()  # [1, N, C]

        y, x = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size), indexing='ij')
        patch_mask = torch.ones(batch_size, seq_len, dtype=torch.bool).cuda()
        row_idx = y.flatten().unsqueeze(0).expand(batch_size, -1).cuda()
        col_idx = x.flatten().unsqueeze(0).expand(batch_size, -1).cuda()
        orig_height = torch.full((batch_size,), H).cuda()
        orig_width = torch.full((batch_size,), W).cuda()

        using_real_image = True
    except Exception as e:
        print(f"  Failed to load real image: {e}, using random noise")
        using_real_image = False

        seq_len = 256
        C = patch_size * patch_size * 3
        grid_size = int(np.sqrt(seq_len))
        y, x = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size), indexing='ij')

        patches = torch.randn(batch_size, seq_len, C).cuda()
        patch_mask = torch.ones(batch_size, seq_len, dtype=torch.bool).cuda()
        row_idx = y.flatten().unsqueeze(0).expand(batch_size, -1).cuda()
        col_idx = x.flatten().unsqueeze(0).expand(batch_size, -1).cuda()
        orig_height = torch.full((batch_size,), grid_size * patch_size).cuda()
        orig_width = torch.full((batch_size,), grid_size * patch_size).cuda()

    # vitokv2 uses yidx/xidx/ptype
    v2_input = {
        'patches': patches.clone(),
        'ptype': patch_mask.clone(),
        'yidx': row_idx.clone(),
        'xidx': col_idx.clone(),
        'original_height': orig_height.clone(),
        'original_width': orig_width.clone(),
    }

    # vitok-release uses row_idx/col_idx/patch_mask
    release_input = {
        'patches': patches.clone(),
        'patch_mask': patch_mask.clone(),
        'row_idx': row_idx.clone(),
        'col_idx': col_idx.clone(),
        'orig_height': orig_height.clone(),
        'orig_width': orig_width.clone(),
        'attention_mask': None,
    }

    # === Run forward passes ===
    # First reload v2 module since we cleared it
    for mod in [k for k in list(sys.modules.keys()) if k.startswith('vitok')]:
        del sys.modules[mod]
    sys.path.remove(VITOK_PATH)
    sys.path.insert(0, V2_PATH)

    # Recreate v2 model with same weights
    from vitok.models.ae import AE as V2_AE_2
    from vitok.configs.vae.base import decode_variant as v2_decode_2

    v2_params_2 = v2_decode_2(variant)
    v2_model_2 = V2_AE_2(
        **v2_params_2,
        variational=False,
        drop_path_rate=0.0,
        use_naflex_posemb=False,
        encoder_output_fn='layernorm',
        decoder_output_fn='none',
    ).cuda().eval()
    v2_weights_2 = remap_release_to_v2(weights) if has_encoder_blocks else weights
    v2_model_2.load_state_dict(v2_weights_2, strict=False)

    with torch.no_grad():
        v2_out = v2_model_2(v2_input)

    # Clear and reload release
    for mod in [k for k in list(sys.modules.keys()) if k.startswith('vitok')]:
        del sys.modules[mod]
    sys.path.remove(V2_PATH)
    sys.path.insert(0, VITOK_PATH)

    with torch.no_grad():
        release_out = release_model(release_input)

    # === Compare outputs ===
    print("\nOutput comparison:")

    # Check for NaN/Inf
    v2_nan = torch.isnan(v2_out['patches']).any().item()
    v2_inf = torch.isinf(v2_out['patches']).any().item()
    rel_nan = torch.isnan(release_out['patches']).any().item()
    rel_inf = torch.isinf(release_out['patches']).any().item()

    print(f"  V2 output - NaN: {v2_nan}, Inf: {v2_inf}")
    print(f"  Release output - NaN: {rel_nan}, Inf: {rel_inf}")

    # Compare reconstructed patches
    v2_patches = v2_out['patches']
    rel_patches = release_out['patches']

    patch_diff = (v2_patches - rel_patches).abs()
    max_diff = patch_diff.max().item()
    mean_diff = patch_diff.mean().item()

    print(f"\n  Patch output shape: V2={v2_patches.shape}, Release={rel_patches.shape}")
    print(f"  Max absolute difference: {max_diff:.6e}")
    print(f"  Mean absolute difference: {mean_diff:.6e}")

    # Output statistics
    print(f"\n  V2 output stats: min={v2_patches.min():.4f}, max={v2_patches.max():.4f}, mean={v2_patches.mean():.4f}")
    print(f"  Release output stats: min={rel_patches.min():.4f}, max={rel_patches.max():.4f}, mean={rel_patches.mean():.4f}")

    # === Save comparison images ===
    if save_images:
        from PIL import Image
        import io
        import base64

        print("\n" + "-" * 70)
        print("Saving comparison images...")
        print("-" * 70)

        # Convert patches to images
        input_img = patches_to_image(
            patches, row_idx, col_idx, patch_mask, orig_height, orig_width, patch_size=16
        )
        v2_img = patches_to_image(
            v2_patches, row_idx, col_idx, patch_mask, orig_height, orig_width, patch_size=16
        )
        release_img = patches_to_image(
            rel_patches, row_idx, col_idx, patch_mask, orig_height, orig_width, patch_size=16
        )

        # Convert to PIL images (scale from [-1, 1] to [0, 255])
        def tensor_to_pil(t):
            # t is [B, 3, H, W], take first image
            img = t[0].cpu().float()
            img = (img + 1) / 2  # [-1, 1] -> [0, 1]
            img = img.clamp(0, 1)
            img = (img * 255).byte()
            img = img.permute(1, 2, 0).numpy()
            return Image.fromarray(img)

        input_pil = tensor_to_pil(input_img)
        v2_pil = tensor_to_pil(v2_img)
        release_pil = tensor_to_pil(release_img)

        # Create side-by-side comparison
        W, H = input_pil.size
        comparison = Image.new('RGB', (W * 3, H + 30), color=(255, 255, 255))

        # Add images
        comparison.paste(input_pil, (0, 30))
        comparison.paste(v2_pil, (W, 30))
        comparison.paste(release_pil, (W * 2, 30))

        # Add labels (simple text via drawing)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(comparison)
        draw.text((W // 2 - 20, 5), "Input", fill=(0, 0, 0))
        draw.text((W + W // 2 - 10, 5), "V2", fill=(0, 0, 0))
        draw.text((W * 2 + W // 2 - 30, 5), "Release", fill=(0, 0, 0))

        # Save locally
        comparison.save("/tmp/v2_vs_release_comparison.png")
        print(f"  Saved comparison to /tmp/v2_vs_release_comparison.png")

        # Also save individual images
        input_pil.save("/tmp/input.png")
        v2_pil.save("/tmp/v2_output.png")
        release_pil.save("/tmp/release_output.png")
        print(f"  Saved individual images to /tmp/")

        # Return base64 encoded images for display
        def pil_to_base64(img):
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')

        images_b64 = {
            'comparison': pil_to_base64(comparison),
            'input': pil_to_base64(input_pil),
            'v2': pil_to_base64(v2_pil),
            'release': pil_to_base64(release_pil),
        }
    else:
        images_b64 = None

    # === Also compare encode outputs ===
    print("\n" + "-" * 70)
    print("Comparing encode outputs (latent z)...")
    print("-" * 70)

    # Run encode on both
    # Reload v2 again
    for mod in [k for k in list(sys.modules.keys()) if k.startswith('vitok')]:
        del sys.modules[mod]
    sys.path.remove(VITOK_PATH)
    sys.path.insert(0, V2_PATH)

    from vitok.models.ae import AE as V2_AE_3
    from vitok.configs.vae.base import decode_variant as v2_decode_3

    v2_params_3 = v2_decode_3(variant)
    v2_model_3 = V2_AE_3(
        **v2_params_3,
        variational=False,
        drop_path_rate=0.0,
        use_naflex_posemb=False,
        encoder_output_fn='layernorm',
        decoder_output_fn='none',
    ).cuda().eval()
    v2_model_3.load_state_dict(v2_weights_2, strict=False)

    with torch.no_grad():
        v2_enc = v2_model_3.encode(v2_input)

    # Clear and reload release
    for mod in [k for k in list(sys.modules.keys()) if k.startswith('vitok')]:
        del sys.modules[mod]
    sys.path.remove(V2_PATH)
    sys.path.insert(0, VITOK_PATH)

    with torch.no_grad():
        release_enc = release_model.encode(release_input)

    # vitokv2 returns 'posterior' which has .mode() method, vitok-release returns 'z' directly
    print(f"  V2 encode keys: {v2_enc.keys()}")
    print(f"  Release encode keys: {release_enc.keys()}")

    # Handle different return formats
    if 'z' in v2_enc:
        v2_z = v2_enc['z']
    elif 'posterior' in v2_enc:
        # For variational models, posterior has .mode() to get latent
        v2_z = v2_enc['posterior'].mode() if hasattr(v2_enc['posterior'], 'mode') else v2_enc['posterior']
    else:
        v2_z = None

    if 'z' in release_enc:
        rel_z = release_enc['z']
    elif 'posterior' in release_enc:
        rel_z = release_enc['posterior'].mode() if hasattr(release_enc['posterior'], 'mode') else release_enc['posterior']
    else:
        rel_z = None

    if v2_z is not None and rel_z is not None:
        z_diff = (v2_z - rel_z).abs()
        z_max_diff = z_diff.max().item()
        z_mean_diff = z_diff.mean().item()

        print(f"  Latent shape: V2={v2_z.shape}, Release={rel_z.shape}")
        print(f"  Max absolute difference: {z_max_diff:.6e}")
        print(f"  Mean absolute difference: {z_mean_diff:.6e}")
        print(f"  V2 z stats: min={v2_z.min():.4f}, max={v2_z.max():.4f}, mean={v2_z.mean():.4f}")
        print(f"  Release z stats: min={rel_z.min():.4f}, max={rel_z.max():.4f}, mean={rel_z.mean():.4f}")
    else:
        z_max_diff = 0.0
        z_mean_diff = 0.0
        print("  Skipping latent comparison (incompatible encode outputs)")

    # === Summary ===
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    status = "pass"
    if max_diff > 0.1:
        status = "significant_diff"
        print(f"SIGNIFICANT DIFFERENCE in reconstruction: {max_diff:.4f}")
    elif max_diff > 1e-4:
        status = "minor_diff"
        print(f"Minor difference in reconstruction: {max_diff:.6e}")
    else:
        print(f"Outputs match within tolerance: {max_diff:.6e}")

    if z_max_diff > 0.1:
        print(f"SIGNIFICANT DIFFERENCE in latent: {z_max_diff:.4f}")
    elif z_max_diff > 1e-4:
        print(f"Minor difference in latent: {z_max_diff:.6e}")

    return {
        "status": status,
        "max_patch_diff": max_diff,
        "mean_patch_diff": mean_diff,
        "max_latent_diff": z_max_diff,
        "mean_latent_diff": z_mean_diff,
        "v2_missing_keys": len(v2_result.missing_keys),
        "release_missing_keys": len(release_result.missing_keys),
        "images_b64": images_b64,
    }


@app.local_entrypoint()
def main(
    model: str = "philippehansen/ViTok-L-16x64",
    variant: str = "Ld4-Ld24/1x16x64",
    save_images: bool = True,
):
    """Run forward pass comparison.

    Usage:
        modal run tests/gpu/test_v2_comparison.py
        modal run tests/gpu/test_v2_comparison.py --model philippehansen/ViTok-L-16x32 --variant Ld4-Ld24/1x16x32
    """
    import base64
    from pathlib import Path

    print(f"Comparing {model} ({variant})...")
    result = compare_forward_pass.remote(model, variant, save_images)

    # Extract and save images locally
    images_b64 = result.pop("images_b64", None)
    print(f"\nResult: {result}")

    if images_b64:
        output_dir = Path("results/v2_comparison")
        output_dir.mkdir(parents=True, exist_ok=True)

        for name, b64_data in images_b64.items():
            img_path = output_dir / f"{name}.png"
            with open(img_path, "wb") as f:
                f.write(base64.b64decode(b64_data))
            print(f"Saved: {img_path}")

        print(f"\nComparison images saved to {output_dir}/")
