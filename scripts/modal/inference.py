"""Run ViTok AE inference on Modal.

This script downloads pretrained weights and runs encode/decode on an image.
Weights are cached in a Modal volume for fast subsequent runs.

Usage:
    # Run with default model (L-64) and astronaut test image
    modal run scripts/modal/inference.py

    # Specify model
    modal run scripts/modal/inference.py --model T-64

    # Use custom input image
    modal run scripts/modal/inference.py --model L-64 --image path/to/image.jpg

    # Save output locally
    modal run scripts/modal/inference.py --output reconstructed.png

    # List available models
    modal run scripts/modal/inference.py --list-models
"""

import modal
import sys
from pathlib import Path

VOLUME_NAME = "vitok-weights"

app = modal.App("vitok-inference")

# Inference image with all required packages
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4.0",
        "torchvision>=0.19.0",
        "safetensors>=0.4.0",
        "huggingface_hub>=0.23.0",
        "pillow>=10.0.0",
        "scikit-image",
        "numpy>=1.24.0",
    )
    .add_local_dir("vitok", remote_path="/root/vitok-release/vitok")
)

# Create volume for caching weights
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",
    volumes={"/cache": vol},
    timeout=600,
)
def run_inference(model_name: str, image_bytes: bytes | None = None) -> tuple[bytes, dict]:
    """Run AE encode/decode inference.

    Args:
        model_name: Pretrained model name (e.g., "L-64", "T-128")
        image_bytes: Optional input image bytes. If None, uses astronaut test image.

    Returns:
        Tuple of (output_image_bytes, info_dict)
    """
    import io
    import torch
    import numpy as np
    from PIL import Image
    from safetensors.torch import load_file

    # Add vitok to path
    sys.path.insert(0, "/root/vitok-release")

    from vitok import AE, decode_variant, preprocess, postprocess
    from vitok.utils.pretrained import download_pretrained, get_pretrained_info, list_pretrained

    # Get model info
    repo_id, filename, variant = get_pretrained_info(model_name)
    print(f"Model: {model_name}")
    print(f"  Variant: {variant}")
    print(f"  HuggingFace: {repo_id}/{filename}")

    # Download weights (cached in volume via HF_HOME)
    import os
    os.environ["HF_HOME"] = "/cache/huggingface"

    print("\nDownloading weights (or using cache)...")
    weights_path = download_pretrained(model_name)
    print(f"  Weights: {weights_path}")

    # Commit volume changes so weights persist
    vol.commit()

    # Load encoder and decoder separately
    print("\nLoading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    weights = load_file(weights_path)

    # Create encoder-only model
    encoder = AE(**decode_variant(variant), decoder=False)
    encoder.to(device=device, dtype=dtype)
    encoder.load_state_dict(weights, strict=False)
    encoder.eval()

    # Create decoder-only model
    decoder = AE(**decode_variant(variant), encoder=False)
    decoder.to(device=device, dtype=dtype)
    decoder.load_state_dict(weights, strict=False)
    decoder.eval()

    # Get input image
    if image_bytes is not None:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        print(f"\nInput image: {img.size[0]}x{img.size[1]}")
    else:
        from skimage import data
        astronaut = data.astronaut()
        img = Image.fromarray(astronaut)
        print(f"\nUsing astronaut test image: {img.size[0]}x{img.size[1]}")

    # Preprocess
    # Parse spatial stride from variant (e.g., "Ld4-Ld24/1x16x64" -> 16)
    spatial_stride = int(variant.split("/")[1].split("x")[1])
    pp_string = f"to_tensor|normalize(minus_one_to_one)|patchify({spatial_stride}, 256)"
    patch_dict = preprocess(img, pp=pp_string, device=device)

    # Cast patches to model dtype
    patch_dict = {
        k: v.to(dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() else v
        for k, v in patch_dict.items()
    }

    # Encode and decode
    print("Running encode/decode...")
    with torch.no_grad():
        encoded = encoder.encode(patch_dict)
        decoded = decoder.decode(encoded)

    # Get latent info
    z = encoded["z"]
    print(f"  Latent shape: {z.shape} (B, N, C)")

    # Postprocess to image
    images = postprocess(decoded, output_format="0_255", do_unpack=True, patch=spatial_stride)
    out_img = images[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    out_pil = Image.fromarray(out_img)

    # Save to bytes
    buf = io.BytesIO()
    out_pil.save(buf, format="PNG")
    out_bytes = buf.getvalue()

    info = {
        "model": model_name,
        "variant": variant,
        "input_size": (img.size[0], img.size[1]),
        "output_size": (out_pil.size[0], out_pil.size[1]),
        "latent_shape": list(z.shape),
        "device": device,
    }

    print(f"\nOutput image: {out_pil.size[0]}x{out_pil.size[1]}")
    print("Done!")

    return out_bytes, info


@app.local_entrypoint()
def main(
    model: str = "L-64",
    image: str | None = None,
    output: str | None = None,
    list_models: bool = False,
):
    """Run ViTok inference on Modal.

    Args:
        model: Pretrained model name (default: L-64)
        image: Path to input image (default: astronaut test image)
        output: Path to save output image (default: prints info only)
        list_models: List available pretrained models and exit
    """
    if list_models:
        # Import locally to list models
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from vitok.utils.pretrained import list_pretrained, PRETRAINED_ALIASES

        print("Available pretrained models:")
        print()
        print("Aliases (recommended):")
        for alias, full_name in sorted(PRETRAINED_ALIASES.items()):
            print(f"  {alias:10s} -> {full_name}")
        print()
        print("Full names:")
        from vitok.utils.pretrained import PRETRAINED_MODELS
        for name in sorted(PRETRAINED_MODELS.keys()):
            print(f"  {name}")
        return

    # Read input image if provided
    image_bytes = None
    if image is not None:
        image_path = Path(image)
        if not image_path.exists():
            print(f"Error: Image not found: {image}")
            return
        image_bytes = image_path.read_bytes()
        print(f"Input: {image}")

    print(f"Model: {model}")
    print(f"Running inference on Modal...\n")

    # Run inference
    out_bytes, info = run_inference.remote(model, image_bytes)

    print(f"\nResults:")
    print(f"  Model: {info['model']}")
    print(f"  Variant: {info['variant']}")
    print(f"  Input size: {info['input_size'][0]}x{info['input_size'][1]}")
    print(f"  Output size: {info['output_size'][0]}x{info['output_size'][1]}")
    print(f"  Latent shape: {info['latent_shape']}")

    if output is not None:
        output_path = Path(output)
        output_path.write_bytes(out_bytes)
        print(f"\nSaved to: {output}")
    else:
        print(f"\nTip: Use --output reconstructed.png to save the result")
