"""Run ViTok AE inference on Modal.

This script downloads pretrained weights and runs encode/decode on images or videos.
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

    # Video inference
    modal run scripts/modal/inference.py --video path/to/video.mp4 --max-frames 8 --output-dir output/

    # Video from directory of frames
    modal run scripts/modal/inference.py --video frames_dir/ --max-frames 16 --output-dir output/

    # List available models
    modal run scripts/modal/inference.py --list-models
"""

import modal
import sys
from pathlib import Path

VOLUME_NAME = "vitok-weights"
CHECKPOINTS_VOLUME = "vitok-checkpoints"

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
        "webdataset",  # Required by vitok.data import
        "numpy>=1.24.0",
    )
    .add_local_dir("vitok", remote_path="/root/vitok-release/vitok")
)

# Create volumes for caching weights and accessing checkpoints
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
checkpoints_vol = modal.Volume.from_name(CHECKPOINTS_VOLUME, create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",
    volumes={"/cache": vol, "/checkpoints": checkpoints_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=600,
)
def run_inference(
    model_name: str,
    image_bytes: bytes | None = None,
    weights_path: str | None = None,
) -> tuple[bytes, dict]:
    """Run AE encode/decode inference.

    Args:
        model_name: Pretrained model name (e.g., "L-64", "T-128") - defines architecture
        image_bytes: Optional input image bytes. If None, uses astronaut test image.
        weights_path: Optional path to local weights file in /checkpoints volume.
                     If None, downloads from HuggingFace.

    Returns:
        Tuple of (output_image_bytes, info_dict)
    """
    import io
    import os
    import torch
    import numpy as np
    from PIL import Image
    from safetensors.torch import load_file

    # Add vitok to path
    sys.path.insert(0, "/root/vitok-release")

    from vitok import AE, decode_variant, preprocess, postprocess
    from vitok.pretrained import get_pretrained_info

    # Get model info (for variant/architecture)
    repo_id, filename, variant = get_pretrained_info(model_name)
    print(f"Model: {model_name}")
    print(f"  Variant: {variant}")

    # Get weights - either from local path or HuggingFace
    if weights_path is not None:
        # Use local weights from checkpoints volume
        full_weights_path = f"/checkpoints/{weights_path}"
        if not os.path.exists(full_weights_path):
            raise FileNotFoundError(f"Weights not found: {full_weights_path}")
        print(f"  Weights: {full_weights_path} (local)")
    else:
        # Download from HuggingFace
        from vitok.pretrained import download_pretrained
        print(f"  HuggingFace: {repo_id}/{filename}")
        os.environ["HF_HOME"] = "/cache/huggingface"
        print("\nDownloading weights (or using cache)...")
        full_weights_path = download_pretrained(model_name)
        print(f"  Weights: {full_weights_path}")
        vol.commit()

    # Load encoder and decoder separately
    print("\nLoading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    weights = load_file(full_weights_path)

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
        # Resize to 256x256 to fit max_tokens=256 with patch_size=16
        img = img.resize((256, 256), Image.LANCZOS)
        print(f"\nUsing astronaut test image (resized): {img.size[0]}x{img.size[1]}")

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


@app.function(
    image=image,
    gpu="T4",
    volumes={"/cache": vol, "/checkpoints": checkpoints_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=600,
)
def run_video_inference(
    model_name: str,
    video_bytes: bytes | None = None,
    frame_bytes_list: list[bytes] | None = None,
    max_frames: int = 8,
    temporal_stride: int = 1,
    weights_path: str | None = None,
) -> tuple[list[bytes], dict]:
    """Run AE encode/decode on video frames.

    Processes video in batch mode - each frame is encoded/decoded independently.

    Args:
        model_name: Pretrained model name (e.g., "L-64", "T-128") - defines architecture
        video_bytes: Video file bytes (.mp4, etc.). Mutually exclusive with frame_bytes_list.
        frame_bytes_list: List of frame image bytes. Mutually exclusive with video_bytes.
        max_frames: Maximum number of frames to process
        temporal_stride: Sample every Nth frame (currently only stride=1)
        weights_path: Optional path to local weights file in /checkpoints volume.
                     If None, downloads from HuggingFace.

    Returns:
        Tuple of (list_of_frame_bytes, info_dict)
    """
    import io
    import os
    import tempfile
    import torch
    import numpy as np
    from PIL import Image
    from safetensors.torch import load_file

    # Add vitok to path
    sys.path.insert(0, "/root/vitok-release")

    from vitok import AE, decode_variant, postprocess
    from vitok.pretrained import get_pretrained_info
    from vitok.video import extract_frames
    from vitok.pp import preprocess_video, postprocess_video

    # Get model info (for variant/architecture)
    repo_id, filename, variant = get_pretrained_info(model_name)
    print(f"Model: {model_name}")
    print(f"  Variant: {variant}")

    # Get weights - either from local path or HuggingFace
    if weights_path is not None:
        full_weights_path = f"/checkpoints/{weights_path}"
        if not os.path.exists(full_weights_path):
            raise FileNotFoundError(f"Weights not found: {full_weights_path}")
        print(f"  Weights: {full_weights_path} (local)")
    else:
        from vitok.pretrained import download_pretrained
        print(f"  HuggingFace: {repo_id}/{filename}")
        os.environ["HF_HOME"] = "/cache/huggingface"
        print("\nDownloading weights (or using cache)...")
        full_weights_path = download_pretrained(model_name)
        print(f"  Weights: {full_weights_path}")
        vol.commit()

    # Load model
    print("\nLoading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    weights = load_file(full_weights_path)

    encoder = AE(**decode_variant(variant), decoder=False)
    encoder.to(device=device, dtype=dtype)
    encoder.load_state_dict(weights, strict=False)
    encoder.eval()

    decoder = AE(**decode_variant(variant), encoder=False)
    decoder.to(device=device, dtype=dtype)
    decoder.load_state_dict(weights, strict=False)
    decoder.eval()

    # Extract frames from video or load from bytes
    if video_bytes is not None:
        # Save video to temp file and extract frames
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(video_bytes)
            temp_path = f.name
        try:
            frames = extract_frames(temp_path, max_frames=max_frames, temporal_stride=temporal_stride)
        finally:
            os.unlink(temp_path)
        print(f"\nExtracted {len(frames)} frames from video")
    elif frame_bytes_list is not None:
        frames = [Image.open(io.BytesIO(b)).convert("RGB") for b in frame_bytes_list[:max_frames]]
        print(f"\nLoaded {len(frames)} frames from bytes")
    else:
        # Use test frames (repeat astronaut)
        from skimage import data
        astronaut = data.astronaut()
        img = Image.fromarray(astronaut)
        # Resize to 256x256 to fit max_tokens=256 with patch_size=16
        img = img.resize((256, 256), Image.LANCZOS)
        frames = [img] * min(4, max_frames)
        print(f"\nUsing {len(frames)} copies of astronaut test image (resized to 256x256)")

    if len(frames) == 0:
        raise ValueError("No frames to process")

    print(f"Frame size: {frames[0].size[0]}x{frames[0].size[1]}")

    # Preprocess all frames as batch
    spatial_stride = int(variant.split("/")[1].split("x")[1])
    pp_string = f"to_tensor|normalize(minus_one_to_one)|patchify({spatial_stride}, 256)"
    patch_dict = preprocess_video(frames, pp=pp_string, mode="batch", device=device)

    # Cast to model dtype
    patch_dict = {
        k: v.to(dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() else v
        for k, v in patch_dict.items()
    }

    print(f"Batch patches shape: {patch_dict['patches'].shape}")

    # Encode and decode
    print("Running encode/decode on all frames...")
    with torch.no_grad():
        encoded = encoder.encode(patch_dict)
        decoded = decoder.decode(encoded)

    z = encoded["z"]
    print(f"  Latent shape: {z.shape} (B, N, C)")

    # Postprocess to frames
    output_frames = postprocess_video(decoded, mode="batch", patch=spatial_stride)

    # Convert frames to bytes
    frame_bytes_list = []
    for frame in output_frames:
        buf = io.BytesIO()
        frame.save(buf, format="PNG")
        frame_bytes_list.append(buf.getvalue())

    info = {
        "model": model_name,
        "variant": variant,
        "num_frames": len(frames),
        "frame_size": (frames[0].size[0], frames[0].size[1]),
        "latent_shape": list(z.shape),
        "device": device,
    }

    print(f"\nProcessed {len(output_frames)} frames")
    print("Done!")

    return frame_bytes_list, info


@app.local_entrypoint()
def main(
    model: str = "L-64",
    image: str | None = None,
    output: str | None = None,
    video: str | None = None,
    max_frames: int = 8,
    temporal_stride: int = 1,
    output_dir: str | None = None,
    weights: str | None = None,
    list_models: bool = False,
    list_checkpoints: bool = False,
):
    """Run ViTok inference on Modal.

    Args:
        model: Pretrained model name (default: L-64) - defines architecture
        image: Path to input image (default: astronaut test image)
        output: Path to save output image (default: prints info only)
        video: Path to video file or directory of frames
        max_frames: Maximum frames to process for video (default: 8)
        temporal_stride: Sample every Nth frame (default: 1)
        output_dir: Directory to save output frames (for video)
        weights: Path to weights in vitok-checkpoints volume (e.g., vae/last/model.safetensors)
        list_models: List available pretrained models and exit
        list_checkpoints: List available checkpoints in Modal volume and exit
    """
    if list_checkpoints:
        import subprocess
        print("Checkpoints in vitok-checkpoints volume:")
        result = subprocess.run(
            ["modal", "volume", "ls", "vitok-checkpoints", "-r"],
            capture_output=True, text=True
        )
        print(result.stdout or "(empty)")
        if result.stderr:
            print(result.stderr)
        return

    if list_models:
        # Define models locally to avoid importing torch
        aliases = {
            "L-64": "Ld4-Ld24/1x16x64",
            "L-32": "Ld4-Ld24/1x32x64",
            "L-16": "Ld4-Ld24/1x16x16",
            "T-64": "Td2-Td12/1x16x64",
            "T-128": "Td2-Td12/1x16x128",
            "T-256": "Td2-Td12/1x16x256",
        }
        print("Available pretrained models:")
        print()
        print("Aliases (recommended):")
        for alias, full_name in sorted(aliases.items()):
            print(f"  {alias:10s} -> {full_name}")
        print()
        print("Tip: Use --list-checkpoints to see local weights in Modal volume")
        return

    # Check for mutually exclusive inputs
    if video is not None and image is not None:
        print("Error: Cannot specify both --image and --video")
        return

    # Video inference
    if video is not None:
        video_path = Path(video)
        if not video_path.exists():
            print(f"Error: Video/directory not found: {video}")
            return

        print(f"Model: {model}")
        if weights:
            print(f"Weights: {weights} (local)")
        print(f"Video: {video}")
        print(f"Max frames: {max_frames}")
        print(f"Running video inference on Modal...\n")

        # Determine if it's a video file or directory
        video_bytes = None
        frame_bytes_list = None

        if video_path.is_file():
            video_bytes = video_path.read_bytes()
        elif video_path.is_dir():
            # Collect frame paths and read their bytes
            image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
            frame_paths = sorted(
                p for p in video_path.iterdir()
                if p.suffix.lower() in image_extensions
            )
            if not frame_paths:
                print(f"Error: No image files found in {video}")
                return
            print(f"Found {len(frame_paths)} frames in directory")
            # Read frame bytes locally
            frame_bytes_list = [p.read_bytes() for p in frame_paths[:max_frames]]

        # Run video inference
        output_frame_bytes, info = run_video_inference.remote(
            model, video_bytes, frame_bytes_list, max_frames, temporal_stride, weights
        )

        print(f"\nResults:")
        print(f"  Model: {info['model']}")
        print(f"  Variant: {info['variant']}")
        print(f"  Num frames: {info['num_frames']}")
        print(f"  Frame size: {info['frame_size'][0]}x{info['frame_size'][1]}")
        print(f"  Latent shape: {info['latent_shape']}")

        if output_dir is not None:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            for i, frame_bytes in enumerate(output_frame_bytes):
                frame_path = out_dir / f"frame_{i:04d}.png"
                frame_path.write_bytes(frame_bytes)
            print(f"\nSaved {len(output_frame_bytes)} frames to: {output_dir}")
        else:
            print(f"\nTip: Use --output-dir output/ to save the frames")

        return

    # Image inference (original behavior)
    image_bytes = None
    if image is not None:
        image_path = Path(image)
        if not image_path.exists():
            print(f"Error: Image not found: {image}")
            return
        image_bytes = image_path.read_bytes()
        print(f"Input: {image}")

    print(f"Model: {model}")
    if weights:
        print(f"Weights: {weights} (local)")
    print(f"Running inference on Modal...\n")

    # Run inference
    out_bytes, info = run_inference.remote(model, image_bytes, weights)

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
