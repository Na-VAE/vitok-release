"""Download and cache model weights on Modal volume.

Usage:
    modal run scripts/modal/setup_weights.py

This downloads:
- FID InceptionV3 weights
- DINO ViT-B weights
- ViTok pretrained models (5B variants)

Weights are stored on vitok-data volume at /data/weights/
"""

import modal

app = modal.App("vitok-setup-weights")

# Minimal image for downloading
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        "safetensors>=0.4.0",
        "huggingface_hub>=0.23.0",
        "pytorch-fid>=0.3.0",
        "dino-perceptual>=0.1.0",
    )
)

data_vol = modal.Volume.from_name("vitok-data", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

VITOK_MODELS = [
    ("5B-f16x32", "philippehansen/ViTok-v2-5B-f16x32"),
    ("5B-f16x64", "philippehansen/ViTok-v2-5B-f16x64"),
    ("5B-f32x64", "philippehansen/ViTok-v2-5B-f32x64"),
    ("5B-f32x128", "philippehansen/ViTok-v2-5B-f32x128"),
]


@app.function(image=image, volumes={"/data": data_vol}, secrets=[hf_secret], timeout=1800)
def download_all_weights():
    """Download all model weights to the volume."""
    import os
    import shutil
    from pathlib import Path

    weights_dir = Path("/data/weights")
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Set cache dirs to volume
    os.environ["HF_HOME"] = str(weights_dir / "huggingface")
    os.environ["TORCH_HOME"] = str(weights_dir / "torch")

    # 1. FID InceptionV3
    print("=" * 60)
    print("Downloading FID InceptionV3 weights...")
    print("=" * 60)
    from pytorch_fid.inception import InceptionV3
    _ = InceptionV3(output_blocks=(InceptionV3.BLOCK_INDEX_BY_DIM[2048],))
    print("FID InceptionV3: OK")

    # 2. DINO ViT-B
    print("\n" + "=" * 60)
    print("Downloading DINO ViT-B weights...")
    print("=" * 60)
    from dino_perceptual import DINOModel
    _ = DINOModel(model_size='B', target_size=512)
    print("DINO ViT-B: OK")

    # 3. ViTok pretrained models
    print("\n" + "=" * 60)
    print("Downloading ViTok pretrained models...")
    print("=" * 60)
    from huggingface_hub import hf_hub_download

    for name, repo_id in VITOK_MODELS:
        print(f"\n  {name}:")
        enc_path = hf_hub_download(repo_id=repo_id, filename="encoder.safetensors")
        dec_path = hf_hub_download(repo_id=repo_id, filename="decoder.safetensors")
        print(f"    encoder: {enc_path}")
        print(f"    decoder: {dec_path}")

    # Commit volume
    data_vol.commit()

    # Show what was downloaded
    print("\n" + "=" * 60)
    print("Weights directory contents:")
    print("=" * 60)
    for root, dirs, files in os.walk(weights_dir):
        level = root.replace(str(weights_dir), '').count(os.sep)
        indent = '  ' * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = '  ' * (level + 1)
        for file in files[:5]:  # Limit files shown
            print(f"{subindent}{file}")
        if len(files) > 5:
            print(f"{subindent}... and {len(files) - 5} more files")

    print("\n" + "=" * 60)
    print("All weights downloaded successfully!")
    print("=" * 60)


@app.local_entrypoint()
def main():
    download_all_weights.remote()
