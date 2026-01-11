"""Clean weights from Modal volume and upload to HuggingFace.

Reads from extracted tar files, remaps keys, casts to bf16, and uploads as public repos.

Usage:
    # Clean and upload all models
    modal run scripts/modal/clean_and_upload_from_volume.py

    # Upload specific model
    modal run scripts/modal/clean_and_upload_from_volume.py --model T-32x64

    # Dry run (no upload)
    modal run scripts/modal/clean_and_upload_from_volume.py --dry-run
"""

import modal

app = modal.App("clean-upload-volume")

volume = modal.Volume.from_name("vitok-downloads")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "safetensors",
    "huggingface_hub",
    "torch",
    "numpy",
)

# Mapping: volume folder name -> (HuggingFace repo, variant)
MODELS = {
    # T models (Td4 = encoder depth 4, decoder uses T width)
    "Td4-T_x32x64": ("philippehansen/ViTok-T-32x64", "1x32x64"),
    "Td4-T_x32x128": ("philippehansen/ViTok-T-32x128", "1x32x128"),
    "Td4-T_x32x256": ("philippehansen/ViTok-T-32x256", "1x32x256"),
    # L models (Ld4 = encoder depth 4, decoder uses L width)
    "Ld4-L_x16x16_test": ("philippehansen/ViTok-L-16x16", "1x16x16"),
    "Ld4-L_x16x32_test": ("philippehansen/ViTok-L-16x32", "1x16x32"),
    "Ld4-L_x16x64_test": ("philippehansen/ViTok-L-16x64", "1x16x64"),
}


def remap_keys(state_dict):
    """Remap keys from training checkpoint format to clean inference format."""
    new_dict = {}
    for key, value in state_dict.items():
        new_key = key

        # Remove _orig_mod. prefix (from torch.compile)
        if new_key.startswith("_orig_mod."):
            new_key = new_key[len("_orig_mod."):]

        # Rename encoder.N -> encoder_blocks.N
        if new_key.startswith("encoder.") and new_key.split(".")[1].isdigit():
            parts = new_key.split(".")
            parts[0] = "encoder_blocks"
            new_key = ".".join(parts)

        # Rename decoder.N -> decoder_blocks.N
        if new_key.startswith("decoder.") and new_key.split(".")[1].isdigit():
            parts = new_key.split(".")
            parts[0] = "decoder_blocks"
            new_key = ".".join(parts)

        new_dict[new_key] = value

    return new_dict


@app.function(
    image=image,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,  # 1 hour per model (large files)
    memory=32768,  # 32GB for large models
)
def clean_and_upload(volume_name: str, repo_id: str, dry_run: bool = False):
    """Clean weights from volume and upload to HuggingFace."""
    import os
    import torch
    from safetensors.torch import load_file, save_file
    from huggingface_hub import HfApi, login

    token = os.environ.get("HF_TOKEN")
    if not dry_run:
        login(token=token)
        api = HfApi()

    print(f"\n{'='*60}")
    print(f"Processing: {volume_name} -> {repo_id}")
    print(f"{'='*60}")

    # Load from volume
    source_path = f"/data/extracted/dir_for_transfer/{volume_name}/model.safetensors"
    if not os.path.exists(source_path):
        print(f"  ERROR: Source not found: {source_path}")
        return {"status": "error", "error": "source_not_found"}

    size_gb = os.path.getsize(source_path) / (1024**3)
    print(f"  Source: {source_path} ({size_gb:.2f} GB)")

    # Load weights
    print("  Loading weights...")
    weights = load_file(source_path)
    print(f"  Loaded {len(weights)} tensors")

    # Show sample original keys
    sample_keys = list(weights.keys())[:3]
    print(f"  Original keys: {sample_keys}")

    # Remap keys
    print("  Remapping keys...")
    cleaned = remap_keys(weights)
    sample_keys = list(cleaned.keys())[:3]
    print(f"  Remapped keys: {sample_keys}")

    # Cast to bf16
    print("  Casting to bf16...")
    cleaned = {k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v
               for k, v in cleaned.items()}

    # Count params
    total_params = sum(v.numel() for v in cleaned.values())
    print(f"  Total params: {total_params:,} ({total_params/1e9:.2f}B)")

    # Save cleaned weights
    cleaned_path = f"/tmp/{volume_name}_cleaned.safetensors"
    print(f"  Saving to: {cleaned_path}")
    save_file(cleaned, cleaned_path)

    cleaned_size_gb = os.path.getsize(cleaned_path) / (1024**3)
    print(f"  Cleaned size: {cleaned_size_gb:.2f} GB (was {size_gb:.2f} GB)")

    if dry_run:
        print("  [DRY RUN] Skipping upload")
        os.remove(cleaned_path)
        return {"status": "dry_run", "size_gb": cleaned_size_gb}

    # Upload to HuggingFace
    print(f"  Creating/updating repo: {repo_id}")
    api.create_repo(repo_id, exist_ok=True, repo_type="model", private=False)

    print(f"  Uploading...")
    api.upload_file(
        path_or_fileobj=cleaned_path,
        path_in_repo="model.safetensors",
        repo_id=repo_id,
    )

    print(f"  SUCCESS: https://huggingface.co/{repo_id}")

    os.remove(cleaned_path)
    return {"status": "success", "repo_id": repo_id, "size_gb": cleaned_size_gb}


@app.local_entrypoint()
def main(model: str = "all", dry_run: bool = False):
    """Clean and upload models from volume to HuggingFace.

    Args:
        model: Model to upload ("all" or specific like "T-32x64")
        dry_run: If True, don't actually upload
    """
    if dry_run:
        print("DRY RUN MODE - No uploads will be made")

    if model == "all":
        print(f"Processing all {len(MODELS)} models...")

        results = []
        for volume_name, (repo_id, variant) in MODELS.items():
            result = clean_and_upload.remote(volume_name, repo_id, dry_run)
            results.append(result)
            print(f"  {volume_name}: {result.get('status')}")

        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        success = sum(1 for r in results if r.get("status") == "success")
        print(f"Success: {success}/{len(results)}")

        for (vn, (repo, _)), r in zip(MODELS.items(), results):
            status = r.get("status", "unknown")
            if status == "success":
                print(f"  ✅ {repo}")
            elif status == "dry_run":
                print(f"  ⏸️  {repo} (dry run)")
            else:
                print(f"  ❌ {vn}: {r.get('error', 'unknown error')}")

    else:
        # Find model by name
        volume_name = None
        repo_id = None

        for vn, (repo, _) in MODELS.items():
            short_name = repo.split("/")[-1].replace("ViTok-", "")
            if model == vn or model == short_name or repo.endswith(model):
                volume_name = vn
                repo_id = repo
                break

        if not volume_name:
            print(f"Unknown model: {model}")
            print("Available models:")
            for vn, (repo, variant) in MODELS.items():
                name = repo.split("/")[-1]
                print(f"  {name} ({variant})")
            return

        result = clean_and_upload.remote(volume_name, repo_id, dry_run)
        print(f"\nResult: {result}")
