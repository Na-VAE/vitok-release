"""Clean up model weights and re-upload to HuggingFace.

Removes _orig_mod. prefix and remaps key names to match inference model.
"""

import modal

app = modal.App("clean-weights")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "safetensors",
    "huggingface_hub",
    "torch",
    "numpy",
)

# Only models that have model.safetensors uploaded
MODELS = [
    "philippehansen/ViTok-L-16x64",
    "philippehansen/ViTok-L-16x32",
    "philippehansen/ViTok-L-16x16",
    "philippehansen/ViTok-T-32x64",
    # T-32x128 and T-32x256 need to be re-uploaded first
]


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
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=1800,
)
def clean_and_upload(repo_id: str):
    """Download, clean, and re-upload weights for a single model."""
    import os
    from safetensors.torch import load_file, save_file
    from huggingface_hub import hf_hub_download, HfApi, login

    token = os.environ.get("HF_TOKEN")
    login(token=token)
    api = HfApi()

    print(f"Processing {repo_id}...")

    # Download
    path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
    print(f"  Downloaded: {path}")

    # Load and check
    weights = load_file(path)
    sample_keys = list(weights.keys())[:5]
    print(f"  Original keys ({len(weights)} total): {sample_keys}")

    # Remap
    cleaned = remap_keys(weights)
    sample_keys = list(cleaned.keys())[:5]
    print(f"  Cleaned keys ({len(cleaned)} total): {sample_keys}")

    # Save locally
    local_path = f"/tmp/cleaned.safetensors"
    save_file(cleaned, local_path)
    size_gb = os.path.getsize(local_path) / (1024**3)
    print(f"  Saved: {size_gb:.2f} GB")

    # Upload
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo="model.safetensors",
        repo_id=repo_id,
    )
    print(f"  Uploaded to {repo_id}")

    os.remove(local_path)
    return repo_id


@app.local_entrypoint()
def main():
    """Clean all model weights in parallel."""
    print(f"Cleaning {len(MODELS)} models in parallel...")

    results = []
    for result in clean_and_upload.map(MODELS):
        print(f"  Done: {result}")
        results.append(result)

    print(f"\nAll {len(results)} models cleaned!")
