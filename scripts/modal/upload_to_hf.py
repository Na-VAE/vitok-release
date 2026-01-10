"""Upload model weights from Modal volume directly to HuggingFace Hub."""

import modal

app = modal.App("upload-to-hf")

volume = modal.Volume.from_name("vitok-downloads")

image = modal.Image.debian_slim(python_version="3.11").pip_install("huggingface_hub")

# Mapping: volume path -> (HuggingFace repo, variant description)
# Naming: ViTok-{size}-{spatial}x{latent}
# e.g., ViTok-L-16x64 = Large model, 16x spatial compression, 64 latent channels
MODELS = {
    # Large models (Ld4-Ld24 = encoder depth 4, decoder depth 24)
    "Ld4-L_x16x64_test": ("philippehansen/ViTok-L-16x64", "1x16x64"),  # spatial 16, latent 64
    "Ld4-L_x16x32_test": ("philippehansen/ViTok-L-16x32", "1x16x32"),  # spatial 16, latent 32
    "Ld4-L_x16x16_test": ("philippehansen/ViTok-L-16x16", "1x16x16"),  # spatial 16, latent 16
    # Tiny models (Td2-Td12 = encoder depth 2, decoder depth 12)
    "Td4-T_x32x64": ("philippehansen/ViTok-T-32x64", "1x32x64"),   # spatial 32, latent 64
    "Td4-T_x32x128": ("philippehansen/ViTok-T-32x128", "1x32x128"), # spatial 32, latent 128
    "Td4-T_x32x256": ("philippehansen/ViTok-T-32x256", "1x32x256"), # spatial 32, latent 256
}


@app.function(
    image=image,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=1800,
)
def upload_model(volume_name: str, repo_id: str, private: bool = True):
    """Upload a single model to HuggingFace."""
    import os
    from huggingface_hub import HfApi, login

    # Login with token from secret
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN not found in huggingface-secret")
    login(token=token)

    # Find the safetensors file
    model_path = f"/data/extracted/dir_for_transfer/{volume_name}/model.safetensors"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    file_size = os.path.getsize(model_path) / (1024 * 1024 * 1024)
    print(f"Uploading {volume_name} ({file_size:.2f} GB) -> {repo_id}")

    api = HfApi()

    # Create repo if it doesn't exist (private by default)
    api.create_repo(repo_id, exist_ok=True, repo_type="model", private=private)

    # Upload the file
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="model.safetensors",
        repo_id=repo_id,
    )

    print(f"Done! https://huggingface.co/{repo_id}")
    return repo_id


@app.local_entrypoint()
def main(model: str = "all", public: bool = False):
    """Upload models to HuggingFace.

    Args:
        model: Model to upload ("all" or specific name like "L-16x64")
        public: If True, make repos public (default: private)
    """
    private = not public

    if model == "all":
        print(f"Uploading all {len(MODELS)} models in PARALLEL...")
        print(f"Privacy: {'private' if private else 'public'}")

        # Fire off all uploads in parallel using starmap
        args = [(vol_name, repo_id, private) for vol_name, (repo_id, _) in MODELS.items()]

        results = []
        for result in upload_model.starmap(args):
            print(f"  Completed: {result}")
            results.append(result)

        print(f"\nAll {len(results)} models uploaded!")
        for r in results:
            print(f"  https://huggingface.co/{r}")
    else:
        # Find by name
        volume_name = None
        repo_id = None

        # Check direct volume name match
        if model in MODELS:
            volume_name = model
            repo_id = MODELS[model][0]
        else:
            # Check by repo suffix (e.g., "L-16x64" matches "philippehansen/ViTok-L-16x64")
            for vol, (repo, _) in MODELS.items():
                if repo.endswith(model) or repo.endswith(f"ViTok-{model}"):
                    volume_name = vol
                    repo_id = repo
                    break

        if not volume_name:
            print(f"Unknown model: {model}")
            print(f"Available:")
            for vol, (repo, variant) in MODELS.items():
                name = repo.split("/")[-1]
                print(f"  {name} ({variant})")
            return

        print(f"Uploading {model}...")
        result = upload_model.remote(volume_name, repo_id, private)
        print(f"Done: https://huggingface.co/{result}")
