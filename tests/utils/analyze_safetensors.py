"""Extract tar.gz and analyze safetensors files."""

import modal

app = modal.App("analyze-safetensors")

volume = modal.Volume.from_name("vitok-downloads")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "safetensors",
    "torch",
)


@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=3600,
    cpu=4,
    memory=32768,  # 32GB for large files
)
def extract_and_analyze(tar_path: str = "/data/gdrive_file", extract_dir: str = "/data/extracted"):
    """Extract tar.gz and analyze safetensors."""
    import os
    import tarfile
    from safetensors import safe_open

    # Check if already extracted
    if os.path.exists(extract_dir) and os.listdir(extract_dir):
        print(f"Already extracted to {extract_dir}")
    else:
        # Extract
        print(f"Extracting {tar_path}...")
        os.makedirs(extract_dir, exist_ok=True)

        with tarfile.open(tar_path, "r:gz") as tar:
            members = tar.getmembers()
            print(f"Archive contains {len(members)} files")

            print("\nFirst 20 files:")
            for m in members[:20]:
                print(f"  {m.name} ({m.size / 1024 / 1024:.2f} MB)")

            print("\nExtracting...")
            tar.extractall(extract_dir)

        print("Extraction complete!")
        volume.commit()

    # Find safetensors files
    print("\n" + "=" * 60)
    print("SAFETENSORS ANALYSIS")
    print("=" * 60)

    safetensor_files = []
    for root, dirs, files in os.walk(extract_dir):
        for f in files:
            if f.endswith(".safetensors"):
                safetensor_files.append(os.path.join(root, f))

    print(f"\nFound {len(safetensor_files)} safetensor files:")

    for sf_path in safetensor_files:
        print(f"\n{'=' * 60}")
        print(f"File: {os.path.basename(sf_path)}")
        print(f"Size: {os.path.getsize(sf_path) / 1024 / 1024:.2f} MB")
        print("=" * 60)

        try:
            with safe_open(sf_path, framework="pt") as f:
                keys = list(f.keys())
                print(f"\nTensors ({len(keys)} total):")

                # Group by prefix
                prefixes = {}
                for key in keys:
                    prefix = key.split(".")[0] if "." in key else key
                    if prefix not in prefixes:
                        prefixes[prefix] = []
                    prefixes[prefix].append(key)

                for prefix, pkeys in sorted(prefixes.items()):
                    print(f"\n  {prefix}/ ({len(pkeys)} tensors)")
                    for key in pkeys[:5]:
                        tensor = f.get_tensor(key)
                        print(f"    {key}: {list(tensor.shape)} {tensor.dtype}")
                    if len(pkeys) > 5:
                        print(f"    ... and {len(pkeys) - 5} more")

                # Total params
                total_params = 0
                for key in keys:
                    tensor = f.get_tensor(key)
                    total_params += tensor.numel()

                print(f"\nTotal parameters: {total_params:,} ({total_params / 1e6:.1f}M)")

        except Exception as e:
            print(f"Error reading: {e}")

    # List other file types
    print("\n" + "=" * 60)
    print("OTHER FILES")
    print("=" * 60)

    other_files = []
    for root, dirs, files in os.walk(extract_dir):
        for f in files:
            if not f.endswith(".safetensors"):
                fpath = os.path.join(root, f)
                other_files.append((fpath, os.path.getsize(fpath)))

    for fpath, fsize in sorted(other_files, key=lambda x: -x[1])[:20]:
        print(f"  {os.path.relpath(fpath, extract_dir)}: {fsize / 1024 / 1024:.2f} MB")

    return True


@app.local_entrypoint()
def main():
    extract_and_analyze.remote()
