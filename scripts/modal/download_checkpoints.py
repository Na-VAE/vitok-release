"""Download checkpoints from Google Drive to Modal volume using rclone.

Usage:
    modal run scripts/modal/download_checkpoints.py
    modal run scripts/modal/download_checkpoints.py --check
"""

import modal

VOLUME_NAME = "vitok-weights"

app = modal.App("vitok-download-checkpoints")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "unzip")
    .run_commands("curl https://rclone.org/install.sh | bash")
)

vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Files to download
FILES = [
    ("1qhbpX3ruisCu5Ow_66XasMhiZSIkuHLN", "transfer_for_eval.tar.gz"),
    ("16Kns7sp0wphpPDfVmbwSn2GNNv6fnHGR", "transfer_for_eval_1.tar.gz"),
]


@app.function(
    image=image,
    volumes={"/weights": vol},
    secrets=[modal.Secret.from_name("rclone-config")],
    timeout=14400,  # 4 hours
)
def download_gdrive(file_id: str, filename: str) -> dict:
    """Download from Google Drive using rclone with user's config."""
    import base64
    import os
    import subprocess
    from pathlib import Path

    # Set up rclone config from secret
    config_b64 = os.environ.get("RCLONE_CONFIG_B64")
    if not config_b64:
        return {"status": "error", "error": "RCLONE_CONFIG_B64 not set"}

    config_dir = Path("/root/.config/rclone")
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "rclone.conf"
    config_path.write_bytes(base64.b64decode(config_b64))
    print(f"Wrote rclone config to {config_path}")

    download_path = Path("/weights") / filename

    # Check if already downloaded
    if download_path.exists():
        size_gb = download_path.stat().st_size / (1024**3)
        if size_gb > 1:  # More than 1GB means real file
            return {"status": "cached", "size_gb": round(size_gb, 2)}

    print(f"Downloading {file_id} -> {download_path}")

    # Use rclone backend copyid
    result = subprocess.run(
        ["rclone", "backend", "copyid", "gdrive:", file_id, str(download_path)],
        capture_output=True,
        text=True,
    )

    print(f"stdout: {result.stdout}")
    print(f"stderr: {result.stderr}")

    if download_path.exists():
        size_gb = download_path.stat().st_size / (1024**3)
        vol.commit()
        return {"status": "downloaded", "size_gb": round(size_gb, 2)}

    return {"status": "error", "stderr": result.stderr}


@app.function(image=image, volumes={"/weights": vol}, timeout=60)
def check_volume() -> dict:
    """Check what's in the volume."""
    from pathlib import Path

    results = {}
    weights_dir = Path("/weights")
    if not weights_dir.exists():
        return {"status": "empty"}

    for item in weights_dir.iterdir():
        if item.is_file():
            results[item.name] = {"size_gb": round(item.stat().st_size / (1024**3), 2)}
        elif item.is_dir():
            files = [f for f in item.rglob("*") if f.is_file()]
            size = sum(f.stat().st_size for f in files)
            results[item.name] = {"files": len(files), "size_gb": round(size / (1024**3), 2)}
    return results


@app.local_entrypoint()
def main(check: bool = False):
    if check:
        print("Checking volume...")
        results = check_volume.remote()
        if not results or results.get("status") == "empty":
            print("  Volume is empty")
        else:
            for name, info in results.items():
                print(f"  {name}: {info}")
        return

    print(f"Downloading {len(FILES)} files in parallel...")

    # Launch both downloads in parallel
    results = list(download_gdrive.starmap(FILES))

    print("\n" + "=" * 50)
    print("Results:")
    print("=" * 50)
    for (file_id, filename), result in zip(FILES, results):
        print(f"{filename}: {result}")
