"""Download a single file from Google Drive to a Modal volume using rclone."""

import modal

app = modal.App("download-gdrive-rclone")

volume = modal.Volume.from_name("vitok-downloads", create_if_missing=True)

image = modal.Image.debian_slim().apt_install("curl", "unzip").run_commands(
    "curl -O https://downloads.rclone.org/rclone-current-linux-amd64.zip",
    "unzip rclone-current-linux-amd64.zip",
    "cp rclone-*-linux-amd64/rclone /usr/bin/",
    "chmod 755 /usr/bin/rclone",
)


@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=7200,
    cpu=2,
    memory=4096,
)
def find_and_download(file_id: str, output_name: str, rclone_config: str):
    """Find a file by ID in your own Drive and download it."""
    import subprocess
    import os
    import json

    # Write rclone config
    config_dir = os.path.expanduser("~/.config/rclone")
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, "rclone.conf")

    with open(config_path, "w") as f:
        f.write(rclone_config)

    output_path = f"/data/{output_name}"
    print(f"Looking for file ID: {file_id}")

    # List files in your own drive (no --drive-shared-with-me)
    result = subprocess.run(
        [
            "rclone", "lsjson", "gdrive:",
            "--files-only",
            "-R",
            "--no-modtime",
        ],
        capture_output=True,
        text=True
    )

    # Parse JSON and find file by ID
    files = json.loads(result.stdout) if result.stdout else []
    target_file = None
    for f in files:
        if f.get("ID") == file_id:
            target_file = f
            print(f"Found: {f['Path']} (ID: {f['ID']}, Size: {f['Size'] / 1024 / 1024:.1f} MB)")
            break

    if not target_file:
        print(f"File ID {file_id} not found.")
        print(f"Checked {len(files)} files.")
        print("\nFirst 10 file IDs found:")
        for f in files[:10]:
            print(f"  {f.get('ID')}: {f.get('Path')}")
        return False

    # Now copy just that file
    file_path = target_file["Path"]
    print(f"\nDownloading: {file_path} -> {output_path}")

    result = subprocess.run(
        [
            "rclone", "copyto",
            f"gdrive:{file_path}",
            output_path,
            "--progress",
            "-v",
        ],
    )

    if os.path.exists(output_path):
        fsize = os.path.getsize(output_path)
        print(f"\nSuccess! {output_path} ({fsize / 1024 / 1024:.2f} MB)")
        volume.commit()
        return True
    else:
        print("\nDownload failed")
        return False


@app.local_entrypoint()
def main(
    file_id: str = "1qhbpX3ruisCu5Ow_66XasMhiZSIkuHLN",
    output_name: str = "gdrive_file",
):
    """Download a Google Drive file to Modal volume."""
    import os

    config_path = os.path.expanduser("~/.config/rclone/rclone.conf")
    if not os.path.exists(config_path):
        print("ERROR: Run 'rclone config' first")
        return

    with open(config_path, "r") as f:
        rclone_config = f.read()

    print("Starting download on Modal...")
    success = find_and_download.remote(file_id, output_name, rclone_config)

    if success:
        print("\nDone! Check with: modal volume ls vitok-downloads")
