"""Download a file from Google Drive to a Modal volume."""

import modal

app = modal.App("download-gdrive")

# Create or get the volume
volume = modal.Volume.from_name("vitok-downloads", create_if_missing=True)

@app.function(
    volumes={"/data": volume},
    timeout=3600,
)
def download_from_gdrive(file_id: str, output_name: str = "downloaded_file"):
    """Download a file from Google Drive using gdown."""
    import subprocess
    import os

    # Install gdown
    subprocess.run(["pip", "install", "gdown"], check=True)

    # Download the file
    output_path = f"/data/{output_name}"
    url = f"https://drive.google.com/uc?id={file_id}"

    print(f"Downloading from Google Drive file ID: {file_id}")
    print(f"Output path: {output_path}")

    result = subprocess.run(
        ["gdown", url, "-O", output_path],
        capture_output=True,
        text=True
    )

    print(result.stdout)
    if result.stderr:
        print(f"Stderr: {result.stderr}")

    if result.returncode != 0:
        # Try with --fuzzy flag for shared links
        print("Trying with --fuzzy flag...")
        result = subprocess.run(
            ["gdown", "--fuzzy", f"https://drive.google.com/file/d/{file_id}/view?usp=sharing", "-O", output_path],
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print(f"Stderr: {result.stderr}")

    # Check if file exists and show info
    if os.path.exists(output_path):
        size = os.path.getsize(output_path)
        print(f"Downloaded successfully: {output_path} ({size / 1024 / 1024:.2f} MB)")

        # List the /data directory
        print("\nContents of /data:")
        for f in os.listdir("/data"):
            fpath = os.path.join("/data", f)
            fsize = os.path.getsize(fpath) if os.path.isfile(fpath) else 0
            print(f"  {f}: {fsize / 1024 / 1024:.2f} MB")

        # Commit the volume
        volume.commit()
        return True
    else:
        print("Download failed!")
        return False


@app.local_entrypoint()
def main(
    file_id: str = "1qhbpX3ruisCu5Ow_66XasMhiZSIkuHLN",
    output_name: str = "downloaded_file",
):
    """Download a Google Drive file to Modal volume."""
    success = download_from_gdrive.remote(file_id, output_name)
    if success:
        print("\nFile saved to Modal volume 'vitok-downloads'")
        print("You can access it with: modal volume ls vitok-downloads")
    else:
        print("\nDownload failed!")
