"""Clean up volume - keep only the tar file."""

import modal

app = modal.App("cleanup-volume")

volume = modal.Volume.from_name("vitok-downloads")


@app.function(
    volumes={"/data": volume},
    timeout=600,
)
def cleanup(keep_file: str = "gdrive_file"):
    """Delete everything except the specified file."""
    import os
    import shutil

    print("Current contents of /data:")
    items = os.listdir("/data")
    print(f"  {len(items)} items")

    deleted = 0
    kept = 0

    for item in items:
        path = os.path.join("/data", item)
        if item == keep_file:
            size = os.path.getsize(path) if os.path.isfile(path) else 0
            print(f"  KEEPING: {item} ({size / 1024 / 1024 / 1024:.2f} GB)")
            kept += 1
        else:
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
                deleted += 1
            except Exception as e:
                print(f"  ERROR deleting {item}: {e}")

    print(f"\nDeleted {deleted} items, kept {kept}")

    volume.commit()

    # Verify
    print("\nRemaining contents:")
    for item in os.listdir("/data"):
        path = os.path.join("/data", item)
        size = os.path.getsize(path) if os.path.isfile(path) else 0
        print(f"  {item}: {size / 1024 / 1024 / 1024:.2f} GB")


@app.local_entrypoint()
def main():
    cleanup.remote()
