"""Create Modal volume for caching ViTok model weights.

This script creates a persistent volume to store downloaded model weights.
Weights are downloaded once and reused across inference runs.

Usage:
    modal run scripts/modal/setup_volume.py

    # Delete and recreate the volume
    modal run scripts/modal/setup_volume.py --delete
"""

import modal

VOLUME_NAME = "vitok-weights"

app = modal.App("vitok-setup-volume")


@app.local_entrypoint()
def main(delete: bool = False):
    """Create or manage the vitok-weights volume."""
    if delete:
        try:
            vol = modal.Volume.from_name(VOLUME_NAME)
            vol.delete()
            print(f"Deleted volume '{VOLUME_NAME}'")
        except modal.exception.NotFoundError:
            print(f"Volume '{VOLUME_NAME}' does not exist")
            return

    # Create volume (or verify it exists)
    try:
        vol = modal.Volume.from_name(VOLUME_NAME)
        print(f"Volume '{VOLUME_NAME}' already exists")
    except modal.exception.NotFoundError:
        vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
        print(f"Created volume '{VOLUME_NAME}'")

    print()
    print("=" * 50)
    print("Volume Management Commands")
    print("=" * 50)
    print(f"  List contents:   modal volume ls {VOLUME_NAME}")
    print(f"  Delete volume:   modal volume rm {VOLUME_NAME} --recursive")
    print(f"  Download file:   modal volume get {VOLUME_NAME} /path/to/file ./local/")
    print("=" * 50)
    print()
    print("You can now run: modal run scripts/modal/inference.py")
