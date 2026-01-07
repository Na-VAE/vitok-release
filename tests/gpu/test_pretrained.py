"""Test the pretrained model loading API."""

import modal

app = modal.App("vitok-pretrained-api-test")

VITOK_PATH = "/root/vitok-release"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.4.0",
        "torchvision>=0.19.0",
        "safetensors>=0.4.0",
        "numpy>=1.24.0",
        "pillow>=10.0.0",
        "packaging>=21.0",
        "webdataset>=0.2.86",
        "huggingface_hub>=0.23.0,<1.0",
    )
    .add_local_dir("vitok", remote_path=f"{VITOK_PATH}/vitok")
)

downloads_volume = modal.Volume.from_name("vitok-downloads")


@app.function(
    image=image,
    gpu="T4",
    volumes={"/downloads": downloads_volume},
    timeout=300,
)
def test_api():
    import sys
    sys.path.insert(0, VITOK_PATH)

    import torch
    import numpy as np
    from PIL import Image

    print("=" * 60)
    print("Test 1: List pretrained models")
    print("=" * 60)

    import vitok
    print(f"Available pretrained models:")
    for name in vitok.list_pretrained():
        print(f"  {name}")

    print("\n" + "=" * 60)
    print("Test 2: Load with local checkpoint path")
    print("=" * 60)

    # Test loading from local path (simulating what pretrained would do)
    ckpt_path = "/downloads/extracted/dir_for_transfer/Ld4-L_x16x64_test/model.safetensors"
    config = vitok.AEConfig(variant="Ld4-Ld24/1x16x64")
    ae = vitok.load_ae(ckpt_path, config=config, device="cuda", dtype="bfloat16")
    print(f"Loaded model: {type(ae)}")
    print(f"Parameters: {sum(p.numel() for p in ae.parameters()):,}")

    print("\n" + "=" * 60)
    print("Test 3: Quick encode/decode test")
    print("=" * 60)

    # Create a simple test image
    img = Image.new('RGB', (256, 256), color='red')
    patches = vitok.preprocess_images(
        img,
        pp="to_tensor|normalize(minus_one_to_one)|patchify(256, 16, 256)",
        device="cuda"
    )
    # Convert patches to model dtype
    patches['patches'] = patches['patches'].to(torch.bfloat16)

    with torch.no_grad():
        encoded = ae.encode(patches)
        decoded = ae.decode(encoded)

    print(f"Input patches: {patches['patches'].shape}")
    print(f"Latent z: {encoded['z'].shape}")
    print(f"Output patches: {decoded['patches'].shape}")

    # Reconstruct image
    recon = vitok.postprocess_images(
        decoded,
        output_format="zero_to_one",
        current_format="minus_one_to_one",
        unpack=True,
        patch=16,
    )
    print(f"Reconstructed image shape: {recon[0].shape}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

    return True


@app.local_entrypoint()
def main():
    result = test_api.remote()
    print(f"\nTest result: {'PASSED' if result else 'FAILED'}")
