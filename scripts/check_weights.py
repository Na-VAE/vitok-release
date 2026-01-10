"""Check weight keys for T-32x64 model."""
import modal

app = modal.App("weight-check")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "safetensors", "huggingface_hub"
)

@app.function(image=image, secrets=[modal.Secret.from_name("huggingface-secret")])
def check_weights():
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download

    # Download T-32x64 weights
    path = hf_hub_download('philippehansen/ViTok-T-32x64', 'model.safetensors')
    weights = load_file(path)

    print('=== Weight keys (first 40) ===')
    for k in sorted(weights.keys())[:40]:
        print(f'  {k}: {list(weights[k].shape)}')
    print(f'  ... ({len(weights)} total keys)')

    # Check for any keys with unexpected prefixes
    prefixes = set()
    for k in weights.keys():
        prefix = k.split('.')[0]
        prefixes.add(prefix)
    print(f'\n=== Key prefixes: {sorted(prefixes)} ===')

    # Check output_fn weights specifically
    output_fn_keys = [k for k in weights.keys() if 'output_fn' in k]
    print(f'\n=== output_fn keys: {output_fn_keys} ===')

    # Check to_code and to_pixels
    to_code_keys = [k for k in weights.keys() if 'to_code' in k]
    to_pixels_keys = [k for k in weights.keys() if 'to_pixels' in k]
    print(f'=== to_code keys: {to_code_keys} ===')
    print(f'=== to_pixels keys: {to_pixels_keys} ===')

    # Check decoder_embed
    dec_embed_keys = [k for k in weights.keys() if 'decoder_embed' in k]
    print(f'=== decoder_embed keys: {dec_embed_keys} ===')

@app.local_entrypoint()
def main():
    check_weights.remote()
