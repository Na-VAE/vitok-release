"""ViTok Gradio Demo - Interactive image reconstruction visualization.

Run locally:
    cd demo && gradio app.py

Deploy to HuggingFace Spaces:
    Copy this directory to a new Space with Gradio SDK.
"""

import functools
import io
import sys
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Add parent to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent))

from vitok import AE, decode_variant, preprocess, postprocess
from vitok.pretrained import download_pretrained, get_pretrained_info, PRETRAINED_ALIASES

# Available models
MODELS = list(PRETRAINED_ALIASES.keys())

# Device setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32


@functools.lru_cache(maxsize=6)
def load_model(model_name: str):
    """Load and cache a ViTok model."""
    print(f"Loading model: {model_name}")

    # Get model info and download weights
    _, _, variant = get_pretrained_info(model_name)
    weights_paths = download_pretrained(model_name)
    weights = {}
    for path in weights_paths:
        weights.update(load_file(path))

    # Create full model (encoder + decoder)
    model = AE(**decode_variant(variant))
    model.load_state_dict(weights)
    model.to(device=DEVICE, dtype=DTYPE)
    model.eval()

    # Parse spatial stride from variant (e.g., "Ld4-Ld24/1x16x64" -> 16)
    spatial_stride = int(variant.split("/")[1].split("x")[1])

    return model, variant, spatial_stride


def compute_diff_heatmap(original: np.ndarray, reconstruction: np.ndarray) -> np.ndarray:
    """Generate a heatmap showing reconstruction differences.

    Args:
        original: Original image as numpy array (H, W, C) in range [0, 255]
        reconstruction: Reconstructed image as numpy array (H, W, C) in range [0, 255]

    Returns:
        Heatmap as numpy array (H, W, C) in RGB format
    """
    # Ensure same size
    if original.shape != reconstruction.shape:
        original = cv2.resize(original, (reconstruction.shape[1], reconstruction.shape[0]))

    # Compute absolute difference per channel, then mean across channels
    diff = np.abs(original.astype(np.float32) - reconstruction.astype(np.float32))
    diff_gray = np.mean(diff, axis=2)

    # Normalize to 0-255 (with some headroom for visibility)
    max_diff = max(diff_gray.max(), 1.0)
    diff_normalized = (diff_gray / max_diff * 255).astype(np.uint8)

    # Apply colormap (COLORMAP_JET: blue=low error, red=high error)
    heatmap_bgr = cv2.applyColorMap(diff_normalized, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    return heatmap_rgb


def compute_metrics(original: np.ndarray, reconstruction: np.ndarray) -> dict:
    """Compute image quality metrics.

    Args:
        original: Original image as numpy array (H, W, C) in range [0, 255]
        reconstruction: Reconstructed image as numpy array (H, W, C) in range [0, 255]

    Returns:
        Dictionary with SSIM and PSNR values
    """
    # Ensure same size for metrics
    if original.shape != reconstruction.shape:
        original = cv2.resize(original, (reconstruction.shape[1], reconstruction.shape[0]))

    # PSNR (higher is better)
    psnr = peak_signal_noise_ratio(original, reconstruction, data_range=255)

    # SSIM (higher is better, range 0-1)
    # Use channel_axis for color images
    ssim = structural_similarity(
        original, reconstruction,
        channel_axis=2,
        data_range=255
    )

    return {"ssim": ssim, "psnr": psnr}


def reconstruct_image(image: Image.Image, model_name: str):
    """Encode and decode an image, returning reconstruction and analysis.

    Args:
        image: Input PIL Image
        model_name: Name of the model to use (e.g., "L-64")

    Returns:
        Tuple of (original, reconstruction, diff_heatmap, metrics_text, latent_info)
    """
    if image is None:
        return None, None, None, "Please upload an image", ""

    # Load model
    model, variant, spatial_stride = load_model(model_name)

    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Store original for display
    original_np = np.array(image)

    # Preprocess
    pp_string = f"to_tensor|normalize(minus_one_to_one)|patchify({spatial_stride}, 256)"
    patch_dict = preprocess(image, pp=pp_string, device=DEVICE)

    # Cast to model dtype
    patch_dict = {
        k: v.to(DTYPE) if isinstance(v, torch.Tensor) and v.is_floating_point() else v
        for k, v in patch_dict.items()
    }

    # Encode and decode
    with torch.no_grad():
        encoded = model.encode(patch_dict)
        decoded = model.decode(encoded)

    # Get latent info
    z = encoded["z"]
    latent_shape = list(z.shape)

    # Postprocess to image
    images = postprocess(decoded, output_format="0_255", do_unpack=True, patch=spatial_stride)
    recon_np = images[0].permute(1, 2, 0).cpu().numpy()

    # Compute diff heatmap
    diff_heatmap = compute_diff_heatmap(original_np, recon_np)

    # Compute metrics
    metrics = compute_metrics(original_np, recon_np)

    # Format metrics text
    metrics_text = f"SSIM: {metrics['ssim']:.4f}  |  PSNR: {metrics['psnr']:.2f} dB"

    # Format latent info
    latent_info = (
        f"Model: {model_name} ({variant})\n"
        f"Input: {original_np.shape[1]}x{original_np.shape[0]}\n"
        f"Output: {recon_np.shape[1]}x{recon_np.shape[0]}\n"
        f"Latent: {latent_shape} (B, N, C)\n"
        f"Compression: {original_np.shape[0]*original_np.shape[1]*3} -> {latent_shape[1]*latent_shape[2]} "
        f"({original_np.shape[0]*original_np.shape[1]*3 / (latent_shape[1]*latent_shape[2]):.1f}x)"
    )

    return original_np, recon_np, diff_heatmap, metrics_text, latent_info


# Build Gradio interface
with gr.Blocks(
    title="ViTok Image Tokenizer Demo",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        """
        # ViTok: Vision Transformer Image Tokenizer

        Upload an image to see how ViTok encodes and reconstructs it.
        The difference heatmap shows reconstruction error (blue=low, red=high).

        **Models:** L (Large, 1.1B params) and T (Tiny) variants with different latent channel sizes.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                type="pil",
                label="Upload Image",
                sources=["upload", "clipboard"],
            )
            model_dropdown = gr.Dropdown(
                choices=MODELS,
                value="L-64",
                label="Model",
                info="L=Large (1.1B), T=Tiny. Number=latent channels."
            )
            submit_btn = gr.Button("Reconstruct", variant="primary", size="lg")

            metrics_output = gr.Textbox(
                label="Quality Metrics",
                interactive=False,
                placeholder="Metrics will appear here..."
            )
            latent_output = gr.Textbox(
                label="Latent Info",
                interactive=False,
                lines=5,
                placeholder="Latent details will appear here..."
            )

        with gr.Column(scale=2):
            with gr.Row():
                original_display = gr.Image(label="Original", show_download_button=False)
                recon_display = gr.Image(label="Reconstruction", show_download_button=True)
                diff_display = gr.Image(label="Difference Heatmap", show_download_button=True)

    # Examples
    example_dir = Path(__file__).parent / "examples"
    if example_dir.exists():
        example_images = list(example_dir.glob("*.jpg")) + list(example_dir.glob("*.png"))
        if example_images:
            gr.Examples(
                examples=[[str(img), "L-64"] for img in example_images[:4]],
                inputs=[input_image, model_dropdown],
                outputs=[original_display, recon_display, diff_display, metrics_output, latent_output],
                fn=reconstruct_image,
                cache_examples=False,
            )

    # Connect button
    submit_btn.click(
        fn=reconstruct_image,
        inputs=[input_image, model_dropdown],
        outputs=[original_display, recon_display, diff_display, metrics_output, latent_output],
    )

    # Also trigger on image upload
    input_image.change(
        fn=reconstruct_image,
        inputs=[input_image, model_dropdown],
        outputs=[original_display, recon_display, diff_display, metrics_output, latent_output],
    )

    gr.Markdown(
        """
        ---
        **Links:** [GitHub](https://github.com/Na-VAE/vitok-release) |
        [Paper](https://arxiv.org/abs/2501.09755) |
        [Models](https://huggingface.co/Na-VAE)

        **Metrics:** SSIM (Structural Similarity, 0-1, higher=better) |
        PSNR (Peak Signal-to-Noise Ratio, dB, higher=better)
        """
    )


if __name__ == "__main__":
    demo.launch()
