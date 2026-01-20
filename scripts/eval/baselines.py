"""Baseline VAE wrappers for evaluation comparison.

Provides unified interface for Flux.2 VAE, SD-VAE, and Qwen-Image VAE.

Usage:
    from scripts.eval.baselines import BaselineVAE

    vae = BaselineVAE("flux", device="cuda", dtype=torch.float16)
    reconstructed = vae.encode_decode(image_tensor)
"""
import torch
from typing import Literal


BASELINE_MODELS = {
    "flux": {
        "repo": "black-forest-labs/FLUX.2-dev",
        "subfolder": "vae",
        "class": "AutoencoderKL",
        "scale": None,
    },
    "sd": {
        "repo": "stabilityai/sd-vae-ft-mse",
        "subfolder": None,
        "class": "AutoencoderKL",
        "scale": 0.18215,
    },
    "qwen": {
        "repo": "REPA-E/e2e-qwenimage-vae",
        "subfolder": None,
        "class": "AutoencoderKLQwenImage",
        "scale": None,
    },
}


class BaselineVAE:
    """Unified interface for baseline VAEs."""

    def __init__(
        self,
        name: Literal["flux", "sd", "qwen"],
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        """Initialize a baseline VAE.

        Args:
            name: Model name - "flux", "sd", or "qwen"
            device: Device to load model on
            dtype: Data type (float16 recommended)
        """
        if name not in BASELINE_MODELS:
            raise ValueError(f"Unknown baseline: {name}. Choose from {list(BASELINE_MODELS.keys())}")

        self.name = name
        self.device = torch.device(device)
        self.dtype = dtype

        config = BASELINE_MODELS[name]
        self.scale = config["scale"]

        # Import and load model
        if config["class"] == "AutoencoderKL":
            from diffusers import AutoencoderKL
            import os
            kwargs = {"torch_dtype": dtype}
            if config["subfolder"]:
                kwargs["subfolder"] = config["subfolder"]
            # Use HF_TOKEN from environment for gated models like Flux
            if os.environ.get("HF_TOKEN"):
                kwargs["token"] = os.environ["HF_TOKEN"]
            self.vae = AutoencoderKL.from_pretrained(config["repo"], **kwargs)
        elif config["class"] == "AutoencoderKLQwenImage":
            from diffusers import AutoencoderKLQwenImage
            self.vae = AutoencoderKLQwenImage.from_pretrained(config["repo"], torch_dtype=dtype)
        else:
            raise ValueError(f"Unknown model class: {config['class']}")

        self.vae = self.vae.to(device).eval()

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image to latents.

        Args:
            image: Input tensor [B, C, H, W] in [0, 1] or [-1, 1] range
                   (model expects [-1, 1], will be converted if needed)

        Returns:
            Latent tensor
        """
        # Normalize to [-1, 1] if needed
        if image.min() >= 0:
            image = image * 2 - 1

        image = image.to(device=self.device, dtype=self.dtype)

        # Qwen expects extra frame dimension
        if self.name == "qwen":
            image = image.unsqueeze(2)

        with torch.no_grad():
            latents = self.vae.encode(image).latent_dist.sample()

        if self.scale is not None:
            latents = latents * self.scale

        if self.name == "qwen":
            latents = latents.squeeze(2)

        return latents

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to image.

        Args:
            latents: Latent tensor

        Returns:
            Reconstructed image [B, C, H, W] in [0, 1] range
        """
        latents = latents.to(device=self.device, dtype=self.dtype)

        # Qwen expects extra frame dimension
        if self.name == "qwen":
            latents = latents.unsqueeze(2)

        if self.scale is not None:
            latents = latents / self.scale

        with torch.no_grad():
            recon = self.vae.decode(latents).sample

        if self.name == "qwen":
            recon = recon.squeeze(2)

        # Convert from [-1, 1] to [0, 1]
        return (recon / 2 + 0.5).clamp(0, 1)

    def encode_decode(self, image: torch.Tensor) -> torch.Tensor:
        """Full reconstruction pass.

        Args:
            image: Input tensor [B, C, H, W] in [0, 1] range

        Returns:
            Reconstructed image [B, C, H, W] in [0, 1] range (same size as input)
        """
        # Normalize to [-1, 1]
        if image.min() >= 0:
            image = image * 2 - 1

        image = image.to(device=self.device, dtype=self.dtype)

        # Store original size for cropping back
        _, _, orig_h, orig_w = image.shape

        # Pad to be divisible by 8 (VAE spatial stride)
        pad_h = (8 - orig_h % 8) % 8
        pad_w = (8 - orig_w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            # Pad on right and bottom (reflect padding for better edge handling)
            image = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h), mode="reflect")

        # Qwen expects extra frame dimension
        if self.name == "qwen":
            image = image.unsqueeze(2)

        with torch.no_grad():
            latents = self.vae.encode(image).latent_dist.sample()

            if self.scale is not None:
                latents = latents * self.scale
                recon = self.vae.decode(latents / self.scale).sample
            else:
                recon = self.vae.decode(latents).sample

        if self.name == "qwen":
            recon = recon.squeeze(2)

        # Crop back to original size
        if pad_h > 0 or pad_w > 0:
            recon = recon[:, :, :orig_h, :orig_w]

        # Convert from [-1, 1] to [0, 1]
        return (recon / 2 + 0.5).clamp(0, 1)

    @property
    def spatial_stride(self) -> int:
        """Get spatial compression factor."""
        # Both Flux and SD use 8x compression
        if self.name in ["flux", "sd"]:
            return 8
        # Qwen uses variable compression
        return 8  # Approximate

    def __repr__(self) -> str:
        return f"BaselineVAE(name={self.name}, device={self.device}, dtype={self.dtype})"


def list_baselines() -> list[str]:
    """List available baseline models."""
    return list(BASELINE_MODELS.keys())


if __name__ == "__main__":
    # Quick test
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list_baselines(), default="sd")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    vae = BaselineVAE(args.model, device=args.device)
    print(f"Loaded: {vae}")

    # Test with dummy image
    dummy = torch.randn(1, 3, 256, 256)
    recon = vae.encode_decode(dummy)
    print(f"Input shape: {dummy.shape}")
    print(f"Output shape: {recon.shape}")
    print(f"Output range: [{recon.min():.3f}, {recon.max():.3f}]")
