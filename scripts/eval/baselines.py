"""Baseline VAE wrappers for evaluation comparison.

Provides unified interface for various production VAEs:
- Flux.2 VAE, SDXL VAE, SD VAE (diffusers AutoencoderKL)
- Qwen-Image VAE (diffusers AutoencoderKLQwenImage)
- DC-AE f32/f64 (diffusers AutoencoderDC)
- NVIDIA Cosmos CI8x8/CI16x16 (cosmos_tokenizer)

Usage:
    from scripts.eval.baselines import BaselineVAE

    vae = BaselineVAE("flux", device="cuda", dtype=torch.float16)
    reconstructed = vae.encode_decode(image_tensor)

Note on dtypes:
- SDXL: Uses madebyollin/sdxl-vae-fp16-fix to avoid NaN in fp16
- Cosmos: Requires bf16 (NVIDIA recommendation for Ampere+ GPUs)
"""
import torch
from typing import Literal


BASELINE_MODELS = {
    "flux": {
        "repo": "black-forest-labs/FLUX.2-dev",
        "subfolder": "vae",
        "class": "AutoencoderKL",
        "scale": None,
        "dtype_override": None,
    },
    "sdxl": {
        # Use fp16-fix version to avoid NaN issues in fp16
        "repo": "madebyollin/sdxl-vae-fp16-fix",
        "subfolder": None,
        "class": "AutoencoderKL",
        "scale": 0.13025,  # SDXL scaling factor
        "dtype_override": None,  # Works with fp16 using this fixed version
    },
    "sd": {
        "repo": "stabilityai/sd-vae-ft-mse",
        "subfolder": None,
        "class": "AutoencoderKL",
        "scale": 0.18215,
        "dtype_override": None,
    },
    "qwen": {
        "repo": "REPA-E/e2e-qwenimage-vae",
        "subfolder": None,
        "class": "AutoencoderKLQwenImage",
        "scale": None,
        "dtype_override": None,
    },
    "dc-ae-f32": {
        "repo": "mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers",
        "subfolder": None,
        "class": "AutoencoderDC",
        "scale": None,
        "dtype_override": None,
    },
    "dc-ae-f64": {
        "repo": "mit-han-lab/dc-ae-f64c128-in-1.0-diffusers",
        "subfolder": None,
        "class": "AutoencoderDC",
        "scale": None,
        "dtype_override": None,
    },
}


class BaselineVAE:
    """Unified interface for baseline VAEs."""

    def __init__(
        self,
        name: Literal["flux", "sdxl", "sd", "qwen", "dc-ae-f32", "dc-ae-f64", "cosmos-ci8x8", "cosmos-ci16x16"],
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float16,
        compile: bool = False,
    ):
        """Initialize a baseline VAE.

        Args:
            name: Model name - "flux", "sdxl", "sd", "qwen", "dc-ae-f32", "dc-ae-f64",
                  "cosmos-ci8x8", or "cosmos-ci16x16"
            device: Device to load model on
            dtype: Data type (float16 recommended, some models override)
            compile: Whether to use torch.compile
        """
        if name not in BASELINE_MODELS:
            raise ValueError(f"Unknown baseline: {name}. Choose from {list(BASELINE_MODELS.keys())}")

        self.name = name
        self.device = torch.device(device)
        self.compiled = compile

        config = BASELINE_MODELS[name]
        self.scale = config["scale"]

        # Check for dtype override (some models require specific dtypes to avoid NaN)
        if config.get("dtype_override") is not None:
            dtype = config["dtype_override"]
            print(f"Note: {name} requires {dtype} for numerical stability")
        self.dtype = dtype

        # Import and load model
        import os
        if config["class"] == "AutoencoderKL":
            from diffusers import AutoencoderKL
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
        elif config["class"] == "AutoencoderDC":
            from diffusers import AutoencoderDC
            self.vae = AutoencoderDC.from_pretrained(config["repo"], torch_dtype=dtype)
        elif config["class"] == "CausalVideoTokenizer":
            # NVIDIA Cosmos tokenizer uses a different API
            # Download model files from HuggingFace and load with cosmos_tokenizer
            from huggingface_hub import hf_hub_download
            from cosmos_tokenizer.image_lib import ImageTokenizer

            # Download encoder and decoder JIT files
            encoder_path = hf_hub_download(
                repo_id=config["repo"],
                filename="encoder.jit",
                token=os.environ.get("HF_TOKEN"),
            )
            decoder_path = hf_hub_download(
                repo_id=config["repo"],
                filename="decoder.jit",
                token=os.environ.get("HF_TOKEN"),
            )

            # Create tokenizer (dtype as string without "torch." prefix)
            dtype_str = str(dtype).replace("torch.", "")
            self.vae = ImageTokenizer(
                checkpoint_enc=encoder_path,
                checkpoint_dec=decoder_path,
                dtype=dtype_str,
            )
            self._is_cosmos = True
        else:
            raise ValueError(f"Unknown model class: {config['class']}")

        self._is_cosmos = config["class"] == "CausalVideoTokenizer"

        # Move to device and set eval mode (Cosmos handles this internally)
        if not self._is_cosmos:
            self.vae = self.vae.to(device).eval()
            if compile:
                # Compile encoder and decoder separately for cleaner graphs
                self.vae.encoder = torch.compile(self.vae.encoder, fullgraph=True)
                self.vae.decoder = torch.compile(self.vae.decoder, fullgraph=True)

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
            enc_out = self.vae.encode(image)
            # DC-AE returns .latent directly, others return .latent_dist
            if self.name.startswith("dc-ae"):
                latents = enc_out.latent
            else:
                latents = enc_out.latent_dist.sample()

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
        # Store original size for cropping back
        _, _, orig_h, orig_w = image.shape

        # Handle Cosmos tokenizer separately (different API)
        if self._is_cosmos:
            image = image.to(device=self.device, dtype=self.dtype)
            # Pad to be divisible by spatial stride
            stride = self.spatial_stride
            pad_h = (stride - orig_h % stride) % stride
            pad_w = (stride - orig_w % stride) % stride
            if pad_h > 0 or pad_w > 0:
                image = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h), mode="reflect")

            with torch.no_grad():
                # Cosmos expects [0, 1] input and returns [0, 1] output
                latents, _ = self.vae.encode(image)
                recon = self.vae.decode(latents)

            # Crop back to original size
            if pad_h > 0 or pad_w > 0:
                recon = recon[:, :, :orig_h, :orig_w]
            return recon.clamp(0, 1)

        # Normalize to [-1, 1]
        if image.min() >= 0:
            image = image * 2 - 1

        image = image.to(device=self.device, dtype=self.dtype)

        # Pad to be divisible by spatial stride
        stride = self.spatial_stride
        pad_h = (stride - orig_h % stride) % stride
        pad_w = (stride - orig_w % stride) % stride
        if pad_h > 0 or pad_w > 0:
            # Pad on right and bottom (reflect padding for better edge handling)
            image = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h), mode="reflect")

        # Qwen expects extra frame dimension
        if self.name == "qwen":
            image = image.unsqueeze(2)

        with torch.no_grad():
            enc_out = self.vae.encode(image)
            # DC-AE returns .latent directly, others return .latent_dist
            if self.name.startswith("dc-ae"):
                latents = enc_out.latent
            else:
                latents = enc_out.latent_dist.sample()

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
        # Flux, SDXL, SD, and Qwen use 8x compression
        if self.name in ["flux", "sdxl", "sd", "qwen", "cosmos-ci8x8"]:
            return 8
        # DC-AE and Cosmos-16x16 use higher compression
        if self.name == "dc-ae-f32":
            return 32
        if self.name == "dc-ae-f64":
            return 64
        if self.name == "cosmos-ci16x16":
            return 16
        return 8  # Default

    def __repr__(self) -> str:
        return f"BaselineVAE(name={self.name}, device={self.device}, dtype={self.dtype})"

    def warmup(self, size: int = 256) -> None:
        """Warmup with a dummy forward pass to trigger compilation.

        Args:
            size: Image size to use for warmup
        """
        if not self.compiled:
            return

        # Run a dummy encode-decode pass
        dummy = torch.randn(1, 3, size, size, device=self.device, dtype=self.dtype)
        _ = self.encode_decode(dummy)


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
