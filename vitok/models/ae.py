"""Vision Transformer Autoencoder."""

import torch
import torch.nn as nn
from typing import Literal, Optional, Tuple, Dict, List, Any
from torch.utils.checkpoint import checkpoint as _checkpoint

from vitok.models.modules.mlp import SwiGLU
from vitok.models.modules.layerscale import LayerScale
from vitok.models.modules.attention import Attention
from vitok.models.modules.norm import RMSNorm, LayerNorm
from vitok.models.modules.rotary_embedding import compute_2d_freqs_cis


def drop_path(
    x: torch.Tensor,
    drop_prob: float = 0.0,
    training: bool = False,
    scale_by_keep: bool = True,
) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - float(drop_prob)
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    if scale_by_keep and keep_prob > 0.0:
        x = x.div(keep_prob)
    return x * random_tensor


class Block(nn.Module):
    """Transformer block with parallel attention + MLP and LayerScale."""

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        use_layer_scale: bool = True,
        layer_scale_init: float = 1e-6,
        drop_path: float = 0.0,
        sliding_window: Optional[int] = None,
        attn_backend: Literal["flash", "sdpa"] = "flash",
    ) -> None:
        super().__init__()
        self.sliding_window = sliding_window
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim=dim, num_heads=num_heads, backend=attn_backend)
        self.ffn = SwiGLU(dim, hidden_dim=ffn_dim)
        self.layer_scale = LayerScale(dim, layer_scale_init) if use_layer_scale else nn.Identity()
        self.drop_path_rate = drop_path

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.norm1(x)
        attn_out = self.attn(h, freqs_cis=freqs_cis, sliding_window=self.sliding_window, attn_mask=attn_mask)
        mlp_out = self.ffn(h)
        combined = self.layer_scale(attn_out + mlp_out)
        return x + drop_path(combined, self.drop_path_rate, self.training)


class AE(nn.Module):
    """Vision Transformer Autoencoder with NaFlex support."""

    def __init__(
        self,
        pixels_per_token=768,
        channels_per_token=32,
        encoder_width=1024,
        decoder_width=1024,
        encoder_depth=4,
        decoder_depth=24,
        encoder_heads=12,
        decoder_heads=12,
        mlp_factor=2.67,
        checkpoint: int = 0,
        spatial_stride: int = 16,
        temporal_stride: int = 1,
        use_layer_scale: bool = True,
        layer_scale_init: float = 1e-4,
        drop_path_rate: float = 0.0,
        encoder: bool = True,
        decoder: bool = True,
        sw: Optional[int] = None,
        attn_backend: Literal["flash", "sdpa"] = "flash",
        **kwargs,
    ):
        super().__init__()

        if not encoder and not decoder:
            raise ValueError("At least one of encoder or decoder must be True")

        sw = sw if (sw is None or sw > 0) else None

        self.pixels_per_token = pixels_per_token
        self.channels_per_token = channels_per_token
        self.rope_theta = 10000.0
        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth
        self.encoder_width = encoder_width
        self.decoder_width = decoder_width
        self.encoder_heads = encoder_heads
        self.decoder_heads = decoder_heads
        self.checkpoint = checkpoint
        self.spatial_stride = spatial_stride
        self.is_encoder = encoder
        self.is_decoder = decoder
        self.sw = sw
        self.attn_backend = attn_backend
        self._quantization_applied = False

        # Encoder
        if encoder:
            self.patch_embed = nn.Linear(pixels_per_token, encoder_width)
            self.to_code = nn.Linear(encoder_width, channels_per_token)
            self.output_fn = LayerNorm(channels_per_token)

            self.encoder_blocks = nn.ModuleList([
                Block(
                    dim=encoder_width,
                    ffn_dim=int(encoder_width * mlp_factor),
                    num_heads=encoder_heads,
                    use_layer_scale=use_layer_scale,
                    layer_scale_init=layer_scale_init,
                    drop_path=0.0,
                    sliding_window=sw,
                    attn_backend=attn_backend,
                )
                for _ in range(encoder_depth)
            ])

        # Decoder
        if decoder:
            self.decoder_embed = nn.Linear(channels_per_token, decoder_width)
            self.to_pixels = nn.Linear(decoder_width, pixels_per_token)

            decoder_dpr = [drop_path_rate * i / max(decoder_depth - 1, 1) for i in range(decoder_depth)]

            self.decoder_blocks = nn.ModuleList([
                Block(
                    dim=decoder_width,
                    ffn_dim=int(decoder_width * mlp_factor),
                    num_heads=decoder_heads,
                    use_layer_scale=use_layer_scale,
                    layer_scale_init=layer_scale_init,
                    drop_path=decoder_dpr[i],
                    sliding_window=sw,
                    attn_backend=attn_backend,
                )
                for i in range(decoder_depth)
            ])

    def _should_checkpoint(self, layer_idx: int) -> bool:
        return self.checkpoint > 0 and self.training and (layer_idx % self.checkpoint == 0)

    def _get_rope_freqs(
        self,
        patch_dict: Dict[str, torch.Tensor],
        head_dim: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        row = patch_dict["row_idx"].to(device=device, dtype=torch.float32)
        col = patch_dict["col_idx"].to(device=device, dtype=torch.float32)
        freqs_cos, freqs_sin = compute_2d_freqs_cis(row, col, head_dim, self.rope_theta)
        return freqs_cos.float(), freqs_sin.float()

    def _get_attn_mask(self, patch_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Generate 2D attention mask from patch mask for SDPA backend.

        Args:
            patch_mask: [B, N] bool tensor where True = valid patch

        Returns:
            [B, 1, N, N] bool tensor for SDPA where True = attend, False = mask out
        """
        if patch_mask is None:
            return None
        # patch_mask: [B, N] -> mask_2d: [B, N, N]
        # Position (i, j) is True if both patch i and patch j are valid
        mask_2d = patch_mask.unsqueeze(2) & patch_mask.unsqueeze(1)  # [B, N, N]
        return mask_2d.unsqueeze(1)  # [B, 1, N, N] for head broadcast

    def encode(self, patch_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Encode patches to latent codes."""
        x = self.patch_embed(patch_dict["patches"])
        freqs_cis = self._get_rope_freqs(
            patch_dict,
            head_dim=self.encoder_width // self.encoder_heads,
            device=x.device,
        )

        # Get attention mask for SDPA backend
        attn_mask = self._get_attn_mask(patch_dict.get("patch_mask")) if self.attn_backend == "sdpa" else None

        for i, block in enumerate(self.encoder_blocks):
            if self._should_checkpoint(i):
                x = _checkpoint(lambda _x: block(_x, freqs_cis=freqs_cis, attn_mask=attn_mask), x, use_reentrant=False)
            else:
                x = block(x, freqs_cis=freqs_cis, attn_mask=attn_mask)

        z = self.output_fn(self.to_code(x))

        return {
            "patch_mask": patch_dict.get("patch_mask"),
            "row_idx": patch_dict["row_idx"],
            "col_idx": patch_dict["col_idx"],
            "orig_height": patch_dict.get("orig_height"),
            "orig_width": patch_dict.get("orig_width"),
            "z": z,
        }

    def decode(self, encode_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Decode latent codes to patches."""
        x = self.decoder_embed(encode_dict["z"])
        freqs_cis = self._get_rope_freqs(
            encode_dict,
            head_dim=self.decoder_width // self.decoder_heads,
            device=x.device,
        )

        # Get attention mask for SDPA backend
        attn_mask = self._get_attn_mask(encode_dict.get("patch_mask")) if self.attn_backend == "sdpa" else None

        for i, block in enumerate(self.decoder_blocks):
            if self._should_checkpoint(i):
                x = _checkpoint(lambda _x: block(_x, freqs_cis=freqs_cis, attn_mask=attn_mask), x, use_reentrant=False)
            else:
                x = block(x, freqs_cis=freqs_cis, attn_mask=attn_mask)

        return {
            "patch_mask": encode_dict.get("patch_mask"),
            "row_idx": encode_dict.get("row_idx"),
            "col_idx": encode_dict.get("col_idx"),
            "orig_height": encode_dict.get("orig_height"),
            "orig_width": encode_dict.get("orig_width"),
            "patches": self.to_pixels(x),
        }

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Full forward pass."""
        if self.is_encoder:
            x = self.encode(x)
        if self.is_decoder:
            x = self.decode(x)
        return x

    def quantize(self) -> "AE":
        """Apply FP8 dynamic quantization for inference."""
        if self._quantization_applied:
            return self

        from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig

        blocks = []
        if self.is_encoder and hasattr(self, "encoder_blocks"):
            blocks.extend(self.encoder_blocks)
        if self.is_decoder and hasattr(self, "decoder_blocks"):
            blocks.extend(self.decoder_blocks)

        for block in blocks:
            quantize_(block, Float8DynamicActivationFloat8WeightConfig())

        self._quantization_applied = True
        return self


def Model(**kw):
    """Factory function for AE model."""
    return AE(**kw)


# --- Variant Parsing ---

_BASE_WIDTHS = {"B": 768, "L": 1024, "G": 1728, "T": 3072, "E": 4096}
_BASE_DEPTHS = {"B": 12, "L": 24, "G": 32, "T": 40, "E": 48}
_BASE_HEADS = {"B": 12, "L": 16, "G": 24, "T": 24, "E": 32}
_BASE_MLP = 2.67


def _parse_variant_name(variant_name: str) -> Dict[str, Any]:
    """Parse a single variant name into config dict."""
    import re

    # Custom underscore format: w{width}_d{depth}_h{heads}[_m{mlp}]
    if variant_name.startswith("w") and "_d" in variant_name and "_h" in variant_name:
        parts = variant_name.split("_")
        width = int(parts[0][1:])
        depth = int(parts[1][1:])
        heads = int(parts[2][1:])
        mlp_factor = float(parts[3][1:]) if len(parts) > 3 and parts[3].startswith("m") else _BASE_MLP
        return {"width": width, "depth": depth, "heads": heads, "mlp_factor": mlp_factor}

    # Inline modifiers
    width_match = re.search(r"w(\d+)", variant_name)
    depth_match = re.search(r"d(\d+)", variant_name)
    heads_match = re.search(r"h(\d+)", variant_name)
    mlp_match = re.search(r"m(\d+(?:\.\d+)?)", variant_name)

    base_name = re.sub(r"w\d+|d\d+|h\d+|m\d+(?:\.\d+)?", "", variant_name)

    if base_name and base_name not in _BASE_WIDTHS:
        raise ValueError(f"Unknown base variant: {base_name}. Available: {list(_BASE_WIDTHS.keys())}")

    return {
        "width": int(width_match.group(1)) if width_match else _BASE_WIDTHS.get(base_name, 768),
        "depth": int(depth_match.group(1)) if depth_match else _BASE_DEPTHS.get(base_name, 12),
        "heads": int(heads_match.group(1)) if heads_match else _BASE_HEADS.get(base_name, 12),
        "mlp_factor": float(mlp_match.group(1)) if mlp_match else _BASE_MLP,
    }


def decode_variant(variant: str) -> Dict[str, Any]:
    """Parse AE variant string like "B/1x16x64" or "Ld2-Ld22/1x16x64"."""
    v, rest = variant.split("/")
    enc_v, dec_v = v.split("-") if "-" in v else (v, v)

    parts = list(map(int, rest.split("x")))
    if len(parts) == 3:
        temporal_stride, spatial_stride, channel_size = parts
    elif len(parts) == 2:
        temporal_stride, spatial_stride, channel_size = 1, parts[0], parts[1]
    else:
        raise ValueError(f"Invalid variant format: {variant}")

    enc_config = _parse_variant_name(enc_v)
    dec_config = _parse_variant_name(dec_v)

    return {
        "encoder_width": enc_config["width"],
        "decoder_width": dec_config["width"],
        "encoder_depth": enc_config["depth"],
        "decoder_depth": dec_config["depth"],
        "encoder_heads": enc_config["heads"],
        "decoder_heads": dec_config["heads"],
        "mlp_factor": max(enc_config["mlp_factor"], dec_config["mlp_factor"]),
        "temporal_stride": temporal_stride,
        "spatial_stride": spatial_stride,
        "channels_per_token": channel_size,
        "pixels_per_token": spatial_stride * spatial_stride * temporal_stride * 3,
    }
