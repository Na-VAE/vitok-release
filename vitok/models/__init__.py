"""ViTok models."""

from vitok.models.ae import AE, decode_variant
from vitok.models.dit import DiT, decode_variant as decode_dit_variant

__all__ = ["AE", "DiT", "decode_variant", "decode_dit_variant"]
