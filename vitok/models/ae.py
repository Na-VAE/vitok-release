"""Vision Transformer Autoencoder."""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List, Any
from torch.utils.checkpoint import checkpoint as _checkpoint

from vitok.models.modules.mlp import SwiGLU
from vitok.models.modules.layerscale import LayerScale
from vitok.models.modules.attention import Attention, create_2d_block_mask
from vitok.models.modules.norm import RMSNorm, LayerNorm
from vitok.models.modules.rotary_embedding import compute_2d_freqs_cis

try:
    from torchao.float8 import convert_to_float8_training
except ImportError:
    convert_to_float8_training = None

def _make_score_mod(attn_mask: torch.Tensor, num_special: int = 0):
    """Create score_mod from attention mask, handling special tokens inline."""
    S = num_special

    def score_mod(score, b, h, q_idx, kv_idx):
        is_special = (q_idx < S) | (kv_idx < S)
        patch_q = q_idx - S
        patch_kv = kv_idx - S
        valid_patch = attn_mask[b, patch_q, patch_kv]
        valid = is_special | valid_patch
        return torch.where(valid, score, torch.full_like(score, float('-inf')))
    return score_mod

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
        num_special_tokens: int = 0,
        use_layer_scale: bool = True,
        layer_scale_init: float = 1e-6,
        drop_path: float = 0.0,
        sliding_window: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.sliding_window = sliding_window

        self.norm1 = RMSNorm(dim)

        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            num_special_tokens=num_special_tokens,
        )
        self.ffn = SwiGLU(dim, hidden_dim=ffn_dim)
        self.layer_scale = LayerScale(dim, layer_scale_init) if use_layer_scale else nn.Identity()
        self.drop_path_rate = drop_path

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        score_mod=None,
        block_mask=None,
    ) -> torch.Tensor:
        h = self.norm1(x)
        attn_out = self.attn(
            h,
            freqs_cis=freqs_cis,
            score_mod=score_mod,
            block_mask=block_mask,
        )
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
        encoder_depth=2,
        decoder_depth=22,
        encoder_heads=12,
        decoder_heads=12,
        mlp_factor=2.67,
        checkpoint: int = 0,
        spatial_stride: int = 16,
        temporal_stride: int = 1,
        class_token: bool = False,
        reg_tokens: int = 0,
        use_layer_scale: bool = True,
        layer_scale_init: float = 1e-4,
        drop_path_rate: float = 0.0,
        float8: bool = False,
        encoder: bool = True,
        decoder: bool = True,
        sw: Optional[int] = None,
        train_seq_len: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()

        if not encoder and not decoder:
            raise ValueError("At least one of encoder or decoder must be True")

        # Hardcoded hyperparameters
        norm_type = 'rmsnorm'
        qk_norm = 'rmsnorm'
        rope_theta = 10000.0
        parallel_mlp_attn = True
        sw_every = 1

        sw = sw if (sw is None or sw > 0) else None

        self.pixels_per_token = pixels_per_token
        self.channels_per_token = channels_per_token
        self.rope_theta = rope_theta
        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth
        self.encoder_width = encoder_width
        self.decoder_width = decoder_width
        self.encoder_heads = encoder_heads
        self.decoder_heads = decoder_heads
        self.checkpoint = checkpoint
        self.spatial_stride = spatial_stride
        self.class_token = class_token
        self.reg_tokens = reg_tokens
        self.encoder = encoder
        self.decoder = decoder
        self.float8 = bool(float8)
        self.sw = sw
        self.sw_every = max(1, sw_every)
        self.train_seq_len = train_seq_len

        self.num_special_tokens = (1 if class_token else 0) + reg_tokens
        self._block_mask = None

        # Initialize encoder components
        if encoder:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_width)) if class_token else None
            self.reg_token = nn.Parameter(torch.zeros(1, reg_tokens, encoder_width)) if reg_tokens > 0 else None

            self.patch_embed = nn.Linear(self.pixels_per_token, self.encoder_width)

            self.to_code = nn.Linear(encoder_width, channels_per_token)

            blocks = []
            for layer_idx in range(encoder_depth):
                sliding_window = None
                if self.sw is not None and ((layer_idx + 1) % self.sw_every == 0):
                    sliding_window = self.sw

                block = Block(
                    dim=encoder_width,
                    ffn_dim=int(encoder_width * mlp_factor),
                    num_heads=encoder_heads,
                    num_special_tokens=self.num_special_tokens,
                    use_layer_scale=use_layer_scale,
                    layer_scale_init=layer_scale_init,
                    drop_path=0.0,
                    sliding_window=sliding_window,
                )
                if self.float8:
                    convert_to_float8_training(block)
                blocks.append(block)
            self.encoder_blocks = nn.ModuleList(blocks)

            self.output_fn = LayerNorm(channels_per_token)

        # Initialize decoder components
        if decoder:
            self.decoder_cls_token = nn.Parameter(torch.zeros(1, 1, decoder_width)) if class_token else None
            self.decoder_reg_token = nn.Parameter(torch.zeros(1, reg_tokens, decoder_width)) if reg_tokens > 0 else None

            self.decoder_embed = nn.Linear(channels_per_token, self.decoder_width)
            self.to_pixels = nn.Linear(decoder_width, pixels_per_token)

            decoder_dpr = [drop_path_rate * i / (decoder_depth - 1) for i in range(decoder_depth)] if decoder_depth > 1 else [0.0]

            blocks = []
            for layer_idx in range(decoder_depth):
                sliding_window = None
                if self.sw is not None and ((layer_idx + 1) % self.sw_every == 0):
                    sliding_window = self.sw

                block = Block(
                    dim=decoder_width,
                    ffn_dim=int(decoder_width * mlp_factor),
                    num_heads=decoder_heads,
                    num_special_tokens=self.num_special_tokens,
                    use_layer_scale=use_layer_scale,
                    layer_scale_init=layer_scale_init,
                    drop_path=decoder_dpr[layer_idx],
                    sliding_window=sliding_window,
                )
                if self.float8:
                    convert_to_float8_training(block)
                blocks.append(block)
            self.decoder_blocks = nn.ModuleList(blocks)

        if self.sw is not None and self.train_seq_len is not None:
            self._block_mask = create_2d_block_mask(
                window=self.sw,
                seq_len=self.train_seq_len,
                num_special=self.num_special_tokens,
                device=torch.device('cuda'),
            )

    def _should_checkpoint(self, layer_idx: int) -> bool:
        return self.checkpoint > 0 and self.training and (layer_idx % self.checkpoint == 0)

    def _get_rope_freqs(
        self,
        patch_dict: Dict[str, torch.Tensor],
        head_dim: int,
        device: torch.device,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Support both old and new key names
        row = patch_dict.get('row_idx', patch_dict.get('yidx'))
        col = patch_dict.get('col_idx', patch_dict.get('xidx'))
        row = row.to(device=device, dtype=torch.float32)
        col = col.to(device=device, dtype=torch.float32)
        freqs_cos, freqs_sin = compute_2d_freqs_cis(row, col, head_dim, self.rope_theta)
        S = self.num_special_tokens
        if S > 0:
            rope_dim = freqs_cos.shape[-1]
            ones = torch.ones(batch_size, S, rope_dim, device=device, dtype=freqs_cos.dtype)
            zeros = torch.zeros_like(ones)
            freqs_cos = torch.cat([ones, freqs_cos], dim=1)
            freqs_sin = torch.cat([zeros, freqs_sin], dim=1)
        return freqs_cos.float(), freqs_sin.float()

    def encode(self, patch_dict: Dict[str, torch.Tensor]):
        """Encode patches to latent codes."""
        if not self.encoder:
            raise RuntimeError("Cannot call encode() on a decoder-only model")

        x = self.patch_embed(patch_dict['patches'])
        B, patch_len, _ = x.shape

        if self.cls_token is not None or self.reg_token is not None:
            tokens = []
            if self.cls_token is not None:
                tokens.append(self.cls_token.expand(B, -1, -1))
            if self.reg_token is not None:
                tokens.append(self.reg_token.expand(B, -1, -1))
            tokens.append(x)
            x = torch.cat(tokens, dim=1)

        freqs_cis = self._get_rope_freqs(
            patch_dict,
            head_dim=self.encoder_width // self.encoder_heads,
            device=x.device, batch_size=B,
        )

        score_mod = None
        attn_mask = patch_dict.get('attention_mask')
        if attn_mask is not None:
            score_mod = _make_score_mod(attn_mask, self.num_special_tokens)

        for i, block in enumerate(self.encoder_blocks):
            if self._should_checkpoint(i):
                x = _checkpoint(lambda _x: block(_x, freqs_cis=freqs_cis, score_mod=score_mod, block_mask=self._block_mask), x, use_reentrant=False)
            else:
                x = block(x, freqs_cis=freqs_cis, score_mod=score_mod, block_mask=self._block_mask)

        if self.num_special_tokens > 0:
            x = x[:, self.num_special_tokens:]

        z = self.to_code(x)
        # Apply output normalization (LayerNorm on latent codes)
        z = self.output_fn(z)

        # Build output with new key names (support old names for reading)
        mask = patch_dict.get('patch_mask', patch_dict.get('ptype'))
        row = patch_dict.get('row_idx', patch_dict.get('yidx'))
        col = patch_dict.get('col_idx', patch_dict.get('xidx'))
        orig_h = patch_dict.get('orig_height', patch_dict.get('original_height'))
        orig_w = patch_dict.get('orig_width', patch_dict.get('original_width'))

        encode_dict = {
            'patch_mask': mask,
            'row_idx': row,
            'col_idx': col,
            'orig_height': orig_h,
            'orig_width': orig_w,
            'z': z,
        }
        if 'attention_mask' in patch_dict:
            encode_dict['attention_mask'] = patch_dict['attention_mask']
        return encode_dict

    def decode(self, encode_dict: Dict[str, torch.Tensor], max_grid_size=None):
        """Decode latent codes to patches."""
        if not self.decoder:
            raise RuntimeError("Cannot call decode() on an encoder-only model")

        x = self.decoder_embed(encode_dict['z'])
        B, _, _ = x.shape

        if self.decoder_cls_token is not None or self.decoder_reg_token is not None:
            tokens = []
            if self.decoder_cls_token is not None:
                tokens.append(self.decoder_cls_token.expand(B, -1, -1))
            if self.decoder_reg_token is not None:
                tokens.append(self.decoder_reg_token.expand(B, -1, -1))
            tokens.append(x)
            x = torch.cat(tokens, dim=1)

        freqs_cis = self._get_rope_freqs(
            encode_dict,
            head_dim=self.decoder_width // self.decoder_heads,
            device=x.device, batch_size=B,
        )

        score_mod = None
        attn_mask = encode_dict.get('attention_mask')
        if attn_mask is not None:
            score_mod = _make_score_mod(attn_mask, self.num_special_tokens)

        for i, block in enumerate(self.decoder_blocks):
            if self._should_checkpoint(i):
                x = _checkpoint(lambda _x: block(_x, freqs_cis=freqs_cis, score_mod=score_mod, block_mask=self._block_mask), x, use_reentrant=False)
            else:
                x = block(x, freqs_cis=freqs_cis, score_mod=score_mod, block_mask=self._block_mask)

        if self.num_special_tokens > 0:
            x = x[:, self.num_special_tokens:]

        pred_patches = self.to_pixels(x)

        # Build output with new key names (support old names for reading)
        mask = encode_dict.get('patch_mask', encode_dict.get('ptype'))
        row = encode_dict.get('row_idx', encode_dict.get('yidx'))
        col = encode_dict.get('col_idx', encode_dict.get('xidx'))
        orig_h = encode_dict.get('orig_height', encode_dict.get('original_height'))
        orig_w = encode_dict.get('orig_width', encode_dict.get('original_width'))

        decode_dict = {
            'patch_mask': mask,
            'row_idx': row,
            'col_idx': col,
            'patches': pred_patches,
        }
        if orig_h is not None:
            decode_dict['orig_height'] = orig_h
        if orig_w is not None:
            decode_dict['orig_width'] = orig_w
        if 'attention_mask' in encode_dict:
            decode_dict['attention_mask'] = encode_dict['attention_mask']
        return decode_dict

    def forward(self, x: Dict[str, torch.Tensor]):
        """Full forward pass: encode then decode.

        If encoder+decoder: returns decoded patches dict
        If encoder-only: returns encode dict with 'z'
        If decoder-only: expects dict with 'z', returns patches dict
        """
        if self.encoder:
            x = self.encode(x)
        if self.decoder:
            x = self.decode(x)
        return x

    def get_encoder_decoder_param_groups(self) -> Tuple[List[nn.Parameter], List[nn.Parameter], List[nn.Parameter], List[nn.Parameter]]:
        """Return (enc_decay, enc_no_decay, dec_decay, dec_no_decay) parameter groups."""
        enc_modules: List[nn.Module] = []
        dec_modules: List[nn.Module] = []

        for name in ("patch_embed", "encoder_blocks", "to_code", "output_fn"):
            m = getattr(self, name, None)
            if m is not None:
                enc_modules.append(m)
        for name in ("decoder_embed", "decoder_blocks", "to_pixels"):
            m = getattr(self, name, None)
            if m is not None:
                dec_modules.append(m)

        enc_params = set()
        for mod in enc_modules:
            for p in mod.parameters(recurse=True):
                enc_params.add(p)
        if getattr(self, "cls_token", None) is not None:
            enc_params.add(self.cls_token)
        if getattr(self, "reg_token", None) is not None:
            enc_params.add(self.reg_token)

        dec_params = set()
        for mod in dec_modules:
            for p in mod.parameters(recurse=True):
                dec_params.add(p)
        if getattr(self, "decoder_cls_token", None) is not None:
            dec_params.add(self.decoder_cls_token)
        if getattr(self, "decoder_reg_token", None) is not None:
            dec_params.add(self.decoder_reg_token)

        try:
            named_iter = self.named_parameters(remove_duplicate=True)
        except TypeError:
            named_iter = self.named_parameters()

        enc_decay: List[nn.Parameter] = []
        enc_no: List[nn.Parameter] = []
        dec_decay: List[nn.Parameter] = []
        dec_no: List[nn.Parameter] = []

        for name, param in named_iter:
            if not param.requires_grad:
                continue
            lname = name.lower()
            is_weight = name.endswith(".weight")
            is_normish = ("norm" in lname) or (".bn" in lname)
            is_embedish = any(k in lname for k in ("embedding", "pos_embed", "position", "relative_position", "token"))
            use_decay = is_weight and not (is_normish or is_embedish)

            if param in enc_params:
                (enc_decay if use_decay else enc_no).append(param)
            elif param in dec_params:
                (dec_decay if use_decay else dec_no).append(param)
            else:
                dec_no.append(param)

        return enc_decay, enc_no, dec_decay, dec_no


def Model(**kw):
    """Factory function for AE model."""
    return AE(**kw)


# --- Variant Parsing ---

# Base presets for model architectures
_BASE_WIDTHS = {"B": 768, "L": 1024, "G": 1728, "T": 3072, "E": 4096}
_BASE_DEPTHS = {"B": 12, "L": 24, "G": 32, "T": 40, "E": 48}
_BASE_HEADS = {"B": 12, "L": 16, "G": 24, "T": 24, "E": 32}
_BASE_MLP = 2.67


def _parse_variant_name(variant_name: str) -> Dict[str, Any]:
    """Parse a single variant name into config dict.

    Supports:
      - Base names: B, L, G, T, E
      - Inline modifiers: w{width} d{depth} h{heads} m{mlp_factor}
        e.g., "Lw1280", "Ld32h20", "Gm3.28"
      - Underscore format: w{width}_d{depth}_h{heads}[_m{mlp}]
        e.g., "w4096_d40_h32_m2.0"
    """
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

    # Remove modifiers to get base name
    base_name = re.sub(r"w\d+|d\d+|h\d+|m\d+(?:\.\d+)?", "", variant_name)

    if base_name and base_name not in _BASE_WIDTHS:
        raise ValueError(f"Unknown base variant: {base_name}. Available: {list(_BASE_WIDTHS.keys())}")

    width = int(width_match.group(1)) if width_match else _BASE_WIDTHS.get(base_name, 768)
    depth = int(depth_match.group(1)) if depth_match else _BASE_DEPTHS.get(base_name, 12)
    heads = int(heads_match.group(1)) if heads_match else _BASE_HEADS.get(base_name, 12)
    mlp_factor = float(mlp_match.group(1)) if mlp_match else _BASE_MLP

    return {"width": width, "depth": depth, "heads": heads, "mlp_factor": mlp_factor}


def decode_variant(variant: str) -> Dict[str, Any]:
    """Parse AE variant string like "B/1x16x64" or "Ld2-Ld22/1x16x64".

    Format: {encoder}[-{decoder}]/{temporal}x{spatial}x{channels}

    Examples:
        - "B/1x16x64": Base encoder/decoder, stride 16, 64 channels
        - "Ld2-Ld22/1x16x64": 2-layer Large encoder, 22-layer Large decoder
        - "w4096_d40_h32/1x16x64": Custom width/depth/heads

    Returns:
        Dict ready to unpack into AE constructor:
        encoder_width, decoder_width, encoder_depth, decoder_depth,
        encoder_heads, decoder_heads, mlp_factor, spatial_stride, temporal_stride,
        channels_per_token, pixels_per_token
    """
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
