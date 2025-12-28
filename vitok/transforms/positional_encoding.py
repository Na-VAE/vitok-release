"""2D Positional Encoding utilities."""

import torch
import torch.nn.functional as F


def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_h: int,
    grid_w: int,
    temperature: float = 10000,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Generate 2-D sine/cosine positional embeddings.

    Returns a tensor of shape `(grid_h*grid_w, embed_dim)` laid out in
    row-major (y-major) order.
    """
    assert embed_dim % 4 == 0, "embed_dim must be a multiple of 4 for 2-D sin-cos PE"

    y, x = torch.meshgrid(
        torch.arange(grid_h, dtype=dtype, device=device),
        torch.arange(grid_w, dtype=dtype, device=device),
        indexing="ij",
    )
    grid = torch.stack((y, x), dim=0).reshape(2, -1)

    omega = torch.arange(embed_dim // 4, dtype=dtype, device=device) / (embed_dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    pos_y = (grid[0].unsqueeze(1) * omega).view(-1, embed_dim // 4)
    pos_x = (grid[1].unsqueeze(1) * omega).view(-1, embed_dim // 4)

    emb = torch.cat([
        torch.sin(pos_x), torch.cos(pos_x),
        torch.sin(pos_y), torch.cos(pos_y)
    ], dim=1)

    return emb.to(dtype=dtype, device=device)


def create_fixed_2d_table(
    embed_dim: int,
    grid_h: int,
    grid_w: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return a (1, embed_dim, grid_h, grid_w) fixed 2-D sin-cos positional table."""
    table_flat = get_2d_sincos_pos_embed(embed_dim, grid_h, grid_w, dtype=dtype, device=device)
    table = table_flat.reshape(grid_h, grid_w, embed_dim).unsqueeze(0)
    table = table.permute(0, 3, 1, 2).float()
    return table


def _make_pe_from_fixed_table(
    grid_h: int,
    grid_w: int,
    fixed_table: torch.Tensor,
    ar_preserving: bool = True,
    interp_mode: str = "bilinear"
) -> torch.Tensor:
    """Generate positional embedding for given grid size using interpolation."""
    table_nchw = fixed_table
    _, C, H0, W0 = table_nchw.shape
    dtype = table_nchw.dtype

    if (grid_h == H0) and (grid_w == W0):
        pe_flat = fixed_table.reshape(1, -1, C)
    else:
        target_size = (max(grid_h, grid_w),) * 2 if ar_preserving else (grid_h, grid_w)
        pe = F.interpolate(
            table_nchw,
            size=target_size,
            mode=interp_mode,
            align_corners=False,
            antialias=True,
        )[:, :, :grid_h, :grid_w]
        pe_flat = pe.flatten(2).transpose(1, 2)
    return pe_flat.squeeze(0).to(dtype=dtype)


class PositionalEncoding2D:
    """Simple positional encoding that computes on-demand from a fixed table."""

    def __init__(
        self,
        embed_dim: int,
        max_grid_size: int,
        temperature: float = 10000,
        dtype: torch.dtype = torch.float32,
        max_seq_len: int | None = None,
        cache_size: int = 4096
    ):
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.max_grid_size = max_grid_size
        self.temperature = temperature
        self.dtype = dtype
        self.cache_size = cache_size
        self._cache = {}

        self.fixed_table = create_fixed_2d_table(
            embed_dim,
            max_grid_size,
            max_grid_size,
            dtype=dtype,
            device='cpu'
        )

    def __call__(self, patch_dict: dict) -> torch.Tensor:
        """Compute positional encoding for a single example with caching."""
        H = patch_dict["grid_h"]
        W = patch_dict["grid_w"]
        patches = patch_dict['patches']
        seq_len, _ = patches.shape

        h = int(H.item())
        w = int(W.item())

        cache_key = (h, w)
        if cache_key in self._cache:
            pos_embed = self._cache[cache_key].clone()
        else:
            pos_embed = _make_pe_from_fixed_table(
                h, w, self.fixed_table,
                ar_preserving=True,
                interp_mode='bilinear'
            )

            if len(self._cache) < self.cache_size:
                self._cache[cache_key] = pos_embed.clone()
            else:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._cache[cache_key] = pos_embed.clone()

        if pos_embed.shape[0] > seq_len:
            pos_embed = pos_embed[:seq_len]
        elif pos_embed.shape[0] < seq_len:
            padded = torch.zeros(seq_len, self.embed_dim, dtype=self.dtype, device='cpu')
            padded[:pos_embed.shape[0]] = pos_embed
            pos_embed = padded

        return pos_embed.to(dtype=patches.dtype)
