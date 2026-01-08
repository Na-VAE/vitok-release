"""MetricCalculator for computing image quality metrics.

Computes FID, FDD, SSIM, PSNR for VAE reconstruction evaluation.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from typing import List, Tuple, Optional
from scipy import linalg

# FID Inception
from pytorch_fid.inception import InceptionV3 as PytorchFIDInceptionV3

# DINO for FDD
from dino_perceptual import DINOModel

# Torchmetrics for SSIM/PSNR
try:
    from torchmetrics.image import (
        peak_signal_noise_ratio as PSNR,
        structural_similarity_index_measure as SSIM,
    )
except ImportError:
    from torchmetrics.functional import (
        peak_signal_noise_ratio as PSNR,
        structural_similarity_index_measure as SSIM,
    )


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Calculate Frechet Distance between two Gaussians."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def distributed_mean_cov(features: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and covariance across distributed ranks."""
    features = features.to(dtype=torch.float64)
    n_local = torch.tensor(float(features.shape[0]), device=features.device, dtype=torch.float64)
    sum_local = features.sum(dim=0)

    if dist.is_initialized():
        dist.all_reduce(n_local, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_local, op=dist.ReduceOp.SUM)

    N = n_local.item()
    mu = (sum_local / N).cpu().numpy()

    # Compute covariance
    centered = features - torch.from_numpy(mu).to(features.device)
    cov_local = centered.T @ centered

    if dist.is_initialized():
        dist.all_reduce(cov_local, op=dist.ReduceOp.SUM)

    cov = (cov_local / (N - 1)).cpu().numpy()
    return mu, cov


def dist_mean_1d(values: torch.Tensor) -> float:
    """Distributed mean of 1D values."""
    if values.numel() == 0:
        sum_val = torch.tensor(0.0, dtype=torch.float64)
        count = torch.tensor(0.0, dtype=torch.float64)
    else:
        values = values.to(dtype=torch.float64)
        sum_val = values.sum()
        count = torch.tensor(float(values.numel()), dtype=torch.float64)

    if dist.is_initialized():
        sum_val = sum_val.to(values.device if values.numel() > 0 else 'cuda')
        count = count.to(sum_val.device)
        dist.all_reduce(sum_val, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)

    return float(sum_val / count) if count > 0 else 0.0


def compute_ssim(preds, target, data_range=2.0, max_kernel_size: int = 11):
    """Compute SSIM with adaptive kernel size for small images."""
    if preds.dim() == 3:
        preds = preds.unsqueeze(0)
    if target.dim() == 3:
        target = target.unsqueeze(0)

    preds = preds.float()
    target = target.float()

    h, w = preds.shape[-2], preds.shape[-1]
    k = int(min(h, w, max_kernel_size))
    if k % 2 == 0:
        k = max(1, k - 1)

    try:
        return SSIM(preds=preds, target=target, data_range=data_range, kernel_size=k)
    except RuntimeError:
        return SSIM(preds=preds, target=target, data_range=data_range, kernel_size=1)


def compute_psnr(preds, target, data_range=(-1.0, 1.0)):
    """Compute PSNR between predicted and target images."""
    return PSNR(preds=preds, target=target, data_range=data_range)


class MetricCalculator:
    """Unified calculator for image generation/reconstruction metrics.

    Supports: FID, FDD, SSIM, PSNR

    Example:
        calc = MetricCalculator(metrics=('fid', 'fdd', 'ssim', 'psnr'))
        calc.move_model_to_device('cuda')
        calc.update(real_images, generated_images)
        stats = calc.gather()
        print(stats)  # {'fid': 8.5, 'fdd': 12.3, 'ssim': 0.92, 'psnr': 28.4}
    """

    def __init__(self, metrics: Tuple[str, ...] = ('fid', 'fdd', 'ssim', 'psnr')):
        self.metrics = metrics

        # Initialize models based on requested metrics
        if 'fid' in metrics:
            self.fid_inception = PytorchFIDInceptionV3(
                output_blocks=(PytorchFIDInceptionV3.BLOCK_INDEX_BY_DIM[2048],),
                resize_input=False,
                normalize_input=False,
                requires_grad=False,
            ).to('cpu')
            self.fid_inception.eval()

        if 'fdd' in metrics:
            self.dino_model = DINOModel(model_size='B', target_size=512).to('cpu')
            self.dino_model.eval()

        self.reset()

    def reset(self):
        """Reset all accumulators."""
        self.inception_real = []
        self.inception_fake = []
        self.dino_real = []
        self.dino_fake = []
        self.ssim_values = []
        self.psnr_values = []

    def move_model_to_device(self, device, dtype=None):
        """Move feature extraction models to device and optionally set dtype.

        Note: FID Inception stays in float32 as it expects float32 inputs.
        DINO can use bf16 for memory efficiency.
        """
        if 'fid' in self.metrics:
            self.fid_inception = self.fid_inception.to(device)
            # Keep FID Inception in float32 - it expects float32 inputs
        if 'fdd' in self.metrics:
            self.dino_model = self.dino_model.to(device)
            if dtype is not None:
                self.dino_model = self.dino_model.to(dtype)

    @torch.no_grad()
    def update(self, real: List[torch.Tensor], generated: List[torch.Tensor]):
        """Update metrics from lists of images.

        Args:
            real: List of real images (C,H,W) or (B,C,H,W) in [-1, 1]
            generated: List of generated images in [-1, 1]
        """
        # Flatten to per-image tensors
        def to_image_list(x):
            out = []
            if isinstance(x, (list, tuple)):
                for t in x:
                    if t.dim() == 4:
                        out.extend([t[i] for i in range(t.shape[0])])
                    elif t.dim() == 3:
                        out.append(t)
            elif isinstance(x, torch.Tensor):
                if x.dim() == 4:
                    out.extend([x[i] for i in range(x.shape[0])])
                elif x.dim() == 3:
                    out.append(x)
            return out

        reals = to_image_list(real)
        gens = to_image_list(generated)

        if len(reals) != len(gens):
            raise ValueError(f"Mismatched lengths: {len(reals)} vs {len(gens)}")

        n = len(reals)
        if n == 0:
            return

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Perceptual metrics on original sizes
        if 'ssim' in self.metrics or 'psnr' in self.metrics:
            for r_img, g_img in zip(reals, gens):
                r_b = r_img.to(dtype=torch.float32, device=device).unsqueeze(0)
                g_b = g_img.to(dtype=torch.float32, device=device).unsqueeze(0)
                if 'ssim' in self.metrics:
                    self.ssim_values.append(compute_ssim(g_b, r_b).unsqueeze(0))
                if 'psnr' in self.metrics:
                    self.psnr_values.append(compute_psnr(g_b, r_b).unsqueeze(0))

        # FID: resize to 299x299, extract Inception features
        if 'fid' in self.metrics:
            r299 = [TF.resize(r.to(device=device, dtype=torch.float32), [299, 299],
                             interpolation=InterpolationMode.BICUBIC, antialias=True) for r in reals]
            g299 = [TF.resize(g.to(device=device, dtype=torch.float32), [299, 299],
                             interpolation=InterpolationMode.BICUBIC, antialias=True) for g in gens]

            chunk = 64
            for i in range(0, n, chunk):
                rb = torch.stack(r299[i:i+chunk], dim=0).clamp_(-1.0, 1.0)
                gb = torch.stack(g299[i:i+chunk], dim=0).clamp_(-1.0, 1.0)
                # Convert to [0, 1] for FID Inception
                rb_fid = ((rb + 1.0) / 2.0).clamp_(0.0, 1.0)
                gb_fid = ((gb + 1.0) / 2.0).clamp_(0.0, 1.0)

                real_feat = self.fid_inception(rb_fid)[0]
                fake_feat = self.fid_inception(gb_fid)[0]
                if real_feat.dim() == 4:
                    real_feat = torch.flatten(real_feat, 1)
                if fake_feat.dim() == 4:
                    fake_feat = torch.flatten(fake_feat, 1)

                self.inception_real.append(real_feat)
                self.inception_fake.append(fake_feat)

        # FDD: resize to 512x512, extract DINO features
        if 'fdd' in self.metrics:
            dino_dtype = next(self.dino_model.parameters()).dtype
            r512 = [TF.resize(r.to(device=device, dtype=dino_dtype), [512, 512],
                             interpolation=InterpolationMode.BICUBIC, antialias=True) for r in reals]
            g512 = [TF.resize(g.to(device=device, dtype=dino_dtype), [512, 512],
                             interpolation=InterpolationMode.BICUBIC, antialias=True) for g in gens]

            chunk = 64
            for i in range(0, n, chunk):
                rb = torch.stack(r512[i:i+chunk], dim=0).clamp_(-1.0, 1.0)
                gb = torch.stack(g512[i:i+chunk], dim=0).clamp_(-1.0, 1.0)

                real_d, _ = self.dino_model(rb)
                fake_d, _ = self.dino_model(gb)

                self.dino_real.append(real_d)
                self.dino_fake.append(fake_d)

    def gather(self) -> dict:
        """Gather and compute final metrics across all processes.

        Returns:
            Dictionary of computed metrics
        """
        stats = {}

        # FID
        if 'fid' in self.metrics and self.inception_real:
            inc_real = torch.cat(self.inception_real, dim=0)
            inc_fake = torch.cat(self.inception_fake, dim=0)
            mu_r, sigma_r = distributed_mean_cov(inc_real)
            mu_f, sigma_f = distributed_mean_cov(inc_fake)
            stats['fid'] = calculate_frechet_distance(mu_r, sigma_r, mu_f, sigma_f)

        # FDD
        if 'fdd' in self.metrics and self.dino_real:
            dino_real = torch.cat(self.dino_real, dim=0)
            dino_fake = torch.cat(self.dino_fake, dim=0)
            mu_r, sigma_r = distributed_mean_cov(dino_real)
            mu_f, sigma_f = distributed_mean_cov(dino_fake)
            stats['fdd'] = calculate_frechet_distance(mu_r, sigma_r, mu_f, sigma_f)

        # SSIM
        if 'ssim' in self.metrics and self.ssim_values:
            ssim_all = torch.cat(self.ssim_values, dim=0)
            stats['ssim'] = dist_mean_1d(ssim_all)

        # PSNR
        if 'psnr' in self.metrics and self.psnr_values:
            psnr_all = torch.cat(self.psnr_values, dim=0)
            stats['psnr'] = dist_mean_1d(psnr_all)

        return stats
