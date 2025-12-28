"""Tests for example helper functions."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
from PIL import Image


_EXAMPLE_CACHE = {}


def load_example_module(name: str):
    """Load an example module by filename without requiring a package."""
    if name in _EXAMPLE_CACHE:
        return _EXAMPLE_CACHE[name]

    path = Path(__file__).resolve().parents[1] / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"examples_{name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load example module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _EXAMPLE_CACHE[name] = module
    return module


def test_compute_psnr_identical():
    eval_vae = load_example_module("eval_vae")
    pred = torch.rand(2, 3, 8, 8)
    psnr = eval_vae.compute_psnr(pred, pred)
    assert psnr > 90.0


def test_compute_ssim_identical():
    eval_vae = load_example_module("eval_vae")
    pred = torch.rand(2, 3, 8, 8)
    ssim = eval_vae.compute_ssim(pred, pred)
    assert ssim == pytest.approx(1.0, rel=1e-4, abs=1e-4)


def test_create_grid_layout():
    eval_dit = load_example_module("eval_dit")
    img1 = Image.new("RGB", (10, 8), (10, 20, 30))
    img2 = Image.new("RGB", (10, 8), (40, 50, 60))
    img3 = Image.new("RGB", (10, 8), (70, 80, 90))

    grid = eval_dit.create_grid([img1, img2, img3], n_cols=2)

    assert grid.size == (20, 16)
    assert grid.getpixel((0, 0)) == (10, 20, 30)
    assert grid.getpixel((10, 0)) == (40, 50, 60)
    assert grid.getpixel((0, 8)) == (70, 80, 90)


def test_train_vae_compute_loss_mask_and_kl():
    train_vae = load_example_module("train_vae")

    class DummyPosterior:
        def __init__(self, value):
            self._value = value

        def kl(self):
            return self._value

    class DummyModel:
        def __init__(self, pred_patches, posterior):
            self._pred_patches = pred_patches
            self._posterior = posterior

        def __call__(self, batch, sample_posterior=True):
            return {
                "patches": self._pred_patches,
                "posterior": self._posterior,
            }

    pred = torch.tensor([[[2.0], [1.0]]])
    target = torch.tensor([[[1.0], [3.0]]])
    ptype = torch.tensor([[True, False]])
    batch = {"patches": target, "ptype": ptype}

    posterior = DummyPosterior(torch.tensor([2.0]))
    model = DummyModel(pred, posterior)

    losses = train_vae.compute_loss(model, batch, kl_weight=0.5)

    assert losses["recon_loss"] == pytest.approx(1.0)
    assert losses["kl_loss"] == pytest.approx(2.0)
    assert losses["loss"] == pytest.approx(2.0)
