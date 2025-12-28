"""Image comparison evaluator for VAE reconstruction quality."""

import time
import torch
import torch.distributed as dist
from typing import Callable, Dict, List, Optional, Tuple
from tqdm import tqdm

from vitok.evaluators.metrics import MetricCalculator
from vitok.data import create_dataloader


def get_rank() -> int:
    """Get current process rank."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get world size."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


class Evaluator:
    """Evaluator for image reconstruction quality.

    Runs a predict function over a dataset and computes metrics.

    Example:
        def predict_fn(batch):
            output = model(batch)
            recon = postprocess_images(output, unpack=True)
            ref = postprocess_images(batch, unpack=True)
            return {'images': recon, 'ref_images': ref}

        evaluator = Evaluator(
            predict_fn=predict_fn,
            data_source="path/to/images",
            metrics=('fid', 'fdd', 'ssim', 'psnr'),
        )
        stats = evaluator.run()
    """

    def __init__(
        self,
        predict_fn: Callable,
        data_source: str,
        pp: str = "to_tensor|normalize(minus_one_to_one)|patchify(512, 16, 256)",
        batch_size: int = 32,
        num_workers: int = 4,
        val_size: int = 50000,
        metrics: Tuple[str, ...] = ('fid', 'fdd', 'ssim', 'psnr'),
        max_visuals: int = 0,
        seed: int = 42,
    ):
        """Initialize evaluator.

        Args:
            predict_fn: Function that takes batch and returns dict with
                        'images' (reconstructions) and 'ref_images' (references)
            data_source: Path to data or HuggingFace pattern
            pp: Preprocessing string
            batch_size: Global batch size
            num_workers: Number of dataloader workers
            val_size: Number of samples to evaluate
            metrics: Tuple of metrics to compute
            max_visuals: Number of sample images to return
            seed: Random seed for dataloader
        """
        self.predict_fn = predict_fn
        self.val_size = val_size
        self.max_visuals = max_visuals

        world_size = get_world_size()
        self.per_gpu_batch_size = max(1, batch_size // world_size)
        self.global_batch_size = self.per_gpu_batch_size * world_size
        self.total_steps = max(1, val_size // self.global_batch_size)

        # Create dataloader
        self.dataloader = create_dataloader(
            source=data_source,
            pp=pp,
            batch_size=self.per_gpu_batch_size,
            num_workers=num_workers,
            seed=seed,
        )

        # Initialize metrics calculator
        self.metrics = MetricCalculator(metrics=metrics)

    def run(self, step: int = 0) -> Dict:
        """Run evaluation.

        Args:
            step: Current training step (for logging)

        Returns:
            Dictionary of computed metrics
        """
        rank = get_rank()
        disable_tqdm = rank != 0

        if dist.is_initialized():
            dist.barrier()

        t_start = time.perf_counter()

        all_real = []
        all_fake = []

        # Collect samples
        data_iter = iter(self.dataloader)
        for _ in tqdm(range(self.total_steps), disable=disable_tqdm, desc="Evaluating"):
            try:
                batch, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch, _ = next(data_iter)

            # Move to GPU
            batch = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Run prediction
            with torch.no_grad():
                result = self.predict_fn(batch)

            ref_images = result['ref_images']
            recon_images = result['images']

            # Store samples (on CPU)
            if isinstance(ref_images, list):
                all_real.extend([x.cpu() for x in ref_images])
            else:
                all_real.append(ref_images.cpu())

            if isinstance(recon_images, list):
                all_fake.extend([x.cpu() for x in recon_images])
            else:
                all_fake.append(recon_images.cpu())

        # Compute metrics
        if rank == 0:
            print("Computing metrics...")

        self.metrics.move_model_to_device('cuda')
        self.metrics.update(all_real, all_fake)
        stats = self.metrics.gather()
        self.metrics.reset()
        self.metrics.move_model_to_device('cpu')

        # Add sample images if requested
        if self.max_visuals > 0:
            stats['sample_reference'] = all_real[:self.max_visuals]
            stats['sample_recons'] = all_fake[:self.max_visuals]

        # Timing
        elapsed = time.perf_counter() - t_start
        stats['eval_time_s'] = elapsed

        if rank == 0:
            print(f"Evaluation complete in {elapsed:.1f}s")
            for k, v in stats.items():
                if k not in ('sample_reference', 'sample_recons') and isinstance(v, (int, float)):
                    print(f"  {k}: {v:.4f}")

        if dist.is_initialized():
            dist.barrier()

        return stats
