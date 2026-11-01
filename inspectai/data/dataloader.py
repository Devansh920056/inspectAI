"""DataLoader construction with reproducible, macOS/MPS-friendly defaults."""

from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def build_dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    shuffle: bool | None = None,
    num_workers: int = 0,
    seed: int = 0,
    drop_last: bool = False,
    device: str | torch.device | None = None,
) -> DataLoader:
    """Create a DataLoader.

    * ``shuffle`` defaults to True for a train split, False otherwise.
    * ``num_workers=0`` is the default: on macOS workers start via ``spawn``, which
      has noticeable startup cost for small datasets.
    * ``pin_memory`` is only enabled for CUDA targets (not supported for MPS/CPU).
    * Shuffling uses a seeded generator, so an epoch's order is reproducible.
    """
    if shuffle is None:
        shuffle = getattr(dataset, "split", None) == "train"
    generator = torch.Generator()
    generator.manual_seed(seed)
    pin = device is not None and torch.device(device).type == "cuda"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin,
        generator=generator,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )


def move_batch(batch: dict, device: str | torch.device) -> dict:
    """Move tensor entries of a batch to ``device``; strings/lists stay on the host."""
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
