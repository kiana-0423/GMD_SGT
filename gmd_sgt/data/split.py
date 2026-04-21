"""Train / val / test dataset splitting."""

from __future__ import annotations

import random
from typing import Tuple

from .dataset import AtomicDataset


def split_dataset(
    dataset: AtomicDataset,
    val_fraction: float = 0.1,
    test_fraction: float = 0.05,
    seed: int = 42,
) -> Tuple[AtomicDataset, AtomicDataset, AtomicDataset]:
    """
    Randomly split a dataset into train / val / test subsets.

    Parameters
    ----------
    dataset       : full AtomicDataset
    val_fraction  : fraction of data for validation
    test_fraction : fraction of data for test  (0 = no test set)
    seed          : random seed for reproducibility

    Returns
    -------
    train_set, val_set, test_set
    (test_set will be empty if test_fraction == 0)
    """
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    n = len(dataset)
    n_test = int(n * test_fraction)
    n_val  = int(n * val_fraction)
    n_train = n - n_val - n_test

    train_idx = indices[:n_train]
    val_idx   = indices[n_train : n_train + n_val]
    test_idx  = indices[n_train + n_val :]

    train_set = AtomicDataset([dataset[i] for i in train_idx])
    val_set   = AtomicDataset([dataset[i] for i in val_idx])
    test_set  = AtomicDataset([dataset[i] for i in test_idx])

    print(
        f"[split] train={len(train_set)}  val={len(val_set)}  test={len(test_set)}"
        f"  (seed={seed})"
    )
    return train_set, val_set, test_set
