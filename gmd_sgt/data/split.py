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

    Notes
    -----
    Split sizes are ``floor(n * fraction)``. A positive ``val_fraction``
    always yields at least one validation structure when the dataset can
    spare it (at least one training structure remains); ``val_fraction=0``
    deliberately produces an empty validation set, in which case the
    Trainer selects checkpoints on the training loss.

    Raises
    ------
    ValueError
        If a fraction is outside ``[0, 1)``, the fractions sum to ``>= 1``,
        or no training structure would remain.
    """
    for name, value in (("val_fraction", val_fraction), ("test_fraction", test_fraction)):
        if not 0.0 <= float(value) < 1.0:
            raise ValueError(f"{name} must be in [0, 1), got {value}")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError(
            f"val_fraction + test_fraction must be < 1, got {val_fraction} + {test_fraction}"
        )

    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    n = len(dataset)
    n_test = int(n * test_fraction)
    n_val  = int(n * val_fraction)
    if val_fraction > 0 and n_val == 0 and n - n_test >= 2:
        n_val = 1
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ValueError(
            f"Dataset of {n} structures leaves no training data with "
            f"val_fraction={val_fraction}, test_fraction={test_fraction}"
        )

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
