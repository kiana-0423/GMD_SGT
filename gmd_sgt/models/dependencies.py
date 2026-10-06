"""Optional third-party dependencies used by the model package."""

from __future__ import annotations

try:
    from e3nn import o3
    from e3nn.nn import BatchNorm as IrrepsBatchNorm

    E3NN_AVAILABLE = True
except ImportError:
    o3 = None
    IrrepsBatchNorm = None
    E3NN_AVAILABLE = False
    print(
        "[WARNING] e3nn not found. Only scalar-irreps UnifiedEquivariantMLIP and "
        "l_max <= 2 backbones are available."
    )

try:
    from torch_scatter import scatter_add, scatter_mean

    SCATTER_AVAILABLE = True
except ImportError:
    scatter_add = None
    scatter_mean = None
    SCATTER_AVAILABLE = False
    print("[WARNING] torch_scatter not found (not required; pure-torch scatter is used).")

try:
    from torch_cluster import radius_graph

    CLUSTER_AVAILABLE = True
except ImportError:
    radius_graph = None
    CLUSTER_AVAILABLE = False
    print("[WARNING] torch_cluster not found. Using the dense O(N^2) neighbor search.")

__all__ = [
    "CLUSTER_AVAILABLE",
    "E3NN_AVAILABLE",
    "IrrepsBatchNorm",
    "SCATTER_AVAILABLE",
    "o3",
    "radius_graph",
    "scatter_add",
    "scatter_mean",
]
