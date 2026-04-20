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
    print("[WARNING] e3nn not found. Equivariant tensor products will use PLACEHOLDERS.")

try:
    from torch_scatter import scatter_add, scatter_mean

    SCATTER_AVAILABLE = True
except ImportError:
    scatter_add = None
    scatter_mean = None
    SCATTER_AVAILABLE = False
    print("[WARNING] torch_scatter not found. Using torch.scatter_add_ fallback.")

try:
    from torch_cluster import radius_graph

    CLUSTER_AVAILABLE = True
except ImportError:
    radius_graph = None
    CLUSTER_AVAILABLE = False
    print("[WARNING] torch_cluster not found. Neighbor graph construction disabled.")

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
