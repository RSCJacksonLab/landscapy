"""Graph matching utilities for landscapy.

The reversible-jump aligners remain available via ``phylo-landscapy``. Core
latent graph reconstruction helpers are provided here.
"""

from __future__ import annotations

from .minimum_spanning_graph import (
    graph_to_length_matrix,
    landmark_mds,
    detect_gap_pairs_kdtree,
    self_tuned_graph,
    reconstruct_latent_graph_with_steiner,
    reconstruct_latent_graph_midpoint,
)
from .isorank import (
    normalize_adj_matrix,
    cosine_similarity_matrix,
    isorank_with_features,
)

__all__ = [
    "graph_to_length_matrix",
    "landmark_mds",
    "detect_gap_pairs_kdtree",
    "self_tuned_graph",
    "reconstruct_latent_graph_with_steiner",
    "reconstruct_latent_graph_midpoint",
    "normalize_adj_matrix",
    "cosine_similarity_matrix",
    "isorank_with_features",
]

_ALIGNER_NAMES = {
    "RJMCMCAligner",
    "HierarchicalRJMCMCAligner",
    "auto_anchors_by_cosine",
}

try:  # pragma: no cover
    from phylo_landscapy.graph_matching import (  # type: ignore
        RJMCMCAligner,
        HierarchicalRJMCMCAligner,
        auto_anchors_by_cosine,
    )

    __all__ += sorted(_ALIGNER_NAMES)
except ModuleNotFoundError as exc:  # pragma: no cover
    if exc.name and not exc.name.startswith("phylo_landscapy"):
        raise
    _IMPORT_ERROR = exc
    _ERR = (
        "RJMCMC alignment utilities have moved to 'phylo-landscapy'. "
        "Install phylo-landscapy to access RJMCMCAligner and related helpers."
    )

    def __getattr__(name):  # type: ignore
        if name in _ALIGNER_NAMES:
            raise ModuleNotFoundError(_ERR) from _IMPORT_ERROR
        raise AttributeError(name)
