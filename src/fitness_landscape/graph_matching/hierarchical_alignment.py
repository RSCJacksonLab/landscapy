"""Compat shim for hierarchical RJMCMC alignment.

Delegates to the implementation in ``phylo_landscapy`` when available.
"""

from __future__ import annotations

try:  # pragma: no cover
    from phylo_landscapy.graph_matching.hierarchical_alignment import *  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover
    if exc.name and not exc.name.startswith("phylo_landscapy"):
        raise
    _IMPORT_ERROR = exc
    _ERR = (
        "Hierarchical RJMCMC alignment has moved to 'phylo-landscapy'. "
        "Install phylo-landscapy to use HierarchicalRJMCMCAligner."
    )

    __all__ = ()

    def __getattr__(name):  # type: ignore
        raise ModuleNotFoundError(_ERR) from _IMPORT_ERROR
