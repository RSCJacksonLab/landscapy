"""Compat shim for legacy latent alignment imports.

The implementation is provided by ``phylo_landscapy``.
"""

from __future__ import annotations

try:  # pragma: no cover
    from phylo_landscapy.graph_matching.latent_alignment import *  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover
    _ERR = (
        "RJMCMC latent alignment has moved to 'phylo-landscapy'. "
        "Install phylo-landscapy to access RJMCMCAligner and related helpers."
    )

    __all__ = ()

    def __getattr__(name):  # type: ignore
        raise ModuleNotFoundError(_ERR) from exc
