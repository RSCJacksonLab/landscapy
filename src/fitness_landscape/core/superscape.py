"""Superscape compatibility shim.

The full superscape implementation now lives in ``phylo_landscapy``.
Importing from ``fitness_landscape.core.superscape`` will re-export that
functionality when the optional dependency is installed, otherwise an
informative error is raised.
"""

from __future__ import annotations

try:  # pragma: no cover - thin compatibility wrapper
    from phylo_landscapy.core.superscape import *  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover
    if exc.name and not exc.name.startswith("phylo_landscapy"):
        raise
    _IMPORT_ERROR = exc
    _ERR = (
        "Superscape utilities have moved to 'phylo-landscapy'. "
        "Install phylo-landscapy to access FitnessSuperscape and related APIs."
    )

    __all__ = ("FitnessSuperscape", "NullAligner")

    class FitnessSuperscape:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(_ERR) from _IMPORT_ERROR

    class NullAligner:  # pragma: no cover - placeholder for type checkers
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(_ERR) from _IMPORT_ERROR

    def __getattr__(name):  # type: ignore
        raise ModuleNotFoundError(_ERR) from _IMPORT_ERROR
