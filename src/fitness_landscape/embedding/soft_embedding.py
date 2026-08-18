"""Compatibility import for the consolidated ESM embedder.

The authoritative implementation lives in :mod:`fitness_landscape.embedding.esm`.
"""

from .esm import ESMEmbedder

__all__ = ["ESMEmbedder"]
