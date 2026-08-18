"""Protein language-model embeddings loaded on first use."""

from importlib import import_module

__all__ = ["ESMEmbedder"]


def __getattr__(name):
    if name != "ESMEmbedder":
        raise AttributeError(name)
    value = import_module(f"{__name__}.esm").ESMEmbedder
    globals()[name] = value
    return value
