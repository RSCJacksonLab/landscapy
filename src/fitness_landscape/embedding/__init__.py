"""Protein language-model embeddings loaded on first use."""

from importlib import import_module

__all__ = ["ESMEmbedder"]


def __getattr__(name):
    if name != "ESMEmbedder":
        raise AttributeError(name)
    value = import_module(f"{__name__}.soft_embedding").ESMEmbedder
    globals()[name] = value
    return value
