"""Spectral transforms loaded on first use."""

from importlib import import_module

_EXPORTS = {
    "walsh_transform": "walsh_hadamard",
    "walsh_coefficients": "walsh_hadamard",
    "graph_fourier_transform": "graph_fourier",
    "eigenmode_decomposition": "eigenmode",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value
