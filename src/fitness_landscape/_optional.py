"""Helpers for optional backend imports and owned runtime lifecycles."""

from __future__ import annotations

from contextlib import contextmanager
from importlib import import_module
from types import ModuleType
from typing import Iterator


def require_optional(module: str, *, extra: str, purpose: str) -> ModuleType:
    """Import an optional module or raise an actionable installation error."""

    try:
        return import_module(module)
    except ModuleNotFoundError as error:
        missing = error.name or ""
        root = module.split(".", 1)[0]
        if missing != root and not missing.startswith(f"{root}."):
            raise
        message = (
            f"{purpose} requires the optional dependency {root!r}. "
            f"Install it with `python -m pip install 'landscapy[{extra}]'`."
        )
        raise ModuleNotFoundError(message, name=missing) from error


@contextmanager
def ray_runtime(num_cpus: int, *, purpose: str) -> Iterator[ModuleType]:
    """Yield Ray while owning and cleaning up only runtimes started here."""

    if num_cpus < 1:
        raise ValueError("num_cpus must be at least 1")
    ray = require_optional("ray", extra="parallel", purpose=purpose)
    owned = not ray.is_initialized()
    if owned:
        ray.init(num_cpus=num_cpus, ignore_reinit_error=True)
    try:
        yield ray
    finally:
        if owned and ray.is_initialized():
            ray.shutdown()
