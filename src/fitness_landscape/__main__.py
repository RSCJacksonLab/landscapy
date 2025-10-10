"""CLI shim for landscapy.

The legacy superscape and phylogenetic tooling has moved to the
``phylo-landscapy`` package.  When that dependency is available the CLI
subcommands are delegated to it; otherwise, informative errors are raised.
"""

from __future__ import annotations

import importlib
from typing import Any

import click


def _load_phylo_module() -> Any:
    """Attempt to import the CLI module from ``phylo_landscapy``."""

    try:
        return importlib.import_module("phylo_landscapy.__main__")
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise click.UsageError(
            "Superscape and phylogenetic CLI commands now live in 'phylo-landscapy'. "
            "Install phylo-landscapy to re-enable these commands."
        ) from exc


@click.group()
def cli() -> None:
    """Entry point maintained for backwards compatibility."""


@cli.command()
@click.pass_context
def diffusion_evol_superscape(ctx: click.Context, *args: Any, **kwargs: Any) -> None:
    module = _load_phylo_module()
    ctx.forward(module.diffusion_evol_superscape)


@cli.command()
@click.pass_context
def evol_diffusion_landscape(ctx: click.Context, *args: Any, **kwargs: Any) -> None:
    module = _load_phylo_module()
    ctx.forward(module.evol_diffusion_landscape)


@cli.command()
@click.pass_context
def phylo_landscape(ctx: click.Context, *args: Any, **kwargs: Any) -> None:
    module = _load_phylo_module()
    ctx.forward(module.phylo_landscape)


@cli.command()
@click.pass_context
def phylo_superscape(ctx: click.Context, *args: Any, **kwargs: Any) -> None:
    module = _load_phylo_module()
    ctx.forward(module.phylo_superscape)
