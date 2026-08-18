"""Release smoke tests for the supported import and command surfaces."""

from __future__ import annotations

import importlib
import pkgutil

from click.testing import CliRunner

import fitness_landscape
from fitness_landscape.__main__ import (
    cli,
    evol_diffusion_landscape,
    knn_landscape,
    phylo_landscape,
)


def test_every_declared_public_name_is_importable():
    modules = [fitness_landscape]
    modules.extend(
        importlib.import_module(module_info.name)
        for module_info in pkgutil.walk_packages(
            fitness_landscape.__path__, fitness_landscape.__name__ + "."
        )
    )

    missing = []
    for module in modules:
        for name in getattr(module, "__all__", ()):
            try:
                getattr(module, name)
            except (AttributeError, ModuleNotFoundError) as error:
                missing.append(f"{module.__name__}.{name}: {error}")

    assert not missing, "Invalid public exports:\n" + "\n".join(missing)


def test_star_imports_resolve_declared_names():
    for package in (
        "fitness_landscape",
        "fitness_landscape.analysis",
        "fitness_landscape.core",
        "fitness_landscape.io",
        "fitness_landscape.models",
        "fitness_landscape.phylo",
        "fitness_landscape.transforms",
    ):
        namespace = {}
        exec(f"from {package} import *", namespace)


def test_all_console_commands_render_help():
    runner = CliRunner()

    for command in (cli, evol_diffusion_landscape, knn_landscape, phylo_landscape):
        result = runner.invoke(command, ["--help"])
        assert result.exit_code == 0, result.output
        assert "Usage:" in result.output
