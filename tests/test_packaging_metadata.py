"""Checks the default install and optional import contracts."""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((REPOSITORY / "pyproject.toml").read_text(encoding="utf-8"))
PROJECT_DEPENDENCIES = PYPROJECT["project"]["dependencies"]
EXTRAS = PYPROJECT["project"]["optional-dependencies"]
IMPORT_DISTRIBUTIONS = {
    "faiss": "faiss-cpu",
    "sklearn": "scikit-learn",
    "torch_geometric": "torch-geometric",
}


def _requirement_name(requirement: str) -> str:
    match = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", requirement)
    assert match is not None, f"Unable to parse requirement {requirement!r}"
    return match.group().lower().replace("_", "-")


def _declared_names(extra: str) -> set[str]:
    return {_requirement_name(requirement) for requirement in EXTRAS[extra]}


def test_default_install_contains_core_and_every_non_ml_user_extra():
    core = {
        "networkx>=3.2",
        "numpy>=1.24",
        "pandas>=2.3",
        "scipy>=1.10",
    }
    user_extras = set(EXTRAS) - {"all", "dev", "ml"}
    non_ml_requirements = {
        requirement
        for extra in user_extras
        for requirement in EXTRAS[extra]
    }

    assert set(PROJECT_DEPENDENCIES) == core | non_ml_requirements


def test_all_remains_a_backward_compatible_alias_for_the_default_install():
    assert EXTRAS["all"] == []


def test_default_install_excludes_dependencies_unique_to_ml():
    shared_embedding_requirements = set(EXTRAS["embeddings"])
    ml_only = set(EXTRAS["ml"]) - shared_embedding_requirements

    assert "torch-geometric>=2.6" in ml_only
    assert ml_only.isdisjoint(PROJECT_DEPENDENCIES)


def test_optional_imports_have_direct_dependencies_in_their_named_extra():
    missing = []
    for source in (REPOSITORY / "src" / "fitness_landscape").rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
            if not isinstance(call.func, ast.Name) or call.func.id != "require_optional":
                continue
            if not call.args or not isinstance(call.args[0], ast.Constant):
                continue
            extra_keyword = next(
                (keyword for keyword in call.keywords if keyword.arg == "extra"),
                None,
            )
            if extra_keyword is None or not isinstance(extra_keyword.value, ast.Constant):
                continue

            module_root = str(call.args[0].value).split(".", 1)[0]
            distribution = IMPORT_DISTRIBUTIONS.get(module_root, module_root)
            extra = str(extra_keyword.value.value)
            if extra not in EXTRAS or distribution not in _declared_names(extra):
                missing.append(f"{source.relative_to(REPOSITORY)}: {module_root} -> {extra}")

    assert not missing, "Optional imports missing direct extra dependencies:\n" + "\n".join(
        missing
    )
