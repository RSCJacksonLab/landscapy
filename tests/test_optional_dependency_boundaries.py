"""Release checks for optional dependencies and backend lifecycle ownership."""

from __future__ import annotations

import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from fitness_landscape import _optional
from fitness_landscape.analysis import statistics


def test_publication_core_does_not_import_optional_backends():
    repository = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repository / "src")
    subprocess.run(
        [sys.executable, str(repository / "scripts" / "minimal_install_smoke.py")],
        cwd=repository,
        env=env,
        check=True,
    )


def test_package_import_does_not_patch_cogent3_moltype():
    code = """
from cogent3.core.moltype import MolType
original = MolType.make_seq
import fitness_landscape
assert MolType.make_seq is original
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_missing_default_dependency_has_actionable_reinstall(monkeypatch):
    def missing(_module):
        raise ModuleNotFoundError("No module named 'ray'", name="ray")

    monkeypatch.setattr(_optional, "import_module", missing)

    with pytest.raises(ModuleNotFoundError, match=r"force-reinstall landscapy"):
        _optional.require_optional(
            "ray",
            extra="parallel",
            purpose="parallel analysis",
        )


def test_missing_ml_dependency_keeps_actionable_extra(monkeypatch):
    def missing(_module):
        raise ModuleNotFoundError(
            "No module named 'torch_geometric'",
            name="torch_geometric",
        )

    monkeypatch.setattr(_optional, "import_module", missing)

    with pytest.raises(ModuleNotFoundError, match=r"landscapy\[ml\]"):
        _optional.require_optional(
            "torch_geometric",
            extra="ml",
            purpose="PyTorch Geometric export",
        )


def test_optional_import_does_not_mask_transitive_dependency_errors(monkeypatch):
    def missing(_module):
        raise ModuleNotFoundError("No module named 'backend_helper'", name="backend_helper")

    monkeypatch.setattr(_optional, "import_module", missing)

    with pytest.raises(ModuleNotFoundError, match="backend_helper") as error:
        _optional.require_optional(
            "ray",
            extra="parallel",
            purpose="parallel analysis",
        )

    assert "force-reinstall landscapy" not in str(error.value)


@pytest.mark.parametrize("already_initialized", [False, True])
def test_ray_runtime_only_stops_a_runtime_it_started(
    monkeypatch,
    already_initialized,
):
    state = {"initialized": already_initialized, "init_calls": 0, "shutdown_calls": 0}

    def init(**_kwargs):
        state["initialized"] = True
        state["init_calls"] += 1

    def shutdown():
        state["initialized"] = False
        state["shutdown_calls"] += 1

    fake_ray = SimpleNamespace(
        is_initialized=lambda: state["initialized"],
        init=init,
        shutdown=shutdown,
    )
    monkeypatch.setattr(_optional, "require_optional", lambda *_args, **_kwargs: fake_ray)

    with _optional.ray_runtime(2, purpose="test") as yielded:
        assert yielded is fake_ray
        assert state["initialized"] is True

    if already_initialized:
        assert state["init_calls"] == 0
        assert state["shutdown_calls"] == 0
        assert state["initialized"] is True
    else:
        assert state["init_calls"] == 1
        assert state["shutdown_calls"] == 1
        assert state["initialized"] is False


def test_parallel_statistics_uses_scoped_ray_runtime(monkeypatch):
    class Landscape:
        def __init__(self, value):
            self.value = value
            self.viewed = None

        def view(self, layer_name):
            self.viewed = layer_name

    class RemoteFunction:
        def __init__(self, function):
            self.function = function

        def remote(self, *args):
            return self.function(*args)

    class FakeRay:
        @staticmethod
        def remote(function):
            return RemoteFunction(function)

        @staticmethod
        def put(value):
            return value

        @staticmethod
        def get(values):
            return values

    runtime_calls = []

    @contextmanager
    def fake_runtime(workers, *, purpose):
        runtime_calls.append((workers, purpose))
        yield FakeRay()

    monkeypatch.setattr(statistics, "ray_runtime", fake_runtime)
    landscapes = [Landscape(2), Landscape(3)]

    result = statistics._parallel_analyze_landscapes(
        landscapes,
        lambda landscape: landscape.value * 2,
        layer_name="fitness",
        use_ray=True,
        num_workers=8,
    )

    assert result == [4, 6]
    assert [landscape.viewed for landscape in landscapes] == ["fitness", "fitness"]
    assert runtime_calls == [(2, "parallel landscape statistics")]


def test_parallel_statistics_serial_and_empty_paths():
    class Landscape:
        def __init__(self, value):
            self.value = value

        def view(self, _layer_name):
            self.value += 1

    landscapes = [Landscape(1), Landscape(4)]
    result = statistics._parallel_analyze_landscapes(
        landscapes,
        lambda landscape: landscape.value,
        layer_name="fitness",
        use_ray=False,
    )

    assert result == [2, 5]
    assert statistics._parallel_analyze_landscapes([], lambda value: value) == []
