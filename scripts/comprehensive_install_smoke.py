"""Exercise the one-command non-ML install through the installed CLI."""

from __future__ import annotations

import importlib
import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from fitness_landscape import FitnessLandscape


NON_ML_MODULES = (
    "click",
    "cogent3",
    "gudhi",
    "piqtree",
    "pyarrow",
    "ray",
    "sklearn",
    "softalign",
    "torch",
    "transformers",
    "tqdm",
)


def _landscapy_executable() -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    executable = Path(sys.executable).parent / f"landscapy{suffix}"
    if not executable.is_file():
        raise AssertionError(f"Landscapy CLI entry point was not installed at {executable}")
    return executable


def main() -> None:
    for module in NON_ML_MODULES:
        importlib.import_module(module)
    if importlib.util.find_spec("torch_geometric") is not None:
        raise AssertionError(
            "landscapy[all] unexpectedly installed the ml-only "
            "torch-geometric dependency"
        )

    cli = _landscapy_executable()
    subprocess.run([str(cli), "--help"], check=True, capture_output=True, text=True)

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        fasta = directory / "sequences.fasta"
        output = directory / "knn.pkl"
        fasta.write_text(
            ">seq-1\nACDE\n"
            ">seq-2\nACDF\n"
            ">seq-3\nACDG\n"
            ">seq-4\nWCDG\n",
            encoding="utf-8",
        )

        subprocess.run(
            [
                str(cli),
                "knn-landscape",
                "--sequences",
                str(fasta),
                "--output",
                str(output),
                "--k",
                "2",
                "--backend",
                "balltree",
                "--embedding-domain",
                "ohe",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        landscape = FitnessLandscape.load(output)

    assert landscape.graph.number_of_nodes() == 4
    assert landscape.graph.number_of_edges() > 0


if __name__ == "__main__":
    main()
