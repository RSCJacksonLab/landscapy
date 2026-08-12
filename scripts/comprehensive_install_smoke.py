"""Exercise the one-command non-ML install through the installed CLI."""

from __future__ import annotations

import importlib.util
from importlib.metadata import PackageNotFoundError, version
import os
import shutil
import subprocess
import sysconfig
import tempfile
from pathlib import Path

from fitness_landscape import FitnessLandscape


NON_ML_DISTRIBUTIONS = (
    "click",
    "cogent3",
    "faiss-cpu",
    "gudhi",
    "piqtree",
    "pyarrow",
    "ray",
    "scikit-learn",
    "softalign",
    "torch",
    "transformers",
    "tqdm",
)


def _landscapy_executable() -> Path:
    executable = shutil.which("landscapy")
    if executable is not None:
        return Path(executable)

    suffix = ".exe" if os.name == "nt" else ""
    scripts_directory = Path(sysconfig.get_path("scripts"))
    installed = scripts_directory / f"landscapy{suffix}"
    if not installed.is_file():
        raise AssertionError("Landscapy CLI entry point is not installed")
    return installed


def main() -> None:
    for distribution in NON_ML_DISTRIBUTIONS:
        try:
            version(distribution)
        except PackageNotFoundError as error:
            raise AssertionError(
                f"landscapy[all] did not install {distribution}"
            ) from error
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
