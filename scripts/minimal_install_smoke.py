"""Exercise the publication core from a wheel with no optional extras."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from fitness_landscape import FitnessLandscape, NumericFitness
from fitness_landscape import analysis, phylo, transforms
from fitness_landscape.core.sequence import BinarySequence
import fitness_landscape.embedding as embedding


OPTIONAL_ROOTS = {
    "click",
    "cogent3",
    "faiss",
    "gudhi",
    "ray",
    "sklearn",
    "softalign",
    "torch",
    "torch_geometric",
    "transformers",
}


def main() -> None:
    sequences = [
        BinarySequence.from_bits(bits)
        for bits in ([0, 0], [0, 1], [1, 0], [1, 1])
    ]
    fitness = NumericFitness("fitness", [[0.0], [1.0], [1.0], [2.0]])
    landscape = FitnessLandscape.build(
        sequences,
        graph="hamming",
        fitness_layers={"fitness": fitness},
        attach_embeddings=False,
    )

    assert landscape.graph.number_of_nodes() == 4
    assert landscape.graph.number_of_edges() == 4
    assert landscape.active_layer.to_scalar().tolist() == [0.0, 1.0, 1.0, 2.0]
    assert analysis.__name__.endswith(".analysis")
    assert transforms.__name__.endswith(".transforms")
    assert phylo.__name__.endswith(".phylo")
    assert embedding.__name__.endswith(".embedding")

    imported_optional = sorted(OPTIONAL_ROOTS.intersection(sys.modules))
    if imported_optional:
        raise AssertionError(
            "minimal core imported optional dependencies: "
            + ", ".join(imported_optional)
        )

    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / "minimal-landscape"
        landscape.save_bundle_dir(
            bundle,
            metadata={"dataset_name": "minimal-install-smoke"},
        )
        loaded = FitnessLandscape.load_bundle_dir(bundle)

    assert loaded.graph.number_of_nodes() == 4
    assert loaded.graph.number_of_edges() == 4
    assert loaded.active_layer.to_scalar().tolist() == [0.0, 1.0, 1.0, 2.0]


if __name__ == "__main__":
    main()
