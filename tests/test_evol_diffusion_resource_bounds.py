"""Resource-bound validation for evolutionary-diffusion construction."""

import numpy as np
import pytest

from fitness_landscape._const import PROT_20
from fitness_landscape.core.graph import create_evol_diffusion_graph
from fitness_landscape.core.sequence import BaseNumpySequence


def test_max_in_flight_alignment_validation_precedes_optional_runtime():
    sequence = BaseNumpySequence.from_string(
        "A",
        alphabet=PROT_20,
        sequence_id="only",
    )
    embedding = np.zeros((1, 2), dtype=float)

    graph = create_evol_diffusion_graph(
        [sequence],
        embeddings=embedding,
        k=1,
        max_in_flight_alignments=1,
    )
    assert graph.number_of_nodes() == 1

    with pytest.raises(ValueError, match="max_in_flight_alignments"):
        create_evol_diffusion_graph(
            [sequence],
            embeddings=embedding,
            k=1,
            max_in_flight_alignments=0,
        )
