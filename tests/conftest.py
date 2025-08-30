import numpy as np
import pytest
import networkx as nx
from pathlib import Path

from fitness_landscape.core.sequence import generate_sequences
from fitness_landscape.core.fitness import NumericFitness
from fitness_landscape.core.landscape import FitnessLandscape


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(1234)


@pytest.fixture
def binary_3bit_landscape(rng):
    seqs = generate_sequences(length=3, alphabet=[0, 1])
    vals = [[float(x)] for x in rng.random(len(seqs))]
    layers = {"default": NumericFitness(name="default", values=vals)}
    return FitnessLandscape.from_sequences(seqs, fitness_layers=layers, graph_type="hamming")


@pytest.fixture
def tmp_text(tmp_path):
    def _write(name: str, text: str) -> Path:
        p = tmp_path / name
        p.write_text(text)
        return p
    return _write

