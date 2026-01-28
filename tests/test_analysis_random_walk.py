import networkx as nx
import numpy as np

from fitness_landscape.analysis.random_walk import calculate_ruggedness_autocorrelation_analytical
from fitness_landscape.core.fitness import NumericFitness
from fitness_landscape.core.landscape import FitnessLandscape
from fitness_landscape.core.sequence import BaseNumpySequence
from fitness_landscape.transforms.eigenmode import eigenmode_decomposition


def _toy_landscape():
    seqs = [
        BaseNumpySequence([0], sequence_id="s0"),
        BaseNumpySequence([1], sequence_id="s1"),
        BaseNumpySequence([2], sequence_id="s2"),
        BaseNumpySequence([3], sequence_id="s3"),
    ]
    G = nx.path_graph(len(seqs))
    for idx, seq in enumerate(seqs):
        G.nodes[idx]["sequence"] = seq

    fitness = NumericFitness(name="fitness", values=[0.0, 1.0, 0.5, 1.5])
    return FitnessLandscape(sequences=seqs, graph=G, fitness_layers={"fitness": fitness})


def test_autocorrelation_analytical_with_precomputed_eigenpairs_matches():
    landscape = _toy_landscape()
    evals, evecs = eigenmode_decomposition(landscape.graph, matrix="laplacian", k=None)
    base = calculate_ruggedness_autocorrelation_analytical(landscape)
    pre = calculate_ruggedness_autocorrelation_analytical(
        landscape, _eigenvalues=evals, _eigenvectors=evecs
    )
    np.testing.assert_allclose(base["autocorrelation"], pre["autocorrelation"], rtol=1e-6, atol=1e-6)
    assert base["correlation_length"] == pre["correlation_length"]
