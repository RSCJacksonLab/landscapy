import numpy as np
import pytest
from fitness_landscape.utils import cosine_similarity_matrix, get_landscape_dist_mat
from fitness_landscape.core.landscape import FitnessLandscape
from fitness_landscape.core.sequence import generate_sequences
from fitness_landscape.core.fitness import NumericFitness

@pytest.fixture
def util_landscape():
    """Provides a basic FitnessLandscape for utility testing."""
    sequences = generate_sequences(length=3, alphabet=[0, 1])
    fitness_values = [[val] for val in np.random.rand(8)]
    fitness_layers = {
        'default': NumericFitness(name='default', values=fitness_values)
    }
    return FitnessLandscape.from_sequences(
        sequences=sequences,
        fitness_layers=fitness_layers,
        graph_type='hamming'
    )

def test_cosine_similarity_matrix():
    """Tests the cosine similarity matrix computation."""
    A = np.array([[1, 0], [0, 1]])
    B = np.array([[1, 1], [1, -1]])
    sim_matrix = cosine_similarity_matrix(A, B)
    assert np.allclose(sim_matrix, np.array([[0.70710678, 0.70710678], [0.70710678, -0.70710678]]))

def test_get_landscape_dist_mat(util_landscape):
    """Tests getting the landscape distance matrix."""
    dist_mat = get_landscape_dist_mat(util_landscape)
    assert dist_mat.shape == (len(util_landscape.sequences), len(util_landscape.sequences))
    assert np.all(np.diag(dist_mat) == 0)