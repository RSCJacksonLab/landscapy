import numpy as np

from fitness_landscape.utils import get_landscape_dist_mat, _reorder_matrix, calculate_gapped_soft_score


def test_get_landscape_dist_mat_weighted(binary_3bit_landscape):
    L = binary_3bit_landscape
    # attach weights to edges
    for u, v in L.graph.edges():
        L.graph[u][v]['weight'] = 2.0
    D = get_landscape_dist_mat(L, weighted=True)
    # diagonal zero, off-diagonals should be >= 0
    assert np.allclose(np.diag(D), 0.0)


def test_reorder_matrix_and_gapped_soft_score():
    Q = np.array([[1, 2], [3, 4]])
    alph = ['A', 'B']
    target = ['B', 'A']
    R = _reorder_matrix(Q, matrix_alphabet=alph, target_alphabet=target)
    assert np.allclose(R, np.array([[4, 3], [2, 1]]))

    # gapped soft score small example
    A = np.array([[1, 0, 0], [0, 1, 0]])
    B = np.array([[1, 0, 0], [1, 0, 0]])
    s = calculate_gapped_soft_score(A, B, q=np.array([[1, 0], [0, 1]]), gap_penalty=-1.0)
    # first site match score=1, second site mismatch score=0
    assert np.isclose(s, 1.0)

