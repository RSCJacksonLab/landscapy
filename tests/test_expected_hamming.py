import numpy as np

from fitness_landscape.core.graph import expected_hamming_from_aligned


def test_expected_hamming_pair_soft_with_gap():
    # Two aligned soft arrays with explicit gap channel at -1
    # L=3, A=2 (aa channels), last column is gap prob
    A = np.array([
        [1.0, 0.0, 0.0],  # site 0: AA0 certain, no gap
        [0.0, 1.0, 0.0],  # site 1: AA1 certain, no gap
        [0.5, 0.5, 0.0],  # site 2: equal mix, no gap
    ])
    B = np.array([
        [1.0, 0.0, 0.0],  # match
        [1.0, 0.0, 0.0],  # mismatch
        [0.0, 1.0, 0.0],  # mixture vs AA1
    ])
    mut, eff, dist = expected_hamming_from_aligned(A, B, gap_at=-1)
    assert eff == 3.0
    # site0: match (0); site1: mismatch (1); site2: 1 - (0.5*1 + 0.5*0) = 0.5
    assert np.isclose(mut, 1.5)
    assert np.isclose(dist, 0.5)


def test_expected_hamming_batch_soft_gapped_last():
    # Three sequences, aligned, with explicit gap channel last
    S = np.array([
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],  # [AA0, AA1]
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],  # [AA0, AA0]
        [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],  # [AA1, AA1]
    ])
    exp_mut, eff_len, dist = expected_hamming_from_aligned(S, return_norm=True)
    # Check shapes
    assert exp_mut.shape == (3, 3)
    assert eff_len.shape == (3, 3)
    assert dist.shape == (3, 3)
    # diagonal zero
    assert np.allclose(np.diag(dist), 0.0)

