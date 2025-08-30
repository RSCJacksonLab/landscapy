import numpy as np
import pytest

from fitness_landscape.core.sequence import BaseNumpySequence


def test_to_one_hot_gap_alias_and_case():
    # Alphabet includes 'gap'; lowercase and '-' should map to same index
    alph = ['A', 'B', 'gap']
    s = BaseNumpySequence.from_string('a-B-', alphabet=alph)
    M = s.to_one_hot()
    # length 4, 3 classes
    assert M.shape == (4, 3)
    # positions 0: 'a' -> A; 1: '-' -> gap; 2: 'B' -> B; 3: '-' -> gap
    idx = np.argmax(M, axis=1)
    assert list(idx) == [0, 2, 1, 2]


def test_distance_and_remove_gap_arr():
    alph = ['A', 'B', 'gap']
    s1 = BaseNumpySequence.from_string('AB-', alphabet=alph)
    s2 = BaseNumpySequence.from_string('A--', alphabet=alph)
    # Hamming on raw chars
    assert s1.distance(s2) == 1.0
    # remove_gap_arr drops gap channel and renormalises
    ungapped = s1.remove_gap_arr(gap_threshold=0.5)
    assert ungapped.shape[1] == 2


def test_from_integer_and_one_hot_roundtrip():
    alph = ['X', 'Y', 'Z']
    ints = [0, 2, 1, 2]
    s = BaseNumpySequence.from_integer(ints, alphabet=alph)
    oh = s.to_one_hot()
    s2 = BaseNumpySequence.from_one_hot(oh, alphabet=alph)
    assert s.to_str() == s2.to_str()

