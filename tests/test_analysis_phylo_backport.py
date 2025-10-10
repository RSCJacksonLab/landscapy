import numpy as np
import networkx as nx
import pytest

from fitness_landscape.analysis.graph_induction_alignment import procrustes
from fitness_landscape.analysis.statistics import hypothesis_testing
from fitness_landscape.graph_matching import (
    graph_to_length_matrix,
    normalize_adj_matrix,
)


def test_procrustes_identity_alignment():
    X = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    Y = X.copy()
    aligned, rmse = procrustes(X, Y)

    assert aligned.shape == X.shape
    assert rmse == pytest.approx(0.0, abs=1e-8)


def test_graph_to_length_matrix_returns_square_distance():
    G = nx.Graph()
    G.add_edge(0, 1, weight=2.0)
    lengths = graph_to_length_matrix(G, transform="reciprocal")

    assert lengths.shape == (2, 2)
    assert lengths[0, 1] == pytest.approx(0.5)
    assert lengths[0, 0] == 0.0


def test_normalize_adj_matrix_handles_sink_rows():
    G = nx.DiGraph()
    G.add_edge(0, 1, weight=2.0)
    G.add_node(2)  # sink row

    mat = normalize_adj_matrix(G)

    assert mat.shape == (3, 3)
    assert mat[2].tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_hypothesis_testing_basic_groups():
    groups = {
        "A": np.array([1.0, 2.0, 3.0]),
        "B": np.array([1.5, 2.0, 2.5]),
    }

    result = hypothesis_testing(groups=groups, run_tests=("ttest",))

    assert set(result.keys()) == {"group_stats", "pairwise_tests"}
    assert set(result["group_stats"].keys()) == {"A", "B"}
    assert "B" in result["pairwise_tests"]["A"]
    t_result = result["pairwise_tests"]["A"]["B"]["t_test"]
    assert {"statistic", "p_value", "significant"} <= t_result.keys()
