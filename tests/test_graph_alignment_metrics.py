import networkx as nx
import pytest

np = pytest.importorskip("numpy")

from fitness_landscape.analysis.graph_induction_alignment import (
    compare_density_matched_geometry_to_phylogeny,
    compare_pairwise_rankings_to_phylogeny,
)


def _toy_graphs():
    nodes = ["A", "B", "C"]
    sequences = {name: np.array([idx], dtype=float) for idx, name in enumerate(nodes)}

    phy = nx.Graph()
    for node in nodes:
        phy.add_node(node, sequence=sequences[node])
    phy.add_edge("A", "B", branch_length=1.0)
    phy.add_edge("B", "C", branch_length=1.0)

    diffusion = nx.Graph()
    for node in nodes:
        diffusion.add_node(node, sequence=sequences[node])
    diffusion.add_edge("A", "B", sim=0.9)
    diffusion.add_edge("B", "C", sim=0.8)
    diffusion.add_edge("A", "C", sim=0.1)

    knn = nx.Graph()
    for node in nodes:
        knn.add_node(node, sequence=sequences[node])
    knn.add_edge("A", "C", weight=1.0)

    return diffusion, knn, phy


def test_pairwise_ranking_prefers_diffusion():
    diffusion, knn, phy = _toy_graphs()

    result = compare_pairwise_rankings_to_phylogeny(
        diffusion,
        knn,
        phy,
        tree_k_for_labels=1,
        diffusion_weight_key="sim",
        knn_weight_key="weight",
    )

    assert result["n_nodes"] == 3
    assert len(result["pairs"]) == 3
    assert result["diffusion"]["average_precision"] > result["knn"]["average_precision"]
    assert result["diffusion"]["roc_auc"] > result["knn"]["roc_auc"]
    assert result["labels"] == [1, 0, 1]


def test_density_matched_geometry_diffusion_outperforms_knn():
    diffusion, knn, phy = _toy_graphs()

    result = compare_density_matched_geometry_to_phylogeny(
        diffusion,
        knn,
        phy,
        k_values=[1, 2],
        diffusion_weight_key="sim",
        knn_weight_key="weight",
    )

    assert result["n_nodes"] == 3
    assert [row["k"] for row in result["rows"]] == [1, 2]

    k1, k2 = result["rows"]

    assert k1["precision_at_k_diffusion"] == pytest.approx(2.0 / 3.0, rel=1e-6, abs=1e-6)
    assert k1["precision_at_k_knn"] == pytest.approx(0.0, rel=1e-6, abs=1e-6)

    assert k2["precision_at_k_diffusion"] == pytest.approx(1.0, rel=1e-6, abs=1e-6)
    assert k2["precision_at_k_knn"] == pytest.approx(1.0 / 3.0, rel=1e-6, abs=1e-6)
    assert k2["precision_at_k_diffusion"] > k2["precision_at_k_knn"]
