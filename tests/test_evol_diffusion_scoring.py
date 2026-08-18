import numpy as np
from scipy.sparse import coo_matrix

from fitness_landscape._const import PROT_20
from fitness_landscape.core.graph import (
    _evolutionary_log_odds_matrix,
    _length_normalized_gapped_soft_score,
    _reversible_lazy_transition,
    _symmetric_affinity_from_scores,
    create_evol_diffusion_graph,
)
from fitness_landscape.core.edge_schema import EDGE_SCHEMA_GRAPH_KEY
from fitness_landscape.core.sequence import BaseNumpySequence
from fitness_landscape.phylo._sub_matrices import lg


def _gapped_one_hot(sequence: str) -> np.ndarray:
    encoded = np.zeros((len(sequence), len(PROT_20) + 1), dtype=np.float64)
    for position, residue in enumerate(sequence):
        if residue == "-":
            encoded[position, -1] = 1.0
        else:
            encoded[position, PROT_20.index(residue)] = 1.0
    return encoded


def _sequence(text: str, sequence_id: str) -> BaseNumpySequence:
    return BaseNumpySequence.from_string(
        text,
        alphabet=PROT_20,
        sequence_id=sequence_id,
    )


def _edge_weights_by_sequence_id(graph) -> dict[tuple[str, str], float]:
    weights = {}
    for node_a, node_b, data in graph.edges(data=True):
        sequence_a = graph.nodes[node_a]["sequence"].id
        sequence_b = graph.nodes[node_b]["sequence"].id
        key = tuple(sorted((sequence_a, sequence_b)))
        weights[key] = data["kernel_weight"]
    return weights


def test_lg_transition_log_odds_are_symmetric_and_similarity_oriented():
    score_matrix = _evolutionary_log_odds_matrix(lg, evolutionary_time=1.0)

    assert score_matrix.shape == (len(PROT_20), len(PROT_20))
    assert np.all(np.isfinite(score_matrix))
    assert np.allclose(score_matrix, score_matrix.T, atol=1e-12)

    off_diagonal = score_matrix.copy()
    np.fill_diagonal(off_diagonal, -np.inf)
    assert np.all(np.diag(score_matrix) > np.max(off_diagonal, axis=1))


def test_length_normalized_score_is_symmetric_ranked_and_length_invariant():
    score_matrix = _evolutionary_log_odds_matrix(lg, evolutionary_time=1.0)
    reference = _gapped_one_hot("AAAA")
    identical = _gapped_one_hot("AAAA")
    one_mutation = _gapped_one_hot("AAAR")
    divergent = _gapped_one_hot("RRRR")

    identity_score = _length_normalized_gapped_soft_score(
        reference, identical, score_matrix
    )
    mutation_score = _length_normalized_gapped_soft_score(
        reference, one_mutation, score_matrix
    )
    divergent_score = _length_normalized_gapped_soft_score(
        reference, divergent, score_matrix
    )
    reverse_score = _length_normalized_gapped_soft_score(
        one_mutation, reference, score_matrix
    )
    repeated_score = _length_normalized_gapped_soft_score(
        _gapped_one_hot("AAAAAAAA"),
        _gapped_one_hot("AAARAAAR"),
        score_matrix,
    )

    assert identity_score > mutation_score > divergent_score
    assert np.isclose(mutation_score, reverse_score)
    assert np.isclose(mutation_score, repeated_score)


def test_sparse_reversible_affinity_does_not_collapse_large_negative_scores():
    rows = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
    cols = np.array([1, 2, 0, 2, 0, 1], dtype=np.int32)
    values = np.array(
        [-400.0, -350.0, -400.0, -390.0, -350.0, -390.0],
        dtype=np.float64,
    )
    scores = coo_matrix((values, (rows, cols)), shape=(3, 3)).tocsr()

    affinity = _symmetric_affinity_from_scores(scores, tau=1.0)
    transition, stationary, _ = _reversible_lazy_transition(affinity)
    row_sums = np.asarray(transition.sum(axis=1)).ravel()
    flux = stationary[:, None] * transition.toarray()

    assert transition.dtype == np.float64
    assert transition.nnz > 0
    assert np.all(np.isfinite(transition.data))
    assert np.all(transition.data > 0.0)
    assert np.allclose(row_sums, 1.0)
    assert np.allclose(flux, flux.T)


def test_long_protein_construction_retains_edges_after_scoring():
    sequences = [
        _sequence("A" * 411, "identical"),
        _sequence(("A" * 410) + "R", "one_mutation"),
        _sequence(("A" * 409) + "RR", "two_mutations"),
    ]
    embeddings = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]])

    graph = create_evol_diffusion_graph(
        sequences,
        embeddings=embeddings,
        backend="balltree",
        k=2,
        t=1,
        tau=1.0,
        connectivity_threshold=0.01,
        cpus=1,
    )

    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 3
    assert graph.graph[EDGE_SCHEMA_GRAPH_KEY]["conductance"]["key"] == "weight"
    assert all(
        np.isfinite(data["kernel_weight"])
        and data["kernel_weight"] == data["affinity"] == data["weight"]
        and data["weight"] > 0.0
        for _, _, data in graph.edges(data=True)
    )


def test_graph_weights_do_not_depend_on_sequence_node_order():
    sequences = [
        _sequence("AAAA", "a"),
        _sequence("AAAR", "b"),
        _sequence("AARR", "c"),
    ]
    embeddings = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]])

    graph = create_evol_diffusion_graph(
        sequences,
        embeddings=embeddings,
        backend="balltree",
        k=2,
        t=1,
        connectivity_threshold=0.0,
        cpus=1,
    )

    order = [2, 0, 1]
    reordered_graph = create_evol_diffusion_graph(
        [sequences[index] for index in order],
        embeddings=embeddings[order],
        backend="balltree",
        k=2,
        t=1,
        connectivity_threshold=0.0,
        cpus=1,
    )

    expected = _edge_weights_by_sequence_id(graph)
    observed = _edge_weights_by_sequence_id(reordered_graph)
    assert expected.keys() == observed.keys()
    assert all(np.isclose(expected[key], observed[key]) for key in expected)
