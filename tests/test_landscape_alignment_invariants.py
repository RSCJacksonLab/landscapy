"""Regression tests for canonical landscape row alignment."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from fitness_landscape.core.annotation import AnnotationLayer
from fitness_landscape.core.fitness import CategoricalFitness, NumericFitness
from fitness_landscape.core.landscape import FitnessLandscape
from fitness_landscape.core.sequence import BaseNumpySequence, BinarySequence


def _aligned_graph(sequences, nodes=None):
    nodes = list(range(len(sequences))) if nodes is None else nodes
    graph = nx.Graph()
    for node, sequence in zip(nodes, sequences):
        graph.add_node(node, sequence=sequence)
    return graph


def _unique_landscape(nodes=None):
    sequences = [
        BinarySequence([0, 0]),
        BinarySequence([0, 1]),
        BinarySequence([1, 1]),
    ]
    return FitnessLandscape(sequences, _aligned_graph(sequences, nodes))


def _duplicate_landscape():
    sequences = [
        BinarySequence([0, 0]),
        BinarySequence([0, 0]),
        BinarySequence([0, 1]),
    ]
    graph = _aligned_graph(sequences, ["first", 19, ("last", 1)])
    with pytest.warns(UserWarning, match="Duplicate sequences"):
        return FitnessLandscape(sequences, graph)


def test_constructor_establishes_one_mapping_for_all_aligned_domains():
    sequences = [
        BaseNumpySequence(["A"]),
        BaseNumpySequence(["B"]),
        BaseNumpySequence(["C"]),
    ]
    nodes = ["alpha", 19, ("node", 3)]
    graph = _aligned_graph(sequences, nodes)
    fitness = NumericFitness("score", [[1.0], [2.0], [3.0]])
    annotations = AnnotationLayer("group", {"label": ["x", "y", "z"]})
    embeddings = {
        "latent": np.arange(6, dtype=float).reshape(3, 2),
        "one-dimensional": np.arange(3, dtype=float).reshape(3, 1),
    }

    landscape = FitnessLandscape(
        sequences,
        graph,
        fitness_layers={"score": fitness},
        annotation_layers={"group": annotations},
        embeddings=embeddings,
        active_embedding_domain="latent",
    )

    assert landscape.node_to_sequence_index == {
        "alpha": 0,
        19: 1,
        ("node", 3): 2,
    }
    assert landscape.sequence_index_to_node == {
        0: "alpha",
        1: 19,
        2: ("node", 3),
    }
    for index, node in enumerate(nodes):
        assert graph.nodes[node]["fitness_score"] == [float(index + 1)]
        assert graph.nodes[node]["annotations"]["group"]["label"] == "xyz"[index]
        np.testing.assert_array_equal(
            graph.nodes[node]["emb_arr"], embeddings["latent"][index]
        )

    # Mapping accessors cannot be used to mutate the canonical mapping.
    landscape.node_to_sequence_index["alpha"] = 99
    assert landscape.node_to_sequence_index["alpha"] == 0


def test_build_resolves_reordered_graph_and_rejects_unmatched_sequences():
    sequences = [BinarySequence([0]), BinarySequence([1])]
    graph = _aligned_graph(list(reversed(sequences)), ["left", "right"])
    fitness = NumericFitness("score", [[10.0], [20.0]])

    landscape = FitnessLandscape.build(
        sequences,
        graph=graph,
        fitness_layers={"score": fitness},
    )

    assert landscape.node_to_sequence_index == {"left": 1, "right": 0}
    assert graph.nodes["left"]["fitness_score"] == [20.0]
    assert graph.nodes["right"]["fitness_score"] == [10.0]

    mismatched = _aligned_graph(
        [sequences[0], BaseNumpySequence([2])], ["left", "right"]
    )
    with pytest.raises(ValueError, match="no matching provided sequence"):
        FitnessLandscape.build(sequences, graph=mismatched)

    assert all("fitness_score" not in data for _, data in mismatched.nodes(data=True))


def test_constructor_rejects_graph_count_and_sequence_attribute_mismatches():
    sequences = [BinarySequence([0]), BinarySequence([1])]

    with pytest.raises(ValueError, match="number of provided sequences"):
        FitnessLandscape(sequences, _aligned_graph(sequences[:1]))

    missing = nx.Graph()
    missing.add_nodes_from(["a", "b"])
    missing.nodes["a"]["sequence"] = sequences[0]
    with pytest.raises(ValueError, match="missing its 'sequence'"):
        FitnessLandscape(sequences, missing)

    invalid = _aligned_graph(sequences)
    invalid.nodes[1]["sequence"] = [1]
    with pytest.raises(ValueError, match="invalid 'sequence'"):
        FitnessLandscape(sequences, invalid)


def test_constructor_rejects_every_misaligned_layer_domain():
    sequences = [BinarySequence([0]), BinarySequence([1])]
    graph = _aligned_graph(sequences)

    with pytest.raises(ValueError, match="landscape construction"):
        FitnessLandscape(
            sequences,
            graph,
            fitness_layers={"short": NumericFitness("short", [[1.0]])},
        )

    with pytest.raises(ValueError, match="landscape construction"):
        FitnessLandscape(
            sequences,
            graph,
            annotation_layers={"short": AnnotationLayer("short", {"x": [1]})},
        )

    with pytest.raises(ValueError, match="'bad'.*1 rows; expected 2"):
        FitnessLandscape(
            sequences,
            graph,
            embeddings={
                "good": np.zeros((2, 3)),
                "bad": np.zeros((1, 4)),
            },
        )


@pytest.mark.parametrize("dtype", ["numeric", "categorical"])
def test_default_duplicate_policy_succeeds_for_unique_sequences(dtype):
    landscape = _unique_landscape(["a", 7, ("c",)])
    kwargs = {}
    values = {"00": 1.0, "01": 2.0, "11": 3.0}
    if dtype == "categorical":
        values = {"00": "low", "01": "middle", "11": "high"}
        kwargs["categories"] = ["low", "middle", "high"]

    landscape.attach(
        name=f"unique_{dtype}",
        values=values,
        dtype=dtype,
        map_by="sequence",
        **kwargs,
    )

    assert f"unique_{dtype}" in landscape.fitness_layers
    assert all(
        f"fitness_unique_{dtype}" in data
        for _, data in landscape.graph.nodes(data=True)
    )


@pytest.mark.parametrize("dtype", ["numeric", "categorical"])
def test_error_duplicate_policy_fails_only_on_actual_duplicates(dtype):
    landscape = _duplicate_landscape()
    values = {"00": 1.0, "01": 2.0}
    kwargs = {}
    if dtype == "categorical":
        values = {"00": "same", "01": "other"}
        kwargs["categories"] = ["same", "other"]

    with pytest.raises(ValueError, match="Duplicate sequences"):
        landscape.attach(
            name=f"error_{dtype}",
            values=values,
            dtype=dtype,
            map_by="sequence",
            on_duplicates="error",
            **kwargs,
        )


def test_duplicate_first_all_and_aggregate_policies_preserve_node_rows():
    landscape = _duplicate_landscape()

    landscape.attach(
        name="first",
        values={"00": [1.0, 2.0], "01": 3.0},
        dtype="numeric",
        map_by="sequence",
        on_duplicates="first",
        allow_missing=True,
    )
    assert landscape.graph.nodes["first"]["fitness_first"] == [1.0, 2.0]
    assert np.isnan(landscape.graph.nodes[19]["fitness_first"][0])
    assert landscape.graph.nodes[("last", 1)]["fitness_first"] == [3.0]

    landscape.attach(
        name="all",
        values={"00": "same", "01": "other"},
        dtype="categorical",
        categories=["same", "other"],
        map_by="sequence",
        on_duplicates="all",
    )
    assert landscape.graph.nodes["first"]["fitness_all"] == "same"
    assert landscape.graph.nodes[19]["fitness_all"] == "same"
    assert landscape.graph.nodes[("last", 1)]["fitness_all"] == "other"

    landscape.attach(
        name="aggregate",
        values={"00": [4.0, 6.0], "01": [8.0]},
        dtype="numeric",
        map_by="sequence",
        on_duplicates="aggregate",
    )
    # Duplicate nodes share the supplied replicate sample; duplicates do not
    # manufacture extra observations by repeating that sample.
    assert landscape.graph.nodes["first"]["fitness_aggregate"] == [4.0, 6.0]
    assert landscape.graph.nodes[19]["fitness_aggregate"] == [4.0, 6.0]

    with pytest.raises(ValueError, match="not supported for categorical"):
        landscape.attach(
            name="bad_aggregate",
            values={"00": "same", "01": "other"},
            dtype="categorical",
            categories=["same", "other"],
            map_by="sequence",
            on_duplicates="aggregate",
        )


def test_ready_layers_annotations_queries_and_detach_use_exact_node_rows():
    landscape = _duplicate_landscape()
    layer = NumericFitness("distinct", [[1.0], [9.0], [3.0]])
    landscape.attach(layer)

    assert landscape.graph.nodes["first"]["fitness_distinct"] == [1.0]
    assert landscape.graph.nodes[19]["fitness_distinct"] == [9.0]

    annotation = AnnotationLayer("row", {"label": ["first", "second", "last"]})
    landscape.attach_annotation(annotation)
    result = landscape.query_annotations("row", {"label": "second"})
    assert result.sequence_indices == [1]
    assert result.node_ids == [19]

    landscape.detach("distinct")
    assert all(
        "fitness_distinct" not in data
        for _, data in landscape.graph.nodes(data=True)
    )


def test_duplicate_input_keys_are_rejected_instead_of_silently_overwritten():
    landscape = _unique_landscape()

    with pytest.raises(ValueError, match="Multiple input values"):
        landscape.attach(
            name="ambiguous",
            values=[("00", 1.0), ((0, 0), 2.0), ("01", 3.0), ("11", 4.0)],
            dtype="numeric",
            map_by="sequence",
        )
