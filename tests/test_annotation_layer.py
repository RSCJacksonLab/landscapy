import numpy as np
import networkx as nx
import pandas as pd
import pytest

from fitness_landscape.core.annotation import AnnotationLayer, register_auto_annotation
from fitness_landscape.core.landscape import FitnessLandscape
from fitness_landscape.core.sequence import BaseNumpySequence
from fitness_landscape.core.fitness import CategoricalFitness


def test_annotation_layer_query_supports_membership():
    data = {
        "superkingdom": ["A", "B", "A"],
        "kingdom": ["K1", "K1", "K2"],
    }
    layer = AnnotationLayer("taxonomy", data)

    assert len(layer) == 3
    assert layer.columns == ["superkingdom", "kingdom"]

    filtered = layer.query({"superkingdom": "A"})
    assert filtered.index.tolist() == [0, 2]
    assert filtered["kingdom"].tolist() == ["K1", "K2"]

    filtered_all = layer.query({"superkingdom": ["A", "B"]})
    assert len(filtered_all) == 3

    with pytest.raises(KeyError):
        layer.query({"phylum": "Chordata"})

    assert layer.matching_indices({"kingdom": ["K2"]}) == [2]


def test_annotation_layer_validates_inputs_and_supports_helpers():
    with pytest.raises(ValueError):
        AnnotationLayer("empty_df", pd.DataFrame())

    with pytest.raises(TypeError):
        AnnotationLayer("bad", 123)

    with pytest.raises(ValueError):
        AnnotationLayer("empty_map", {})

    with pytest.raises(ValueError):
        AnnotationLayer("empty_cols", {"a": []})

    with pytest.raises(ValueError):
        AnnotationLayer("mismatch", {"a": [1], "b": [1, 2]})

    layer = AnnotationLayer(
        "anno",
        {"row1": {"group": "A", "score": 1}, "row2": {"group": "B", "score": 2}},
        metadata={"source": "test"},
    )
    assert layer.metadata["source"] == "test"
    assert layer.get_record(1) == {"group": "B", "score": 2}

    with pytest.raises(IndexError):
        layer.get_record(2)

    with pytest.raises(ValueError, match="ctx"):
        layer.validate_length(3, context="ctx")

    frame = layer.to_dataframe(copy=False)
    assert frame is layer.to_dataframe(copy=False)
    assert layer.query(None).equals(layer.to_dataframe())
    assert layer.matching_indices({"group": np.array(["A"])}) == [0]

    with pytest.raises(KeyError):
        layer.query({"missing": "value"})


def test_register_auto_annotation_normalizes_records_and_metadata():
    graph = nx.Graph()

    register_auto_annotation(
        graph,
        "kind",
        {0: {"class": "A"}, 1: "B"},
        metadata={"origin": "unit"},
    )

    payload = graph.graph["_auto_annotations"]["kind"]
    assert payload["records"][0] == {"class": "A"}
    assert payload["records"][1] == {"kind": "B"}
    assert payload["metadata"] == {"origin": "unit"}


def test_fitness_landscape_annotation_query_and_detach():
    sequences = [
        BaseNumpySequence([0, 1, 0], sequence_id="s0"),
        BaseNumpySequence([0, 1, 1], sequence_id="s1"),
        BaseNumpySequence([1, 1, 0], sequence_id="s2"),
    ]

    graph = nx.Graph()
    graph.add_node("n0", sequence=sequences[0])
    graph.add_node("n1", sequence=sequences[1])
    graph.add_node("n2", sequence=sequences[2])
    graph.add_edge("n0", "n1")
    graph.add_edge("n1", "n2")

    landscape = FitnessLandscape(sequences=sequences, graph=graph)

    annotation_data = {
        "superkingdom": ["A", "B", "A"],
        "kingdom": ["K1", "K1", "K2"],
    }

    layer = landscape.attach_annotation(name="taxonomy", data=annotation_data)
    assert layer is landscape.get_annotation_layer("taxonomy")

    assert graph.nodes["n0"]["annotations"]["taxonomy"]["superkingdom"] == "A"
    assert graph.nodes["n1"]["annotations"]["taxonomy"]["kingdom"] == "K1"
    assert graph.nodes["n2"]["annotations"]["taxonomy"]["superkingdom"] == "A"

    res_superkingdom = landscape.query_annotations("taxonomy", {"superkingdom": "A"})
    assert res_superkingdom.sequence_indices == [0, 2]
    assert res_superkingdom.node_ids == ["n0", "n2"]
    assert res_superkingdom.edges == []
    assert res_superkingdom.dataframe.index.tolist() == [0, 2]
    assert res_superkingdom.dataframe.index.name == "sequence_index"
    assert res_superkingdom.sequences[0] is sequences[0]

    res_kingdom = landscape.query_annotations("taxonomy", {"kingdom": ["K1"]})
    assert res_kingdom.sequence_indices == [0, 1]
    assert set(res_kingdom.node_ids) == {"n0", "n1"}
    assert {frozenset(edge) for edge in res_kingdom.edges} == {frozenset(("n0", "n1"))}

    res_no_edges = landscape.query_annotations("taxonomy", {"kingdom": ["K1"]}, include_edges=False)
    assert res_no_edges.edges == []

    with pytest.raises(ValueError):
        landscape.attach_annotation(name="taxonomy", data=annotation_data)

    landscape.detach_annotation("taxonomy")
    assert "taxonomy" not in landscape.annotation_layers
    for node in graph.nodes:
        annotations = graph.nodes[node].get("annotations", {})
        assert "taxonomy" not in annotations

    with pytest.raises(KeyError):
        landscape.get_annotation_layer("taxonomy")


def test_attach_annotation_map_by_sequence():
    sequences = [
        BaseNumpySequence([0, 1, 0], sequence_id="s0"),
        BaseNumpySequence([0, 1, 1], sequence_id="s1"),
        BaseNumpySequence([1, 1, 0], sequence_id="s2"),
    ]

    graph = nx.Graph()
    for idx, seq in enumerate(sequences):
        graph.add_node(f"n{idx}", sequence=seq)

    landscape = FitnessLandscape(sequences=sequences, graph=graph)

    data = {
        "010": {"group": "X"},
        "011": {"group": "Y"},
        (1, 1, 0): {"group": "Z"},
    }

    layer = landscape.attach_annotation(name="grouping", data=data, map_by="sequence")
    assert layer.to_dataframe()["group"].tolist() == ["X", "Y", "Z"]

    for node, expected in zip(graph.nodes, ["X", "Y", "Z"]):
        assert graph.nodes[node]["annotations"]["grouping"]["group"] == expected


def test_attach_annotation_map_by_name_with_allow_missing():
    sequences = [
        BaseNumpySequence([0, 1, 0], sequence_id="Alpha"),
        BaseNumpySequence([0, 1, 1], sequence_id="Beta"),
        BaseNumpySequence([1, 1, 0], sequence_id="Gamma"),
    ]

    graph = nx.Graph()
    for idx, seq in enumerate(sequences):
        graph.add_node(f"n{idx}", sequence=seq)

    landscape = FitnessLandscape(sequences=sequences, graph=graph)

    data = {
        "Alpha": {"lineage": "L1"},
        "Gamma": {"lineage": "L2"},
    }

    layer = landscape.attach_annotation(
        name="lineage", data=data, map_by="name", allow_missing=True
    )

    df = layer.to_dataframe()
    assert df.loc[0, "lineage"] == "L1"
    assert pd.isna(df.loc[1, "lineage"])
    assert df.loc[2, "lineage"] == "L2"


def test_attach_annotation_map_by_index_records_allow_missing():
    sequences = [
        BaseNumpySequence([0, 1, 0], sequence_id="s0"),
        BaseNumpySequence([0, 1, 1], sequence_id="s1"),
        BaseNumpySequence([1, 1, 0], sequence_id="s2"),
    ]

    graph = nx.Graph()
    for idx, seq in enumerate(sequences):
        graph.add_node(f"n{idx}", sequence=seq)

    landscape = FitnessLandscape(sequences=sequences, graph=graph)

    data = {
        0: {"region": "north"},
        2: {"region": "south"},
    }

    layer = landscape.attach_annotation(
        name="region", data=data, map_by="index", allow_missing=True
    )

    df = layer.to_dataframe()
    assert df.loc[0, "region"] == "north"
    assert pd.isna(df.loc[1, "region"])
    assert df.loc[2, "region"] == "south"


def test_fitness_from_annotation_layer_defaults_categorical_and_handles_missing():
    layer = AnnotationLayer("anno", {"label": ["A", None, "B", np.nan]})
    seqs = [BaseNumpySequence([i]) for i in range(len(layer))]
    G = nx.Graph()
    for idx, seq in enumerate(seqs):
        G.add_node(idx, sequence=seq)
    fl = FitnessLandscape(seqs, graph=G, annotation_layers={"anno": layer})
    fit = fl.annotation_to_fitness("anno")
    assert isinstance(fit, CategoricalFitness)
    assert len(fit) == len(layer)
    assert "__missing__" in fit.categories
    missing_idx = fit.categories.index("__missing__")
    tensor = fit.get_tensor()
    assert tensor.shape[0] == len(layer)
    assert tensor[1, missing_idx] == 1.0
    assert tensor[3, missing_idx] == 1.0


def test_fitness_from_annotation_layer_requires_field_for_multi_column():
    layer = AnnotationLayer("anno", {"a": [1], "b": [2]})
    seqs = [BaseNumpySequence([0])]
    G = nx.Graph()
    for idx, seq in enumerate(seqs):
        G.add_node(idx, sequence=seq)
    fl = FitnessLandscape(seqs, graph=G, annotation_layers={"anno": layer})
    with pytest.raises(ValueError):
        fl.annotation_to_fitness("anno")
