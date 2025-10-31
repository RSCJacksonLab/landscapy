import networkx as nx
import pytest

from fitness_landscape.core.annotation import AnnotationLayer
from fitness_landscape.core.landscape import FitnessLandscape
from fitness_landscape.core.sequence import BaseNumpySequence


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
