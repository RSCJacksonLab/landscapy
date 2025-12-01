import numpy as np

from fitness_landscape.core.landscape import FitnessLandscape
from fitness_landscape.core.fitness import (
    NumericFitness,
    CategoricalFitness,
    ProbabilisticCategoricalFitness,
)
from fitness_landscape.core.annotation import AnnotationLayer
from fitness_landscape.core.sequence import generate_sequences


def _make_base_layers():
    seqs = generate_sequences(length=2, alphabet=[0, 1])
    num = NumericFitness.from_scalars("num", [1, 3, 5, 7])
    cat = CategoricalFitness.from_values("cat", ["X", "X", "Y", "Y"], categories=["X", "Y"])
    prob = ProbabilisticCategoricalFitness.from_probabilities(
        "prob",
        probabilities=np.array(
            [
                [0.2, 0.8],
                [0.5, 0.5],
                [0.9, 0.1],
                [0.8, 0.2],
            ],
            dtype=float,
        ),
        categories=["X", "Y"],
    )
    annotations = AnnotationLayer(
        name="communities",
        data={
            "community": ["A", "A", "B", "B"],
            "family": ["f1", "f1", "f2", "f3"],
        },
    )
    landscape = FitnessLandscape.build(
        seqs,
        graph="hamming",
        fitness_layers={"num": num, "cat": cat, "prob": prob},
        annotation_layers={"communities": annotations},
    )
    # Tag edges with a numeric attribute for aggregation checks.
    for u, v in landscape.graph.edges():
        landscape.graph[u][v]["w"] = float(u + v)
    return landscape


def test_quotient_landscape_annotation_partition():
    landscape = _make_base_layers()

    quotient = landscape.quotient_landscape(
        partition="communities",
        annotation_field="community",
    )

    assert len(quotient.sequences) == 2

    num_layer = quotient.fitness_layers["num"]
    assert np.allclose(num_layer.to_scalar(), [2.0, 6.0])

    cat_layer = quotient.fitness_layers["cat"]
    assert isinstance(cat_layer, ProbabilisticCategoricalFitness)
    assert np.allclose(cat_layer.probabilities, [[1.0, 0.0], [0.0, 1.0]])

    prob_layer = quotient.fitness_layers["prob"]
    expected_prob = np.array([[0.35, 0.65], [0.85, 0.15]])
    assert np.allclose(prob_layer.probabilities, expected_prob)

    # Aggregated annotations should be present and deduplicated per block.
    assert "communities" in quotient.annotation_layers
    ann_df = quotient.annotation_layers["communities"].to_dataframe()
    assert ann_df["community"].tolist() == ["A", "B"]
    assert ann_df["family"].tolist() == ["f1", "f2;f3"]

    # Edge attributes are aggregated across inter-block edges.
    assert quotient.graph.number_of_nodes() == 2
    assert quotient.graph.number_of_edges() == 1
    (u, v, data) = next(iter(quotient.graph.edges(data=True)))
    assert {u, v} == {0, 1}
    assert data["w"] == 3.0


def test_quotient_landscape_custom_partition_and_no_annotation_aggregation():
    seqs = generate_sequences(length=2, alphabet=[0, 1])
    num = NumericFitness.from_replicates(
        "num",
        [[1, 2], [10, 12], [20, 22], [30, 32]],
    )
    landscape = FitnessLandscape.build(seqs, graph="hamming", fitness_layers={"num": num})
    for u, v in landscape.graph.edges():
        landscape.graph[u][v]["score"] = float(u * v)

    labels = ["left", "left", "right", "right"]
    quotient = landscape.quotient_landscape(
        partition=labels,
        aggregate_annotations=False,
        aggregation_function=lambda arr: np.nanmax(arr),
        edge_aggregation_function="median",
    )

    assert not quotient.annotation_layers
    num_layer = quotient.fitness_layers["num"]
    assert np.allclose(num_layer.to_scalar(), [12.0, 32.0])

    (u, v, data) = next(iter(quotient.graph.edges(data=True)))
    assert data["score"] == 1.5
