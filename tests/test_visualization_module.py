import pandas as pd
import networkx as nx
import numpy as np

from fitness_landscape.core.sequence import BaseNumpySequence
from fitness_landscape.core.fitness import NumericFitness
from fitness_landscape.core.landscape import FitnessLandscape
from fitness_landscape.visualization import (
    VisualizationDatasetBuilder,
    AnnotationRegistry,
    PaletteStore,
    LayoutSpec,
)
from fitness_landscape.visualization.adapters import (
    import_pct_annotations,
    register_pct_palette,
)


def _build_simple_landscape():
    seqs = [
        BaseNumpySequence([0, 1, 0], sequence_id="s0"),
        BaseNumpySequence([0, 1, 1], sequence_id="s1"),
        BaseNumpySequence([1, 1, 0], sequence_id="s2"),
    ]
    G = nx.Graph()
    for idx, seq in enumerate(seqs):
        G.add_node(f"n{idx}", sequence=seq)
    G.add_edges_from([("n0", "n1"), ("n1", "n2")])

    fitness = NumericFitness.from_scalars("score", [0.1, 0.5, 0.9])
    landscape = FitnessLandscape(sequences=seqs, graph=G, fitness_layers={"score": fitness})
    landscape.attach_annotation(
        name="taxonomy",
        data=[
            {"superkingdom": "A", "kingdom": "K1"},
            {"superkingdom": "B", "kingdom": "K1"},
            {"superkingdom": "A", "kingdom": "K2"},
        ],
        map_by="index",
    )
    return landscape


def test_builder_graph_layout_with_annotations():
    landscape = _build_simple_landscape()
    registry = AnnotationRegistry()
    palette_store = PaletteStore()

    builder = VisualizationDatasetBuilder(landscape, annotation_registry=registry)
    dataset = builder.build(
        layout=LayoutSpec("graph", {"seed": 0}),
        fitness_layer="score",
        annotation="taxonomy",
        palette_store=palette_store,
    )

    assert dataset.fitness_name == "score"
    np.testing.assert_allclose(dataset.fitness_values, np.array([0.1, 0.5, 0.9]))
    assert dataset.annotation_name == "taxonomy"
    assert set(dataset.annotation_values.keys()) == {"superkingdom", "kingdom"}
    assert dataset.positions.shape == (3, 2)
    assert len(dataset.edges) == 2


def test_builder_query_filters_nodes():
    landscape = _build_simple_landscape()
    builder = VisualizationDatasetBuilder(landscape)
    dataset = builder.build(
        layout="graph",
        annotation="taxonomy",
        query={"superkingdom": "A"},
    )
    assert dataset.nodes == ["n0", "n2"]
    np.testing.assert_allclose(dataset.fitness_values, np.array([0.1, 0.9]))
    assert all(val in ("A",) for val in dataset.annotation_values["superkingdom"])


def test_import_pct_annotations_and_palette_registration():
    landscape = _build_simple_landscape()
    registry = AnnotationRegistry()
    palette_store = PaletteStore()

    clusters = pd.DataFrame(
        {
            "Entry": ["s0", "s1", "s2"],
            "L1": ["c1", "c1", "c2"],
            "L2": ["c1a", "c1b", "c2a"],
        }
    )
    palette_dict = {
        "value": "L1",
        "method": "counts",
        "levels": ["L1"],
        "L1": {"id": ["c1", "c2"], "value": ["c1", "c2"]},
        "categories": {"c1": "rgba(1,2,3,4)", "c2": "rgba(5,6,7,8)"},
    }

    layer = import_pct_annotations(
        landscape,
        clusters,
        annotation_name="pct",
        registry=registry,
        palette_store=palette_store,
        palette=palette_dict,
    )

    assert layer.name == "pct"
    assert "pct" in registry
    descriptor = registry.get("pct")
    assert descriptor.source == "proteinclustertools"
    assert descriptor.palette_key == "pct:pct"
    assert palette_store.get_palette("pct:pct") == palette_dict

    builder = VisualizationDatasetBuilder(landscape, annotation_registry=registry)
    dataset = builder.build(
        layout="graph",
        annotation="pct",
        palette_store=palette_store,
    )
    assert dataset.annotation_name == "pct"
    assert "pct:pct" in dataset.palettes
    assert dataset.annotation_values["L1"] == ["c1", "c1", "c2"]


def test_register_pct_palette_returns_key():
    palette_store = PaletteStore()
    palette = {"example": "payload"}
    key = register_pct_palette(palette_store, "example_annot", palette)
    assert key == "example_annot:pct"
    assert palette_store.get_palette(key) == palette
