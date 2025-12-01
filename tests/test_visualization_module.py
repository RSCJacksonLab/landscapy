import matplotlib

matplotlib.use("Agg")

import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pytest
import xml.etree.ElementTree as ET

from fitness_landscape.core.sequence import BaseNumpySequence
from fitness_landscape.core.fitness import (
    NumericFitness,
    CategoricalFitness,
    ProbabilisticCategoricalFitness,
)
from fitness_landscape.core.annotation import AnnotationLayer
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
from fitness_landscape.visualization.renderers import plot_landscape_matplotlib, resolve_node_colours
from fitness_landscape.analysis.graph import resistance_distance_matrix


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


def _landscape_with_embeddings():
    landscape = _build_simple_landscape()
    base = np.arange(len(landscape.sequences) * 4, dtype=float).reshape(len(landscape.sequences), 4)
    landscape.embeddings = {
        "beta": base + 10.0,
        "alpha": base + 20.0,
    }
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

    fig, ax = plot_landscape_matplotlib(
        dataset,
        annotation_field="L1",
        palette_key="pct:pct",
    )
    assert ax.legend_ is not None
    labels = [text.get_text() for text in ax.legend_.get_texts()]
    assert "c1" in labels
    plt.close(fig)


def test_register_pct_palette_returns_key():
    palette_store = PaletteStore()
    palette = {"example": "payload"}
    key = register_pct_palette(palette_store, "example_annot", palette)
    assert key == "example_annot:pct"
    assert palette_store.get_palette(key) == palette


def test_plot_matplotlib_with_numeric_fitness_only():
    landscape = _build_simple_landscape()
    builder = VisualizationDatasetBuilder(landscape)
    dataset = builder.build(layout="graph", fitness_layer="score")
    fig, ax = plot_landscape_matplotlib(dataset)
    # Expect a single scatter collection
    assert len(ax.collections) == 1
    scatter = ax.collections[0]
    assert scatter.get_offsets().shape[0] == len(dataset.nodes)
    plt.close(fig)


def test_landscape_plot_accepts_categorical_cmap_passthrough():
    landscape = _build_simple_landscape()
    fig, ax = landscape.plot(
        fitness_layer="score",
        color_by="fitness",
        categorical_cmap="Pastel1",
        interactive=False,
        show=False,
    )
    assert len(ax.collections) >= 1
    plt.close(fig)


def test_resolve_node_colours_categorical_fitness_overrides_annotation():
    landscape = _build_simple_landscape()
    cat_layer = CategoricalFitness.from_values("cat", ["low", "medium", "high"])
    landscape.fitness_layers["cat"] = cat_layer
    builder = VisualizationDatasetBuilder(landscape)
    dataset = builder.build(layout="graph", fitness_layer="cat", annotation="taxonomy")

    colours, legend, is_continuous = resolve_node_colours(
        dataset,
        annotation_field=None,
        palette_key=None,
        cmap="viridis",
        color_by="fitness",
    )

    assert not is_continuous
    assert len(colours) == len(dataset.nodes)
    assert legend is not None
    labels = {label for label, _ in legend}
    assert labels == set(cat_layer.categories)


def test_probabilistic_fitness_colour_mixing():
    landscape = _build_simple_landscape()
    probs = np.array([[0.75, 0.25], [0.0, 1.0], [0.5, 0.5]])
    prob_layer = ProbabilisticCategoricalFitness.from_probabilities(
        "prob",
        probabilities=probs,
        categories=["A", "B"],
    )
    landscape.fitness_layers["prob"] = prob_layer

    builder = VisualizationDatasetBuilder(landscape)
    dataset = builder.build(layout="graph", fitness_layer="prob")
    palette = {"A": "#0000ff", "B": "#ff0000"}

    colours, legend, is_continuous = resolve_node_colours(
        dataset,
        annotation_field=None,
        palette_key=None,
        palette=palette,
        cmap="viridis",
        categorical_cmap="Set2",
        color_by="fitness",
    )

    assert not is_continuous
    assert legend is not None
    assert {label for label, _ in legend} == {"A", "B"}
    np.testing.assert_allclose(colours[0], np.array([0.25, 0.0, 0.75, 1.0]))
    np.testing.assert_allclose(colours[1], np.array([1.0, 0.0, 0.0, 1.0]))


def test_diffusion_layout_positions():
    landscape = _build_simple_landscape()
    builder = VisualizationDatasetBuilder(landscape)
    dataset = builder.build(layout=LayoutSpec("diffusion", {"dimensions": 2}))
    assert dataset.positions.shape == (3, 2)


def test_builder_embedding_matrix_default_key():
    landscape = _landscape_with_embeddings()
    builder = VisualizationDatasetBuilder(landscape)
    matrix, key = builder._get_embedding_matrix(None)
    assert key == "beta"
    np.testing.assert_array_equal(matrix, landscape.embeddings["beta"])


def test_umap_layout_defaults_to_first_embedding_key():
    pytest.importorskip("umap")
    landscape = _landscape_with_embeddings()
    builder = VisualizationDatasetBuilder(landscape)
    dataset = builder.build(layout="umap")
    assert dataset.positions.shape == (3, 2)


def test_landscape_plot_method_matplotlib():
    landscape = _build_simple_landscape()
    fig, ax = landscape.plot(interactive=False, show=False)
    assert len(ax.collections) >= 1
    plt.close(fig)


def test_landscape_plot_method_plotly():
    plotly = pytest.importorskip("plotly")  # noqa: F841
    landscape = _build_simple_landscape()
    fig = landscape.plot(interactive=True, show=False)
    assert fig.data


def test_export_xgmml_includes_annotations(tmp_path):
    landscape = _build_simple_landscape()
    path = tmp_path / "graph.xgmml"
    landscape.export_xgmml(path, annotation_layers=["taxonomy"])
    assert path.exists()
    ns = {"x": "http://www.cs.rpi.edu/XGMML"}
    tree = ET.parse(path)
    root = tree.getroot()
    nodes = root.findall("x:node", ns)
    assert len(nodes) == 3
    attr_map = {
        att.get("name"): att.get("value")
        for att in nodes[0].findall("x:att", ns)
    }
    assert attr_map["taxonomy::superkingdom"] in {"A", "B"}
    assert "fitness::score" in attr_map


def test_get_components_preserves_layers():
    seqs = [
        BaseNumpySequence([0], sequence_id="s0"),
        BaseNumpySequence([1], sequence_id="s1"),
        BaseNumpySequence([2], sequence_id="s2"),
    ]
    G = nx.Graph()
    G.add_node("n0", sequence=seqs[0])
    G.add_node("n1", sequence=seqs[1])
    G.add_node("n2", sequence=seqs[2])
    G.add_edge("n0", "n1")

    fitness = NumericFitness.from_scalars("score", [0.1, 0.2, 0.3])
    annotations = AnnotationLayer(name="labels", data={"label": ["A", "B", "C"]})

    landscape = FitnessLandscape(
        sequences=seqs,
        graph=G,
        fitness_layers={"score": fitness},
        annotation_layers={"labels": annotations},
    )

    components = landscape.get_components()
    assert len(components) == 2
    assert components[0].graph.number_of_nodes() == 2
    assert components[1].graph.number_of_nodes() == 1

    comp0_scores = components[0].fitness_layers["score"].to_scalar()
    np.testing.assert_allclose(comp0_scores, np.array([0.1, 0.2]))

    comp1_scores = components[1].fitness_layers["score"].to_scalar()
    np.testing.assert_allclose(comp1_scores, np.array([0.3]))

    labels = components[1].annotation_layers["labels"].to_dataframe()["label"].tolist()
    assert labels == ["C"]


def test_resistance_distance_matrix_sparse_switch():
    G = nx.path_graph(4)
    dense = resistance_distance_matrix(G, sparse_threshold=1000)
    sparse = resistance_distance_matrix(G, sparse_threshold=1)
    np.testing.assert_allclose(dense, sparse, atol=1e-8)


def test_landscape_plot_umap_with_emb_key():
    pytest.importorskip("umap")
    landscape = _landscape_with_embeddings()
    fig, ax = landscape.plot(layout="umap", emb_key="alpha", interactive=False, show=False)
    assert len(ax.collections) >= 1
    plt.close(fig)
