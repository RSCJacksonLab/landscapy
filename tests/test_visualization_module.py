import matplotlib

matplotlib.use("Agg")

import sys
import types
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pytest
import xml.etree.ElementTree as ET

from fitness_landscape.core.annotation import AnnotationLayer
from fitness_landscape.core.fitness import (
    CategoricalFitness,
    NumericFitness,
    ProbabilisticCategoricalFitness,
)
from fitness_landscape.core.landscape import FitnessLandscape
from fitness_landscape.core.sequence import BaseNumpySequence
from fitness_landscape.analysis.graph import resistance_distance_matrix
from fitness_landscape.visualization import (
    AnnotationRegistry,
    LayoutSpec,
    PaletteStore,
    VisualizationDatasetBuilder,
)
from fitness_landscape.visualization.adapters import (
    import_pct_annotations,
    register_pct_palette,
)
from fitness_landscape.visualization.dataset import VisualizationDataset
from fitness_landscape.visualization.renderers import (
    plot_landscape_matplotlib,
    resolve_node_colours,
)
from fitness_landscape.visualization.renderers.color_utils import (
    _categorical_colours,
    _default_colour_map,
    _fitness_colours,
    _mix_probability_colours,
    _normalize_colour,
    _palette_for_categories,
    _resolve_colour_mode,
    _to_rgba_tuple,
    rgba_to_plotly,
)
from fitness_landscape.visualization.renderers.plotly_renderer import (
    plot_landscape_plotly,
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


def _landscape_with_embeddings():
    landscape = _build_simple_landscape()
    base = np.arange(len(landscape.sequences) * 4, dtype=float).reshape(len(landscape.sequences), 4)
    landscape.embeddings = {
        "beta": base + 10.0,
        "alpha": base + 20.0,
    }
    return landscape


def _make_dataset(**overrides) -> VisualizationDataset:
    base = dict(
        nodes=["n0", "n1", "n2"],
        positions=np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]], dtype=float),
        edges=[("n0", "n1"), ("n1", "n2")],
        fitness_name="score",
        fitness_values=np.array([0.1, 0.5, 0.9], dtype=float),
        fitness_kind="numeric",
        annotation_name="labels",
        annotation_values={"group": ["A", "B", "A"]},
        palettes={},
        metadata={"title": "Demo"},
    )
    base.update(overrides)
    return VisualizationDataset(**base)


class _FakeScattergl:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


class _FakeFigure:
    def __init__(self):
        self.data = []
        self.layout = {}
        self.shown = False

    def add_trace(self, trace):
        self.data.append(trace)

    def update_layout(self, **kwargs):
        self.layout.update(kwargs)

    def show(self):
        self.shown = True


def _install_fake_plotly(monkeypatch):
    go = types.ModuleType("plotly.graph_objects")
    go.Figure = _FakeFigure
    go.Scattergl = _FakeScattergl

    plotly = types.ModuleType("plotly")
    plotly.__path__ = []
    plotly.graph_objects = go

    monkeypatch.setitem(sys.modules, "plotly", plotly)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", go)


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
    dense = resistance_distance_matrix(G, sparse_threshold=1000)["resistance_mat"]
    sparse = resistance_distance_matrix(G, sparse_threshold=1)["resistance_mat"]
    np.testing.assert_allclose(dense, sparse, atol=1e-8)


def test_landscape_plot_umap_with_emb_key():
    pytest.importorskip("umap")
    landscape = _landscape_with_embeddings()
    fig, ax = landscape.plot(layout="umap", emb_key="alpha", interactive=False, show=False)
    assert len(ax.collections) >= 1
    plt.close(fig)


def test_annotation_registry_and_palette_store_lifecycle():
    layer = AnnotationLayer("anno", {"group": ["A", "B"]})
    registry = AnnotationRegistry()
    descriptor = registry.register("anno", layer, source="test", metadata={"v": 1})

    assert descriptor.metadata["v"] == 1
    assert "anno" in registry
    assert len(registry) == 1
    assert list(registry.items())[0][0] == "anno"

    with pytest.raises(ValueError):
        registry.register("anno", layer)

    registry.update_palette("anno", "palette-key")
    assert registry.get("anno").palette_key == "palette-key"

    with pytest.raises(KeyError):
        registry.update_palette("missing", "palette")

    with pytest.raises(KeyError):
        registry.get("missing")

    registry.discard("anno")
    registry.discard("anno")
    assert len(registry) == 0

    store = PaletteStore()
    assert store.get_palette(None) is None
    store.register_palette("p", {"A": "#ff0000"})
    assert store.has_palette("p")
    assert store.get_palette("p") == {"A": "#ff0000"}
    store.clear()
    assert not store.has_palette("p")


def test_visualization_dataset_serializes_and_subsets():
    dataset = VisualizationDataset(
        nodes=["n0", "n1", "n2"],
        positions=np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]], dtype=float),
        edges=[("n0", "n1"), ("n1", "n2"), ("n0", "missing")],
        fitness_name="fit",
        fitness_values=np.array([1.0, 2.0, 3.0]),
        fitness_kind="probabilistic",
        fitness_categories=["A", "B"],
        fitness_labels=["A", "B", "A"],
        fitness_probabilities=np.array([[1.0, 0.0], [0.2, 0.8], [0.5, 0.5]]),
        annotation_name="anno",
        annotation_values={"group": ["x", "y", "z"]},
        palettes={"p": {"A": "#ff0000"}},
        metadata={"title": "Example"},
    )

    as_dict = dataset.to_dict()
    assert as_dict["positions"] == [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]
    assert as_dict["fitness_probabilities"][1] == [0.2, 0.8]
    assert as_dict["annotation_values"]["group"] == ["x", "y", "z"]

    subset = dataset.subset(np.array([True, False, True]))
    assert subset.nodes == ["n0", "n2"]
    assert subset.edges == []
    assert subset.fitness_labels == ["A", "A"]
    np.testing.assert_allclose(
        subset.fitness_probabilities,
        np.array([[1.0, 0.0], [0.5, 0.5]]),
    )

    no_fitness = VisualizationDataset(
        nodes=["n0"],
        positions=np.array([[0.0, 0.0]], dtype=float),
        annotation_values={"group": ["x"]},
    )
    subset_no_fitness = no_fitness.subset(np.array([True]))
    assert subset_no_fitness.fitness_values is None
    assert subset_no_fitness.fitness_probabilities is None

    with pytest.raises(ValueError):
        dataset.subset(np.array([True]))


def test_resolve_node_colours_falls_back_to_default_mode():
    empty = VisualizationDataset(nodes=["n0"], positions=np.array([[0.0, 0.0]], dtype=float))
    assert _resolve_colour_mode(empty, color_by="annotation") == "default"
    assert _resolve_colour_mode(empty, color_by="fitness") == "default"
    assert _resolve_colour_mode(empty, color_by="auto") == "default"

    annotation_only = VisualizationDataset(
        nodes=["n0"],
        positions=np.array([[0.0, 0.0]], dtype=float),
        annotation_values={"group": ["A"]},
    )
    assert _resolve_colour_mode(annotation_only, color_by="annotation") == "annotation"
    assert _resolve_colour_mode(annotation_only, color_by="fitness") == "annotation"

    fitness_only = VisualizationDataset(
        nodes=["n0"],
        positions=np.array([[0.0, 0.0]], dtype=float),
        fitness_values=np.array([1.0]),
    )
    assert _resolve_colour_mode(fitness_only, color_by="annotation") == "fitness"
    assert _resolve_colour_mode(fitness_only, color_by="fitness") == "fitness"

    colours, legend, is_continuous = resolve_node_colours(
        empty,
        annotation_field=None,
        palette_key=None,
        cmap="viridis",
    )
    assert not is_continuous
    assert legend == [("nodes", colours[0])]


def test_fitness_colour_helpers_cover_default_categorical_and_probabilistic_modes():
    default_colours, default_legend, default_continuous = _default_colour_map(2)
    assert len(default_colours) == 2
    assert default_legend[0][0] == "nodes"
    assert not default_continuous

    numeric_auto = VisualizationDataset(
        nodes=["n0"],
        positions=np.array([[0.0, 0.0]], dtype=float),
        fitness_values=np.array([1.0]),
    )
    colours, legend, is_continuous = _fitness_colours(
        numeric_auto,
        palette=None,
        cmap="viridis",
        categorical_cmap="Set2",
    )
    assert legend is None
    assert is_continuous
    np.testing.assert_allclose(colours, np.array([1.0]))

    numeric_missing = VisualizationDataset(
        nodes=["n0"],
        positions=np.array([[0.0, 0.0]], dtype=float),
        fitness_kind="numeric",
    )
    assert _fitness_colours(
        numeric_missing,
        palette=None,
        cmap="viridis",
        categorical_cmap="Set2",
    )[2] is False

    categorical_numpy = VisualizationDataset(
        nodes=["n0", "n1"],
        positions=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float),
        fitness_kind="categorical",
        fitness_values=np.array([0, 1]),
    )
    colours_np, legend_np, _ = _fitness_colours(
        categorical_numpy,
        palette=None,
        cmap="viridis",
        categorical_cmap="Set2",
    )
    assert len(colours_np) == 2
    assert [label for label, _ in legend_np] == ["0", "1"]

    categorical_list = VisualizationDataset(
        nodes=["n0", "n1"],
        positions=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float),
        fitness_kind="categorical",
        fitness_values=[0, 1],
    )
    colours_list, legend_list, _ = _fitness_colours(
        categorical_list,
        palette=None,
        cmap="viridis",
        categorical_cmap="Set2",
    )
    assert len(colours_list) == 2
    assert [label for label, _ in legend_list] == ["0", "1"]

    categorical_empty = VisualizationDataset(
        nodes=["n0"],
        positions=np.array([[0.0, 0.0]], dtype=float),
        fitness_kind="categorical",
    )
    assert _fitness_colours(
        categorical_empty,
        palette=None,
        cmap="viridis",
        categorical_cmap="Set2",
    )[2] is False

    probabilistic_empty = VisualizationDataset(
        nodes=["n0"],
        positions=np.array([[0.0, 0.0]], dtype=float),
        fitness_kind="probabilistic",
        fitness_categories=["A", "B"],
    )
    assert _fitness_colours(
        probabilistic_empty,
        palette=None,
        cmap="viridis",
        categorical_cmap="Set2",
    )[2] is False

    unknown = VisualizationDataset(
        nodes=["n0"],
        positions=np.array([[0.0, 0.0]], dtype=float),
        fitness_kind="mystery",
    )
    assert _fitness_colours(
        unknown,
        palette=None,
        cmap="viridis",
        categorical_cmap="Set2",
    )[2] is False


def test_categorical_colour_helpers_handle_palette_variants_and_probability_mixing():
    colours, legend = _categorical_colours(
        ["A", "B"],
        palette={"A": "#ff0000", "other": object()},
        cmap="Set2",
    )
    assert len(colours) == 2
    assert legend[-1][0] == "Other"

    colours_with_categories, legend_with_categories = _categorical_colours(
        ["A", "B"],
        palette={
            "categories": {"A": "rgb(255,0,0)"},
            "other": "rgba(0,255,0,0.5)",
        },
        cmap="Set2",
    )
    assert colours_with_categories[1] == pytest.approx((0.0, 1.0, 0.0, 0.5))
    assert legend_with_categories[-1][0] == "Other"

    colours_generated, legend_generated = _categorical_colours(
        ["A"],
        palette=None,
        cmap="TAB10",
    )
    assert len(colours_generated) == 1
    assert legend_generated[0][0] == "A"

    palette_map, palette_legend = _palette_for_categories(
        ["A", "B"],
        palette={"A": "#0000ff", "B": "#ff0000"},
        cmap="Set2",
    )
    assert set(palette_map) >= {"A", "B"}
    assert [label for label, _ in palette_legend] == ["A", "B"]

    with pytest.raises(ValueError):
        _mix_probability_colours(
            np.array([[1.0], [0.0]]),
            ["A", "B"],
            {"A": (1, 0, 0, 1)},
        )

    mixed = _mix_probability_colours(
        np.array([[0.5, 0.5]]),
        ["A", "B"],
        {"A": (1.0, 0.0, 0.0, 1.0)},
    )
    assert mixed[0] == pytest.approx((0.5, 0.0, 0.0, 0.5))

    probabilistic = VisualizationDataset(
        nodes=["n0"],
        positions=np.array([[0.0, 0.0]], dtype=float),
        fitness_kind="probabilistic",
        fitness_categories=["A", "B"],
        fitness_probabilities=np.array([[0.25, 0.75]]),
    )
    colours_prob, legend_prob, is_continuous = _fitness_colours(
        probabilistic,
        palette={"A": "#0000ff", "B": "#ff0000"},
        cmap="viridis",
        categorical_cmap="Set2",
    )
    assert not is_continuous
    assert legend_prob is not None
    assert colours_prob[0] == pytest.approx((0.75, 0.0, 0.25, 1.0))


def test_colour_normalization_helpers_validate_inputs():
    assert _normalize_colour("rgba(255,0,0,128)") == pytest.approx(
        (1.0, 0.0, 0.0, 128 / 255)
    )
    assert _normalize_colour("rgb(0,128,255)") == pytest.approx(
        (0.0, 128 / 255, 1.0, 1.0)
    )
    assert _normalize_colour("red") == pytest.approx((1.0, 0.0, 0.0, 1.0))
    assert _normalize_colour([0.2, 0.3, 0.4]) == pytest.approx((0.2, 0.3, 0.4, 1.0))
    assert _normalize_colour((0.1, 0.2, 0.3, 0.4)) == pytest.approx((0.1, 0.2, 0.3, 0.4))
    assert _to_rgba_tuple([0.1, 0.2, 0.3]) == pytest.approx((0.1, 0.2, 0.3, 1.0))
    assert rgba_to_plotly((0.5, 0.25, 0.0, 0.75)) == "rgba(128,64,0,0.75)"

    with pytest.raises(ValueError):
        _normalize_colour("rgba(1,2,3)")

    with pytest.raises(ValueError):
        _normalize_colour("rgb(1,2)")

    with pytest.raises(TypeError):
        _normalize_colour(object())


def test_plot_landscape_matplotlib_uses_existing_axes_and_skips_missing_edges(monkeypatch):
    dataset = _make_dataset(
        edges=[("n0", "n1"), ("missing", "n2")],
        fitness_kind=None,
        annotation_values={},
    )
    fig, ax = plt.subplots()
    called = {"show": False}

    monkeypatch.setattr(plt, "show", lambda: called.__setitem__("show", True))
    returned_fig, returned_ax = plot_landscape_matplotlib(dataset, ax=ax, show=True)

    assert returned_fig is fig
    assert returned_ax is ax
    assert called["show"] is True
    plt.close(fig)


def test_plot_landscape_plotly_requires_plotly(monkeypatch):
    monkeypatch.setitem(sys.modules, "plotly", None)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", None)

    with pytest.raises(RuntimeError, match="plotly is required"):
        plot_landscape_plotly(_make_dataset())


def test_plot_landscape_plotly_handles_numeric_and_categorical_branches(monkeypatch):
    _install_fake_plotly(monkeypatch)

    numeric = _make_dataset(edges=[("n0", "n1"), ("missing", "n2")], annotation_values={})
    fig = plot_landscape_plotly(numeric, color_by="fitness")
    assert len(fig.data) == 2
    assert fig.data[0].name == "edges"
    np.testing.assert_allclose(fig.data[1].marker["color"], np.array([0.1, 0.5, 0.9]))
    assert fig.layout["legend"]["title"] == ""

    categorical = _make_dataset(
        fitness_kind=None,
        fitness_values=None,
        annotation_values={"group": ["A", "B", "A"]},
        palettes={"grp": {"categories": {"A": "#ff0000", "B": "#00ff00"}}},
    )
    categorical_fig = plot_landscape_plotly(
        categorical,
        annotation_field="group",
        palette_key="grp",
        color_by="annotation",
        show=True,
    )
    assert categorical_fig.shown is True
    assert categorical_fig.layout["legend"]["title"] == "labels"
    assert [trace.name for trace in categorical_fig.data[1:]] == ["nodes", "A", "B"]


def test_plot_landscape_plotly_uses_fitness_legend_without_annotations(monkeypatch):
    _install_fake_plotly(monkeypatch)
    dataset = _make_dataset(
        fitness_kind="categorical",
        fitness_values=None,
        fitness_labels=["low", "high", "low"],
        annotation_values={},
    )
    fig = plot_landscape_plotly(dataset, color_by="annotation")
    assert fig.layout["legend"]["title"] == "score"


def test_builder_validates_layout_inputs_and_embedding_edge_cases():
    landscape = _build_simple_landscape()
    builder = VisualizationDatasetBuilder(landscape)
    nodes = ["n0", "n1", "n2"]

    with pytest.raises(TypeError):
        builder._normalise_layout(123)

    with pytest.raises(ValueError):
        builder.build(layout="graph", query={"superkingdom": "A"})

    with pytest.raises(ValueError):
        builder.build(layout="external")

    with pytest.raises(ValueError):
        builder.build(layout="unknown")

    spring = builder._graph_layout(nodes, {"engine": "spring", "seed": 0})
    assert spring.shape == (3, 2)

    with pytest.raises(ValueError):
        builder._graph_layout(nodes, {"engine": "bogus"})

    dataset = builder.build(layout="graph", include_edges=False)
    assert dataset.edges == []

    landscape.embeddings = np.arange(6, dtype=float).reshape(3, 2)
    matrix, key = builder._get_embedding_matrix(None)
    assert key == "default"
    np.testing.assert_array_equal(matrix, landscape.embeddings)

    with pytest.raises(KeyError):
        builder._get_embedding_matrix("missing")

    landscape.embeddings = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        builder._get_embedding_matrix(None)

    landscape.embeddings = np.array([[1.0], [2.0], [3.0]])
    with pytest.raises(ValueError):
        builder._get_embedding_matrix(None)

    landscape.embeddings = {}
    with pytest.raises(ValueError):
        builder._get_embedding_matrix(None)

    landscape.embeddings = "bad"
    with pytest.raises(TypeError):
        builder._get_embedding_matrix(None)

    landscape.embeddings = None
    with pytest.raises(ValueError):
        builder._get_embedding_matrix(None)

    with pytest.raises(KeyError):
        builder._external_layout(["n0", "n1"], {"n0": (0.0, 0.0)})

    with pytest.raises(ValueError):
        builder._external_layout(["n0"], {"n0": (0.0,)})

    coords = builder._external_layout(["n0"], {"n0": (1.0, 2.0, 3.0)})
    np.testing.assert_allclose(coords, np.array([[1.0, 2.0]]))


def test_builder_graph_layout_falls_back_when_graphviz_is_unavailable(monkeypatch):
    landscape = _build_simple_landscape()
    builder = VisualizationDatasetBuilder(landscape)

    monkeypatch.setattr(builder, "_graphviz_sfdp_available", lambda: False)
    monkeypatch.setattr(
        builder,
        "_graphviz_sfdp_layout",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected graphviz call")),
    )

    dataset = builder.build(layout=LayoutSpec("graph", {"seed": 0}))
    assert dataset.positions.shape == (3, 2)


def test_builder_graphviz_layout_and_embedding_error_branches(monkeypatch):
    landscape = _build_simple_landscape()
    builder = VisualizationDatasetBuilder(landscape)

    empty = builder._graphviz_sfdp_layout(nx.Graph(), [])
    np.testing.assert_allclose(empty, np.zeros((0, 2)))

    import networkx.drawing.nx_pydot as nx_pydot

    def fake_graphviz_layout(graph, prog="sfdp", args=""):
        assert prog == "sfdp"
        assert args == "-Grankdir=LR"
        return {node: idx for idx, node in enumerate(graph.nodes())}

    monkeypatch.setattr(nx_pydot, "graphviz_layout", fake_graphviz_layout)
    coords = builder._graphviz_sfdp_layout(
        landscape.graph.subgraph(["n0", "n1"]),
        ["n0", "n1"],
        args="-Grankdir=LR",
    )
    assert coords.shape == (2, 2)
    np.testing.assert_allclose(coords[:, 1], np.zeros(2))

    def raising_graphviz_layout(graph, prog="sfdp", args=""):
        raise RuntimeError("boom")

    monkeypatch.setattr(nx_pydot, "graphviz_layout", raising_graphviz_layout)
    with pytest.raises(RuntimeError, match="Graphviz 'sfdp' layout failed"):
        builder._graphviz_sfdp_layout(landscape.graph.subgraph(["n0", "n1"]), ["n0", "n1"])

    monkeypatch.setattr(builder, "_graphviz_sfdp_layout", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="fallback is disabled"):
        builder._graph_layout(["n0", "n1"], {"engine": "sfdp"})

    landscape.embeddings = {"default": np.arange(6, dtype=float).reshape(3, 2)}
    with pytest.raises(KeyError):
        builder._embedding_layout(["missing"], {"emb_key": "default"})


def test_annotation_registry_and_palette_store_support_updates_and_deletes():
    layer = AnnotationLayer("anno", {"group": ["A", "B"]})
    registry = AnnotationRegistry()
    descriptor = registry.register("anno", layer, source="test", metadata={"v": 1})
    assert descriptor.metadata["v"] == 1
    assert "anno" in registry
    assert len(registry) == 1
    assert list(registry.items())[0][0] == "anno"

    with pytest.raises(ValueError):
        registry.register("anno", layer)

    registry.update_palette("anno", "palette-key")
    assert registry.get("anno").palette_key == "palette-key"

    with pytest.raises(KeyError):
        registry.update_palette("missing", "palette")

    with pytest.raises(KeyError):
        registry.get("missing")

    registry.discard("anno")
    registry.discard("anno")
    assert len(registry) == 0

    store = PaletteStore()
    assert store.get_palette(None) is None
    store.register_palette("p", {"A": "#ff0000"})
    assert store.has_palette("p")
    assert store.get_palette("p") == {"A": "#ff0000"}
    store.clear()
    assert not store.has_palette("p")


def test_visualization_dataset_to_dict_and_subset_preserve_aligned_fields():
    dataset = VisualizationDataset(
        nodes=["n0", "n1", "n2"],
        positions=np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]], dtype=float),
        edges=[("n0", "n1"), ("n1", "n2"), ("n0", "missing")],
        fitness_name="fit",
        fitness_values=np.array([1.0, 2.0, 3.0]),
        fitness_kind="probabilistic",
        fitness_categories=["A", "B"],
        fitness_labels=["A", "B", "A"],
        fitness_probabilities=np.array([[1.0, 0.0], [0.2, 0.8], [0.5, 0.5]]),
        annotation_name="anno",
        annotation_values={"group": ["x", "y", "z"]},
        palettes={"p": {"A": "#ff0000"}},
        metadata={"title": "Example"},
    )
    as_dict = dataset.to_dict()
    assert as_dict["positions"] == [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]
    assert as_dict["fitness_probabilities"][1] == [0.2, 0.8]
    assert as_dict["annotation_values"]["group"] == ["x", "y", "z"]

    subset = dataset.subset(np.array([True, False, True]))
    assert subset.nodes == ["n0", "n2"]
    assert subset.edges == []
    assert subset.fitness_labels == ["A", "A"]
    np.testing.assert_allclose(
        subset.fitness_probabilities,
        np.array([[1.0, 0.0], [0.5, 0.5]]),
    )

    no_fitness = VisualizationDataset(
        nodes=["n0"],
        positions=np.array([[0.0, 0.0]], dtype=float),
        annotation_values={"group": ["x"]},
    )
    subset2 = no_fitness.subset(np.array([True]))
    assert subset2.fitness_values is None
    assert subset2.fitness_probabilities is None

    with pytest.raises(ValueError):
        dataset.subset(np.array([True]))


def test_resolve_node_colours_falls_back_between_annotation_fitness_and_default():
    empty = VisualizationDataset(
        nodes=["n0"],
        positions=np.array([[0.0, 0.0]], dtype=float),
    )
    assert _resolve_colour_mode(empty, color_by="annotation") == "default"
    assert _resolve_colour_mode(empty, color_by="fitness") == "default"
    assert _resolve_colour_mode(empty, color_by="auto") == "default"

    annotation_only = VisualizationDataset(
        nodes=["n0"],
        positions=np.array([[0.0, 0.0]], dtype=float),
        annotation_values={"group": ["A"]},
    )
    assert _resolve_colour_mode(annotation_only, color_by="annotation") == "annotation"
    assert _resolve_colour_mode(annotation_only, color_by="fitness") == "annotation"

    fitness_only = VisualizationDataset(
        nodes=["n0"],
        positions=np.array([[0.0, 0.0]], dtype=float),
        fitness_values=np.array([1.0]),
    )
    assert _resolve_colour_mode(fitness_only, color_by="annotation") == "fitness"
    assert _resolve_colour_mode(fitness_only, color_by="fitness") == "fitness"

    colours, legend, is_continuous = resolve_node_colours(
        empty,
        annotation_field=None,
        palette_key=None,
        cmap="viridis",
    )
    assert not is_continuous
    assert legend == [("nodes", colours[0])]


def test_fitness_colour_helpers_cover_numeric_categorical_and_invalid_inputs():
    default_colours, default_legend, default_continuous = _default_colour_map(2)
    assert len(default_colours) == 2
    assert default_legend[0][0] == "nodes"
    assert not default_continuous

    numeric_auto = VisualizationDataset(
        nodes=["n0"],
        positions=np.array([[0.0, 0.0]], dtype=float),
        fitness_values=np.array([1.0]),
    )
    colours, legend, is_continuous = _fitness_colours(
        numeric_auto,
        palette=None,
        cmap="viridis",
        categorical_cmap="Set2",
    )
    assert legend is None
    assert is_continuous
    np.testing.assert_allclose(colours, np.array([1.0]))

    numeric_missing = VisualizationDataset(
        nodes=["n0"],
        positions=np.array([[0.0, 0.0]], dtype=float),
        fitness_kind="numeric",
    )
    assert _fitness_colours(
        numeric_missing,
        palette=None,
        cmap="viridis",
        categorical_cmap="Set2",
    )[2] is False

    categorical_numpy = VisualizationDataset(
        nodes=["n0", "n1"],
        positions=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float),
        fitness_kind="categorical",
        fitness_values=np.array([0, 1]),
    )
    colours_np, legend_np, _ = _fitness_colours(
        categorical_numpy,
        palette=None,
        cmap="viridis",
        categorical_cmap="Set2",
    )
    assert len(colours_np) == 2
    assert [label for label, _ in legend_np] == ["0", "1"]

    categorical_list = VisualizationDataset(
        nodes=["n0", "n1"],
        positions=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float),
        fitness_kind="categorical",
        fitness_values=[0, 1],
    )
    colours_list, legend_list, _ = _fitness_colours(
        categorical_list,
        palette=None,
        cmap="viridis",
        categorical_cmap="Set2",
    )
    assert len(colours_list) == 2
    assert [label for label, _ in legend_list] == ["0", "1"]

    categorical_empty = VisualizationDataset(
        nodes=["n0"],
        positions=np.array([[0.0, 0.0]], dtype=float),
        fitness_kind="categorical",
    )
    assert _fitness_colours(
        categorical_empty,
        palette=None,
        cmap="viridis",
        categorical_cmap="Set2",
    )[2] is False

    probabilistic_empty = VisualizationDataset(
        nodes=["n0"],
        positions=np.array([[0.0, 0.0]], dtype=float),
        fitness_kind="probabilistic",
        fitness_categories=["A", "B"],
    )
    assert _fitness_colours(
        probabilistic_empty,
        palette=None,
        cmap="viridis",
        categorical_cmap="Set2",
    )[2] is False

    unknown = VisualizationDataset(
        nodes=["n0"],
        positions=np.array([[0.0, 0.0]], dtype=float),
        fitness_kind="mystery",
    )
    assert _fitness_colours(
        unknown,
        palette=None,
        cmap="viridis",
        categorical_cmap="Set2",
    )[2] is False


def test_categorical_colour_helpers_apply_palettes_and_mix_probabilities():
    colours, legend = _categorical_colours(
        ["A", "B"],
        palette={"A": "#ff0000", "other": object()},
        cmap="Set2",
    )
    assert len(colours) == 2
    assert legend[-1][0] == "Other"

    colours2, legend2 = _categorical_colours(
        ["A", "B"],
        palette={
            "categories": {"A": "rgb(255,0,0)"},
            "other": "rgba(0,255,0,0.5)",
        },
        cmap="Set2",
    )
    assert colours2[1] == pytest.approx((0.0, 1.0, 0.0, 0.5))
    assert legend2[-1][0] == "Other"

    colours3, legend3 = _categorical_colours(["A"], palette=None, cmap="TAB10")
    assert len(colours3) == 1
    assert legend3[0][0] == "A"

    palette_map, palette_legend = _palette_for_categories(
        ["A", "B"],
        palette={"A": "#0000ff", "B": "#ff0000"},
        cmap="Set2",
    )
    assert set(palette_map) >= {"A", "B"}
    assert [label for label, _ in palette_legend] == ["A", "B"]

    with pytest.raises(ValueError):
        _mix_probability_colours(
            np.array([[1.0], [0.0]]),
            ["A", "B"],
            {"A": (1, 0, 0, 1)},
        )

    mixed = _mix_probability_colours(
        np.array([[0.5, 0.5]]),
        ["A", "B"],
        {"A": (1.0, 0.0, 0.0, 1.0)},
    )
    assert mixed[0] == pytest.approx((0.5, 0.0, 0.0, 0.5))

    probabilistic = VisualizationDataset(
        nodes=["n0"],
        positions=np.array([[0.0, 0.0]], dtype=float),
        fitness_kind="probabilistic",
        fitness_categories=["A", "B"],
        fitness_probabilities=np.array([[0.25, 0.75]]),
    )
    colours4, legend4, is_continuous = _fitness_colours(
        probabilistic,
        palette={"A": "#0000ff", "B": "#ff0000"},
        cmap="viridis",
        categorical_cmap="Set2",
    )
    assert not is_continuous
    assert legend4 is not None
    assert colours4[0] == pytest.approx((0.75, 0.0, 0.25, 1.0))


def test_colour_normalization_utilities_parse_supported_formats_and_errors():
    assert _normalize_colour("rgba(255,0,0,128)") == pytest.approx(
        (1.0, 0.0, 0.0, 128 / 255),
    )
    assert _normalize_colour("rgb(0,128,255)") == pytest.approx(
        (0.0, 128 / 255, 1.0, 1.0),
    )
    assert _normalize_colour("red") == pytest.approx((1.0, 0.0, 0.0, 1.0))
    assert _normalize_colour([0.2, 0.3, 0.4]) == pytest.approx((0.2, 0.3, 0.4, 1.0))
    assert _normalize_colour((0.1, 0.2, 0.3, 0.4)) == pytest.approx(
        (0.1, 0.2, 0.3, 0.4),
    )
    assert _to_rgba_tuple([0.1, 0.2, 0.3]) == pytest.approx((0.1, 0.2, 0.3, 1.0))
    assert rgba_to_plotly((0.5, 0.25, 0.0, 0.75)) == "rgba(128,64,0,0.75)"

    with pytest.raises(ValueError):
        _normalize_colour("rgba(1,2,3)")

    with pytest.raises(ValueError):
        _normalize_colour("rgb(1,2)")

    with pytest.raises(TypeError):
        _normalize_colour(object())


def test_plot_landscape_matplotlib_uses_existing_axes_and_skips_missing_edges(monkeypatch):
    dataset = _make_dataset(
        edges=[("n0", "n1"), ("missing", "n2")],
        fitness_kind=None,
        annotation_values={},
    )
    fig, ax = plt.subplots()
    called = {"show": False}

    monkeypatch.setattr(plt, "show", lambda: called.__setitem__("show", True))
    returned_fig, returned_ax = plot_landscape_matplotlib(dataset, ax=ax, show=True)
    assert returned_fig is fig
    assert returned_ax is ax
    assert called["show"] is True
    plt.close(fig)


def test_plot_landscape_plotly_raises_when_plotly_is_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "plotly", None)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", None)

    with pytest.raises(RuntimeError, match="plotly is required"):
        plot_landscape_plotly(_make_dataset())


def test_plot_landscape_plotly_supports_numeric_and_annotation_legends(monkeypatch):
    _install_fake_plotly(monkeypatch)

    numeric = _make_dataset(edges=[("n0", "n1"), ("missing", "n2")], annotation_values={})
    fig = plot_landscape_plotly(numeric, color_by="fitness")
    assert len(fig.data) == 2
    assert fig.data[0].name == "edges"
    np.testing.assert_allclose(fig.data[1].marker["color"], np.array([0.1, 0.5, 0.9]))
    assert fig.layout["legend"]["title"] == ""

    categorical = _make_dataset(
        fitness_kind=None,
        fitness_values=None,
        annotation_values={"group": ["A", "B", "A"]},
        palettes={"grp": {"categories": {"A": "#ff0000", "B": "#00ff00"}}},
    )
    fig2 = plot_landscape_plotly(
        categorical,
        annotation_field="group",
        palette_key="grp",
        color_by="annotation",
        show=True,
    )
    assert fig2.shown is True
    assert fig2.layout["legend"]["title"] == "labels"
    assert [trace.name for trace in fig2.data[1:]] == ["nodes", "A", "B"]


def test_plot_landscape_plotly_uses_fitness_legend_when_annotations_are_absent(monkeypatch):
    _install_fake_plotly(monkeypatch)
    dataset = _make_dataset(
        fitness_kind="categorical",
        fitness_values=None,
        fitness_labels=["low", "high", "low"],
        annotation_values={},
    )
    fig = plot_landscape_plotly(dataset, color_by="annotation")
    assert fig.layout["legend"]["title"] == "score"


def test_visualization_builder_validates_layout_and_embedding_inputs():
    landscape = _build_simple_landscape()
    builder = VisualizationDatasetBuilder(landscape)

    with pytest.raises(TypeError):
        builder._normalise_layout(123)

    with pytest.raises(ValueError):
        builder.build(layout="graph", query={"superkingdom": "A"})

    with pytest.raises(ValueError):
        builder.build(layout="external")

    with pytest.raises(ValueError):
        builder.build(layout="unknown")

    spring = builder._graph_layout(["n0", "n1", "n2"], {"engine": "spring", "seed": 0})
    assert spring.shape == (3, 2)

    with pytest.raises(ValueError):
        builder._graph_layout(["n0", "n1", "n2"], {"engine": "bogus"})

    dataset = builder.build(layout="graph", include_edges=False)
    assert dataset.edges == []

    landscape.embeddings = np.arange(6, dtype=float).reshape(3, 2)
    matrix, key = builder._get_embedding_matrix(None)
    assert key == "default"
    np.testing.assert_array_equal(matrix, landscape.embeddings)

    with pytest.raises(KeyError):
        builder._get_embedding_matrix("missing")

    landscape.embeddings = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        builder._get_embedding_matrix(None)

    landscape.embeddings = np.array([[1.0], [2.0], [3.0]])
    with pytest.raises(ValueError):
        builder._get_embedding_matrix(None)

    landscape.embeddings = {}
    with pytest.raises(ValueError):
        builder._get_embedding_matrix(None)

    landscape.embeddings = "bad"
    with pytest.raises(TypeError):
        builder._get_embedding_matrix(None)

    landscape.embeddings = None
    with pytest.raises(ValueError):
        builder._get_embedding_matrix(None)

    with pytest.raises(KeyError):
        builder._external_layout([0, 1], {0: (0.0, 0.0)})

    with pytest.raises(ValueError):
        builder._external_layout([0], {0: (0.0,)})

    coords = builder._external_layout([0], {0: (1.0, 2.0, 3.0)})
    np.testing.assert_allclose(coords, np.array([[1.0, 2.0]]))


def test_visualization_builder_graph_layout_falls_back_when_graphviz_is_unavailable(monkeypatch):
    landscape = _build_simple_landscape()
    builder = VisualizationDatasetBuilder(landscape)

    monkeypatch.setattr(builder, "_graphviz_sfdp_available", lambda: False)
    monkeypatch.setattr(
        builder,
        "_graphviz_sfdp_layout",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected graphviz call")),
    )

    dataset = builder.build(layout=LayoutSpec("graph", {"seed": 0}))
    assert dataset.positions.shape == (3, 2)


def test_visualization_builder_graphviz_and_embedding_helpers_cover_error_paths(monkeypatch):
    landscape = _build_simple_landscape()
    builder = VisualizationDatasetBuilder(landscape)

    empty = builder._graphviz_sfdp_layout(nx.Graph(), [])
    np.testing.assert_allclose(empty, np.zeros((0, 2)))

    import networkx.drawing.nx_pydot as nx_pydot

    def fake_graphviz_layout(graph, prog="sfdp", args=""):
        assert prog == "sfdp"
        assert args == "-Grankdir=LR"
        return {node: idx for idx, node in enumerate(graph.nodes())}

    monkeypatch.setattr(nx_pydot, "graphviz_layout", fake_graphviz_layout)
    coords = builder._graphviz_sfdp_layout(
        landscape.graph.subgraph(["n0", "n1"]),
        ["n0", "n1"],
        args="-Grankdir=LR",
    )
    assert coords.shape == (2, 2)
    np.testing.assert_allclose(coords[:, 1], np.zeros(2))

    def raising_graphviz_layout(graph, prog="sfdp", args=""):
        raise RuntimeError("boom")

    monkeypatch.setattr(nx_pydot, "graphviz_layout", raising_graphviz_layout)
    with pytest.raises(RuntimeError, match="Graphviz 'sfdp' layout failed"):
        builder._graphviz_sfdp_layout(landscape.graph.subgraph(["n0", "n1"]), ["n0", "n1"])

    monkeypatch.setattr(builder, "_graphviz_sfdp_layout", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="fallback is disabled"):
        builder._graph_layout(["n0", "n1"], {"engine": "sfdp"})

    landscape.embeddings = {"default": np.arange(6, dtype=float).reshape(3, 2)}
    with pytest.raises(KeyError):
        builder._embedding_layout(["missing"], {"emb_key": "default"})
