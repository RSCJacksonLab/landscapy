import sys
import types

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pytest

from fitness_landscape.core.annotation import AnnotationLayer, register_auto_annotation
from fitness_landscape.core.fitness import NumericFitness
from fitness_landscape.core.landscape import FitnessLandscape
from fitness_landscape.core.sequence import BaseNumpySequence
from fitness_landscape.graph_matching.isorank import (
    cosine_similarity_matrix,
    isorank_with_features,
    normalize_adj_matrix,
)
from fitness_landscape.visualization.builder import (
    LayoutSpec,
    VisualizationDatasetBuilder,
)
from fitness_landscape.visualization.dataset import VisualizationDataset
from fitness_landscape.visualization.registry import AnnotationRegistry, PaletteStore
from fitness_landscape.visualization.renderers.color_utils import (
    _categorical_colours,
    _default_colour_map,
    _fitness_colours,
    _mix_probability_colours,
    _normalize_colour,
    _palette_for_categories,
    _resolve_colour_mode,
    _to_rgba_tuple,
    resolve_node_colours,
    rgba_to_plotly,
)
from fitness_landscape.visualization.renderers.matplotlib_renderer import (
    plot_landscape_matplotlib,
)
from fitness_landscape.visualization.renderers.plotly_renderer import (
    plot_landscape_plotly,
)


def _build_landscape(n: int = 3) -> FitnessLandscape:
    seqs = [BaseNumpySequence([idx], sequence_id=f"s{idx}") for idx in range(n)]
    graph = nx.path_graph(n)
    for idx, seq in enumerate(seqs):
        graph.nodes[idx]["sequence"] = seq

    landscape = FitnessLandscape(
        sequences=seqs,
        graph=graph,
        fitness_layers={"score": NumericFitness.from_scalars("score", np.linspace(0.1, 0.9, n))},
    )
    landscape.attach_annotation(
        name="labels",
        data={"group": ["A", "B", "A"][:n], "rank": list(range(n))},
        map_by="index",
    )
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


def test_annotation_layer_error_paths_and_helpers():
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


def test_annotation_registry_and_palette_store_branches():
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


def test_visualization_dataset_to_dict_and_subset_branches():
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
    np.testing.assert_allclose(subset.fitness_probabilities, np.array([[1.0, 0.0], [0.5, 0.5]]))

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


def test_isorank_normalize_and_similarity_branches():
    with pytest.raises(TypeError):
        normalize_adj_matrix("bad")

    empty = normalize_adj_matrix(nx.Graph())
    assert empty.shape == (0, 0)

    graph = nx.DiGraph()
    graph.add_nodes_from([0, 1, 2])
    graph.add_edge(0, 1)
    mat = normalize_adj_matrix(graph)
    np.testing.assert_allclose(mat[1], np.array([1 / 3, 1 / 3, 1 / 3]))
    np.testing.assert_allclose(mat[2], np.array([1 / 3, 1 / 3, 1 / 3]))

    sim = cosine_similarity_matrix(np.eye(2), np.eye(2))
    np.testing.assert_allclose(np.diag(sim), np.ones(2))


def test_isorank_with_features_handles_validation_and_uniform_prior():
    g1 = nx.path_graph(2)
    g2 = nx.path_graph(2)

    with pytest.raises(ValueError):
        isorank_with_features(g1, g2, np.ones((1, 2)), np.ones((2, 2)))

    zero_prior = isorank_with_features(
        g1,
        g2,
        np.zeros((2, 1)),
        np.zeros((2, 1)),
        alpha=0.7,
        max_iter=5,
        tol=0.0,
    )
    np.testing.assert_allclose(zero_prior, np.full((2, 2), 0.25))

    aligned = isorank_with_features(
        g1,
        g2,
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        alpha=0.5,
        max_iter=20,
        tol=1e9,
    )
    assert aligned.shape == (2, 2)
    assert aligned[0, 0] > aligned[0, 1]


def test_colour_mode_and_default_resolution_branches():
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


def test_fitness_colour_helpers_cover_fallback_modes():
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
        numeric_auto, palette=None, cmap="viridis", categorical_cmap="Set2"
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
        numeric_missing, palette=None, cmap="viridis", categorical_cmap="Set2"
    )[2] is False

    categorical_numpy = VisualizationDataset(
        nodes=["n0", "n1"],
        positions=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float),
        fitness_kind="categorical",
        fitness_values=np.array([0, 1]),
    )
    colours_np, legend_np, _ = _fitness_colours(
        categorical_numpy, palette=None, cmap="viridis", categorical_cmap="Set2"
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
        categorical_list, palette=None, cmap="viridis", categorical_cmap="Set2"
    )
    assert len(colours_list) == 2
    assert [label for label, _ in legend_list] == ["0", "1"]

    categorical_empty = VisualizationDataset(
        nodes=["n0"],
        positions=np.array([[0.0, 0.0]], dtype=float),
        fitness_kind="categorical",
    )
    assert _fitness_colours(
        categorical_empty, palette=None, cmap="viridis", categorical_cmap="Set2"
    )[2] is False

    probabilistic_empty = VisualizationDataset(
        nodes=["n0"],
        positions=np.array([[0.0, 0.0]], dtype=float),
        fitness_kind="probabilistic",
        fitness_categories=["A", "B"],
    )
    assert _fitness_colours(
        probabilistic_empty, palette=None, cmap="viridis", categorical_cmap="Set2"
    )[2] is False

    unknown = VisualizationDataset(
        nodes=["n0"],
        positions=np.array([[0.0, 0.0]], dtype=float),
        fitness_kind="mystery",
    )
    assert _fitness_colours(
        unknown, palette=None, cmap="viridis", categorical_cmap="Set2"
    )[2] is False


def test_categorical_colour_palettes_and_probability_mixing():
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
        _mix_probability_colours(np.array([[1.0], [0.0]]), ["A", "B"], {"A": (1, 0, 0, 1)})

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


def test_colour_normalization_utilities_and_errors():
    assert _normalize_colour("rgba(255,0,0,128)") == pytest.approx((1.0, 0.0, 0.0, 128 / 255))
    assert _normalize_colour("rgb(0,128,255)") == pytest.approx((0.0, 128 / 255, 1.0, 1.0))
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


def test_plot_landscape_matplotlib_with_existing_axes_and_skipped_edges(monkeypatch):
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
    monkeypatch.delitem(sys.modules, "plotly", raising=False)
    monkeypatch.delitem(sys.modules, "plotly.graph_objects", raising=False)
    with pytest.raises(RuntimeError, match="plotly is required"):
        plot_landscape_plotly(_make_dataset())


def test_plot_landscape_plotly_numeric_and_categorical_branches(monkeypatch):
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


def test_visualization_builder_error_and_layout_branches():
    landscape = _build_landscape()
    builder = VisualizationDatasetBuilder(landscape)

    with pytest.raises(TypeError):
        builder._normalise_layout(123)

    with pytest.raises(ValueError):
        builder.build(layout="graph", query={"group": "A"})

    with pytest.raises(ValueError):
        builder.build(layout="external")

    with pytest.raises(ValueError):
        builder.build(layout="unknown")

    spring = builder._graph_layout([0, 1, 2], {"engine": "spring", "seed": 0})
    assert spring.shape == (3, 2)

    with pytest.raises(ValueError):
        builder._graph_layout([0, 1, 2], {"engine": "bogus"})

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


def test_visualization_builder_graphviz_and_embedding_branches(monkeypatch):
    landscape = _build_landscape()
    builder = VisualizationDatasetBuilder(landscape)

    empty = builder._graphviz_sfdp_layout(nx.Graph(), [])
    np.testing.assert_allclose(empty, np.zeros((0, 2)))

    import networkx.drawing.nx_pydot as nx_pydot

    def fake_graphviz_layout(graph, prog="sfdp", args=""):
        assert prog == "sfdp"
        assert args == "-Grankdir=LR"
        return {node: idx for idx, node in enumerate(graph.nodes())}

    monkeypatch.setattr(nx_pydot, "graphviz_layout", fake_graphviz_layout)
    coords = builder._graphviz_sfdp_layout(landscape.graph.subgraph([0, 1]), [0, 1], args="-Grankdir=LR")
    assert coords.shape == (2, 2)
    np.testing.assert_allclose(coords[:, 1], np.zeros(2))

    def raising_graphviz_layout(graph, prog="sfdp", args=""):
        raise RuntimeError("boom")

    monkeypatch.setattr(nx_pydot, "graphviz_layout", raising_graphviz_layout)
    with pytest.raises(RuntimeError, match="Graphviz 'sfdp' layout failed"):
        builder._graphviz_sfdp_layout(landscape.graph.subgraph([0, 1]), [0, 1])

    monkeypatch.setattr(builder, "_graphviz_sfdp_layout", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="fallback is disabled"):
        builder._graph_layout([0, 1], {"engine": "sfdp"})

    landscape.embeddings = {"default": np.arange(6, dtype=float).reshape(3, 2)}
    with pytest.raises(KeyError):
        builder._embedding_layout([99], {"emb_key": "default"})
