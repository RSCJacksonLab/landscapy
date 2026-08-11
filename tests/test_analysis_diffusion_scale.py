import networkx as nx
import numpy as np
import pytest

import fitness_landscape.analysis.diffusion_scale as diffusion_scale_module
from fitness_landscape.analysis.diffusion_scale import compute_ruggedness_diffusion_scale
from fitness_landscape.core.fitness import NumericFitness
from fitness_landscape.core.landscape import FitnessLandscape
from fitness_landscape.core.sequence import BaseNumpySequence
from fitness_landscape.transforms.eigenmode import eigenmode_decomposition


def _toy_landscape():
    seqs = [
        BaseNumpySequence([0], sequence_id="s0"),
        BaseNumpySequence([1], sequence_id="s1"),
        BaseNumpySequence([2], sequence_id="s2"),
        BaseNumpySequence([3], sequence_id="s3"),
        BaseNumpySequence([4], sequence_id="s4"),
    ]
    G = nx.path_graph(len(seqs))
    for idx, seq in enumerate(seqs):
        G.nodes[idx]["sequence"] = seq

    fitness = NumericFitness(name="fitness", values=[0.0, 1.0, 0.5, 1.5, 1.0])
    return FitnessLandscape(sequences=seqs, graph=G, fitness_layers={"fitness": fitness})


def _two_layer_landscape():
    seqs = [
        BaseNumpySequence([0], sequence_id="s0"),
        BaseNumpySequence([1], sequence_id="s1"),
        BaseNumpySequence([2], sequence_id="s2"),
    ]
    graph = nx.Graph()
    for node, sequence_index in [("third", 2), ("first", 0), ("second", 1)]:
        graph.add_node(node, sequence=seqs[sequence_index])
    graph.add_edges_from([("first", "second"), ("second", "third")])
    layers = {
        "active": NumericFitness.from_scalars("active", [10.0, 20.0, 30.0]),
        "selected": NumericFitness.from_scalars("selected", [1.0, 2.0, 4.0]),
    }
    return FitnessLandscape(sequences=seqs, graph=graph, fitness_layers=layers)


def _capture_grid_signal(monkeypatch):
    captured = {}

    def fake_fit(graph, signal, **kwargs):
        captured["graph"] = graph
        captured["signal"] = np.array(signal, copy=True)
        return 1.0, 0.5, 1.5, -2.0, 3.0

    monkeypatch.setattr(diffusion_scale_module, "fit_t_grid_posterior", fake_fit)
    return captured


@pytest.mark.parametrize("method", ["grid", "profile", "bootstrap", "laplace"])
def test_diffusion_scale_methods_return_valid(method):
    landscape = _toy_landscape()
    kwargs = dict(t_min=0.01, t_max=5.0, method=method, grid_size=64)
    if method == "bootstrap":
        kwargs.update(bootstrap_samples=25, random_state=0)
    out = compute_ruggedness_diffusion_scale(landscape, **kwargs)

    assert {"t_map", "t_lower_confidence_interval", "t_upper_confidence_interval", "t_logposterior_map", "variance_approximate"} <= set(out.keys())
    assert np.isfinite(out["t_map"])
    assert np.isfinite(out["t_logposterior_map"])
    assert np.isfinite(out["variance_approximate"])
    assert kwargs["t_min"] <= out["t_map"] <= kwargs["t_max"]
    assert kwargs["t_min"] <= out["t_lower_confidence_interval"] <= out["t_upper_confidence_interval"] <= kwargs["t_max"]


def test_diffusion_scale_default_is_grid():
    landscape = _toy_landscape()
    out_default = compute_ruggedness_diffusion_scale(landscape, t_min=0.01, t_max=5.0, grid_size=64)
    out_grid = compute_ruggedness_diffusion_scale(
        landscape, t_min=0.01, t_max=5.0, grid_size=64, method="grid"
    )
    assert out_default["t_map"] == pytest.approx(out_grid["t_map"], rel=1e-6, abs=1e-6)


def test_bootstrap_reproducible_with_seed():
    landscape = _toy_landscape()
    out1 = compute_ruggedness_diffusion_scale(
        landscape,
        t_min=0.01,
        t_max=5.0,
        method="bootstrap",
        grid_size=32,
        bootstrap_samples=20,
        random_state=123,
    )
    out2 = compute_ruggedness_diffusion_scale(
        landscape,
        t_min=0.01,
        t_max=5.0,
        method="bootstrap",
        grid_size=32,
        bootstrap_samples=20,
        random_state=123,
    )
    assert out1["t_map"] == pytest.approx(out2["t_map"], rel=1e-6, abs=1e-6)
    assert out1["t_lower_confidence_interval"] == pytest.approx(out2["t_lower_confidence_interval"], rel=1e-6, abs=1e-6)
    assert out1["t_upper_confidence_interval"] == pytest.approx(out2["t_upper_confidence_interval"], rel=1e-6, abs=1e-6)


def test_diffusion_scale_with_precomputed_eigenpairs_matches():
    landscape = _toy_landscape()
    evals, evecs = eigenmode_decomposition(landscape.graph, matrix="norm_laplacian", k=None)
    out_default = compute_ruggedness_diffusion_scale(
        landscape, t_min=0.01, t_max=5.0, grid_size=64, method="grid"
    )
    out_pre = compute_ruggedness_diffusion_scale(
        landscape,
        t_min=0.01,
        t_max=5.0,
        grid_size=64,
        method="grid",
        _eigenvalues=evals,
        _eigenvectors=evecs,
    )
    assert out_default["t_map"] == pytest.approx(out_pre["t_map"], rel=1e-6, abs=1e-6)


def test_named_layer_is_selected_in_node_order_without_mutating_view(monkeypatch):
    landscape = _two_layer_landscape()
    captured = _capture_grid_signal(monkeypatch)

    result = compute_ruggedness_diffusion_scale(
        landscape,
        fitness_layer="selected",
    )

    assert landscape.active_layer_name == "active"
    assert captured["graph"] is landscape.graph
    np.testing.assert_array_equal(captured["signal"], [4.0, 1.0, 2.0])
    assert result == {
        "t_map": 1.0,
        "t_lower_confidence_interval": 0.5,
        "t_upper_confidence_interval": 1.5,
        "t_logposterior_map": -2.0,
        "variance_approximate": 3.0,
    }
    assert all(isinstance(value, float) for value in result.values())


def test_implicit_layer_uses_active_view_without_mutating_it(monkeypatch):
    landscape = _two_layer_landscape()
    landscape.view("selected")
    captured = _capture_grid_signal(monkeypatch)

    compute_ruggedness_diffusion_scale(landscape)

    assert landscape.active_layer_name == "selected"
    np.testing.assert_array_equal(captured["signal"], [4.0, 1.0, 2.0])


def test_missing_named_layer_fails_without_mutating_view():
    landscape = _two_layer_landscape()

    with pytest.raises(KeyError, match="Layer 'missing' not found"):
        compute_ruggedness_diffusion_scale(landscape, fitness_layer="missing")

    assert landscape.active_layer_name == "active"


def test_missing_active_layer_has_clear_failure():
    source = _two_layer_landscape()
    landscape = FitnessLandscape(
        sequences=source.sequences,
        graph=source.graph,
        fitness_layers={},
    )

    with pytest.raises(ValueError, match="requires an active fitness layer"):
        compute_ruggedness_diffusion_scale(landscape)

    assert landscape.active_layer_name is None


@pytest.mark.parametrize(
    ("layer", "message"),
    [
        (
            type(
                "NonNumericLayer",
                (),
                {"to_scalar": lambda self: ["low", "middle", "high"]},
            )(),
            "must be scalarizable to numeric values",
        ),
        (
            type("WrongShapeLayer", (), {"to_scalar": lambda self: [[1.0], [2.0], [3.0]]})(),
            "must provide one scalar per sequence",
        ),
        (
            type("NonFiniteLayer", (), {"to_scalar": lambda self: [1.0, np.nan, 3.0]})(),
            "contains missing or non-finite values",
        ),
    ],
)
def test_invalid_named_layer_signal_fails_without_mutating_view(layer, message):
    landscape = _two_layer_landscape()
    landscape.fitness_layers["invalid"] = layer

    with pytest.raises(ValueError, match=message):
        compute_ruggedness_diffusion_scale(landscape, fitness_layer="invalid")

    assert landscape.active_layer_name == "active"


def test_fitter_failure_does_not_mutate_active_view(monkeypatch):
    landscape = _two_layer_landscape()

    def fail_fit(*args, **kwargs):
        raise RuntimeError("fit failed")

    monkeypatch.setattr(diffusion_scale_module, "fit_t_grid_posterior", fail_fit)

    with pytest.raises(RuntimeError, match="fit failed"):
        compute_ruggedness_diffusion_scale(
            landscape,
            fitness_layer="selected",
        )

    assert landscape.active_layer_name == "active"


def test_graph_sequence_count_mismatch_fails_without_mutating_view():
    landscape = _two_layer_landscape()
    landscape.graph.remove_node("third")

    with pytest.raises(ValueError, match="one graph node per sequence"):
        compute_ruggedness_diffusion_scale(
            landscape,
            fitness_layer="selected",
        )

    assert landscape.active_layer_name == "active"
