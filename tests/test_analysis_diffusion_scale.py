import networkx as nx
import numpy as np
import pytest

from fitness_landscape.analysis.diffusion_scale import compute_ruggedness_diffusion_scale
from fitness_landscape.core.fitness import NumericFitness
from fitness_landscape.core.landscape import FitnessLandscape
from fitness_landscape.core.sequence import BaseNumpySequence


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
