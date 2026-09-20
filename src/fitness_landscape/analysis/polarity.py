"""Reference-relative polarity fields on graph-based sequence landscapes."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Dict

import numpy as np
import pandas as pd

from ..core.edge_schema import AUTO_EDGE_KEY
from .bottleneck import calculate_graph_occupancy

if TYPE_CHECKING:
    from ..core.landscape import FitnessLandscape


def _polarity_and_support(
    reference: np.ndarray, observed: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the extended-real log density ratio and support labels."""
    polarity = np.full(reference.shape, np.nan, dtype=float)
    support = np.full(reference.shape, "zero_mass", dtype=object)
    shared = (reference > 0.0) & (observed > 0.0)
    observed_only = (reference == 0.0) & (observed > 0.0)
    reference_only = (reference > 0.0) & (observed == 0.0)
    polarity[shared] = np.log(observed[shared]) - np.log(reference[shared])
    polarity[observed_only] = np.inf
    polarity[reference_only] = -np.inf
    support[shared] = "shared"
    support[observed_only] = "observed_only"
    support[reference_only] = "reference_only"
    return polarity, support


def _edge_polarity_field(
    landscape: "FitnessLandscape",
    nodes: pd.DataFrame,
    *,
    weight_key: str | None,
) -> pd.DataFrame:
    """Orient undirected edges by sequence index and calculate their tilt."""
    graph = landscape.graph
    edge_count = graph.number_of_edges()
    node_labels = nodes["node"].tolist()
    node_to_index = {node: index for index, node in enumerate(node_labels)}
    polarity = nodes["polarity"].to_numpy(dtype=float)

    source_index = np.empty(edge_count, dtype=np.int64)
    target_index = np.empty(edge_count, dtype=np.int64)
    conductance = np.empty(edge_count, dtype=float)
    for row, (left, right, data) in enumerate(graph.edges(data=True)):
        left_index = node_to_index[left]
        right_index = node_to_index[right]
        if left_index <= right_index:
            source_index[row], target_index[row] = left_index, right_index
        else:
            source_index[row], target_index[row] = right_index, left_index
        conductance[row] = 1.0 if weight_key is None else float(data[weight_key])

    source_polarity = polarity[source_index]
    target_polarity = polarity[target_index]
    tilt_defined = np.isfinite(source_polarity) & np.isfinite(target_polarity)
    tilt = np.full(edge_count, np.nan, dtype=float)
    tilt[tilt_defined] = (
        target_polarity[tilt_defined] - source_polarity[tilt_defined]
    )
    return pd.DataFrame(
        {
            "source_sequence_index": source_index,
            "target_sequence_index": target_index,
            "source_node": np.asarray(node_labels, dtype=object)[source_index],
            "target_node": np.asarray(node_labels, dtype=object)[target_index],
            "conductance": conductance,
            "source_polarity": source_polarity,
            "target_polarity": target_polarity,
            "tilt": tilt,
            "tilt_defined": tilt_defined,
        }
    )


def calculate_polarity_field(
    landscape: "FitnessLandscape",
    reference_indices: Iterable[int],
    observed_indices: Iterable[int],
    *,
    weighting: str = "global",
    family_labels: Sequence[Hashable] | None = None,
    family_weights: Mapping[Hashable, float] | None = None,
    weight_key: str | None = AUTO_EDGE_KEY,
    diffusion_time: float = 1.0,
) -> Dict:
    """Calculate a graph-smoothed local log-density-ratio field and edge tilt.

    The node field is the dimensionless reference-relative polarity

    ``phi(i) = log(rho_observed(i) / rho_reference(i))``,

    where both source-normalized empirical measures are smoothed by the same
    continuous-time random-walk heat kernel. Each undirected edge is oriented
    from its lower to higher canonical sequence index and has tilt
    ``phi(target) - phi(source)``. Reversing an edge reverses the tilt.

    Parameters
    ----------
    landscape : FitnessLandscape
        Joint simple undirected landscape containing both source panels.
    reference_indices : iterable of int
        Nonempty unique sequence rows defining the reference panel.
    observed_indices : iterable of int
        Nonempty unique sequence rows defining the observed panel.
    weighting : {"global", "matched_family"}, default="global"
        Source-mass construction passed to
        :func:`calculate_graph_occupancy`.
    family_labels : sequence of hashable, optional
        One family label per sequence row, required for matched-family mass.
    family_weights : mapping of hashable to float, optional
        Positive relative family masses used identically in both panels.
    weight_key : str or None, default="auto"
        Conductance attribute used by the random walk, or ``None`` for
        unweighted geometry.
    diffusion_time : float, default=1.0
        Finite non-negative heat time in graph jump-rate units.

    Returns
    -------
    dict
        ``node_field`` contains the two smoothed probabilities, enrichment,
        dimensionless ``polarity`` and an explicit support label per sequence
        row. ``edge_field`` contains the canonically oriented endpoints,
        conductance, endpoint polarities, edge ``tilt`` and whether that tilt
        is finite. The remaining entries preserve the occupancy divergence,
        source weights, family summary and augmented metadata.

    Notes
    -----
    No pseudocount is added. Observed-only support has polarity ``+inf``,
    reference-only support has ``-inf``, and zero mass from both panels is
    undefined. An edge tilt is reported only when both endpoint polarities are
    finite. This is a descriptive potential-of-mean-force-style field on the
    supplied graph and smoothing scale. It does not by itself identify an
    evolutionary force, selection coefficient, physical energy, equilibrium,
    foldability or mutational accessibility. The graph layout is not used.
    """
    occupancy = calculate_graph_occupancy(
        landscape,
        reference_indices,
        observed_indices,
        weighting=weighting,
        family_labels=family_labels,
        family_weights=family_weights,
        weight_key=weight_key,
        kernel="heat",
        diffusion_time=diffusion_time,
    )
    nodes = occupancy["distribution"].copy()
    reference = nodes["reference_probability"].to_numpy(dtype=float)
    observed = nodes["observed_probability"].to_numpy(dtype=float)
    polarity, support = _polarity_and_support(reference, observed)
    nodes["polarity"] = polarity
    nodes["support"] = support
    edges = _edge_polarity_field(
        landscape,
        nodes,
        weight_key=occupancy["metadata"]["weight_key"],
    )
    metadata = dict(occupancy["metadata"])
    metadata.update(
        {
            "method": "graph_polarity_field",
            "polarity_definition": "log(observed_probability/reference_probability)",
            "edge_orientation": "lower_to_higher_sequence_index",
            "tilt_definition": "target_polarity-source_polarity",
            "pseudocount": None,
            "beta": None,
        }
    )
    return {
        "node_field": nodes,
        "edge_field": edges,
        "divergence": occupancy["divergence"],
        "sequence_weights": occupancy["sequence_weights"],
        "family_summary": occupancy["family_summary"],
        "metadata": metadata,
    }
