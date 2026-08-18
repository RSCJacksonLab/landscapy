"""Dirichlet-energy analyses for scalar fitness landscapes."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Hashable

import networkx as nx
import numpy as np

from ..core.edge_schema import AUTO_EDGE_KEY, resolve_edge_attribute
from ..core.landscape import FitnessLandscape


def _validate_landscape(landscape: FitnessLandscape) -> nx.Graph:
    """Return a supported undirected simple graph."""
    if not isinstance(landscape, FitnessLandscape) or landscape.graph is None:
        raise TypeError("Input must be a FitnessLandscape with an initialized graph.")
    graph = landscape.graph
    if graph.is_directed():
        raise TypeError("Dirichlet energy requires an undirected graph.")
    if graph.is_multigraph():
        raise TypeError("Dirichlet energy requires a simple graph, not a multigraph.")
    return graph


def _resolve_dirichlet_weight_key(
    graph: nx.Graph,
    weight_key: str | None,
    weighted_laplacian: bool | None,
) -> str | None:
    """Resolve explicit weighting and the legacy weighted selector."""
    if weighted_laplacian is not None and not isinstance(weighted_laplacian, bool):
        raise TypeError("weighted_laplacian must be a boolean or None.")
    if weighted_laplacian is False and weight_key is not None:
        raise ValueError(
            "weight_key requests weighted analysis but weighted_laplacian=False."
        )

    requested_key = weight_key
    if weighted_laplacian is True and requested_key is None:
        requested_key = AUTO_EDGE_KEY

    return resolve_edge_attribute(
        graph,
        "conductance",
        requested_key,
        required=requested_key is not None,
    )


def _validate_edge_weight_bins(
    edge_weight_bins: object,
) -> list[tuple[float, float]] | None:
    """Return validated half-open edge-weight intervals."""
    if edge_weight_bins is None:
        return None

    try:
        raw_bins = list(edge_weight_bins)
    except TypeError as error:
        raise TypeError("edge_weight_bins must be an iterable of (lower, upper) pairs.") from error

    bins: list[tuple[float, float]] = []
    for bin_range in raw_bins:
        try:
            lower, upper = bin_range
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Each edge-weight bin must contain exactly two bounds."
            ) from error
        try:
            lower = float(lower)
            upper = float(upper)
        except (TypeError, ValueError) as error:
            raise ValueError("Edge-weight bin bounds must be numeric.") from error
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError(
                "Edge-weight bin bounds must be finite and satisfy lower < upper."
            )
        bins.append((lower, upper))
    return bins


def _fitness_by_node(landscape: FitnessLandscape, node_order: list[Hashable]) -> dict:
    """Return the active scalar layer aligned to graph nodes."""
    signal = landscape.get_node_signal(node_order)
    if signal.shape != (len(node_order),):
        raise ValueError("The active fitness layer must provide one scalar per graph node.")
    if not np.all(np.isfinite(signal)):
        raise ValueError("The active fitness layer must contain only finite scalar values.")
    return dict(zip(node_order, signal, strict=True))


def _edge_energy_records(
    graph: nx.Graph,
    fitness_by_node: dict,
    weight_key: str | None,
) -> list[tuple[Hashable, Hashable, float, float]]:
    """Return ``(u, v, conductance, energy)`` for each undirected edge."""
    records = []
    for u, v, data in graph.edges(data=True):
        conductance = 1.0 if weight_key is None else float(data[weight_key])
        energy = _sum_dirichlet_energy(
            fitness1=fitness_by_node[u],
            fitness2=fitness_by_node[v],
            edge_weight=conductance,
        )
        records.append((u, v, conductance, energy))
    return records


def calculate_ruggedness_dirichlet_energy(
    landscape: FitnessLandscape,
    edge_weight_bins: Iterable[tuple[float, float]] | None = None,
    weighted_laplacian: bool | None = None,
    weight_key: str | None = None,
) -> dict:
    r"""Calculate global and binned Dirichlet energy.

    Parameters
    ----------
    landscape : FitnessLandscape
        Landscape whose current active fitness layer supplies the scalar graph
        signal.
    edge_weight_bins : iterable of pair of float, optional
        Half-open conductance intervals ``[lower, upper)`` used to aggregate
        per-edge energy. With unweighted analysis every edge has conductance
        one. Bins need not be complete or disjoint.
    weighted_laplacian : bool, optional
        Compatibility selector. ``True`` resolves the graph's declared
        conductance when ``weight_key`` is omitted; ``False`` explicitly
        requests unweighted analysis and cannot be combined with a key.
        Prefer ``weight_key`` in new code.
    weight_key : str, optional
        Edge attribute containing non-negative finite conductance. The default
        ``None`` performs unweighted analysis, even when the graph contains
        weight attributes. Passing ``"auto"`` explicitly resolves the
        constructor-declared conductance.

    Returns
    -------
    dict
        ``global_dirichlet_energy`` is the unnormalized quadratic form
        ``f.T @ L @ f``. ``total_dirichlet_energy`` retains the historical
        per-node normalization ``(f.T @ L @ f) / n``. Optional bin energies
        use the unnormalized convention and their contributions divide by the
        global energy.

    Notes
    -----
    For an undirected simple graph, Landscapy defines
    ``f.T @ L @ f = sum_{u,v} w_uv (f_u - f_v)^2``, where the sum is over
    undirected edges. Each undirected edge
    is visited exactly once, so no factor of one half appears in per-edge or
    binned energy. Local node contributions assign half of each incident edge
    energy to each endpoint and therefore sum to the global energy.
    Empty graphs and zero-energy signals return zero for both global and
    per-node energy; bin contributions are also zero when global energy is
    zero.
    """
    graph = _validate_landscape(landscape)
    bins = _validate_edge_weight_bins(edge_weight_bins)
    resolved_weight_key = _resolve_dirichlet_weight_key(
        graph,
        weight_key,
        weighted_laplacian,
    )

    node_order = list(graph.nodes())
    if node_order:
        fitness_by_node = _fitness_by_node(landscape, node_order)
        edge_records = _edge_energy_records(
            graph,
            fitness_by_node,
            resolved_weight_key,
        )
        global_energy = float(sum(record[3] for record in edge_records))
        energy_per_node = global_energy / len(node_order)
    else:
        edge_records = []
        global_energy = 0.0
        energy_per_node = 0.0

    results = {
        "global_dirichlet_energy": global_energy,
        "total_dirichlet_energy": float(energy_per_node),
        "weighted_laplacian": resolved_weight_key is not None,
        "weight_key": resolved_weight_key,
    }

    if bins is not None:
        binned_results = {}
        for lower, upper in bins:
            bin_energy = float(
                sum(
                    edge_energy
                    for _, _, conductance, edge_energy in edge_records
                    if lower <= conductance < upper
                )
            )
            bin_key = str((lower, upper))
            binned_results[f"{bin_key}_dirichlet_energy"] = bin_energy
            binned_results[f"{bin_key}_contribution"] = (
                bin_energy / global_energy if global_energy > 0.0 else 0.0
            )
        results["edge_weight_bins"] = binned_results

    return results


def _sum_dirichlet_energy(
    fitness1: float,
    fitness2: float,
    edge_weight: float = 1.0,
) -> float:
    """Return one undirected edge's contribution to ``f.T @ L @ f``."""
    return float(edge_weight * (fitness1 - fitness2) ** 2)


def local_dirichlet_energy_contribution(
    landscape: FitnessLandscape,
    weight_key: str | None = None,
) -> dict[Hashable, float]:
    """Calculate each node's local contribution to Dirichlet energy.

    Parameters
    ----------
    landscape : FitnessLandscape
        Landscape whose current active fitness layer supplies the scalar graph
        signal.
    weight_key : str, optional
        Edge attribute containing non-negative finite conductance. The default
        ``None`` is unweighted. Passing ``"auto"`` explicitly resolves the
        constructor-declared conductance.

    Returns
    -------
    dict of hashable to float
        Half the sum of incident edge energies for every graph node. Isolated
        nodes contribute zero, and summing all values reproduces the
        unnormalized global ``f.T @ L @ f``.
    """
    graph = _validate_landscape(landscape)
    resolved_weight_key = _resolve_dirichlet_weight_key(
        graph,
        weight_key,
        weighted_laplacian=None,
    )
    node_order = list(graph.nodes())
    if not node_order:
        return {}

    fitness_by_node = _fitness_by_node(landscape, node_order)
    edge_records = _edge_energy_records(
        graph,
        fitness_by_node,
        resolved_weight_key,
    )
    local_energies = {node: 0.0 for node in node_order}
    for u, v, _, edge_energy in edge_records:
        half_energy = 0.5 * edge_energy
        local_energies[u] += half_energy
        local_energies[v] += half_energy
    return local_energies
