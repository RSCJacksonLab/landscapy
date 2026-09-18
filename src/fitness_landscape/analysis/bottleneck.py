"""Absorbing-boundary bottlenecks and reference-relative sequence occupancy."""

from __future__ import annotations

import math
import warnings
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Dict, Iterable, List, Sequence, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import rel_entr

from ..core.edge_schema import AUTO_EDGE_KEY, resolve_edge_attribute

if TYPE_CHECKING:
    from ..core.landscape import FitnessLandscape


BOUNDARY_GRAPH_KEY = "landscapy_boundary_model"
BOUNDARY_NODE_KEY = "is_boundary"
ABSORBING_NODE_KEY = "absorbing_boundary"
BOUNDARY_EDGE_KEY = "boundary_relation"
BOUNDARY_CROSSING_KEY = "crosses_boundary"


@dataclass(frozen=True)
class BoundaryModel:
    """Define an absorbing boundary by positions in the input sequence list.

    ``sequence_indices`` always refers to the exact list passed to
    :class:`~fitness_landscape.core.landscape.FitnessLandscape`. Binding is
    deferred until landscape construction has established its canonical,
    duplicate-safe mapping from sequence rows to graph nodes.

    Parameters
    ----------
    sequence_indices : sequence of int
        Unique, zero-based input rows that define the boundary.
    absorbing : bool, default=True
        Declare the boundary absorbing for downstream Dirichlet analysis.
        Non-absorbing boundaries are not supported by this model.
    """

    sequence_indices: Sequence[int]
    absorbing: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.absorbing, (bool, np.bool_)):
            raise TypeError("absorbing must be a boolean")
        if not bool(self.absorbing):
            raise ValueError("BoundaryModel represents an absorbing boundary")

        indices = tuple(self.sequence_indices)
        if not indices:
            raise ValueError("sequence_indices must contain at least one boundary row")
        for index in indices:
            if isinstance(index, (bool, np.bool_)) or not isinstance(
                index, (int, np.integer)
            ):
                raise TypeError("sequence_indices must contain only integers")
            if int(index) < 0:
                raise ValueError("sequence_indices must be non-negative")
        normalized = tuple(int(index) for index in indices)
        if len(set(normalized)) != len(normalized):
            raise ValueError("sequence_indices must not contain duplicates")
        object.__setattr__(self, "sequence_indices", normalized)
        object.__setattr__(self, "absorbing", True)

    def nodes(self, landscape: "FitnessLandscape") -> tuple:
        """Return graph nodes corresponding to the declared input rows.

        Parameters
        ----------
        landscape : FitnessLandscape
            Landscape whose canonical sequence mapping binds the boundary.

        Returns
        -------
        tuple
            Graph-node labels in declared boundary-row order.
        """

        size = len(landscape.sequences)
        invalid = [index for index in self.sequence_indices if index >= size]
        if invalid:
            raise IndexError(
                "Boundary sequence index outside the landscape: "
                f"{invalid[0]} not in [0, {size})"
            )
        return tuple(
            landscape.node_for_sequence_index(index)
            for index in self.sequence_indices
        )

    def apply_to_landscape(self, landscape: "FitnessLandscape") -> None:
        """Bind the boundary to a constructed landscape and annotate its graph.

        Parameters
        ----------
        landscape : FitnessLandscape
            Landscape to annotate in place with boundary membership and edges.
        """

        boundary_nodes = self.nodes(landscape)
        boundary_set = set(boundary_nodes)
        graph = landscape.graph
        crossing_count = 0

        for node in graph.nodes:
            is_boundary = node in boundary_set
            graph.nodes[node][BOUNDARY_NODE_KEY] = is_boundary
            graph.nodes[node][ABSORBING_NODE_KEY] = is_boundary

        for node_a, node_b, data in graph.edges(data=True):
            a_boundary = node_a in boundary_set
            b_boundary = node_b in boundary_set
            if a_boundary and b_boundary:
                relation = "boundary-boundary"
            elif a_boundary or b_boundary:
                relation = "boundary-interior"
                crossing_count += 1
            else:
                relation = "interior-interior"
            data[BOUNDARY_EDGE_KEY] = relation
            data[BOUNDARY_CROSSING_KEY] = relation == "boundary-interior"

        graph.graph[BOUNDARY_GRAPH_KEY] = {
            "type": "absorbing-sequence-index",
            "sequence_indices": list(self.sequence_indices),
            "boundary_nodes": list(boundary_nodes),
            "boundary_node_count": len(boundary_nodes),
            "interior_node_count": graph.number_of_nodes() - len(boundary_nodes),
            "crossing_edge_count": crossing_count,
            "absorbing": True,
        }
        landscape._boundary_model = self
        landscape._boundary_nodes = boundary_nodes


def _resolve_boundary(
    landscape: "FitnessLandscape",
    boundary: BoundaryModel | None,
) -> tuple[BoundaryModel, tuple, list]:
    if boundary is None:
        boundary = getattr(landscape, "_boundary_model", None)
    if not isinstance(boundary, BoundaryModel):
        raise TypeError(
            "Provide BoundaryModel(sequence_indices=...) or construct the "
            "FitnessLandscape with boundary_model=..."
        )
    boundary.apply_to_landscape(landscape)
    boundary_nodes = boundary.nodes(landscape)
    boundary_set = set(boundary_nodes)
    interior_nodes = [
        node for node in landscape.graph.nodes if node not in boundary_set
    ]
    if not interior_nodes:
        raise ValueError("The boundary contains every landscape node")
    return boundary, boundary_nodes, interior_nodes


def _edge_weight(data: dict, weight_key: str) -> float:
    weight = float(data.get(weight_key, 1.0))
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError(
            f"Edge attribute {weight_key!r} must be finite and non-negative"
        )
    return weight


def _nx_to_sparse_on_nodes(
    graph: nx.Graph,
    nodes: Sequence,
    *,
    weight_key: str,
) -> Tuple[sp.csr_matrix, Dict]:
    """Convert the graph adjacency induced by ``nodes`` to CSR form."""

    index = {node: position for position, node in enumerate(nodes)}
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for node_a, node_b, data in graph.edges(data=True):
        if node_a in index and node_b in index:
            row, column = index[node_a], index[node_b]
            weight = _edge_weight(data, weight_key)
            rows.extend([row, column])
            columns.extend([column, row])
            values.extend([weight, weight])
    adjacency = sp.csr_matrix(
        (values, (rows, columns)),
        shape=(len(nodes), len(nodes)),
        dtype=float,
    )
    return adjacency, index


def _boundary_leakage(
    graph: nx.Graph,
    interior_nodes: Sequence,
    boundary_nodes: Sequence,
    *,
    weight_key: str,
) -> dict:
    """Sum joint-graph conductance from each interior node to the boundary."""

    boundary_set = set(boundary_nodes)
    leakage = {}
    for node in interior_nodes:
        leakage[node] = sum(
            _edge_weight(data, weight_key)
            for neighbour, data in graph[node].items()
            if neighbour in boundary_set
        )
    return leakage


def build_dirichlet_operator(
    landscape: "FitnessLandscape",
    boundary: BoundaryModel | None = None,
    *,
    weight_key: str = "weight",
    normalized: bool = True,
    interior_nodes: Iterable | None = None,
) -> Tuple[sp.csr_matrix, List]:
    """Construct the absorbing-boundary Dirichlet operator.

    The operator acts only on non-boundary nodes. Joint-graph edge weight from
    an interior node to a boundary node is added to that interior node's
    diagonal leakage term. Boundary nodes are therefore absorbing and do not
    appear as operator rows.

    Parameters
    ----------
    landscape : FitnessLandscape
        Joint boundary/interior graph.
    boundary : BoundaryModel, optional
        Explicit absorbing rows, or the model already bound to the landscape.
    weight_key : str, default="weight"
        Non-negative conductance attribute; missing weights default to one.
    normalized : bool, default=True
        Symmetrically normalize using internal degree plus boundary leakage.
    interior_nodes : iterable, optional
        Interior graph-node labels to include, in operator order. Omitted
        interior nodes are excluded rather than treated as absorbing states.

    Returns
    -------
    operator : scipy.sparse.csr_matrix
        Dirichlet Laplacian including boundary leakage on its diagonal.
    nodes : list
        Graph-node labels aligned with the operator rows.
    """

    _, boundary_nodes, all_interior = _resolve_boundary(landscape, boundary)
    if interior_nodes is None:
        nodes = list(all_interior)
    else:
        allowed = set(all_interior)
        nodes = list(interior_nodes)
        if not nodes:
            raise ValueError("interior_nodes must not be empty")
        if len(set(nodes)) != len(nodes):
            raise ValueError("interior_nodes must not contain duplicates")
        unknown = [node for node in nodes if node not in allowed]
        if unknown:
            raise ValueError(f"Node {unknown[0]!r} is not an interior graph node")

    adjacency, _ = _nx_to_sparse_on_nodes(
        landscape.graph,
        nodes,
        weight_key=weight_key,
    )
    internal_degree = np.asarray(adjacency.sum(axis=1)).ravel()
    leakage_map = _boundary_leakage(
        landscape.graph,
        nodes,
        boundary_nodes,
        weight_key=weight_key,
    )
    leakage = np.array([leakage_map[node] for node in nodes], dtype=float)

    degree = sp.diags(internal_degree, offsets=0, format="csr")
    boundary_degree = sp.diags(leakage, offsets=0, format="csr")
    operator = degree - adjacency + boundary_degree
    if not normalized:
        return operator.tocsr(), nodes

    total_degree = internal_degree + leakage
    inverse_square_root = np.zeros_like(total_degree)
    positive = total_degree > 0.0
    inverse_square_root[positive] = 1.0 / np.sqrt(total_degree[positive])
    scale = sp.diags(inverse_square_root, offsets=0, format="csr")
    return (scale @ operator @ scale).tocsr(), nodes


def first_dirichlet_eigenpair(
    operator: sp.spmatrix | np.ndarray,
    *,
    tol: float = 1e-6,
    maxiter: int = 5000,
) -> Tuple[float, np.ndarray]:
    """Return the smallest algebraic eigenpair of a Dirichlet operator.

    Parameters
    ----------
    operator : scipy.sparse.spmatrix or numpy.ndarray
        Nonempty symmetric square Dirichlet matrix.
    tol : float, default=1e-6
        ARPACK convergence tolerance.
    maxiter : int, default=5000
        Maximum number of ARPACK iterations.

    Returns
    -------
    eigenvalue : float
        Smallest algebraic eigenvalue.
    eigenvector : numpy.ndarray
        Corresponding vector, sign-adjusted to have a non-negative sum.
    """

    matrix = sp.csr_matrix(operator, dtype=float)
    if matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("operator must be a non-empty square matrix")
    if matrix.shape[0] <= 2:
        values, vectors = np.linalg.eigh(matrix.toarray())
        position = int(np.argmin(values))
        value = float(values[position])
        vector = vectors[:, position]
    else:
        try:
            values, vectors = spla.eigsh(
                matrix,
                k=1,
                which="SA",
                tol=tol,
                maxiter=maxiter,
            )
        except spla.ArpackNoConvergence as error:
            if error.eigenvalues is None or len(error.eigenvalues) == 0:
                raise
            position = int(np.argmin(error.eigenvalues))
            value = float(error.eigenvalues[position])
            vector = error.eigenvectors[:, position]
        else:
            value = float(values[0])
            vector = vectors[:, 0]
    if float(np.sum(vector)) < 0.0:
        vector = -vector
    return value, np.asarray(vector, dtype=float)


def rank_throat_edges(
    graph: nx.Graph,
    nodes: Sequence,
    eigenvector: np.ndarray,
    *,
    weight_key: str = "weight",
    degree_normalize: bool = True,
) -> pd.DataFrame:
    """Rank interior edges by weighted Dirichlet-eigenfunction gradient.

    Parameters
    ----------
    graph : networkx.Graph
        Joint landscape graph.
    nodes : sequence
        Node labels in eigenvector order.
    eigenvector : numpy.ndarray
        Vector whose edge differences are ranked, without basis conversion.
    weight_key : str, default="weight"
        Non-negative edge conductance; missing weights default to one.
    degree_normalize : bool, default=True
        Divide weighted differences by the square root of summed endpoint
        unweighted degrees in the full graph.

    Returns
    -------
    pandas.DataFrame
        Interior edges sorted by decreasing normalized weighted difference.
    """

    index = {node: position for position, node in enumerate(nodes)}
    rows = []
    for node_a, node_b, data in graph.subgraph(nodes).edges(data=True):
        weight = _edge_weight(data, weight_key)
        gradient = abs(eigenvector[index[node_a]] - eigenvector[index[node_b]]) * weight
        if degree_normalize:
            denominator = math.sqrt(
                max(graph.degree(node_a), 1) + max(graph.degree(node_b), 1)
            )
            normalized_gradient = gradient / denominator
        else:
            normalized_gradient = gradient
        rows.append(
            (
                node_a,
                node_b,
                weight,
                eigenvector[index[node_a]],
                eigenvector[index[node_b]],
                gradient,
                normalized_gradient,
            )
        )
    columns = ["u", "v", "weight", "f_u", "f_v", "grad", "grad_norm"]
    return pd.DataFrame(rows, columns=columns).sort_values(
        "grad_norm", ascending=False, ignore_index=True
    )


def local_cheeger_sweep(
    graph: nx.Graph,
    interior_nodes: Iterable,
    eigenvector: np.ndarray,
    nodes: Sequence,
    *,
    weight_key: str = "weight",
    max_half_volume: bool = True,
) -> Tuple[float, Set]:
    """Estimate local conductance by sweeping the Dirichlet eigenvector.

    Parameters
    ----------
    graph : networkx.Graph
        Full graph supplying weighted cut and volume values.
    interior_nodes : iterable
        Candidate graph-node labels for the sweep.
    eigenvector : numpy.ndarray
        Vector used to order candidates, without basis conversion.
    nodes : sequence
        Node labels aligned with the vector.
    weight_key : str, default="weight"
        Non-negative edge conductance; missing weights default to one.
    max_half_volume : bool, default=True
        Stop multi-node sweep sets once half the candidate volume is exceeded.

    Returns
    -------
    conductance : float
        Smallest cut-to-volume ratio encountered by the sweep heuristic.
    cutset : set
        Candidate node set realizing that ratio.
    """

    interior = list(interior_nodes)
    index = {node: position for position, node in enumerate(nodes)}
    missing = [node for node in interior if node not in index]
    if missing:
        raise ValueError(f"Node {missing[0]!r} has no eigenvector entry")
    order = sorted(interior, key=lambda node: -eigenvector[index[node]])

    degree = {
        node: sum(
            _edge_weight(data, weight_key)
            for _, data in graph[node].items()
        )
        for node in interior
    }
    total_volume = sum(degree.values())
    best_conductance = float("inf")
    best_set: Set = set()
    sweep_set: Set = set()
    sweep_volume = 0.0

    for node in order:
        sweep_set.add(node)
        sweep_volume += degree[node]
        if (
            max_half_volume
            and len(sweep_set) > 1
            and sweep_volume > 0.5 * max(total_volume, 1e-12)
        ):
            break
        cut = sum(
            _edge_weight(data, weight_key)
            for member in sweep_set
            for neighbour, data in graph[member].items()
            if neighbour not in sweep_set
        )
        conductance = cut / max(sweep_volume, 1e-12)
        if conductance < best_conductance:
            best_conductance = conductance
            best_set = set(sweep_set)

    return float(best_conductance), best_set


def _crossing_edge_table(
    graph: nx.Graph,
    boundary_nodes: Sequence,
    *,
    weight_key: str,
) -> pd.DataFrame:
    boundary_set = set(boundary_nodes)
    rows = []
    for node_a, node_b, data in graph.edges(data=True):
        if (node_a in boundary_set) == (node_b in boundary_set):
            continue
        boundary_node, interior_node = (
            (node_a, node_b) if node_a in boundary_set else (node_b, node_a)
        )
        rows.append(
            {
                "boundary_node": boundary_node,
                "interior_node": interior_node,
                "weight": _edge_weight(data, weight_key),
            }
        )
    return pd.DataFrame(rows, columns=["boundary_node", "interior_node", "weight"])


def calculate_local_bottleneck(
    fitness_landscape: "FitnessLandscape",
    boundary: BoundaryModel | None = None,
    *,
    weight_key: str = "weight",
    normalized_laplacian: bool = True,
    normalize_degree: bool = True,
    largest_component_only: bool = True,
) -> Dict:
    """Analyse bottlenecks inside a joint landscape with an absorbing boundary.

    Boundary nodes are taken from input-row indices. The analysed domain is the
    induced graph on all remaining nodes, while domain-to-boundary edge weights
    supply the Dirichlet leakage term.

    Parameters
    ----------
    fitness_landscape : FitnessLandscape
        Joint landscape to analyse.
    boundary : BoundaryModel, optional
        Explicit absorbing rows, or the landscape's bound model.
    weight_key : str, default="weight"
        Non-negative edge conductance; missing weights default to one.
    normalized_laplacian : bool, default=True
        Use the symmetric normalized Dirichlet operator.
    normalize_degree : bool, default=True
        Degree-normalize edge-gradient rankings.
    largest_component_only : bool, default=True
        Restrict a disconnected interior to its largest component with a warning.

    Returns
    -------
    dict
        Boundary and interior node IDs, crossing edges and leakage, smallest
        Dirichlet eigenpair, edge-gradient rankings and sweep conductance.
    """

    if not hasattr(fitness_landscape, "node_for_sequence_index"):
        raise TypeError("fitness_landscape must be a FitnessLandscape")
    model, boundary_nodes, interior_nodes = _resolve_boundary(
        fitness_landscape, boundary
    )
    graph = fitness_landscape.graph
    components = list(nx.connected_components(graph.subgraph(interior_nodes)))
    analysed_nodes = list(interior_nodes)
    if largest_component_only and len(components) > 1:
        largest = max(components, key=len)
        warnings.warn(
            "Interior graph has "
            f"{len(components)} connected components; restricting bottleneck analysis "
            f"to the largest component containing {len(largest)} nodes.",
            RuntimeWarning,
        )
        analysed_nodes = [node for node in interior_nodes if node in largest]

    operator, operator_nodes = build_dirichlet_operator(
        fitness_landscape,
        model,
        weight_key=weight_key,
        normalized=normalized_laplacian,
        interior_nodes=analysed_nodes,
    )
    eigenvalue, eigenvector = first_dirichlet_eigenpair(operator)
    throats = rank_throat_edges(
        graph,
        operator_nodes,
        eigenvector,
        weight_key=weight_key,
        degree_normalize=normalize_degree,
    )
    conductance, cutset = local_cheeger_sweep(
        graph,
        operator_nodes,
        eigenvector,
        operator_nodes,
        weight_key=weight_key,
    )
    leakage = _boundary_leakage(
        graph,
        operator_nodes,
        boundary_nodes,
        weight_key=weight_key,
    )

    return {
        "boundary_sequence_indices": tuple(model.sequence_indices),
        "boundary_nodes": tuple(boundary_nodes),
        "interior_nodes": tuple(operator_nodes),
        "boundary_crossing_edges": _crossing_edge_table(
            graph, boundary_nodes, weight_key=weight_key
        ),
        "boundary_leakage": leakage,
        "first_dirichlet_eigenvalue": eigenvalue,
        "first_dirichlet_eigenvector": eigenvector,
        "dirichlet_eigenfunction_throats": throats,
        "local_cheeger_constant": conductance,
        "local_cheeger_cutset": cutset,
    }


def _occupancy_indices(indices: Iterable[int], size: int, name: str) -> np.ndarray:
    """Validate a nonempty selection in the landscape's sequence-row order."""
    values = list(indices)
    if not values:
        raise ValueError(f"{name} must not be empty")
    if any(
        isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
        for i in values
    ):
        raise TypeError(f"{name} must contain integer sequence indices")
    if any(i < 0 or i >= size for i in values):
        raise IndexError(f"{name} contains a sequence index outside [0, {size})")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicate indices")
    return np.asarray(values, dtype=np.intp)


def _occupancy_scalar(value: float, name: str, *, positive: bool = False) -> float:
    """Validate a finite real smoothing or weighting parameter."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not np.isfinite(value) or value < 0 or (positive and value == 0):
        bound = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be finite and {bound}")
    return value


def _occupancy_inputs(
    landscape: "FitnessLandscape",
    reference_indices: Iterable[int],
    observed_indices: Iterable[int],
    weighting: str,
    family_labels: Sequence[Hashable] | None,
    family_weights: Mapping[Hashable, float] | None,
    weight_key: str | None,
) -> tuple:
    """Build auditable source masses without changing the supplied landscape."""
    from ..core.landscape import FitnessLandscape

    if not isinstance(landscape, FitnessLandscape):
        raise TypeError("landscape must be a FitnessLandscape")
    graph = landscape.graph
    if graph.is_directed() or graph.is_multigraph():
        raise TypeError("Occupancy requires a simple undirected graph")
    size = len(landscape.sequences)
    if size == 0 or graph.number_of_nodes() != size:
        raise ValueError("Occupancy requires one graph node per sequence row")
    nodes = [landscape.node_for_sequence_index(i) for i in range(size)]
    if len(set(nodes)) != size or set(nodes) != set(graph):
        raise ValueError("Graph nodes no longer match the landscape sequence rows")
    reference = _occupancy_indices(reference_indices, size, "reference_indices")
    observed = _occupancy_indices(observed_indices, size, "observed_indices")
    if weighting not in {"global", "matched_family"}:
        raise ValueError("weighting must be 'global' or 'matched_family'")
    resolved_weight = resolve_edge_attribute(graph, "conductance", weight_key)
    masses = np.zeros((size, 2), dtype=float)
    family_rows = []
    if weighting == "global":
        if family_labels is not None or family_weights is not None:
            raise ValueError("Family parameters require weighting='matched_family'")
        masses[reference, 0] = 1.0 / len(reference)
        masses[observed, 1] = 1.0 / len(observed)
    else:
        if family_labels is None or len(family_labels) != size:
            raise ValueError("family_labels must have one label per landscape sequence")
        labels = list(family_labels)
        family_ids: dict[Hashable, int] = {}
        codes = np.full(size, -1, dtype=np.intp)
        for index in np.union1d(reference, observed):
            label = labels[index]
            try:
                hash(label)
            except TypeError as error:
                raise TypeError("Selected family labels must be hashable") from error
            missing = pd.isna(label)
            if isinstance(missing, (bool, np.bool_)) and missing:
                raise ValueError("Selected family labels must not be missing")
            codes[index] = family_ids.setdefault(label, len(family_ids))
        counts = np.column_stack(
            [
                np.bincount(codes[selection], minlength=len(family_ids))
                for selection in (reference, observed)
            ]
        )
        if np.any(counts == 0):
            raise ValueError(
                "matched_family requires every selected family in both panels; "
                "select a common family set explicitly"
            )
        if family_weights is None:
            targets = np.ones(len(family_ids), dtype=float)
        else:
            if not isinstance(family_weights, Mapping):
                raise TypeError("family_weights must be a mapping")
            if set(family_weights) != set(family_ids):
                raise ValueError(
                    "family_weights must name exactly the selected families"
                )
            targets = np.array(
                [
                    _occupancy_scalar(
                        family_weights[label], "family weight", positive=True
                    )
                    for label in family_ids
                ]
            )
        # Scaling first prevents overflow for otherwise valid relative weights.
        targets /= targets.max()
        targets /= targets.sum()
        if np.any(targets[:, None] / counts == 0):
            raise ValueError("family_weights underflow; reduce their dynamic range")
        for column, selection in enumerate((reference, observed)):
            masses[selection, column] = (
                targets[codes[selection]] / counts[codes[selection], column]
            )
        family_rows = [
            (label, int(counts[code, 0]), int(counts[code, 1]), float(targets[code]))
            for label, code in family_ids.items()
        ]
    sequence_table = pd.DataFrame(
        {
            "sequence_index": np.arange(size),
            "node": nodes,
            "reference_weight": masses[:, 0],
            "observed_weight": masses[:, 1],
        }
    )
    if weighting == "matched_family":
        sequence_table["family"] = list(family_labels)
    family_summary = pd.DataFrame(
        family_rows,
        columns=["family", "reference_count", "observed_count", "target_weight"],
    )
    metadata = {
        "weighting": weighting,
        "weight_key": resolved_weight,
        "reference_indices": tuple(int(i) for i in reference),
        "observed_indices": tuple(int(i) for i in observed),
        "log_base": "e",
        "absorbing_boundary_applied": False,
        "networkx_version": nx.__version__,
    }
    return graph, nodes, masses, sequence_table, family_summary, metadata


def _occupancy_divergence(probabilities: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """Summarize observed/reference mass, retaining unsupported observations."""
    reference, observed = probabilities.T
    middle = 0.5 * reference + 0.5 * observed
    contribution = rel_entr(observed, reference)
    ratio = np.full_like(observed, np.nan)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        np.divide(observed, reference, out=ratio, where=reference > 0)
    ratio[(reference == 0) & (observed > 0)] = np.inf
    distribution = pd.DataFrame(
        {
            "reference_probability": reference,
            "observed_probability": observed,
            "enrichment": ratio,
            "kl_contribution": contribution,
        }
    )
    divergence = {
        "kl_observed_reference": max(0.0, float(contribution.sum())),
        "kl_reference_observed": max(0.0, float(rel_entr(reference, observed).sum())),
        "jensen_shannon": max(
            0.0,
            float(
                0.5 * rel_entr(observed, middle).sum()
                + 0.5 * rel_entr(reference, middle).sum()
            ),
        ),
        "total_variation": float(0.5 * np.abs(observed - reference).sum()),
        "observed_mass_without_reference": float(observed[reference == 0].sum()),
    }
    return distribution, divergence


def calculate_sequence_occupancy(
    landscape: "FitnessLandscape",
    reference_indices: Iterable[int],
    observed_indices: Iterable[int],
    *,
    weighting: str = "global",
    family_labels: Sequence[Hashable] | None = None,
    family_weights: Mapping[Hashable, float] | None = None,
    weight_key: str | None = AUTO_EDGE_KEY,
    resolution: float = 1.0,
    threshold: float = 1e-7,
    seed: int | None = 0,
) -> Dict:
    """Compare sequence counts in shared, on-the-fly Louvain communities.

    The graph defines regions; source-normalized sequence counts define their
    probability masses. Communities are computed once on the entire supplied
    graph, independently of source membership and family weights. Absorbing
    boundary annotations are ignored. The landscape is not modified.

    Parameters
    ----------
    landscape : FitnessLandscape
        Joint landscape containing both panels, with one node per sequence row.
        Additional nodes may define geometry without receiving source mass.
    reference_indices : iterable of int
        Nonempty, unique zero-based rows in ``landscape.sequences`` for the
        reference panel. May overlap ``observed_indices``.
    observed_indices : iterable of int
        Nonempty, unique sequence rows for the empirical panel. Duplicate
        sequence strings in distinct rows remain distinct observations.
    weighting : {"global", "matched_family"}, default="global"
        Equal sequence weights within each panel, or equal total family mass
        in both panels. Matched mode requires identical selected family sets;
        no family or observation is silently dropped.
    family_labels : sequence of hashable, optional
        Labels aligned to every landscape sequence row, required in matched
        mode. Selected labels must be nonmissing. For annotations, pass the
        desired column of ``get_annotation_layer(...).to_dataframe()``.
    family_weights : mapping of hashable to float, optional
        Positive finite relative masses for exactly the selected families,
        normalized to sum to one. Matched mode defaults to equal family mass.
        Family parameters are rejected in global mode.
    weight_key : str or None, default="auto"
        Conductance attribute resolved through Landscapy's edge schema.
        ``None`` explicitly requests unweighted geometry. Explicit attributes
        must exist on every edge and be finite and non-negative.
    resolution : float, default=1.0
        Positive Louvain resolution; larger values favor smaller communities.
    threshold : float, default=1e-7
        Non-negative Louvain modularity-gain stopping threshold.
    seed : int or None, default=0
        NetworkX Louvain random seed, recorded in the result metadata.

    Returns
    -------
    dict
        ``distribution`` is a community table with node and raw source counts,
        probabilities, observed/reference enrichment and KL contributions.
        ``sequence_weights`` maps canonical sequence rows and graph nodes to
        source weights and community IDs. ``family_summary`` records matched
        counts and target masses (empty in global mode). ``divergence`` holds
        both KL directions, Jensen-Shannon divergence (not its square root),
        total variation and observed mass with zero reference probability.
        ``metadata`` records selections, weighting and Louvain parameters.

    Notes
    -----
    All logarithms are natural. No pseudocount is added: positive observed mass
    with zero reference mass gives infinite KL, and enrichment is NaN where
    both are zero. These descriptive quantities depend on sampling and region
    resolution; they do not establish selection or physical foldability.
    Zero-conductance edges do not connect regions; edgeless graphs yield
    singleton communities. Self-loops retain NetworkX Louvain semantics.
    """
    resolution = _occupancy_scalar(resolution, "resolution", positive=True)
    threshold = _occupancy_scalar(threshold, "threshold")
    if seed is not None:
        if isinstance(seed, (bool, np.bool_)) or not isinstance(
            seed, (int, np.integer)
        ):
            raise TypeError("seed must be an integer or None")
        seed = int(seed)
    graph, nodes, masses, sequence_table, families, metadata = _occupancy_inputs(
        landscape,
        reference_indices,
        observed_indices,
        weighting,
        family_labels,
        family_weights,
        weight_key,
    )
    key = metadata["weight_key"]
    positive_graph = (
        graph
        if key is None
        else nx.subgraph_view(
            graph, filter_edge=lambda u, v: float(graph[u][v][key]) > 0.0
        )
    )
    if positive_graph.number_of_edges() == 0:
        communities = [{node} for node in nodes]
    else:
        communities = nx.community.louvain_communities(
            positive_graph,
            weight=key,
            resolution=resolution,
            threshold=threshold,
            seed=seed,
        )
    # Canonical IDs do not depend on sorting arbitrary graph-node labels.
    positions = {node: i for i, node in enumerate(nodes)}
    communities = sorted(
        communities, key=lambda group: min(positions[n] for n in group)
    )
    codes = np.empty(len(nodes), dtype=np.intp)
    for community_id, members in enumerate(communities):
        for node in members:
            codes[positions[node]] = community_id
    probabilities = np.column_stack(
        [
            np.bincount(codes, weights=masses[:, column], minlength=len(communities))
            for column in range(2)
        ]
    )
    distribution, divergence = _occupancy_divergence(probabilities)
    distribution.insert(0, "community", np.arange(len(communities)))
    distribution.insert(1, "node_count", np.bincount(codes))
    for column, source in enumerate(("reference", "observed")):
        distribution[f"{source}_count"] = np.bincount(
            codes[list(metadata[f"{source}_indices"])], minlength=len(communities)
        )
    sequence_table["community"] = codes
    metadata.update(
        {
            "method": "sequence_counts",
            "partition": "networkx.community.louvain_communities",
            "resolution": resolution,
            "threshold": threshold,
            "seed": seed,
            "community_count": len(communities),
        }
    )
    return {
        "distribution": distribution,
        "divergence": divergence,
        "sequence_weights": sequence_table,
        "family_summary": families,
        "metadata": metadata,
    }


def calculate_graph_occupancy(
    landscape: "FitnessLandscape",
    reference_indices: Iterable[int],
    observed_indices: Iterable[int],
    *,
    weighting: str = "global",
    family_labels: Sequence[Hashable] | None = None,
    family_weights: Mapping[Hashable, float] | None = None,
    weight_key: str | None = AUTO_EDGE_KEY,
    kernel: str = "random_walk",
    steps: int | None = None,
    laziness: float | None = None,
    diffusion_time: float | None = None,
) -> Dict:
    """Compare source occupancy after a shared, mass-conserving graph smoothing.

    With row-stochastic ``P = D**(-1) W``, random-walk smoothing applies
    ``[laziness*I + (1-laziness)*P]**steps`` to both row distributions. Heat
    smoothing applies ``exp(diffusion_time * (P-I))``, the continuous-time
    random-walk heat kernel. Sparse matrix-vector products and exponential
    actions avoid materializing a dense kernel. Absorbing boundaries are not
    applied, all components are retained, and the landscape is not modified.

    Parameters
    ----------
    landscape : FitnessLandscape
        Joint simple undirected landscape containing both panels. Extra nodes
        participate in smoothing without receiving initial source mass.
    reference_indices : iterable of int
        Nonempty unique zero-based reference rows in ``landscape.sequences``.
    observed_indices : iterable of int
        Nonempty unique observed sequence rows; overlap with reference is
        allowed. Distinct rows with identical strings remain observations.
    weighting : {"global", "matched_family"}, default="global"
        Equal initial sequence mass within each panel, or matched total family
        mass as in :func:`calculate_sequence_occupancy`.
    family_labels : sequence of hashable, optional
        One label per landscape sequence row, required for matched mode.
        Every selected family must occur in both panels; selected labels must
        be nonmissing. An annotation-layer column may supply these labels.
    family_weights : mapping of hashable to float, optional
        Positive finite relative masses for exactly the selected families;
        defaults to equal family mass. Rejected with global weighting.
    weight_key : str or None, default="auto"
        Conductance attribute resolved by the edge schema, or ``None`` for
        unweighted transitions. Weights must be finite and non-negative.
    kernel : {"random_walk", "heat"}, default="random_walk"
        Discrete lazy random walk or continuous-time random-walk heat kernel.
    steps : int, optional
        Non-negative number of random-walk steps; defaults to one. Must be
        omitted for heat. Zero steps leaves initial masses unchanged.
    laziness : float, optional
        Random-walk holding probability in [0, 1]; defaults to 0.5 to avoid
        bipartite oscillations. Must be omitted for heat; heat uses ``P-I``
        without an extra laziness factor.
    diffusion_time : float, optional
        Finite non-negative heat time; defaults to 1.0. Must be omitted for
        random walks. Time is in graph jump-rate units, not biological time;
        zero leaves initial masses unchanged.

    Returns
    -------
    dict
        ``distribution`` contains one row per canonical sequence index and
        node, with smoothed probabilities, enrichment and KL contributions.
        ``sequence_weights`` contains initial source masses; ``family_summary``
        contains matched counts and target masses. ``divergence`` has both KL
        directions, Jensen-Shannon divergence, total variation and observed
        mass with zero reference probability, as in the sequence-count method.
        ``metadata`` records the kernel, effective parameters and selections.

    Notes
    -----
    Zero-degree nodes hold their mass. Disconnected components never exchange
    mass. Self-loop weights count once in the row sum. Source masses, not
    stationary degree weights, define occupancy. Long smoothing erases source
    differences within each connected component. All divergences use natural
    logarithms, with no pseudocounts. Infinite KL may reflect disconnected or
    unreached support, or numerical underflow at very small heat probabilities.
    Tiny negative roundoff is clipped before normalizing; material violations
    of probability conservation raise an error.
    """
    if kernel not in {"random_walk", "heat"}:
        raise ValueError("kernel must be 'random_walk' or 'heat'")
    if kernel == "random_walk":
        if diffusion_time is not None:
            raise ValueError("diffusion_time is only valid for kernel='heat'")
        steps = 1 if steps is None else steps
        if isinstance(steps, (bool, np.bool_)) or not isinstance(
            steps, (int, np.integer)
        ):
            raise TypeError("steps must be a non-negative integer")
        if steps < 0:
            raise ValueError("steps must be a non-negative integer")
        steps = int(steps)
        laziness = _occupancy_scalar(0.5 if laziness is None else laziness, "laziness")
        if laziness > 1:
            raise ValueError("laziness must be in [0, 1]")
    else:
        if steps is not None or laziness is not None:
            raise ValueError(
                "steps and laziness are only valid for kernel='random_walk'"
            )
        diffusion_time = _occupancy_scalar(
            1.0 if diffusion_time is None else diffusion_time, "diffusion_time"
        )
    graph, nodes, masses, sequence_table, families, metadata = _occupancy_inputs(
        landscape,
        reference_indices,
        observed_indices,
        weighting,
        family_labels,
        family_weights,
        weight_key,
    )
    adjacency = sp.csr_matrix(
        nx.to_scipy_sparse_array(
            graph,
            nodelist=nodes,
            weight=metadata["weight_key"],
            dtype=float,
            format="csr",
        )
    )
    adjacency.eliminate_zeros()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    if not np.all(np.isfinite(degree)):
        raise ValueError("Weighted degree overflow; rescale conductances")
    transition = adjacency.copy()
    # Divide entries directly: reciprocal degrees can overflow for valid
    # subnormal conductances, even when the transition probabilities are safe.
    transition.data /= np.repeat(degree, np.diff(transition.indptr))
    transition = (transition + sp.diags((degree == 0).astype(float))).tocsr()
    probabilities = masses.copy()
    if kernel == "random_walk":
        for _ in range(steps):
            probabilities = laziness * probabilities + (1.0 - laziness) * (
                transition.T @ probabilities
            )
    elif diffusion_time > 0:
        generator = diffusion_time * (transition.T - sp.eye(len(nodes), format="csr"))
        probabilities = spla.expm_multiply(
            generator, probabilities, traceA=float(generator.diagonal().sum())
        )
    if (
        not np.all(np.isfinite(probabilities))
        or np.any(probabilities < -1e-12)
        or not np.allclose(probabilities.sum(axis=0), 1.0, atol=1e-10, rtol=1e-10)
    ):
        raise RuntimeError("Graph smoothing failed probability conservation")
    probabilities = np.maximum(probabilities, 0.0)
    probabilities /= probabilities.sum(axis=0)
    distribution, divergence = _occupancy_divergence(probabilities)
    distribution.insert(0, "sequence_index", np.arange(len(nodes)))
    distribution.insert(1, "node", nodes)
    metadata.update(
        {
            "method": "graph_smoothing",
            "kernel": kernel,
            "steps": steps,
            "laziness": laziness,
            "diffusion_time": diffusion_time,
            "zero_degree_node_count": int(np.count_nonzero(degree == 0)),
        }
    )
    return {
        "distribution": distribution,
        "divergence": divergence,
        "sequence_weights": sequence_table,
        "family_summary": families,
        "metadata": metadata,
    }
