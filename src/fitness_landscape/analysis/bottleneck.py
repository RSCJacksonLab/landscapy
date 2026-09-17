"""Bottleneck analysis with an explicit absorbing sequence boundary."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterable, List, Sequence, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla

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
        """Return graph nodes corresponding to the declared input rows."""

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
        """Bind the boundary to a constructed landscape and annotate its graph."""

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
    """Return the smallest algebraic eigenpair of a Dirichlet operator."""

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
    """Rank interior edges by weighted Dirichlet-eigenfunction gradient."""

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
    """Estimate local conductance by sweeping the Dirichlet eigenvector."""

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
