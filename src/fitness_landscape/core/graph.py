"""Construct sequence-aware undirected fitness-landscape graphs."""

from __future__ import annotations

import numpy as np
import networkx as nx
from typing import TYPE_CHECKING, List, Union, Literal, Tuple, Optional, Sequence, Mapping, Hashable, Any
from .sequence import BaseNumpySequence, BinarySequence, sequence_distance, SoftSequence
from pathlib import Path
from .._const import PROT_20
from .._optional import ray_runtime, require_optional
from ..utils import calculate_gapped_soft_score
from .annotation import register_auto_annotation
from .edge_schema import declare_edge_semantics
from scipy import sparse
from scipy.linalg import expm
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from scipy.sparse import triu as sp_triu
import logging
import time

if TYPE_CHECKING:
    from cogent3.core.alignment import Alignment

_BaseSequence = [BaseNumpySequence, BinarySequence, SoftSequence]

_NEIGHBOUR_BACKENDS = {"auto", "faiss", "balltree"}
_FAISS_INDEX_TYPES = {"flat", "hnsw", "ivf"}
_FAISS_METRICS = {"ip", "l2"}
_TIE_POLICIES = {"all", "min_index", "random"}
_EMBEDDING_KNN_DOMAINS = {"plm", "composition"}
_DEFAULT_MAX_DIFFUSION_NNZ = 50_000_000
_DEFAULT_MAX_DIFFUSION_WORK = 1_000_000_000


def _validate_sequence_collection(
    sequences: Sequence[BaseNumpySequence],
    *,
    name: str = "sequences",
    require_aligned: bool = True,
) -> tuple[int, int]:
    """Validate graph-constructor sequence structure.

    Empty collections are supported and return ``(0, 0)``. Non-empty
    collections must contain non-empty ``BaseNumpySequence`` instances and,
    by default, have a common aligned length. Embedding-space graph searches
    may set ``require_aligned=False`` because their geometry is independent of
    raw sequence coordinates.
    """

    if sequences is None:
        raise TypeError(f"`{name}` must be a sequence collection, not None.")

    try:
        n_sequences = len(sequences)
    except TypeError as error:
        raise TypeError(f"`{name}` must be a sized sequence collection.") from error

    if n_sequences == 0:
        return 0, 0

    for index, sequence in enumerate(sequences):
        if not isinstance(sequence, BaseNumpySequence):
            raise TypeError(
                f"`{name}[{index}]` must be a BaseNumpySequence instance."
            )

    lengths = [len(sequence) for sequence in sequences]
    if any(length <= 0 for length in lengths):
        raise ValueError(f"`{name}` entries must be non-empty.")
    if require_aligned and len(set(lengths)) != 1:
        raise ValueError(
            f"`{name}` entries must have a uniform aligned length; found {lengths}."
        )
    return n_sequences, lengths[0] if require_aligned else 0


def _validate_embedding_matrix(
    embeddings: np.ndarray,
    *,
    n_sequences: int,
    name: str = "embeddings",
) -> np.ndarray:
    """Return a finite two-dimensional embedding matrix aligned to sequences."""

    matrix = np.asarray(embeddings)
    if matrix.ndim != 2:
        raise ValueError(f"`{name}` must be 2-D with shape (n_sequences, n_features).")
    if matrix.shape[0] != n_sequences:
        raise ValueError(
            f"`{name}` rows must match `sequences`; found {matrix.shape[0]} rows "
            f"for {n_sequences} sequences."
        )
    if matrix.shape[1] == 0:
        raise ValueError(f"`{name}` must contain at least one feature column.")
    if np.issubdtype(matrix.dtype, np.complexfloating):
        raise ValueError(f"`{name}` must contain real numeric values.")
    if not np.issubdtype(matrix.dtype, np.number):
        try:
            matrix = matrix.astype(np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(f"`{name}` must contain numeric values.") from error
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"`{name}` must contain only finite values.")
    return matrix


def _validate_integer(
    value: Any,
    *,
    name: str,
    minimum: int,
) -> int:
    """Return an integer parameter after rejecting booleans and coercions."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"`{name}` must be an integer >= {minimum}.")
    result = int(value)
    if result < minimum:
        raise ValueError(f"`{name}` must be an integer >= {minimum}.")
    return result


def _validate_boolean(value: Any, *, name: str) -> bool:
    """Return a strict boolean option."""

    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"`{name}` must be a boolean.")
    return bool(value)


def _validate_neighbour_configuration(
    *,
    n_sequences: int,
    k: Any,
    tiebuffer: Any,
    backend: str,
    index_type: str,
    faiss_metric: str,
    include_self: Any,
    use_gpu: Any,
    hnsw_M: Any,
    tie_policy: str | None = None,
) -> tuple[int, str]:
    """Validate shared kNN options and return capped k and resolved backend."""

    requested_k = _validate_integer(k, name="k", minimum=1)
    _validate_integer(tiebuffer, name="tiebuffer", minimum=0)
    _validate_integer(hnsw_M, name="hnsw_M", minimum=1)
    include_self = _validate_boolean(include_self, name="include_self")
    use_gpu = _validate_boolean(use_gpu, name="use_gpu")

    if backend not in _NEIGHBOUR_BACKENDS:
        raise ValueError(
            f"Unsupported backend {backend!r}. Expected one of "
            f"{sorted(_NEIGHBOUR_BACKENDS)}."
        )
    if index_type not in _FAISS_INDEX_TYPES:
        raise ValueError(
            f"Unsupported FAISS index_type {index_type!r}. Expected one of "
            f"{sorted(_FAISS_INDEX_TYPES)}."
        )
    if faiss_metric not in _FAISS_METRICS:
        raise ValueError(
            f"Unsupported FAISS metric {faiss_metric!r}. Expected 'ip' or 'l2'."
        )
    if tie_policy is not None and tie_policy not in _TIE_POLICIES:
        raise ValueError(
            f"Unsupported tie_policy {tie_policy!r}. Expected one of "
            f"{sorted(_TIE_POLICIES)}."
        )

    resolved_backend = (
        "faiss" if backend == "auto" and n_sequences >= 5000
        else "balltree" if backend == "auto"
        else backend
    )
    if use_gpu and resolved_backend != "faiss":
        raise ValueError("`use_gpu=True` requires the FAISS backend.")
    if use_gpu and index_type != "flat":
        raise ValueError("`use_gpu=True` is supported only with index_type='flat'.")

    # k always denotes non-self neighbours. A request beyond the available
    # population is explicitly capped instead of being left to a backend.
    effective_k = min(requested_k, max(n_sequences - 1, 0))
    _ = include_self  # validated above; it controls candidate-query capacity.
    return effective_k, resolved_backend


def _validate_diffusion_power(
    t: Optional[Union[int, float]],
) -> tuple[bool, int | None]:
    """Validate diffusion power and normalize stationary sentinels."""

    if t is None:
        return True, None
    if isinstance(t, (bool, np.bool_)) or not isinstance(
        t, (int, float, np.integer, np.floating)
    ):
        raise TypeError("`t` must be a non-negative integer, None, or positive infinity.")
    value = float(t)
    if np.isnan(value) or value == -np.inf:
        raise ValueError("`t` must be a non-negative integer, None, or positive infinity.")
    if value == np.inf or value == 0.0:
        return True, None
    if not np.isfinite(value) or not value.is_integer() or value < 1.0:
        raise ValueError("Finite `t` must be an integer >= 1.")
    return False, int(value)


def _validate_connectivity_threshold(value: Any) -> float:
    """Return a finite dimensionless diffusion threshold in ``[0, 1]``."""

    if isinstance(value, (bool, np.bool_)):
        raise TypeError("`connectivity_threshold` must be a real number in [0, 1].")
    try:
        threshold = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "`connectivity_threshold` must be a real number in [0, 1]."
        ) from error
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("`connectivity_threshold` must be finite and lie in [0, 1].")
    return threshold


def _validate_diffusion_budget(value: Any, *, name: str) -> int:
    """Return a positive exact-diffusion resource budget."""

    return _validate_integer(value, name=name, minimum=1)


def _csr_storage_bytes(n_rows: int, nnz: int) -> int:
    """Conservatively estimate bytes for one float64 CSR matrix."""

    # SciPy normally uses int32 column indices but may promote to int64 for
    # very large matrices. Count int64 indices and indptr entries so the
    # estimate remains conservative across platforms.
    return int(nnz) * (np.dtype(np.float64).itemsize + np.dtype(np.int64).itemsize) + (
        int(n_rows) + 1
    ) * np.dtype(np.int64).itemsize


def _raise_diffusion_budget_error(
    *,
    stage: str,
    quantity: str,
    estimate: int,
    limit: int,
) -> None:
    """Raise an actionable error before exact sparse diffusion exceeds budget."""

    if quantity == "nonzeros":
        memory = _csr_storage_bytes(0, estimate)
        detail = f"{estimate:,} nonzeros (at least {memory / 2**30:.2f} GiB per CSR matrix)"
        option = "max_diffusion_nnz"
    else:
        detail = f"{estimate:,} scalar products"
        option = "max_diffusion_work"
    raise MemoryError(
        f"Exact sparse diffusion at {stage} requires {detail}, exceeding "
        f"`{option}={limit:,}`. Reduce `k`, `t`, or `tiebuffer`, partition the "
        f"landscape, or deliberately raise `{option}` only after provisioning "
        "the corresponding memory and compute resources. "
        "`connectivity_threshold` is applied after exact diffusion and cannot "
        "reduce intermediate resource use."
    )


def _force_disable_hamming_edge_computation(_requested: bool) -> bool:
    """Normalize the legacy Hamming-edge annotation flag.

    The flag was historically forced off without warning. It now controls an
    edge-local annotation pass, whose memory use is linear in graph size.
    """

    return bool(_requested)


def _distance_affinity(normalized_distance: float) -> float:
    """Convert a non-negative normalized distance to an RBF-like affinity."""
    return float(np.exp(-float(normalized_distance)))


def _declare_hamming_graph_semantics(G: nx.Graph, *, constructor: str) -> None:
    """Declare unit-conductance semantics for a Hamming adjacency graph."""
    declare_edge_semantics(
        G,
        constructor=constructor,
        distance_key="distance",
        distance_units="hamming_count",
        normalized_distance_key="normalized_distance",
        affinity_key="affinity",
        conductance_key="weight",
        legacy_aliases={"sim": "affinity"},
        notes=(
            "Edges connect one-mutant neighbours. Conductance and affinity are "
            "unit-valued topological adjacency weights."
        ),
    )


def _attach_knn_edge_semantics(
    G: nx.Graph,
    *,
    sequence_length: int,
    constructor: str,
    distance_geometry: Literal["hamming", "euclidean"] = "hamming",
) -> None:
    """Attach canonical kNN distances, affinities, and conductances."""
    length = int(sequence_length)
    if distance_geometry == "hamming" and length <= 0 and G.number_of_edges() > 0:
        raise ValueError("kNN edge semantics require sequences of positive length.")
    if distance_geometry not in {"hamming", "euclidean"}:
        raise ValueError(
            "`distance_geometry` must be either 'hamming' or 'euclidean'."
        )

    attributes = {}
    for u, v, data in G.edges(data=True):
        distance = float(data["distance"])
        affinity_distance = (
            distance / length
            if distance_geometry == "hamming" and length > 0
            else distance
        )
        affinity = _distance_affinity(affinity_distance)
        edge_attributes = {
            "distance": distance,
            "affinity": affinity,
            "weight": affinity,
            "knn_weight": distance,
            "sim": affinity,
        }
        if distance_geometry == "hamming":
            edge_attributes["normalized_distance"] = affinity_distance
        attributes[(u, v)] = edge_attributes
    if attributes:
        nx.set_edge_attributes(G, attributes)

    is_hamming = distance_geometry == "hamming"
    declare_edge_semantics(
        G,
        constructor=constructor,
        distance_key="distance",
        distance_units="hamming_count" if is_hamming else "embedding_euclidean",
        normalized_distance_key="normalized_distance" if is_hamming else None,
        affinity_key="affinity",
        conductance_key="weight",
        legacy_aliases={"knn_weight": "distance", "sim": "affinity"},
        notes=(
            "Conductance is exp(-normalized Hamming distance)."
            if is_hamming
            else "Conductance is exp(-Euclidean embedding distance)."
        ),
    )


def _declare_tda_graph_semantics(G: nx.Graph) -> None:
    """Declare canonical alpha-complex distance and conductance semantics."""

    declare_edge_semantics(
        G,
        constructor="tda-alpha-complex",
        distance_key="distance",
        distance_units="pca_euclidean",
        affinity_key="affinity",
        conductance_key="weight",
        legacy_aliases={"tda_distance": "distance"},
        notes="Conductance is 1 / (1 + PCA-space Euclidean distance).",
    )


def _declare_diffusion_graph_semantics(G: nx.Graph, *, constructor: str) -> None:
    """Declare canonical diffusion affinity and conductance semantics."""

    declare_edge_semantics(
        G,
        constructor=constructor,
        affinity_key="affinity",
        conductance_key="weight",
        legacy_aliases={"kernel_weight": "affinity"},
        notes=(
            "Conductance is the retained symmetric stationary-measure "
            "diffusion amplitude."
        ),
    )


def _attach_diffusion_metadata(
    graph: nx.Graph,
    *,
    use_stationary: bool,
    power: int | None,
    threshold: float,
) -> None:
    """Record the shared mathematical diffusion contract on a graph."""

    graph.graph["diffusion_semantics"] = {
        "kernel": "stationary_measure_similarity",
        "formula": "Pi^(1/2) P^t Pi^(-1/2)",
        "power": "componentwise_stationary_limit" if use_stationary else power,
        "lazy_probability": 0.5,
        "threshold": threshold,
        "threshold_units": "dimensionless_diffusion_amplitude",
    }


def _attach_sparse_diffusion_construction_metadata(
    graph: nx.Graph,
    *,
    requested_k: int,
    effective_k: int,
    tiebuffer: int,
    backend: str,
    index_type: str,
    max_nnz: int,
    max_work: int,
    affinity_nnz: int,
    transition_nnz: int,
    kernel_nnz: int,
    directed_candidates: int,
    diffusion_work: int,
) -> None:
    """Record the sparse scientific-kernel and resource-control contract."""

    largest_matrix_nnz = max(affinity_nnz, transition_nnz, kernel_nnz)
    graph.graph["diffusion_construction"] = {
        "affinity_source": "symmetric_union_knn_rbf",
        "storage": "csr",
        "diffusion_accuracy": "exact",
        "requested_k": int(requested_k),
        "effective_k": int(effective_k),
        "tiebuffer": int(tiebuffer),
        "tie_rule": "all_returned_candidates_at_exact_kth_distance",
        "candidate_backend_approximate": bool(
            backend == "faiss" and index_type in {"hnsw", "ivf"}
        ),
        "candidate_backend": str(backend),
        "candidate_index_type": str(index_type) if backend == "faiss" else None,
        "directed_candidates": int(directed_candidates),
        "affinity_nnz": int(affinity_nnz),
        "transition_nnz": int(transition_nnz),
        "kernel_nnz": int(kernel_nnz),
        "estimated_scalar_products": int(diffusion_work),
        "max_diffusion_nnz": int(max_nnz),
        "max_diffusion_work": int(max_work),
        "largest_matrix_estimated_bytes": _csr_storage_bytes(
            graph.number_of_nodes(), largest_matrix_nnz
        ),
        # Scaling the largest CSR estimate by four accounts conservatively for
        # the affinity, transition, current power, and product/kernel matrices
        # that can coexist during one exact multiplication.
        "estimated_peak_working_bytes": 4
        * _csr_storage_bytes(graph.number_of_nodes(), largest_matrix_nnz),
    }


def _attach_unit_hamming_edge_attributes(
    G: nx.Graph,
    sequences: Sequence[BaseNumpySequence],
) -> None:
    """
    Attach exact edge attributes for graphs whose edges are guaranteed
    to connect sequences at Hamming count 1.
    """

    if not sequences or G.number_of_edges() == 0:
        return

    seq_len = len(sequences[0])
    norm_distance = float(1.0 / seq_len) if seq_len > 0 else 0.0

    edge_attrs = {
        (u, v): {
            "distance": 1.0,
            "normalized_distance": norm_distance,
            "affinity": 1.0,
            "weight": 1.0,
            "sim": 1.0,
        }
        for u, v in G.edges()
    }
    nx.set_edge_attributes(G, edge_attrs)


def _reversible_lazy_transition(
    affinity: np.ndarray | sparse.spmatrix,
) -> tuple[np.ndarray | csr_matrix, np.ndarray, np.ndarray]:
    """Build a reversible lazy random walk from symmetric affinities.

    The diagonal of the supplied affinity is ignored. Isolated states receive
    an absorbing self transition. All other states use
    ``P = (I + D^-1 W) / 2``, which preserves detailed balance and removes
    periodicity without changing connected components.
    """

    is_sparse = sparse.issparse(affinity)
    if is_sparse:
        weights = affinity.astype(np.float64, copy=True).tocsr()
        if weights.shape[0] != weights.shape[1]:
            raise ValueError("Diffusion affinity must be square.")
        if weights.data.size and (
            not np.all(np.isfinite(weights.data)) or np.any(weights.data < 0.0)
        ):
            raise ValueError("Diffusion affinity must be finite and non-negative.")
        difference = (weights - weights.T).tocsr()
        if difference.data.size and np.max(np.abs(difference.data)) > 1e-10:
            raise ValueError("Diffusion affinity must be symmetric.")
        weights = (0.5 * (weights + weights.T)).tocsr()
        weights.setdiag(0.0)
        weights.eliminate_zeros()
        degrees = np.asarray(weights.sum(axis=1)).ravel()
        isolated = degrees == 0.0
        if np.any(isolated):
            weights = weights.tolil()
            for index in np.flatnonzero(isolated):
                weights[index, index] = 1.0
            weights = weights.tocsr()
            degrees = np.asarray(weights.sum(axis=1)).ravel()
        transition = sparse.diags(1.0 / degrees) @ weights
        transition = (
            0.5 * (sparse.eye(weights.shape[0], format="csr") + transition)
        ).tocsr()
        support = weights.copy()
        support.data = np.ones_like(support.data)
        _, component_labels = sparse.csgraph.connected_components(
            support,
            directed=False,
            return_labels=True,
        )
    else:
        weights = np.asarray(affinity, dtype=np.float64).copy()
        if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
            raise ValueError("Diffusion affinity must be square.")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("Diffusion affinity must be finite and non-negative.")
        if not np.allclose(weights, weights.T, atol=1e-10, rtol=1e-10):
            raise ValueError("Diffusion affinity must be symmetric.")
        weights = 0.5 * (weights + weights.T)
        np.fill_diagonal(weights, 0.0)
        degrees = weights.sum(axis=1)
        isolated = degrees == 0.0
        weights[isolated, isolated] = 1.0
        degrees = weights.sum(axis=1)
        base_transition = weights / degrees[:, None]
        transition = 0.5 * (np.eye(weights.shape[0]) + base_transition)
        _, component_labels = sparse.csgraph.connected_components(
            csr_matrix(weights > 0.0),
            directed=False,
            return_labels=True,
        )

    if degrees.size == 0:
        return transition, np.zeros(0, dtype=np.float64), component_labels
    total_degree = float(degrees.sum())
    stationary = degrees / total_degree

    # Guard the invariant at the construction boundary so later symmetric
    # kernels cannot hide a non-reversible transition.
    if sparse.issparse(transition):
        flux = sparse.diags(stationary) @ transition
        imbalance = (flux - flux.T).tocsr()
        if imbalance.data.size and np.max(np.abs(imbalance.data)) > 1e-10:
            raise ValueError("Failed to construct a detailed-balance transition.")
    else:
        flux = stationary[:, None] * transition
        if not np.allclose(flux, flux.T, atol=1e-10, rtol=1e-10):
            raise ValueError("Failed to construct a detailed-balance transition.")
    return transition, stationary, component_labels


def _sparse_product_requirements(
    left: csr_matrix,
    right: csr_matrix,
    *,
    max_nnz: int,
    max_work: int,
    stage: str,
    work_already: int = 0,
) -> tuple[int, int]:
    """Count exact product structure and arithmetic before multiplication.

    The marker array is O(number of columns), while traversal stops as soon as
    either public feasibility budget is exceeded. This prevents SciPy from
    allocating an unexpectedly dense sparse-product result merely to discover
    that the exact request is infeasible.
    """

    left = left.tocsr()
    right = right.tocsr()
    if left.shape[1] != right.shape[0]:
        raise ValueError("Sparse diffusion product matrices are misaligned.")

    n_rows, n_cols = left.shape[0], right.shape[1]
    marker = np.full(n_cols, -1, dtype=np.int64)
    output_nnz = 0
    scalar_products = 0
    right_row_nnz = np.diff(right.indptr)

    for row in range(n_rows):
        intermediates = left.indices[left.indptr[row] : left.indptr[row + 1]]
        row_work = int(right_row_nnz[intermediates].sum(dtype=np.int64))
        scalar_products += row_work
        if work_already + scalar_products > max_work:
            _raise_diffusion_budget_error(
                stage=stage,
                quantity="work",
                estimate=work_already + scalar_products,
                limit=max_work,
            )

        row_nnz = 0
        for middle in intermediates:
            columns = right.indices[
                right.indptr[middle] : right.indptr[middle + 1]
            ]
            unseen = marker[columns] != row
            if np.any(unseen):
                new_columns = columns[unseen]
                marker[new_columns] = row
                row_nnz += int(new_columns.size)
                if row_nnz == n_cols:
                    break

        output_nnz += row_nnz
        if output_nnz > max_nnz:
            _raise_diffusion_budget_error(
                stage=stage,
                quantity="nonzeros",
                estimate=output_nnz,
                limit=max_nnz,
            )

    return output_nnz, scalar_products


def _checked_sparse_matrix_power(
    transition: csr_matrix,
    power: int,
    *,
    max_nnz: int,
    max_work: int,
) -> tuple[csr_matrix, int]:
    """Compute an exact sparse power after per-step structure/work checks."""

    n_states = transition.shape[0]
    powered = sparse.eye(n_states, format="csr", dtype=np.float64)
    total_work = 0
    for step in range(1, power + 1):
        _, step_work = _sparse_product_requirements(
            powered,
            transition,
            max_nnz=max_nnz,
            max_work=max_work,
            stage=f"power step {step}/{power}",
            work_already=total_work,
        )
        total_work += step_work
        powered = (powered @ transition).tocsr()
        powered.eliminate_zeros()
        if powered.nnz > max_nnz:  # defensive against unexpected SciPy structure
            _raise_diffusion_budget_error(
                stage=f"power step {step}/{power}",
                quantity="nonzeros",
                estimate=powered.nnz,
                limit=max_nnz,
            )
    return powered, total_work


def _reversible_diffusion_kernel(
    transition: np.ndarray | sparse.spmatrix,
    stationary: np.ndarray,
    component_labels: np.ndarray,
    *,
    stationary_limit: bool,
    power: int | None,
    max_nnz: int | None = None,
    max_work: int | None = None,
    _diagnostics: dict[str, int] | None = None,
) -> np.ndarray | csr_matrix:
    """Return a symmetric stationary-measure diffusion kernel.

    For finite ``t``, the kernel is
    ``K_t = Pi^(1/2) P^t Pi^(-1/2)``. Numerical averaging with its transpose
    uses both orientations explicitly. The stationary limit is evaluated
    component by component, so distinct communicating classes have zero
    pairwise connectivity.
    """

    probabilities = np.asarray(stationary, dtype=np.float64)
    labels = np.asarray(component_labels, dtype=np.int64)
    n_states = probabilities.size
    if transition.shape != (n_states, n_states) or labels.shape != (n_states,):
        raise ValueError("Transition, stationary distribution, and labels are misaligned.")
    if n_states == 0:
        return csr_matrix((0, 0)) if sparse.issparse(transition) else np.empty((0, 0))
    if np.any(probabilities <= 0.0) or not np.isclose(probabilities.sum(), 1.0):
        raise ValueError("Stationary probabilities must be positive and sum to one.")

    if stationary_limit:
        if sparse.issparse(transition) and max_nnz is not None:
            component_sizes = np.bincount(labels)
            required_nnz = sum(int(size) ** 2 for size in component_sizes)
            if required_nnz > max_nnz:
                _raise_diffusion_budget_error(
                    stage="componentwise stationary limit",
                    quantity="nonzeros",
                    estimate=required_nnz,
                    limit=max_nnz,
                )
            if max_work is not None and required_nnz > max_work:
                _raise_diffusion_budget_error(
                    stage="componentwise stationary limit",
                    quantity="work",
                    estimate=required_nnz,
                    limit=max_work,
                )
            if _diagnostics is not None:
                _diagnostics["estimated_scalar_products"] = required_nnz

            rows = np.empty(required_nnz, dtype=np.int64)
            cols = np.empty(required_nnz, dtype=np.int64)
            values = np.empty(required_nnz, dtype=np.float64)
            cursor = 0
            component_order = np.argsort(labels, kind="stable")
            offsets = np.concatenate(([0], np.cumsum(component_sizes)))
            for start, stop in zip(offsets[:-1], offsets[1:]):
                indices = component_order[start:stop]
                mass = float(probabilities[indices].sum())
                root = np.sqrt(probabilities[indices])
                for local_row, row in enumerate(indices):
                    next_cursor = cursor + indices.size
                    rows[cursor:next_cursor] = row
                    cols[cursor:next_cursor] = indices
                    values[cursor:next_cursor] = root[local_row] * root / mass
                    cursor = next_cursor
            return coo_matrix(
                (values, (rows, cols)),
                shape=(n_states, n_states),
            ).tocsr()

        kernel = np.zeros((n_states, n_states), dtype=np.float64)
        for component in np.unique(labels):
            indices = np.flatnonzero(labels == component)
            mass = float(probabilities[indices].sum())
            root = np.sqrt(probabilities[indices])
            kernel[np.ix_(indices, indices)] = np.outer(root, root) / mass
        return kernel

    if power is None or power < 1:
        raise ValueError("Finite diffusion kernels require an integer power >= 1.")
    root = np.sqrt(probabilities)
    inverse_root = 1.0 / root
    if sparse.issparse(transition):
        if max_nnz is not None:
            if max_work is None:
                raise ValueError("`max_work` is required when `max_nnz` is set.")
            powered, total_work = _checked_sparse_matrix_power(
                transition.tocsr(),
                power,
                max_nnz=max_nnz,
                max_work=max_work,
            )
            if _diagnostics is not None:
                _diagnostics["estimated_scalar_products"] = total_work
        else:
            powered = sparse.eye(n_states, format="csr")
            for _ in range(power):
                powered = powered @ transition
        kernel = sparse.diags(root) @ powered @ sparse.diags(inverse_root)
        kernel = (0.5 * (kernel + kernel.T)).tocsr()
        if kernel.data.size:
            kernel.data[np.abs(kernel.data) < 1e-15] = 0.0
            kernel.eliminate_zeros()
        return kernel

    powered = np.linalg.matrix_power(np.asarray(transition), power)
    kernel = root[:, None] * powered * inverse_root[None, :]
    return 0.5 * (kernel + kernel.T)


def _threshold_undirected_kernel(
    kernel: np.ndarray | sparse.spmatrix,
    *,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return upper-triangle kernel entries strictly above ``threshold``."""

    if sparse.issparse(kernel):
        upper = sp_triu(kernel, k=1, format="coo")
        keep = np.isfinite(upper.data) & (upper.data > threshold)
        return upper.row[keep], upper.col[keep], upper.data[keep]

    values = np.asarray(kernel, dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < -1e-12):
        raise ValueError("Diffusion kernel must be finite and non-negative.")
    mask = np.triu(values > threshold, k=1)
    rows, cols = np.where(mask)
    return rows, cols, values[rows, cols]


def _select_diffusion_knn_candidates(
    search_features: np.ndarray,
    neighbour_indices: np.ndarray,
    *,
    k: int,
    distance_geometry: Literal["hamming", "euclidean"],
) -> list[np.ndarray]:
    """Rerank backend candidates and retain all observed ties at rank ``k``.

    Approximate backends define the candidate pool, but distances and ties are
    evaluated exactly in the declared search geometry. ``tiebuffer`` affects
    the scientific graph only when the extra returned candidates tie the kth
    candidate; it cannot recover neighbours omitted by an approximate index.
    """

    features = np.asarray(search_features)
    selected: list[np.ndarray] = []
    for row, raw_candidates in enumerate(np.asarray(neighbour_indices)):
        candidates = np.asarray(raw_candidates, dtype=np.int64)
        candidates = candidates[(candidates >= 0) & (candidates != row)]
        if candidates.size:
            candidates = np.unique(candidates)
        if candidates.size == 0:
            selected.append(np.empty(0, dtype=np.int64))
            continue

        if distance_geometry == "hamming":
            distances = np.mean(features[candidates] != features[row], axis=1)
        else:
            deltas = features[candidates] - features[row]
            distances = np.linalg.norm(deltas, axis=1)
        order = np.argsort(distances, kind="stable")
        candidates = candidates[order]
        distances = distances[order]
        cutoff = distances[min(k, candidates.size) - 1]
        tolerance = np.finfo(np.float64).eps * max(1.0, abs(float(cutoff))) * 8.0
        selected.append(candidates[distances <= cutoff + tolerance])
    return selected


def _sparse_rbf_affinity_from_candidates(
    embeddings: np.ndarray,
    candidates_by_row: Sequence[np.ndarray],
    *,
    gamma: float,
    max_nnz: int,
) -> tuple[csr_matrix, int]:
    """Construct the symmetric union-kNN RBF affinity without densification."""

    n_points = embeddings.shape[0]
    directed_nnz = int(sum(len(candidates) for candidates in candidates_by_row))
    symmetric_bound = min(n_points * n_points, 2 * directed_nnz)
    transition_bound = min(n_points * n_points, symmetric_bound + n_points)
    if transition_bound > max_nnz:
        _raise_diffusion_budget_error(
            stage="kNN affinity and lazy transition",
            quantity="nonzeros",
            estimate=transition_bound,
            limit=max_nnz,
        )

    row_parts = []
    col_parts = []
    value_parts = []
    for row, candidates in enumerate(candidates_by_row):
        candidates = np.asarray(candidates, dtype=np.int64)
        if candidates.size == 0:
            continue
        deltas = embeddings[candidates] - embeddings[row]
        squared_distances = np.einsum("ij,ij->i", deltas, deltas)
        values = np.exp(-float(gamma) * squared_distances)
        keep = np.isfinite(values) & (values > 0.0)
        if np.any(keep):
            row_parts.append(np.full(np.count_nonzero(keep), row, dtype=np.int64))
            col_parts.append(candidates[keep])
            value_parts.append(values[keep])

    if not row_parts:
        return csr_matrix((n_points, n_points), dtype=np.float64), directed_nnz
    directed = coo_matrix(
        (
            np.concatenate(value_parts),
            (np.concatenate(row_parts), np.concatenate(col_parts)),
        ),
        shape=(n_points, n_points),
    ).tocsr()
    affinity = directed.maximum(directed.T).tocsr()
    affinity.setdiag(0.0)
    affinity.eliminate_zeros()
    if affinity.nnz + n_points > max_nnz:
        _raise_diffusion_budget_error(
            stage="kNN affinity and lazy transition",
            quantity="nonzeros",
            estimate=affinity.nnz + n_points,
            limit=max_nnz,
        )
    return affinity, directed_nnz


def _symmetric_affinity_from_scores(scores: csr_matrix, *, tau: float) -> csr_matrix:
    """Exponentiate symmetric sparse scores without row-dependent scaling."""

    affinity = scores.astype(np.float64, copy=True).tocsr()
    difference = (affinity - affinity.T).tocsr()
    if difference.data.size and np.max(np.abs(difference.data)) > 1e-10:
        raise ValueError("Evolutionary edge scores must be symmetric.")
    if affinity.data.size:
        if not np.all(np.isfinite(affinity.data)):
            raise ValueError("Evolutionary edge scores must be finite.")
        scaled = (affinity.data - np.max(affinity.data)) / float(tau)
        scaled = np.maximum(scaled, np.log(np.finfo(np.float64).tiny))
        affinity.data = np.exp(scaled)
    affinity = (0.5 * (affinity + affinity.T)).tocsr()
    affinity.setdiag(0.0)
    affinity.eliminate_zeros()
    return affinity

def _pack_binary(seqs: list[BaseNumpySequence]) -> np.ndarray:
    """
    Helper function to convert a list of `BaseNumpySequences` itno an
    int encoded array
    """
    
    # (n, L)
    arr = np.stack([s.to_array().astype(np.uint8) for s in seqs], axis=0)  
    if not np.isin(arr, [0, 1]).all():

        raise ValueError("Binary builder requires sequences with symbols {0,1}.")
    L = arr.shape[1]
    if L > 64:
        raise ValueError("Bit-pack assumes L <= 64.")
    
    # bit for each pos
    powers = (1 << np.arange(L, dtype=np.uint64))
    
    return (arr.astype(np.uint64) * powers).sum(axis=1, dtype=np.uint64)

def _build_hamming_csr_binary(sequences: list[BinarySequence]) -> csr_matrix:
    """
    Function to build undirected CSR adjacency for a binary Hamming
    graph using XOR neighbor generation.

    Parameters
    ----------
    sequences : List[BinarySequence]
        The input BinarySequence objects used to construct the
        Hamming graph. 
    
    Returns
    -------
    A : sp.csr_matrix
        Sparse adjacency matrix. 
    """
    # Guardrails
    if len(sequences) == 0:
        return csr_matrix((0, 0))
    
    n = len(sequences)
    bitstrings = _pack_binary(sequences)

    # infer L from used bits (safe if all positions vary at least once)
    max_bit = int(max(int(b).bit_length() for b in bitstrings))
    L = max(1, max_bit)

    # Map packed bitstring -> list of indices (to handle duplicates correctly)
    bs_to_indices: dict[int, list[int]] = {}
    for idx, bs in enumerate(bitstrings):
        bs_to_indices.setdefault(int(bs), []).append(idx)

    rows_list: list[int] = []
    cols_list: list[int] = []

    # For each node, flip each bit and connect to all occurrences of the neighbor code
    for i, s in enumerate(bitstrings):
        s_int = int(s)
        for pos in range(L):
            t = s_int ^ (1 << pos)
            js = bs_to_indices.get(int(t))
            if not js:
                continue
            for j in js:
                if i >= j:
                    continue
                rows_list.extend([i, j])
                cols_list.extend([j, i])

    if rows_list:
        rows = np.asarray(rows_list, dtype=np.int32)
        cols = np.asarray(cols_list, dtype=np.int32)
        data = np.ones(len(rows), dtype=np.float32)
    else:
        rows = np.zeros(0, dtype=np.int32)
        cols = np.zeros(0, dtype=np.int32)
        data = np.zeros(0, dtype=np.float32)
    A = csr_matrix((data, (rows, cols)), shape=(n, n))
    return A

def create_hamming_graph_binary(sequences: list[BinarySequence], *, _compute_hamming_edges: bool = False) -> nx.Graph:
    """
    Function to build a undirected Hamming graph using efficiency bit
    wise (XOR) operations. 

    Parameters
    ----------
    sequences : List[BinarySequence]
        The input BinarySequence objects used to construct the
        Hamming graph. 
    _compute_hamming_edges : bool, default=False
        Legacy option retained for compatibility; edge annotation is disabled.

    Returns
    -------
    G : nx.Graph
        The undirected graph that can construct the `FitnessLandscape`
        class. 
    """
    _compute_hamming_edges = _force_disable_hamming_edge_computation(_compute_hamming_edges)
    _validate_sequence_collection(sequences)
    if not all(isinstance(sequence, BinarySequence) for sequence in sequences):
        raise TypeError("`sequences` must contain only BinarySequence instances.")

    A = _build_hamming_csr_binary(sequences)
    G = nx.from_scipy_sparse_array(A) 
    
    # attach node attributes for `FitnessLandscape` constructor.s
    for i, seq in enumerate(sequences):
        G.nodes[i]['sequence'] = seq

    _attach_unit_hamming_edge_attributes(G, sequences)
    _declare_hamming_graph_semantics(G, constructor="hamming-binary")
    
    return G


def _encode_multiallele(seqs: list[BaseNumpySequence]) -> tuple[np.ndarray, dict[str,int]]:
    """
    Helper function to map string symbols in the `BaseNumpySequence`
    alphabet to contiguous integers. 
    """

    # collect alphabet in order of first appearance to keep mapping stable
    seen = {}
    mats = []
    for s in seqs:
        arr = s.to_array()
        mats.append(arr)
        for sym in map(str, arr):
            if sym not in seen:
                seen[sym] = len(seen)
    mapping = seen
    int_mat = np.stack([[mapping[str(x)] for x in s.to_array()] for s in seqs], axis=0).astype(np.int32)
    return int_mat, mapping  # (n,L)

def _radix_keyspace_fits_int64(base: int, length: int) -> bool:
    """
    Return ``True`` when a base-``base`` radix key of ``length`` digits
    can be represented exactly in signed int64.
    """

    limit = int(np.iinfo(np.int64).max)
    keyspace = 1
    for _ in range(length):
        keyspace *= int(base)
        if keyspace - 1 > limit:
            return False
    return True

def _append_cross_allele_edges(
    rows: list[np.ndarray],
    cols: list[np.ndarray],
    block_idx: np.ndarray,
    block_allele: np.ndarray,
) -> None:
    """Append all cross-allele edges for a masked-sequence equivalence block."""

    unique_alleles, inverse = np.unique(block_allele, return_inverse=True)
    groups = [block_idx[inverse == allele_idx] for allele_idx in range(len(unique_alleles))]

    for left_idx, src in enumerate(groups):
        if src.size == 0:
            continue
        for dst in groups[left_idx + 1:]:
            if dst.size == 0:
                continue
            src_rep = np.repeat(src, dst.size)
            dst_tile = np.tile(dst, src.size)
            rows.append(src_rep)
            cols.append(dst_tile)
            rows.append(dst_tile)
            cols.append(src_rep)

def _csr_from_symmetric_edge_lists(
    rows: list[np.ndarray],
    cols: list[np.ndarray],
    *,
    n: int,
) -> csr_matrix:
    """Materialise a deduplicated symmetric CSR adjacency matrix."""

    if not rows:
        empty = np.zeros(0, dtype=np.int32)
        return csr_matrix((np.zeros(0, dtype=np.float32), (empty, empty)), shape=(n, n))

    row_arr = np.concatenate(rows).astype(np.int32, copy=False)
    col_arr = np.concatenate(cols).astype(np.int32, copy=False)

    order = np.lexsort((col_arr, row_arr))
    row_arr, col_arr = row_arr[order], col_arr[order]

    keep = np.ones_like(row_arr, dtype=bool)
    keep[1:] = (row_arr[1:] != row_arr[:-1]) | (col_arr[1:] != col_arr[:-1])
    row_arr, col_arr = row_arr[keep], col_arr[keep]

    data = np.ones(row_arr.size, dtype=np.float32)
    return csr_matrix((data, (row_arr, col_arr)), shape=(n, n))

def _build_hamming_csr_multiallele_masked_radix(X: np.ndarray, *, base: int) -> csr_matrix:
    """Fast radix-key implementation for multiallelic Hamming graphs."""

    n, L = X.shape
    powers = (base ** np.arange(L, dtype=np.int64))
    key_full = (X * powers).sum(axis=1, dtype=np.int64)

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []

    for p in range(L):
        masked = key_full - (X[:, p].astype(np.int64) * powers[p])
        order = np.argsort(masked, kind='stable')
        masked_sorted = masked[order]
        xp = X[:, p][order]

        start = 0
        while start < n:
            end = start + 1
            while end < n and masked_sorted[end] == masked_sorted[start]:
                end += 1
            if end - start >= 2:
                block_idx = order[start:end]
                block_allele = xp[start:end]
                _append_cross_allele_edges(rows, cols, block_idx, block_allele)
            start = end

    return _csr_from_symmetric_edge_lists(rows, cols, n=n)

def _build_hamming_csr_multiallele_masked_exact(X: np.ndarray) -> csr_matrix:
    """
    Exact overflow-safe multiallelic Hamming builder.

    Long protein sequences can overflow the int64 radix keys used by
    the fast path. In those cases, group rows by their masked sequence
    bytes instead of numeric radix encodings.
    """

    n, L = X.shape
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []

    for p in range(L):
        if L == 1:
            order = np.arange(n, dtype=np.int32)
            masked_sorted = np.zeros(n, dtype=np.int8)
        else:
            masked = np.ascontiguousarray(np.concatenate((X[:, :p], X[:, p + 1:]), axis=1))
            key_dtype = np.dtype((np.void, masked.dtype.itemsize * masked.shape[1]))
            masked_keys = masked.view(key_dtype).reshape(-1)
            order = np.argsort(masked_keys, kind='stable')
            masked_sorted = masked_keys[order]

        xp = X[:, p][order]

        start = 0
        while start < n:
            end = start + 1
            while end < n and masked_sorted[end] == masked_sorted[start]:
                end += 1
            if end - start >= 2:
                block_idx = order[start:end]
                block_allele = xp[start:end]
                _append_cross_allele_edges(rows, cols, block_idx, block_allele)
            start = end

    return _csr_from_symmetric_edge_lists(rows, cols, n=n)

def _build_hamming_csr_multiallele_masked(sequences: list[BaseNumpySequence]) -> csr_matrix:
    """
    Function to build a sparse Hamming adjacency matrix using a
    radix-encoded (base B for B alleles) masking algorithm.

    Parameters
    ----------
    sequences : List[BaseNumpySequences] 
        List of input sequences. 
    
    Returns
    -------
    A : sp.csr_matrix
        The sparse Hamming adjacency matrix. 
    """

    # Guardrails
    if len(sequences) == 0:
        return csr_matrix((0, 0))

    X, _ = _encode_multiallele(sequences)  # (n,L) int32
    n, L = X.shape
    base = int(X.max()) + 1

    if _radix_keyspace_fits_int64(base, L):
        return _build_hamming_csr_multiallele_masked_radix(X, base=base)

    return _build_hamming_csr_multiallele_masked_exact(X)

def create_hamming_graph_multiallele(sequences: list[BaseNumpySequence], *, _compute_hamming_edges: bool = False) -> nx.Graph:
    """
    Function to create a Hamming graph using B-radix encoded sequence
    masking to identify Hamming neighbors. 

    Parameters
    ----------
    sequences : List[BaseNumpySequence]
        The list of input sequences to construct the graph from. 
    _compute_hamming_edges : bool, default=False
        Legacy option retained for compatibility; edge annotation is disabled.
    
    Returns
    -------
    G : nx.Graph
        The undirected graph with edge and node features accepted by
        the `FitnessLandscape` from graph constructor.
    """

    _compute_hamming_edges = _force_disable_hamming_edge_computation(_compute_hamming_edges)
    _validate_sequence_collection(sequences)

    A = _build_hamming_csr_multiallele_masked(sequences)
    G = nx.from_scipy_sparse_array(A)
    for i, seq in enumerate(sequences):
        G.nodes[i]['sequence'] = seq

    _attach_unit_hamming_edge_attributes(G, sequences)
    _declare_hamming_graph_semantics(G, constructor="hamming-multiallele")

    if len(sequences) == 0 or G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return G
    
    return G

# Main public method
def create_hamming_graph(sequences: List[BaseNumpySequence],
                         _backend: Literal['auto', 'binary_xor', 'masked'] = 'auto',
                         *,
                         _compute_hamming_edges: bool = False) -> nx.Graph:
    """
    Create a Hamming graph from sequences and fitness values. In a
    Hamming graph, nodes represent sequences and edges connect
    sequences that differ by exactly one position (Hamming
    distance = 1).

    Parameters
    ----------
    sequences : list of Sequence or array-like
        Sequences to connect.
    _backend : str, default=`aut`
        Backend to compute Hamming neighbors. 
        -`binary_xor`: applies binary XOR operation to find bit-encoded
        sequences that differ by precisely 1 bit in an indexed lookup
        table. Scales in O(n * L). Applies exlusively to the
        `BinarySequence` class. 
        - `masked` : applies a position p mask over radix (base B)
        enocoded sequences to find sequences that are identical outside
        of position p. Scales in O(L n log n)
        - `auto` : automatically chooses backend based on the sequence
        type.
    _compute_hamming_edges : bool, default=False
        Accepted for compatibility. Hamming graphs always expose exact raw and
        normalized Hamming distances, so no additional pass is required.

    Returns
    -------
    networkx.Graph
        Hamming graph.
    """
    
    _compute_hamming_edges = _force_disable_hamming_edge_computation(_compute_hamming_edges)
    _validate_sequence_collection(sequences)

    # Safety check all sequences are binary classes.
    is_binary = all(isinstance(s, BinarySequence) for s in sequences)

    if _backend == "auto":
        _backend = "binary_xor" if is_binary else "masked"

    if _backend == "binary_xor":
        if not is_binary:
            raise ValueError("backend='binary_xor' requires binary sequences {0,1}.")
        return create_hamming_graph_binary(sequences, _compute_hamming_edges=_compute_hamming_edges)
    elif _backend == "masked":
        return create_hamming_graph_multiallele(sequences, _compute_hamming_edges=_compute_hamming_edges)
    else:
        raise ValueError(f"Unknown `_backend`: {_backend}")


def _one_hot_matrix_amino(seqs: List[BaseNumpySequence]) -> np.ndarray:
    """
    Helper function to to convert a list of BaseNumpySequence objects
    into a one hot encded matrix. Assumes all sequences share the same
    alphabet.

    Parameters
    ----------
    seqs : List[BaseNumpySequence] 
        The sequences to convert to one-hot. 

    Returns
    -------
    X : np.ndarray
        The one hot encoded matrix.    
    """
    n, L = _validate_sequence_collection(seqs)
    if n == 0:
        return np.empty((0, 0), dtype=np.float32)

    # Use a shared alphabet in stable first-appearance order. Individual
    # sequence objects may expose narrower inferred alphabets even when the
    # collection as a whole is aligned.
    alphabet: list[str] = []
    for sequence in seqs:
        for symbol in map(str, sequence.to_array()):
            if symbol not in alphabet:
                alphabet.append(symbol)
    amap = {symbol: i for i, symbol in enumerate(alphabet)}
    X = np.zeros((n, L * len(alphabet)), dtype=np.float32)
    W = len(alphabet)
    for r, s in enumerate(seqs):
        arr = s.to_array()
        for p, sym in enumerate(arr):
            X[r, p*W + amap[str(sym)]] = 1.0
    return X

def _is_binary_like_matrix(X: np.ndarray, sample: int = 10000) -> bool:
    """
    Lightweight heuristic to decide if an embedding matrix represents
    discrete encodings (e.g., integer labels or one-hot vectors). For
    such matrices Hamming distance is appropriate; otherwise we should
    use a continuous metric like Euclidean.
    """
    if np.issubdtype(X.dtype, np.bool_) or np.issubdtype(X.dtype, np.integer):
        return True

    if not np.issubdtype(X.dtype, np.floating):
        return False

    flat = X.ravel()
    if flat.size > sample:
        flat = flat[:sample]
    return np.all((flat == 0.0) | (flat == 1.0))

def _resolve_balltree_metric(X: np.ndarray, metric: str | None = None) -> str:
    """
    Choose an appropriate BallTree metric when none is provided.
    Defaults to Hamming for discrete encodings and Euclidean otherwise.
    """
    if metric is not None:
        return metric
    return "hamming" if _is_binary_like_matrix(X) else "euclidean"


def _prepare_knn_search_space(
    sequences: List[BaseNumpySequence],
    embeddings: np.ndarray | None,
    *,
    embedding_domain: str | None,
    backend: Literal["faiss", "balltree"],
    faiss_metric: Literal["ip", "l2"],
) -> tuple[np.ndarray, Literal["hamming", "euclidean"], str, str]:
    """Resolve the feature matrix and metric for a kNN search.

    Sequence/OHE searches use Hamming geometry. PLM, composition, and direct
    embedding searches use ordinary Euclidean geometry; FAISS represents that
    geometry with its squared-L2 index, irrespective of the sequence-oriented
    ``faiss_metric`` option.
    """

    use_embedding_geometry = embedding_domain in _EMBEDDING_KNN_DOMAINS or (
        embeddings is not None and embedding_domain != "ohe"
    )
    n_sequences, _ = _validate_sequence_collection(
        sequences,
        require_aligned=not use_embedding_geometry,
    )
    if embedding_domain is not None and embedding_domain not in {
        "ohe",
        *_EMBEDDING_KNN_DOMAINS,
    }:
        raise ValueError(
            "`embedding_domain` must be 'ohe', 'plm', 'composition', or None; "
            f"got {embedding_domain!r}."
        )

    matrix = None
    if embeddings is not None:
        matrix = _validate_embedding_matrix(
            embeddings,
            n_sequences=n_sequences,
        )
    elif embedding_domain in _EMBEDDING_KNN_DOMAINS:
        raise ValueError(
            f"`embedding_domain={embedding_domain!r}` requires an aligned "
            "embedding matrix for kNN construction."
        )

    use_embedding_geometry = matrix is not None and embedding_domain != "ohe"
    if use_embedding_geometry:
        metric = "l2" if backend == "faiss" else "euclidean"
        return matrix, "euclidean", metric, embedding_domain or "embedding"

    if n_sequences == 0:
        metric = faiss_metric if backend == "faiss" else "hamming"
        return np.empty((0, 0), dtype=np.float32), "hamming", metric, "ohe"

    if backend == "faiss":
        features = _one_hot_matrix_amino(sequences)
        metric = faiss_metric
    else:
        features, _ = _encode_multiallele(sequences)
        metric = "hamming"
    return features, "hamming", metric, "ohe"


def _attach_knn_search_metadata(
    graph: nx.Graph,
    *,
    backend: str,
    metric: str,
    distance_geometry: str,
    embedding_domain: str,
    role: Literal["graph", "prefilter"],
) -> None:
    """Record the feature-space contract used by a kNN graph or prefilter."""

    graph.graph["landscapy_knn_search"] = {
        "role": role,
        "backend": str(backend),
        "metric": str(metric),
        "distance_geometry": str(distance_geometry),
        "embedding_domain": str(embedding_domain),
    }

def _find_knn_balltree(X : np.ndarray,
                       k : int,
                       tiebuffer : int = 0,
                       *,
                       include_self: bool = False,
                       metric: str | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Helper function to find nearest neighbors by BallTree.

    Parameters
    ----------
    X : np.ndarray
        The encoded sequence array. 
    
    k : int 
        The number of neighbours to find.
        
    tiebuffer : int, defaut=1
        The tiebuffer for equidistant neighbors above k. 

    metric : str, optional
        Distance metric to use. When ``None`` (default) the function
        auto-selects ``'hamming'`` for discrete/binary encodings and
        ``'euclidean'`` otherwise.
    
    Returns
    -------
    dists, inds : np.ndarray
        Tuple of distances and indices.
    """
    raw = np.asarray(X)
    if raw.ndim != 2:
        raise ValueError("`X` must be 2-D with shape (n_samples, n_features).")
    validated = _validate_embedding_matrix(raw, n_sequences=raw.shape[0], name="X")
    # Preserve integer-coded sequence matrices so automatic metric selection
    # continues to recognize categorical/Hamming inputs.
    X = raw if np.issubdtype(raw.dtype, np.number) else validated
    k = _validate_integer(k, name="k", minimum=1)
    tiebuffer = _validate_integer(tiebuffer, name="tiebuffer", minimum=0)
    include_self = _validate_boolean(include_self, name="include_self")
    n = X.shape[0]
    if n == 0:
        empty = np.empty((0, 0), dtype=np.float64)
        return empty, empty.astype(np.int64)
    sklearn_neighbors = require_optional(
        "sklearn.neighbors",
        extra="knn",
        purpose="BallTree nearest-neighbour graph construction",
    )
    metric = _resolve_balltree_metric(X, metric)
    nn = sklearn_neighbors.NearestNeighbors(algorithm='auto', metric=metric)
    nn.fit(X)
    kq = min(k + (0 if include_self else 1) + tiebuffer, n)
    dists, inds = nn.kneighbors(X, n_neighbors=kq, return_distance=True)
    return dists, inds

def _find_knn_faiss(X: np.ndarray,
                    k: int,
                    index_type: Literal['hnsw', 'flat', 'ivf'] = "hnsw",
                    metric: Literal['ip', 'l2'] = 'ip',
                    use_gpu: bool = False,
                    hnsw_M: int = 32,
                    include_self: bool = False,
                    tiebuffer : int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Helper function to find nearest neighbors by FAISS backend.

    Parameters
    ----------
    X : np.ndarray
        The encoded sequence array. 
    
    k : int 
        The number of neighbours to find.

    index_type : str, default=`hnsw`
        The faiss index type. Options are:
        - flat (exact) for small n. 
        - hnsw (approximate) for large n. 
        - ivf (approximate) for very large n.
    
    metric : str, default=`ip`
        The faiss metric to use specificall for hnsw. Options are:
        - `ip` : Inner product
        - `l2` : L2 norm 
    
    include_self : bool, default=`False`
        Boolean to include self in the neighbor list.
    
    use_gpu : bool, default=`False`
        Boolean to use FAISS GPU acceleration (application only to the
        flat index).
    
    hsnw_M : int, default = 32
        The hnsw dimesnion.
    
    include_self : bool, default=`False`
        Boolean to include self edges.
    
    tiebuffer : int, default=128
        The number of hits kept in buffer to eliminate ties.
    
    Returns
    -------
    dists, inds : np.ndarray
        Tuple of distances and indices.

    """
    raw = np.asarray(X)
    if raw.ndim != 2:
        raise ValueError("`X` must be 2-D with shape (n_samples, n_features).")
    X = _validate_embedding_matrix(raw, n_sequences=raw.shape[0], name="X")
    k = _validate_integer(k, name="k", minimum=1)
    tiebuffer = _validate_integer(tiebuffer, name="tiebuffer", minimum=0)
    hnsw_M = _validate_integer(hnsw_M, name="hnsw_M", minimum=1)
    include_self = _validate_boolean(include_self, name="include_self")
    use_gpu = _validate_boolean(use_gpu, name="use_gpu")
    if index_type not in _FAISS_INDEX_TYPES:
        raise ValueError(
            f"Unsupported FAISS index_type {index_type!r}. Expected one of "
            f"{sorted(_FAISS_INDEX_TYPES)}."
        )
    if metric not in _FAISS_METRICS:
        raise ValueError(
            f"Unsupported FAISS metric {metric!r}. Expected 'ip' or 'l2'."
        )
    if use_gpu and index_type != "flat":
        raise ValueError("`use_gpu=True` is supported only with index_type='flat'.")

    X = np.ascontiguousarray(X, dtype=np.float32)
    n, d = X.shape
    if n == 0:
        empty = np.empty((0, 0), dtype=np.float32)
        return empty, empty.astype(np.int64)
    try:
        faiss = require_optional(
            "faiss",
            extra="faiss",
            purpose="FAISS nearest-neighbour graph construction",
        )
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            f"{error} FAISS wheels are platform-specific; rerun with "
            "`backend='balltree'` to use the portable scikit-learn fallback.",
            name=error.name,
        ) from error
    # Set the FAISS metric so easy conversion back to hamming distance.
    if metric == "ip":
        faiss_metric = faiss.METRIC_INNER_PRODUCT
    elif metric == "l2":
        faiss_metric = faiss.METRIC_L2
    else:  # guarded before importing the backend
        raise AssertionError("unreachable FAISS metric")

    # FAISS index
    if index_type == "flat":
        if metric == "ip":
            index = faiss.IndexFlatIP(d)
        else:  # 'l2'
            index = faiss.IndexFlatL2(d)
    
    elif index_type == "hnsw":
        # Catch error in setting `faiss_metrix`.
        try:
            index = faiss.IndexHNSWFlat(d, hnsw_M, faiss_metric)
        except TypeError:
            # Fallback: default is L2
            index = faiss.IndexHNSWFlat(d, hnsw_M)
            if metric != "l2":
                raise RuntimeError(
                    "IndexHNSWFlat in this FAISS build uses L2 only; set metric='l2' "
                    "or switch to 'flat'/'ivf' with METRIC_INNER_PRODUCT."
                )
            
    elif index_type == "ivf":
        # Keep every coarse centroid trainable on small explicit-IVF inputs.
        nlist = min(n, max(1, int(np.sqrt(n))))
        quant = faiss.IndexFlatIP(d) if metric == "ip" else faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quant, d, nlist, faiss_metric)
        index.train(X)
        index.nprobe = min(64, nlist)
    else:
        raise ValueError(f"Expected `index_type` to be in [`flat`, `hnsw`, `ivf`], found {index_type}")

    if use_gpu:
        if not hasattr(faiss, "StandardGpuResources"):
            raise RuntimeError(
                "`use_gpu=True` is unavailable because the installed FAISS build "
                "has no GPU support. Install a GPU-enabled FAISS build supported "
                "by this OS, rerun with `use_gpu=False` for CPU FAISS, or select "
                "`backend='balltree'`."
            )
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    index.add(X)
    
    # Include consideration for self edges and a tiebuffer.
    kq = min(k + (0 if include_self else 1) + tiebuffer, n)
    dists, inds = index.search(X, kq)
    return dists, inds


def _create_knn_graph_balltree(sequences: List[BaseNumpySequence],
                               k: int,
                               search_features: np.ndarray,
                               distance_geometry: Literal["hamming", "euclidean"],
                               include_self: bool = False,
                               tie_policy: Literal['all', 'min_index', 'random'] = 'all',
                               tiebuffer: int = 128,
                               seed: int = 42,
                               eps: float = 1e-12) -> nx.Graph:
    """
    Function to create an exact KNN using the scipy `BallTree`
    algorithm.

    Parameters
    -----------
    sequences : List[BaseNumpySequence]
        The list of input sequences. 
    k : int
        The number of neighbours to connect each sequence to. 
    
    tie_policy : str, default=`all`
        The tie policy for when there are more than k equidistant
        neighbors found. Options are:
        - `all` : all neighbors are kept and the graph becomes
        irregular.
        - `min_index` : The "first" connection is kept, the others are
        arbitrarily removed. The graph remains k regular. 
        - `random` : equidistant edges are kept at random.
    
    seed : int, default=42
        The random state seed.

    Returns
    -------
    nx.Graph
        The KNN graph. 
    """
    
    n = len(sequences)
    if n == 0:
        return nx.Graph()

    X = search_features
    L = X.shape[1]

    dists, inds = _find_knn_balltree(
        X,
        k=k,
        tiebuffer=tiebuffer,
        include_self=include_self,
        metric="hamming" if distance_geometry == "hamming" else "euclidean",
    )

    rng = np.random.default_rng(seed)
    rows, cols, vals = [], [], []

    for i in range(n):
        ids = inds[i]
        # sklearn reports a Hamming fraction and an ordinary Euclidean length.
        ds = dists[i] * L if distance_geometry == "hamming" else dists[i]

        # drop self
        keep = (ids != i)
        ids = ids[keep]
        ds  = ds[keep]

        if ids.size == 0:
            continue

        # stable sort by distance
        order = np.argsort(ds, kind='stable')
        ids, ds = ids[order], ds[order]

        if ids.size <= k:
            take = np.arange(ids.size)
        else:
            dk = ds[k-1]
            cand = np.nonzero(ds <= dk + 1e-9)[0]  # include all ties at kth distance
            if tie_policy == 'all':
                take = cand
            elif tie_policy == 'min_index':
                take = cand[:k]
            elif tie_policy == 'random':
                take = rng.choice(cand, size=min(k, cand.size), replace=False)
            else:
                raise ValueError(f"Unknown tie_policy: {tie_policy}")

        ids = ids[take]; ds = ds[take]
        rows.append(np.full(ids.size, i, dtype=np.int32))
        cols.append(ids.astype(np.int32))
        vals.append(ds.astype(np.float32))

    I = np.concatenate(rows) if rows else np.array([], dtype=np.int32)
    J = np.concatenate(cols) if cols else np.array([], dtype=np.int32)
    V = np.concatenate(vals) if vals else np.array([], dtype=np.float32)

    # directed k-NN : symmetrize by UNION so degree >= k (and “all” can exceed)
    M = coo_matrix((V, (I, J)), shape=(n, n)).tocsr()
    U = M.maximum(M.T)

    G = nx.from_scipy_sparse_array(U, edge_attribute='distance')
    G.add_nodes_from(range(n))
    for i in range(n):
        G.nodes[i]['sequence'] = sequences[i]

    _attach_knn_edge_semantics(
        G,
        sequence_length=len(sequences[0]),
        constructor="knn-balltree",
        distance_geometry=distance_geometry,
    )

    return G

def _create_knn_graph_faiss(sequences: List[BaseNumpySequence],
                            k: int,
                            search_features: np.ndarray,
                            distance_geometry: Literal["hamming", "euclidean"],
                            *,
                            index_type: Literal['hnsw', 'flat', 'ivf'] = "hnsw",
                            metric: Literal['ip', 'l2'] = 'ip',
                            include_self: bool = False,
                            use_gpu: bool = False,
                            hnsw_M: int = 32,
                            tiebuffer : int = 128,
                            tie_policy: Literal['all', 'min_index', 'random'] = 'all',
                            seed: int = 42,
                            eps: float = 1e-12) -> nx.Graph:
    """
    Function to create an approximate nearest neighbour graph using
    FAISS indexing for efficient neighbour searching. 

    Parameters
    ----------
    sequences : List[BaseNumpySequence]
        The list of input sequences to connect. 

    k : int 
        The number of neighbours to connect each node with. 

    index_type : str, default=`hnsw`
        The faiss index type. Options are:
        - flat (exact) for small n. 
        - hnsw (approximate) for large n. 
        - ivf (approximate) for very large n.
    
    metric : str, default=`ip`
        The faiss metric to use specificall for hnsw. Options are:
        - `ip` : Inner product
        - `l2` : L2 norm 
    
    include_self : bool, default=`False`
        Boolean to include self in the neighbor list.
    
    use_gpu : bool, default=`False`
        Boolean to use FAISS GPU acceleration (application only to the
        flat index).
    
    hsnw_M : int, default = 32
        The hnsw dimesnion.
    
    tiebuffer : int, default=128
        The number of hits kept in buffer to eliminate ties.
    
    tie_policy : str, default=`all`
        The tie policy for when there are more than k equidistant
        neighbors found. Options are:
        - `all` : all neighbors are kept and the graph becomes
        irregular.
        - `min_index` : The "first" connection is kept, the others are
        arbitrarily removed. The graph remains k regular. 
        - `random` : equidistant edges are kept at random.
    
    seed : int, default=42
        The random state seed.

    Returns
    -------
    G : nx.Graph
        The constructed nearest neighbor graph. 
    """
    
    X = search_features
    n, d = X.shape
    L = len(sequences[0])

    D, I = _find_knn_faiss(X, k=k, index_type=index_type, metric=metric,
                           use_gpu=use_gpu, hnsw_M=hnsw_M,
                           include_self=include_self, tiebuffer=tiebuffer)

    # FAISS reports similarities for IP and squared distances for L2.
    if distance_geometry == "euclidean":
        distances_all = np.sqrt(np.maximum(D, 0.0)).astype(np.float32)
    elif metric == "ip":
        distances_all = (L - D).astype(np.float32)
    else:
        distances_all = (0.5 * D).astype(np.float32)
    del X, D

    # Build graph and keep the minimum distance per undirected edge.
    G = nx.Graph()
    for i, s in enumerate(sequences):
        G.add_node(i, sequence=s)

    min_distance = {}

    rng = np.random.default_rng(seed)
    for i in range(n):
        ids = I[i]
        ds  = distances_all[i]

        # filter invalids (-1) and optionally self
        valid = ids >= 0
        if not include_self:
            valid &= (ids != i)
        ids = ids[valid]
        ds  = ds[valid]

        if ids.size == 0:
            continue

        # stable sort by distance
        order = np.argsort(ds, kind="stable")
        ids, ds = ids[order], ds[order]

        # choose top-k with tie handling
        if ids.size <= k:
            take = np.arange(ids.size)
        else:
            dk = ds[k-1]
            cand = np.nonzero(ds <= dk)[0]
            if tie_policy == "min_index":
                take = cand[:k]
            elif tie_policy == "all":
                take = cand
            elif tie_policy == "random":
                take = rng.choice(cand, size=min(k, cand.size), replace=False)
            else:
                raise ValueError(f"Unknown tie_policy: {tie_policy}")

        sel_ids = ids[take].astype(int)
        sel_ds  = ds[take].astype(float)

        for j, dij in zip(sel_ids, sel_ds):
            if i == j:
                continue
            u, v = (i, j) if i < j else (j, i)
            prev = min_distance.get((u, v))
            if prev is None or dij < prev:
                min_distance[(u, v)] = dij
                G.add_edge(u, v)

    del I, distances_all

    if min_distance:
        nx.set_edge_attributes(G, min_distance, "distance")
    _attach_knn_edge_semantics(
        G,
        sequence_length=L,
        constructor=f"knn-faiss-{index_type}-{metric}",
        distance_geometry=distance_geometry,
    )
    return G

def create_knn_graph(sequences: List[BaseNumpySequence],
                     k: int,
                     *,
                     embeddings: np.ndarray | None = None,
                     embedding_domain: Literal["plm", "ohe", "composition"] | None = None,
                     backend: Literal['auto', 'faiss', 'balltree'] = 'auto',
                     index_type: Literal['hnsw', 'flat', 'ivf'] = 'hnsw',
                     faiss_metric: Literal['ip', 'l2'] = 'ip',
                     include_self: bool = False,
                     use_gpu: bool = False,
                     hnsw_M: int = 32,
                     tiebuffer: int = 128,
                     tie_policy: Literal['all', 'min_index', 'random'] = 'all',
                     seed : int = None,
                     _compute_hamming_edges: bool = False) -> nx.Graph:
    """
    Function to create a k-nearest neighbor network graph from
    sequences, using an efficient backend algorithm. 

    Parameters
    ----------
    sequences : List[BaseNumpySequence]
        The sequences to construct the graph from. 
    
    k : int
        Positive number of non-self neighbours to connect. Values greater
        than the available population are capped at ``n - 1``.

    embeddings : np.ndarray, optional
        Finite feature matrix aligned to ``sequences``. When supplied for a
        PLM, composition, or unspecified embedding domain, kNN is constructed
        from ordinary Euclidean distances in this matrix. When omitted, or
        when ``embedding_domain='ohe'``, sequence Hamming geometry is used.

    embedding_domain : {'plm', 'ohe', 'composition'}, optional
        Scientific domain of ``embeddings``. PLM and composition domains
        require ``embeddings`` and force Euclidean/L2 neighbour search.
    
    backend : str, default=`auto`
        The computational backend to use. Options are:
        -`faiss` : use a FAISS-based (sublinear) scalling (but
        approximate) backend to find neighbors. 
        - `balltree` : use the BallTree exact solver, which scales
        poorly with large dimension size. 
        - `auto` : Automatic backend based on dataset size.
    
    index_type : str, default=`hnsw`
        The FAISS indexing algorithm to use. Options are:
        - `hnsw` : hierarchical navigatible small worlds. Effective on
        large n. Approximate.
        - `flat` : Exact flat indexing.
        - `ivf` : inverted file indexing algorithm. Effective on very
        large n. Approximate.
    
    faiss_metric : str, default=`ip`
        The faiss metric. Options are:
        - `ip` : the inner produt.
        - `l2` : the L2 norm. 
        This option applies only to sequence/OHE Hamming searches. Continuous
        embedding searches always use FAISS L2 to preserve Euclidean geometry.

    include_self : bool, default=False
        Whether FAISS candidate queries include each sequence itself. Self
        edges are not added to the final undirected graph.
    
    use_gpu : bool, default=`False`
        Boolean to use GPU on flat indexing. 
    
    hnsw_M : int, default=32
        The hnsw dimension size.
    
    tiebuffer : int, default=128
        Non-negative number of additional candidates retained for ties.
    
    tie_policy : str, default=`all`
        The tie policy for when there are more than k equidistant
        neighbors found. Options are:
        - `all` : all neighbors are kept and the graph becomes
        irregular.
        - `min_index` : The "first" connection is kept, the others are
        arbitrarily removed. The graph remains k regular. 
        - `random` : equidistant edges are kept at random.
    
    seed : int, default=42
        The random state seed. 

    _compute_hamming_edges : bool, default=False
        Reserved compatibility flag. Release builds disable the optional
        post-construction edge mutation pass.

    Returns
    -------
    nx.Graph    
        The constructed KNN graph.
    """
    _compute_hamming_edges = _force_disable_hamming_edge_computation(_compute_hamming_edges)

    use_embedding_geometry = embedding_domain in _EMBEDDING_KNN_DOMAINS or (
        embeddings is not None and embedding_domain != "ohe"
    )
    n, sequence_length = _validate_sequence_collection(
        sequences,
        require_aligned=not use_embedding_geometry,
    )
    k, backend = _validate_neighbour_configuration(
        n_sequences=n,
        k=k,
        tiebuffer=tiebuffer,
        backend=backend,
        index_type=index_type,
        faiss_metric=faiss_metric,
        include_self=include_self,
        use_gpu=use_gpu,
        hnsw_M=hnsw_M,
        tie_policy=tie_policy,
    )
    if seed is not None and (
        isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
    ):
        raise TypeError("`seed` must be an integer or None.")

    search_features, distance_geometry, search_metric, search_domain = (
        _prepare_knn_search_space(
            sequences,
            embeddings,
            embedding_domain=embedding_domain,
            backend=backend,
            faiss_metric=faiss_metric,
        )
    )

    if n <= 1:
        graph = nx.Graph()
        for index, sequence in enumerate(sequences):
            graph.add_node(index, sequence=sequence)
        _attach_knn_edge_semantics(
            graph,
            sequence_length=sequence_length,
            constructor=f"knn-{backend}",
            distance_geometry=distance_geometry,
        )
        _attach_knn_search_metadata(
            graph,
            backend=backend,
            metric=search_metric,
            distance_geometry=distance_geometry,
            embedding_domain=search_domain,
            role="graph",
        )
        return graph

    if backend == 'faiss':
        G = _create_knn_graph_faiss(
            sequences, k, search_features, distance_geometry,
            index_type=index_type,
            metric=search_metric,
            include_self=include_self,
            use_gpu=use_gpu,
            hnsw_M=hnsw_M,
            tiebuffer=tiebuffer,
            tie_policy=tie_policy,
            seed=seed
        )
    elif backend == 'balltree':
        G = _create_knn_graph_balltree(sequences,
                                          k,
                                          search_features,
                                          distance_geometry,
                                          include_self=include_self,
                                          tie_policy=tie_policy,
                                          tiebuffer=tiebuffer,
                                          seed=seed)
    else:
        raise ValueError(f"Unsupported backend {backend!r}. Expected `auto`, `faiss`, or `balltree`.")
    _attach_knn_search_metadata(
        G,
        backend=backend,
        metric=search_metric,
        distance_geometry=distance_geometry,
        embedding_domain=search_domain,
        role="graph",
    )
    # Optionally compute expected Hamming distances if available
    if _compute_hamming_edges and all(
        hasattr(seq, "ungapped_arr") and getattr(seq, "ungapped_arr", None) is not None
        and getattr(seq, "alphabet", None) == PROT_20
        for seq in sequences
    ):
        _annotate_existing_edges_hamming(G)
    return G

def create_tda_graph(sequences: List[BaseNumpySequence],
                     embeddings: np.ndarray,
                     n_components: int = 3,
                     reweight_simplex_edges: bool = False,
                     **kwargs) -> nx.Graph:
    """
    Function to construct a graph based on persisent homology, using
    the alpha complex and dimensionality reduced embedding features.

    Parameters
    ----------
    sequences : List[BaseNumpySequences]
        Sequences to connect.
    
    embeddings : np.ndarray
        Finite two-dimensional sequence embeddings indexed according to
        sequence order.

    n_components : int, default=3
        Positive requested number of principal components. The effective
        value is clipped to the sample count, feature count, and centered
        geometric rank.
    
    reweight_simplex_edges : bool, default=`False`
        Bool to reweight graph edges by triangle simplexes.

    **kwargs
        Reserved for compatibility. No keyword is currently consumed.
    
    Returns
    -------
    G : nx.graph
        The constructed graph with `BaseNumpySequence` features stored
        under `sequence`.
    """
    n_sequences, _ = _validate_sequence_collection(sequences)
    embeddings = _validate_embedding_matrix(
        embeddings,
        n_sequences=n_sequences,
    )
    requested_components = _validate_integer(
        n_components,
        name="n_components",
        minimum=1,
    )
    reweight_simplex_edges = _validate_boolean(
        reweight_simplex_edges,
        name="reweight_simplex_edges",
    )

    G = nx.Graph()
    for index, sequence in enumerate(sequences):
        G.add_node(index, sequence=sequence)
    _declare_tda_graph_semantics(G)
    G.graph["tda_requested_components"] = requested_components
    G.graph["tda_duplicate_policy"] = "reject"

    if n_sequences == 0:
        G.graph["tda_effective_components"] = 0
        return G
    if n_sequences == 1:
        G.graph["tda_effective_components"] = 0
        return G

    if np.unique(embeddings, axis=0).shape[0] != n_sequences:
        raise ValueError(
            "`embeddings` contains duplicate points; TDA construction requires "
            "one distinct point per sequence."
        )

    centered_rank = int(np.linalg.matrix_rank(embeddings - embeddings.mean(axis=0)))
    if centered_rank < 1:
        raise ValueError("`embeddings` is geometrically degenerate after centering.")
    effective_components = min(
        requested_components,
        n_sequences - 1,
        embeddings.shape[1],
        centered_rank,
    )
    G.graph["tda_effective_components"] = effective_components

    gudhi = require_optional(
        "gudhi",
        extra="tda",
        purpose="topological graph construction",
    )
    sklearn_decomposition = require_optional(
        "sklearn.decomposition",
        extra="tda",
        purpose="topological graph construction",
    )
    # Reduce dimensionality with PCA.
    # Alpha complex scales with dimension.
    pca = sklearn_decomposition.PCA(n_components=effective_components)
    low_dim_data = pca.fit_transform(embeddings)
    alpha_complex = gudhi.AlphaComplex(points=low_dim_data)
    simplex_tree = alpha_complex.create_simplex_tree()
    persistence_0d = simplex_tree.persistence(homology_coeff_field=2, min_persistence=0)
    
    # Get all finite death times for 0D features (connected components)
    finite_deaths = [p[1][1] for p in persistence_0d if p[0] == 0 and p[1][1] < float('inf')]
    
    if not finite_deaths:
        # If all points are isolated or form one component, use a small default
        chosen_alpha_square = 0.01 
    else:
        # Choose the 95th percentile of death times as a robust threshold
        chosen_alpha_square = np.percentile(finite_deaths, 95)

    alpha_complex_for_graph = gudhi.AlphaComplex(points=low_dim_data)
    simplex_tree_for_graph = alpha_complex_for_graph.create_simplex_tree(max_alpha_square=chosen_alpha_square)
    edge_generator = simplex_tree_for_graph.get_skeleton(1)

    for simplex, _ in edge_generator:
        if len(simplex) == 2:
            node1, node2 = simplex[0], simplex[1]
            dist = np.linalg.norm(low_dim_data[node1] - low_dim_data[node2])
            affinity = 1.0 / (1.0 + float(dist))
            G.add_edge(
                node1,
                node2,
                distance=float(dist),
                affinity=affinity,
                weight=affinity,
                tda_distance=float(dist),
            )

    _declare_tda_graph_semantics(G)
            
    if reweight_simplex_edges:
        G = _reweight_graph_by_simplices(G=G,
                                         simplex_tree=simplex_tree)

    
    # Attach edge attributes.    
    # if all(hasattr(seq, "ungapped_arr") and seq.alphabet==PROT_20 for seq in sequences):
    #     compute_edge_mutations_star(G=G)
    return G

def _reweight_graph_by_simplices(G: nx.Graph,
                                 simplex_tree) -> nx.Graph:
    """
    Helper function to reweight the edges of a graph based on how many
    triangles are present in the TDA.

    Parameters
    ----------
    G : nx.Graph
        The constructed network graph to reweight.
    
    simplex_tree : Any
        The 0d persistence simplex tree used to construct `G`.
    
    Returns
    -------
    G : nx.Graph
        The input network graph with updated simplex edge weights.
    """
    G_weighted = G.copy()
    
    # A dictionary to count triangle participation for each edge
    triangle_counts = {}
    
    # Iterate through all triangles in the simplex tree
    for simplex, _ in simplex_tree.get_skeleton(2):
        if len(simplex) == 3:
            # For each edge in the triangle, increment its count
            for i in range(3):
                u, v = simplex[i], simplex[(i + 1) % 3]
                # Ensure the edge is stored in a canonical order (u < v)
                edge = tuple(sorted((u, v)))
                triangle_counts[edge] = triangle_counts.get(edge, 0) + 1
    
    # Update the weights in the new graph
    for u, v in G_weighted.edges():
        edge = tuple(sorted((u, v)))
        G_weighted[u][v]['simplicial_weight'] = 1 + triangle_counts.get(edge, 0)
        
    return G_weighted


def create_diffusion_emb_graph(sequences: List[BaseNumpySequence],
                               embeddings: np.ndarray = None,
                               k: int = 128,
                               tiebuffer: int = 0,
                               backend: Literal['auto', 'faiss', 'balltree'] = 'auto',
                               index_type: Literal['hnsw', 'flat', 'ivf'] = 'hnsw',
                               faiss_metric: Literal['ip', 'l2'] = 'ip',
                               include_self: bool = False,
                               use_gpu: bool = False,
                               hnsw_M: int = 32,
                               t: Optional[Union[int, float]] = 5,
                               connectivity_threshold: float = 1e-4,
                               max_diffusion_nnz: int = _DEFAULT_MAX_DIFFUSION_NNZ,
                               max_diffusion_work: int = _DEFAULT_MAX_DIFFUSION_WORK,
                               *,
                               embedding_domain: Literal["plm", "ohe", "composition"] | None = None,
                               _compute_hamming_edges: bool = False,
                               **kwargs) -> nx.Graph:
    """
    Construct a reversible undirected diffusion graph in embedding space.

    RBF affinities on the symmetric union of kNN candidates define a sparse
    lazy detailed-balance transition. Edge weights are the exact symmetric
    stationary-measure kernel
    ``Pi^(1/2) P^t Pi^(-1/2)`` after thresholding.

    Parameters
    ----------
    sequences : List[BaseNumpySequences]
        Sequences to connect.
    
    embeddings : np.ndarray
        Finite two-dimensional sequence embeddings indexed according to
        sequence order.

    k : int, default=128
        Non-self neighbours defining each directed candidate set. Their
        undirected union supports the sparse RBF affinity and also sets its
        global bandwidth.

    tiebuffer : int, default=0
        Additional backend hits inspected for candidates tied at the exact kth
        distance. Non-tied buffered candidates do not enter the affinity.

    backend : str, default=`auto`
        The computational backend to use. Options are:
        -`faiss` : use a FAISS-based (sublinear) scalling (but
        approximate) backend to find neighbors. 
        - `balltree` : use the BallTree exact solver, which scales
        poorly with large dimension size. 
        - `auto` : Automatic backend based on dataset size.
    
    index_type : str, default=`hnsw`
        The FAISS indexing algorithm to use. Options are:
        - `hnsw` : hierarchical navigatible small worlds. Effective on
        large n. Approximate.
        - `flat` : Exact flat indexing.
        - `ivf` : inverted file indexing algorithm. Effective on very
        large n. Approximate.
    
    faiss_metric : str, default=`ip`
        The faiss metric. Options are:
        - `ip` : the inner produt.
        - `l2` : the L2 norm. 
        Applies to sequence/OHE searches. Continuous embedding prefilters
        always use L2 to preserve Euclidean geometry.

    include_self : bool, default=False
        Whether FAISS candidate queries may include the query point. The final
        diffusion graph has no self edges.
    
    use_gpu : bool, default=`False`
        Boolean to use GPU on flat indexing. 
    
    hnsw_M : int, default=32
        The hnsw dimension size.
    
    t : int | float | None, default=`5`
        Diffusion power for the Markov transition matrix. When ``None``,
        ``0`` or ``np.inf``, use the componentwise stationary pair kernel
        instead of an explicit matrix power.

    connectivity_threshold : float, default=`1e-04`
        Finite dimensionless diffusion-amplitude threshold in ``[0, 1]`` used
        to define discrete connectivity.

    max_diffusion_nnz : int, default=50000000
        Maximum nonzeros allowed in any exact sparse affinity, transition, or
        diffusion-power matrix. The constructor raises ``MemoryError`` before
        a multiplication whose exact structural result exceeds this budget.

    max_diffusion_work : int, default=1000000000
        Maximum cumulative scalar products allowed for the exact sparse matrix
        power. This is a work guard, not a numerical approximation control.

    embedding_domain : {'plm', 'ohe', 'composition'}, optional
        Domain governing the sparse kNN prefilter. PLM and composition use
        Euclidean/L2 search in ``embeddings``; OHE uses sequence Hamming
        geometry. When omitted, supplied embeddings are treated as continuous.

    _compute_hamming_edges : bool, default=False
        Reserved compatibility flag. Release builds disable the optional
        post-construction mutation pass.

    **kwargs
        Reserved for compatibility. No keyword is currently consumed.

    Returns
    -------
    G : nx.graph
        The constructed graph with `BaseNumpySequence` features stored
        under `sequence`.
    """
    _compute_hamming_edges = _force_disable_hamming_edge_computation(_compute_hamming_edges)
    use_embedding_geometry = embedding_domain in _EMBEDDING_KNN_DOMAINS or (
        embeddings is not None and embedding_domain != "ohe"
    )
    n_points, _ = _validate_sequence_collection(
        sequences,
        require_aligned=not use_embedding_geometry,
    )
    requested_k = k
    k, backend = _validate_neighbour_configuration(
        n_sequences=n_points,
        k=k,
        tiebuffer=tiebuffer,
        backend=backend,
        index_type=index_type,
        faiss_metric=faiss_metric,
        include_self=include_self,
        use_gpu=use_gpu,
        hnsw_M=hnsw_M,
    )
    use_stationary, t_int = _validate_diffusion_power(t)
    threshold = _validate_connectivity_threshold(connectivity_threshold)
    max_diffusion_nnz = _validate_diffusion_budget(
        max_diffusion_nnz,
        name="max_diffusion_nnz",
    )
    max_diffusion_work = _validate_diffusion_budget(
        max_diffusion_work,
        name="max_diffusion_work",
    )

    search_features, search_geometry, search_metric, search_domain = (
        _prepare_knn_search_space(
            sequences,
            embeddings,
            embedding_domain=embedding_domain,
            backend=backend,
            faiss_metric=faiss_metric,
        )
    )

    if embeddings is None:
        if n_points:
            embeddings, _ = _encode_multiallele(sequences)
        else:
            embeddings = None
    if embeddings is not None:
        embeddings = _validate_embedding_matrix(
            embeddings,
            n_sequences=n_points,
        )

    G = nx.Graph()
    for index, sequence in enumerate(sequences):
        G.add_node(index, sequence=sequence)
    _declare_diffusion_graph_semantics(G, constructor="embedding-diffusion")
    _attach_diffusion_metadata(
        G,
        use_stationary=use_stationary,
        power=t_int,
        threshold=threshold,
    )
    _attach_knn_search_metadata(
        G,
        backend=backend,
        metric=search_metric,
        distance_geometry=search_geometry,
        embedding_domain=search_domain,
        role="prefilter",
    )
    if n_points <= 1:
        _attach_sparse_diffusion_construction_metadata(
            G,
            requested_k=int(requested_k),
            effective_k=k,
            tiebuffer=tiebuffer,
            backend=backend,
            index_type=index_type,
            max_nnz=max_diffusion_nnz,
            max_work=max_diffusion_work,
            affinity_nnz=0,
            transition_nnz=n_points,
            kernel_nnz=n_points,
            directed_candidates=0,
            diffusion_work=0,
        )
        return G

    k_for_scale = k
    if int(requested_k) >= n_points and k_for_scale == n_points - 1:
        # When k overshoots a small dataset, avoid using the farthest neighbour
        # to set the RBF bandwidth; otherwise distant points flatten the kernel
        # and collapse clustered structure (e.g., fully connected graphs).
        k_for_scale = max(1, int(np.sqrt(n_points)))
    
    # Use balltree algorithm (will fail as shape of embeddings >>>)
    # The backend's ``include_self=True`` query capacity counts the query point
    # itself. Ask for one additional hit so public ``k`` continues to denote
    # non-self candidates in either mode.
    backend_query_k = k + int(include_self)
    if backend == 'balltree':
        _, neighbour_indices = _find_knn_balltree(
            search_features,
            backend_query_k,
            tiebuffer,
            include_self=include_self,
            metric=search_metric,
        )
    
    # Use FAISS algorithm (approx or exact).
    elif backend == 'faiss':
        _, neighbour_indices = _find_knn_faiss(search_features,
                                       backend_query_k,
                                       index_type=index_type,
                                       metric=search_metric,
                                       use_gpu=use_gpu,
                                       hnsw_M=hnsw_M,
                                       include_self=include_self,
                                       tiebuffer=tiebuffer) 

    candidates_by_row = _select_diffusion_knn_candidates(
        search_features,
        neighbour_indices,
        k=k,
        distance_geometry=search_geometry,
    )

    # Evaluate scale in a common Euclidean unit after the requested backend
    # has selected and ordered candidates. This avoids interpreting FAISS
    # inner-product scores or squared-L2 scores as ordinary distances.
    sigma = np.zeros(n_points, dtype=np.float64)
    for row in range(n_points):
        candidates = candidates_by_row[row]
        if candidates.size:
            distances = np.linalg.norm(embeddings[candidates] - embeddings[row], axis=1)
            distances.sort()
            scale_index = min(k_for_scale, distances.size) - 1
            sigma[row] = float(distances[scale_index])
    pos = sigma[np.isfinite(sigma) & (sigma > 0)]

    if pos.size == 0:
        median_sigma_sq = 1.0
    else:
        median_sigma_sq = float(np.median(pos))**2
        if not np.isfinite(median_sigma_sq) or median_sigma_sq <= 0:
            median_sigma_sq = 1.0

    gamma = 1.0 / (2 * median_sigma_sq)
    affinity_matrix, directed_candidates = _sparse_rbf_affinity_from_candidates(
        embeddings,
        candidates_by_row,
        gamma=gamma,
        max_nnz=max_diffusion_nnz,
    )
    
    transition_matrix, stationary, component_labels = _reversible_lazy_transition(
        affinity_matrix
    )
    diffusion_diagnostics: dict[str, int] = {}
    diffusion_kernel = _reversible_diffusion_kernel(
        transition_matrix,
        stationary,
        component_labels,
        stationary_limit=use_stationary,
        power=t_int,
        max_nnz=max_diffusion_nnz,
        max_work=max_diffusion_work,
        _diagnostics=diffusion_diagnostics,
    )
    rows, cols, edge_weights = _threshold_undirected_kernel(
        diffusion_kernel,
        threshold=threshold,
    )

    for i, j, value in zip(rows, cols, edge_weights):
        affinity = float(value)
        G.add_edge(
            int(i),
            int(j),
            affinity=affinity,
            weight=affinity,
            kernel_weight=affinity,
        )

    _declare_diffusion_graph_semantics(G, constructor="embedding-diffusion")
    _attach_diffusion_metadata(
        G,
        use_stationary=use_stationary,
        power=t_int,
        threshold=threshold,
    )
    _attach_sparse_diffusion_construction_metadata(
        G,
        requested_k=int(requested_k),
        effective_k=k,
        tiebuffer=tiebuffer,
        backend=backend,
        index_type=index_type,
        max_nnz=max_diffusion_nnz,
        max_work=max_diffusion_work,
        affinity_nnz=affinity_matrix.nnz,
        transition_nnz=transition_matrix.nnz,
        kernel_nnz=diffusion_kernel.nnz,
        directed_candidates=directed_candidates,
        diffusion_work=diffusion_diagnostics["estimated_scalar_products"],
    )
    
    # Optionally compute expected Hamming distances if available
    if _compute_hamming_edges and all(
        hasattr(seq, "ungapped_arr") and getattr(seq, "ungapped_arr", None) is not None
        and getattr(seq, "alphabet", None) == PROT_20
        for seq in sequences
    ):
        _annotate_existing_edges_hamming(G)
    return G

def create_phylo_graph(sequences: Union[Path, Alignment],
                       replacement_matrix: List[str] = ['LG'],
                       model_fitting: bool = True,
                       _log_progress: bool = False,
                       _nested_parallel: bool = False,
                       phylo_backend: str = 'cogent_nj',
                       _dist_calc: str = 'pdist',
                       reconstruct_ancestral_states: bool = True,
                       *,
                       _compute_hamming_edges: bool = False,
                       _lightweight_nodes: bool = False,
                       _hard_ancestors: bool = False,
                       **kwargs) -> nx.Graph:
    """Construct an undirected phylogenetic landscape topology.

    Parameters
    ----------
    sequences : pathlib.Path or cogent3.Alignment
        Alignment of extant sequences used for tree inference and ancestral
        reconstruction.
    replacement_matrix : list of str, default=['LG']
        Candidate amino-acid substitution models.
    model_fitting : bool, default=True
        Fit the maximum-likelihood model selected from ``replacement_matrix``.
    _log_progress : bool, default=False
        Emit progress logging for phylogenetic and edge computations.
    _nested_parallel : bool, default=False
        Permit nested parallelism in optional edge calculations.
    phylo_backend : str, default='cogent_nj'
        Tree-inference backend accepted by :class:`ASRConstructor`.
    _dist_calc : str, default='pdist'
        Pairwise-distance calculator used by compatible tree backends.
    reconstruct_ancestral_states : bool, default=True
        Reconstruct ancestral amino-acid states. If false,
        internal nodes are populated with placeholder sequences so that
        the phylogenetic topology can still be analysed.
    _compute_hamming_edges : bool, default=False
        Compute optional mutation attributes after topology construction.
        The annotation pass operates only on existing edges.
    _lightweight_nodes : bool, default=False
        Remove stored gapped arrays from graph nodes.
    _hard_ancestors : bool, default=False
        Replace probabilistic ancestral sequences with their hard calls.
    **kwargs
        Reserved for compatibility. No keyword is currently consumed.

    Returns
    -------
    networkx.Graph
        Undirected tree topology with sequence and node-role annotations.
    """
    _compute_hamming_edges = _force_disable_hamming_edge_computation(_compute_hamming_edges)

    require_optional(
        "cogent3",
        extra="phylogeny",
        purpose="phylogenetic graph construction",
    )
    from ..phylo.phylogenetic_asr import ASRConstructor

    constructor = ASRConstructor(sequences,
                                  replacement_matrix = replacement_matrix,
                                  model_fitting = model_fitting,
                                  phylo_backend=phylo_backend,
                                  _dist_calc=_dist_calc,
                                  reconstruct_ancestral_states=reconstruct_ancestral_states,
                                  _log_progress=_log_progress)
    
    graph = constructor.construct_topology()

    # Optionally strip heavy arrays and collapse ancestors to hard sequences
    if _lightweight_nodes or _hard_ancestors:
        from .sequence import SoftSequence, BaseNumpySequence
        for node, data in list(graph.nodes(data=True)):
            if _lightweight_nodes:
                data.pop('gapped_arr', None)
            if _hard_ancestors and isinstance(data.get('sequence'), SoftSequence):
                hard_str = ''.join(map(str, data['sequence'].to_array()))
                data['sequence'] = BaseNumpySequence.from_string(hard_str, alphabet=PROT_20, moltype='protein', sequence_id=str(node))
    
    # Attach edge attributes (serial by default to avoid nested Ray OOM)
    if _compute_hamming_edges:
        _annotate_existing_edges_hamming(graph)

    role_records = {
        node: {"node_role": "extant" if node in constructor.tip_names else "ancestral"}
        for node in graph.nodes()
    }
    register_auto_annotation(
        graph,
        "node_role",
        role_records,
        metadata={"description": "Phylogenetic node roles (ancestral vs extant)."},
    )
    declare_edge_semantics(
        graph,
        constructor="phylogeny",
        distance_key="branch_length",
        distance_units="expected_substitutions_per_site",
        normalized_distance_key=(
            "normalized_distance" if _compute_hamming_edges else None
        ),
        affinity_key=None,
        conductance_key=None,
        legacy_aliases={"sim": "hamming_affinity"},
        notes=(
            "Phylogenetic branch length is a distance, not conductance. "
            "Weighted Laplacian analyses require an explicit derived conductance."
        ),
    )
    return graph

def _stationary_frequencies_from_rate_matrix(rate_matrix: np.ndarray) -> np.ndarray:
    """Return the normalized stationary frequencies of a rate generator."""
    Q = np.asarray(rate_matrix, dtype=np.float64)
    if Q.shape != (len(PROT_20), len(PROT_20)):
        raise ValueError(
            "`replacement_matrix` must be a square rate matrix matching PROT_20."
        )
    if not np.all(np.isfinite(Q)):
        raise ValueError("`replacement_matrix` must contain only finite values.")
    if not np.allclose(Q.sum(axis=1), 0.0, atol=1e-10, rtol=1e-8):
        raise ValueError("`replacement_matrix` must have rows that sum to zero.")
    if np.any(np.diag(Q) > 1e-12):
        raise ValueError("`replacement_matrix` must have non-positive diagonal entries.")
    off_diagonal = Q.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    if np.any(off_diagonal < -1e-12):
        raise ValueError("`replacement_matrix` must have non-negative off-diagonal entries.")

    # Solve pi @ Q = 0 subject to sum(pi) = 1. Replacing one redundant
    # stationarity equation avoids selecting an arbitrary eigenvector sign.
    system = Q.T.copy()
    system[-1, :] = 1.0
    rhs = np.zeros(Q.shape[0], dtype=np.float64)
    rhs[-1] = 1.0
    try:
        frequencies = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "`replacement_matrix` must define a unique stationary distribution."
        ) from exc

    if np.any(frequencies <= 0.0) or not np.all(np.isfinite(frequencies)):
        raise ValueError(
            "`replacement_matrix` must define strictly positive stationary frequencies."
        )
    frequencies /= frequencies.sum()
    return frequencies


def _evolutionary_log_odds_matrix(
    rate_matrix: np.ndarray,
    evolutionary_time: float,
    equilibrium_frequencies: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Convert a reversible rate generator into symmetric transition log-odds."""
    Q = np.asarray(rate_matrix, dtype=np.float64)
    inferred_frequencies = _stationary_frequencies_from_rate_matrix(Q)

    if equilibrium_frequencies is None:
        frequencies = inferred_frequencies
    else:
        frequencies = np.asarray(equilibrium_frequencies, dtype=np.float64)
        if frequencies.shape != (Q.shape[0],):
            raise ValueError(
                "`equilibrium_frequencies` must have one value per PROT_20 residue."
            )
        if np.any(frequencies <= 0.0) or not np.all(np.isfinite(frequencies)):
            raise ValueError(
                "`equilibrium_frequencies` must contain finite positive values."
            )
        frequencies = frequencies / frequencies.sum()
        if not np.allclose(frequencies @ Q, 0.0, atol=1e-10, rtol=1e-8):
            raise ValueError(
                "`equilibrium_frequencies` must be stationary for `replacement_matrix`."
            )

    if not np.isfinite(evolutionary_time) or evolutionary_time <= 0.0:
        raise ValueError("`evolutionary_time` must be finite and greater than zero.")

    equilibrium_flux = frequencies[:, None] * Q
    if not np.allclose(equilibrium_flux, equilibrium_flux.T, atol=1e-10, rtol=1e-8):
        raise ValueError(
            "`replacement_matrix` must be reversible for an undirected "
            "evolutionary-diffusion graph."
        )

    transition = expm(float(evolutionary_time) * Q)
    if np.any(transition < -1e-12) or not np.all(np.isfinite(transition)):
        raise ValueError("Failed to obtain a valid transition probability matrix.")
    transition = np.maximum(transition, np.finfo(np.float64).tiny)
    transition /= transition.sum(axis=1, keepdims=True)

    log_odds = np.log(transition) - np.log(frequencies)[None, :]
    # Detailed balance makes this symmetric analytically. Average the two
    # orientations to remove only floating-point asymmetry.
    return 0.5 * (log_odds + log_odds.T)


def _length_normalized_gapped_soft_score(
    aligned_seq1: np.ndarray,
    aligned_seq2: np.ndarray,
    score_matrix: np.ndarray,
) -> float:
    """Return a mean per-column score, including single-gap penalties."""
    total_score = calculate_gapped_soft_score(
        aligned_seq1=aligned_seq1,
        aligned_seq2=aligned_seq2,
        q=score_matrix,
    )
    gap_index = score_matrix.shape[0]
    p1_gap = np.asarray(aligned_seq1, dtype=np.float64)[:, gap_index]
    p2_gap = np.asarray(aligned_seq2, dtype=np.float64)[:, gap_index]
    effective_length = float(np.sum(1.0 - (p1_gap * p2_gap)))
    if not np.isfinite(effective_length) or effective_length <= 0.0:
        raise ValueError("Aligned sequences must contain at least one effective column.")
    return float(total_score / effective_length)


def _score_pair(i, j, seq_i, seq_j, score_matrix):
    soft_alignment = require_optional(
        "softalign.soft_alignment",
        extra="alignment",
        purpose="evolutionary sequence alignment",
    )
    Ai = seq_i.posterior if isinstance(seq_i, SoftSequence) else seq_i.to_one_hot()
    Aj = seq_j.posterior if isinstance(seq_j, SoftSequence) else seq_j.to_one_hot()
    # Ensure float inputs for stability in softalign
    Ai = np.ascontiguousarray(np.asarray(Ai, dtype=np.float64))
    Aj = np.ascontiguousarray(np.asarray(Aj, dtype=np.float64))
    _res = soft_alignment.align_soft_sequences(sequences=[Ai, Aj], alphabet=PROT_20)
    aligned = _res[0] if isinstance(_res, tuple) else _res
    score = _length_normalized_gapped_soft_score(
        aligned_seq1=aligned[0],
        aligned_seq2=aligned[1],
        score_matrix=score_matrix,
    )

    return i, j, score


def create_evol_diffusion_graph(sequences: List[BaseNumpySequence],
                                             embeddings: np.ndarray,
                                             replacement_matrix: np.ndarray | None = None,
                                             tiebuffer: int = 0,
                                             backend: Literal['auto', 'faiss', 'balltree'] = 'auto',
                                             index_type: Literal['hnsw', 'flat', 'ivf'] = 'hnsw',
                                             faiss_metric: Literal['ip', 'l2'] = 'ip',
                                             include_self: bool = False,
                                             use_gpu: bool = False,
                                             hnsw_M: int = 32,
                                             k: int = 50,
                                             t: Optional[Union[int, float]] = 5,
                                             tau: float = 1.0,
                                             connectivity_threshold: float = 1e-4,
                                             cpus: int = 1,
                                             *,
                                             embedding_domain: Literal["plm", "ohe", "composition"] | None = None,
                                             evolutionary_time: float = 1.0,
                                             equilibrium_frequencies: Optional[np.ndarray] = None,
                                             _compute_hamming_edges: bool = False,
                                             **kwargs) -> nx.Graph:
    """
    Construct a reversible evolutionary-diffusion graph from pairwise alignments.

    Candidate pairs are selected in embedding space, then aligned and scored
    in distributed Ray tasks. The supplied reversible rate generator is
    converted to transition log-odds at ``evolutionary_time``. Pair scores are
    averaged over effective alignment length and converted to transition
    affinities with a numerically stable global exponential shift. A lazy
    detailed-balance transition then defines the symmetric
    ``Pi^(1/2) P^t Pi^(-1/2)`` edge kernel.

    Parameters
    ----------
    sequences : List[BaseNumpySequence]
        The list of sequence in the landscape. 

    embeddings : np.ndarray
        Finite two-dimensional sequence embeddings indexed by the entry in
        ``sequences``.

    replacement_matrix : np.ndarray, optional
        Reversible instantaneous amino-acid rate generator in PROT_20 order.
        Defaults to the bundled LG matrix.

    tiebuffer : int, default=0
        Additional nearest-neighbour candidates retained for tie handling.

    backend : str, default=`auto`
        The computational backend to use. Options are:
        -`faiss` : use a FAISS-based (sublinear) scalling (but
        approximate) backend to find neighbors. 
        - `balltree` : use the BallTree exact solver, which scales
        poorly with large dimension size. 
        - `auto` : Automatic backend based on dataset size.
    
    index_type : str, default=`hnsw`
        The FAISS indexing algorithm to use. Options are:
        - `hnsw` : hierarchical navigatible small worlds. Effective on
        large n. Approximate.
        - `flat` : Exact flat indexing.
        - `ivf` : inverted file indexing algorithm. Effective on very
        large n. Approximate.
    
    faiss_metric : str, default=`ip`
        The faiss metric. Options are:
        - `ip` : the inner produt.
        - `l2` : the L2 norm. 
        Applies to sequence/OHE searches. Continuous embedding prefilters
        always use L2 to preserve Euclidean geometry.

    include_self : bool, default=False
        Whether the candidate-neighbour query includes each sequence itself.
        Self edges are removed from the final graph.
    
    use_gpu : bool, default=`False`
        Boolean to use GPU on flat indexing. 
    
    hnsw_M : int, default=32
        The hnsw dimension size.
    
    k : int, default=50
        The number of neighbours to use for kNN pre-filtering.
    
    t : int | float | None, default=5
        Diffusion power for the Markov transition matrix. When ``None``,
        ``0`` or ``np.inf``, use the componentwise stationary pair kernel
        instead of explicitly computing ``P^t``.
    
    tau : float, default=1.0
        Temperature applied to length-normalized evolutionary log-odds before
        the shared symmetric exponential affinity is constructed.

    connectivity_threshold : float, default=1e-4
        Finite threshold in ``[0, 1]``. Symmetric dimensionless diffusion
        amplitudes above this threshold are retained as edges.

    cpus : int, default=1
        Target number of worker CPUs for Ray alignment tasks. Each task
        consumes a single CPU.

    embedding_domain : {'plm', 'ohe', 'composition'}, optional
        Domain governing the sparse kNN prefilter. PLM and composition use
        Euclidean/L2 search in ``embeddings``; OHE uses sequence Hamming
        geometry. When omitted, supplied embeddings are treated as continuous.

    evolutionary_time : float, default=1.0
        Evolutionary time used to obtain transition probabilities from the
        instantaneous rate generator. This is distinct from the graph
        diffusion power ``t``.

    equilibrium_frequencies : np.ndarray, optional
        Stationary amino-acid frequencies in PROT_20 order. When omitted they
        are inferred from ``replacement_matrix``.

    _compute_hamming_edges : bool, default=False
        Reserved compatibility flag. Release builds disable the optional
        post-construction mutation pass.

    **kwargs
        Reserved for compatibility. No keyword is currently consumed.

    Returns
    -------
    nx.Graph
        The constructed graph.
    """
    _compute_hamming_edges = _force_disable_hamming_edge_computation(_compute_hamming_edges)
    use_embedding_geometry = embedding_domain in _EMBEDDING_KNN_DOMAINS or (
        embeddings is not None and embedding_domain != "ohe"
    )
    n_sequences, _ = _validate_sequence_collection(
        sequences,
        require_aligned=not use_embedding_geometry,
    )
    k, backend = _validate_neighbour_configuration(
        n_sequences=n_sequences,
        k=k,
        tiebuffer=tiebuffer,
        backend=backend,
        index_type=index_type,
        faiss_metric=faiss_metric,
        include_self=include_self,
        use_gpu=use_gpu,
        hnsw_M=hnsw_M,
    )
    use_stationary, t_int = _validate_diffusion_power(t)
    thr = _validate_connectivity_threshold(connectivity_threshold)
    num_cpus = _validate_integer(cpus, name="cpus", minimum=1)
    if isinstance(tau, (bool, np.bool_)):
        raise TypeError("`tau` must be a finite real number greater than zero.")
    try:
        tau = float(tau)
    except (TypeError, ValueError) as error:
        raise TypeError("`tau` must be a finite real number greater than zero.") from error
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("`tau` must be finite and greater than zero.")

    search_features, search_geometry, search_metric, search_domain = (
        _prepare_knn_search_space(
            sequences,
            embeddings,
            embedding_domain=embedding_domain,
            backend=backend,
            faiss_metric=faiss_metric,
        )
    )

    if embeddings is None:
        if n_sequences:
            embeddings, _ = _encode_multiallele(sequences)
        else:
            embeddings = None
    if embeddings is not None:
        embeddings = _validate_embedding_matrix(
            embeddings,
            n_sequences=n_sequences,
        )

    # Type check alphabet before optional backend or phylogenetic work.
    for seq in sequences:
        if seq.alphabet != PROT_20:
            raise ValueError("Sequence alphabet must be PROT_20 for all entries.")

    graph = nx.Graph()
    for index, sequence in enumerate(sequences):
        graph.add_node(index, sequence=sequence)
    _declare_diffusion_graph_semantics(
        graph,
        constructor="evolutionary-diffusion",
    )
    _attach_diffusion_metadata(
        graph,
        use_stationary=use_stationary,
        power=t_int,
        threshold=thr,
    )
    _attach_knn_search_metadata(
        graph,
        backend=backend,
        metric=search_metric,
        distance_geometry=search_geometry,
        embedding_domain=search_domain,
        role="prefilter",
    )
    if n_sequences <= 1:
        return graph

    if replacement_matrix is None:
        from ..phylo._sub_matrices import lg

        replacement_matrix = lg

    evolutionary_score_matrix = _evolutionary_log_odds_matrix(
        replacement_matrix,
        evolutionary_time=evolutionary_time,
        equilibrium_frequencies=equilibrium_frequencies,
    )
    _logger = logging.getLogger('fitness_landscape')
    _t_knn0 = time.perf_counter(); _c_knn0 = time.process_time()
    metric_used = None
    if backend == 'balltree':
        _, neighbor_indices = _find_knn_balltree(
            search_features,
            k,
            tiebuffer,
            include_self=include_self,
            metric=search_metric,
        )
        metric_used = search_metric
    
    # Use FAISS algorithm (approx or exact).
    elif backend == 'faiss':
        _, neighbor_indices = _find_knn_faiss(search_features,
                                       k,
                                       index_type=index_type,
                                       metric=search_metric,
                                       use_gpu=use_gpu,
                                       hnsw_M=hnsw_M,
                                       include_self=include_self,
                                       tiebuffer=tiebuffer) 
        metric_used = search_metric
    _logger.info('kNN prefilter done: backend=%s metric=%s k=%d n=%d wall=%.2fs cpu=%.2fs', backend, metric_used, k, n_sequences, time.perf_counter()-_t_knn0, time.process_time()-_c_knn0)

    pairs_to_align = set()
    for i in range(n_sequences):
        for j_idx in neighbor_indices[i]:
            if i != j_idx:
                # Add pairs in a canonical order to avoid duplicates
                pair = tuple(sorted((i, j_idx)))
                pairs_to_align.add(pair)
    _logger.info('Pairs to align: count=%d (~%.2fxN)', len(pairs_to_align), (len(pairs_to_align)/max(n_sequences,1)))

    # Build sparse kernel only on neighbor pairs to avoid dense NxN
    rows_list = []
    cols_list = []
    data_list = []

    # Use an existing Ray runtime without assuming ownership. A runtime started
    # here is always shut down when the alignment work finishes.
    _t_align0 = time.perf_counter(); _c_align0 = time.process_time()
    total_tasks = len(pairs_to_align)
    _logger.info('Submitted alignment tasks: %d', total_tasks)
    if total_tasks:
        with ray_runtime(num_cpus, purpose="parallel evolutionary sequence alignment") as ray:
            score_pair_remote = ray.remote(_score_pair)
            refs = [
                score_pair_remote.options(num_cpus=1).remote(
                    i,
                    j,
                    sequences[i],
                    sequences[j],
                    evolutionary_score_matrix,
                )
                for (i, j) in pairs_to_align
            ]
            pending = list(refs)
            completed = 0
            log_every = max(1, total_tasks // 20)
            while pending:
                num_returns = min(32, len(pending))
                ready, pending = ray.wait(pending, num_returns=num_returns)
                results = ray.get(ready)
                for i, j, score in results:
                    rows_list.append(i); cols_list.append(j); data_list.append(float(score))
                    rows_list.append(j); cols_list.append(i); data_list.append(float(score))
                completed += len(results)
                if completed == total_tasks or completed % log_every == 0:
                    _logger.info('Alignments progress: %d/%d (%.1f%%)', completed, total_tasks, (completed / total_tasks) * 100.0)
    _logger.info('Alignments complete: wall=%.2fs cpu=%.2fs', time.perf_counter()-_t_align0, time.process_time()-_c_align0)

    if rows_list:
        edge_scores = coo_matrix(
            (
                np.asarray(data_list, dtype=np.float64),
                (
                    np.asarray(rows_list, dtype=np.int32),
                    np.asarray(cols_list, dtype=np.int32),
                ),
            ),
            shape=(n_sequences, n_sequences),
        ).tocsr()
    else:
        edge_scores = csr_matrix((n_sequences, n_sequences), dtype=np.float64)

    # Convert symmetric length-normalized log-odds into a shared affinity
    # before row normalization. Row-dependent softmax shifts would obscure the
    # reversible affinity needed to establish detailed balance.
    _t_norm0 = time.perf_counter(); _c_norm0 = time.process_time()
    affinity_matrix = _symmetric_affinity_from_scores(edge_scores, tau=tau)
    transition_matrix, stationary, component_labels = _reversible_lazy_transition(
        affinity_matrix
    )
    diffusion_kernel = _reversible_diffusion_kernel(
        transition_matrix,
        stationary,
        component_labels,
        stationary_limit=use_stationary,
        power=t_int,
    )
    rows, cols, edge_weights = _threshold_undirected_kernel(
        diffusion_kernel,
        threshold=thr,
    )
    _logger.info(
        'Reversible diffusion kernel: nnz=%d edges=%d wall=%.2fs cpu=%.2fs',
        diffusion_kernel.nnz if sparse.issparse(diffusion_kernel) else np.count_nonzero(diffusion_kernel),
        edge_weights.size,
        time.perf_counter()-_t_norm0,
        time.process_time()-_c_norm0,
    )

    _t_graph0 = time.perf_counter(); _c_graph0 = time.process_time()
    # add edges with attribute
    if len(edge_weights):
        for i, j, w in zip(rows, cols, edge_weights):
            affinity = float(w)
            graph.add_edge(
                int(i),
                int(j),
                affinity=affinity,
                weight=affinity,
                kernel_weight=affinity,
            )

    _declare_diffusion_graph_semantics(
        graph,
        constructor="evolutionary-diffusion",
    )
    _attach_diffusion_metadata(
        graph,
        use_stationary=use_stationary,
        power=t_int,
        threshold=thr,
    )

    # Optionally compute expected Hamming distances if available
    _logger.info('Graph nodes/edges added: nodes=%d edges=%d wall=%.2fs cpu=%.2fs', graph.number_of_nodes(), graph.number_of_edges(), time.perf_counter()-_t_graph0, time.process_time()-_c_graph0)
    _t_ham0 = time.perf_counter(); _c_ham0 = time.process_time()
    if _compute_hamming_edges and all(
        hasattr(seq, "ungapped_arr") and getattr(seq, "ungapped_arr", None) is not None
        and getattr(seq, "alphabet", None) == PROT_20
        for seq in sequences
    ):
        _annotate_existing_edges_hamming(graph)
        _logger.info('compute_edge_mutations_star done: wall=%.2fs cpu=%.2fs', time.perf_counter()-_t_ham0, time.process_time()-_c_ham0)
    return graph
    
def expected_hamming_from_aligned(aligned_or_A: Sequence[np.ndarray] | np.ndarray,
                                  B: Optional[np.ndarray] = None,
                                  *,
                                  gap_at: int = -1,
                                  return_norm: bool = True,
                                  block_cols: Optional[int] = None,
                                  eps: float = 1e-12) -> Tuple:
    """
    Unified API to compute expected Hamming distances from aligned sequences.

    Two usage modes:
      1) Pairwise (two arrays):
         (mut, eff, dist) = expected_hamming_from_aligned(A, B, ...)
         where A and B are aligned soft arrays of shape (L, A) or (L, A+1) with gap channel optional.

      2) Batch (list/array of N sequences):
         (exp_mut, eff_len, dist) = expected_hamming_from_aligned([A1, A2, ..., AN], ...)
         where each Ai is aligned; supports both soft (L, A or L, A+1) and hard (N,L) representations.

    Parameters
    ----------
    aligned_or_A : Sequence[np.ndarray] | np.ndarray
        Either a sequence of aligned arrays (batch mode) or the first aligned array for pair mode.

    B : Optional[np.ndarray], default=None
        The second aligned array for pair mode. If provided, pair mode is used.

    gap_at : int, default=-1
        Index of the gap channel in the last axis when gap is explicitly present.
        If negative, counts from the end (-1 = last channel).

    return_norm : bool, default=True
        For batch mode, whether to return normalized mismatch fraction in [0,1].
        (Ignored in pair mode; the function always returns (mut, eff, dist).)

    block_cols : Optional[int], default=None
        Batch mode only. If set, process columns in blocks of this size for memory efficiency.

    eps : float, default=1e-12
        Small value to avoid division by zero in normalization.

    Returns
    -------
    Tuple
        Pair mode: (mut: float, eff: float, dist: float)
        Batch mode: (exp_mut: np.ndarray[N,N], eff_len: np.ndarray[N,N],
                     dist: Optional[np.ndarray[N,N]] depending on return_norm)
    """
    # Pair mode.
    if B is not None:
        Pu = _ensure_gapped_last(np.asarray(aligned_or_A, float))
        Pv = _ensure_gapped_last(np.asarray(B, float))

        if Pu.shape != Pv.shape:
            raise ValueError("Aligned arrays for a pair must have the same shape")

        L, C = Pu.shape
        gap_idx = gap_at if gap_at >= 0 else (C + gap_at)
        if not (0 <= gap_idx < C):
            raise ValueError("gap_at out of range")

        # split into gap and amino channels
        pu_gap = Pu[:, gap_idx]
        pv_gap = Pv[:, gap_idx]
        pu_aa  = np.delete(Pu, gap_idx, axis=1)
        pv_aa  = np.delete(Pv, gap_idx, axis=1)

        w = (1.0 - pu_gap) * (1.0 - pv_gap) # joint non-gap weight per column
        ident = np.sum(pu_aa * pv_aa, axis=1) # expected identity per column
        mut = float(np.sum(w * (1.0 - ident))) # expected mismatches
        eff = float(np.sum(w)) # effective length (non-gap weight)
        dist = float(mut / max(eff, eps))
        return mut, eff, dist

    # Batch mode
    P = np.asarray(aligned_or_A, dtype=object)

    # Soft aligned arrays with optional explicit gap channel: (N, L, C)
    if P.ndim == 3:
        P = np.asarray(aligned_or_A, dtype=np.float64)
        N, L, C = P.shape
        if C < 2:
            raise ValueError("aligned soft arrays need at least 1 AA + 1 gap channel")

        gap_idx = gap_at if gap_at >= 0 else (C + gap_at)
        if not (0 <= gap_idx < C):
            raise ValueError("gap_at out of range for last axis")

        P_gap = P[..., gap_idx]    # (N, L)
        P_aa  = np.delete(P, gap_idx, axis=2)  # (N, L, A)

        Wcol  = (1.0 - P_gap)      # (N, L)

        exp_mut = np.zeros((N, N), dtype=np.float64)
        eff_len = np.zeros((N, N), dtype=np.float64)

        Bsz = block_cols or L
        for s in range(0, L, Bsz):
            e  = min(s + Bsz, L)
            Pa = P_aa[:, s:e, :]                  # (N, b, A)
            W  = Wcol[:, s:e]                      # (N, b)

            id_batch = np.einsum("nka,mka->nmk", Pa, Pa, optimize=True)  # (N,N,b)
            w_batch  = np.einsum("nc,mc->nmc",  W,  W,  optimize=True)   # (N,N,b)

            exp_mut += np.sum(w_batch * (1.0 - id_batch), axis=2)
            eff_len += np.sum(w_batch, axis=2)

        if return_norm:
            dist = exp_mut / np.maximum(eff_len, eps)
            np.clip(dist, 0.0, 1.0, out=dist)
            return exp_mut, eff_len, dist
        else:
            return exp_mut, eff_len, None

    # Hard (label) alignment path: P is (N, L)
    elif P.ndim == 2:
        N, L = P.shape
        eq = (P[:, None, :] == P[None, :, :])
        mism = (~eq)
        exp_mut = mism.sum(axis=2).astype(np.float64)
        eff_len = np.full((N, N), float(L), dtype=np.float64)
        if return_norm:
            dist = exp_mut / np.maximum(eff_len, eps)
            return exp_mut, eff_len, dist
        else:
            return exp_mut, eff_len, None

    else:
        raise ValueError("aligned_or_A must stack to (N,L) or (N,L,C), or provide B for pair mode")

def _ensure_gapped_last(arr: np.ndarray) -> np.ndarray:
    """
    Helper function to ensure array is (L, A+1) with a final gap channel.
    If input is (L, A) (ungapped), append gap = 1 - sum(AA) (clipped).

    Parameters
    ----------
    arr : np.ndarray
        Input array of shape (L, A) or (L, A+1) where
        A is the number of amino acids (20 or 21 including gap).
    
    Returns
    -------
    np.ndarray
        Array of shape (L, A+1) with gap channel appended if needed.
    """
    if arr.ndim != 2:
        raise ValueError("sequence array must be 2-D (L, A or L, A+1)")
    L, C = arr.shape
    if C >= 21:
        return arr
    aa_sum = arr.sum(axis=1, keepdims=True)
    gap = np.clip(1.0 - aa_sum, 0.0, 1.0)
    return np.concatenate([arr, gap], axis=1)


def _star_block(u, neighbors, seq_u, seqs_v, alphabet, chunk_size, eps):
    # Limit thread usage inside each Ray worker to avoid oversubscription
    import os as _os
    _os.environ.setdefault('OMP_NUM_THREADS', '1')
    _os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    _os.environ.setdefault('MKL_NUM_THREADS', '1')
    _os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

    soft_alignment = require_optional(
        "softalign.soft_alignment",
        extra="alignment",
        purpose="edge mutation alignment",
    )
    A = len(alphabet)
    def _sanitize(arr: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
        # If gapped channel included, drop it for alignment with ungapped alphabet
        if x.ndim == 2 and x.shape[1] == A + 1:
            x = x[:, :A]
        if x.ndim != 2 or x.shape[1] != A:
            raise ValueError(f"Expected (L,{A}) array for alignment; got {x.shape}")
        # Replace NaNs and renormalise rows to sum 1
        x = np.where(np.isnan(x), 0.0, x)
        row_sum = x.sum(axis=1, keepdims=True)
        zero_mask = (row_sum <= 0.0)
        if np.any(zero_mask):
            x[zero_mask, :] = 1.0 / A
            row_sum[zero_mask] = 1.0
        x = x / row_sum
        return x

    set_w, set_d, set_s = {}, {}, {}
    Pu = _sanitize(seq_u.ungapped_arr)
    def chunks(lst, k):
        if not k:
            k = 8
        for i in range(0, len(lst), k): yield lst[i:i+k]
    for chunk_ids in chunks(list(range(len(neighbors))), chunk_size):
        seqs = [Pu] + [_sanitize(seqs_v[i].ungapped_arr) for i in chunk_ids]
        # Cast to float64 contiguous to avoid dtype issues in softalign
        seqs = [np.ascontiguousarray(np.asarray(s, dtype=np.float64)) for s in seqs]
        _res = soft_alignment.align_soft_sequences(sequences=seqs, alphabet=alphabet)
        aligned = _res[0] if isinstance(_res, tuple) else _res
        Au = np.asarray(aligned[0])
        for off, idx in enumerate(chunk_ids, start=1):
            v = neighbors[idx]
            Av = np.asarray(aligned[off])
            mut, eff, dist = expected_hamming_from_aligned(Au, Av)
            set_w[(u, v)] = float(mut)
            set_d[(u, v)] = float(dist)
            set_s[(u, v)] = _distance_affinity(float(dist))
    return set_w, set_d, set_s

def compute_edge_mutations_star(G: nx.Graph,
                                *,
                                alphabet: List = PROT_20,
                                chunk_size: Optional[int] = 8,
                                eps: float = 1e-12,
                                _log_progress: bool = False,
                                _nested_parallel: bool = False) -> None:
    """
    Compute expected Hamming distance per edge using star subgraphs, sequentially.

    This sequential implementation avoids Ray workers to mitigate native segfaults
    in highly parallel soft alignment. It preserves chunked star alignment to 
    reduce redundant work and memory usage.

    Parameters
    ----------
    G : nx.Graph
        The undirected graph to compute expected Hamming distances for.
    
    alphabet : List, default=PROT_20
        The ungapped alphabet used for alignment.

    chunk_size : Optional[int], default=8
        Process neighbors in chunks of this size to reduce memory. If falsy, uses 8.
    
    eps : float, default=1e-12
        Small value to avoid division by zero in normalization.
    _log_progress : bool, default=False
        Emit progress messages through the package logger.
    _nested_parallel : bool, default=False
        Compute node-star alignments with Ray workers.
    """
    if G.is_directed():
        raise TypeError("Edge mutation annotation requires an undirected graph.")

    A = len(alphabet)

    def _sanitize(arr: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
        if x.ndim == 2 and x.shape[1] == A + 1:
            x = x[:, :A]
        if x.ndim != 2 or x.shape[1] != A:
            raise ValueError(f"Expected (L,{A}) array for alignment; got {x.shape}")
        x = np.where(np.isnan(x), 0.0, x)
        row_sum = x.sum(axis=1, keepdims=True)
        zero_mask = (row_sum <= 0.0)
        if np.any(zero_mask):
            x[zero_mask, :] = 1.0 / A
            row_sum[zero_mask] = 1.0
        x = x / row_sum
        return x

    def _chunks(lst, k):
        k = 8 if not k or k <= 0 else k
        for i in range(0, len(lst), k):
            yield lst[i:i+k]

    import logging as _logging
    _logger = _logging.getLogger('fitness_landscape')
    if _log_progress:
        _logger.info('compute_edge_mutations_star: start (nodes=%d, edges=%d, chunk=%s)', G.number_of_nodes(), G.number_of_edges(), chunk_size)
    set_w, set_d, set_s = {}, {}, {}
    node_position = {node: index for index, node in enumerate(G.nodes())}

    if G.number_of_edges() == 0:
        return

    if not _nested_parallel:
        soft_alignment = require_optional(
            "softalign.soft_alignment",
            extra="alignment",
            purpose="edge mutation alignment",
        )
        for u in G.nodes():
            # Avoid duplicate pairs in undirected graphs
            nbrs = [
                v for v in G.neighbors(u)
                if node_position[u] < node_position[v]
            ]
            if not nbrs:
                continue
            Pu = _sanitize(G.nodes[u]['sequence'].ungapped_arr)
            for chunk_ids in _chunks(list(range(len(nbrs))), chunk_size):
                seqs = [Pu] + [_sanitize(G.nodes[nbrs[i]]['sequence'].ungapped_arr) for i in chunk_ids]
                _res = soft_alignment.align_soft_sequences(sequences=seqs, alphabet=alphabet)
                aligned = _res[0] if isinstance(_res, tuple) else _res
                Au = np.asarray(aligned[0])
                for off, idx in enumerate(chunk_ids, start=1):
                    v = nbrs[idx]
                    Av = np.asarray(aligned[off])
                    mut, eff, dist = expected_hamming_from_aligned(Au, Av)
                    set_w[(u, v)] = float(mut)
                    set_d[(u, v)] = float(dist)
                    set_s[(u, v)] = _distance_affinity(float(dist))
    else:
        # Ray-parallel star computation per node
        with ray_runtime(1, purpose="parallel edge mutation alignment") as ray:
            star_block_remote = ray.remote(num_cpus=1)(_star_block)
            tasks = []
            node_list = list(G.nodes())
            for u in node_list:
                nbrs = [
                    v for v in G.neighbors(u)
                    if node_position[u] < node_position[v]
                ]
                if not nbrs:
                    continue
                seqs_v = [G.nodes[v]['sequence'] for v in nbrs]
                tasks.append(
                    star_block_remote.remote(
                        u,
                        nbrs,
                        G.nodes[u]['sequence'],
                        seqs_v,
                        alphabet,
                        chunk_size,
                        eps,
                    )
                )
            if tasks:
                pending = set(tasks)
                try:
                    import psutil as _psutil
                except Exception:
                    _psutil = None
                done_count = 0
                total = len(tasks)
                while pending:
                    done, pending = ray.wait(list(pending), num_returns=1, timeout=30.0)
                    if done:
                        W, D, S = ray.get(done[0])
                        set_w.update(W); set_d.update(D); set_s.update(S)
                        done_count += 1
                        if _log_progress:
                            _logger.info('compute_edge_mutations_star (nested): %d/%d completed', done_count, total)
                    elif _log_progress:
                        rss = ''
                        if _psutil is not None:
                            p = _psutil.Process()
                            rss_bytes = p.memory_info().rss
                            rss = f" rss={rss_bytes/1e9:.2f}GB"
                        _logger.info('compute_edge_mutations_star heartbeat: %d/%d completed%s', done_count, total, rss)

    if set_w:
        nx.set_edge_attributes(G, set_w, "hamming_distance")
        nx.set_edge_attributes(G, set_d, "normalized_distance")
        nx.set_edge_attributes(G, set_s, "hamming_affinity")
    if _log_progress:
        _logger.info('compute_edge_mutations_star: complete')

def attach_expected_hamming_to_edges(G: nx.Graph,
                                     aligned: Sequence[np.ndarray],
                                     node_order: Optional[Sequence] = None,
                                     *,
                                     gap_at: int = -1,
                                     eps: float = 1e-12,
                                     block_cols: Optional[int] = None) -> None:
    """
    Function to attach expected Hamming edge attributes to a graph from a 
    precomputed alignment of soft sequences. The expected Hamming distance
    is computed for each edge based on the aligned sequences.

    Parameters
    ----------
    G : nx.Graph
        The undirected graph to attach expected Hamming distances to.
    
    aligned : Sequence[np.ndarray]
        List of aligned soft sequences, each of shape (L_aln, A+1)
        where last axis is A amino acids + gap. Indices must match the node indices in G or the
        indices in `node_order`.
    
    node_order : Optional[Sequence], default=None
        If provided, specifies the order of nodes in G to match the aligned sequences.
    
    gap_at : int, default=-1
        Index of the gap channel in the last axis of aligned[i].
        If negative, counts from the end (-1 = last channel).
    
    eps : float, default=1e-12
        Small value to avoid division by zero in normalization.
    block_cols : int, optional
        Reserved alignment-column block size.
    """
    if G.is_directed():
        raise TypeError("Expected Hamming annotation requires an undirected graph.")

    if node_order is None:
        node_order = list(G.nodes())
    if len(node_order) != len(aligned):
        raise ValueError("node_order length must match len(aligned)")

    idx = {n: i for i, n in enumerate(node_order)}

    # Compute only existing edges. This avoids the legacy O(n^2) pairwise
    # allocation when a sparse graph requests Hamming annotation.
    set_hamming_distance = {}
    set_normalized_distance = {}
    set_hamming_affinity = {}

    for u, v in G.edges():
        i, j = idx[u], idx[v]
        left = np.asarray(aligned[i])
        right = np.asarray(aligned[j])
        if left.shape != right.shape:
            raise ValueError(
                "Hamming edge annotation requires aligned arrays of equal shape."
            )
        if left.ndim == 1:
            effective_length = float(left.shape[0])
            mutations = float(np.count_nonzero(left != right))
            normalized = mutations / max(effective_length, eps)
        elif left.ndim == 2:
            mutations, _effective_length, normalized = expected_hamming_from_aligned(
                left,
                right,
                gap_at=gap_at,
                return_norm=True,
                block_cols=block_cols,
                eps=eps,
            )
        else:
            raise ValueError(
                "Aligned edge arrays must be hard 1-D labels or 2-D posterior matrices."
            )
        set_hamming_distance[(u, v)] = float(mutations)
        set_normalized_distance[(u, v)] = float(normalized)
        set_hamming_affinity[(u, v)] = _distance_affinity(float(normalized))

    if set_hamming_distance:
        nx.set_edge_attributes(G, set_hamming_distance, "hamming_distance")
        nx.set_edge_attributes(G, set_normalized_distance, "normalized_distance")
        nx.set_edge_attributes(G, set_hamming_affinity, "hamming_affinity")


def _annotate_existing_edges_hamming(G: nx.Graph) -> None:
    """Safely annotate only existing edges from already aligned sequences."""
    node_order = list(G.nodes())
    aligned = []
    for node in node_order:
        sequence = G.nodes[node].get("sequence")
        if not isinstance(sequence, BaseNumpySequence):
            raise ValueError(
                f"Graph node {node!r} lacks a BaseNumpySequence under 'sequence'."
            )
        if isinstance(sequence, SoftSequence):
            aligned.append(np.asarray(sequence.posterior, dtype=float))
        else:
            aligned.append(np.asarray(sequence.to_array()))
    attach_expected_hamming_to_edges(G, aligned, node_order=node_order)
