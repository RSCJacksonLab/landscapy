"""Canonical edge-attribute semantics for undirected fitness landscapes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import networkx as nx
import numpy as np


EDGE_SCHEMA_GRAPH_KEY = "landscapy_edge_schema"
EDGE_SCHEMA_VERSION = "1.0.0"
AUTO_EDGE_KEY = "auto"

EdgeSemantic = Literal[
    "distance",
    "normalized_distance",
    "affinity",
    "conductance",
    "transition_probability",
]


def declare_edge_semantics(
    graph: nx.Graph,
    *,
    constructor: str,
    distance_key: str | None = None,
    distance_units: str | None = None,
    normalized_distance_key: str | None = None,
    affinity_key: str | None = None,
    conductance_key: str | None = None,
    transition_probability_key: str | None = None,
    legacy_aliases: Mapping[str, str] | None = None,
    notes: str | None = None,
) -> None:
    """Declare the scientific meaning of edge attributes on ``graph``.

    The NetworkX ``weight`` convention is reserved for conductance. Raw or
    normalized distances must use distinct keys.

    Parameters
    ----------
    graph : networkx.Graph
        Undirected graph to annotate.
    constructor : str
        Name of the graph constructor.
    distance_key : str, optional
        Edge key containing raw distances.
    distance_units : str, optional
        Units of the raw distance.
    normalized_distance_key : str, optional
        Edge key containing distances in ``[0, 1]``.
    affinity_key : str, optional
        Edge key containing dimensionless affinities.
    conductance_key : str, optional
        Edge key containing dimensionless conductances.
    transition_probability_key : str, optional
        Edge key containing transition probabilities.
    legacy_aliases : mapping, optional
        Mapping from legacy keys to canonical semantics.
    notes : str, optional
        Additional schema notes.
    """
    if not isinstance(graph, nx.Graph):
        raise TypeError("graph must be a NetworkX graph")
    if graph.is_directed():
        raise TypeError("The Landscapy 0.9 edge schema supports undirected graphs only.")

    graph.graph[EDGE_SCHEMA_GRAPH_KEY] = {
        "schema_version": EDGE_SCHEMA_VERSION,
        "constructor": str(constructor),
        "distance": {"key": distance_key, "units": distance_units},
        "normalized_distance": {
            "key": normalized_distance_key,
            "units": "fraction" if normalized_distance_key is not None else None,
        },
        "affinity": {"key": affinity_key, "units": "dimensionless"},
        "conductance": {"key": conductance_key, "units": "dimensionless"},
        "transition_probability": {
            "key": transition_probability_key,
            "units": "probability" if transition_probability_key is not None else None,
        },
        "legacy_aliases": dict(legacy_aliases or {}),
        "notes": notes,
    }


def edge_semantics(graph: nx.Graph) -> dict[str, Any] | None:
    """Return a defensive copy of the declared edge schema.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to inspect.

    Returns
    -------
    dict or None
        Declared edge schema, or ``None`` when absent.
    """
    schema = graph.graph.get(EDGE_SCHEMA_GRAPH_KEY)
    return dict(schema) if isinstance(schema, Mapping) else None


def resolve_edge_attribute(
    graph: nx.Graph,
    semantic: EdgeSemantic,
    requested: str | None = AUTO_EDGE_KEY,
    *,
    required: bool = False,
) -> str | None:
    """Resolve and validate an edge key by scientific semantic.

    ``requested=None`` is an explicit request for an unweighted analysis.
    ``requested='auto'`` uses constructor-declared graph metadata. An
    attribute-free graph is treated as explicitly unweighted, while a legacy
    graph containing an ambiguous ``weight`` is rejected.

    Parameters
    ----------
    graph : networkx.Graph
        Graph whose edge attributes are resolved.
    semantic : str
        Scientific edge semantic to resolve.
    requested : str or None, default="auto"
        Explicit edge key, ``"auto"`` for schema lookup, or ``None`` for
        an unweighted analysis.
    required : bool, default=False
        Require a resolved edge key.

    Returns
    -------
    str or None
        Resolved edge key, or ``None`` for an unweighted analysis.
    """
    if requested is None:
        if required:
            raise ValueError(f"A {semantic} edge attribute is required.")
        return None

    if requested == AUTO_EDGE_KEY:
        schema = graph.graph.get(EDGE_SCHEMA_GRAPH_KEY)
        if not isinstance(schema, Mapping):
            has_legacy_weight = any(
                "weight" in data for _, _, data in graph.edges(data=True)
            )
            if not has_legacy_weight and not required:
                return None
            raise ValueError(
                "Graph edge semantics are undeclared. Pass an explicit edge key "
                f"for {semantic!r}, pass None for an unweighted analysis, or construct "
                "the graph with a Landscapy graph constructor."
            )
        spec = schema.get(semantic)
        key = spec.get("key") if isinstance(spec, Mapping) else None
        if key is None:
            if required:
                raise ValueError(
                    f"Graph schema does not declare an edge key for {semantic!r}."
                )
            return None
    else:
        key = requested

    missing = [(u, v) for u, v, data in graph.edges(data=True) if key not in data]
    if missing:
        preview = ", ".join(repr(edge) for edge in missing[:3])
        raise ValueError(
            f"Edge attribute {key!r} is missing from {len(missing)} edge(s): {preview}."
        )

    if semantic in {"distance", "normalized_distance", "affinity", "conductance", "transition_probability"}:
        invalid = []
        for u, v, data in graph.edges(data=True):
            value = data[key]
            try:
                scalar = float(value)
            except (TypeError, ValueError):
                invalid.append((u, v, value))
                continue
            if not np.isfinite(scalar) or scalar < 0.0:
                invalid.append((u, v, value))
            elif semantic in {"normalized_distance", "transition_probability"} and scalar > 1.0:
                invalid.append((u, v, value))
        if invalid:
            u, v, value = invalid[0]
            raise ValueError(
                f"Edge attribute {key!r} has an invalid {semantic} value "
                f"{value!r} on edge {(u, v)!r}."
            )
    return str(key)


def migrate_legacy_edge_semantics(
    graph: nx.Graph,
    *,
    sequence_length: int | None = None,
) -> bool:
    """Migrate recognizable pre-0.9 edge aliases without guessing ambiguity.

    Returns ``True`` when a known legacy constructor schema was migrated.
    Generic graphs containing only ``weight`` remain undeclared because that
    value could be either a distance or a conductance.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to migrate in place.
    sequence_length : int, optional
        Sequence length used to normalize legacy Hamming distances.

    Returns
    -------
    bool
        Whether a known legacy schema was migrated.
    """
    if EDGE_SCHEMA_GRAPH_KEY in graph.graph:
        return False

    attributes = {
        str(key) for _, _, data in graph.edges(data=True) for key in data
    }
    if not attributes:
        return False

    if "kernel_weight" in attributes:
        for _, _, data in graph.edges(data=True):
            value = float(data["kernel_weight"])
            data.setdefault("affinity", value)
            data.setdefault("weight", value)
        declare_edge_semantics(
            graph,
            constructor="legacy-diffusion",
            affinity_key="affinity",
            conductance_key="weight",
            legacy_aliases={"kernel_weight": "affinity"},
            notes="Migrated from a pre-0.9 kernel_weight bundle.",
        )
        return True

    if "tda_distance" in attributes:
        for _, _, data in graph.edges(data=True):
            distance = float(data["tda_distance"])
            affinity = 1.0 / (1.0 + distance)
            data["distance"] = distance
            data.setdefault("affinity", affinity)
            data["weight"] = data["affinity"]
        declare_edge_semantics(
            graph,
            constructor="legacy-tda",
            distance_key="distance",
            distance_units="pca_euclidean",
            affinity_key="affinity",
            conductance_key="weight",
            legacy_aliases={"tda_distance": "distance"},
            notes="Migrated from a pre-0.9 TDA bundle; legacy weight was a distance.",
        )
        return True

    if "knn_weight" in attributes and "distance" in attributes:
        length = int(sequence_length or 0)
        for _, _, data in graph.edges(data=True):
            distance = float(data["distance"])
            normalized = distance / length if length > 0 else None
            affinity = np.exp(-normalized) if normalized is not None else 1.0 / (1.0 + distance)
            data["distance"] = distance
            if normalized is not None:
                data.setdefault("normalized_distance", float(normalized))
            data.setdefault("affinity", float(affinity))
            data["weight"] = data["affinity"]
        declare_edge_semantics(
            graph,
            constructor="legacy-knn",
            distance_key="distance",
            distance_units="hamming_count",
            normalized_distance_key="normalized_distance" if length > 0 else None,
            affinity_key="affinity",
            conductance_key="weight",
            legacy_aliases={"knn_weight": "distance", "sim": "affinity"},
            notes="Migrated from a pre-0.9 kNN bundle; legacy weight was a distance.",
        )
        return True

    return False
