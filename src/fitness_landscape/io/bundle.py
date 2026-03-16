from __future__ import annotations

import hashlib
import importlib
import json
import math
import pickle
import platform
import shutil
import sys
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import networkx as nx
import numpy as np
import pandas as pd

from ..core.annotation import AnnotationLayer
from ..core.fitness import (
    CategoricalFitness,
    NumericFitness,
    ProbabilisticCategoricalFitness,
)
from ..core.sequence import BaseNumpySequence, SoftSequence
from .exceptions import BundleValidationError, ChecksumMismatchError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..core.landscape import FitnessLandscape


ARTIFACT_TYPE = "landscapy.fitness_landscape"
PORTABLE_SCHEMA_VERSION = "1.0.0"
PORTABLE_SERIALIZER_VERSION = "1.0.0"
PORTABLE_BACKEND = "portable"
PICKLE_BACKEND = "pickle"
PICKLE_FORMAT_VERSION = "pickle-v1"
TABULAR_STORAGE_PARQUET = "parquet"
TABULAR_STORAGE_JSON = "json-table-v1"

MANIFEST_FILENAME = "manifest.json"
METADATA_FILENAME = "metadata.json"
NODES_FILENAME = "nodes.json"
SEQUENCES_FILENAME = "sequences.npy"
SOFT_SEQUENCES_FILENAME = "soft_sequences.npy"
GRAPH_FILENAME = "graph_edges.parquet"
LAYERS_DIRNAME = "layers"
ANNOTATIONS_DIRNAME = "annotations"
EMBEDDING_DOMAINS_DIRNAME = "embedding_domains"
EMBEDDINGS_FILENAME = "embeddings.npy"
LEGACY_DIRNAME = "legacy"
LEGACY_PICKLE_FILENAME = "landscape.pkl"

ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
PICKLE_MANIFEST_CREATED_AT = "1980-01-01T00:00:00+00:00"


def save_bundle_dir(
    landscape: "FitnessLandscape",
    path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
    include_embeddings: bool = True,
    include_legacy_pickle: bool = False,
    overwrite: bool = False,
) -> Path:
    destination = Path(path)
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"Bundle directory already exists: {destination}")
        _remove_path(destination)

    with tempfile.TemporaryDirectory(dir=parent, prefix=f".{destination.name}.tmp-") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name) / "bundle"
        _write_portable_bundle_dir(
            landscape,
            tmp_dir,
            metadata=metadata,
            include_embeddings=include_embeddings,
            include_legacy_pickle=include_legacy_pickle,
        )
        shutil.move(str(tmp_dir), str(destination))

    return destination


def load_bundle_dir(path: str | Path):
    bundle_dir = Path(path)
    if not bundle_dir.is_dir():
        raise FileNotFoundError(f"Bundle directory not found: {bundle_dir}")

    manifest = _load_json(bundle_dir / MANIFEST_FILENAME)
    _validate_portable_manifest(manifest)

    metadata_path = bundle_dir / METADATA_FILENAME
    metadata_payload = _load_json(metadata_path)
    _validate_file_checksums(bundle_dir, manifest)

    node_records = _load_json(bundle_dir / manifest["nodes"]["path"])
    node_keys = [_decode_portable_value(rec["node_key"]) for rec in node_records]

    sequence_manifest = manifest["sequences"]
    sequence_matrix = np.load(bundle_dir / sequence_manifest["path"], allow_pickle=False)
    _validate_sequence_matrix(sequence_matrix, manifest["node_count"], manifest["sequence_length"])

    sequence_objects = _load_sequences(
        bundle_dir=bundle_dir,
        manifest=manifest,
        node_records=node_records,
        sequence_matrix=sequence_matrix,
    )

    graph = _load_graph(bundle_dir, manifest, node_keys)
    for node_key, sequence in zip(node_keys, sequence_objects):
        graph.nodes[node_key]["sequence"] = sequence

    fitness_layers = _load_fitness_layers(bundle_dir, manifest)
    annotation_layers = _load_annotation_layers(bundle_dir, manifest)
    embeddings, embedding_metadata = _load_embeddings(bundle_dir, manifest)

    landscape_class = _import_symbol(manifest["landscape_class"])
    if not isinstance(landscape_class, type):
        raise BundleValidationError(
            f"Landscape class path does not resolve to a type: {manifest['landscape_class']!r}"
        )

    landscape = landscape_class(
        sequences=sequence_objects,
        graph=graph,
        fitness_layers=fitness_layers or None,
        annotation_layers=annotation_layers or None,
        embeddings=embeddings or None,
        emb_arr_key=manifest.get("emb_arr_key", "emb_arr"),
        active_embedding_domain=manifest.get("active_embedding_domain"),
        embedding_metadata=embedding_metadata or None,
    )
    active_layer_name = manifest.get("active_layer_name")
    if active_layer_name:
        landscape.view(active_layer_name)

    setattr(landscape, "_bundle_manifest", manifest)
    setattr(landscape, "_bundle_metadata", metadata_payload)
    return landscape


def export_lsbundle(
    landscape: "FitnessLandscape",
    path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
    backend: str = PORTABLE_BACKEND,
    overwrite: bool = False,
) -> Path:
    destination = Path(path)
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"Bundle archive already exists: {destination}")
        _remove_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if backend == PORTABLE_BACKEND:
        with tempfile.TemporaryDirectory(dir=destination.parent, prefix=f".{destination.name}.tmp-") as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name) / "bundle"
            _write_portable_bundle_dir(
                landscape,
                tmp_dir,
                metadata=metadata,
                include_embeddings=True,
                include_legacy_pickle=False,
            )
            _write_deterministic_zip_from_directory(tmp_dir, destination)
        return destination

    if backend == PICKLE_BACKEND:
        payload_bytes = pickle.dumps(landscape, protocol=4)
        metadata_payload = _normalize_legacy_pickle_metadata(landscape, metadata=metadata)
        metadata_bytes = _stable_json_dumps(metadata_payload).encode("utf-8")
        payload_checksum = _sha256_bytes(payload_bytes)
        manifest_payload = {
            "artifact_schema_version": PORTABLE_SCHEMA_VERSION,
            "artifact_type": ARTIFACT_TYPE,
            "serialization_backend": PICKLE_BACKEND,
            "serialization_format_version": PICKLE_FORMAT_VERSION,
            "created_at": PICKLE_MANIFEST_CREATED_AT,
            "python_version": platform.python_version(),
            "payloads": [
                {
                    "path": "payloads/landscape.pkl",
                    "sha256": payload_checksum,
                    "size_bytes": len(payload_bytes),
                }
            ],
            "primary_payload": "payloads/landscape.pkl",
            "metadata_sha256": _sha256_bytes(metadata_bytes),
            "compatibility": {
                "python_specifier": _current_python_specifier(),
                "expected_class": _class_path(type(landscape)),
                "unsafe_deserialization_required": True,
                "package_versions": {},
            },
        }
        archive_payloads = {
            MANIFEST_FILENAME: _stable_json_dumps(manifest_payload).encode("utf-8"),
            METADATA_FILENAME: metadata_bytes,
            "payloads/landscape.pkl": payload_bytes,
        }
        _write_deterministic_zip_from_bytes(archive_payloads, destination)
        return destination

    raise ValueError(f"Unsupported lsbundle backend {backend!r}; expected 'portable' or 'pickle'.")


def _write_portable_bundle_dir(
    landscape: "FitnessLandscape",
    bundle_dir: Path,
    *,
    metadata: Mapping[str, Any] | None,
    include_embeddings: bool,
    include_legacy_pickle: bool,
) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)

    graph = getattr(landscape, "graph", None)
    if not isinstance(graph, nx.Graph):
        raise BundleValidationError("FitnessLandscape.graph must be a networkx Graph or DiGraph.")

    original_node_order = _resolve_original_node_order(landscape)
    node_to_sequence_index = _match_nodes_to_sequence_indices(landscape, original_node_order)
    canonical_records = _build_canonical_node_records(
        landscape,
        original_node_order=original_node_order,
        node_to_sequence_index=node_to_sequence_index,
    )
    canonical_sequence_objects = [record["sequence"] for record in canonical_records]
    canonical_sequence_indices = [record["sequence_index"] for record in canonical_records]
    canonical_node_keys = [record["node_key"] for record in canonical_records]
    node_to_canonical_index = {
        record["node_key"]: idx for idx, record in enumerate(canonical_records)
    }

    layers_dir = bundle_dir / LAYERS_DIRNAME
    layers_dir.mkdir(parents=True, exist_ok=True)

    files_written: list[Path] = []

    metadata_payload = _normalize_bundle_metadata(metadata)
    metadata_path = bundle_dir / METADATA_FILENAME
    _write_json(metadata_path, metadata_payload)
    files_written.append(metadata_path)

    nodes_payload = [
        {
            "node_key": _encode_portable_value(record["node_key"]),
            "sequence_id": record["sequence_id"],
            "sequence_alphabet": _normalize_json(record["sequence"].alphabet),
            "sequence_class": _class_path(type(record["sequence"])),
        }
        for record in canonical_records
    ]
    nodes_path = bundle_dir / NODES_FILENAME
    _write_json(nodes_path, nodes_payload)
    files_written.append(nodes_path)

    sequence_payload = _build_sequence_payload(canonical_sequence_objects)
    sequences_path = bundle_dir / SEQUENCES_FILENAME
    _write_npy(sequences_path, sequence_payload["hard_matrix"])
    files_written.append(sequences_path)

    if sequence_payload["soft_tensor"] is not None:
        soft_sequences_path = bundle_dir / SOFT_SEQUENCES_FILENAME
        _write_npy(soft_sequences_path, sequence_payload["soft_tensor"])
        files_written.append(soft_sequences_path)
    else:
        soft_sequences_path = None

    graph_manifest = _write_graph_edges(
        graph=graph,
        bundle_dir=bundle_dir,
        node_to_canonical_index=node_to_canonical_index,
    )
    files_written.append(bundle_dir / graph_manifest["path"])

    layer_manifests = []
    for layer_key in sorted(getattr(landscape, "fitness_layers", {}).keys()):
        layer = landscape.fitness_layers[layer_key]
        layer_manifest = _write_fitness_layer(
            layer_key=layer_key,
            layer=layer,
            canonical_sequence_indices=canonical_sequence_indices,
            layers_dir=layers_dir,
        )
        layer_manifests.append(layer_manifest)
        files_written.append(bundle_dir / layer_manifest["path"])

    annotation_manifests = []
    annotation_layers = getattr(landscape, "annotation_layers", {}) or {}
    if annotation_layers:
        annotations_dir = bundle_dir / ANNOTATIONS_DIRNAME
        annotations_dir.mkdir(parents=True, exist_ok=True)
        for layer_key in sorted(annotation_layers.keys()):
            layer = annotation_layers[layer_key]
            annotation_manifest = _write_annotation_layer(
                layer_key=layer_key,
                layer=layer,
                canonical_sequence_indices=canonical_sequence_indices,
                annotations_dir=annotations_dir,
            )
            annotation_manifests.append(annotation_manifest)
            files_written.append(bundle_dir / annotation_manifest["path"])

    embeddings_manifest = None
    if include_embeddings:
        embeddings_manifest = _write_embeddings(
            landscape=landscape,
            bundle_dir=bundle_dir,
            canonical_records=canonical_records,
        )
        if embeddings_manifest is not None:
            for item in embeddings_manifest["domains"]:
                files_written.append(bundle_dir / item["path"])

    legacy_manifest = None
    if include_legacy_pickle:
        legacy_dir = bundle_dir / LEGACY_DIRNAME
        legacy_dir.mkdir(parents=True, exist_ok=True)
        legacy_path = legacy_dir / LEGACY_PICKLE_FILENAME
        legacy_path.write_bytes(pickle.dumps(landscape, protocol=4))
        files_written.append(legacy_path)
        legacy_manifest = {"path": _relative_path(legacy_path, bundle_dir)}

    files_manifest = {
        _relative_path(file_path, bundle_dir): _file_descriptor(file_path)
        for file_path in sorted(files_written, key=lambda p: _relative_path(p, bundle_dir))
    }

    manifest = {
        "schema_version": PORTABLE_SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "serializer_backend": PORTABLE_BACKEND,
        "serializer_version": PORTABLE_SERIALIZER_VERSION,
        "landscape_class": _class_path(type(landscape)),
        "graph_class": _class_path(type(graph)),
        "graph_directed": bool(graph.is_directed()),
        "node_count": len(canonical_records),
        "edge_count": int(graph.number_of_edges()),
        "sequence_length": len(canonical_sequence_objects[0]) if canonical_sequence_objects else 0,
        "molecule_type": _infer_global_molecule_type(canonical_sequence_objects),
        "alphabet": _collect_global_alphabet(canonical_sequence_objects),
        "emb_arr_key": getattr(landscape, "_emb_arr_key", "emb_arr"),
        "active_layer_name": getattr(landscape, "active_layer_name", None),
        "active_embedding_domain": (
            embeddings_manifest["active_domain"] if embeddings_manifest is not None else None
        ),
        "node_ordering": {
            "kind": "canonical",
            "declaration": (
                "Nodes are stored in canonical order sorted by sequence content, sequence_id, "
                "and original node key when it can be encoded portably."
            ),
        },
        "nodes": {
            "path": NODES_FILENAME,
            "key_codec": "tagged-json-v1",
        },
        "sequences": {
            "path": SEQUENCES_FILENAME,
            "representation": sequence_payload["representation"],
            "token_kind": sequence_payload["token_kind"],
            "array_dtype": str(sequence_payload["hard_matrix"].dtype),
            "soft_posteriors_path": (
                SOFT_SEQUENCES_FILENAME if soft_sequences_path is not None else None
            ),
            "soft_posterior_dtype": (
                str(sequence_payload["soft_tensor"].dtype)
                if sequence_payload["soft_tensor"] is not None
                else None
            ),
        },
        "graph": graph_manifest,
        "layers": layer_manifests,
        "annotation_layers": annotation_manifests,
        "embeddings": embeddings_manifest,
        "legacy_pickle": legacy_manifest,
        "files": files_manifest,
    }

    manifest_path = bundle_dir / MANIFEST_FILENAME
    _write_json(manifest_path, manifest)


def _build_sequence_payload(sequences: Sequence[BaseNumpySequence]) -> dict[str, Any]:
    if not sequences:
        raise BundleValidationError("Portable bundles require at least one sequence.")

    contains_soft = any(isinstance(seq, SoftSequence) for seq in sequences)
    if contains_soft and not all(isinstance(seq, SoftSequence) for seq in sequences):
        raise BundleValidationError("Mixed hard and soft sequence bundles are not currently supported.")

    hard_matrix, token_kind = _build_hard_sequence_matrix(sequences)

    if not contains_soft:
        return {
            "representation": "hard",
            "token_kind": token_kind,
            "hard_matrix": hard_matrix,
            "soft_tensor": None,
        }

    first_soft = sequences[0]
    posterior_shape = np.asarray(first_soft.posterior, dtype=float).shape
    posterior_alphabet = list(first_soft.alphabet)
    soft_arrays = []
    for seq in sequences:
        if list(seq.alphabet) != posterior_alphabet:
            raise BundleValidationError("Soft sequences must share the same alphabet in portable bundles.")
        posterior = np.asarray(seq.posterior, dtype=float)
        if posterior.shape != posterior_shape:
            raise BundleValidationError("Soft sequences must share the same posterior tensor shape.")
        soft_arrays.append(posterior)

    return {
        "representation": "soft",
        "token_kind": token_kind,
        "hard_matrix": hard_matrix,
        "soft_tensor": np.asarray(soft_arrays, dtype=float),
    }


def _build_hard_sequence_matrix(sequences: Sequence[BaseNumpySequence]) -> tuple[np.ndarray, str]:
    rows = [np.asarray(seq.to_array()).reshape(-1) for seq in sequences]
    lengths = {row.shape[0] for row in rows}
    if len(lengths) != 1:
        raise BundleValidationError("All sequences must share the same length.")

    scalar_values = [_coerce_scalar_token(value) for row in rows for value in row.tolist()]
    token_kind = _infer_scalar_kind(scalar_values)

    if token_kind == "bool":
        matrix = np.asarray([[bool(_coerce_scalar_token(v)) for v in row.tolist()] for row in rows], dtype=bool)
    elif token_kind == "int":
        matrix = np.asarray([[int(_coerce_scalar_token(v)) for v in row.tolist()] for row in rows], dtype=np.int64)
    elif token_kind == "float":
        matrix = np.asarray([[float(_coerce_scalar_token(v)) for v in row.tolist()] for row in rows], dtype=np.float64)
    else:
        matrix = np.asarray(
            [[str(_coerce_scalar_token(v)) for v in row.tolist()] for row in rows],
            dtype=_string_dtype_from_values(scalar_values),
        )

    return matrix, token_kind


def _write_graph_edges(
    *,
    graph: nx.Graph,
    bundle_dir: Path,
    node_to_canonical_index: Mapping[Any, int],
) -> dict[str, Any]:
    edge_rows = []
    attribute_values: dict[str, list[Any]] = defaultdict(list)
    attribute_names = sorted({str(key) for _, _, data in graph.edges(data=True) for key in data.keys()})

    def normalize_endpoints(u: Any, v: Any) -> tuple[int, int]:
        src = int(node_to_canonical_index[u])
        dst = int(node_to_canonical_index[v])
        if graph.is_directed():
            return src, dst
        return (src, dst) if src <= dst else (dst, src)

    ordered_edges = []
    for u, v, data in graph.edges(data=True):
        src, dst = normalize_endpoints(u, v)
        ordered_edges.append((src, dst, data))
    ordered_edges.sort(key=lambda item: (item[0], item[1]))

    for src, dst, data in ordered_edges:
        row = {"source_index": src, "target_index": dst}
        edge_rows.append(row)
        for attribute_name in attribute_names:
            attribute_values[attribute_name].append(data.get(attribute_name))

    frame = pd.DataFrame(edge_rows)
    attribute_manifest = []
    for attribute_name in attribute_names:
        codec = _infer_attribute_codec(attribute_values[attribute_name])
        column_name = f"attr_{attribute_name}"
        frame[column_name] = _build_attribute_series(attribute_values[attribute_name], codec=codec)
        attribute_manifest.append(
            {
                "name": attribute_name,
                "column": column_name,
                "codec": codec,
            }
        )

    graph_path = bundle_dir / GRAPH_FILENAME
    storage_backend = _write_parquet(frame, graph_path)
    return {
        "path": GRAPH_FILENAME,
        "storage_backend": storage_backend,
        "edge_attributes": attribute_manifest,
    }


def _write_fitness_layer(
    *,
    layer_key: str,
    layer: Any,
    canonical_sequence_indices: Sequence[int],
    layers_dir: Path,
) -> dict[str, Any]:
    filename = f"{_safe_filename(layer_key)}.parquet"
    path = layers_dir / filename
    metadata = _normalize_json(getattr(layer, "metadata", {}) or {})

    if isinstance(layer, NumericFitness):
        replicates = [
            [float(value) for value in layer.get_value(idx)]
            for idx in canonical_sequence_indices
        ]
        max_replicates = max((len(row) for row in replicates), default=0)
        tensor = np.full((len(replicates), max_replicates), np.nan, dtype=float)
        replicate_counts = np.zeros(len(replicates), dtype=np.int64)
        for row_index, row in enumerate(replicates):
            replicate_counts[row_index] = len(row)
            if row:
                tensor[row_index, : len(row)] = row
        frame = pd.DataFrame({"sequence_index": np.arange(tensor.shape[0], dtype=np.int64)})
        frame["replicate_count"] = replicate_counts
        replicate_columns = []
        for idx in range(tensor.shape[1]):
            column_name = f"replicate_{idx:04d}"
            frame[column_name] = tensor[:, idx]
            replicate_columns.append(column_name)
        storage_backend = _write_parquet(frame, path)
        return {
            "key": layer_key,
            "name": layer.name,
            "class_path": _class_path(type(layer)),
            "dtype": layer.dtype,
            "path": _relative_path(path, layers_dir.parent),
            "storage_backend": storage_backend,
            "encoding": "numeric_replicates_matrix",
            "replicate_count_column": "replicate_count",
            "replicate_columns": replicate_columns,
            "metadata": metadata,
        }

    if isinstance(layer, ProbabilisticCategoricalFitness):
        probabilities = np.asarray(layer.probabilities, dtype=float)[list(canonical_sequence_indices)]
        frame = pd.DataFrame({"sequence_index": np.arange(probabilities.shape[0], dtype=np.int64)})
        probability_columns = []
        for idx, _category in enumerate(layer.categories):
            column_name = f"probability_{idx:04d}"
            frame[column_name] = probabilities[:, idx]
            probability_columns.append(column_name)
        storage_backend = _write_parquet(frame, path)
        return {
            "key": layer_key,
            "name": layer.name,
            "class_path": _class_path(type(layer)),
            "dtype": layer.dtype,
            "path": _relative_path(path, layers_dir.parent),
            "storage_backend": storage_backend,
            "encoding": "categorical_probabilities",
            "categories": _normalize_json(list(layer.categories)),
            "probability_columns": probability_columns,
            "metadata": metadata,
        }

    if isinstance(layer, CategoricalFitness):
        values = [layer.get_value(idx) for idx in canonical_sequence_indices]
        frame = pd.DataFrame(
            {
                "sequence_index": np.arange(len(values), dtype=np.int64),
                "value": pd.Series(values, dtype="string"),
            }
        )
        storage_backend = _write_parquet(frame, path)
        return {
            "key": layer_key,
            "name": layer.name,
            "class_path": _class_path(type(layer)),
            "dtype": layer.dtype,
            "path": _relative_path(path, layers_dir.parent),
            "storage_backend": storage_backend,
            "encoding": "categorical_values",
            "categories": _normalize_json(list(layer.categories)),
            "metadata": metadata,
        }

    raise BundleValidationError(
        f"Unsupported fitness layer type for portable serialization: {type(layer).__name__}"
    )


def _write_annotation_layer(
    *,
    layer_key: str,
    layer: AnnotationLayer,
    canonical_sequence_indices: Sequence[int],
    annotations_dir: Path,
) -> dict[str, Any]:
    filename = f"{_safe_filename(layer_key)}.parquet"
    path = annotations_dir / filename
    frame = layer.to_dataframe(copy=True).iloc[list(canonical_sequence_indices)].reset_index(drop=True)
    frame.insert(0, "sequence_index", np.arange(len(frame), dtype=np.int64))
    storage_backend = _write_parquet(frame, path)
    return {
        "key": layer_key,
        "name": layer.name,
        "path": _relative_path(path, annotations_dir.parent),
        "storage_backend": storage_backend,
        "columns": _normalize_json(list(layer.columns)),
        "metadata": _normalize_json(layer.metadata),
    }


def _write_embeddings(
    *,
    landscape: "FitnessLandscape",
    bundle_dir: Path,
    canonical_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    embeddings = getattr(landscape, "embeddings", {}) or {}
    if not embeddings:
        return None

    original_node_order = _resolve_original_node_order(landscape)
    original_index = {node_key: idx for idx, node_key in enumerate(original_node_order)}
    reorder = [original_index[record["node_key"]] for record in canonical_records]

    active_domain = getattr(landscape, "active_embedding_domain", None)
    if active_domain is None:
        active_domain = next(iter(embeddings), None)

    domains_manifest = []
    extra_dir = bundle_dir / EMBEDDING_DOMAINS_DIRNAME
    extra_dir.mkdir(parents=True, exist_ok=True)

    for domain_name in sorted(embeddings.keys()):
        matrix = np.asarray(embeddings[domain_name])
        if matrix.shape[0] != len(original_node_order):
            raise BundleValidationError(
                f"Embedding domain {domain_name!r} has {matrix.shape[0]} rows, "
                f"expected {len(original_node_order)} to match the graph nodes."
            )
        reordered = matrix[reorder]
        if domain_name == active_domain:
            target = bundle_dir / EMBEDDINGS_FILENAME
        else:
            target = extra_dir / f"{_safe_filename(domain_name)}.npy"
        _write_npy(target, reordered)
        domains_manifest.append(
            {
                "name": domain_name,
                "path": _relative_path(target, bundle_dir),
                "dtype": str(reordered.dtype),
                "shape": _normalize_json(list(reordered.shape)),
                "metadata": _normalize_json(
                    getattr(landscape, "get_embedding_metadata", lambda _domain: None)(domain_name)
                    or {}
                ),
            }
        )

    return {
        "active_domain": active_domain,
        "domains": domains_manifest,
    }


def _build_canonical_node_records(
    landscape: "FitnessLandscape",
    *,
    original_node_order: Sequence[Any],
    node_to_sequence_index: Mapping[Any, int],
) -> list[dict[str, Any]]:
    records = []
    for original_position, node_key in enumerate(original_node_order):
        sequence_index = node_to_sequence_index[node_key]
        sequence = landscape.sequences[sequence_index]
        sequence_tokens = [_coerce_scalar_token(value) for value in sequence.to_array().tolist()]
        node_sort_key = _portable_sort_key(node_key)
        records.append(
            {
                "node_key": node_key,
                "original_position": original_position,
                "sequence_index": sequence_index,
                "sequence": sequence,
                "sequence_id": _normalize_sequence_id(sequence),
                "sequence_sort_key": _stable_json_dumps(sequence_tokens, indent=None),
                "node_sort_key": node_sort_key if node_sort_key is not None else f"ordinal:{original_position:08d}",
            }
        )

    records.sort(
        key=lambda record: (
            record["sequence_sort_key"],
            record["sequence_id"] or "",
            record["node_sort_key"],
            record["original_position"],
        )
    )
    return records


def _resolve_original_node_order(landscape: "FitnessLandscape") -> list[Any]:
    graph_nodes = list(landscape.graph.nodes())
    stored = list(getattr(landscape, "_node_order", graph_nodes))
    if len(stored) != len(graph_nodes) or set(stored) != set(graph_nodes):
        return graph_nodes
    return stored


def _match_nodes_to_sequence_indices(
    landscape: "FitnessLandscape",
    node_order: Sequence[Any],
) -> dict[Any, int]:
    if len(landscape.sequences) != len(node_order):
        raise BundleValidationError(
            "Portable bundle serialization currently requires one sequence per graph node."
        )

    by_key_and_id: dict[tuple[tuple[Any, ...], str | None], list[int]] = defaultdict(list)
    by_key: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for idx, sequence in enumerate(landscape.sequences):
        key = _sequence_lookup_key(sequence)
        seq_id = _normalize_sequence_id(sequence)
        by_key_and_id[(key, seq_id)].append(idx)
        by_key[key].append(idx)

    used: set[int] = set()
    mapping: dict[Any, int] = {}

    for node_key in node_order:
        node_sequence = landscape.graph.nodes[node_key].get("sequence")
        if not isinstance(node_sequence, BaseNumpySequence):
            raise BundleValidationError(
                f"Graph node {node_key!r} does not expose a BaseNumpySequence under 'sequence'."
            )

        key = _sequence_lookup_key(node_sequence)
        seq_id = _normalize_sequence_id(node_sequence)
        candidates = [idx for idx in by_key_and_id[(key, seq_id)] if idx not in used]
        if not candidates:
            candidates = [idx for idx in by_key[key] if idx not in used]
        if not candidates:
            raise BundleValidationError(
                f"Unable to align graph node {node_key!r} to a unique sequence record."
            )
        selected = candidates[0]
        used.add(selected)
        mapping[node_key] = selected

    if len(mapping) != len(node_order):
        raise BundleValidationError("Failed to align every graph node to a sequence.")
    return mapping


def _load_sequences(
    *,
    bundle_dir: Path,
    manifest: Mapping[str, Any],
    node_records: Sequence[Mapping[str, Any]],
    sequence_matrix: np.ndarray,
) -> list[BaseNumpySequence]:
    if len(node_records) != manifest["node_count"]:
        raise BundleValidationError("nodes.json length does not match node_count in manifest.")

    sequence_manifest = manifest["sequences"]
    representation = sequence_manifest["representation"]
    molecule_type = manifest.get("molecule_type")
    result = []

    if representation == "hard":
        for idx, record in enumerate(node_records):
            result.append(
                BaseNumpySequence(
                    sequence_matrix[idx],
                    sequence_id=record.get("sequence_id"),
                    alphabet=record.get("sequence_alphabet") or manifest.get("alphabet"),
                    moltype=molecule_type,
                )
            )
        return result

    if representation != "soft":
        raise BundleValidationError(f"Unsupported sequence representation {representation!r}")

    soft_path = sequence_manifest.get("soft_posteriors_path")
    if not soft_path:
        raise BundleValidationError("Soft sequence bundle is missing soft_posteriors_path.")
    posterior_tensor = np.load(bundle_dir / soft_path, allow_pickle=False)
    if posterior_tensor.shape[0] != manifest["node_count"]:
        raise BundleValidationError("Soft posterior tensor row count does not match node_count.")

    for idx, record in enumerate(node_records):
        alphabet = list(record.get("sequence_alphabet") or manifest.get("alphabet") or [])
        posterior = np.asarray(posterior_tensor[idx], dtype=float)
        if "gap" in alphabet:
            gap_index = alphabet.index("gap")
            base_alphabet = [item for item in alphabet if item != "gap"]
            gap_posterior = posterior[:, gap_index : gap_index + 1]
            aa_posterior = np.delete(posterior, gap_index, axis=1)
            sequence = SoftSequence.from_posteriors(
                aa_posterior,
                alphabet=base_alphabet,
                gap_posterior=gap_posterior,
                hard_rule="argmax",
            )
        else:
            sequence = SoftSequence.from_posteriors(
                posterior,
                alphabet=alphabet,
                hard_rule="argmax",
            )
        sequence.id = record.get("sequence_id")
        hard_proxy = np.asarray(sequence.to_array())
        if hard_proxy.shape != sequence_matrix[idx].shape or not np.array_equal(
            hard_proxy.astype(str), np.asarray(sequence_matrix[idx]).astype(str)
        ):
            raise BundleValidationError(
                "Soft sequence hard proxy does not match sequences.npy contents."
            )
        result.append(sequence)

    return result


def _load_graph(bundle_dir: Path, manifest: Mapping[str, Any], node_keys: Sequence[Any]) -> nx.Graph:
    graph_class = _load_graph_class(manifest)
    graph = graph_class()
    graph.add_nodes_from(node_keys)

    graph_manifest = manifest["graph"]
    frame = _read_parquet(
        bundle_dir / graph_manifest["path"],
        storage_backend=graph_manifest.get("storage_backend"),
    )
    edge_attributes = graph_manifest.get("edge_attributes", [])

    for row in frame.to_dict(orient="records"):
        src_index = int(row["source_index"])
        dst_index = int(row["target_index"])
        attrs = {}
        for spec in edge_attributes:
            value = row.get(spec["column"])
            attrs[spec["name"]] = _decode_attribute_value(value, codec=spec["codec"])
        attrs = {key: value for key, value in attrs.items() if value is not None}
        graph.add_edge(node_keys[src_index], node_keys[dst_index], **attrs)

    return graph


def _load_fitness_layers(bundle_dir: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    layers = {}
    for layer_spec in manifest.get("layers", []):
        frame = _read_parquet(
            bundle_dir / layer_spec["path"],
            storage_backend=layer_spec.get("storage_backend"),
        )
        _validate_sequence_index_column(frame, manifest["node_count"])
        data_frame = frame.drop(columns=["sequence_index"])
        metadata = layer_spec.get("metadata") or {}
        encoding = layer_spec["encoding"]

        if encoding == "numeric_replicates_matrix":
            matrix = data_frame[layer_spec["replicate_columns"]].to_numpy(dtype=float)
            count_column = layer_spec.get("replicate_count_column")
            if count_column is not None and count_column in frame.columns:
                counts = frame[count_column].tolist()
                replicates = []
                for row_index, count in enumerate(counts):
                    n = int(count)
                    replicates.append(matrix[row_index, :n].tolist())
                layer = NumericFitness.from_replicates(
                    layer_spec["name"],
                    replicates,
                    metadata=metadata,
                )
            else:
                layer = NumericFitness.from_tensor(
                    layer_spec["name"],
                    matrix,
                    metadata=metadata,
                    pad_strategy="trim_tail_nans",
                )
        elif encoding == "categorical_values":
            values = [None if pd.isna(value) else str(value) for value in data_frame["value"].tolist()]
            layer = CategoricalFitness(
                name=layer_spec["name"],
                values=values,
                categories=list(layer_spec["categories"]),
                metadata=metadata,
            )
        elif encoding == "categorical_probabilities":
            probabilities = data_frame[layer_spec["probability_columns"]].to_numpy(dtype=float)
            layer = ProbabilisticCategoricalFitness.from_probabilities(
                layer_spec["name"],
                probabilities,
                categories=list(layer_spec["categories"]),
                metadata=metadata,
            )
        else:
            raise BundleValidationError(f"Unsupported layer encoding {encoding!r}")

        layers[layer_spec["key"]] = layer
    return layers


def _load_annotation_layers(bundle_dir: Path, manifest: Mapping[str, Any]) -> dict[str, AnnotationLayer]:
    layers = {}
    for layer_spec in manifest.get("annotation_layers", []):
        frame = _read_parquet(
            bundle_dir / layer_spec["path"],
            storage_backend=layer_spec.get("storage_backend"),
        )
        _validate_sequence_index_column(frame, manifest["node_count"])
        data_frame = frame.drop(columns=["sequence_index"]).reset_index(drop=True)
        layer = AnnotationLayer(
            name=layer_spec["name"],
            data=data_frame,
            metadata=layer_spec.get("metadata") or {},
        )
        layers[layer_spec["key"]] = layer
    return layers


def _load_embeddings(
    bundle_dir: Path, manifest: Mapping[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    embeddings_manifest = manifest.get("embeddings")
    if not embeddings_manifest:
        return {}, {}

    embeddings = {}
    metadata = {}
    for spec in embeddings_manifest.get("domains", []):
        matrix = np.load(bundle_dir / spec["path"], allow_pickle=False)
        if matrix.shape[0] != manifest["node_count"]:
            raise BundleValidationError(
                f"Embedding domain {spec['name']!r} row count does not match node_count."
            )
        embeddings[spec["name"]] = np.asarray(matrix)
        metadata[spec["name"]] = spec.get("metadata") or {}

    return embeddings, metadata


def _validate_portable_manifest(manifest: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "artifact_type",
        "serializer_backend",
        "serializer_version",
        "landscape_class",
        "node_count",
        "edge_count",
        "sequence_length",
        "alphabet",
        "nodes",
        "sequences",
        "graph",
        "layers",
        "files",
        "node_ordering",
    }
    missing = sorted(required - set(manifest.keys()))
    if missing:
        raise BundleValidationError(f"Manifest is missing required keys: {missing}")
    if manifest["schema_version"] != PORTABLE_SCHEMA_VERSION:
        raise BundleValidationError(
            f"Unsupported bundle schema_version {manifest['schema_version']!r}; "
            f"expected {PORTABLE_SCHEMA_VERSION!r}."
        )
    if manifest["artifact_type"] != ARTIFACT_TYPE:
        raise BundleValidationError(
            f"Unsupported artifact_type {manifest['artifact_type']!r}; expected {ARTIFACT_TYPE!r}."
        )
    if manifest["serializer_backend"] != PORTABLE_BACKEND:
        raise BundleValidationError(
            f"Unsupported serializer_backend {manifest['serializer_backend']!r}; "
            f"expected {PORTABLE_BACKEND!r}."
        )


def _validate_file_checksums(bundle_dir: Path, manifest: Mapping[str, Any]) -> None:
    files_manifest = manifest.get("files", {})
    for relative_path, descriptor in files_manifest.items():
        file_path = bundle_dir / relative_path
        if not file_path.exists():
            raise BundleValidationError(f"Bundle payload listed in manifest is missing: {relative_path}")
        actual = _sha256_file(file_path)
        expected = descriptor["sha256"]
        if actual != expected:
            raise ChecksumMismatchError(
                f"Checksum mismatch for {relative_path}: expected {expected}, got {actual}"
            )
        actual_size = file_path.stat().st_size
        if actual_size != descriptor["size_bytes"]:
            raise BundleValidationError(
                f"Size mismatch for {relative_path}: expected {descriptor['size_bytes']}, got {actual_size}"
            )


def _validate_sequence_matrix(sequence_matrix: np.ndarray, node_count: int, sequence_length: int) -> None:
    if sequence_matrix.ndim != 2:
        raise BundleValidationError("sequences.npy must be a 2-D array.")
    if sequence_matrix.shape[0] != node_count:
        raise BundleValidationError("sequences.npy row count does not match node_count.")
    if sequence_matrix.shape[1] != sequence_length:
        raise BundleValidationError("sequences.npy width does not match sequence_length.")


def _validate_sequence_index_column(frame: pd.DataFrame, node_count: int) -> None:
    if "sequence_index" not in frame.columns:
        raise BundleValidationError("Bundle parquet payload is missing the sequence_index column.")
    indices = frame["sequence_index"].tolist()
    if indices != list(range(node_count)):
        raise BundleValidationError("sequence_index column is not in canonical order.")


def _load_graph_class(manifest: Mapping[str, Any]):
    class_path = manifest.get("graph_class")
    if class_path:
        try:
            graph_class = _import_symbol(class_path)
            if isinstance(graph_class, type) and issubclass(graph_class, nx.Graph):
                return graph_class
        except Exception:
            pass
    return nx.DiGraph if manifest.get("graph_directed") else nx.Graph


def _normalize_bundle_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = dict(metadata or {})
    normalized = {
        "dataset_name": _pop_alias(payload, "dataset_name", "dataset"),
        "source_name": _pop_alias(payload, "source_name", "source"),
        "protein_gene": _pop_alias(payload, "protein_gene", "protein", "gene", "gene_id"),
        "assay_type": _pop_alias(payload, "assay_type", "assay"),
        "organism": _pop_alias(payload, "organism"),
        "version": _pop_alias(payload, "version"),
        "tags": _normalize_tags(payload.pop("tags", [])),
        "provenance": _normalize_json(payload.pop("provenance", {}) or {}),
        "metadata": {},
    }

    user_metadata = payload.pop("metadata", {}) or {}
    if not isinstance(user_metadata, Mapping):
        raise TypeError("metadata['metadata'] must be a mapping when provided.")
    normalized["metadata"] = _normalize_json(dict(user_metadata))

    for key in sorted(payload.keys(), key=str):
        normalized["metadata"][str(key)] = _normalize_json(payload[key])

    return normalized


def _normalize_legacy_pickle_metadata(
    landscape: "FitnessLandscape",
    *,
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    payload = dict(metadata or {})
    landscape_id = payload.pop("landscape_id", None)
    if landscape_id is None:
        raise ValueError(
            "Pickle compatibility export requires metadata['landscape_id'] for landscape-store v1 ingestion."
        )

    normalized = _normalize_bundle_metadata(payload)
    protein_gene = normalized["protein_gene"]
    assay_type = normalized["assay_type"]
    version = normalized["version"]
    if not protein_gene or not assay_type or not version:
        raise ValueError(
            "Pickle compatibility export requires metadata with protein_gene, assay_type, and version."
        )

    alphabet = _collect_global_alphabet(landscape.sequences)
    return {
        "landscape_id": str(landscape_id),
        "dataset_name": normalized["dataset_name"],
        "source_name": normalized["source_name"],
        "protein_gene": protein_gene,
        "assay_type": assay_type,
        "organism": normalized["organism"],
        "sequence_length": len(landscape.sequences[0]) if landscape.sequences else None,
        "alphabet": alphabet,
        "molecule_type": _infer_global_molecule_type(landscape.sequences),
        "available_fitness_layers": sorted(landscape.fitness_layers.keys()),
        "default_active_layer": getattr(landscape, "active_layer_name", None),
        "version": version,
        "serialization_format_version": PICKLE_FORMAT_VERSION,
        "tags": normalized["tags"],
        "metadata": normalized["metadata"],
        "provenance": normalized["provenance"],
    }


def _normalize_tags(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        items = [value]
    else:
        items = list(value)
    tags = sorted({str(item).strip().lower() for item in items if str(item).strip()})
    return tags


def _pop_alias(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload:
            value = payload.pop(key)
            return None if value is None else str(value)
    return None


def _coerce_scalar_token(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def _infer_scalar_kind(values: Sequence[Any]) -> str:
    non_null = [value for value in values if value is not None]
    if not non_null:
        return "string"
    if all(isinstance(value, bool) for value in non_null):
        return "bool"
    if all(isinstance(value, int) and not isinstance(value, bool) for value in non_null):
        return "int"
    if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in non_null):
        return "float"
    return "string"


def _infer_attribute_codec(values: Sequence[Any]) -> str:
    non_null = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (Mapping, list, tuple, set, np.ndarray)):
            return "json"
        non_null.append(value)
    if not non_null:
        return "json"
    if all(isinstance(value, bool) for value in non_null):
        return "bool"
    if all(isinstance(value, int) and not isinstance(value, bool) for value in non_null):
        return "int"
    if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in non_null):
        return "float"
    if all(isinstance(value, str) for value in non_null):
        return "str"
    return "json"


def _build_attribute_series(values: Sequence[Any], *, codec: str) -> pd.Series:
    if codec == "bool":
        normalized = [
            pd.NA if value is None or (isinstance(value, float) and math.isnan(value)) else bool(value)
            for value in values
        ]
        return pd.Series(normalized, dtype="boolean")
    if codec == "int":
        normalized = [
            pd.NA if value is None or (isinstance(value, float) and math.isnan(value)) else int(value)
            for value in values
        ]
        return pd.Series(normalized, dtype="Int64")
    if codec == "float":
        normalized = [
            np.nan if value is None else float(value)
            for value in values
        ]
        return pd.Series(normalized, dtype=np.float64)
    if codec == "str":
        normalized = [None if value is None else str(value) for value in values]
        return pd.Series(normalized, dtype="string")
    if codec == "json":
        normalized = [
            None if value is None else _stable_json_dumps(_normalize_json(value), indent=None)
            for value in values
        ]
        return pd.Series(normalized, dtype="string")
    raise ValueError(f"Unsupported attribute codec {codec!r}")


def _decode_attribute_value(value: Any, *, codec: str) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if codec == "bool":
        return bool(value)
    if codec == "int":
        return int(value)
    if codec == "float":
        return float(value)
    if codec == "str":
        return str(value)
    if codec == "json":
        return json.loads(str(value))
    raise ValueError(f"Unsupported attribute codec {codec!r}")


def _sequence_lookup_key(sequence: BaseNumpySequence) -> tuple[Any, ...]:
    return tuple(_coerce_scalar_token(item) for item in np.asarray(sequence.to_array()).tolist())


def _normalize_sequence_id(sequence: BaseNumpySequence) -> str | None:
    value = getattr(sequence, "id", None)
    return None if value is None else str(value)


def _infer_global_molecule_type(sequences: Sequence[BaseNumpySequence]) -> str | None:
    values = {value for value in (_extract_molecule_type(seq) for seq in sequences) if value}
    if len(values) == 1:
        return next(iter(values))
    return None


def _extract_molecule_type(sequence: BaseNumpySequence) -> str | None:
    c3_sequence = getattr(sequence, "_c3_seq", None)
    if c3_sequence is None:
        return None
    moltype = getattr(c3_sequence, "moltype", None)
    if moltype is None:
        return None
    for attr in ("label", "name"):
        value = getattr(moltype, attr, None)
        if value:
            return str(value)
    return str(moltype)


def _collect_global_alphabet(sequences: Sequence[BaseNumpySequence]) -> list[Any]:
    values = []
    seen = set()
    for sequence in sequences:
        for item in getattr(sequence, "alphabet", []):
            normalized = _coerce_scalar_token(item)
            marker = _stable_json_dumps(normalized, indent=None)
            if marker not in seen:
                seen.add(marker)
                values.append(normalized)
    values.sort(key=lambda item: _stable_json_dumps(item, indent=None))
    return values


def _normalize_json(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _normalize_json(value.item())
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError("Non-finite floats are not supported in bundle JSON payloads.")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return _normalize_json(value.tolist())
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_json(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, set):
        normalized = [_normalize_json(item) for item in value]
        normalized.sort(key=lambda item: _stable_json_dumps(item, indent=None))
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [_normalize_json(item) for item in value]
    return str(value)


def _encode_portable_value(value: Any) -> dict[str, Any]:
    if isinstance(value, np.generic):
        return _encode_portable_value(value.item())
    if value is None:
        return {"type": "none"}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int) and not isinstance(value, bool):
        return {"type": "int", "value": value}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError("Node keys must be finite when exporting portable bundles.")
        return {"type": "float", "value": repr(value)}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    if isinstance(value, list):
        return {"type": "list", "value": [_encode_portable_value(item) for item in value]}
    if isinstance(value, tuple):
        return {"type": "tuple", "value": [_encode_portable_value(item) for item in value]}
    raise TypeError(f"Unsupported portable node key type: {type(value).__name__}")


def _decode_portable_value(payload: Mapping[str, Any]) -> Any:
    kind = payload["type"]
    if kind == "none":
        return None
    if kind == "bool":
        return bool(payload["value"])
    if kind == "int":
        return int(payload["value"])
    if kind == "float":
        return float(payload["value"])
    if kind == "str":
        return str(payload["value"])
    if kind == "list":
        return [_decode_portable_value(item) for item in payload["value"]]
    if kind == "tuple":
        return tuple(_decode_portable_value(item) for item in payload["value"])
    raise BundleValidationError(f"Unsupported portable value type tag {kind!r}")


def _portable_sort_key(value: Any) -> str | None:
    try:
        return _stable_json_dumps(_encode_portable_value(value), indent=None)
    except Exception:
        return None


def _stable_json_dumps(payload: Any, *, indent: int | None = 2) -> str:
    separators = (",", ":") if indent is None else (",", ": ")
    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=True,
        indent=indent,
        separators=separators,
        allow_nan=False,
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(_stable_json_dumps(payload) + "\n", encoding="utf-8")


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise BundleValidationError(f"Failed to parse JSON file {path}") from exc


def _write_npy(path: Path, array: np.ndarray) -> None:
    with path.open("wb") as handle:
        np.save(handle, np.asarray(array), allow_pickle=False)


def _write_parquet(frame: pd.DataFrame, path: Path) -> str:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        payload = _frame_to_json_payload(frame)
        _write_json(path, payload)
        return TABULAR_STORAGE_JSON

    table = pa.Table.from_pandas(frame, preserve_index=False)
    schema_meta = table.schema.metadata or {}
    if schema_meta:
        table = table.replace_schema_metadata(None)
    pq.write_table(
        table,
        path,
        compression="NONE",
        use_dictionary=False,
        write_statistics=False,
    )
    return TABULAR_STORAGE_PARQUET


def _read_parquet(path: Path, *, storage_backend: str | None = None) -> pd.DataFrame:
    backend = storage_backend or _detect_tabular_storage_backend(path)
    if backend == TABULAR_STORAGE_JSON:
        payload = _load_json(path)
        return _frame_from_json_payload(payload)

    if backend != TABULAR_STORAGE_PARQUET:
        raise BundleValidationError(f"Unsupported tabular storage backend {backend!r}")

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - dependency failure
        raise ModuleNotFoundError(
            "Portable landscape bundle contains native parquet payloads, but no parquet engine is installed."
        ) from exc
    table = pq.read_table(path)
    return table.to_pandas()


def _detect_tabular_storage_backend(path: Path) -> str:
    with path.open("rb") as handle:
        prefix = handle.read(32).lstrip()
    if prefix.startswith(b"{"):
        return TABULAR_STORAGE_JSON
    return TABULAR_STORAGE_PARQUET


def _frame_to_json_payload(frame: pd.DataFrame) -> dict[str, Any]:
    columns = [str(column) for column in frame.columns]
    records = []
    for row in frame.itertuples(index=False, name=None):
        record = {}
        for column_name, value in zip(columns, row):
            record[column_name] = _normalize_tabular_cell(value)
        records.append(record)
    return {
        "storage_backend": TABULAR_STORAGE_JSON,
        "columns": [
            {"name": column_name, "dtype": str(frame[column_name].dtype)}
            for column_name in columns
        ],
        "records": records,
    }


def _frame_from_json_payload(payload: Mapping[str, Any]) -> pd.DataFrame:
    if payload.get("storage_backend") != TABULAR_STORAGE_JSON:
        raise BundleValidationError("Invalid JSON tabular payload.")
    column_specs = payload.get("columns")
    records = payload.get("records")
    if not isinstance(column_specs, list) or not isinstance(records, list):
        raise BundleValidationError("Malformed JSON tabular payload.")

    column_names = [str(spec["name"]) for spec in column_specs]
    frame = pd.DataFrame.from_records(records, columns=column_names)
    for spec in column_specs:
        column_name = str(spec["name"])
        dtype = str(spec.get("dtype", "object"))
        if column_name not in frame.columns:
            continue
        if dtype == "Int64":
            frame[column_name] = pd.Series(frame[column_name], dtype="Int64")
        elif dtype == "boolean":
            frame[column_name] = pd.Series(frame[column_name], dtype="boolean")
        elif dtype == "string":
            frame[column_name] = pd.Series(frame[column_name], dtype="string")
        elif dtype.startswith("float"):
            frame[column_name] = pd.to_numeric(frame[column_name], errors="coerce").astype(dtype)
        elif dtype.startswith("int"):
            frame[column_name] = pd.to_numeric(frame[column_name], errors="raise").astype(dtype)
    return frame


def _normalize_tabular_cell(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, np.floating):
        numeric = float(value)
        if math.isnan(numeric):
            return None
        return numeric
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (str, bool, int, float)):
        return value
    return _normalize_json(value)


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _relative_path(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _file_descriptor(path: Path) -> dict[str, Any]:
    return {
        "sha256": _sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _sha256_bytes(data: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(data)
    return digest.hexdigest()


def _class_path(obj_type: type) -> str:
    return f"{obj_type.__module__}.{obj_type.__name__}"


def _import_symbol(path: str) -> Any:
    module_name, _, attr = path.rpartition(".")
    if not module_name or not attr:
        raise BundleValidationError(f"Invalid import path {path!r}")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise BundleValidationError(f"Could not resolve symbol {path!r}") from exc


def _current_python_specifier() -> str:
    major = sys.version_info.major
    minor = sys.version_info.minor
    return f">={major}.{minor},<{major}.{minor + 1}"


def _safe_filename(value: str) -> str:
    text = str(value).strip()
    if not text:
        return "unnamed"
    safe = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", "."}:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("._") or "unnamed"


def _string_dtype_from_values(values: Sequence[Any]) -> str:
    max_length = max((len(str(value)) for value in values), default=1)
    return f"<U{max_length}"


def _write_deterministic_zip_from_directory(source_dir: Path, destination: Path) -> None:
    payloads = {}
    for file_path in sorted(source_dir.rglob("*")):
        if file_path.is_dir():
            continue
        payloads[_relative_path(file_path, source_dir)] = file_path.read_bytes()
    _write_deterministic_zip_from_bytes(payloads, destination)


def _write_deterministic_zip_from_bytes(payloads: Mapping[str, bytes], destination: Path) -> None:
    with ZipFile(destination, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(payloads.keys()):
            info = ZipInfo(filename=name, date_time=ZIP_TIMESTAMP)
            info.compress_type = ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payloads[name])
