from __future__ import annotations

import pickle
import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms.minors import equivalence_classes
from typing import TYPE_CHECKING, List, Union, Dict, Any, Iterable, Literal, Protocol, runtime_checkable, Hashable, Tuple, Mapping, Callable, Optional, Sequence
from dataclasses import dataclass
from .sequence import BaseNumpySequence, SoftSequence, make_sequence
from .graph import (
    create_diffusion_emb_graph,
    create_hamming_graph,
    create_tda_graph,
    create_knn_graph,
    _encode_multiallele,
    create_phylo_graph,
    create_evol_diffusion_graph,
    compute_edge_mutations_star,
)
from .fitness import (
    NumericFitness,
    CategoricalFitness,
    BaseFitnessLayer,
    ProbabilisticCategoricalFitness,
    FitnessModifierLike,
    apply_fitness_modifier,
)
from .annotation import AnnotationLayer
from abc import ABC, abstractmethod
from ..utils import _compute_embeddings_from_sequences, alignment_to_base_numpy_sequences
from .._optional import require_optional
import inspect
from collections import defaultdict
from pathlib import Path
import warnings
from .._const import PROT_20, ALPHABET_21
from xml.etree.ElementTree import Element, SubElement, ElementTree

if TYPE_CHECKING:
    import torch
    from cogent3.core.alignment import Alignment
    from cogent3.core.tree import PhyloNode
    from torch_geometric.data import Data


GraphCtor = Callable[..., nx.Graph]

@dataclass(frozen=True)
class _GraphRegistryItem:
    fn_name: str
    needs_embeddings: bool

    def resolve(self) -> GraphCtor:
        fn = globals().get(self.fn_name)
        if not callable(fn):
            raise RuntimeError(f"Graph constructor {self.fn_name!r} is not callable.")
        return fn

_GRAPH_REGISTRY: dict[str, _GraphRegistryItem] = {
    "hamming":         _GraphRegistryItem("create_hamming_graph", needs_embeddings=False),
    "knn":             _GraphRegistryItem("create_knn_graph", needs_embeddings=False),
    "tda":             _GraphRegistryItem("create_tda_graph", needs_embeddings=True),
    "diffusion":       _GraphRegistryItem("create_diffusion_emb_graph", needs_embeddings=True),
    "evol_diffusion":  _GraphRegistryItem("create_evol_diffusion_graph", needs_embeddings=True),
    "diffusion_evol":  _GraphRegistryItem("create_evol_diffusion_graph", needs_embeddings=True),
    # phylogenetic handled separately (alignment/ASR path)
}


@dataclass(frozen=True)
class AnnotationQueryResult:
    layer: str
    criteria: dict[str, Any]
    dataframe: pd.DataFrame
    sequence_indices: list[int]
    node_ids: list[Hashable]
    edges: list[tuple[Hashable, Hashable]]
    sequences: list[BaseNumpySequence]

    def to_subgraph(self, graph: nx.Graph, *, copy: bool = True) -> nx.Graph:
        sub = graph.subgraph(self.node_ids)
        return sub.copy() if copy else sub

def _resolve_embeddings_for_graph(sequences: list[BaseNumpySequence],
                                  graph_type: str,
                                  embeddings: Optional[np.ndarray],
                                  embedding_domain: Literal['plm', 'ohe'],
                                  *,
                                  model_name: str,
                                  batch_size: int,
                                  device: Optional[str],) -> Tuple[Optional[np.ndarray], dict]:
    """
    Helper function to resolve embeddings for a graph type.

    Parameters
    ----------
    sequences : list[BaseNumpySequence]
        List of sequences to compute embeddings for.
    graph_type : str
        The type of graph to create (e.g., 'hamming', 'knn', 'tda',
        'diffusion').
    
    embeddings : Optional[np.ndarray]
        Pre-computed embeddings, if available. If `None`, embeddings
        will be computed.

    embedding_domain : Literal['plm', 'ohe']
        The domain of the embeddings. 'plm' for pre-trained language
        model embeddings, 'ohe' for one-hot encoded sequences.
    
    model_name : str
        The name of the pre-trained model to use for embeddings if
        `embedding_domain` is 'plm'.

    batch_size : int
        The batch size to use for computing embeddings if
        `embedding_domain` is 'plm'.

    device : Optional[str]
        The device to use for computing embeddings if
        `embedding_domain` is 'plm'. If `None`, defaults to the

    Returns 
    -------
    Tuple[Optional[np.ndarray], dict]
        Returns a tuple containing the embeddings and a dictionary
        with additional keyword arguments for the graph constructor.
    """
    reg = _GRAPH_REGISTRY.get(graph_type)
    if reg is None or not reg.needs_embeddings:
        return embeddings, {}

    if embeddings is not None:
        return embeddings, {"embeddings": embeddings}

    use_soft = embedding_domain == "plm" and any(isinstance(seq, SoftSequence) for seq in sequences)

    if embedding_domain == "plm":
        E = _compute_embeddings_from_sequences(
            sequences,
            model_name=model_name,
            batch_size=batch_size,
            device=device,
            embedding_mode="soft" if use_soft else "hard",
        )
        return E, {"embeddings": E}

    if embedding_domain == "ohe":
        E, _ = _encode_multiallele(sequences)
        return E, {"embeddings": E}

    raise ValueError(f"embedding_domain must be 'plm' or 'ohe', got {embedding_domain!r}")


def _prepare_embedding_store(
    embeddings: Mapping[str, np.ndarray] | np.ndarray | None,
    embedding_domain: str,
) -> tuple[np.ndarray | None, dict[str, np.ndarray]]:
    """
    Normalize embedding inputs into a dict keyed by domain.
    Returns the array corresponding to the requested domain (if any)
    alongside the full store for attachment.
    """
    if embeddings is None:
        return None, {}
    if isinstance(embeddings, np.ndarray):
        return embeddings, {embedding_domain: embeddings}
    store: dict[str, np.ndarray] = {
        str(domain): np.asarray(arr) for domain, arr in embeddings.items()
    }
    return store.get(embedding_domain), store


def _choose_active_embedding_domain(
    store: Mapping[str, np.ndarray],
    preferred: str | None,
    attach_embeddings: bool,
) -> str | None:
    if not attach_embeddings or not store:
        return None
    if preferred and preferred in store:
        return preferred
    return next(iter(store))


def _collect_auto_annotation_layers(graph: nx.Graph) -> dict[str, AnnotationLayer]:
    """
    Convert auto-annotation metadata stored on a graph into
    AnnotationLayer instances keyed by layer name.
    """
    specs = graph.graph.get("_auto_annotations")
    if not specs:
        return {}

    node_order = list(graph.nodes())
    layers: dict[str, AnnotationLayer] = {}

    for name, payload in specs.items():
        records = payload.get("records", {})
        metadata = payload.get("metadata") or {}

        keyed: dict[Hashable, dict[str, Any]] = {}
        columns: set[str] = set()

        for node, rec in records.items():
            clean = {str(k): v for k, v in rec.items()}
            keyed[node] = clean
            columns.update(clean.keys())

        if not columns:
            continue

        data = {col: [] for col in columns}
        for node in node_order:
            record = keyed.get(node, {})
            for col in columns:
                data[col].append(record.get(col))

        frame = pd.DataFrame(data)
        try:
            layer = AnnotationLayer(name=name, data=frame, metadata=metadata)
        except ValueError:
            continue
        layers[name] = layer

    return layers


def _merge_annotation_layers(
    base_layers: dict[str, AnnotationLayer] | None,
    auto_layers: dict[str, AnnotationLayer],
) -> dict[str, AnnotationLayer] | None:
    if not auto_layers:
        return base_layers
    merged = dict(base_layers) if base_layers else {}
    for name, layer in auto_layers.items():
        if name in merged:
            warnings.warn(
                f"Annotation layer '{name}' already provided; skipping auto-generated layer.",
                RuntimeWarning,
            )
            continue
        merged[name] = layer
    return merged
SeqKey = Union['BaseNumpySequence', str, Tuple]

class FitnessLandscape:
    """
    FitnessLandscape is a class that represents a fitness landscape
    constructed from a networkx graph. It allows for the analysis of
    fitness layers, sequences, and their relationships.

    Attributes
    ---------- 
    sequences : List[BaseNumpySequence]
        The sequences in the fitness landscape

    graph : nx.Graph
        The instantianted graph.
    
    embeddings : Mapping[str, np.ndarray] | np.ndarray | None, default=`None`
        Mapping from embedding domain to aligned embedding arrays. Plain
        numpy arrays are accepted for backwards compatibility.
    
    emb_arr_key : str, default=`'emb_arr'`
        The keyword embeddings are stored under.

    active_embedding_domain : str | None
        The domain key used when annotating graph nodes or exporting
        tensors. Defaults to the first available domain.

    annotation_layers : Dict[str, AnnotationLayer], optional
        User-supplied metadata layers aligned with the landscape sequences.
    """
    def __init__(self,
                 sequences: List[BaseNumpySequence],
                 graph: nx.Graph,
                 fitness_layers: Dict[str, BaseFitnessLayer] | None = None,
                 annotation_layers: Dict[str, AnnotationLayer] | None = None,
                 embeddings: Mapping[str, np.ndarray] | np.ndarray | None = None,
                 emb_arr_key: str = 'emb_arr',
                 active_embedding_domain: str | None = None,
                 embedding_metadata: Mapping[str, Mapping[str, Any]] | None = None,
                 _build_sequence_indexes: bool = True):
        
        if graph.is_directed():
            raise TypeError(
                "FitnessLandscape requires an undirected networkx graph."
            )

        # Initialize Core Attributes with pre-computed objects
        self.sequences = sequences
        self.graph = graph
        self.fitness_layers = fitness_layers if fitness_layers is not None else {}
        self.annotation_layers = (
            annotation_layers if annotation_layers is not None else {}
        )
        if embeddings is None:
            self.embeddings: dict[str, np.ndarray] = {}
        elif isinstance(embeddings, np.ndarray):
            key = active_embedding_domain or "default"
            self.embeddings = {key: embeddings}
        else:
            self.embeddings = {str(domain): np.asarray(arr) for domain, arr in embeddings.items()}
        if active_embedding_domain is not None and active_embedding_domain not in self.embeddings:
            raise KeyError(
                f"Active embedding domain {active_embedding_domain!r} not found in provided embeddings."
            )
        self._active_embedding_domain = (
            active_embedding_domain
            if active_embedding_domain is not None
            else (next(iter(self.embeddings), None))
        )
        self._embedding_metadata: dict[str, dict[str, Any]] = {
            str(domain): dict(meta) for domain, meta in (embedding_metadata or {}).items()
        }
        self._emb_arr_key = emb_arr_key

        # Finalize Setup and Annotate Graph
        # Safe canonical node ordering.
        self._node_order = list(graph.nodes())  
        if not _build_sequence_indexes and (self.fitness_layers or self.annotation_layers):
            raise ValueError(
                "_build_sequence_indexes=False is only valid without fitness or annotation layers."
            )
        self._seq_to_nodes = (
            self._build_seq_multimap() if _build_sequence_indexes else {}
        )  # duplicate-safe
        self._nodes_by_index = {i: n for i, n in enumerate(self._node_order)}  # 0..N-1 -> node key
        self._annotate_graph_nodes_with_fitness()
        self._annotate_graph_nodes_with_annotations()
        if self.get_embedding() is not None:
            self._annotate_graph_nodes_with_embeddings()
        self._records = self._build_sequence_index() if _build_sequence_indexes else {}
        if _build_sequence_indexes:
            self._enforce_unique_sequences()
        
        if self.fitness_layers:
            self._active_view_name = (
                'default' if 'default' in self.fitness_layers
                else next(iter(self.fitness_layers.keys()))
            )
        else:
            self._active_view_name = None

    def _build_seq_multimap(self) -> Dict[Tuple, List]:
        """
        Helper function to map sequence array tuple. Safe for
        duplicates. 

        Returns
        -------
        mm : Dict
            The sequnce array to node value mapping.
        """
        mm: dict[tuple, list] = {}
        for n, data in self.graph.nodes(data=True):
            arr = tuple(data['sequence'].to_array())
            mm.setdefault(arr, []).append(n)
        return mm
    
    def _build_sequence_index(self) -> Dict[Tuple, int]:
        """
        Keep first occurrence index for fast get_fitness.
        """
        idx = {}
        for i, seq in enumerate(self.sequences):
            key = tuple(seq.to_array())
            if key not in idx:
                idx[key] = i
        return idx
    
    def _index_map(self) -> Dict[Tuple, list[int]]:
        """
        Map sequence-array tuple -> [indices in self.sequences]
        (duplicate-safe).
        """
        m: dict[Tuple, list[int]] = {}
        for i, s in enumerate(self.sequences):
            key = tuple(s.to_array())
            m.setdefault(key, []).append(i)
        return m

    def _index_map_by_name(self) -> tuple[dict[str, list[int]], list[int]]:
        """
        Map sequence identifier -> [indices] and collect indices without ids.
        """
        mapping: dict[str, list[int]] = {}
        missing: list[int] = []
        for i, seq in enumerate(self.sequences):
            seq_id = getattr(seq, "id", None)
            if seq_id is None:
                missing.append(i)
                continue
            mapping.setdefault(str(seq_id), []).append(i)
        return mapping, missing

    def _ensure_embedding_state(self) -> None:
        """
        Backwards-compatible guard to ensure embeddings and active domain
        exist even for historical pickles or pre-refactor instances.
        """
        if not isinstance(self.embeddings, dict):
            if self.embeddings is None:
                self.embeddings = {}
            else:
                default_key = getattr(self, "_active_embedding_domain", None) or "default"
                self.embeddings = {default_key: np.asarray(self.embeddings)}
        if not hasattr(self, "_active_embedding_domain"):
            self._active_embedding_domain = next(iter(self.embeddings), None)
        if not hasattr(self, "_embedding_metadata") or not isinstance(self._embedding_metadata, dict):
            self._embedding_metadata = {}
    
    @property
    def active_embedding_domain(self) -> str | None:
        """Domain key currently used for graph annotations and tensors."""
        self._ensure_embedding_state()
        return self._active_embedding_domain
    
    def set_active_embedding_domain(self, domain: str) -> None:
        """
        Set the active embedding domain used for downstream exports.
        """
        self._ensure_embedding_state()
        if domain not in self.embeddings:
            raise KeyError(f"Embedding domain {domain!r} is not available.")
        self._active_embedding_domain = domain
    
    @property
    def embedding_metadata(self) -> dict[str, dict[str, Any]]:
        """Per-domain metadata describing how embeddings were produced."""
        self._ensure_embedding_state()
        return self._embedding_metadata
    
    def get_embedding_metadata(self, domain: str | None = None) -> dict[str, Any] | None:
        """
        Retrieve embedding provenance for a given domain (or the active domain).
        """
        self._ensure_embedding_state()
        key = domain if domain is not None else self._active_embedding_domain
        if key is None:
            return None
        return self._embedding_metadata.get(key)
    
    @property
    def embedding_model(self) -> str | None:
        """
        Convenience property returning the model identifier for the active embeddings.
        """
        meta = self.get_embedding_metadata()
        if meta is None:
            return None
        return meta.get("model_name")
    
    def get_embedding(self, domain: str | None = None) -> np.ndarray | None:
        """
        Retrieve the embedding array for the requested domain.
        """
        self._ensure_embedding_state()
        if not self.embeddings:
            return None
        key = domain if domain is not None else self._active_embedding_domain
        if key is None and self.embeddings:
            # If no active domain is set but embeddings exist, default to the first
            key = next(iter(self.embeddings))
            self._active_embedding_domain = key
        return self.embeddings.get(key)
    
    def _normalize_seq_key(self, k: SeqKey) -> Tuple:
        """
        Normalize a sequence-like key to a tuple of symbols (hashable).
        """
        if hasattr(k, "to_array"):
            return tuple(k.to_array())
        if isinstance(k, str):
            dtype = self.sequences[0].to_array().dtype
            return tuple(np.array(list(k)).astype(dtype))
        if isinstance(k, (tuple, list, np.ndarray)):
            return tuple(list(k))
        raise TypeError(f"Unsupported sequence key type: {type(k)}")
    
    # Annotation methods.
    def _annotate_graph_nodes_with_fitness(self):
        """
        Helper function to add all fitness layer data to graph nodes.
        """
        if not self.graph or not self.fitness_layers:
            return
            
        for name, layer in self.fitness_layers.items():
            # Raise error if sequences are missing labels
            layer._validate_length(len(self.sequences), name=f"during annotation ({name})")

        for i, seq in enumerate(self.sequences):
            key = tuple(seq.to_array())
            nodes = self._seq_to_nodes.get(key, [])
            if not nodes:
                # Skip quietly, or raise error?
                continue
            
            for node in nodes:
                for name, layer in self.fitness_layers.items():
                    self.graph.nodes[node][f"fitness_{name}"] = layer.get_value(i)

    def _annotate_graph_nodes_with_annotations(self) -> None:
        """
        Helper function to add annotation layer data to graph nodes.
        """
        if not self.graph or not self.annotation_layers:
            return

        for name, layer in self.annotation_layers.items():
            layer.validate_length(len(self.sequences), context=f"during annotation ({name})")
            self._apply_annotation_layer(layer)

    def _apply_annotation_layer(self, layer: AnnotationLayer) -> None:
        """
        Attach a single annotation layer to graph nodes.
        """
        if not self.graph:
            return

        for idx, seq in enumerate(self.sequences):
            key = tuple(seq.to_array())
            nodes = self._seq_to_nodes.get(key, [])
            if not nodes:
                continue
            record = layer.get_record(idx)
            for node in nodes:
                annotations = self.graph.nodes[node].setdefault("annotations", {})
                annotations[layer.name] = dict(record)

    def _prepare_annotation_frame(
        self,
        data: pd.DataFrame | Mapping[Any, Any] | Sequence[Any],
        *,
        map_by: Literal["index", "sequence", "name"],
        allow_missing: bool,
    ) -> pd.DataFrame:
        if data is None:
            raise ValueError("`data` must be provided when constructing an annotation layer.")

        if map_by not in {"index", "sequence", "name"}:
            raise ValueError(f"Unsupported `map_by` option: {map_by!r}")

        frame = None
        if map_by == "index":
            frame = self._prepare_annotation_frame_index(data)

        if frame is not None:
            return frame.reset_index(drop=True)

        return self._prepare_annotation_frame_keyed(
            data,
            map_by=map_by,
            allow_missing=allow_missing,
        )

    def _prepare_annotation_frame_index(
        self,
        data: pd.DataFrame | Mapping[Any, Any] | Sequence[Any],
    ) -> pd.DataFrame | None:
        n = len(self.sequences)

        if isinstance(data, pd.DataFrame):
            if len(data) != n:
                raise ValueError(
                    f"`data` length {len(data)} does not match number of sequences {n} when map_by='index'."
                )
            return data.copy(deep=True)

        if isinstance(data, Mapping):
            if not data:
                raise ValueError("Annotation `data` mapping is empty.")
            sample = next(iter(data.values()))
            if isinstance(sample, Mapping):
                return None
            frame = pd.DataFrame(data)
            if len(frame) != n:
                raise ValueError(
                    f"`data` length {len(frame)} does not match number of sequences {n} when map_by='index'."
                )
            return frame

        if isinstance(data, (list, tuple)):
            if len(data) != n:
                raise ValueError(
                    f"`data` length {len(data)} does not match number of sequences {n} when map_by='index'."
                )
            if not all(isinstance(row, Mapping) for row in data):
                raise TypeError("Sequence-based annotation data must contain mapping rows.")
            return pd.DataFrame(list(data))

        return None

    def _prepare_annotation_frame_keyed(
        self,
        data: pd.DataFrame | Mapping[Any, Any] | Sequence[Any],
        *,
        map_by: Literal["index", "sequence", "name"],
        allow_missing: bool,
    ) -> pd.DataFrame:
        pairs = list(self._iter_annotation_pairs(data))
        if not pairs:
            raise ValueError("Annotation data is empty; no records were provided.")

        n = len(self.sequences)
        rows: list[dict[str, Any] | None] = [None] * n
        column_order: list[str] = []
        seen_columns: set[str] = set()

        if map_by == "index":
            def resolve(key: Any) -> list[int]:
                try:
                    idx = int(key)
                except (TypeError, ValueError) as exc:
                    raise KeyError(f"Invalid sequence index key {key!r}") from exc
                if idx < 0 or idx >= n:
                    raise KeyError(f"Sequence index {idx} is outside valid range [0, {n}).")
                return [idx]

        elif map_by == "sequence":
            seq_map = self._index_map()

            def resolve(key: Any) -> list[int]:
                normalized = self._normalize_seq_key(key)
                return seq_map.get(normalized, [])

        else:  # map_by == "name"
            name_map, missing = self._index_map_by_name()
            if missing and not allow_missing:
                raise ValueError(
                    "Cannot attach annotations by sequence name: some sequences lack identifiers."
                )

            def resolve(key: Any) -> list[int]:
                if isinstance(key, BaseNumpySequence):
                    key = getattr(key, "id", None)
                if key is None:
                    raise KeyError("Annotation key does not provide a sequence identifier.")
                return name_map.get(str(key), [])

        assigned: set[int] = set()

        for raw_key, record_obj in pairs:
            record = self._coerce_annotation_record(record_obj)
            for column in record.keys():
                if column not in seen_columns:
                    seen_columns.add(column)
                    column_order.append(column)

            indices = resolve(raw_key)
            if not indices:
                if allow_missing:
                    continue
                raise KeyError(
                    f"Annotation key {raw_key!r} could not be matched using map_by='{map_by}'."
                )

            for idx in indices:
                if idx in assigned:
                    raise ValueError(f"Annotation values already assigned for sequence index {idx}.")
                rows[idx] = dict(record)
                assigned.add(idx)

        if not column_order:
            raise ValueError("Annotation records must contain at least one column.")

        filled_rows: list[dict[str, Any]] = []
        for idx, row in enumerate(rows):
            if row is None:
                if not allow_missing:
                    raise ValueError(
                        f"No annotation provided for sequence index {idx}; "
                        "set allow_missing=True to permit missing records."
                    )
                filled_rows.append({col: None for col in column_order})
            else:
                filled_rows.append({col: row.get(col) for col in column_order})

        return pd.DataFrame(filled_rows, columns=column_order)

    @staticmethod
    def _coerce_annotation_record(record: Any) -> dict[str, Any]:
        if isinstance(record, pd.Series):
            return record.to_dict()
        if isinstance(record, Mapping):
            return dict(record)
        raise TypeError(
            f"Annotation record must be a mapping or pandas Series; received {type(record).__name__}."
        )

    def _iter_annotation_pairs(
        self,
        data: pd.DataFrame | Mapping[Any, Any] | Sequence[Any],
    ) -> Iterable[tuple[Any, Any]]:
        if isinstance(data, pd.DataFrame):
            for key, row in data.iterrows():
                yield key, row
            return

        if isinstance(data, Mapping):
            for key, value in data.items():
                yield key, value
            return

        if isinstance(data, (list, tuple)):
            for item in data:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    raise TypeError(
                        "Sequence annotation data must contain (key, record) pairs when provided as a list."
                    )
                yield item[0], item[1]
            return

        raise TypeError(
            "Unsupported annotation data container. Expected DataFrame, mapping, or sequence of key-record pairs."
        )

    def _enforce_unique_sequences(self):
        """
        Helper function to enforce only unique sequences.
        """
        dupes = [k for k, v in self._seq_to_nodes.items() if len(v) > 1]
        if dupes:
            warnings.warn(f"Duplicate sequences detected for {len(dupes)} keys; "
                      f"downstream `attach()` policies will handle them.")
    
    def _annotate_graph_nodes_with_embeddings(self):
        """
        Helper to attach the stored embeddings to the graph nodes.
        """
        emb_array = self.get_embedding()
        if self.graph is None or emb_array is None:
            return
        if emb_array.shape[0] != len(self._node_order):
            raise ValueError("Embeddings rows != number of graph nodes; cannot annotate safely.")
        attrs = {node: {self._emb_arr_key: emb_array[i]}
                for i, node in enumerate(self._node_order)}
        nx.set_node_attributes(self.graph, attrs)

    # Validation method.
    def _validate_data_against_graph(self,
                                     sequences: List[BaseNumpySequence],
                                     fitness_layers: Dict[str, BaseFitnessLayer]):
        """
        Method to validate the provided sequences and fitness layers
        against the current graph structure. This ensures that the
        sequences match the nodes in the graph and that the fitness
        layers are consistent with the node attributes.

        Parameters
        ----------
        sequences : List[BaseNumpySequence]
            List of sequences to validate against the graph.
        fitness_layers : Dict[str, BaseFitnessLayer]
            Dictionary of fitness layers to validate against the
            graph.

        Raises
        ------
        ValueError
            If there is a mismatch between the sequences and the graph
            nodes, or if the fitness layers do not match the attributes
            of the graph nodes.
        
        """
        if len(sequences) != self.graph.number_of_nodes():
            raise ValueError(
                f"Data inconsistency: The number of provided sequences ({len(sequences)}) "
                f"does not match the number of nodes in the graph ({self.graph.number_of_nodes()})."
            )

        graph_sequences = {
            node: tuple(data['sequence'].to_array())
            for node, data in self.graph.nodes(data=True)
        }
        provided_sequences = {i: tuple(s.to_array()) for i, s in enumerate(sequences)}

        if len(graph_sequences) != len(provided_sequences) or \
           set(graph_sequences.values()) != set(provided_sequences.values()):
            raise ValueError(
                "Data inconsistency: The set of provided sequences does not match "
                "the set of sequences stored in the graph nodes."
            )

        seq_to_node_map = {data['sequence']: node 
                           for node, data in self.graph.nodes(data=True)}

        for i, seq in enumerate(sequences):
            node_idx = seq_to_node_map.get(tuple(seq.to_array()))
            if node_idx is None:

                continue

            graph_node_data = self.graph.nodes[node_idx]

            for layer_name, layer in fitness_layers.items():
                attribute_name = f"fitness_{layer_name}"
                
                if attribute_name not in graph_node_data:
                    raise ValueError(
                        f"Data inconsistency: Fitness layer '{layer_name}' exists in the "
                        f"provided dictionary but no corresponding '{attribute_name}' "
                        f"attribute was found on node {node_idx} in the graph."
                    )
                
                layer_value = layer.get_value(i)
                graph_value = graph_node_data[attribute_name]
                
                if layer_value != graph_value:
                    raise ValueError(
                        f"Data inconsistency for layer '{layer_name}' at sequence index {i} "
                        f"(node {node_idx}): The provided layer value ({layer_value}) does not "
                        f"match the graph attribute value ({graph_value})."
                    )

    @property
    def active_layer(self) -> BaseFitnessLayer:
        """
        Dynamic property to get the active fitness layer.
        """
        if self._active_view_name is None:
            raise ValueError("No active fitness layer. Use .view(layer_name) to set one.")
        return self.fitness_layers[self._active_view_name]
    

    #Fitness layer appending, modifying and viewing methods.

    def view(self,
             name: str) -> BaseFitnessLayer:
        """
        Retrieves a fitness layer and sets it as the new active view.
        Main entry point for accessing fitness layers.

        Parameters
        ----------
        name : str
            The name of the fitness layer to retrieve.
        
        Returns
        -------
        BaseFitnessLayer
            The fitness layer corresponding to the provided name.
        """
        if name not in self.fitness_layers:
            raise KeyError(f"Fitness layer '{name}' not found.")
        self._active_view_name = name
        return self.fitness_layers[name]
    
    def add(self,
            **kwargs):
        """
        Convenience function to expedite fitness layer construction via
        the `attach` method.
        """
        if 'layer' in kwargs and kwargs['layer'] is not None:
            raise ValueError("`.add` builds from values; use `.attach(layer=...)` to attach a ready layer.")
        return self.attach(**kwargs)

    def safe_layer_name(self, name: str, *, ensure_unique: bool = True) -> str:
        """
        Return a layer name that will not collide with existing fitness layers.
        """
        if not name:
            raise ValueError("Layer name must be non-empty.")
        base = str(name)
        if not ensure_unique or base not in self.fitness_layers:
            return base
        suffix = 1
        candidate = f"{base}_{suffix}"
        while candidate in self.fitness_layers:
            suffix += 1
            candidate = f"{base}_{suffix}"
        return candidate

    def attach(self,
            layer: BaseFitnessLayer | None = None,
            *,
            name: str = None,
            values = None,
            dtype: Literal['numeric','categorical'] = None,
            categories: list[str] = None,
            map_by: Literal['index','sequence'] = 'index',
            on_duplicates: Literal['error','first','all','aggregate'] = 'error',
            allow_missing: bool = False) -> None:
        """
        Method to attach a fitness layer to the landscape.
        
        Parameters
        ----------
        layer : BaseFitnessLayer, optional
            A pre-constructed fitness layer to attach. If provided,
            it overrides the other parameters (name, values, dtype,
            categories). If `None`, the other parameters must be
            provided.
        
        name : str, optional
            The name of the fitness layer to create. Required if
            `layer` is not provided.
        
        values : list, dict, or iterable, optional
            The values to use for the fitness layer. If `map_by` is
            'index', this should be a list of values aligned with the
            sequences. If `map_by` is 'sequence', this should be a
            mapping of sequence keys to values (e.g., dict or iterable
            of tuples). Required if `layer` is not provided.
        
        dtype : Literal['numeric', 'categorical'], optional
            The data type of the fitness layer. Must be 'numeric' or
            'categorical'. Required if `layer` is not provided.

        categories : list[str], optional
            The categories for a categorical fitness layer. Required if
            `dtype` is 'categorical' and `layer` is not provided.
        
        map_by : Literal['index', 'sequence'], default='index'
            How to map the `values` to sequences. If 'index', the
            `values` should be a list aligned with the sequences.
            If 'sequence', the `values` should be a mapping of
            sequence keys to values (e.g., dict or iterable of tuples).
        
        on_duplicates : Literal['error', 'first', 'all', 'aggregate'], default='error'
            How to handle duplicate sequences when mapping values.
            - 'error': Raise an error if duplicates are found.
            - 'first': Use the first value for duplicates.
            - 'all': Use the value for all duplicates.
            - 'aggregate': Merge values for duplicates (only for numeric).
        
        allow_missing : bool, default=False
            If `True`, allows sequences to not have a value assigned.
        """

        if layer is not None:
            if any(x is not None for x in (name, values, dtype, categories)):
                raise ValueError("Provide either `layer` or (name, values, dtype...), not both.")
            if len(layer.to_scalar()) != len(self.sequences):
                raise ValueError(
                    f"Cannot attach layer '{layer.name}': its length ({len(layer.to_scalar())}) "
                    f"does not match the number of sequences ({len(self.sequences)})."
                )
            layer_name = layer.name
            if layer_name in self.fitness_layers:
                raise ValueError(f"A layer with the name '{layer_name}' already exists.")
            self.fitness_layers[layer_name] = layer
            # annotate graph
            if self.graph:
                seq_to_node_map = {tuple(data['sequence'].to_array()): node_idx
                                for node_idx, data in self.graph.nodes(data=True)}
                for i, seq in enumerate(self.sequences):
                    node_idx = seq_to_node_map.get(tuple(seq.to_array()))
                    if node_idx is not None:
                        self.graph.nodes[node_idx][f"fitness_{layer_name}"] = layer.get_value(i)
            if self._active_view_name is None:
                self._active_view_name = layer_name
            return

        # Construct from values
        if name is None or values is None or dtype is None:
            raise ValueError("When not passing `layer`, you must provide name, values, and dtype.")

        n = len(self.sequences)

        # Resolve mapping by index
        if map_by == 'index':
            # Expect values to be a list aligned to sequences length
            if len(values) != n:
                raise ValueError(f"`values` length {len(values)} != number of sequences {n}")
            # Normalize to concrete layer
            if dtype == 'numeric':
                norm = [[v] if not isinstance(v, (list, tuple, np.ndarray)) else list(v) for v in values]
                new_layer = NumericFitness(name=name, values=norm)
            elif dtype == 'categorical':
                if categories is None:
                    categories = sorted(list(set(values)))
                new_layer = CategoricalFitness(name=name, values=list(values), categories=categories)
            else:
                raise ValueError("For probabilistic categories, use dtype='categorical' with `values`=probabilities and pass categories, or attach a ProbabilisticCategoricalFitness layer explicitly.")
            
            # Delegate to regular layer attachment.
            return self.attach(new_layer)

        # Resolve mapping by sequence key
        if map_by != 'sequence':
            raise ValueError(f"Unknown map_by: {map_by}")

        # Normalize `values` into a dict {Tuple(seq) -> value}
        if isinstance(values, Mapping):
            items = list(values.items())
        else:

            items = list(values)

        key_to_val = {self._normalize_seq_key(k): v for k, v in items}

        # Create a per-index container
        if dtype == 'numeric':
            idx_values: list[list[float]] = [[] for _ in range(n)]
        elif dtype == 'categorical':
            idx_values: list[Any] = [None] * n
        else:
            raise ValueError("dtype must be `numeric` or `categorical` here; pass a ready layer object otherwise.")

        # Build index map for duplicates.
        idx_map = self._index_map()

        # Private helper.
        def _apply_numeric(idx_list: list[int],
                           v):
            
            reps = v if isinstance(v, (list, tuple, np.ndarray)) else [float(v)]
            
            if on_duplicates == 'error' and len(idx_list) > 1:
                raise ValueError("Duplicate sequences found; set `on_duplicates` to `first`, `all`, or `aggregate`.")
            
            # Collect only first
            if on_duplicates == 'first':
                idx_values[idx_list[0]] = list(reps)
            
            # Collect all
            elif on_duplicates == 'all':
                for i in idx_list:
                    idx_values[i] = list(reps)
            
            # merge replicate lists across all matches
            elif on_duplicates == 'aggregate':
            
                merged = []
                for i in idx_list:
                    merged.extend(reps)
                for i in idx_list:
                    idx_values[i] = list(merged)
            else:
                raise ValueError(f"Unknown `on_duplicates` option: {on_duplicates}")

        # Private helper.
        def _apply_categorical(idx_list: list[int],
                               v):
            
            if on_duplicates == 'error' and len(idx_list) > 1:
                raise ValueError("Duplicate sequences found; set on_duplicates to 'first' or 'all'.")
            
            if on_duplicates == 'first':
                idx_values[idx_list[0]] = v
            
            elif on_duplicates == 'all':
                for i in idx_list:
                    idx_values[i] = v
            
            elif on_duplicates == 'aggregate':
                raise ValueError("on_duplicates='aggregate' is not supported for categorical.")
            
            else:
                raise ValueError(f"Unknown on_duplicates: {on_duplicates}")

        # Fill index containers
        seen = set()
        for key, v in key_to_val.items():
            idxs = idx_map.get(key, [])
            if not idxs:
                if allow_missing:
                    continue
                raise KeyError(f"Sequence {key} not found in landscape.")
            
            seen.add(key)
            
            if dtype == 'numeric':
                _apply_numeric(idxs, v)
            
            else:
                _apply_categorical(idxs, v)

        # If unfilled indices:
        if dtype == 'numeric':
            missing = [i for i, r in enumerate(idx_values) if len(r) == 0]
        
        else:
            missing = [i for i, r in enumerate(idx_values) if r is None]
        
        if missing and not allow_missing:
            raise ValueError(f"{len(missing)} sequences were not assigned a value. Use `allow_missing=True` to skip.")

        # Build the concrete layer
        if dtype == 'numeric':
            # For any unassigned (allow_missing=True), give NaN replicate so shape is valid
            idx_values = [r if r else [np.nan] for r in idx_values]
            new_layer = NumericFitness(name=name, values=idx_values)
        else:
            if categories is None:
                categories = sorted(list({v for v in idx_values if v is not None}))
            # Replace None with a placeholder category if allow_missing
            if allow_missing:
                if "__MISSING__" not in categories:
                    categories = categories + ["__MISSING__"]
                idx_values = [v if v is not None else "__MISSING__" for v in idx_values]
            new_layer = CategoricalFitness(name=name, values=idx_values, categories=categories)

        # Delegate to regular constructor.
        return self.attach(new_layer)

    def apply_fitness_modifier(
        self,
        modifier: FitnessModifierLike,
        *,
        source_layer: str | BaseFitnessLayer | None = None,
        output_name: str | None = None,
        attach: bool = True,
        ensure_unique_name: bool = True,
    ) -> BaseFitnessLayer:
        """
        Transform an existing fitness layer with a modifier and
        optionally attach the result to the landscape.

        Parameters
        ----------
        modifier :
            A callable or BaseFitnessModifier that returns a new
            BaseFitnessLayer when applied to an input layer.
        source_layer :
            Name or instance of the source fitness layer. Defaults to
            the active view if not provided.
        output_name :
            Optional name for the new layer. When omitted, the modifier
            decides the name. If ``ensure_unique_name`` is True, a
            non-colliding name is generated.
        attach :
            When True (default), the resulting layer is attached to the
            landscape and returned.
        ensure_unique_name :
            If True, generated names are made unique with
            ``safe_layer_name`` when a collision is detected.
        """
        if source_layer is None:
            if self._active_view_name is None:
                raise ValueError("No source_layer provided and no active view is set.")
            base_layer = self.view(self._active_view_name)
        elif isinstance(source_layer, str):
            base_layer = self.view(source_layer)
        elif isinstance(source_layer, BaseFitnessLayer):
            base_layer = source_layer
            base_layer._validate_length(len(self.sequences), name="source_layer")
        else:
            raise TypeError("source_layer must be None, a layer name, or a BaseFitnessLayer instance.")

        transformed = apply_fitness_modifier(base_layer, modifier, name=output_name)
        transformed._validate_length(len(self.sequences), name="modifier output")

        if not attach:
            return transformed

        final_name = self.safe_layer_name(transformed.name, ensure_unique=ensure_unique_name)
        transformed.name = final_name
        self.attach(layer=transformed)
        return transformed

    def detach(self,
               layer_name: str):
        """
        Detaches a fitness layer from the landscape.

        layer_name : str
            The layer key to remove.
        """
        if layer_name not in self.fitness_layers:
            raise KeyError(f"Layer '{layer_name}' not found in the landscape.")

        # Remove the layer from the dictionary
        del self.fitness_layers[layer_name]

        # If a graph exists, remove the corresponding node attributes
        if self.graph:
            attribute_name = f"fitness_{layer_name}"

        for i, seq in enumerate(self.sequences):
            key = tuple(seq.to_array())
            for node in self._seq_to_nodes.get(key, []):
                self.graph.nodes[node].pop(attribute_name, None)

        # If the detached layer was the active one, update the active view
        if self._active_view_name == layer_name:
            if self.fitness_layers:
                # Set the new active layer to the first available one
                self._active_view_name = next(iter(self.fitness_layers.keys()))
            else:
                # No layers lefts
                self._active_view_name = None

    # Annotation layer management

    def attach_annotation(
        self,
        layer: AnnotationLayer | None = None,
        *,
        name: str | None = None,
        data: pd.DataFrame | Mapping[Any, Any] | Sequence[Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        map_by: Literal["index", "sequence", "name"] = "index",
        allow_missing: bool = False,
    ) -> AnnotationLayer:
        """
        Attach an annotation layer to the landscape.

        Parameters
        ----------
        layer :
            Ready-made annotation layer. If provided, other keyword arguments
            must be omitted.
        name :
            Name for the new annotation layer when constructing from raw data.
        data :
            Columnar annotation data aligned to the sequence order.
        metadata :
            Optional metadata to store on the layer when constructing inline.
        map_by :
            Strategy for aligning the provided data to existing sequences.
            - `"index"`: data is ordered by sequence index or keyed by index.
            - `"sequence"`: keys refer to sequence objects, tuples, lists, or
              strings that can be normalized to the landscape sequences.
            - `"name"`: keys refer to sequence identifiers (``sequence.id``).
        allow_missing :
            Allow sequences to be missing annotations when constructing from a
            mapping. Missing records are filled with ``None``.
        """
        if layer is not None:
            if any(x is not None for x in (name, data, metadata)):
                raise ValueError("Provide either `layer` or (name, data, metadata), not both.")
        else:
            if name is None or data is None:
                raise ValueError("When not providing `layer`, both `name` and `data` are required.")
            frame = self._prepare_annotation_frame(
                data,
                map_by=map_by,
                allow_missing=allow_missing,
            )
            layer = AnnotationLayer(name=name, data=frame, metadata=metadata)

        if layer.name in self.annotation_layers:
            raise ValueError(f"An annotation layer named '{layer.name}' already exists.")

        layer.validate_length(len(self.sequences))
        self.annotation_layers[layer.name] = layer
        self._apply_annotation_layer(layer)
        return layer

    def get_annotation_layer(self, name: str) -> AnnotationLayer:
        if name not in self.annotation_layers:
            raise KeyError(f"Annotation layer '{name}' not found.")
        return self.annotation_layers[name]

    def detach_annotation(self, name: str) -> None:
        if name not in self.annotation_layers:
            raise KeyError(f"Annotation layer '{name}' not found.")

        del self.annotation_layers[name]

        if not self.graph:
            return

        for node in self.graph.nodes:
            annotations = self.graph.nodes[node].get("annotations")
            if not annotations:
                continue
            annotations.pop(name, None)
            if not annotations:
                self.graph.nodes[node].pop("annotations", None)

    def annotation_to_fitness(
        self,
        annotation: AnnotationLayer | str,
        *,
        field: str | None = None,
        name: str | None = None,
        dtype: Literal["categorical", "numeric"] = "categorical",
        categories: list[Any] | None = None,
        missing_category: Any = "__missing__",
        metadata: Mapping[str, Any] | None = None,
        attach: bool = False,
    ) -> BaseFitnessLayer:
        """
        Convert an annotation column into a fitness layer.

        Parameters
        ----------
        annotation : AnnotationLayer or str
            Layer instance or name to convert.
        field : str, optional
            Column to use. If omitted and the layer has a single column,
            that column is used.
        name : str, optional
            Name for the resulting fitness layer. Defaults to ``field`` or
            the annotation layer name.
        dtype : {"categorical", "numeric"}, default="categorical"
            Target fitness layer type.
        categories : list, optional
            Explicit categories for categorical conversion. When omitted,
            categories are inferred (including a placeholder for missing
            values when present).
        missing_category : Any, default="__missing__"
            Placeholder used for None/NaN entries. Set to ``None`` to keep
            ``None`` as a category.
        metadata : mapping, optional
            Metadata to attach to the new layer.
        attach : bool, default=False
            When True, attach the resulting fitness layer to the landscape.

        Returns
        -------
        BaseFitnessLayer
            The constructed fitness layer.
        """
        layer = self.get_annotation_layer(annotation) if isinstance(annotation, str) else annotation
        if not isinstance(layer, AnnotationLayer):
            raise TypeError("`annotation` must be an AnnotationLayer instance or name.")

        df = layer.to_dataframe(copy=True)
        if field is None:
            if df.shape[1] != 1:
                raise ValueError("Annotation layer has multiple columns; specify `field`.")
            field = df.columns[0]
        if field not in df.columns:
            raise KeyError(f"Field '{field}' not found in annotation layer '{layer.name}'.")

        values = df[field].tolist()

        def _is_missing(v: Any) -> bool:
            try:
                return pd.isna(v)
            except Exception:
                return False

        normalized: list[Any] = []
        missing_seen = False
        for v in values:
            if _is_missing(v):
                missing_seen = True
                normalized.append(missing_category)
            else:
                normalized.append(v)

        resolved_name = name or field or layer.name

        if dtype == "numeric":
            if missing_seen and missing_category is not None:
                raise ValueError("Cannot convert to numeric fitness with missing values present.")
            try:
                scalars = np.asarray(normalized, dtype=float).tolist()
            except Exception as exc:
                raise ValueError("Annotation values cannot be coerced to numeric.") from exc
            fitness_layer = NumericFitness.from_scalars(resolved_name, scalars, metadata=metadata)
        elif dtype == "categorical":
            def _unique_preserve(seq: Iterable[Any]) -> list[Any]:
                seen = set()
                out = []
                for x in seq:
                    if x not in seen:
                        seen.add(x)
                        out.append(x)
                return out

            if categories is None:
                categories = _unique_preserve(normalized if missing_seen else list(normalized))
            fitness_layer = CategoricalFitness(
                name=resolved_name,
                values=normalized,
                categories=categories,
                metadata=metadata,
            )
        else:
            raise ValueError(f"Unsupported dtype {dtype!r}; use 'categorical' or 'numeric'.")

        if attach:
            self.attach(fitness_layer)
        return fitness_layer

    def query_annotations(
        self,
        layer_name: str,
        criteria: Mapping[str, Any] | None = None,
        *,
        include_edges: bool = True,
    ) -> AnnotationQueryResult:
        layer = self.get_annotation_layer(layer_name)

        seq_indices = layer.matching_indices(criteria)
        frame = layer.query(criteria)
        frame.index.name = "sequence_index"

        sequences = [self.sequences[i] for i in seq_indices]

        seen_nodes: set[Hashable] = set()
        node_ids: list[Hashable] = []
        for idx in seq_indices:
            key = tuple(self.sequences[idx].to_array())
            for node in self._seq_to_nodes.get(key, []):
                if node not in seen_nodes:
                    seen_nodes.add(node)
                    node_ids.append(node)

        edges: list[tuple[Hashable, Hashable]] = []
        if include_edges and self.graph is not None and node_ids:
            subgraph = self.graph.subgraph(node_ids)
            edges = list(subgraph.edges())

        return AnnotationQueryResult(
            layer=layer_name,
            criteria=dict(criteria) if criteria else {},
            dataframe=frame,
            sequence_indices=seq_indices,
            node_ids=node_ids,
            edges=edges,
            sequences=sequences,
        )

    @property
    def active_layer_name(self) -> str | None:
        return getattr(self, "_active_view_name", None)

    def get_layer(self,
                  name: str,
                  *,
                  allow_active_default: bool = True):
        """
        Method to get return a layer. 

        Parameters
        ----------
        name : str
            The layer name. 
        
        allow_active_default : bool, default=`True`
            Boolean to include the active layer in be resolved by the
            method.
        
        Returns
        -------
        FitnessLayer
            The resolved fitness layer.
        """
        d = self.fitness_layers
        if name in d:
            return d[name]
        for lyr in d.values():
            if getattr(lyr, "name", None) == name:
                return lyr
        if allow_active_default and name == "default":
            active = self.active_layer_name
            if active and active in d:
                return d[active]

        raise KeyError(f"Layer '{name}' not found. Available keys: {list(d.keys())}; "
                       f"active={self.active_layer_name!r}")




    def get_components(self) -> list["FitnessLandscape"]:
        """
        Split the landscape into connected components.

        Returns
        -------
        List[FitnessLandscape]
            Landscapes corresponding to each connected component, ordered
            from largest to smallest. All fitness and annotation layers are
            restricted to the sequences present in each component.
        """
        if self.graph is None:
            return [self]

        if isinstance(self.graph, nx.DiGraph):
            components_iter = nx.weakly_connected_components(self.graph)
        else:
            components_iter = nx.connected_components(self.graph)

        components = sorted((set(comp) for comp in components_iter), key=len, reverse=True)
        node_index_map = {node: idx for idx, node in enumerate(self._node_order)}
        self._ensure_embedding_state()
        out: list[FitnessLandscape] = []

        for comp_nodes in components:
            ordered_nodes = [node for node in self._node_order if node in comp_nodes]
            comp_indices = [node_index_map[node] for node in ordered_nodes]
            comp_sequences = [self.sequences[i] for i in comp_indices]
            comp_graph = self.graph.subgraph(ordered_nodes).copy()
            comp_fitness = self._subset_fitness_layers(comp_indices)
            comp_annotations = self._subset_annotation_layers(comp_indices)
            comp_embeddings = (
                {
                    domain: emb[comp_indices].copy()
                    for domain, emb in self.embeddings.items()
                }
                if self.embeddings
                else None
            )

            out.append(
                FitnessLandscape(
                    sequences=comp_sequences,
                    graph=comp_graph,
                    fitness_layers=comp_fitness,
                    annotation_layers=comp_annotations,
                    embeddings=comp_embeddings,
                    emb_arr_key=self._emb_arr_key,
                    active_embedding_domain=self._active_embedding_domain,
                )
            )

        return out

    def quotient_landscape(
        self,
        partition: AnnotationLayer
        | str
        | Mapping[Hashable, Any]
        | Sequence[Any]
        | None = None,
        *,
        annotation_field: str | None = None,
        aggregation_function: Literal["mean", "median", "mode"]
        | Callable[[np.ndarray], Any] = "mean",
        aggregate_annotations: bool = True,
        annotation_delimiter: str = ";",
        aggregate_edge_attributes: bool = True,
        edge_attributes: Sequence[str] | None = None,
        edge_aggregation_function: Literal["mean", "median", "mode"]
        | Callable[[np.ndarray], Any]
        | None = None,
    ) -> "FitnessLandscape":
        """
        Collapse the landscape according to a node partition using
        :func:`networkx.algorithms.minors.quotient_graph`.

        Parameters
        ----------
        partition :
            Partition specification. When ``None``, an annotation layer must
            be supplied (by name or instance) via ``partition``. Accepted
            forms include:
            - AnnotationLayer instance or name (uses ``annotation_field`` to
              choose the column as the block label).
            - Mapping of node -> block label.
            - Sequence of block sets/lists/tuples of node ids.
            - Sequence of block labels aligned to the node order.
        annotation_field : str, optional
            Column within an annotation layer to use when deriving the
            partition. Required when the layer has multiple columns.
        aggregation_function : {'mean', 'median', 'mode'} or callable, optional
            Aggregation applied to numeric values and category probabilities.
            Callable receives a 1-D NumPy array and must return a scalar or
            array-like.
        aggregate_annotations : bool, default=True
            When True, aggregates all annotation layers using the provided
            delimiter and attaches them to the quotient landscape.
        annotation_delimiter : str, default=";"
            Delimiter used when joining annotation values across merged nodes.
        aggregate_edge_attributes : bool, default=True
            Whether to aggregate edge attributes across merged edges.
        edge_attributes : Sequence[str], optional
            Explicit edge attribute keys to aggregate. When omitted, all edge
            attributes present in the original graph are considered.
        edge_aggregation_function : {'mean', 'median', 'mode'} or callable, optional
            Aggregation applied to numeric edge attributes. Defaults to
            ``aggregation_function``.

        Returns
        -------
        FitnessLandscape
            A new landscape whose nodes correspond to the partition blocks
            with fitness, annotations, and edge attributes aggregated.
        """
        if self.graph is None:
            raise ValueError("Landscape has no graph; cannot build a quotient graph.")

        agg_spec = aggregation_function
        edge_agg_spec = edge_aggregation_function or aggregation_function

        def _aggregate_scalar(values: Sequence[Any], spec) -> float:
            arr = np.asarray(values, dtype=float).ravel()
            arr = arr[~np.isnan(arr)]
            if arr.size == 0:
                return float("nan")
            if isinstance(spec, str):
                key = spec.lower()
                if key == "mean":
                    return float(np.nanmean(arr))
                if key == "median":
                    return float(np.nanmedian(arr))
                if key == "mode":
                    uniq, counts = np.unique(arr, return_counts=True)
                    return float(uniq[np.argmax(counts)])
                raise ValueError(f"Unknown aggregation_function '{spec}'.")
            res = spec(arr)
            res_arr = np.asarray(res, dtype=float)
            if res_arr.size == 0:
                return float("nan")
            if res_arr.ndim == 0:
                return float(res_arr)
            return float(res_arr.ravel()[0])

        def _aggregate_replicates(values: Sequence[Any], spec) -> list[float]:
            collated: list[float] = []
            for v in values:
                if isinstance(v, (list, tuple, np.ndarray)):
                    collated.extend(np.asarray(v, dtype=float).ravel().tolist())
                elif v is None:
                    collated.append(float("nan"))
                else:
                    collated.append(float(v))
            if not collated:
                return [float("nan")]
            if isinstance(spec, str):
                return [_aggregate_scalar(collated, spec)]
            res = spec(np.asarray(collated, dtype=float))
            arr = np.asarray(res, dtype=float)
            if arr.size == 0:
                return [float("nan")]
            if arr.ndim == 0:
                return [float(arr)]
            flat = arr.ravel()
            return flat.tolist() if flat.size else [float("nan")]

        def _aggregate_probabilities(matrix: np.ndarray, spec, num_categories: int) -> np.ndarray:
            if matrix.size == 0:
                return np.ones(num_categories, dtype=float) / num_categories
            if isinstance(spec, str) and spec.lower() == "mode":
                counts = np.zeros(num_categories, dtype=float)
                for row in matrix:
                    if np.isnan(row).all():
                        continue
                    winner = int(np.nanargmax(row))
                    counts[winner] += 1.0
                vec = counts
            else:
                vec = np.zeros(num_categories, dtype=float)
                for j in range(num_categories):
                    col = matrix[:, j]
                    col = col[~np.isnan(col)]
                    if col.size == 0:
                        vec[j] = 0.0
                    else:
                        vec[j] = _aggregate_scalar(col, spec)
            vec = np.nan_to_num(vec, nan=0.0)
            total = float(vec.sum())
            if total <= 0.0:
                return np.ones(num_categories, dtype=float) / num_categories
            return vec / total

        def _aggregate_embedding_block(block_indices: list[int]) -> dict[str, np.ndarray]:
            aggregated: dict[str, np.ndarray] = {}
            if not self.embeddings:
                return aggregated
            for domain, emb in self.embeddings.items():
                if emb.shape[0] != len(self._node_order):
                    continue
                sub = emb[np.asarray(block_indices)]
                if sub.size == 0:
                    aggregated[domain] = np.full(emb.shape[1], np.nan)
                    continue
                if isinstance(agg_spec, str):
                    key = agg_spec.lower()
                    if key == "mean":
                        aggregated[domain] = np.nanmean(sub, axis=0)
                        continue
                    if key == "median":
                        aggregated[domain] = np.nanmedian(sub, axis=0)
                        continue
                    if key == "mode":
                        res = []
                        for j in range(sub.shape[1]):
                            col = sub[:, j]
                            col = col[~np.isnan(col)]
                            if col.size == 0:
                                res.append(np.nan)
                            else:
                                uniq, counts = np.unique(col, return_counts=True)
                                res.append(float(uniq[np.argmax(counts)]))
                        aggregated[domain] = np.asarray(res, dtype=float)
                        continue
                res_vec = []
                for j in range(sub.shape[1]):
                    res_vec.append(_aggregate_scalar(sub[:, j], agg_spec))
                aggregated[domain] = np.asarray(res_vec, dtype=float)
            return aggregated

        def _is_missing(val: Any) -> bool:
            try:
                return bool(pd.isna(val))
            except Exception:
                return False

        def _partition_from_annotation(layer_like: AnnotationLayer | str) -> Mapping[Hashable, Any]:
            layer = self.get_annotation_layer(layer_like) if isinstance(layer_like, str) else layer_like
            if not isinstance(layer, AnnotationLayer):
                raise TypeError("partition must reference an AnnotationLayer when provided as a string.")
            df = layer.to_dataframe(copy=False)
            col = annotation_field
            if col is None:
                if df.shape[1] != 1:
                    raise ValueError("annotation_field is required when the annotation layer has multiple columns.")
                col = df.columns[0]
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in annotation layer '{layer.name}'.")
            labels = df[col].tolist()
            label_map: dict[Hashable, Any] = {}
            for idx, label in enumerate(labels):
                lbl: Any
                if _is_missing(label):
                    lbl = "__missing__"
                else:
                    lbl = label
                try:
                    hash(lbl)
                except Exception:
                    lbl = str(lbl)
                key = tuple(self.sequences[idx].to_array())
                for node in self._seq_to_nodes.get(key, []):
                    label_map[node] = lbl
            return label_map

        def _normalize_partition(part_obj: Any) -> list[set]:
            if part_obj is None:
                raise ValueError(
                    "A partition specification or an annotation layer name must be provided to build a quotient landscape."
                )
            if callable(part_obj):
                blocks = list(equivalence_classes(self.graph, part_obj))
            elif isinstance(part_obj, Mapping):
                if not part_obj:
                    raise ValueError("Partition mapping is empty.")
                sample_val = next(iter(part_obj.values()))
                if isinstance(sample_val, (list, tuple, set, frozenset)):
                    blocks = [set(v) for v in part_obj.values()]
                else:
                    grouped: dict[Any, set] = {}
                    for node, label in part_obj.items():
                        lab = label
                        try:
                            hash(lab)
                        except Exception:
                            lab = str(lab)
                        grouped.setdefault(lab, set()).add(node)
                    blocks = list(grouped.values())
            elif isinstance(part_obj, Sequence) and not isinstance(part_obj, (str, bytes)):
                if part_obj and all(
                    not isinstance(b, (list, tuple, set, frozenset)) for b in part_obj
                ):
                    if len(part_obj) != self.graph.number_of_nodes():
                        raise ValueError(
                            "Partition labels length does not match the number of graph nodes."
                        )
                    grouped: dict[Any, set] = {}
                    for node, label in zip(self._node_order, part_obj):
                        lab = label
                        try:
                            hash(lab)
                        except Exception:
                            lab = str(lab)
                        grouped.setdefault(lab, set()).add(node)
                    blocks = list(grouped.values())
                else:
                    blocks = [set(b) for b in part_obj]
            else:
                raise TypeError(
                    "Unsupported partition type. Provide an annotation layer, mapping, or sequence of blocks/labels."
                )
            blocks = [set(b) for b in blocks if b]
            all_nodes = set(self.graph.nodes())
            seen: set = set()
            for block in blocks:
                overlap = seen & block
                if overlap:
                    raise ValueError(
                        f"Nodes {overlap} appear in multiple partition blocks; partitions must be disjoint."
                    )
                seen.update(block)
            missing = all_nodes - seen
            blocks.extend([{n} for n in missing])
            return blocks

        part_source = partition
        if part_source is None:
            part_source = None
        if isinstance(part_source, (str, AnnotationLayer)):
            part_source = _partition_from_annotation(part_source)
        partition_sets = _normalize_partition(part_source)

        block_indices: list[list[int]] = []
        node_to_idx = {node: idx for idx, node in enumerate(self._node_order)}
        for block in partition_sets:
            indices = [node_to_idx[n] for n in block if n in node_to_idx]
            block_indices.append(indices)

        edge_keys: set[str] = set(edge_attributes) if edge_attributes else set()
        if aggregate_edge_attributes and not edge_attributes:
            for _, _, data in self.graph.edges(data=True):
                edge_keys.update(data.keys())

        def _edge_data(block_a: set, block_b: set) -> dict[str, Any]:
            if not aggregate_edge_attributes or not edge_keys:
                return {}
            values: dict[str, list[Any]] = {k: [] for k in edge_keys}
            if self.graph.is_directed():
                edges_iter = (
                    (u, v, d) for u, v, d in self.graph.edges(data=True) if u in block_a and v in block_b
                )
            else:
                edges_iter = (
                    (u, v, d)
                    for u, v, d in self.graph.edges(data=True)
                    if (u in block_a and v in block_b) or (u in block_b and v in block_a)
                )
            found = False
            for _, _, data in edges_iter:
                found = True
                for key in edge_keys:
                    if key in data:
                        values[key].append(data[key])
            if not found:
                return {}
            result: dict[str, Any] = {}
            for key, vals in values.items():
                if not vals:
                    continue
                numeric_like = all(
                    isinstance(v, (int, float, np.integer, np.floating, bool)) or _is_missing(v) for v in vals
                )
                if numeric_like:
                    result[key] = _aggregate_scalar(vals, edge_agg_spec)
                else:
                    seen_vals: set[str] = set()
                    ordered: list[str] = []
                    for v in vals:
                        if v is None:
                            continue
                        s = str(v)
                        if s not in seen_vals:
                            seen_vals.add(s)
                            ordered.append(s)
                    if ordered:
                        result[key] = annotation_delimiter.join(ordered)
            return result

        quotient_graph = nx.quotient_graph(
            self.graph,
            partition_sets,
            relabel=True,
            create_using=self.graph.__class__(),
            edge_data=_edge_data if aggregate_edge_attributes else None,
        )

        block_to_node: dict[int, set] = {idx: block for idx, block in enumerate(partition_sets)}

        #TODO: Update to SoftSequence consensus of all sequences in the agggregation.
        new_sequences: list[BaseNumpySequence] = []
        seq_indices_per_block: list[list[int]] = []
        for block_idx, block in block_to_node.items():
            indices = block_indices[block_idx]
            if not indices:
                raise ValueError("Partition produced an empty block with no associated sequences.")
            seq_indices_per_block.append(indices)
            representative = self.sequences[indices[0]]
            new_sequences.append(representative)
            quotient_graph.nodes[block_idx]["sequence"] = representative
            quotient_graph.nodes[block_idx]["source_nodes"] = tuple(block)

        aggregated_layers: dict[str, BaseFitnessLayer] = {}
        for name, layer in self.fitness_layers.items():
            metadata = dict(layer.metadata) if getattr(layer, "metadata", None) else None
            if isinstance(layer, NumericFitness):
                replicates: list[list[float]] = []
                for indices in seq_indices_per_block:
                    vals: list[Any] = [layer.get_value(idx) for idx in indices]
                    replicates.append(_aggregate_replicates(vals, agg_spec))
                aggregated_layers[name] = NumericFitness.from_replicates(
                    name=name, replicates=replicates, metadata=metadata
                )
            elif isinstance(layer, ProbabilisticCategoricalFitness):
                prob_rows: list[np.ndarray] = []
                for indices in seq_indices_per_block:
                    mat = np.asarray([layer.probabilities[i] for i in indices], dtype=float)
                    prob_rows.append(_aggregate_probabilities(mat, agg_spec, len(layer.categories)))
                aggregated_layers[name] = ProbabilisticCategoricalFitness.from_probabilities(
                    name=name,
                    probabilities=np.vstack(prob_rows),
                    categories=list(layer.categories),
                    metadata=metadata,
                )
            elif isinstance(layer, CategoricalFitness):
                cat_map = {c: i for i, c in enumerate(layer.categories)}
                prob_rows: list[np.ndarray] = []
                for indices in seq_indices_per_block:
                    mat = np.zeros((len(indices), len(layer.categories)), dtype=float)
                    for row, idx in enumerate(indices):
                        val = layer.get_value(idx)
                        if val not in cat_map:
                            raise KeyError(
                                f"Category '{val}' not found in layer '{layer.name}' categories."
                            )
                        mat[row, cat_map[val]] = 1.0
                    prob_rows.append(_aggregate_probabilities(mat, agg_spec, len(layer.categories)))
                aggregated_layers[name] = ProbabilisticCategoricalFitness.from_probabilities(
                    name=name,
                    probabilities=np.vstack(prob_rows),
                    categories=list(layer.categories),
                    metadata=metadata,
                )
            else:
                raise TypeError(
                    f"Unsupported fitness layer type {type(layer).__name__} for quotient aggregation."
                )

        aggregated_annotations: dict[str, AnnotationLayer] = {}
        if aggregate_annotations and self.annotation_layers:
            for name, layer in self.annotation_layers.items():
                df = layer.to_dataframe(copy=False)
                records: list[dict[str, Any]] = []
                for indices in seq_indices_per_block:
                    rec: dict[str, Any] = {}
                    for col in df.columns:
                        vals = [df.iloc[i][col] for i in indices if not _is_missing(df.iloc[i][col])]
                        if not vals:
                            rec[col] = None
                            continue
                        seen_vals: set[str] = set()
                        ordered: list[str] = []
                        for v in vals:
                            s = str(v)
                            if s not in seen_vals:
                                seen_vals.add(s)
                                ordered.append(s)
                        rec[col] = annotation_delimiter.join(ordered)
                    records.append(rec)
                metadata = dict(layer.metadata) if getattr(layer, "metadata", None) else None
                aggregated_annotations[name] = AnnotationLayer(
                    name=name, data=pd.DataFrame(records), metadata=metadata
                )

        aggregated_embeddings: dict[str, np.ndarray] = {}
        for block_idx, indices in enumerate(seq_indices_per_block):
            emb_block = _aggregate_embedding_block(indices)
            for domain, vec in emb_block.items():
                aggregated_embeddings.setdefault(domain, []).append(vec)
        active_domain = None
        if aggregated_embeddings:
            for domain, rows in aggregated_embeddings.items():
                aggregated_embeddings[domain] = np.vstack(rows)
            if self._active_embedding_domain in aggregated_embeddings:
                active_domain = self._active_embedding_domain
            else:
                active_domain = next(iter(aggregated_embeddings))

        return FitnessLandscape(
            sequences=new_sequences,
            graph=quotient_graph,
            fitness_layers=aggregated_layers if aggregated_layers else None,
            annotation_layers=aggregated_annotations if aggregated_annotations else None,
            embeddings=aggregated_embeddings if aggregated_embeddings else None,
            emb_arr_key=self._emb_arr_key,
            active_embedding_domain=active_domain,
        )

    def _subset_fitness_layers(self, indices: list[int]) -> dict[str, BaseFitnessLayer]:
        subset: dict[str, BaseFitnessLayer] = {}
        for name, layer in self.fitness_layers.items():
            metadata = dict(layer.metadata) if getattr(layer, "metadata", None) else None
            if isinstance(layer, NumericFitness):
                values = []
                for idx in indices:
                    val = layer.get_value(idx)
                    if isinstance(val, (list, tuple, np.ndarray)):
                        values.append(list(val))
                    else:
                        values.append([val])
                subset[name] = NumericFitness(name=name, values=values, metadata=metadata)
            elif isinstance(layer, ProbabilisticCategoricalFitness):
                probs = layer.probabilities[np.asarray(indices)]
                subset[name] = ProbabilisticCategoricalFitness(
                    name=name,
                    probabilities=probs,
                    categories=list(layer.categories),
                    metadata=metadata,
                )
            elif isinstance(layer, CategoricalFitness):
                vals = [layer.get_value(idx) for idx in indices]
                subset[name] = CategoricalFitness(
                    name=name,
                    values=vals,
                    categories=list(layer.categories),
                    metadata=metadata,
                )
            else:
                raise TypeError(
                    f"Unsupported fitness layer type {type(layer).__name__} for component extraction."
                )
        return subset

    def _subset_annotation_layers(self, indices: list[int]) -> dict[str, AnnotationLayer]:
        subset: dict[str, AnnotationLayer] = {}
        if not self.annotation_layers:
            return subset
        for name, layer in self.annotation_layers.items():
            df = layer.to_dataframe().iloc[indices].reset_index(drop=True)
            metadata = dict(layer.metadata) if getattr(layer, "metadata", None) else None
            subset[name] = AnnotationLayer(name=name, data=df, metadata=metadata)
        return subset

    def export_xgmml(
        self,
        filepath: str | Path,
        *,
        annotation_layers: list[str] | None = None,
        include_fitness: bool = True,
        include_annotations: bool = True,
    ) -> Path:
        """
        Export the current landscape to an XGMML file for Cytoscape.

        Parameters
        ----------
        filepath : str | Path
            Path to write the XGMML document.
        annotation_layers : list[str], optional
            Annotation layers to include; defaults to all layers.
        include_fitness : bool, default=True
            Whether to attach scalar values for each fitness layer.
        include_annotations : bool, default=True
            Whether to attach annotation columns.
        """
        if self.graph is None:
            raise ValueError("Landscape has no graph; cannot export XGMML.")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        directed = isinstance(self.graph, nx.DiGraph)
        root = Element(
            "graph",
            attrib={
                "label": getattr(self, "name", "FitnessLandscape"),
                "directed": "1" if directed else "0",
                "xmlns": "http://www.cs.rpi.edu/XGMML",
            },
        )

        selected_layers = (
            annotation_layers if annotation_layers is not None else list(self.annotation_layers.keys())
        )

        seq_index = {
            tuple(seq.to_array()): idx
            for idx, seq in enumerate(self.sequences)
        }

        def _write_attribute(parent, name, value):
            if value is None:
                return
            if isinstance(value, bool):
                attr_type = "boolean"
            elif isinstance(value, (int, np.integer)):
                attr_type = "integer"
            elif isinstance(value, (float, np.floating)):
                attr_type = "real"
            else:
                attr_type = "string"
                value = str(value)
            SubElement(parent, "att", attrib={"name": name, "value": str(value), "type": attr_type})

        fitness_cache: dict[str, np.ndarray] = {}
        if include_fitness:
            for name, layer in self.fitness_layers.items():
                fitness_cache[name] = layer.to_scalar()

        for node in self.graph.nodes():
            node_el = SubElement(root, "node", attrib={"id": str(node), "label": str(node)})
            seq = self.graph.nodes[node].get("sequence")
            seq_idx = None
            if seq is not None:
                arr = seq.to_array()
                _write_attribute(node_el, "sequence", "".join(map(str, arr)))
                _write_attribute(node_el, "sequence_id", getattr(seq, "id", None))
                seq_idx = seq_index.get(tuple(arr))

            if include_fitness and seq_idx is not None:
                for name, values in fitness_cache.items():
                    if seq_idx < len(values):
                        _write_attribute(node_el, f"fitness::{name}", float(values[seq_idx]))

            if include_annotations and seq_idx is not None:
                for layer_name in selected_layers:
                    layer = self.annotation_layers.get(layer_name)
                    if layer is None:
                        continue
                    record = layer.get_record(seq_idx)
                    for field, value in record.items():
                        if value is None:
                            continue
                        if isinstance(value, float) and np.isnan(value):
                            continue
                        _write_attribute(node_el, f"{layer_name}::{field}", value)

        for edge_id, (source, target, data) in enumerate(self.graph.edges(data=True)):
            edge_el = SubElement(
                root,
                "edge",
                attrib={"id": str(edge_id), "source": str(source), "target": str(target)},
            )
            for key, value in data.items():
                if key == "sequence":
                    continue
                _write_attribute(edge_el, key, value)

        ElementTree(root).write(filepath, encoding="UTF-8", xml_declaration=True)
        return filepath


    def to_graph_tensor(self, *, tokenizer: Any | str | None = "facebook/esm2_t6_8M_UR50D") -> 'Data':
        """
        Exports the entire fitness landscape to a PyTorch Geometric
        Data object.

        This method converts the landscape's graph structure, node
        features (from embeddings or sequences), and all associated
        fitness layers into a format suitable for graph machine
        learning with PyTorch Geometric.

        Parameters
        ----------
        tokenizer : huggingface tokenizer | str | None, default=`None`
            - If provided (as instance or model name), adds `token_ids` and `attention_mask`
              tensors to the returned Data, padded to the longest tokenized sequence.
            - If `None` or if tokenization is unavailable, these attributes are omitted.

        Returns
        -------
        torch_geometric.data.Data
            A PyG Data object with the following attributes:
            - x: Node features (embeddings or one-hot encoded
            sequences).
            - edge_index: Graph connectivity in COO format.
            - edge_attr: Edge weights, if they exist.
            - Additional attributes corresponding to each fitness
            layer, named after the layer.
            - token_ids (optional): LongTensor [N, Lmax] of token ids when tokenizer provided.
            - attention_mask (optional): LongTensor [N, Lmax] mask (1=real token, 0=pad).
        """
        torch = require_optional(
            "torch",
            extra="ml",
            purpose="PyTorch Geometric landscape export",
        )
        pyg_utils = require_optional(
            "torch_geometric.utils",
            extra="ml",
            purpose="PyTorch Geometric landscape export",
        )
        if not self.graph:
            raise ValueError("Graph not constructed.")
        pyg_data = pyg_utils.from_networkx(self.graph)
        emb_array = self.get_embedding()
        if emb_array is not None:
            pyg_data.x = torch.tensor(emb_array, dtype=torch.float32)
        else:
            x_tensor = torch.tensor(np.array([s.to_one_hot() for s in self.sequences]), dtype=torch.float32)
            pyg_data.x = x_tensor.view(len(self.sequences), -1)
        for name, layer in self.fitness_layers.items():
            setattr(pyg_data, name, layer.get_tensor())
        pyg_data.num_nodes = self.graph.number_of_nodes()

        # Optional: add tokenized sequences with padding
        if tokenizer is not None:
            if isinstance(tokenizer, str):
                transformers = require_optional(
                    "transformers",
                    extra="ml",
                    purpose="tokenized landscape export",
                )
                tok = transformers.AutoTokenizer.from_pretrained(tokenizer)
            else:
                tok = tokenizer

            if tok is not None:
                seq_texts: list[str] = []
                for s in self.sequences:
                    arr = [str(x) for x in s.to_array()]
                    arr = ['-' if x == 'gap' else x for x in arr]
                    seq_texts.append(' '.join(arr))

                input_id_list: list[torch.Tensor] = []
                max_len = 0
                for t in seq_texts:
                    enc = tok(t, add_special_tokens=True, return_tensors='pt')
                    ids = enc['input_ids'].squeeze(0).to(torch.long)
                    input_id_list.append(ids)
                    if ids.numel() > max_len:
                        max_len = int(ids.numel())

                N = len(input_id_list)
                token_ids = torch.zeros((N, max_len), dtype=torch.long)
                attn_mask = torch.zeros((N, max_len), dtype=torch.long)
                for i, ids in enumerate(input_id_list):
                    L = ids.numel()
                    token_ids[i, :L] = ids
                    attn_mask[i, :L] = 1
                pyg_data.token_ids = token_ids
                pyg_data.attention_mask = attn_mask

        return pyg_data

    def to_sequence_tensors(self,
                            *,
                            sequence_idx: Union[List[int], int] = None,
                            sequence: Union[List[str], str] = None,
                            tokenizer: Any | str | None = None,
                            feature_view: Literal["auto", "tokens", "embedding", "ohe"] = "auto",
                            include_embeddings: bool = False,
                            as_batch: bool = False) -> List[Dict[str, Any]] | Dict[str, Any]:
        """
        Exports the sequences and their fitness layers as a list of
        dictionaries containing tensors, or as a stacked batch when
        `as_batch=True`. Supports indexing by sequence and by int.

        Parameters
        ----------
        sequence_idx : List or int, default=`None`
            Indices of sequences to export as tensors. If `None`, all
            sequences are exported.
        
        sequence : List of str, default=`None`
            Sequence to export as tensors. If `None`, all sequences
            are exported.
        
        tokenizer : huggingface tokenizer | str | None, default=`None`
            - If a tokenizer instance or model name is provided, sequences are tokenized
              using the Hugging Face tokenizer and the returned 'sequence_tensor' is a
              1-D LongTensor of token ids (including special tokens as per the tokenizer).
            - If explicitly set to `None`, behavior matches current defaults: sequences are
              exported as embeddings when available, otherwise one-hot tensors per position.
        
        feature_view : {"auto", "tokens", "embedding", "ohe"}, default=`"auto"`
            Controls what is placed in `sequence_tensor`. `"auto"` prefers embeddings
            when available, otherwise tokenized text when a tokenizer is provided,
            otherwise one-hot encodings.

        include_embeddings : bool, default=`False`
            When True and embeddings are available, also include them under an
            `embedding` key in the returned structure (in addition to whichever
            `sequence_tensor` view is selected).

        as_batch : bool, default=`False`
            When True, returns a single dictionary with stacked tensors instead of
            a list of per-sequence dictionaries.

        Returns
        -------
        List[Dict[str, Any]] or Dict[str, Any]
            Per-sequence dictionaries (default) or a stacked batch when
            `as_batch=True`. Each record contains:
            - 'sequence_tensor': token ids, embeddings, or one-hot encodings.
            - 'fitness_tensors': dict of fitness layer tensors.
            - 'attention_mask': only when tokenized.
            - 'embedding': optional extra view when `include_embeddings=True`.
        """
        torch = require_optional(
            "torch",
            extra="ml",
            purpose="sequence tensor export",
        )
        target_indices: list[int] = []
        if sequence_idx is not None:
            target_indices = [sequence_idx] if isinstance(sequence_idx, int) else list(sequence_idx)
        elif sequence is not None:
            sequence_list = [sequence] if isinstance(sequence, str) else sequence
            dtype = self.sequences[0].to_array().dtype
            for seq_str in sequence_list:
                seq_tuple = tuple(np.array(list(seq_str)).astype(dtype))
                idx = self._records.get(seq_tuple)
                if idx is not None: target_indices.append(idx)
                else: raise ValueError(f"Sequence '{seq_str}' not found.")
        else:
            target_indices = list(range(len(self.sequences)))

        emb_array = self.get_embedding()
        emb_tensor: torch.Tensor | None = None
        if emb_array is not None:
            emb_tensor = torch.as_tensor(emb_array, dtype=torch.float32)
            if emb_tensor.shape[0] != len(self.sequences):
                raise ValueError(
                    f"Embeddings rows {emb_tensor.shape[0]} != number of sequences {len(self.sequences)}; "
                    "cannot export tensors safely."
                )

        # Decide which feature view to export.
        mode = feature_view
        if feature_view == "auto":
            if emb_tensor is not None:
                mode = "embedding"
            elif tokenizer is not None:
                mode = "tokens"
            else:
                mode = "ohe"
        if mode == "tokens" and tokenizer is None:
            raise ValueError("feature_view='tokens' requires a tokenizer.")
        if mode not in {"tokens", "embedding", "ohe"}:
            raise ValueError(f"Unsupported feature_view: {feature_view!r}")
        if mode == "embedding" and emb_tensor is None:
            raise ValueError(
                "feature_view='embedding' requires embeddings to be attached to the landscape."
            )

        fitness_dict = {
            name: layer.get_tensor()[target_indices] for name, layer in self.fitness_layers.items()
        }

        # Tokenization path (padded)
        if mode == "tokens":
            if isinstance(tokenizer, str):
                transformers = require_optional(
                    "transformers",
                    extra="ml",
                    purpose="tokenized sequence export",
                )
                tok = transformers.AutoTokenizer.from_pretrained(tokenizer)
            else:
                tok = tokenizer

            if tok is None:
                # Fallback to OHE/emb when tokenization failed
                mode = "embedding" if emb_tensor is not None else "ohe"
            else:
                ids_list: list[torch.Tensor] = []
                max_len = 0
                seq_texts: dict[int, str] = {}
                for i in target_indices:
                    s = self.sequences[i]
                    arr = [str(x) for x in s.to_array()]
                    arr = ['-' if x == 'gap' else x for x in arr]
                    seq_text = ''.join(arr)
                    spaced = ' '.join(list(seq_text))
                    seq_texts[i] = spaced
                    enc = tok(spaced, add_special_tokens=True, return_tensors='pt')
                    ids = enc['input_ids'].squeeze(0).to(torch.long)
                    ids_list.append(ids)
                    if ids.numel() > max_len:
                        max_len = int(ids.numel())

                token_ids = torch.zeros((len(ids_list), max_len), dtype=torch.long)
                attn_mask = torch.zeros((len(ids_list), max_len), dtype=torch.long)
                for row, ids in enumerate(ids_list):
                    L = ids.numel()
                    token_ids[row, :L] = ids
                    attn_mask[row, :L] = 1

                if as_batch:
                    batch: dict[str, Any] = {
                        "sequence_tensor": token_ids,
                        "attention_mask": attn_mask,
                        "fitness_tensors": fitness_dict,
                    }
                    if include_embeddings and emb_tensor is not None:
                        batch["embedding"] = emb_tensor[target_indices]
                    return batch

                out: list[dict[str, Any]] = []
                for row, i in enumerate(target_indices):
                    rec = {
                        "sequence_tensor": token_ids[row],
                        "attention_mask": attn_mask[row],
                        "fitness_tensors": {name: tensor[row] for name, tensor in fitness_dict.items()},
                    }
                    if include_embeddings and emb_tensor is not None:
                        rec["embedding"] = emb_tensor[i]
                    out.append(rec)
                return out

        # Non-token views: embeddings or one-hot encodings.
        def _base_tensor(i: int) -> torch.Tensor:
            if mode == "embedding":
                return emb_tensor[i]  # type: ignore[index]
            return torch.tensor(self.sequences[i].to_one_hot(), dtype=torch.float32)

        if as_batch:
            features = torch.stack([_base_tensor(i) for i in target_indices], dim=0)
            batch = {
                "sequence_tensor": features,
                "fitness_tensors": fitness_dict,
            }
            if include_embeddings and emb_tensor is not None and mode != "embedding":
                batch["embedding"] = emb_tensor[target_indices]
            return batch

        records: list[dict[str, Any]] = []
        for row, i in enumerate(target_indices):
            rec = {
                "sequence_tensor": _base_tensor(i),
                "fitness_tensors": {name: tensor[row] for name, tensor in fitness_dict.items()}
            }
            if include_embeddings and emb_tensor is not None:
                rec["embedding"] = emb_tensor[i]
            records.append(rec)
        return records

    def compute_plm_embeddings(
        self,
        *,
        domain: str = "plm",
        model_name: str = "facebook/esm2_t6_8M_UR50D",
        batch_size: int = 64,
        device: str | None = None,
        embedding_mode: Literal["hard", "soft"] | None = None,
        attach_to_graph: bool = True,
    ) -> np.ndarray:
        """
        Compute PLM embeddings for the landscape sequences and store them
        under ``self.embeddings[domain]``.

        Parameters
        ----------
        domain : str, default=`"plm"`
            Dictionary key under which the embeddings will be stored.
        model_name : str, default=`"facebook/esm2_t6_8M_UR50D"`
            HuggingFace model identifier for the PLM.
        batch_size : int, default=`64`
            Batch size used during embedding.
        device : str or None, default=`None`
            Device to run the model on. Defaults to GPU when available.
        embedding_mode : {'hard', 'soft'} or None, default=`None`
            Whether to embed discrete sequences or relaxed posteriors. When
            ``None``, defaults to `"soft"` if any sequence is a SoftSequence,
            otherwise `"hard"`.
        attach_to_graph : bool, default=`True`
            When True, sets the new domain as active and annotates graph
            nodes with the computed embeddings.

        Returns
        -------
        np.ndarray
            The computed embedding matrix.
        """
        self._ensure_embedding_state()
        mode = embedding_mode
        if mode is None:
            mode = "soft" if any(isinstance(seq, SoftSequence) for seq in self.sequences) else "hard"
        embeddings = _compute_embeddings_from_sequences(
            self.sequences,
            model_name=model_name,
            batch_size=batch_size,
            device=device,
            embedding_mode=mode,
        )
        self.embeddings[domain] = embeddings
        torch = require_optional(
            "torch",
            extra="ml",
            purpose="protein language-model embeddings",
        )
        self._embedding_metadata[domain] = {
            "model_name": model_name,
            "embedding_mode": mode,
            "device": device or ("cuda" if torch.cuda.is_available() else "cpu"),
            "batch_size": batch_size,
        }
        if self._active_embedding_domain is None:
            self._active_embedding_domain = domain
        if attach_to_graph:
            self._active_embedding_domain = domain
            self._annotate_graph_nodes_with_embeddings()
        return embeddings


    # Legacy methods for compatibility with old code.
    def get_fitness(self, sequence: BaseNumpySequence) -> float:
        """
        [Legacy] Method to retrieve the fitness of a sequence.

        Returns
        -------
        float
            Fitness value of the sequence. If the sequence is not
            found, returns the default value if provided, otherwise
            raises KeyError.
        """
        seq_index = self._records.get(tuple(sequence.to_array()))
        if seq_index is None:
            raise KeyError("Sequence not found in landscape.")
        return self.active_layer.to_scalar()[seq_index]

    def get_signal(self) -> np.ndarray:
        """
        [Legacy] Method to retrieve the graph signal vector.

        Returns
        -------
        np.ndarray
            Array of fitness values for each sequence in the landscape.
        """
        # Uses the new 'active_layer' property
        return self.active_layer.to_scalar()
    
    @classmethod
    def from_graph(cls,
                   graph: nx.Graph, **kwargs) -> 'FitnessLandscape':
        """
        Factory method to create a FitnessLandscape from an existing,
        annotated networkx graph.
        """

        node_list = list(graph.nodes())
        sequences = []
        raw_layer_data = defaultdict(list)

        for node in node_list:
            data = graph.nodes[node]
            if 'sequence' not in data:
                raise ValueError(f"Node {node} is missing 'sequence' attribute.")
            sequences.append(data['sequence'])
            for k, v in data.items():
                if k.startswith('fitness_'):
                    raw_layer_data[k[8:]].append(v)

        # length validation
        for name, values in raw_layer_data.items():
            if len(values) != len(node_list):
                raise ValueError(f"Layer '{name}' length {len(values)} != node count {len(node_list)}.")
        
        fitness_layers = {}
        for name, values in raw_layer_data.items():

            is_numeric = isinstance(values[0], (list, float, int, np.number))
            if is_numeric:
                numeric_values = [v if isinstance(v, list) else [v] for v in values]
                fitness_layers[name] = NumericFitness(name=name, values=numeric_values)
            else:
                all_categories = sorted(list(set(values)))
                fitness_layers[name] = CategoricalFitness(name=name, values=values, categories=all_categories)
        
        # Pop irrelevant keywords.
        kwargs.pop('graph_type', None)
        kwargs.pop('emb_nodes', None)
        
        # Call the simple constructor
        return cls(sequences=sequences,
                   graph=graph,
                   fitness_layers=fitness_layers,
                   **kwargs)
    
    
    @classmethod
    def build(cls,
              sequences: list[BaseNumpySequence],
              *,
              graph: str | nx.Graph = "hamming",
              fitness_layers: dict[str, BaseFitnessLayer] | None = None,
              annotation_layers: dict[str, AnnotationLayer] | None = None,
              embeddings: Mapping[str, np.ndarray] | np.ndarray | None = None,
              embedding_domain: Literal["plm", "ohe"] = "ohe",
              attach_embeddings: bool = True,
              emb_arr_key: str = "emb_arr",
              # PLM knobs (ignored for ohe/hamming)
              model_name: str = "facebook/esm2_t6_8M_UR50D",
              batch_size: int = 64,
              device: str | None = None,
              **graph_kwargs) -> "FitnessLandscape":
        """
        Constructor method for main entry to FitnessLandscape initialisation.

        Parameters
        ----------
        sequences : list[BaseNumpySequence]
            List of sequences to build the landscape from.
        
        graph : str or nx.Graph, default=`"hamming"`
            The graph type or an existing networkx graph. If a string,
            it should be one of the registered graph types (e.g.,
            `"hamming"`, `"knn"`, `"tda"`, `"diffusion"`, `"evol_diffusion"`).
        
        fitness_layers : dict[str, BaseFitnessLayer], optional
            Dictionary of fitness layers to attach to the landscape.

        annotation_layers : dict[str, AnnotationLayer], optional
            Dictionary of annotation layers aligned to the sequence order.
        
        embeddings : Mapping[str, np.ndarray] | np.ndarray | None, optional
            Pre-computed embeddings for the sequences. Plain numpy arrays
            are assumed to correspond to the provided `embedding_domain`.
            If `None`, embeddings will be computed when the selected graph
            requires them.
        
        embedding_domain : str, default=`"ohe"`
            The domain for embeddings. Options are:
            - `"plm"`: Protein language model embeddings.
            - `"ohe"`: One-hot encoded sequences.
        
        attach_embeddings : bool, default=`True`
            Whether to attach embeddings as node attributes in the graph.
        
        emb_arr_key : str, default=`"emb_arr"`
            The key under which embeddings will be stored in the graph
            nodes.
        
        model_name : str, default=`"facebook/esm2_t6_8M_UR50D"`
            The name of the model to use for PLM embeddings.
        
        batch_size : int, default=`64`
            Batch size for PLM embedding computation.
        
        device : str or None, default=`None`
            Device to use for PLM embedding computation (e.g., "cpu" or "cuda").
        
        graph_kwargs : dict
            Additional keyword arguments to pass to the graph constructor.
        
        Returns
        -------
        FitnessLandscape
            The constructed fitness landscape object.
        """
        graph_embeddings, embedding_store = _prepare_embedding_store(embeddings, embedding_domain)

        if isinstance(graph, nx.Graph):
            # annotate & return
            G = graph
        else:
            gtype = str(graph)
            if gtype == "phylogenetic":
                raise ValueError("Use FitnessLandscape.from_alignment(...) for phylogenetic graphs.")
            reg = _GRAPH_REGISTRY.get(gtype)
            if reg is None:
                raise ValueError(f"Unknown graph type {gtype!r}. Options: {list(_GRAPH_REGISTRY)}")

            # resolve embeddings only if needed
            graph_embeddings, extra = _resolve_embeddings_for_graph(
                sequences,
                gtype,
                graph_embeddings,
                embedding_domain,
                model_name=model_name,
                batch_size=batch_size,
                device=device,
            )
            if graph_embeddings is not None:
                embedding_store[embedding_domain] = graph_embeddings
            ctor = reg.resolve()
            G = ctor(sequences, **graph_kwargs, **extra)

        active_domain = _choose_active_embedding_domain(
            embedding_store, embedding_domain, attach_embeddings
        )
        return cls(sequences=sequences,
                   graph=G,
                   fitness_layers=fitness_layers,
                   annotation_layers=annotation_layers,
                   embeddings=(embedding_store or None),
                   emb_arr_key=emb_arr_key,
                   active_embedding_domain=active_domain)

    @classmethod
    def from_alignment(cls,
                       alignment: Alignment | Path,
                       *,
                       fitness_layers: dict[str, BaseFitnessLayer] | None = None,
                       annotation_layers: dict[str, AnnotationLayer] | None = None,
                       attach_embeddings: bool = True,
                       emb_arr_key: str = "emb_arr",
                       # PLM knobs for auto-embeddings on extant+ancestral
                       embedding_domain: Literal["plm", "ohe"] = "ohe",
                       model_name: str = "facebook/esm2_t6_8M_UR50D",
                       batch_size: int = 64,
                       device: str | None = None,
                       _compute_phylo_embeddings: bool = False,
                       **phylo_kwargs) -> "FitnessLandscape":
        """
        Constructor method to create a FitnessLandscape from an
        alignment or a path to an alignment file. Convenience wrapper
        around the phylogenetic graph constructor.

        Parameters
        ----------
        alignment : Alignment or Path
            The alignment object or path to an alignment file.
        
        fitness_layers : dict[str, BaseFitnessLayer], optional
            Dictionary of fitness layers to attach to the landscape.

        annotation_layers : dict[str, AnnotationLayer], optional
            Dictionary of annotation layers aligned to the sequence order.
        
        attach_embeddings : bool, default=`True`
            Whether to attach embeddings as node attributes in the graph.
        
        emb_arr_key : str, default=`"emb_arr"`
            The key under which embeddings will be stored in the graph
            nodes.
        
        embedding_domain : str, default=`"ohe"`
            The domain for embeddings. Options are:
            - `"plm"`: Protein language model embeddings.
            - `"ohe"`: One-hot encoded sequences.
        
        model_name : str, default=`"facebook/esm2_t6_8M_UR50D"`
            The name of the model to use for PLM embeddings.
        
        batch_size : int, default=`64`
            Batch size for PLM embedding computation.
        
        device : str or None, default=`None`
            Device to use for PLM embedding computation (e.g., "cpu" or "cuda").
        
        _compute_phylo_embeddings : bool, default=`False`
            Whether to compute embeddings for phylogenetic sequences.
        
        phylo_kwargs : dict
            Additional keyword arguments to pass to the phylogenetic graph constructor.
        
        Returns
        -------
        FitnessLandscape
            The constructed fitness landscape object.
        """

        cogent3 = require_optional(
            "cogent3",
            extra="phylogeny",
            purpose="phylogenetic landscape construction",
        )
        aln = cogent3.load_aligned_seqs(alignment) if isinstance(alignment, Path) else alignment
        G = create_phylo_graph(aln, **phylo_kwargs)
        seqs = [data["sequence"] for _, data in G.nodes(data=True)]

        embedding_store: dict[str, np.ndarray] = {}
        if _compute_phylo_embeddings:
            if embedding_domain == "plm":
                E = _compute_embeddings_from_sequences(
                    seqs, model_name=model_name, batch_size=batch_size, device=device
                )
            elif embedding_domain == "ohe":
                E, _ = _encode_multiallele(seqs)
            else:
                raise ValueError(f"embedding_domain must be 'plm' or 'ohe', got {embedding_domain!r}")
            embedding_store[embedding_domain] = E

        return cls(sequences=seqs,
                   graph=G,
                   fitness_layers=fitness_layers,
                   annotation_layers=annotation_layers,
                   embeddings=(embedding_store or None),
                   emb_arr_key=emb_arr_key,
                   active_embedding_domain=_choose_active_embedding_domain(
                       embedding_store, embedding_domain, attach_embeddings
                   ))

    @classmethod
    def from_phylogeny(cls,
                       tree: Union[str, Path, 'PhyloNode'],
                       fasta: Union[str, Path, Alignment],
                       *,
                       fitness_layers: dict[str, BaseFitnessLayer] | None = None,
                       annotation_layers: dict[str, AnnotationLayer] | None = None,
                       strip_gap_columns: bool = True,
                       emb_arr_key: str = "emb_arr",
                       moltype: str = "protein",
                       _compute_hamming_edges: bool = False,
                       replacement_matrix: Sequence[str] = ("LG",),
                       reconstruct_ancestral_states: bool = True,
                       model_fitting: bool = False,
                       phylo_backend: str = "cogent_nj",
                       _dist_calc: Literal['paralinear', 'pdist', 'hamming'] = 'pdist',
                       _log_progress: bool = False,
                       _nested_parallel: bool = False) -> "FitnessLandscape":
        """
        Construct a FitnessLandscape directly from a supplied phylogeny and
        an alignment containing both ancestral and extant sequences.

        Parameters
        ----------
        tree : str | Path | PhyloNode
            Newick string, file path, or cogent3 PhyloNode describing the tree.
            Every node must be named so that it can be matched to sequences.
        fasta : str | Path | Alignment
            FASTA alignment (path or Alignment object). If ancestral sequences
            are missing, they will be inferred using the supplied tree.
        strip_gap_columns : bool, default=True
            If True, remove alignment columns that contain a gap in any sequence
            before constructing the hard sequences (ensures PROT_20 alphabet).
            When False, the stored sequences retain gaps using the 21-character
            alphabet that includes ``"gap"``.
        moltype : str, default="protein"
            Moltype hint passed to cogent3 sequence constructors.
        _compute_hamming_edges : bool, default=True
            Whether to annotate edges with expected Hamming counts using the
            existing soft-alignment routine.
        replacement_matrix : Sequence[str], default=("LG",)
            Replacement model(s) passed to the ancestral reconstruction
            workflow when inference is required.
        reconstruct_ancestral_states : bool, default=True
            Whether to perform amino-acid ancestral state reconstruction.
            When False, internal nodes in the resulting graph are populated
            with placeholder sequences so that the graph topology can still
            be analysed.
        model_fitting : bool, default=False
            Whether to perform model selection during ancestral
            reconstruction when inference is triggered.
        phylo_backend : str, default="cogent_nj"
            Backend hint forwarded to the ancestral reconstruction engine.
        _dist_calc : {'paralinear', 'pdist', 'hamming'}, default='pdist'
            Distance metric used when the reconstruction backend computes
            pairwise distances (only relevant if inference is required).
        _log_progress : bool, default=False
            Enable verbose logging during ancestral sequence reconstruction.
        _nested_parallel : bool, default=False
            Forwarded to the edge annotation helper to allow nested parallelism
            when computing expected mutation statistics.

        Returns
        -------
        FitnessLandscape
            Landscape whose nodes follow the supplied phylogeny and whose
            sequences are taken directly from the FASTA records.
        """

        cogent3 = require_optional(
            "cogent3",
            extra="phylogeny",
            purpose="phylogenetic landscape construction",
        )
        cogent_tree = require_optional(
            "cogent3.core.tree",
            extra="phylogeny",
            purpose="phylogenetic landscape construction",
        )
        cogent_alignment = require_optional(
            "cogent3.core.alignment",
            extra="phylogeny",
            purpose="phylogenetic landscape construction",
        )
        phylo_node_type = cogent_tree.PhyloNode

        def _coerce_tree(obj: Union[str, Path, 'PhyloNode']):
            if isinstance(obj, phylo_node_type):
                return obj
            if hasattr(obj, 'children') and hasattr(obj, 'name'):
                return obj
            if isinstance(obj, Path):
                return cogent3.load_tree(str(obj))
            if isinstance(obj, str):
                candidate = Path(obj)
                if candidate.exists():
                    return cogent3.load_tree(str(candidate))
                return cogent3.load_tree(obj)
            raise TypeError("tree must be a Newick string, Path, or PhyloNode")

        alignment_type = cogent_alignment.Alignment

        def _coerce_alignment(obj: Union[str, Path, Alignment]) -> Alignment:
            if isinstance(obj, alignment_type):
                return obj
            if hasattr(obj, 'names') and hasattr(obj, 'get_gapped_seq'):
                return obj  # duck-typed Alignment-like object
            if isinstance(obj, Path):
                return cogent3.load_aligned_seqs(str(obj), moltype=moltype)
            if isinstance(obj, str):
                candidate = Path(obj)
                if candidate.exists():
                    return cogent3.load_aligned_seqs(str(candidate), moltype=moltype)
                return cogent3.load_aligned_seqs(obj, moltype=moltype)
            raise TypeError("fasta must be an Alignment, FASTA string, or Path")

        tree_obj = _coerce_tree(tree)
        alignment = _coerce_alignment(fasta)

        names = [str(n) for n in alignment.names]
        if not names:
            raise ValueError("Alignment is empty; no sequences were provided.")
        if len(names) != len(set(names)):
            raise ValueError("Alignment contains duplicate sequence identifiers.")

        legal = set(PROT_20)

        def _clean_char(ch: str, seq_name: str) -> str:
            if ch in {'-', '.'}:
                return '-'
            up = ch.upper()
            if up not in legal:
                raise ValueError(f"Non-canonical residue '{ch}' found in sequence '{seq_name}'.")
            return up

        gapped_strings: dict[str, str] = {}
        for raw_name in names:
            seq_str = str(alignment.get_gapped_seq(raw_name))
            cleaned = ''.join(_clean_char(ch, raw_name) for ch in seq_str)
            gapped_strings[raw_name] = cleaned

        aln_len = len(next(iter(gapped_strings.values())))
        if any(len(seq) != aln_len for seq in gapped_strings.values()):
            raise ValueError("Alignment sequences must all have the same length.")

        keep_mask: list[bool]
        if strip_gap_columns:
            keep_mask = [all(seq[pos] != '-' for seq in gapped_strings.values()) for pos in range(aln_len)]
            if not any(keep_mask):
                raise ValueError("All alignment columns contain gaps; cannot build ungapped sequences.")
        else:
            keep_mask = [True] * aln_len

        if strip_gap_columns:
            alignment_map_for_asr = {
                name: ''.join(ch for ch, keep in zip(seq, keep_mask) if keep)
                for name, seq in gapped_strings.items()
            }
            for name, trimmed in alignment_map_for_asr.items():
                if not trimmed:
                    raise ValueError(f"Sequence '{name}' is empty after removing gap columns.")
        else:
            alignment_map_for_asr = dict(gapped_strings)

        node_lookup: dict[str, Any] = {}

        def _dfs(node) -> None:
            node_name = getattr(node, 'name', None)
            if not node_name:
                raise ValueError("Encountered an unnamed node in the tree; all nodes must be labelled.")
            key = str(node_name)
            if key in node_lookup:
                raise ValueError(f"Duplicate node name '{key}' encountered in the tree.")
            node_lookup[key] = node
            for child in getattr(node, 'children', []) or []:
                _dfs(child)

        _dfs(tree_obj)

        provided_names = set(gapped_strings)
        missing = sorted(set(node_lookup) - provided_names)
        extra = sorted(provided_names - set(node_lookup))
        if extra:
            raise ValueError(f"Sequences provided without matching tree nodes: {', '.join(extra)}")

        if missing:
            tips = {
                name for name, node in node_lookup.items()
                if not (getattr(node, 'children', []) or [])
            }
            missing_tips = sorted(set(missing) & tips)
            if missing_tips:
                raise ValueError(
                    "Sequences are missing for tree tip nodes: " + ', '.join(missing_tips)
                )

            asr_alignment_map = {
                name: seq for name, seq in alignment_map_for_asr.items() if name in tips
            }
            if not asr_alignment_map:
                raise ValueError(
                    "Ancestral reconstruction requires at least one tip sequence; "
                    "none were provided after filtering internal nodes."
                )
            aln_for_asr = cogent3.make_aligned_seqs(asr_alignment_map, moltype=moltype)
            from ..phylo.phylogenetic_asr import ASRConstructor

            constructor = ASRConstructor(
                aln_for_asr,
                phylogenetic_tree=tree_obj,
                model_fitting=model_fitting,
                replacement_matrix=list(replacement_matrix),
                phylo_backend=phylo_backend,
                _dist_calc=_dist_calc,
                reconstruct_ancestral_states=reconstruct_ancestral_states,
                _log_progress=_log_progress,
            )

            graph = constructor.construct_dag(graph_type='undirected')

            # Stamp branch lengths from the supplied tree onto the inferred graph.
            for child_name, child_node in node_lookup.items():
                parent_node = getattr(child_node, 'parent', None)
                if parent_node is None:
                    continue
                parent_name = getattr(parent_node, 'name', None)
                if not parent_name:
                    continue
                if graph.has_edge(parent_name, child_name):
                    branch_length = getattr(child_node, 'length', None)
                    if branch_length is not None:
                        try:
                            graph[parent_name][child_name]['branch_length'] = float(branch_length)
                        except (TypeError, ValueError):
                            pass

            if _compute_hamming_edges and graph.number_of_edges() > 0:
                compute_edge_mutations_star(
                    graph,
                    _log_progress=_log_progress,
                    _nested_parallel=_nested_parallel,
                )

            node_order = list(graph.nodes())
            sequences = [graph.nodes[name]['sequence'] for name in node_order]

            return cls(sequences=sequences,
                       graph=graph,
                       fitness_layers=fitness_layers,
                       annotation_layers=annotation_layers,
                       embeddings=None,
                       emb_arr_key=emb_arr_key)

        seq_records: dict[str, dict[str, Any]] = {}
        for name, gapped in gapped_strings.items():
            gapped_seq = BaseNumpySequence(list(gapped),
                                           sequence_id=name,
                                           alphabet=ALPHABET_21,
                                           moltype=moltype)
            if strip_gap_columns:
                ungapped = alignment_map_for_asr[name]
                hard_seq = BaseNumpySequence.from_string(ungapped,
                                                         alphabet=PROT_20,
                                                         moltype=moltype,
                                                         sequence_id=name)
            else:
                hard_seq = gapped_seq
            seq_records[name] = {
                'sequence': hard_seq,
                'gapped_arr': gapped_seq.to_one_hot(),
            }

        edges: list[tuple[str, str, dict[str, float]]] = []
        for child_name, child_node in node_lookup.items():
            parent_node = getattr(child_node, 'parent', None)
            if parent_node is None:
                continue
            parent_name = getattr(parent_node, 'name', None)
            if not parent_name:
                raise ValueError(f"Parent of node '{child_name}' lacks a name; unable to create edge.")
            attr: dict[str, float] = {}
            branch_length = getattr(child_node, 'length', None)
            if branch_length is not None:
                try:
                    attr['branch_length'] = float(branch_length)
                except (TypeError, ValueError):
                    pass
            edges.append((str(parent_name), child_name, attr))

        G = nx.Graph()
        G.add_nodes_from(node_lookup.keys())
        for parent_name, child_name, attr in edges:
            G.add_edge(parent_name, child_name, **attr)

        for name, record in seq_records.items():
            G.nodes[name]['sequence'] = record['sequence']
            G.nodes[name]['gapped_arr'] = record['gapped_arr']

        if _compute_hamming_edges and G.number_of_edges() > 0:
            compute_edge_mutations_star(
                G,
                _log_progress=_log_progress,
                _nested_parallel=_nested_parallel,
            )

        node_order = list(G.nodes())
        sequences = [G.nodes[name]['sequence'] for name in node_order]

        return cls(sequences=sequences,
                   graph=G,
                   fitness_layers=fitness_layers,
                   annotation_layers=annotation_layers,
                   embeddings=None,
                   emb_arr_key=emb_arr_key)

    @classmethod
    def from_graph_annotated(cls, graph: nx.Graph, **kwargs) -> "FitnessLandscape":
        """
        Thin alias around  existing `from_graph` for parity with other APIs.
        """
        return cls.from_graph(graph, **kwargs)

    def save_bundle_dir(
        self,
        path: str | Path,
        *,
        metadata: Mapping[str, Any] | None = None,
        include_embeddings: bool = True,
        include_legacy_pickle: bool = False,
        overwrite: bool = False,
    ) -> Path:
        """
        Save the landscape in the canonical portable directory bundle format.
        """
        from ..io import save_bundle_dir as _save_bundle_dir

        return _save_bundle_dir(
            self,
            path,
            metadata=metadata,
            include_embeddings=include_embeddings,
            include_legacy_pickle=include_legacy_pickle,
            overwrite=overwrite,
        )

    @classmethod
    def load_bundle_dir(cls, path: str | Path) -> "FitnessLandscape":
        """
        Load a landscape from a canonical portable directory bundle.
        """
        from ..io import load_bundle_dir as _load_bundle_dir

        landscape = _load_bundle_dir(path)
        if not isinstance(landscape, cls):
            raise TypeError(
                f"Bundle contains {type(landscape).__name__}, which cannot be loaded as {cls.__name__}."
            )
        return landscape

    def export_lsbundle(
        self,
        path: str | Path,
        *,
        metadata: Mapping[str, Any] | None = None,
        backend: str = "portable",
        overwrite: bool = False,
    ) -> Path:
        """
        Export the landscape as an `.lsbundle` archive.
        """
        from ..io import export_lsbundle as _export_lsbundle

        return _export_lsbundle(
            self,
            path,
            metadata=metadata,
            backend=backend,
            overwrite=overwrite,
        )

    def save(self, filepath: Path):
        """Saves the FitnessLandscape object to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: Path):
        """Loads a FitnessLandscape object from a file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.get_fitness(self.sequences[idx])
    
    def __iter__(self):
        for seq in self.sequences:
            yield seq, self.get_fitness(seq)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(n_sequences={len(self.sequences)})"


def read_csv_landscape(path: str | Path,
                       *,
                       sequence_col: str = "sequence",
                       id_col: str | None = None,
                       alphabet: Iterable | None = None,
                       moltype: str | None = None,
                       graph: str | nx.Graph = "hamming",
                       
                       # layer parsing
                       numeric_layers: list[str] | None = None, # e.g., ["fitness", "score"]
                       replicate_prefixes: dict[str, list[str]] | None = None, # {"fitness": ["fitness.rep1","fitness.rep2"]}
                       categorical_layers: list[str] | None = None, # e.g., ["label"]
                       probabilistic_specs: dict[str, list[str]] | None = None, # {"label": ["label=A","label=B","label=C"]}
                       
                       # embeddings for graph if needed
                       embeddings: Mapping[str, np.ndarray] | np.ndarray | None = None,
                       embedding_domain: Literal["plm", "ohe"] = "ohe",
                       attach_embeddings: bool = True,
                       emb_arr_key: str = "emb_arr",
                       model_name: str = "facebook/esm2_t6_8M_UR50D",
                       batch_size: int = 64,
                       device: str | None = None) -> FitnessLandscape:
    """
    Function to initialise a FitnessLandscape from a CSV file.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file containing the landscape data.
    
    sequence_col : str, default=`"sequence"`
        The column name in the CSV that contains the sequences.
    
    id_col : str or None, default=`None`
        Optional column name for sequence IDs (not used in landscape).
    
    alphabet : Iterable, optional
        The alphabet to use for sequence encoding. If None, defaults to
        the standard alphabet for the specified moltype.
    
    moltype : str, optional
        The molecular type of the sequences (e.g., "protein", "dna").
    
    graph : str or nx.Graph, default=`"hamming"`
        The graph type or an existing networkx graph to use.
    
    numeric_layers : list[str] | None, optional
        List of numeric layer names to parse from the CSV.
    
    replicate_prefixes : dict[str, list[str]] | None, optional
        Dictionary mapping layer names to lists of replicate column names.
    
    categorical_layers : list[str] | None, optional
        List of categorical layer names to parse from the CSV.
    
    probabilistic_specs : dict[str, list[str]] | None, optional
        Dictionary mapping layer names to lists of probabilistic column names.
    
    embeddings : Mapping[str, np.ndarray] | np.ndarray | None, optional
        Pre-computed embeddings for the sequences. Plain numpy arrays are
        assumed to correspond to `embedding_domain`. If None, embeddings
        will be computed when the selected graph type requires them.
    
    embedding_domain : Literal["plm", "ohe"], default=`"ohe"`
        The domain for embeddings. Options are:
        - `"plm"`: Protein language model embeddings.
        - `"ohe"`: One-hot encoded sequences.
    
    attach_embeddings : bool, default=`True`
        Whether to attach embeddings as node attributes in the graph.
    
    emb_arr_key : str, default=`"emb_arr"`
        The key under which embeddings will be stored in the graph nodes.
    
    model_name : str, default=`"facebook/esm2_t6_8M_UR50D"`
        The name of the model to use for PLM embeddings.
    
    batch_size : int, default=`64`
        Batch size for PLM embedding computation.
    
    device : str or None, default=`None`
        Device to use for PLM embedding computation (e.g., "cpu" or "cuda").
    
    Returns
    -------
    FitnessLandscape
        The constructed fitness landscape object.
    """

    df = pd.read_csv(path)

    if sequence_col not in df.columns:
        raise ValueError(f"sequence_col '{sequence_col}' not found in CSV columns {list(df.columns)}")

    # build sequences
    seqs = [make_sequence(s, alphabet=alphabet, moltype=moltype) for s in df[sequence_col].tolist()]

    layers: dict[str, BaseFitnessLayer] = {}

    # numeric scalar columns
    if numeric_layers:
        for name in numeric_layers:
            if name not in df.columns:
                raise ValueError(f"Numeric layer column '{name}' not found")
            layers[name] = NumericFitness.from_scalars(name, df[name].to_numpy())

    # numeric replicate groups
    if replicate_prefixes:
        for name, cols in replicate_prefixes.items():
            for c in cols:
                if c not in df.columns:
                    raise ValueError(f"Replicate column '{c}' for layer '{name}' not found")
            reps = df[cols].to_numpy(dtype=float)  # shape (N, R)
            # convert rows to list[list]
            rep_lists = [row[~np.isnan(row)].tolist() if np.isnan(row).any() else row.tolist()
                         for row in reps]
            layers[name] = NumericFitness.from_replicates(name, rep_lists)

    # categorical single-column layers
    if categorical_layers:
        for name in categorical_layers:
            if name not in df.columns:
                raise ValueError(f"Categorical layer column '{name}' not found")
            vals = df[name].astype(str).tolist()
            layers[name] = CategoricalFitness.from_values(name, vals)

    # probabilistic layers (wide)
    if probabilistic_specs:
        for name, cols in probabilistic_specs.items():
            for c in cols:
                if c not in df.columns:
                    raise ValueError(f"Probabilistic column '{c}' for layer '{name}' not found")
            P = df[cols].to_numpy(dtype=float)
            cats = [c.split("=", 1)[1] if "=" in c else c for c in cols]
            layers[name] = ProbabilisticCategoricalFitness.from_probabilities(name, P, categories=cats)

    # build graph qnd landscape (using the unified builder).
    L = FitnessLandscape.build(
        sequences=seqs,
        graph=graph,
        fitness_layers=layers if layers else None,
        embeddings=embeddings,
        embedding_domain=embedding_domain,
        attach_embeddings=attach_embeddings,
        emb_arr_key=emb_arr_key,
        model_name=model_name,
        batch_size=batch_size,
        device=device,
    )
    return L

def to_csv_landscape(L: FitnessLandscape,
                     path: str | Path,
                     *,
                     sequence_col: str = "sequence",
                     include_layers: bool = True) -> None:
    
    """
    Function to write a FitnessLandscape to a CSV file.
    
    Parameters
    ----------
    L : FitnessLandscape
        The fitness landscape object to write to CSV.
    
    path : str or Path
        The path where the CSV file will be saved.
    
    sequence_col : str, default=`"sequence"`
        The column name for sequences in the output CSV.
    
    include_layers : bool, default=`True`
        Whether to include fitness layers in the output CSV.
    """
    rows = []
    for i, s in enumerate(L.sequences):
        row = {sequence_col: s.to_str()}
        if include_layers:
            for name, layer in L.fitness_layers.items():
                if layer.dtype == "numeric":
                    row[name] = float(layer.to_scalar()[i])
                elif layer.dtype == "categorical":
                    row[name] = layer.get_value(i)
            # (extend to probabilistic; emit wide columns).
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
