from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from ..core.landscape import FitnessLandscape
from ..core.annotation import AnnotationLayer
from ..core.fitness import (
    BaseFitnessLayer,
    CategoricalFitness,
    NumericFitness,
    ProbabilisticCategoricalFitness,
)
from ..transforms.eigenmode import eigenmode_decomposition
from .dataset import VisualizationDataset
from .registry import AnnotationRegistry, PaletteStore, AnnotationDescriptor


@dataclass(slots=True)
class LayoutSpec:
    """
    Specification for building node coordinates.
    """

    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)


class VisualizationDatasetBuilder:
    """
    Build :class:`VisualizationDataset` instances from a :class:`FitnessLandscape`.
    """

    def __init__(
        self,
        landscape: FitnessLandscape,
        *,
        annotation_registry: AnnotationRegistry | None = None,
    ) -> None:
        self.landscape = landscape
        self.annotation_registry = annotation_registry or AnnotationRegistry()

    def build(
        self,
        *,
        layout: LayoutSpec | str = "graph",
        fitness_layer: str | None = None,
        annotation: str | None = None,
        query: Mapping[str, Any] | None = None,
        include_edges: bool = True,
        palette_store: PaletteStore | None = None,
        external_positions: Mapping[Hashable, Sequence[float]] | None = None,
    ) -> VisualizationDataset:
        layout_spec = self._normalise_layout(layout)
        layout_params = dict(layout_spec.parameters)
        if layout_spec.name in {"graph", "sfdp"} and "engine" not in layout_params:
            layout_params["engine"] = "sfdp"
        layout_spec = LayoutSpec(name=layout_spec.name, parameters=layout_params)

        indices = self._resolve_indices(annotation=annotation, query=query)
        nodes = [self._node_for_index(i) for i in indices]
        node_set = set(nodes)

        fitness_layer_obj = self._resolve_fitness_layer(fitness_layer)
        fitness_values = None
        fitness_kind: str | None = None
        fitness_categories: list[str] | None = None
        fitness_labels: list[Any] | None = None
        fitness_probabilities = None
        fitness_name = fitness_layer_obj.name if fitness_layer_obj is not None else None

        if fitness_layer_obj is not None:
            dtype = getattr(fitness_layer_obj, "dtype", None)
            if isinstance(fitness_layer_obj, ProbabilisticCategoricalFitness) or (
                dtype == "categorical" and hasattr(fitness_layer_obj, "probabilities")
            ):
                fitness_kind = "probabilistic"
                fitness_categories = list(fitness_layer_obj.categories)
                fitness_probabilities = np.asarray(fitness_layer_obj.probabilities)[indices]
            elif isinstance(fitness_layer_obj, CategoricalFitness) or dtype == "categorical":
                fitness_kind = "categorical"
                fitness_categories = list(getattr(fitness_layer_obj, "categories", [])) or None
                fitness_labels = [
                    fitness_layer_obj.get_value(int(idx)) for idx in indices.tolist()
                ]
            else:
                fitness_kind = "numeric"
                fitness_values = fitness_layer_obj.to_scalar()[indices]

        annotation_layer, descriptor = self._resolve_annotation_layer(annotation)
        annotation_values = (
            self._collect_annotations(annotation_layer, indices) if annotation_layer else {}
        )

        positions = self._build_positions(
            nodes,
            layout_spec,
            indices=indices,
            external_positions=external_positions,
        )

        edges: Iterable[tuple[Hashable, Hashable]]
        if include_edges and self.landscape.graph is not None:
            edges = [
                (u, v)
                for (u, v) in self.landscape.graph.edges()
                if u in node_set and v in node_set
            ]
        else:
            edges = []

        palettes: Dict[str, Any] = {}
        if descriptor and palette_store is not None and descriptor.palette_key:
            palette = palette_store.get_palette(descriptor.palette_key)
            if palette is not None:
                palettes[descriptor.palette_key] = palette

        metadata = {
            "layout": layout_spec.name,
            "layout_parameters": dict(layout_spec.parameters),
            "query": dict(query) if query else None,
        }

        return VisualizationDataset(
            nodes=nodes,
            positions=positions,
            edges=edges,
            fitness_name=fitness_name,
            fitness_values=fitness_values,
            fitness_kind=fitness_kind,
            fitness_categories=fitness_categories,
            fitness_labels=fitness_labels,
            fitness_probabilities=fitness_probabilities,
            annotation_name=annotation_layer.name if annotation_layer else None,
            annotation_values=annotation_values,
            palettes=palettes,
            metadata=metadata,
        )

    def _normalise_layout(self, layout: LayoutSpec | str) -> LayoutSpec:
        if isinstance(layout, LayoutSpec):
            return layout
        if not isinstance(layout, str):
            raise TypeError("layout must be a string or LayoutSpec.")
        return LayoutSpec(name=layout)

    def _resolve_indices(
        self,
        *,
        annotation: str | None,
        query: Mapping[str, Any] | None,
    ) -> np.ndarray:
        if query and not annotation:
            raise ValueError("Annotation name must be provided when using query filters.")

        if query and annotation:
            result = self.landscape.query_annotations(annotation, query)
            return np.asarray(result.sequence_indices, dtype=int)

        return np.arange(len(self.landscape.sequences))

    def _node_for_index(self, index: int) -> Hashable:
        try:
            return self.landscape._nodes_by_index[index]  # type: ignore[attr-defined]
        except AttributeError as exc:  # pragma: no cover - defensive fallback
            raise RuntimeError("Landscape does not expose node index mapping.") from exc

    def _resolve_fitness_layer(self, name: str | None) -> BaseFitnessLayer | None:
        if name is None:
            active = self.landscape.active_layer_name
            if active is None:
                return None
            return self.landscape.get_layer(active)
        return self.landscape.get_layer(name)

    def _resolve_annotation_layer(
        self, name: str | None
    ) -> tuple[AnnotationLayer | None, AnnotationDescriptor | None]:
        if name is None:
            return None, None
        if name in self.annotation_registry:
            descriptor = self.annotation_registry.get(name)
            return descriptor.layer, descriptor
        layer = self.landscape.get_annotation_layer(name)
        descriptor = self.annotation_registry.register(name, layer, source="landscape")
        return layer, descriptor

    def _collect_annotations(
        self,
        layer: AnnotationLayer,
        indices: np.ndarray,
    ) -> Dict[str, List[Any]]:
        columns = layer.columns
        data: Dict[str, List[Any]] = {c: [] for c in columns}
        for idx in indices.tolist():
            record = layer.get_record(int(idx))
            for col in columns:
                data[col].append(record.get(col))
        return data

    def _build_positions(
        self,
        nodes: List[Hashable],
        layout_spec: LayoutSpec,
        *,
        indices: np.ndarray,
        external_positions: Mapping[Hashable, Sequence[float]] | None,
    ) -> np.ndarray:
        name = layout_spec.name
        params = layout_spec.parameters

        if name == "graph":
            return self._graph_layout(nodes, params)
        if name == "sfdp":
            merged = {"engine": "sfdp"}
            merged.update(params)
            return self._graph_layout(nodes, merged)
        if name == "embedding":
            return self._embedding_layout(nodes, params)
        if name == "diffusion":
            return self._diffusion_layout(nodes, params)
        if name == "external":
            if not external_positions:
                raise ValueError("external_positions must be provided for external layout.")
            return self._external_layout(nodes, external_positions)
        if name == "umap":
            return self._umap_layout(nodes, params)

        raise ValueError(
            f"Unknown layout '{name}'. Supported: graph, sfdp, embedding, diffusion, umap, external."
        )

    def _graph_layout(self, nodes: List[Hashable], params: Mapping[str, Any]) -> np.ndarray:
        subgraph = self.landscape.graph.subgraph(nodes) if self.landscape.graph else nx.Graph()
        layout_params = dict(params)
        engine = layout_params.pop("engine", layout_params.pop("algorithm", "sfdp"))
        graphviz_args = layout_params.pop("graphviz_args", "")
        seed = layout_params.pop("seed", 0)
        if engine == "sfdp":
            coords = self._graphviz_sfdp_layout(subgraph, nodes, args=graphviz_args)
            if coords is None:
                raise RuntimeError("Graphviz 'sfdp' layout failed and fallback is disabled.")
            return coords
        elif engine not in {None, "spring"}:
            raise ValueError(f"Unknown graph layout engine '{engine}'. Use 'sfdp' or 'spring'.")

        positions = nx.spring_layout(subgraph, seed=seed, **layout_params)
        return np.array([positions[node] for node in nodes], dtype=float)

    def _graphviz_sfdp_layout(
        self,
        subgraph: nx.Graph,
        nodes: List[Hashable],
        *,
        args: str = "",
    ) -> np.ndarray | None:
        if not nodes:
            return np.zeros((0, 2), dtype=float)
        try:
            from networkx.drawing.nx_pydot import graphviz_layout
            import inspect
        except ImportError as exc:
            raise RuntimeError(
                "Graphviz layout requires the 'pydot' package and Graphviz binaries."
            ) from exc

        try:
            sig = inspect.signature(graphviz_layout)
            call_kwargs = {"prog": "sfdp"}
            if "args" in sig.parameters and args:
                call_kwargs["args"] = args
            # Relabel nodes and strip attributes so Graphviz doesn't choke on
            # complex Python object strings or huge labels.
            relabel_map = {node: f"n{idx}" for idx, node in enumerate(subgraph.nodes())}
            bare = nx.Graph() if not subgraph.is_directed() else nx.DiGraph()
            bare.add_nodes_from(relabel_map.values())
            bare.add_edges_from(
                (relabel_map[u], relabel_map[v]) for u, v in subgraph.edges()
            )
            positions = graphviz_layout(bare, **call_kwargs)
            coords = np.array(
                [positions[relabel_map[node]] for node in nodes],
                dtype=float,
            )
        except (OSError, RuntimeError, KeyError) as exc:
            raise RuntimeError(f"Graphviz 'sfdp' layout failed: {exc}") from exc

        if coords.ndim == 1:
            coords = coords.reshape(-1, 1)
        if coords.shape[1] < 2:
            padding = np.zeros((coords.shape[0], 2 - coords.shape[1]), dtype=float)
            coords = np.hstack([coords, padding])
        return coords[:, :2]

    def _embedding_layout(self, nodes: List[Hashable], params: Mapping[str, Any]) -> np.ndarray:
        embeddings, _ = self._get_embedding_matrix(params.get("emb_key"))
        node_to_index = {node: i for i, node in enumerate(self.landscape._node_order)}  # type: ignore[attr-defined]
        coords = []
        for node in nodes:
            idx = node_to_index.get(node)
            if idx is None:
                raise KeyError(f"Node '{node}' not found in embedding matrix.")
            coords.append(embeddings[idx][:2])
        coords_array = np.asarray(coords, dtype=float)
        if coords_array.shape[1] < 2:
            raise ValueError("Embeddings must have at least two dimensions for plotting.")
        return coords_array[:, :2]

    def _diffusion_layout(self, nodes: List[Hashable], params: Mapping[str, Any]) -> np.ndarray:
        graph = self.landscape.graph
        if graph is None:
            raise ValueError("Landscape does not contain a graph; cannot use 'diffusion' layout.")
        total_nodes = graph.number_of_nodes()
        if total_nodes == 0:
            return np.zeros((len(nodes), 2), dtype=float)

        dims = int(params.get("dimensions", params.get("components", params.get("k", 2))))
        dims = max(dims, 1)

        matrix_type = "transition"
        if total_nodes <= dims + 1:
            eigvals, eigvecs = eigenmode_decomposition(graph, k=None, matrix=matrix_type)
        else:
            eigvals, eigvecs = eigenmode_decomposition(graph, k=dims + 1, matrix=matrix_type)
        if eigvecs.shape[1] <= 1:
            coords_full = np.zeros((total_nodes, dims), dtype=float)
        else:
            components = eigvecs[:, 1 : min(eigvecs.shape[1], dims + 1)]
        if components.shape[1] < dims:
            padding = np.zeros((components.shape[0], dims - components.shape[1]), dtype=float)
            components = np.hstack([components, padding])
        coords_full = components[:, :dims]

        node_to_index = {node: i for i, node in enumerate(self.landscape._node_order)}  # type: ignore[attr-defined]
        coords: list[np.ndarray] = []
        for node in nodes:
            idx = node_to_index.get(node)
            if idx is None:
                raise KeyError(f"Node '{node}' not found when building diffusion layout.")
            coords.append(coords_full[idx])
        return np.asarray(coords, dtype=float)

    def _umap_layout(self, nodes: List[Hashable], params: Mapping[str, Any]) -> np.ndarray:
        embeddings, _ = self._get_embedding_matrix(params.get("emb_key"))
        total = embeddings.shape[0]
        if total == 0:
            return np.zeros((len(nodes), 2), dtype=float)
        if total < 2:
            coords_full = np.zeros((total, 2), dtype=float)
        elif total <= 3:
            # For very small datasets, UMAP's spectral init can fail (k >= N). Fall back
            # to the first two embedding dimensions padded with zeros if needed.
            coords_full = np.asarray(embeddings, dtype=float)
            if coords_full.shape[1] < 2:
                padding = np.zeros((coords_full.shape[0], 2 - coords_full.shape[1]), dtype=float)
                coords_full = np.hstack([coords_full, padding])
            coords_full = coords_full[:, :2]
        else:
            try:
                import umap  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("layout='umap' requires the 'umap-learn' package.") from exc
            max_neighbors = max(2, total - 1)
            n_neighbors = int(params.get("n_neighbors", min(15, max_neighbors)))
            n_neighbors = max(2, min(n_neighbors, max_neighbors))
            min_dist = float(params.get("min_dist", 0.1))
            metric = params.get("metric", "euclidean")
            random_state = params.get("random_state", 42)
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric=metric,
                random_state=random_state,
            )
            coords_full = reducer.fit_transform(embeddings)

        node_to_index = {node: i for i, node in enumerate(self.landscape._node_order)}  # type: ignore[attr-defined]
        coords = []
        for node in nodes:
            idx = node_to_index.get(node)
            if idx is None:
                raise KeyError(f"Node '{node}' not found when building UMAP layout.")
            coords.append(coords_full[idx])
        return np.asarray(coords, dtype=float)

    def _get_embedding_matrix(self, emb_key: str | None) -> Tuple[np.ndarray, str]:
        embeddings = getattr(self.landscape, "embeddings", None)
        if embeddings is None:
            raise ValueError("Landscape does not contain embeddings; cannot use this layout.")
        if isinstance(embeddings, np.ndarray):
            embeddings_store: Mapping[str, np.ndarray] = {"default": embeddings}
        else:
            if not isinstance(embeddings, Mapping):
                raise TypeError("Landscape embeddings must be a mapping or numpy array.")
            embeddings_store = embeddings
        if not embeddings_store:
            raise ValueError("No embeddings are available on the landscape.")
        key = emb_key if emb_key is not None else next(iter(embeddings_store))
        if key not in embeddings_store:
            raise KeyError(
                f"Embedding domain '{key}' not found. Available keys: {list(embeddings_store.keys())}"
            )
        matrix = np.asarray(embeddings_store[key], dtype=float)
        if matrix.ndim != 2:
            raise ValueError("Embeddings must be a 2-D array.")
        if matrix.shape[1] < 2:
            raise ValueError("Embeddings must have at least two dimensions for plotting.")
        return matrix, key

    def _external_layout(
        self,
        nodes: List[Hashable],
        external_positions: Mapping[Hashable, Sequence[float]],
    ) -> np.ndarray:
        coords = []
        for node in nodes:
            if node not in external_positions:
                raise KeyError(f"Node '{node}' missing from external positions.")
            pos = external_positions[node]
            if len(pos) < 2:
                raise ValueError("External positions must provide at least two dimensions.")
            coords.append(pos[:2])
        return np.asarray(coords, dtype=float)
