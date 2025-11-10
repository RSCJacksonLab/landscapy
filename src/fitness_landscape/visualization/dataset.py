from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, Iterable, List, Mapping, MutableMapping, Optional

import numpy as np


@dataclass(slots=True)
class VisualizationDataset:
    """
    Lightweight container describing the data required to render a landscape view.

    Attributes
    ----------
    nodes :
        Ordered list of node ids present in the visualization.
    positions :
        Array of shape (N, 2) giving x/y coordinates per node in the same order
        as ``nodes``.
    edges :
        Iterable of (u, v) tuples describing the edges to draw. Edges are
        filtered to the subset of nodes contained in ``nodes``.
    fitness_name :
        Name of the active fitness layer, if any.
    fitness_values :
        1D array of fitness scalars aligned with ``nodes``. ``None`` if no
        fitness layer is selected.
    annotation_name :
        Name of the active annotation scheme, if any.
    annotation_values :
        Mapping of annotation column -> list of values aligned with ``nodes``.
    palettes :
        Mapping of palette name -> palette payload (opaque dict; consumers
        interpret it according to their rendering backend).
    metadata :
        Additional free-form metadata describing the dataset (e.g., layout
        parameters, query filters).
    """

    nodes: List[Hashable]
    positions: np.ndarray
    edges: Iterable[tuple[Hashable, Hashable]] = field(default_factory=list)
    fitness_name: Optional[str] = None
    fitness_values: Optional[np.ndarray] = None
    annotation_name: Optional[str] = None
    annotation_values: Mapping[str, List[Any]] = field(default_factory=dict)
    palettes: MutableMapping[str, Any] = field(default_factory=dict)
    metadata: MutableMapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the dataset to a JSON-friendly dictionary.
        """
        return {
            "nodes": list(self.nodes),
            "positions": self.positions.tolist(),
            "edges": list(self.edges),
            "fitness_name": self.fitness_name,
            "fitness_values": None if self.fitness_values is None else self.fitness_values.tolist(),
            "annotation_name": self.annotation_name,
            "annotation_values": {k: list(v) for k, v in self.annotation_values.items()},
            "palettes": dict(self.palettes),
            "metadata": dict(self.metadata),
        }

    def subset(self, mask: np.ndarray) -> "VisualizationDataset":
        """
        Create a filtered copy of the dataset according to a boolean mask.
        """
        if mask.shape[0] != len(self.nodes):
            raise ValueError("Mask length must equal the number of nodes in the dataset.")

        nodes = [n for n, keep in zip(self.nodes, mask) if keep]
        positions = self.positions[mask]

        if self.fitness_values is not None:
            fitness_values = self.fitness_values[mask]
        else:
            fitness_values = None

        ann_values = {
            key: [val for val, keep in zip(values, mask) if keep]
            for key, values in self.annotation_values.items()
        }

        node_set = set(nodes)
        edges = [(u, v) for (u, v) in self.edges if u in node_set and v in node_set]

        return VisualizationDataset(
            nodes=nodes,
            positions=positions,
            edges=edges,
            fitness_name=self.fitness_name,
            fitness_values=fitness_values,
            annotation_name=self.annotation_name,
            annotation_values=ann_values,
            palettes=dict(self.palettes),
            metadata=dict(self.metadata),
        )
