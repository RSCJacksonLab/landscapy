# Import an external adjacency matrix or edge table

External topology is supported when node order and edge meaning are made
explicit before wrapping the graph in `FitnessLandscape`.

## Install and input

```bash
python -m pip install landscapy
```

This example treats nonzero matrix/table values as raw distances. A zero means
no edge, not zero distance. Inputs must be symmetric and undirected.

## Worked example

```python
# cookbook: test
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

from fitness_landscape import BinarySequence, FitnessLandscape

distance_adjacency = sparse.csr_array(
    np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 2.0, 0.0],
            [0.0, 2.0, 0.0, 1.5],
            [0.0, 0.0, 1.5, 0.0],
        ]
    )
)
assert np.allclose(distance_adjacency.toarray(), distance_adjacency.toarray().T)
graph = nx.from_scipy_sparse_array(distance_adjacency, edge_attribute="distance")

sequences = [BinarySequence(f"{index:02b}", sequence_id=f"s{index}") for index in range(4)]
for node, sequence in zip(graph.nodes, sequences):
    graph.nodes[node]["sequence"] = sequence
for _, _, data in graph.edges(data=True):
    data["normalized_distance"] = data["distance"] / 2.0
    data["affinity"] = float(np.exp(-data["normalized_distance"]))
    data["weight"] = data["affinity"]

graph.graph["landscapy_edge_schema"] = {
    "schema_version": "1.0.0",
    "constructor": "external-distance-adjacency",
    "distance": {"key": "distance", "units": "declared_external_units"},
    "normalized_distance": {"key": "normalized_distance", "units": "fraction_of_2"},
    "affinity": {"key": "affinity", "units": "dimensionless"},
    "conductance": {"key": "weight", "units": "dimensionless"},
    "transition_probability": {"key": None, "units": None},
    "legacy_aliases": {},
    "notes": "weight = exp(-distance / 2); source matrix v1",
}
landscape = FitnessLandscape(sequences, graph)

assert landscape.node_to_sequence_index == {0: 0, 1: 1, 2: 2, 3: 3}
assert all(graph.nodes[node]["sequence"] == sequences[node] for node in graph)
assert all(data["weight"] != data["distance"] for _, _, data in graph.edges(data=True))

# Equivalent edge-table route with an explicit source/target schema.
edge_table = pd.DataFrame(
    {"source": [0, 1, 2], "target": [1, 2, 3], "distance": [1.0, 2.0, 1.5]}
)
table_graph = nx.from_pandas_edgelist(
    edge_table, source="source", target="target", edge_attr="distance", create_using=nx.Graph
)
assert set(table_graph.edges) == set(graph.edges)

try:
    FitnessLandscape(sequences, nx.DiGraph(graph))
except TypeError as error:
    assert "undirected" in str(error)
else:
    raise AssertionError("directed inputs must be rejected in 0.9")

print(landscape.node_to_sequence_index)
print(list(graph.edges(data=True)))
```

The wrapped graph is a four-node path with three raw distances and an explicit
distance-to-conductance transform. Matrix row/column order, edge-table IDs, and
landscape sequence order are identical and asserted.

## Interpretation

The graph preserves an externally specified topology; Landscapy does not
validate the scientific reason for each edge. Follow the [edge-semantics
contract](../../edge_semantics.md): raw `distance` increases with separation,
whereas canonical `weight` increases with connection strength. Record units and
the transform. Reject an input whose bare `weight` could mean distance, count,
probability, or conductance until its meaning is resolved.

## Common failures

- Sparse-matrix row order differs from sequence order.
- Zero is ambiguously used for both no edge and observed zero distance.
- A directed matrix is silently symmetrized.
- A distance column is renamed `weight` without transformation.
- Duplicate/parallel edge-table rows are collapsed without a declared rule.
