# Export a PyTorch Geometric graph

`FitnessLandscape.to_graph_tensor` converts node features, connectivity, edge
attributes, and fitness layers to a PyTorch Geometric `Data` object. The
exported object is a transport format; edge semantics still come from the
Landscapy graph schema.

## Install and input

```bash
python -m pip install "landscapy[ml]"
```

This example uses OHE features and passes `tokenizer=None`, so it performs no
model download. Every edge must have the same attribute keys before PyG export.

## Worked example

```python
# cookbook: test
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from fitness_landscape import BinarySequence
from fitness_landscape.core import FitnessLandscape, NumericFitness, create_hamming_graph

table = pd.read_csv(
    Path("docs/cookbook/data/toy_landscape.csv"), dtype={"sequence": "string"}
)
sequences = [BinarySequence(text, sequence_id=f"toy-{i}") for i, text in enumerate(table.sequence)]
graph = create_hamming_graph(sequences)
graph.remove_edges_from(list(graph.edges(7)))  # preregistered unsupported query node
graph = nx.relabel_nodes(graph, {node: f"genotype-{node}" for node in graph}, copy=True)
landscape = FitnessLandscape(
    sequences,
    graph,
    fitness_layers={"measured": NumericFitness.from_scalars("measured", table.fitness)},
)

node_order = list(landscape.graph.nodes())
row_order = [landscape.sequence_index_for_node(node) for node in node_order]
assert row_order == list(range(len(sequences)))
edge_key_sets = {frozenset(data) for _, _, data in landscape.graph.edges(data=True)}
assert len(edge_key_sets) == 1
schema = landscape.graph.graph["landscapy_edge_schema"]
conductance_key = schema["conductance"]["key"]
assert conductance_key == "weight"
assert all(np.isfinite(data[conductance_key]) for _, _, data in landscape.graph.edges(data=True))

data = landscape.to_graph_tensor(tokenizer=None)
assert data.x.shape == (8, 3 * 2)
assert data.edge_index.shape[1] == 2 * landscape.graph.number_of_edges()
assert data.measured.shape[0] == 8
assert data.weight.shape[0] == data.edge_index.shape[1]
assert data.num_nodes == 8
isolated_row = landscape.sequence_index_for_node("genotype-7")
assert isolated_row not in data.edge_index
print(data, schema["conductance"])
```

PyG represents each undirected edge in both directions. Named arrays such as
`weight`, `distance`, and `affinity` retain distinct meanings; do not choose an
`edge_attr` merely because it is present. For arbitrary node labels, the
sequence-index order check is mandatory before assigning row-aligned `x`.

## Common failures

- One undirected graph edge is expected to produce one `edge_index` column.
- Distance is supplied as message-passing conductance without a defined transform.
- Missing attributes on only some edges are allowed through as implicit zeroes.
- Isolated nodes are dropped because they never occur in `edge_index`.
- Graph node iteration and landscape sequence order are assumed to match.
