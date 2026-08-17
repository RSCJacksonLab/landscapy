# Construct a TDA alpha-complex graph

The 0.9 TDA constructor reduces an embedding with PCA, builds an alpha complex,
and exposes its one-skeleton as an undirected graph.

## Input

Embedding rows must be finite, distinct, and aligned with sequences. PCA rank is
bounded by centered geometric rank, sample count, feature count, and requested
`n_components`.

## Worked example

```python
# cookbook: test
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from fitness_landscape import BaseNumpySequence
from fitness_landscape.core import create_tda_graph

cache = pd.read_csv(Path("docs/cookbook/data/toy_protein_embeddings.csv"))
alphabet = list("ACDEFGHIKLMNPQRSTVWY")
sequences = [BaseNumpySequence(list(text), alphabet=alphabet) for text in cache.sequence]
embeddings = cache[["pc1", "pc2", "pc3", "pc4"]].to_numpy()
centered_rank = int(np.linalg.matrix_rank(embeddings - embeddings.mean(axis=0)))

graph = create_tda_graph(
    sequences,
    embeddings,
    n_components=3,
    reweight_simplex_edges=True,
)
assert centered_rank >= 3
assert graph.graph["tda_requested_components"] == 3
assert graph.graph["tda_effective_components"] == 3
assert graph.graph["tda_duplicate_policy"] == "reject"
assert list(graph.nodes) == list(range(len(sequences)))
assert all(
    {"distance", "affinity", "weight", "simplicial_weight"} <= set(data)
    for _, _, data in graph.edges(data=True)
)
assert all(np.isclose(data["weight"], 1.0 / (1.0 + data["distance"])) for _, _, data in graph.edges(data=True))

try:
    create_tda_graph(sequences, np.vstack([embeddings[:-1], embeddings[0]]))
except ValueError as error:
    assert "duplicate points" in str(error)
else:
    raise AssertionError("duplicate embedding points must be rejected")

print(
    {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "components": nx.number_connected_components(graph),
        "effective_rank": graph.graph["tda_effective_components"],
    }
)
print(list(graph.edges(data=True)))
```

The fixture yields six nodes, four edges, two components, and three effective
PCA dimensions. `distance` is Euclidean in PCA space. `weight` is the canonical
conductance `1 / (1 + distance)`. Optional triangle participation is stored
separately as `simplicial_weight`; it does not overwrite conductance.

## Alpha selection and scope

The 0.9 public constructor automatically chooses the squared alpha threshold as
the 95th percentile of finite zero-dimensional persistence death times (or
`0.01` when none exist). It does not expose a user alpha parameter or record the
chosen value, so report this algorithm and test embedding/PCA sensitivity. A
future API should expose the selected alpha for stronger provenance.

Alpha-complex graph construction is in release scope. Persistent-homology
analysis and interpretation are explicitly outside the 0.9 release scope; an
edge set alone does not support a claim about topological features.

## Common failures

- Duplicate or geometrically degenerate points are passed to Gudhi.
- Requested PCA dimension is reported instead of effective rank.
- PCA-space distance is used as NetworkX conductance.
- `simplicial_weight` is mistaken for canonical `weight`.
- Automatic alpha selection is treated as biologically calibrated.
