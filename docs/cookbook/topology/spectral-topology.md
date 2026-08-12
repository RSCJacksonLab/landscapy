# Inspect spectral topology

Spectra summarize a chosen graph operator. Operator, weighting, component
structure, and mode count are part of the result—not optional implementation
details.

## Install and input

```bash
python -m pip install landscapy
```

This recipe uses a simple undirected graph with two connected components and no
isolates. It requests unweighted operators explicitly with `weight_key=None`.
See the [spectral-operator contract](../../spectral_operators.md) and [edge
semantics](../../edge_semantics.md) before using conductance weights.

## Worked example

```python
# cookbook: test
import networkx as nx
import numpy as np

from fitness_landscape import BaseNumpySequence, FitnessLandscape
from fitness_landscape.analysis import graph_spectral_analysis
from fitness_landscape.transforms import eigenmode_decomposition

graph = nx.Graph([(0, 1), (1, 2), (3, 4)])
sequences = [BaseNumpySequence([index], sequence_id=f"s{index}") for index in range(5)]
for node, sequence in zip(graph.nodes, sequences):
    graph.nodes[node]["sequence"] = sequence
landscape = FitnessLandscape(sequences, graph)

reported = {}
for matrix in ("laplacian", "norm_laplacian"):
    result = graph_spectral_analysis(
        landscape, matrix=matrix, k=None, weight_key=None
    )
    reported[matrix] = result
    assert result["node_order"] == list(graph.nodes)
    assert result["weight_key"] is None
    assert np.count_nonzero(np.isclose(result["eigenvalues"], 0.0)) == 2
    assert result["spectral_gap"] == 0.0

full = {}
for matrix in ("adjacency", "laplacian", "norm_laplacian", "transition"):
    eigenvalues, eigenvectors = eigenmode_decomposition(
        landscape, matrix=matrix, k=None, weight_key=None
    )
    full[matrix] = eigenvalues
    assert eigenvalues.shape == (5,)
    assert eigenvectors.shape == (5, 5)

sparse_values, sparse_vectors = eigenmode_decomposition(
    landscape,
    matrix="laplacian",
    k=3,
    weight_key=None,
    dense_threshold=0,
)
np.testing.assert_allclose(sparse_values, full["laplacian"][:3], atol=1e-8)
assert sparse_vectors.shape == (5, 3)

print({name: values.tolist() for name, values in full.items()})
print("zero modes", np.count_nonzero(np.isclose(full["laplacian"], 0.0)))
```

The Laplacian and normalized-Laplacian spectra have two zero modes, one per
connected component. Their reported graph-level spectral gap is therefore
zero. An isolate would add another zero mode under the documented operator
conventions. The `transition` option is the random-walk Laplacian `I - P`, not
the transition matrix `P` itself.

## Interpretation

Adjacency, combinatorial Laplacian, normalized Laplacian, and random-walk
operators answer different questions. `k=3` returns only a low-mode subspace;
it is not a full basis for exact reconstruction. Repeated eigenvalues define a
subspace, but individual eigenvectors may rotate or change sign across runs and
solvers. Compare invariant subspaces or derived quantities, not column labels.

## Common failures

- Using `weight_key="auto"` without an auditable edge schema, or using a raw
  distance as conductance.
- Reading the first nonzero eigenvalue as a connected-graph gap when multiple
  zero modes exist.
- Treating eigenvector signs or bases in a degenerate eigenspace as identifiable.
- Requesting sparse `k` modes and then assuming a complete transform basis.
- Assigning a biological mechanism to a spectral pattern without a graph-choice
  sensitivity analysis.
