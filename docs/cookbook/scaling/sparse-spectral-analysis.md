# Audit a truncated sparse spectral analysis

Request only the low-frequency modes needed for a declared analysis on a
larger sparse connected component. A `k`-mode result is a projection, never a
full spectrum or exact signal reconstruction unless `k` spans the graph.

## Install and input

```bash
python -m pip install landscapy
```

This example uses the combinatorial Laplacian with unit conductance on a
20-by-20 grid. Node order is fixed before building the sparse matrix and signal.

## Worked example

```python
# cookbook: test
import networkx as nx
import numpy as np

from fitness_landscape.transforms import graph_fourier_transform

graph = nx.convert_node_labels_to_integers(nx.grid_2d_graph(20, 20), ordering="sorted")
node_order = list(graph.nodes())
signal = np.array([divmod(node, 20)[0] + divmod(node, 20)[1] for node in node_order], dtype=float)
signal -= signal.mean()
requested_modes = 12
vectors, values, coefficients = graph_fourier_transform(
    graph,
    signal=signal,
    matrix="laplacian",
    k=requested_modes,
    weight_key=None,
)
laplacian = nx.laplacian_matrix(graph, nodelist=node_order, weight=None).astype(float)
residuals = np.linalg.norm(
    laplacian @ vectors - vectors * values[None, :], axis=0
)
orthogonality_error = np.linalg.norm(vectors.T @ vectors - np.eye(requested_modes))
projection = vectors @ coefficients
truncation_coverage = float(np.sum(coefficients**2) / np.sum(signal**2))
memory = {
    "sparse_laplacian_bytes": int(laplacian.data.nbytes + laplacian.indices.nbytes + laplacian.indptr.nbytes),
    "returned_modes_bytes": int(vectors.nbytes + values.nbytes),
}
report = {
    "solver": "sparse eigensolver through eigenmode_decomposition",
    "operator": "combinatorial_laplacian",
    "requested_modes": requested_modes,
    "returned_modes": vectors.shape[1],
    "zero_modes": int(np.count_nonzero(np.isclose(values, 0.0, atol=1e-9))),
    "max_residual": float(residuals.max()),
    "orthogonality_error": float(orthogonality_error),
    "truncation_coverage": truncation_coverage,
    "projection_rmse": float(np.sqrt(np.mean((signal - projection) ** 2))),
    "memory": memory,
}

assert nx.is_connected(graph) and report["zero_modes"] == 1
assert vectors.shape == (400, requested_modes)
assert report["max_residual"] < 1e-7
assert report["orthogonality_error"] < 1e-7
assert 0.0 <= truncation_coverage <= 1.0
assert requested_modes < graph.number_of_nodes()
print(report)
```

Low-frequency projection and mode-local coefficients are available. Exact
reconstruction, a full spectral distribution, and full-basis cumulative power
remain unavailable. Dirichlet energy can instead be computed directly from
edges. Sum quantities over degenerate eigenspaces rather than naming arbitrary
basis vectors.

## Common failures

- `returned_modes` is reported as graph size or a full spectrum.
- Sparse-solver residuals and orthogonality are not checked.
- Repeated eigenvalues are assigned stable individual interpretations.
- Operator, conductance key, node order, and component are omitted.
- Projection coverage is called variance explained by a fitted causal model.
