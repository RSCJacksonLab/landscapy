# Construct elementary landscapes on Hamming and kNN graphs

An elementary landscape uses one graph-Laplacian eigenvector as its fitness
signal. It is a known-answer control for spectral and random-walk analysis, but
its meaning is inseparable from the chosen graph and operator.

## Install and input

```bash
python -m pip install "landscapy[knn]"
```

Select eigenvector index `j` and graph type. A kNN graph additionally requires
a representation and neighbourhood size; the current factory uses OHE kNN for
sequence-only input.

## Worked example

```python
# cookbook: test
import networkx as nx
import numpy as np

from fitness_landscape.models import create_elementary_landscape

report = {}
for graph_type, kwargs in (("hamming", {}), ("knn", {"k": 3})):
    landscape = create_elementary_landscape(
        j=2,
        N=4,
        alphabet=[0, 1],
        graph_type=graph_type,
        **kwargs,
    )
    nodes = list(landscape.graph.nodes())
    signal_by_node = {
        node: landscape.get_signal()[landscape.sequence_index_for_node(node)]
        for node in nodes
    }
    signal = np.array([signal_by_node[node] for node in nodes])
    laplacian = nx.laplacian_matrix(
        landscape.graph, nodelist=nodes, weight="weight"
    ).astype(float)
    rayleigh = float(signal @ (laplacian @ signal) / (signal @ signal))
    residual = np.linalg.norm(laplacian @ signal - rayleigh * signal)
    report[graph_type] = {
        "nodes": len(nodes),
        "edges": landscape.graph.number_of_edges(),
        "eigenvalue": rayleigh,
        "relative_residual": residual / np.linalg.norm(signal),
        "metadata": landscape.active_layer.metadata,
    }

assert all(row["nodes"] == 16 for row in report.values())
assert all(row["relative_residual"] < 1e-10 for row in report.values())
assert report["hamming"]["metadata"]["graph_type"] == "hamming"
assert report["knn"]["metadata"]["graph_type"] == "knn"
print(report)
```

The residual tests the defining property against the same combinatorial
Laplacian and edge weights used to construct the signal. A different operator
requires a newly defined known answer, not reuse of this threshold.

## Common failures

- `j` is treated as an eigenvalue rather than an ordered eigenvector index.
- Fitness row order is assumed to equal arbitrary graph-node iteration order.
- A normalized or random-walk Laplacian is used to verify a combinatorial mode.
- OHE kNN topology is described as equivalent to biological mutational adjacency.
- Degenerate eigenvalues are expected to return an identical basis vector across solvers.
