# Decompose fitness into graph Fourier modes

Graph Fourier analysis expands a node-aligned signal in eigenvectors of a
the graph Laplacian matrix. Operator and edge-weight choices define the basis.

## Input

The signal must follow graph node order. Full bases are suitable only for small
graphs; `k`-truncated bases cover selected modes, not the full spectrum. See the
[spectral operator contract](spectral-operators.md).

## Worked example

```python
# cookbook: test
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from fitness_landscape.analysis import graph_spectral_analysis
from fitness_landscape.models import create_nk_binary_landscape
from fitness_landscape.transforms import eigenmode_decomposition, graph_fourier_transform

landscape = create_nk_binary_landscape(N=3, K=1, seed=17)
signal = landscape.view(landscape.active_layer_name).to_scalar()

bases = {}
for operator in ["adjacency", "laplacian", "norm_laplacian", "transition"]:
    eigenvalues, eigenvectors = eigenmode_decomposition(
        landscape, matrix=operator, weight_key=None
    )
    assert eigenvectors.shape == (8, 8)
    assert np.isfinite(eigenvalues).all() and np.isfinite(eigenvectors).all()
    bases[operator] = eigenvalues

eigenvectors, eigenvalues, coefficients = graph_fourier_transform(
    landscape, signal=signal, matrix="laplacian", weight_key=None
)
laplacian = nx.laplacian_matrix(
    landscape.graph, nodelist=list(landscape.graph.nodes), weight=None
).toarray()
residuals = np.linalg.norm(
    laplacian @ eigenvectors - eigenvectors * eigenvalues[None, :], axis=0
)
np.testing.assert_allclose(eigenvectors.T @ eigenvectors, np.eye(8), atol=1e-10)
np.testing.assert_allclose(eigenvectors @ coefficients, signal, atol=1e-10)
assert residuals.max() < 1e-10

power = coefficients**2
cumulative = np.cumsum(power) / power.sum()
summary = graph_spectral_analysis(
    landscape, matrix="laplacian", weight_key=None
)
assert np.isclose(summary["spectral_gap"], 2.0)

with TemporaryDirectory() as tmp:
    fig, axis = plt.subplots(figsize=(4, 3))
    axis.step(eigenvalues, cumulative, where="post")
    axis.set(xlabel="Laplacian eigenvalue", ylabel="cumulative spectral power")
    figure = Path(tmp) / "spectral_power.png"
    fig.savefig(figure, dpi=100)
    plt.close(fig)
    assert figure.stat().st_size > 0

print(eigenvalues.tolist(), residuals.max(), cumulative[-1])
```


## Common failures

- Fitness order differs from the eigenvector node order.
- A normalized or transition spectrum is interpreted as combinatorial.
- A `k`-truncated transform is described as a full reconstruction.
- Individual degenerate eigenvectors receive stable biological labels.
