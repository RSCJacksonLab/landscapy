# Compute Dirichlet energy

Dirichlet energy sums squared fitness differences once per undirected edge.
It depends on fitness units, graph density, and optional conductance.

## Install and input

```bash
python -m pip install landscapy
```

Use a finite scalar active layer on a simple undirected graph. A weighted
analysis must name a finite non-negative conductance key; raw distance is not
conductance. See the [edge contract](../graph-construction/edge-semantics.md).

## Worked example

```python
# cookbook: test
import numpy as np

from fitness_landscape.analysis import calculate_ruggedness_dirichlet_energy
from fitness_landscape.models import create_nk_binary_landscape

landscape = create_nk_binary_landscape(N=3, K=1, seed=17)
fitness = landscape.view(landscape.active_layer_name).to_scalar()
for source, target in landscape.graph.edges:
    landscape.graph[source][target]["conductance"] = 2.0

unweighted = calculate_ruggedness_dirichlet_energy(
    landscape, edge_weight_bins=[(0.0, 0.5), (0.5, 1.1)]
)
weighted = calculate_ruggedness_dirichlet_energy(
    landscape, weight_key="conductance"
)

manual = sum(
    float(fitness[source] - fitness[target]) ** 2
    for source, target in landscape.graph.edges
)
np.testing.assert_allclose(unweighted["global_dirichlet_energy"], manual)
np.testing.assert_allclose(weighted["global_dirichlet_energy"], 2.0 * manual)
np.testing.assert_allclose(
    unweighted["total_dirichlet_energy"], manual / len(landscape)
)
assert unweighted["weighted_laplacian"] is False
assert weighted["weight_key"] == "conductance"
assert landscape.graph.number_of_edges() == 12

report = {
    "nodes": len(landscape),
    "edges": landscape.graph.number_of_edges(),
    "fitness_units": "seeded NK model units",
    "global": unweighted["global_dirichlet_energy"],
    "per_node": unweighted["total_dirichlet_energy"],
    "weighted_global": weighted["global_dirichlet_energy"],
}
print(report)
```

`global_dirichlet_energy` is the once-per-edge sum. Despite its historical
name, `total_dirichlet_energy` is normalized by node count. Edge bins partition
diagnostics by stored edge values; report their definitions and coverage.

## Interpretation

Larger energy means adjacent values differ more under the declared graph and
units. It does not by itself identify a mechanism or support comparison across
graphs with different edge counts, components, conductance scales, or fitness
units.

## Common failures

- Each undirected edge is counted in both orientations.
- `total_dirichlet_energy` is reported as an unnormalized total.
- Raw Hamming or embedding distance is passed as conductance.
- Fitness is standardized after looking at the preferred graph result.
- Values from graphs with different density or support are directly ranked.
