# Compute Dirichlet energy

Dirichlet energy sums squared fitness differences once per undirected edge.
It depends on fitness units, graph density, and optional conductance. Applications
to protein fitness and learned representation landscapes include
[Matthews et al. (2024)](https://doi.org/10.1038/s42256-024-00935-2),
[Vongsouthi et al. (2025)](https://doi.org/10.1126/sciadv.ads8318), and
[Castro et al. (2022)](https://doi.org/10.1038/s42256-022-00532-1).

## Input

Use a finite scalar active layer on a simple undirected graph. A weighted
analysis must name a finite non-negative conductance key. Raw distance is not
conductance. See the [edge contract](../graph-construction/edge-semantics.md).

## Worked example

```python
# cookbook: test
import numpy as np

from fitness_landscape.analysis import calculate_ruggedness_dirichlet_energy
from fitness_landscape.models import create_nk_binary_landscape

landscape = create_nk_binary_landscape(N=3, K=1, seed=17)
fitness = landscape.view(landscape.active_layer_name).to_scalar()

unweighted = calculate_ruggedness_dirichlet_energy(
    landscape,
    edge_weight_bins=[(0.0, 0.5), (0.5, 1.1)],
    weight_key=None,
)

manual = sum(
    float(fitness[source] - fitness[target]) ** 2
    for source, target in landscape.graph.edges
)
np.testing.assert_allclose(unweighted["global_dirichlet_energy"], manual)
np.testing.assert_allclose(
    unweighted["total_dirichlet_energy"], manual / len(landscape)
)
assert unweighted["weighted_laplacian"] is False
assert unweighted["weight_key"] is None
assert landscape.graph.number_of_edges() == 12

report = {
    "nodes": len(landscape),
    "edges": landscape.graph.number_of_edges(),
    "fitness_units": "seeded NK model units",
    "global": unweighted["global_dirichlet_energy"],
    "per_node": unweighted["total_dirichlet_energy"],
    "weight_key": unweighted["weight_key"],
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
