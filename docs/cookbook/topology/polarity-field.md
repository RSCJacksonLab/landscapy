# Calculate a local polarity field and edge tilts

`calculate_polarity_field` compares an observed sequence panel with an explicit
reference panel on one shared graph. It applies the same continuous-time
random-walk heat kernel to both source-normalized empirical measures and returns
the dimensionless node field

```text
phi(i) = log(rho_observed(i) / rho_reference(i)).
```

Positive polarity means that the node's graph neighbourhood is enriched in the
observed panel relative to the reference. Negative polarity means reference
enrichment. For a canonically oriented edge `i -> j`, the returned tilt is
`phi(j) - phi(i)`. The orientation is only a storage convention: reversing the
edge reverses the tilt.

```python
# cookbook: test
import networkx as nx
import numpy as np

from fitness_landscape import BaseNumpySequence, FitnessLandscape
from fitness_landscape.analysis import calculate_polarity_field

sequences = [BaseNumpySequence([i], sequence_id=f"s{i}") for i in range(2)]
graph = nx.path_graph(2)
for index, sequence in enumerate(sequences):
    graph.nodes[index]["sequence"] = sequence
landscape = FitnessLandscape.build(sequences, graph=graph)

result = calculate_polarity_field(
    landscape,
    reference_indices=[0],
    observed_indices=[1],
    weight_key=None,
    diffusion_time=0.5,
)
nodes = result["node_field"]
edges = result["edge_field"]
assert nodes.loc[0, "polarity"] < 0 < nodes.loc[1, "polarity"]
np.testing.assert_allclose(
    edges.loc[0, "tilt"],
    nodes.loc[1, "polarity"] - nodes.loc[0, "polarity"],
)
assert result["metadata"]["pseudocount"] is None
```

The returned node table retains the reference and observed probabilities,
enrichment and local KL contribution from `calculate_graph_occupancy`. Its
`support` column distinguishes `shared`, `observed_only`, `reference_only` and
`zero_mass` nodes. No pseudocount is added: one-sided support produces infinite
polarity, and zero mass from both sources is undefined. Edge tilt is defined
only where both endpoint polarities are finite.

Heat time is a graph smoothing scale measured in jump-rate units, not
biological time. Results at different times are different estimands and should
be reported separately. Increasing time eventually erases within-component
source differences; disconnected components never exchange mass.

This is a descriptive potential-of-mean-force-style field relative to the
chosen finite reference panel. Interpreting it as evolutionary force or energy
requires an independently justified dynamical model whose reference stationary
distribution and transition geometry have biological meaning. The field alone
does not establish selection, equilibrium, foldability or mutational
accessibility.
