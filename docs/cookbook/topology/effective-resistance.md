# Compute effective resistance

Effective resistance is finite only within an electrically connected component
and depends on edge conductance. Read the [component-wise resistance
contract](effective-resistance-contract.md) and [edge-semantics
contract](../graph-construction/edge-semantics.md) first.

## Input

The input must be a simple undirected graph. The selected edge key must contain
finite non-negative conductance, not distance. Node sampling, approximation,
normalization, and jitter settings are part of the method.

## Worked example

```python
# cookbook: test
import networkx as nx
import numpy as np

from fitness_landscape import BaseNumpySequence, CategoricalFitness, FitnessLandscape
from fitness_landscape.analysis import resistance_distance_matrix

connected = nx.path_graph(3)
for u, v in connected.edges:
    connected[u][v]["conductance"] = 2.0
connected_result = resistance_distance_matrix(
    connected,
    weight_key="conductance",
    compute_resistance_matrix=True,
    jitter=1e-10,
    sparse_threshold=1000,
    hutchinson_samples=32,
    hutchinson_seed=17,
    sample_seed=17,
)
np.testing.assert_allclose(
    connected_result["resistance_mat"],
    [[0.0, 0.5, 1.0], [0.5, 0.0, 0.5], [1.0, 0.5, 0.0]],
    atol=1e-10,
)
assert connected_result["weight_key"] == "conductance"
assert connected_result["jitter_used"] is False

disconnected = nx.Graph([(0, 1), (2, 3)])
sequences = [BaseNumpySequence([i], sequence_id=f"s{i}") for i in range(4)]
for node, sequence in zip(disconnected.nodes, sequences):
    disconnected.nodes[node]["sequence"] = sequence
landscape = FitnessLandscape(
    sequences,
    disconnected,
    fitness_layers={
        "classes": CategoricalFitness(
            "classes", ["A", "A", "B", "B"], categories=["A", "B"]
        )
    },
)
disconnected_result = resistance_distance_matrix(
    landscape,
    weight_key=None,
    compute_resistance_matrix=True,
    layers=["classes"],
    aggregation_function="expected_pairwise",
)
resistance = disconnected_result["resistance_mat"]
assert disconnected_result["component_count"] == 2
assert np.isinf(resistance[0, 2])
assert np.isinf(disconnected_result["classes"]["distance_mat"][0, 1])

settings = {
    "weight_key": "conductance",
    "jitter": 1e-10,
    "sparse_threshold": 1000,
    "hutchinson_samples": 32,
    "hutchinson_seed": 17,
    "sample_seed": 17,
    "full_matrix": True,
}
print(connected_result["resistance_mat"])
print(disconnected_result["components"], settings)
```

On a path, resistance adds reciprocal conductances: each edge has resistance
`1 / 2`, so the endpoints are distance 1. Cross-component resistance is
infinite, including the category-level expected-pairwise result.

## Interpretation

Resistance quantifies redundancy of paths under the declared electrical
network. It does not convert disconnected pairs to large finite distances.
`jitter_used` indicates numerical conditioning, not new connectivity. Report
whether the full matrix, Hutchinson approximation, or node sampling was used,
including all seeds and sample counts.

## Common failures

- Passing raw distance as `weight_key` reverses the intended electrical meaning.
- Zero-conductance edges are treated as connected support.
- Infinite cross-component entries are dropped before category aggregation.
- Jitter is described as a biological regularization rather than numerical
  conditioning.
- A sampled or approximate result is reported without its seed and denominator.
