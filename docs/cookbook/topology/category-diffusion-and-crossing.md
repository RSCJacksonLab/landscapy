# Analyse category diffusion and boundary crossing

Category diffusion compares group locations in a graph embedding. Boundary
crossing simulates first visits between categories. Both depend on graph and
transition semantics and are descriptive unless paired with a predeclared null
or permutation design.

## Input

The landscape needs a categorical or probabilistic-categorical fitness layer.
Probability rows must sum to one. This connected path graph uses unweighted
transitions explicitly; a conductance-weighted analysis must name the canonical
conductance key from the [edge-semantics contract](../graph-construction/edge-semantics.md).

## Worked example

```python
# cookbook: test
import numpy as np
import networkx as nx

from fitness_landscape import (
    BinarySequence,
    FitnessLandscape,
    ProbabilisticCategoricalFitness,
)
from fitness_landscape.analysis import (
    category_boundary_crossing_times,
    category_diffusion_hierarchy,
)

texts = ["000", "001", "010", "011", "100", "101", "110", "111"]
sequences = [BinarySequence(text, sequence_id=f"s{i}") for i, text in enumerate(texts)]
probabilities = np.array(
    [
        [0.90, 0.10], [0.85, 0.15], [0.75, 0.25], [0.35, 0.65],
        [0.80, 0.20], [0.30, 0.70], [0.15, 0.85], [0.05, 0.95],
    ]
)
layer = ProbabilisticCategoricalFitness(
    "activity", probabilities, categories=["low", "high"]
)
graph = nx.path_graph(len(sequences))
for node, sequence in zip(graph.nodes, sequences):
    graph.nodes[node]["sequence"] = sequence
landscape = FitnessLandscape(
    sequences, graph, fitness_layers={"activity": layer}
)
landscape.view("activity")

hierarchy = category_diffusion_hierarchy(
    landscape,
    layer="activity",
    embedding_dim=2,
    diffusion_matrix="norm_laplacian",
    weight_key=None,
    skip_first=True,
    filter_small_embedding=False,
)
crossing = category_boundary_crossing_times(
    landscape,
    layer="activity",
    n_walks=80,
    max_steps=20,
    seed=23,
    weight_key=None,
)

assert hierarchy["categories"] == ["low", "high"]
assert hierarchy["embedding"].shape == (8, 2)
assert hierarchy["pairwise_distances"].shape == (2, 2)
assert crossing["mean_crossing_time"].shape == (2, 2)
assert crossing["params"] == {
    "layer": "activity",
    "n_walks": 80,
    "max_steps": 20,
    "seed": 23,
    "weight_key": None,
}
assert np.all(crossing["hit_counts"] <= 80)

# One fixed-seed label permutation illustrates the required null workflow. It
# is not enough permutations for inference and is reported only as a comparator.
rng = np.random.default_rng(29)
null_layer = ProbabilisticCategoricalFitness(
    "activity", probabilities[rng.permutation(len(probabilities))], categories=["low", "high"]
)
null_landscape = FitnessLandscape.build(
    sequences,
    graph=landscape.graph.copy(),
    fitness_layers={"activity": null_layer},
)
null_crossing = category_boundary_crossing_times(
    null_landscape,
    layer="activity",
    n_walks=80,
    max_steps=20,
    seed=29,
    weight_key=None,
)

print(hierarchy["pairwise_distances"])
print(crossing["mean_crossing_time"], crossing["hit_counts"])
print("permuted comparator", null_crossing["mean_crossing_time"])
```

The hierarchy returns category centroids, pairwise distances, spreads, and a
linkage when estimable. Crossing means are conditional on successful hits;
`hit_counts / n_walks` is therefore a required censoring diagnostic. A zero
diagonal is defined by construction.

## Interpretation

The outputs describe separation and simulated passage under the selected graph,
layer probabilities, operator, and walk kernel. They do not prove an evolutionary
barrier. For disconnected support, restrict analysis to scientifically relevant
components or report zero-hit/undefined category pairs; never convert them to a
finite crossing time. Formal inference needs many predeclared component-aware
permutations and a stated statistic, not the single illustrative comparator.

## Common failures

- The wrong active layer or category column is selected.
- Probabilistic category rows or graph-node order are misaligned.
- `weight_key=None` is mistaken for conductance-weighted movement.
- Missed walks are omitted without reporting hit counts and `max_steps`.
- Categories that occupy different components are assigned finite crossings.
