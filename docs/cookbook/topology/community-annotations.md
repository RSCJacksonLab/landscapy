# Attach external community annotations

Landscapy 0.9 does not expose a built-in community-detection API. Run a named
external NetworkX algorithm, record its parameters, then attach its output as an
annotation for auditable downstream workflows.

## Install and input

```bash
python -m pip install landscapy
```

This example uses NetworkX Louvain communities on a simple undirected,
conductance-weighted graph. The algorithm, weight key, resolution, and seed are
fixed. Edge `weight` means conductance under the Landscapy [edge-semantics
contract](../graph-construction/edge-semantics.md).

## Worked example

```python
# cookbook: test
import networkx as nx
import pandas as pd

from fitness_landscape import BaseNumpySequence, FitnessLandscape
from fitness_landscape.analysis import category_diffusion_hierarchy
from fitness_landscape.core import AnnotationLayer

graph = nx.Graph()
graph.add_edges_from(
    [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)],
    weight=1.0,
)
graph.add_edge(2, 3, weight=0.1)
sequences = [BaseNumpySequence([i], sequence_id=f"s{i}") for i in range(6)]
for node, sequence in zip(graph.nodes, sequences):
    graph.nodes[node]["sequence"] = sequence
landscape = FitnessLandscape(sequences, graph)

algorithm = "networkx.community.louvain_communities"
resolution = 1.0
seed = 31
communities = nx.community.louvain_communities(
    graph, weight="weight", resolution=resolution, seed=seed
)
community_by_node = {
    node: community_id
    for community_id, members in enumerate(communities)
    for node in members
}
community_ids = [community_by_node[node] for node in graph.nodes]
modularity = nx.community.modularity(graph, communities, weight="weight", resolution=resolution)

landscape.attach_annotation(
    AnnotationLayer(
        "communities",
        {"community": community_ids},
        metadata={
            "algorithm": algorithm,
            "weight_key": "weight",
            "resolution": resolution,
            "seed": seed,
            "modularity": modularity,
        },
    )
)
landscape.attach_annotation(
    AnnotationLayer("known", {"group": ["left"] * 3 + ["right"] * 3})
)

sizes = sorted(len(members) for members in communities)
comparison = pd.crosstab(
    landscape.get_annotation_layer("known").to_dataframe()["group"],
    landscape.get_annotation_layer("communities").to_dataframe()["community"],
)
quotient = landscape.quotient_landscape(
    partition="communities", annotation_field="community"
)
hierarchy = category_diffusion_hierarchy(
    landscape,
    layer="communities",
    annotation_field="community",
    embedding_dim=2,
    weight_key="weight",
    filter_small_embedding=False,
)

assert sizes == [3, 3]
assert modularity > 0.45
assert quotient.graph.number_of_nodes() == 2
assert quotient.graph.number_of_edges() == 1
assert hierarchy["pairwise_distances"].shape == (2, 2)
assert comparison.to_numpy().max() == 3

report = {
    "algorithm": algorithm,
    "weight_key": "weight",
    "resolution": resolution,
    "seed": seed,
    "modularity": modularity,
    "community_sizes": sizes,
}
print(report)
print(comparison)
```

The algorithm recovers two communities of three nodes connected by one weak
bridge. The cross-tab compares them with known metadata; the quotient and
category hierarchy provide group-level views without converting community IDs
to fitness.

## Interpretation

Community assignments and modularity describe the supplied graph under one
algorithm and parameter set. Integer community labels are arbitrary and
non-identifiable across runs; compare member sets, not label numbers. Agreement
with known metadata is descriptive unless a valid, component-aware null model
and test statistic were specified in advance.

## Common failures

- Algorithm, NetworkX version, resolution, seed, or weight key is omitted.
- Raw distances are passed as `weight` even though Louvain treats larger values
  as stronger connections.
- Community IDs are treated as ordered or stable across runs.
- A high modularity value is interpreted as evidence for biological classes
  without a comparator.
- Disconnected components force communities, but their support structure is not
  reported separately.
