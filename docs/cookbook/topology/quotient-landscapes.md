# Build quotient landscapes

A quotient landscape collapses nodes into declared groups. It is useful for
background, taxonomy, or experimental-class summaries, but aggregation removes
within-group variation and multiplicity.

## Install and input

```bash
python -m pip install landscapy
```

The partition may be an annotation layer, mapping, or aligned sequence. This
example partitions a complete two-site binary cube by a `group` annotation and
uses explicit aggregation rules.

## Worked example

```python
# cookbook: test
import networkx as nx
import numpy as np

from fitness_landscape import (
    CategoricalFitness,
    FitnessLandscape,
    NumericFitness,
    ProbabilisticCategoricalFitness,
    generate_sequences,
)
from fitness_landscape.core import AnnotationLayer

sequences = generate_sequences(length=2, alphabet=[0, 1])
groups = ["A", "A", "B", "B"]
landscape = FitnessLandscape.build(
    sequences,
    graph="hamming",
    fitness_layers={
        "assay": NumericFitness("assay", [1.0, 3.0, 5.0, 7.0]),
        "class": CategoricalFitness(
            "class", ["low", "low", "high", "high"], categories=["low", "high"]
        ),
    },
    annotation_layers={
        "groups": AnnotationLayer(
            "groups",
            {"group": groups, "family": ["f1", "f1", "f2", "f3"]},
        )
    },
)
for edge_index, (u, v) in enumerate(landscape.graph.edges, start=1):
    landscape.graph[u][v]["support"] = float(edge_index)

quotient = landscape.quotient_landscape(
    partition="groups",
    annotation_field="group",
    aggregation_function="mean",
    aggregate_annotations=True,
    aggregate_edge_attributes=True,
    edge_attributes=["support"],
    edge_aggregation_function="mean",
)

np.testing.assert_allclose(quotient.get_layer("assay").to_scalar(), [2.0, 6.0])
assert isinstance(quotient.get_layer("class"), ProbabilisticCategoricalFitness)
np.testing.assert_allclose(
    quotient.get_layer("class").probabilities, [[1.0, 0.0], [0.0, 1.0]]
)
annotations = quotient.get_annotation_layer("groups").to_dataframe()
assert annotations["group"].tolist() == ["A", "B"]
assert annotations["family"].tolist() == ["f1", "f2;f3"]
assert quotient.graph.number_of_nodes() == 2
assert quotient.graph.number_of_edges() == 1
assert "support" in next(iter(quotient.graph.edges(data=True)))[2]

# Audit source-pure topology before collapse.
node_groups = {landscape.sequence_index_to_node[i]: group for i, group in enumerate(groups)}
pure_components = {
    group: nx.number_connected_components(
        landscape.graph.subgraph([node for node, label in node_groups.items() if label == group])
    )
    for group in sorted(set(groups))
}
assert pure_components == {"A": 1, "B": 1}
print(annotations.to_dict(orient="records"))
print(list(quotient.graph.edges(data=True)), pure_components)
```

The quotient has two nodes and one cross-group edge. Numeric fitness is averaged;
categorical fitness becomes a probability distribution; annotation strings are
deduplicated and joined; the selected edge attribute is averaged over source
edges.

## Interpretation

The result supports a group-level summary under the declared aggregation rules.
It no longer contains within-group fitness variance, individual edge counts,
node multiplicity, or the identities of alternative paths unless those are
recorded separately. A connected quotient can join groups even when the source
support within one group is fragmented, so report source-pure components and
cross-group edge counts before collapse.

## Common failures

- Partition labels are misaligned with graph node order.
- Mean aggregation is applied to incompatible assay scales or categorical codes.
- Edge attributes with different semantics are aggregated together.
- A quotient edge is interpreted as one biological transition although it may
  summarize many observed edges.
- Source component structure and group sizes are discarded from the report.
