# Import external NetworkX graphs

Use an external graph when adjacency was defined outside Landscapy and can be
stated as an undirected relation over the same sequences.

## Input

Every node must carry a public sequence object in its `sequence` attribute.
`FitnessLandscape.from_graph` also reconstructs node attributes named
`fitness_<layer>`. Arbitrary hashable node labels are allowed; list order is not
a substitute for checking the mapping.

## Worked example

```python
# cookbook: test
import networkx as nx
import numpy as np

from fitness_landscape import BinarySequence, FitnessLandscape, NumericFitness

sequences = [
    BinarySequence("000", sequence_id="s0"),
    BinarySequence("001", sequence_id="s1"),
    BinarySequence("011", sequence_id="s2"),
]
labels = ["wild-type", 17, ("sample", 2)]

external = nx.Graph(source="external edge table v1")
for label, sequence in zip(labels, sequences):
    external.add_node(label, sequence=sequence)
external.add_edges_from([("wild-type", 17), (17, ("sample", 2))])

landscape = FitnessLandscape.build(
    sequences,
    graph=external,
    fitness_layers={"assay": NumericFitness("assay", [0.1, 0.4, 0.8])},
)
assert landscape.node_to_sequence_index == {
    "wild-type": 0,
    17: 1,
    ("sample", 2): 2,
}
assert landscape.sequence_index_to_node[2] == ("sample", 2)
assert landscape.graph.nodes[17]["fitness_assay"] == [0.4]

# The same topology can arrive as an adjacency matrix, provided nodelist order
# is explicit and node attributes are restored before wrapping it.
adjacency = nx.to_numpy_array(external, nodelist=labels, dtype=int)
np.testing.assert_array_equal(
    adjacency,
    np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]),
)
matrix_graph = nx.from_numpy_array(adjacency)
for index, sequence in enumerate(sequences):
    matrix_graph.nodes[index]["sequence"] = sequence
    matrix_graph.nodes[index]["fitness_assay"] = [0.1, 0.4, 0.8][index]

restored = FitnessLandscape.from_graph(matrix_graph)
alias = FitnessLandscape.from_graph_annotated(matrix_graph)
assert restored.graph.number_of_edges() == 2
assert restored.node_to_sequence_index == {0: 0, 1: 1, 2: 2}
np.testing.assert_allclose(restored.get_layer("assay").to_scalar(), [0.1, 0.4, 0.8])
np.testing.assert_allclose(alias.get_layer("assay").to_scalar(), [0.1, 0.4, 0.8])

print(landscape.node_to_sequence_index)
print(adjacency.tolist())
```

The expected mapping preserves the three arbitrary labels and the expected
adjacency is a three-node path. `from_graph_annotated` is an explicit alias for
the same annotated-graph contract.

## Interpretation and edge semantics

Wrapping a graph validates node/sequence alignment; it does not validate why an
edge exists. Declare raw distance, normalized distance, affinity, and
conductance using the canonical keys in the [edge-semantics
contract](../graph-construction/edge-semantics.md). Do not place a raw distance in NetworkX
`weight`, because weighted Landscapy analyses interpret `weight` as
conductance.

Landscapy 0.9 supports undirected graphs only. Reject or explicitly transform a
directed input upstream; silent symmetrization changes the scientific model.

## Common failures

- A node lacks `sequence`, or graph sequences do not match the supplied list.
- The adjacency nodelist differs from the sequence/fitness order.
- A `fitness_<name>` attribute is absent on some nodes, producing an incomplete
  reconstructed layer.
- Parallel edges or self-loops have not been reduced according to a declared
  rule before conversion to a simple graph.
- An ambiguous `weight` key mixes distance and conductance semantics.
