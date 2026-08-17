# Construct OHE kNN graphs from non-binary sequences

kNN fixes a local neighbour count rather than an absolute mutation radius. It
is therefore a different representation from an exact single-substitution
graph even when both use Hamming geometry. The sequence alphabet does not have
to be binary: aligned protein, DNA, RNA, or other categorical sequences can be
represented in the OHE domain.

## Input

OHE/sequence-domain search requires equal aligned length and compatible symbol
alphabets. `BaseNumpySequence` is the general multi-allelic sequence class;
`BinarySequence` is not required. `k` counts directed non-self candidates
before the graph is symmetrized by the union rule.

## Worked example

```python
# cookbook: test
import networkx as nx
import numpy as np

from fitness_landscape import BaseNumpySequence, FitnessLandscape
from fitness_landscape.core import create_knn_graph

texts = ["MKT", "MKS", "MRT", "AKT", "ART", "ARS", "CRS", "CRD"]
sequences = [
    BaseNumpySequence.from_string(text, sequence_id=f"protein-{index}")
    for index, text in enumerate(texts)
]

# The default BaseNumpySequence string alphabet is the 20 canonical amino
# acids. Landscapy constructs this sequence-domain representation internally;
# it is shown here so its dimensions and row order are explicit.
ohe = np.stack([sequence.to_one_hot().reshape(-1) for sequence in sequences])
assert ohe.shape == (8, 3 * 20)
assert np.all(ohe.reshape(8, 3, 20).sum(axis=2) == 1)

graph = create_knn_graph(
    sequences,
    k=2,
    embedding_domain="ohe",
    backend="balltree",
    tiebuffer=8,
    tie_policy="all",
)
landscape = FitnessLandscape(
    sequences,
    graph,
    embeddings={"ohe": ohe},
    active_embedding_domain="ohe",
)

assert landscape.active_embedding_domain == "ohe"
assert graph.number_of_edges() == 15
assert nx.is_connected(graph)
assert graph.graph["landscapy_knn_search"] == {
    "role": "graph",
    "backend": "balltree",
    "metric": "hamming",
    "distance_geometry": "hamming",
    "embedding_domain": "ohe",
}

# Edge distances are counts of mismatched aligned residues, not binary XORs.
for source, target, data in graph.edges(data=True):
    expected = sum(
        left != right
        for left, right in zip(
            sequences[source].to_array(), sequences[target].to_array()
        )
    )
    assert data["distance"] == expected

assert {data["distance"] for _, _, data in graph.edges(data=True)} == {1.0, 2.0}
print(landscape, graph.number_of_edges(), graph.graph["landscapy_knn_search"])
```

The eight amino-acid sequences produce a connected, 15-edge graph. Both
single- and double-substitution edges occur because kNN selects the nearest
observed rows; it does not require every selected neighbour to be a
single-substitution variant. The explicit OHE array is attached to the
landscape for downstream use, but `create_knn_graph` derives its OHE-domain
search representation directly from `sequences`.

For a high-level construction, the equivalent graph can be created with
`FitnessLandscape.build(sequences, graph="knn", embedding_domain="ohe", k=2,
backend="balltree", tie_policy="all")`.

## Backend and tie choices

BallTree is the portable exact backend. FAISS `flat` is exact; HNSW and IVF are
approximate and require index parameters, software/hardware provenance, and a
recall check against an exact subset. `tiebuffer` controls how many additional
returned candidates are inspected for equality at the kth distance; it does
not make an approximate index exact. `tie_policy="all"` retains every candidate
at the kth distance; `min_index` is row-order dependent, and `random` requires a
recorded seed. The undirected edge set is the union of directed selections, so
final degrees can exceed `k`.

## Common failures

- Reporting `k` as the final undirected degree.
- Supplying unaligned or unequal-length sequences to the OHE domain.
- Letting individual sequences infer incompatible alphabets instead of using a
  common explicit alphabet for non-protein categorical data.
- Interpreting OHE/Hamming distance as biochemical similarity: every symbol
  substitution has the same cost.
- Breaking ties by row index without documenting row order.
- Omitting the random seed or FAISS index configuration.
- Comparing approximate and exact graphs without a recall/sensitivity audit.
- Choosing `k` after inspecting the preferred biological conclusion.
