# Construct kNN graphs in sequence space

kNN fixes a local neighbour count rather than an absolute mutation radius. It
is therefore a different representation from an exact single-substitution
graph even when both use Hamming geometry.

## Install and input

```bash
python -m pip install "landscapy[knn]"
```

OHE/sequence-domain search requires equal aligned length. `k` counts directed
non-self candidates before the graph is symmetrized by the union rule.

## Worked example

```python
# cookbook: test
import networkx as nx

from fitness_landscape import BinarySequence
from fitness_landscape.analysis import graph_properties
from fitness_landscape.core import create_knn_graph

sequences = [BinarySequence(f"{value:03b}", sequence_id=f"s{value}") for value in range(8)]

def build(k, tie_policy, seed=41):
    return create_knn_graph(
        sequences,
        k=k,
        embedding_domain="ohe",
        backend="balltree",
        tiebuffer=4,
        tie_policy=tie_policy,
        seed=seed,
    )

k1_all = build(1, "all")
k4_all = build(4, "all")
k1_min = build(1, "min_index")
k1_random = build(1, "random")

audit = {}
for name, graph in {
    "k1_all": k1_all,
    "k4_all": k4_all,
    "k1_min": k1_min,
    "k1_random": k1_random,
}.items():
    properties = graph_properties(graph)
    audit[name] = {
        "edges": graph.number_of_edges(),
        "components": properties["components"],
        "degree": properties["degree"],
        "isolates": list(nx.isolates(graph)),
        "search": graph.graph["landscapy_knn_search"],
    }

assert audit["k1_all"]["edges"] == 12  # all three tied one-mutant neighbours
assert audit["k4_all"]["edges"] == 24  # distance-two ties also enter
assert audit["k1_min"]["edges"] == 6
assert audit["k1_random"]["edges"] == 7
assert audit["k1_all"]["search"] == {
    "role": "graph",
    "backend": "balltree",
    "metric": "hamming",
    "distance_geometry": "hamming",
    "embedding_domain": "ohe",
}
assert audit["k1_all"]["components"]["count"] == 1
assert audit["k4_all"]["components"]["count"] == 1
assert audit["k1_random"]["components"]["count"] == 1
assert audit["k1_min"]["components"]["count"] == 3
print(audit)
```

Because the binary cube has ties, `k=1, tie_policy="all"` retains three
neighbours per node. `min_index` forces exactly one directed choice before union
symmetrization, is index-dependent, and fragments this fixture into three
components; `random` is reproducible only with its seed. The `k=1` versus `k=4`
comparison shows that increasing `k` changes density and degree even on
identical nodes.

## Backend and tie choices

BallTree is the portable exact backend. FAISS `flat` is exact; HNSW and IVF are
approximate and require index parameters, software/hardware provenance, and a
recall check against an exact subset. `tiebuffer` controls how many additional
returned candidates are inspected for equality at the kth distance; it does
not make an approximate index exact. The undirected edge set is the union of
directed selections, so final degrees can exceed `k`.

## Common failures

- Reporting `k` as the final undirected degree.
- Breaking ties by row index without documenting row order.
- Omitting the random seed or FAISS index configuration.
- Comparing approximate and exact graphs without a recall/sensitivity audit.
- Choosing `k` after inspecting the preferred biological conclusion.
