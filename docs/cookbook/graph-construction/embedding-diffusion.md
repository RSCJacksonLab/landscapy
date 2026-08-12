# Construct an embedding-diffusion graph

Embedding diffusion starts from sparse Euclidean kNN candidate support, applies
an RBF affinity, constructs a lazy reversible walk, and thresholds a symmetric
finite-time or stationary diffusion kernel.

## Install and input

```bash
python -m pip install "landscapy[knn]"
```

The embedding matrix must be finite, two-dimensional, and aligned with the
sequence list. This recipe uses the versioned PLM/PCA cache. Read the [diffusion
contract](../../diffusion_semantics.md) and [kNN-domain
contract](../../knn_embedding_domains.md).

## Worked example

```python
# cookbook: test
from pathlib import Path

import networkx as nx
import pandas as pd

from fitness_landscape import BaseNumpySequence
from fitness_landscape.core import create_diffusion_emb_graph

cache = pd.read_csv(Path("docs/cookbook/data/toy_protein_embeddings.csv"))
alphabet = list("ACDEFGHIKLMNPQRSTVWY")
sequences = [BaseNumpySequence(list(text), alphabet=alphabet) for text in cache.sequence]
embeddings = cache[["pc1", "pc2", "pc3", "pc4"]].to_numpy()

def build(t, threshold):
    return create_diffusion_emb_graph(
        sequences,
        embeddings,
        k=2,
        backend="balltree",
        embedding_domain="plm",
        t=t,
        connectivity_threshold=threshold,
        max_diffusion_nnz=10_000,
        max_diffusion_work=100_000,
    )

finite_t1 = build(1, 1e-4)
finite_t2 = build(2, 1e-4)
thresholded_t2 = build(2, 0.15)
stationary = build(None, 1e-4)

def audit(graph):
    construction = graph.graph["diffusion_construction"]
    return {
        "edges": graph.number_of_edges(),
        "density": nx.density(graph),
        "components": nx.number_connected_components(graph),
        "candidate_directed": construction["directed_candidates"],
        "candidate_affinity_nnz": construction["affinity_nnz"],
        "kernel_nnz": construction["kernel_nnz"],
        "power": graph.graph["diffusion_semantics"]["power"],
        "threshold": graph.graph["diffusion_semantics"]["threshold"],
        "estimated_work": construction["estimated_scalar_products"],
    }

diagnostics = {
    "finite_t1": audit(finite_t1),
    "finite_t2": audit(finite_t2),
    "thresholded_t2": audit(thresholded_t2),
    "stationary": audit(stationary),
}
assert diagnostics["finite_t1"]["edges"] == 7
assert diagnostics["finite_t2"]["edges"] == 11
assert diagnostics["thresholded_t2"]["edges"] == 7
assert diagnostics["stationary"]["edges"] == 15
assert diagnostics["stationary"]["power"] == "componentwise_stationary_limit"
assert finite_t2.graph["landscapy_knn_search"]["role"] == "prefilter"
assert all(data["weight"] == data["affinity"] for _, _, data in finite_t2.edges(data=True))
print(diagnostics)
```

Finite `t=1` retains only the seven candidate-support edges. At `t=2`, exact
sparse propagation creates eleven retained pairs; a higher threshold removes
weak amplitudes. The stationary limit is dense within the connected component.
These counts are fixture-specific diagnostics, not recommended defaults.

## Interpretation

The sparse kNN union defines where the RBF affinity is evaluated; it is not a
dense all-pairs kernel. The walk is 0.5-lazy and reversible. Finite integer `t`
is a graph diffusion power; `None`, zero, or infinity requests the
component-wise stationary limit. Retained `affinity` and `weight` are the same
dimensionless symmetric amplitude, with `weight` serving as conductance.

`max_diffusion_nnz` and `max_diffusion_work` are hard guards for exact sparse
computation. A `MemoryError` is a request to revise scale or parameters, not to
silently densify or approximate the graph.

## Common failures

- Confusing sparse candidate support with the final diffusion edge set.
- Treating finite `t` and the stationary limit as interchangeable.
- Comparing thresholds without their dimensionless kernel scale.
- Ignoring components created by thresholding.
- Raising resource guards until an unreviewed job exhausts memory.
