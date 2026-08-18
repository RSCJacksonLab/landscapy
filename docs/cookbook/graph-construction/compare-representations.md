# Compare graph representations on identical nodes

Representation sensitivity must hold the node set and node order fixed. Compare
support diagnostics before comparing any fitness estimator.

## Input

This example builds Hamming, PLM kNN, finite-time embedding diffusion, and TDA
views over the same six protein rows and cached embedding order.

## Worked example

```python
# cookbook: test
from pathlib import Path

import networkx as nx
import pandas as pd

from fitness_landscape import BaseNumpySequence
from fitness_landscape.analysis import graph_properties
from fitness_landscape.core import (
    create_diffusion_emb_graph,
    create_hamming_graph,
    create_knn_graph,
    create_tda_graph,
)

cache = pd.read_csv(Path("docs/cookbook/data/toy_protein_embeddings.csv"))
alphabet = list("ACDEFGHIKLMNPQRSTVWY")
sequences = [
    BaseNumpySequence(list(text), sequence_id=f"toy-{i}", alphabet=alphabet)
    for i, text in enumerate(cache.sequence)
]
embeddings = cache[["pc1", "pc2", "pc3", "pc4"]].to_numpy()

graphs = {
    "hamming": create_hamming_graph(sequences),
    "plm_knn": create_knn_graph(
        sequences, k=2, embeddings=embeddings, embedding_domain="plm", backend="balltree"
    ),
    "diffusion": create_diffusion_emb_graph(
        sequences,
        embeddings,
        k=2,
        t=2,
        connectivity_threshold=0.15,
        embedding_domain="plm",
        backend="balltree",
        max_diffusion_nnz=10_000,
        max_diffusion_work=100_000,
    ),
    "tda": create_tda_graph(sequences, embeddings, n_components=3),
}

report = {}
for name, graph in graphs.items():
    assert list(graph.nodes) == list(range(len(sequences)))
    assert all(graph.nodes[index]["sequence"] == sequences[index] for index in graph.nodes)
    properties = graph_properties(graph)
    report[name] = {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "density": properties["density"],
        "components": properties["components"],
        "isolates": list(nx.isolates(graph)),
        "degree": properties["degree"],
        "edge_schema": graph.graph["landscapy_edge_schema"],
        "knn_search": graph.graph.get("landscapy_knn_search"),
        "diffusion": graph.graph.get("diffusion_semantics"),
        "diffusion_construction": graph.graph.get("diffusion_construction"),
        "tda_effective_components": graph.graph.get("tda_effective_components"),
    }

assert {row["nodes"] for row in report.values()} == {6}
assert set(report) == {"hamming", "plm_knn", "diffusion", "tda"}
assert report["hamming"]["edge_schema"]["distance"]["units"] == "hamming_count"
assert report["plm_knn"]["knn_search"]["distance_geometry"] == "euclidean"
assert report["diffusion"]["diffusion"]["power"] == 2
assert report["tda"]["tda_effective_components"] == 3
print(report)
```

The report makes graph-dependent node support, edge count, density, components,
isolates, degree distribution, canonical edge schema, and constructor metadata
comparable without changing node order. On this fixture, the representations
do not have identical topology, so downstream metric eligibility and values can
differ even with identical fitness observations.

## Interpretation and design rule

Choose graph family and parameters a priori from the scientific question and
measurement process. Predeclare a sensitivity set (for example, plausible `k`,
diffusion `t`/threshold, PCA dimension, or exact Hamming) and report all results,
including ineligible component cases. Do not choose the graph post hoc because
it produces the preferred biological conclusion.

Common failures are comparing graphs with reordered or filtered nodes, omitting
edge units, treating denser graphs as inherently better supported, dropping
isolates before reporting denominators, and interpreting agreement between two
related constructors as independent biological evidence.
