# Benchmark kNN backends

Benchmark neighbour backends on a fixed feature matrix before selecting one.
BallTree and flat FAISS are exact; HNSW and IVF are approximate candidate
indices. Approximation can alter graph topology, not only runtime.

## Install and input

```bash
python -m pip install "landscapy[faiss]"
```

This synthetic continuous embedding has a fixed seed. Runtime and Python-level
peak memory are deployment measurements, not portable expected values.

## Worked example

```python
# cookbook: test
import importlib.util
import time
import tracemalloc

import numpy as np

from fitness_landscape import BaseNumpySequence
from fitness_landscape.core import create_knn_graph

rng = np.random.default_rng(229)
embeddings = rng.normal(size=(128, 12)).astype(np.float32)
sequences = [BaseNumpySequence([row], sequence_id=f"row-{row}") for row in range(128)]
settings = {
    "balltree": {"backend": "balltree", "exact": True},
    "faiss_flat": {"backend": "faiss", "index_type": "flat", "exact": True},
    "faiss_hnsw": {"backend": "faiss", "index_type": "hnsw", "hnsw_M": 16, "exact": False},
    "faiss_ivf": {"backend": "faiss", "index_type": "ivf", "exact": False},
}

def edge_set(graph):
    return {tuple(sorted(edge)) for edge in graph.edges()}

report = {}
reference_edges = None
faiss_available = importlib.util.find_spec("faiss") is not None
for name, config in settings.items():
    if config["backend"] == "faiss" and not faiss_available:
        report[name] = {"status": "unavailable", "fallback": "balltree"}
        continue
    arguments = {key: value for key, value in config.items() if key != "exact"}
    tracemalloc.start()
    started = time.perf_counter()
    graph = create_knn_graph(
        sequences,
        k=8,
        embeddings=embeddings,
        embedding_domain="plm",
        tie_policy="min_index",
        seed=229,
        **arguments,
    )
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    edges = edge_set(graph)
    if name == "balltree":
        reference_edges = edges
    report[name] = {
        "status": "ok",
        "exact": config["exact"],
        "metric": graph.graph["landscapy_knn_search"]["metric"],
        "index_parameters": arguments,
        "tie_policy": "min_index",
        "runtime_seconds": elapsed,
        "python_peak_bytes": peak,
        "edges": len(edges),
        "edge_recall_vs_exact": len(edges & reference_edges) / len(reference_edges),
        "edge_difference": len(edges ^ reference_edges),
    }

assert report["balltree"]["edge_recall_vs_exact"] == 1.0
assert report["balltree"]["metric"] == "euclidean"
if faiss_available:
    assert report["faiss_flat"]["edge_recall_vs_exact"] == 1.0
    assert all(report[name]["status"] == "ok" for name in settings)
print(report)
```

For large data, repeat the recall calculation on a preregistered exact subset
and record its selection rule. `python_peak_bytes` excludes much native-library
allocation, so add process-level CPU/GPU memory monitoring for production.

## Common failures

- Approximate recall is measured against the same approximate index.
- Inner product and L2 results are compared as if they represented one metric.
- `k`, HNSW `M`, IVF defaults, tie policy, or seed are omitted from provenance.
- Only runtime is reported while edge differences are ignored.
- A changed candidate universe is described as the same scientific graph.
