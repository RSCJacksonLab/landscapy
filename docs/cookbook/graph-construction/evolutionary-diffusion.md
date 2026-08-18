# Construct an evolutionary-diffusion graph

Evolutionary diffusion uses embedding kNN only as a pair prefilter, aligns those
pairs, scores them with an instantaneous amino-acid rate generator, and then
diffuses a reversible affinity.

## Input

Sequences must use the 20-symbol alphabet in the package's documented
alphabetical order. Embeddings must align with sequence order. An external
matrix must be a reversible instantaneous generator: non-negative off-diagonal
rates, zero row sums, and a declared equilibrium distribution.

## Worked example

```python
# cookbook: test
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from fitness_landscape import BaseNumpySequence
from fitness_landscape.analysis import graph_properties
from fitness_landscape.core import create_evol_diffusion_graph, create_hamming_graph

cache = pd.read_csv(Path("docs/cookbook/data/toy_protein_embeddings.csv"))
alphabet = list("ACDEFGHIKLMNPQRSTVWY")
sequences = [
    BaseNumpySequence(list(text), sequence_id=f"toy-{i}", alphabet=alphabet)
    for i, text in enumerate(cache.sequence)
]
embeddings = cache[["pc1", "pc2", "pc3", "pc4"]].to_numpy()

parameters = {
    "evolutionary_time": 0.5,
    "t": 2,
    "tau": 1.0,
    "k": 2,
    "connectivity_threshold": 0.01,
    "cpus": 1,
}

bundled_lg = create_evol_diffusion_graph(
    sequences,
    embeddings,
    replacement_matrix=None,
    backend="balltree",
    embedding_domain="plm",
    **parameters,
)

# A simple uniform reversible generator, validated before use. It is a software
# example, not a defensible amino-acid model for biological inference. A biologically
# realistic reversible generator may, for example, be the LG or WAG replacement matrices,
# in which replacement rates are estimated from large alignments. Note that more modern
# NQ models are not compatible with evol diffusion as they are no time reverisble with 
# asymmetric off-diagonal entries.

external_generator = np.full((20, 20), 1.0 / 19.0)
np.fill_diagonal(external_generator, -1.0)
equilibrium = np.full(20, 1.0 / 20.0)
assert np.allclose(external_generator.sum(axis=1), 0.0)
assert np.all(external_generator[~np.eye(20, dtype=bool)] >= 0.0)
assert np.allclose(
    equilibrium[:, None] * external_generator,
    equilibrium[None, :] * external_generator.T,
)
external = create_evol_diffusion_graph(
    sequences,
    embeddings,
    replacement_matrix=external_generator,
    equilibrium_frequencies=equilibrium,
    backend="balltree",
    embedding_domain="plm",
    **parameters,
)

hamming = create_hamming_graph(sequences)
diagnostics = {}
for name, graph in {"bundled_lg": bundled_lg, "external": external, "hamming": hamming}.items():
    properties = graph_properties(graph)
    diagnostics[name] = {
        "edges": graph.number_of_edges(),
        "density": properties["density"],
        "components": properties["components"],
        "degree": properties["degree"],
        "isolates": list(nx.isolates(graph)),
    }

assert bundled_lg.graph["landscapy_edge_schema"]["constructor"] == "evolutionary-diffusion"
assert bundled_lg.graph["landscapy_knn_search"]["role"] == "prefilter"
assert bundled_lg.graph["diffusion_semantics"]["power"] == 2
assert all(data["weight"] == data["affinity"] for _, _, data in bundled_lg.edges(data=True))
assert diagnostics["bundled_lg"]["components"]["count"] == 1
print(parameters)
print(diagnostics)
```

Both evolutionary graphs retain a connected affinity graph for this fixture;
the Hamming comparator contains only exact observed single substitutions. Exact
edge counts can change with the generator and declared scales, so they are
reported rather than treated as universal expected values.

## Keep the scales separate

- `evolutionary_time` exponentiates the instantaneous generator.
- `tau` temperatures the length-normalized alignment log-odds.
- `k` limits embedding-neighbour pairs sent to alignment.
- graph `t` is the lazy-walk diffusion power after affinity construction.
- `connectivity_threshold` filters the final symmetric diffusion amplitude.
- `cpus` configures the Ray alignment runtime; this example uses one CPU.

Store this parameter record with the graph because not every scientific parameter is
encoded in graph metadata. 

## Interpretation and failures

The result is a kNN-prefiltered alignment-affinity graph. It is not a
phylogenetic tree, branch-length estimate, or test of common ancestry. Common
failures include passing a probability matrix instead of an instantaneous
generator, using the wrong amino-acid order, conflating `evolutionary_time`
with graph `t`, omitting Ray configuration, ignoring pairs excluded by kNN, and
selecting `tau` or threshold after seeing a preferred conclusion.
