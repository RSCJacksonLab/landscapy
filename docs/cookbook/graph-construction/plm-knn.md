# Construct a PLM-embedding kNN graph

PLM kNN uses Euclidean distance between protein-language-model embeddings. It
does not use site-wise Hamming distance, even when the source sequences are
aligned.

## Install and input

Use `embeddings` to generate ESM features and `knn` to build the graph:

```bash
python -m pip install "landscapy[embeddings,knn]"
```

The offline example loads the versioned four-component ESM/PCA cache described
in the [data provenance](../data/README.md). The CSV primary key is `sequence`;
the four numeric columns are aligned embedding coordinates.

## Worked example: load the cache

```python
# cookbook: test
from pathlib import Path

import pandas as pd

from fitness_landscape import BaseNumpySequence, FitnessLandscape
from fitness_landscape.core import create_hamming_graph, create_knn_graph

cache_path = Path("docs/cookbook/data/toy_protein_embeddings.csv")
cache = pd.read_csv(cache_path)
alphabet = list("ACDEFGHIKLMNPQRSTVWY")
sequences = [
    BaseNumpySequence(list(text), sequence_id=f"toy-protein-{i}", alphabet=alphabet)
    for i, text in enumerate(cache.sequence)
]
embeddings = cache[["pc1", "pc2", "pc3", "pc4"]].to_numpy(dtype=float)
assert ["".join(sequence.to_array()) for sequence in sequences] == cache.sequence.tolist()
assert embeddings.shape == (len(sequences), 4)

graph = create_knn_graph(
    sequences,
    k=2,
    embeddings=embeddings,
    embedding_domain="plm",
    backend="balltree",
    tie_policy="all",
)
provenance = {
    "model_name": "facebook/esm2_t6_8M_UR50D",
    "model_revision": "c731040fcd8d73dceaa04b0a8e6329b345b0f5df",
    "pooling": "Landscapy mean-pooled residues",
    "reduction": "4-component full-SVD PCA on cookbook fixture",
    "cache": str(cache_path),
    "device": "cpu",
    "batch_size": 6,
}
landscape = FitnessLandscape(
    sequences,
    graph,
    embeddings={"plm": embeddings},
    active_embedding_domain="plm",
    embedding_metadata={"plm": provenance},
)

assert landscape.active_embedding_domain == "plm"
assert landscape.get_embedding("plm").shape == (6, 4)
assert landscape.get_embedding_metadata("plm") == provenance
assert graph.graph["landscapy_knn_search"]["metric"] == "euclidean"
assert graph.graph["landscapy_knn_search"]["embedding_domain"] == "plm"
assert set(graph.edges) != set(create_hamming_graph(sequences).edges)
assert all(graph.nodes[node]["sequence"] == sequences[index] for index, node in landscape.sequence_index_to_node.items())
print(graph.number_of_edges(), graph.graph["landscapy_knn_search"])
print(landscape.get_embedding_metadata())
```

The fixture produces seven PLM-kNN edges, a Euclidean search record, and an
active `plm` embedding domain. Its edge set differs from the six exact
single-substitution Hamming edges; neither geometry is a default biological
truth.

## Generate and cache embeddings

Run this explicitly when model download and the `embeddings` extra are
available. Record the resolved model revision as well as model name.

```python
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from fitness_landscape import BaseNumpySequence, FitnessLandscape
from fitness_landscape.core import create_hamming_graph

alphabet = list("ACDEFGHIKLMNPQRSTVWY")
texts = ["ACDEFGH", "ACDEYGH", "ACDDFGH", "VCDEFGH", "VCDEYGH", "VCDDYGH"]
sequences = [BaseNumpySequence(list(text), alphabet=alphabet) for text in texts]
seed_graph = create_hamming_graph(sequences)
seed_landscape = FitnessLandscape(sequences, seed_graph)
full = seed_landscape.compute_plm_embeddings(
    domain="plm",
    model_name="facebook/esm2_t6_8M_UR50D",
    batch_size=6,
    device="cpu",
)
output_dir = Path("artifacts")
output_dir.mkdir(parents=True, exist_ok=True)
np.save(output_dir / "toy_protein_esm2_t6.npy", full)
reduced = PCA(n_components=4, svd_solver="full").fit_transform(full)
pd.DataFrame(reduced, index=texts).to_csv(output_dir / "toy_protein_esm2_t6_pca.csv")
```

`ESMEmbedder` in `fitness_landscape.embedding` provides the same supported
model interface when a landscape object is not yet needed. GPU and CPU outputs
should be versioned and tolerance-checked rather than assumed bit-identical.

## Common failures

- Embedding rows and sequence rows are reordered independently.
- `embedding_domain="ohe"` is used with PLM vectors, silently changing the
  intended geometry.
- Model revision, pooling, reduction, device, or batch size is omitted.
- A reduced documentation cache is reused as a scientific benchmark feature.
- PLM proximity is interpreted as evolutionary or functional equivalence
  without validation.
