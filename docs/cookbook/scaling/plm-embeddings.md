# Batch, cache, and audit PLM embeddings

Treat PLM embeddings as versioned derived data. Preserve caller sequence order,
model and tokenizer provenance, dtype, pooling contract, device, and a content
fingerprint before reusing one array across graph views.

## Input

`ESMEmbedder` selects CUDA when available unless a device is supplied. Tune
batch size against measured accelerator memory; sequence length, not only row
count, determines token memory.

```python
from fitness_landscape.embedding import ESMEmbedder

embedder = ESMEmbedder(
    model_name="facebook/esm2_t6_8M_UR50D",
    device="cpu",  # use "cuda" only after checking available memory
    batch_size=16,
)
embeddings = embedder.embed_sequences(sequence_strings)
embedder.save_embeddings(embeddings, cache_path)
```

## Worked cache audit

```python
# cookbook: test
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from fitness_landscape import BaseNumpySequence
from fitness_landscape.core import create_knn_graph

cache_path = Path("docs/cookbook/data/toy_protein_embeddings.csv")
cache = pd.read_csv(cache_path)
sequence_strings = cache.pop("sequence").tolist()
embeddings = cache.to_numpy(dtype=np.float32)
sequences = [BaseNumpySequence(text, sequence_id=f"toy-protein-{i}") for i, text in enumerate(sequence_strings)]
fingerprint = hashlib.sha256(cache_path.read_bytes()).hexdigest()
provenance = {
    "source_model": "facebook/esm2_t6_8M_UR50D",
    "source_model_revision": "c731040fcd8d73dceaa04b0a8e6329b345b0f5df",
    "source_pooling": "mean residue positions; special/padding tokens excluded",
    "cache_transform": "four-component full-SVD PCA",
    "device": "cpu",
    "dtype": str(embeddings.dtype),
    "shape": list(embeddings.shape),
    "array_bytes": embeddings.nbytes,
    "sha256": fingerprint,
}

assert sequence_strings == ["".join(map(str, sequence.to_array())) for sequence in sequences]
assert embeddings.shape == (6, 4) and np.isfinite(embeddings).all()
assert fingerprint == "d0783f5e58049f72bfa0c317ae8b2f854e15c9642a62e7e505929878dd478243"
graphs = {
    k: create_knn_graph(
        sequences, k=k, embeddings=embeddings, embedding_domain="plm", backend="balltree"
    )
    for k in (2, 3)
}
assert all(graph.number_of_nodes() == len(sequences) for graph in graphs.values())
assert all(graph.graph["landscapy_knn_search"]["embedding_domain"] == "plm" for graph in graphs.values())
print(provenance, {k: graph.number_of_edges() for k, graph in graphs.items()})
```

The fixture is an offline PCA cache derived from ESM output, not the raw model
array. A new model revision, tokenizer, pooling rule, alphabet mapping, or PCA
fit creates a new feature artifact and requires a new fingerprint.
