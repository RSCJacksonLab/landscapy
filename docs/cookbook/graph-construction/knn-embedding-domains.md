# kNN embedding domains

<!-- cookbook: reference -->

The embedding domain determines the scientific geometry of every
nearest-neighbour search used during graph construction.

| Domain | Search features | Distance |
| --- | --- | --- |
| `ohe` or no embeddings | aligned categorical sequence/OHE representation | Hamming |
| `plm` | supplied protein language-model embeddings | Euclidean |
| `composition` | supplied composition embeddings | Euclidean |
| direct embeddings with no domain | supplied embedding matrix | Euclidean |

This contract applies both to `create_knn_graph` and to the sparse kNN
prefilters inside `create_diffusion_emb_graph` and
`create_evol_diffusion_graph`. For a FAISS backend, Euclidean searches always
use an L2 index. The `faiss_metric` option only selects the equivalent IP or L2
implementation used for OHE/Hamming searches. FAISS squared-L2 outputs are
converted back to ordinary Euclidean distances before they are stored on a
direct kNN graph.

For embedding diffusion, the returned neighbours support the sparse RBF
affinity after exact reranking and kth-distance tie handling. Thus `k` changes
the scientific kernel support. `tiebuffer` can add tied candidates but does not
add farther candidates. HNSW and IVF can omit exact neighbours, so their
candidate support—and therefore their kernel—is explicitly approximate. Flat
FAISS and BallTree provide exact candidate searches. Diffusion itself is exact
conditional on that support.

PLM kNN graphs therefore do not require aligned or equal-length protein
sequences, provided the embedding matrix has one finite row per sequence. OHE
and sequence-Hamming graphs support non-binary categorical alphabets, including
aligned amino-acid or nucleotide sequences. They still require a common aligned
length and compatible alphabets across rows.

The graph-level `landscapy_knn_search` record stores the backend, effective
metric, distance geometry, embedding domain, and whether the search constructed
the graph itself or served as a sparse prefilter.

```python
landscape = FitnessLandscape.build(
    sequences,
    graph="knn",
    embeddings={"plm": plm_embeddings},
    embedding_domain="plm",
    k=20,
)
```
