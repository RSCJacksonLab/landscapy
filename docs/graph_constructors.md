# Graph-constructor input contract

Landscapy validates graph inputs before calling scikit-learn, FAISS, GUDHI, or
Ray. Invalid public inputs therefore raise a `TypeError` or `ValueError` that
names the Landscapy parameter rather than exposing backend-specific failures.

## Sequences and embeddings

Sequence collections must be sized collections of `BaseNumpySequence`
instances with one common, non-zero aligned length. Empty collections are
supported and produce an empty graph. A singleton produces one node and no
self edge.

User-supplied embeddings must be a finite numeric matrix with shape
`(n_sequences, n_features)` and at least one feature column.

## Nearest-neighbour options

`k` is a positive integer denoting non-self neighbours. When `k >= n`, it is
capped at `n - 1`; query sizes, including the non-negative `tiebuffer`, are
also capped at `n`. The accepted tie policies are `all`, `min_index`, and
`random`.

The accepted backends are `auto`, `balltree`, and `faiss`. FAISS accepts the
`flat`, `hnsw`, and `ivf` indices with either `ip` or `l2`. Each index is built
with the requested metric. IVF selects a positive centroid count no larger
than the number of samples. GPU execution requires the FAISS backend, a
GPU-enabled FAISS build, and the `flat` index.

`include_self` controls candidate-query capacity only. Landscapy's release
graphs are undirected and never contain self edges.

## TDA

TDA automatically clips `n_components` to the available samples, features,
and centered geometric rank. The requested and effective values are recorded
as `tda_requested_components` and `tda_effective_components` in graph
metadata. Duplicate embedding points are rejected because an alpha complex
cannot preserve their one-row-per-sequence node identity. A singleton is
returned directly without invoking PCA or GUDHI.

## Diffusion

Finite diffusion powers must be integers greater than or equal to one.
`None`, zero, and positive infinity select the documented stationary regime;
negative values, non-integral values, NaN, and negative infinity are invalid.
Connectivity thresholds must be finite probabilities in `[0, 1]`.
Evolutionary-diffusion temperature `tau` must be finite and strictly positive.
