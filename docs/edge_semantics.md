# Edge attribute semantics

Landscapy 0.9 treats edge attributes as scientific quantities rather than
interchangeable NetworkX weights. Constructors record their contract under the
graph-level `landscapy_edge_schema` metadata key.

| Semantic | Canonical key | Meaning |
| --- | --- | --- |
| Raw distance | `distance` | Constructor-specific non-negative distance in the units declared by graph metadata. |
| Normalized distance | `normalized_distance` | Dimensionless distance in `[0, 1]`, when the constructor has a defined normalization. |
| Affinity | `affinity` | Non-negative dimensionless similarity; larger values mean stronger association. |
| Conductance | `weight` | Non-negative coupling used by weighted Laplacians, Dirichlet energy, effective resistance, community detection, and spectral operators. |
| Transition probability | constructor-declared | Directional Markov probability. Undirected diffusion affinities are not described as transition probabilities. |

The `weight` key is reserved for conductance. A raw distance is never copied to
`weight`. Hamming adjacency graphs use unit conductance. Hamming kNN graphs use
`exp(-normalized_distance)`. TDA graphs use
`1 / (1 + PCA-space Euclidean distance)`. Diffusion constructors use their
retained undirected affinity as conductance. Phylogenetic `branch_length` is a
distance and does not imply a conductance.

Weighted analyses accept either an explicit edge key or `"auto"`. Automatic
resolution uses `landscapy_edge_schema`; an attribute-free graph is treated as
unweighted, but Landscapy will not guess whether an undeclared legacy `weight`
represents a distance or conductance. Passing `None` explicitly requests an
unweighted analysis.

Recognizable portable bundles from earlier versions are migrated on load:

- `kernel_weight` becomes the `affinity` alias for diffusion graphs;
- `tda_distance` remains a distance and a new conductance is derived;
- `knn_weight` is recorded as a legacy alias for Hamming `distance` and a new
  conductance is derived.

Generic bundles containing only an ambiguous `weight` attribute are preserved
without a declared semantic. Callers must select the intended key explicitly.

The legacy `_compute_hamming_edges=True` constructor option now runs an
edge-local annotation pass. Expected mutation count is stored as
`hamming_distance`, its normalized value as `normalized_distance`, and the
derived similarity as `hamming_affinity`; it never overwrites conductance.
