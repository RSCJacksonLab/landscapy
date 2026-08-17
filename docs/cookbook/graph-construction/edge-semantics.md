# Edge attribute semantics

<!-- cookbook: reference -->

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
`exp(-normalized_distance)`, while embedding-space kNN graphs use
`exp(-Euclidean distance)`. TDA graphs use
`1 / (1 + PCA-space Euclidean distance)`. Diffusion constructors use their
retained undirected affinity as conductance. Phylogenetic `branch_length` is a
distance and does not imply a conductance.

Weighted analyses accept either an explicit edge key or `"auto"`. Automatic
resolution uses `landscapy_edge_schema`; an attribute-free graph is treated as
unweighted, but Landscapy will default to `weight_key=None`, so they remain unweighted unless weighting is requested
explicitly.

#TODO: set all analysis weight_key to none.

For an undirected graph, global Dirichlet energy is the unnormalized quadratic
form `f.T @ L @ f = sum_{u,v} w_uv (f_u - f_v)^2`, where the sum is over
undirected edges. Per-edge and edge-bin energies use that same once-per-edge
convention. The historical
`total_dirichlet_energy` result is normalized by node count; the accompanying
`global_dirichlet_energy` result is not. Local contributions assign half of
each edge's energy to each endpoint, so their sum equals
`global_dirichlet_energy`.

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
derived similarity as `hamming_affinity`.
