# Changelog

All notable changes to Landscapy are documented here.

## [Unreleased]

### Added

- Deterministic, portable landscape directory bundles and `.lsbundle` export.
- Weighted random-walk support.
- Corrected evolutionary-diffusion edge scoring.
- Python 3.11 and 3.12 package metadata and build verification.
- Release-blocking lint, public-import, CLI-smoke, and branch-coverage gates.
- Cross-platform clean-install smoke tests for the comprehensive non-ML extra,
  including an end-to-end kNN CLI workflow.

### Changed

- Consolidated package metadata in `pyproject.toml`.
- Reduced the publication API to the methods required by the application note.
- Improved memory use when loading and exporting large landscapes.
- Compatibility imports for moved APIs now preserve their original cause and
  report the required optional package without masking transitive import errors.
- Renamed phylogenetic `construct_dag()` to the undirected-only
  `construct_topology()` and removed directed graph state from portable bundle
  manifests.
- Standardized the documented 0.9 public API on NumPy-style parameter and
  return contracts, with a scoped `numpydoc` CI gate.
- Made sequence storage immutable, rejected invalid binary and soft-posterior
  inputs before coercion, and preserved sequence subclasses and identifiers
  across mutation and factory construction.
- Enforced finite non-negative categorical probabilities and counts, unique
  categories, defensive storage, and strict one-hot decoding.
- Enforced canonical graph-node row alignment across sequences, layers,
  annotations, and every embedding domain, including duplicate-safe attachment.
- Made publication-facing graph analyses independent of node-label type and
  preserved graph labels alongside explicit sequence-row indices in results.
- Standardized edge distance, normalized-distance, affinity, transition, and
  conductance semantics across constructors, analyses, and portable bundles.
- Added shared graph-constructor validation for aligned sequences, embedding
  matrices, nearest-neighbour options, diffusion parameters, and small-sample
  TDA behavior; FAISS IVF now honors its requested metric and uses a trainable
  centroid count.
- Replaced order-dependent diffusion edge weights and stationary marginals
  with a reversible lazy transition and symmetric stationary-measure kernel;
  stationary limits are now evaluated within communicating components.
- Corrected random-walk Laplacian eigenmodes by solving the similar symmetric
  normalized operator and mapping real right modes back to the transition
  operator, including explicit stationary modes for isolated nodes.
- Made diffusion-scale layer selection non-mutating, node-aligned, and strict
  about scalar shape and finite numeric values.
- Corrected Dirichlet edge-bin and local conservation, made weighting
  explicitly opt-in by edge key, and documented global versus per-node energy.
- Made effective resistance component-wise with infinite cross-component
  costs, explicit jitter reporting, and defined empty/singleton and category
  aggregation behavior.
- Consolidated hard-token and relaxed-distribution ESM embeddings behind one
  implementation with explicit alphabet, masking, pooling, dtype, and ordering
  contracts; fixed the recursive ``extract_features`` alias.
- Made distribution and pairwise tests strict about finite inputs and valid
  sample sizes; added explicit Shapiro-Wilk boundary policies, seeded
  permutation generators with replayable state, finite-sample Monte Carlo
  p-values and uncertainty, and Holm/Bonferroni/BH multiplicity correction.
- Made `landscapy[all]` the exact union of user-facing extras except `ml`, added
  the directly imported `tqdm` embedding dependency, constrained Python 3.11 to
  a compatible `piqtree`, and gated FAISS wheels by supported OS/architecture.
- Added actionable CPU and BallTree fallbacks to unavailable FAISS diagnostics.
- Consolidated generalized NK construction in `create_gnk_landscape`, repaired
  uniform and per-site multiallelic sequence construction and metadata, and
  made `create_nk_multi_landscape` a deprecated compatibility alias.
- Defined full-cube Walsh, sampled-binary regression, and general categorical
  epistasis domains; corrected Walsh position labels and normalization,
  enforced regression identifiability, and made higher-order empirical
  marginal decompositions subtract every lower-order subset.
- Made kNN construction domain-aware: PLM and composition embeddings now
  define Euclidean/L2 neighbourhoods for direct kNN graphs and the sparse
  prefilters used by diffusion constructors, while sequence/OHE searches retain
  Hamming geometry.

### Removed

- Directed landscapes and directed-landscape CLI commands.
- Bottleneck analysis.
- Coupling analysis.
- Persistent-homology analysis; TDA graph construction remains supported.
- Built-in visualisation and plotting.
