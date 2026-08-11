# Changelog

All notable changes to Landscapy are documented here.

## [Unreleased]

### Added

- Deterministic, portable landscape directory bundles and `.lsbundle` export.
- Weighted random-walk support.
- Corrected evolutionary-diffusion edge scoring.
- Python 3.11 and 3.12 package metadata and build verification.
- Release-blocking lint, public-import, CLI-smoke, and branch-coverage gates.

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

### Removed

- Directed landscapes and directed-landscape CLI commands.
- Bottleneck analysis.
- Coupling analysis.
- Built-in visualisation and plotting.
