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

### Removed

- Directed landscapes and directed-landscape CLI commands.
- Bottleneck analysis.
- Coupling analysis.
- Built-in visualisation and plotting.
