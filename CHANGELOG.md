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

### Removed

- Directed landscapes and directed-landscape CLI commands.
- Bottleneck analysis.
- Coupling analysis.
- Built-in visualisation and plotting.
