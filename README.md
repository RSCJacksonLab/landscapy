# Landscapy

Landscapy is a Python package for constructing and analysing graph-based
fitness landscapes. It supports sequence-aware graph construction, layered
fitness and annotation data, landscape models, ruggedness and epistasis
analyses, graph alignment, phylogenetic reconstruction, and portable export.

## Release scope

The `0.9` publication release is intentionally limited to undirected fitness
landscapes. It includes the analysis methods used by the accompanying
application note and deterministic export to portable landscape bundles.

The following experimental areas are not part of this release:

- directed graphs and directed fitness landscapes;
- bottleneck analysis;
- coupling analysis; and
- built-in plotting or interactive visualisation.

Those areas are maintained independently on feature branches so that they do
not enlarge or destabilise the publication API. Landscapy exports data for use
with external plotting and visualisation tools.

## Requirements and installation

Landscapy supports Python 3.11 and 3.12.

```bash
python -m pip install landscapy
```

Install the optional Parquet backend for native Parquet payloads in portable
bundles:

```bash
python -m pip install "landscapy[parquet]"
```

For development from a checkout:

```bash
python -m pip install -e ".[dev,parquet]"
python -m pytest
```

## Quick start

```python
from fitness_landscape.analysis.dirichlet_energy import (
    calculate_ruggedness_dirichlet_energy,
)
from fitness_landscape.models import create_nk_binary_landscape

landscape = create_nk_binary_landscape(N=4, K=1, seed=42)
result = calculate_ruggedness_dirichlet_energy(landscape)
print(result["total_dirichlet_energy"])
```

## Portable landscape export

The portable directory bundle is the canonical, inspectable export format.
It does not require pickle and is suitable for checksummed artifact storage.

```python
landscape.save_bundle_dir(
    "artifacts/example_landscape",
    metadata={
        "dataset_name": "example-dataset",
        "assay_type": "DMS",
        "version": "v1",
        "provenance": {"pipeline": "application-note"},
    },
    include_embeddings=True,
)

reloaded = landscape.load_bundle_dir("artifacts/example_landscape")
landscape.export_lsbundle(
    "artifacts/example_landscape.lsbundle",
    backend="portable",
)
```

The directory contains a versioned manifest, metadata, canonical node and edge
tables, sequence arrays, layers, annotations, and optional embedding arrays.
When `pyarrow` is installed, tabular payloads use Parquet; otherwise Landscapy
uses its deterministic JSON-table fallback. A pickle-backed compatibility
export remains available via `backend="pickle"`, but it is not the recommended
long-term storage format.

CSV export is also available through `to_csv_landscape` and
`read_csv_landscape` in `fitness_landscape.core.landscape`.

## Main modules

- `fitness_landscape.core`: sequences, fitness and annotation layers,
  undirected graph construction, and `FitnessLandscape`.
- `fitness_landscape.models`: NK, Rough Mount Fuji, and elementary landscapes.
- `fitness_landscape.analysis`: ruggedness, epistasis, adaptive walks,
  statistics, diffusion scale, persistent homology, and alignment metrics.
- `fitness_landscape.transforms`: Walsh-Hadamard, eigenmode, and graph Fourier
  transforms.
- `fitness_landscape.phylo`: phylogenetic inference and ancestral-state
  reconstruction used to construct undirected landscapes.
- `fitness_landscape.io`: deterministic portable bundles and compatibility
  archives.

## Command-line interface

```bash
landscapy --help
landscapy-evol --help
landscapy-phylo --help
```

## Citation

Citation metadata is provided in `CITATION.cff`. Release changes are recorded
in `CHANGELOG.md`.

## Authors

- Matthew A. Spence
- Barnabas Gall
- Dana S. Matthews

## License

Landscapy is distributed under the MIT License. See `LICENSE`.
