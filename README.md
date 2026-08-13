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
- coupling analysis;
- persistent-homology analysis; and
- built-in plotting or interactive visualisation.

Those areas are maintained independently on feature branches so that they do
not enlarge or destabilise the publication API. Landscapy exports data for use
with external plotting and visualisation tools.

## Requirements and installation

Landscapy supports Python 3.11 and 3.12.

For a standard installation, install every feature supported by the current OS
except the separately maintained ML export stack:

```bash
python -m pip install "landscapy[all]"
```

This is the recommended path for CLI users. It installs the CLI, portable
Parquet export, kNN and TDA constructors, analyses, CPU parallelism, alignment,
phylogeny, CPU FAISS where a compatible wheel exists, and protein language-model
embeddings. It deliberately excludes the `ml` extra and its PyTorch Geometric
export dependency.

Install only the lightweight core when optional functionality is not required:

```bash
python -m pip install landscapy
```

Install the optional Parquet backend for native Parquet payloads in portable
bundles:

```bash
python -m pip install "landscapy[parquet]"
```

Individual extras remain available for constrained environments:

- `knn` for scikit-learn nearest-neighbour and diffusion graphs;
- `tda` for topological graph construction;
- `faiss` for accelerated nearest-neighbour search;
- `alignment` for soft sequence alignment;
- `phylogeny` for tree inference and ancestral reconstruction;
- `parallel` for Ray-backed parallel execution;
- `embeddings` for protein language-model embeddings;
- `ml` for embeddings plus PyTorch Geometric export; and
- `cli` for the command-line entry points.

`landscapy[ml]` remains an explicit, separate install while its dependency
contract is revised:

```bash
python -m pip install "landscapy[ml]"
```

Always quote an extras expression such as `"landscapy[all]"`; shells including
zsh otherwise interpret the square brackets as a filename pattern.

FAISS availability depends on upstream binary wheels. `landscapy[all]` installs
`faiss-cpu` on supported Linux x86-64/ARM64, current macOS Intel/Apple Silicon,
and Windows x86-64/ARM64 platforms. On other platforms, the comprehensive
install keeps scikit-learn's portable BallTree backend available. Select it with
`--backend balltree`. GPU FAISS is not supplied by `faiss-cpu`; when a compatible
GPU build is unavailable, omit `--use-gpu` to use CPU FAISS or select BallTree.

For development from a checkout:

```bash
python -m pip install -e ".[dev,all]"
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
print(result["global_dirichlet_energy"])
print(result["total_dirichlet_energy"])  # Per-node normalization.

# Weighting is opt-in and requires an explicit conductance key.
weighted = calculate_ruggedness_dirichlet_energy(
    landscape,
    weight_key="weight",
)
```

For nearest-neighbour graphs, `embedding_domain="plm"` uses Euclidean
distances in the supplied or computed PLM embeddings. The same domain-aware
geometry is used by sparse kNN prefilters in diffusion constructors; see the
[kNN embedding-domain contract](docs/knn_embedding_domains.md).
Embedding diffusion evaluates its RBF affinity only on that sparse candidate
graph and uses resource-guarded exact sparse powers, avoiding dense `n x n`
kernel construction.

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

The exact supported names and import namespaces are listed in the
[0.9 public API contract](docs/public_api.md). CI validates that contract
against the exported objects and their NumPy-style docstrings.

Graph constructors and weighted analyses follow the documented
[edge distance and conductance contract](docs/edge_semantics.md). In
particular, NetworkX `weight` is reserved for conductance and is never a raw
distance. Constructor input and small-sample behavior are specified in the
[graph-constructor contract](docs/graph_constructors.md), and diffusion graphs
follow the [reversible diffusion contract](docs/diffusion_semantics.md).
Transition eigenmodes follow the documented
[random-walk spectral contract](docs/spectral_operators.md). Effective
resistance and disconnected-category aggregation follow the
[component-wise resistance contract](docs/resistance_distance.md).
Publication-facing distribution, pairwise, and permutation inference follows
the documented [statistical inference contract](docs/statistical_inference.md).
Walsh, regression, ensemble, and reference-free epistasis methods follow the
documented [epistasis domain and coefficient contract](docs/epistasis.md).

- `fitness_landscape.core`: sequences, fitness and annotation layers,
  undirected graph construction, and `FitnessLandscape`.
- `fitness_landscape.models`: NK, Rough Mount Fuji, and elementary landscapes.
- `fitness_landscape.analysis`: ruggedness, epistasis, adaptive walks,
  statistics, diffusion scale, and alignment metrics.
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

Given an aligned protein FASTA file, a minimal portable kNN CLI workflow is:

```bash
landscapy knn-landscape \
  --sequences sequences.fasta \
  --output landscape.pkl \
  --k 5 \
  --backend balltree \
  --embedding-domain ohe
```

The BallTree example works across supported operating systems and does not
require a platform-specific FAISS build.

## Cookbook

The [worked-example cookbook](docs/cookbook/README.md) starts with empirical
tables and the layered `FitnessLandscape` data model, then covers graph
construction, topology, and analysis. Every recipe states its assumptions,
expected outputs, and limits of interpretation, and its executable example is
checked in CI.

- [Foundations](docs/cookbook/foundations/README.md)
- [Components, topology, and annotated groups](docs/cookbook/topology/README.md)
- [Graph construction and representation choice](docs/cookbook/graph-construction/README.md)
- [Saving, sharing, CLI use, and external visualization](docs/cookbook/io/README.md)
- [Ruggedness, autocorrelation, and spectral analysis](docs/cookbook/ruggedness/README.md)
- [Adaptive walks, accessibility, basins, optima, and neutral networks](docs/cookbook/accessibility/README.md)
- [Epistasis on complete, sampled, and categorical landscapes](docs/cookbook/epistasis/README.md)

## Citation

Citation metadata is provided in `CITATION.cff`. Release changes are recorded
in `CHANGELOG.md`.

## Authors

- Matthew A. Spence
- Barnabas Gall
- Dana S. Matthews

## License

Landscapy is distributed under the MIT License. See `LICENSE`.
