# Landscapy: Fitness Landscape Analysis
Landscapy is a Python package for the construction and analysis of fitness landscapes. This package provides a comprehensive toolkit for researchers in evolutionary biology, bioinformatics, and machine learning to study the relationships between protein sequence and function.

## Key Features
- Flexible Landscape Construction: Build fitness landscapes from sequence data using various graph representations, including Hamming distance, k-Nearest Neighbors (k-NN), and advanced methods based on Topological Data Analysis (TDA) and diffusion maps. Landscapy supports construction and analysis methods for both sparse and dense fitness landscapes.

- Rich Analysis Suite: A comprehensive set of analysis tools to quantify landscape ruggedness, epistasis, and evolutionary dynamics. Methods include:

- Epistasis Analysis: Decompose fitness effects using Walsh-Hadamard transforms, regression models, and reference-free methods.

- Ruggedness Metrics: Quantify landscape ruggedness using Dirichlet energy, local optima counts, and autocorrelation analysis.

- Evolutionary Path Analysis: Simulate and analyze adaptive walks, identify greedy accessible paths, and compute basins of attraction.

- Topological Analysis: Explore the shape of fitness landscapes using persistent homology to uncover higher-order structural features.

- Advanced Modeling: Generate synthetic landscapes using established models like NK models and Rough Mount Fuji (RMF) models.

- Graph Matching and Alignment: Align multiple fitness landscapes into a common latent space using a novel Hierarchical RJMCMC Aligner, enabling comparative landscape analysis and graph alignment in linear time.

- Deep Learning Integration: ntegrate with modern deep learning workflows through: protein Language Model (PLM) embeddings and PyTorch Geometric Support.

## Installation
You can install Landscapy directly from PyPI:

```Bash
pip install landscapy
``` 

## Quick Start

```python

import numpy as np
from fitness_landscape.core.landscape import FitnessLandscape
from fitness_landscape.core.sequence import BinarySequence
from fitness_landscape.analysis.epistasis import calculate_epistasis_walsh
from fitness_landscape.models import create_nk_binary_landscape

# Generate a simple NK landscape
landscape = create_nk_binary_landscape(N=4, K=1, seed=42)

# Analyze epistasis using the Walsh-Hadamard transform
epistasis_results = calculate_epistasis_walsh(landscape, order=2)

# Print the second-order epistatic coefficients
print("Second-order epistasis:")
for term, value in epistasis_results['by_order'][2].items():
    print(f"  {term}: {value:.4f}")

# Output the total Dirichlet energy as a measure of ruggedness
from fitness_landscape.analysis.dirichlet_energy import calculate_ruggedness_dirichlet_energy
energy = calculate_ruggedness_dirichlet_energy(landscape)
print(f"\nTotal Dirichlet Energy: {energy['total_dirichlet_energy']:.4f}")
```

## Portable Landscape Bundles
`FitnessLandscape` now supports a first-class portable on-disk bundle format for deterministic, DVC-friendly I/O.

Recommended workflow:

```python
from fitness_landscape.core.landscape import FitnessLandscape

# Save the canonical portable directory bundle.
landscape.save_bundle_dir(
    "artifacts/example_landscape",
    metadata={
        "dataset_name": "example-dataset",
        "source_name": "synthetic-benchmark",
        "protein_gene": "P53",
        "assay_type": "DMS",
        "organism": "human",
        "version": "v1",
        "tags": ["benchmark", "portable"],
        "provenance": {"pipeline": "pytest"},
        "metadata": {"lab": "unit-test"},
    },
    include_embeddings=True,
)

# Load it back without using pickle.
reloaded = FitnessLandscape.load_bundle_dir("artifacts/example_landscape")

# Export the portable bundle as a deterministic .lsbundle archive.
landscape.export_lsbundle("artifacts/example_landscape.lsbundle", backend="portable")
```

Canonical bundle layout:

```text
bundle_dir/
  manifest.json
  metadata.json
  nodes.json
  sequences.npy
  graph_edges.parquet
  layers/
    <layer_name>.parquet
  annotations/
    <annotation_name>.parquet
  embeddings.npy
  embedding_domains/
    <domain>.npy
  legacy/
    landscape.pkl
```

Notes:
- The directory bundle is the canonical format and the recommended artifact to track with `dvc add`.
- `manifest.json` is versioned, deterministic, and records file checksums plus the information needed to rebuild the landscape without pickle.
- `metadata.json` stores scientific/user metadata separately from structural payloads.
- `graph_edges.parquet` stores edges in canonical integer node order, while `nodes.json` preserves original node labels and sequence identifiers.
- The manifest records the physical tabular storage backend. When a parquet engine is installed, these payloads are written as native parquet; otherwise landscapy falls back to a deterministic JSON table encoding under the same paths so save/load still works.
- `legacy/landscape.pkl` is optional and only written when `include_legacy_pickle=True`.

Compatibility `.lsbundle` export for current `landscape-store` v1 ingestion is also available, but it remains pickle-backed and should be treated as a compatibility mode rather than the primary storage format:

```python
landscape.export_lsbundle(
    "artifacts/example_landscape_v1.lsbundle",
    backend="pickle",
    metadata={
        "landscape_id": "example-landscape-v1",
        "protein_gene": "P53",
        "assay_type": "DMS",
        "version": "v1",
    },
)
```

Use the portable directory bundle for long-term storage, DVC tracking, and inspection. Use `backend="pickle"` only when you need to interoperate with the current `landscape-store` v1 ingestion path.
## Main Components
### Core (fitness_landscape.core)
- landscape.py: The central FitnessLandscape class that integrates sequences, fitness data, and a graph representation.

- sequence.py: Flexible sequence objects, including BinarySequence, MultialleleSequence, and SoftSequence for probabilistic representations.

- fitness.py: A layered system for handling different types of fitness data, including NumericFitness, CategoricalFitness, and ProbabilisticCategoricalFitness.

- graph.py: Functions for constructing landscape graphs (create_hamming_graph, create_knn_graph, create_tda_graph).

### Models (fitness_landscape.models)
- nk.py: Generate Kauffman's NK model landscapes.

- rmf.py: Create Rough Mount Fuji (RMF) landscapes.

- lementary_landscape.py: Construct landscapes based on graph Laplacian eigenfunctions.

### Analysis (fitness_landscape.analysis)
- epistasis.py: Functions for calculating epistasis (calculate_epistasis_walsh, calculate_epistasis_regression).

- adaptive_walk.py: Tools for analyzing evolutionary trajectories (find_greedy_accessible_paths, adaptive_walk_stochastic).

- dirichlet_energy.py: Quantify landscape ruggedness using Dirichlet energy.

- persistent_homology.py: Compute Betti curves and other topological features.

- statistics.py: A suite of statistical tests for fitness distributions and correlations.

- Transforms (fitness_landscape.transforms)
walsh_hadamard.py: Perform Walsh-Hadamard transforms for epistasis analysis.

- graph_fourier.py: Analyze fitness signals in the frequency domain using Graph Fourier transforms.

#### Performance tip: cache eigenpairs for large graphs
For large landscapes, computing eigenpairs can dominate runtime. You can cache them once
and pass the precomputed arrays into GFT-based analyses via the private `_eigenvalues`
and `_eigenvectors` parameters.

```python
from fitness_landscape.transforms.eigenmode import eigenmode_decomposition
from fitness_landscape.transforms.graph_fourier import graph_fourier_transform
from fitness_landscape.analysis.diffusion_scale import compute_ruggedness_diffusion_scale
from fitness_landscape.analysis.random_walk import calculate_ruggedness_autocorrelation_analytical

# Example: normalized Laplacian eigenpairs (used by diffusion scale).
evals_norm, evecs_norm = eigenmode_decomposition(landscape.graph, matrix="norm_laplacian", k=None)

# Cache on the landscape if you want (private convention).
landscape._cached_eigs = {"norm_laplacian": (evals_norm, evecs_norm)}

# Reuse in diffusion scale:
fit = compute_ruggedness_diffusion_scale(
    landscape,
    method="grid",
    _eigenvalues=evals_norm,
    _eigenvectors=evecs_norm,
)

# Example: standard Laplacian eigenpairs (used by analytical autocorrelation).
evals_lap, evecs_lap = eigenmode_decomposition(landscape.graph, matrix="laplacian", k=None)
autocorr = calculate_ruggedness_autocorrelation_analytical(
    landscape,
    _eigenvalues=evals_lap,
    _eigenvectors=evecs_lap,
)

# GFT with cached eigenpairs:
U, w, coeffs = graph_fourier_transform(
    landscape,
    _eigenvalues=evals_lap,
    _eigenvectors=evecs_lap,
)
```

Note: the cached eigenpairs must match the matrix type expected by the analysis
(e.g., normalized Laplacian vs Laplacian).

## Authors
Matthew A Spence

Barnabas Gall

Dana S Matthews


## License
This project is licensed under the MIT License - see the LICENSE file for details.
