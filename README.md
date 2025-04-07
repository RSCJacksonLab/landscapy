"
Landscapy: Fitness Landscape Analysis
=================================

A Python package for analyzing fitness landscapes modeled as network graphs.

This package provides tools for analyzing fitness landscapes using various mathematical
transformations and analysis methods. It supports modeling landscapes as Hamming graphs,
KNN graphs, or custom NetworkX graphs, and includes efficient implementations of
Walsh-Hadamard transformations, graph Fourier transforms, and eigenmode decomposition.

Features
--------
* Core data structures for sequences and fitness landscapes
* Efficient mathematical transformations
* Comprehensive analysis methods
* Compatible with torch, numpy compute backends.

Installation
-----------
```bash
pip install landscapy
```

Quick Start
----------
```python
import numpy as np
from fitness_landscape.core.sequence import BinarySequence
from fitness_landscape.core.landscape import FitnessLandscape
from fitness_landscape.analysis.epistasis import calculate_epistasis

# Create a simple fitness landscape
sequences = ["000", "001", "010", "011", "100", "101", "110", "111"]
fitnesses = [0.1, 0.3, 0.2, 0.8, 0.3, 0.6, 0.5, 1.0]
landscape = FitnessLandscape(dict(zip(sequences, fitnesses)))

# Analyze epistasis
epistasis = calculate_epistasis(landscape, method='walsh')
print(epistasis)
```

Main Components
--------------

### Core
* `sequence.py`: Sequence representations (binary and multiallelic)
* `landscape.py`: Fitness landscape class
* `graph.py`: Graph operations (Hamming, KNN, custom NetworkX)

### Transforms
* `walsh_hadamard.py`: Walsh-Hadamard transforms (standard and multiallelic)
* `graph_fourier.py`: Graph Fourier transforms
* `diffusion_fourier.py`: Markov / Diffusion Fourier transforms

### Analysis
* `epistasis.py`: Epistasis analysis methods
* `eigenmode.py`: Eigenmode decomposition
* `random_walk.py`: Ruggedness random walk analysis
* `adaptive_walk.py`: Evolutionary path analysis
* `dirichlet_energy.py`: Dirichlet energy ruggedness analysis
* `graph`: Graph analysis methods
* `statistics.py`: Statistical analysis methods

### Models
* `nk.py`: NK fitness landscape model
* `rmf.py`: Rough mount Fuji fitness landscape model
* `elementary_landscape.py`: Elementary fitness landscape model


To do
-----------
* Sparsity methods (Brookes et al., 2020)
* Minimum epistasis interpolation (Zhou et al., 2020)
* Global epistasis (Otwinowski et al., 2018)

License
-------
MIT License
"""
