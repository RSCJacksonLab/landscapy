"
Fitness Landscape Analysis Package
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
* Visualization tools
* Compatible with torch, numpy, and vectorized operations

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
* `eigenmode.py`: Eigenmode decomposition

### Analysis
* `epistasis.py`: Epistasis analysis methods
* `ruggedness.py`: Ruggedness analysis methods
* `path.py`: Evolutionary path analysis
* `statistics.py`: Statistical analysis methods
* `sparsity.py`: Sparsity analysis methods (Listgarten)

### Models
* `minimum_epistasis.py`: Minimum Epistasis Interpolation (McCandlish)

### Utils
* `visualization.py`: Visualization tools including Diffusion Axes (McCandlish)

New Features
-----------

### Minimum Epistasis Interpolation
Based on David McCandlish's work, this method infers the least epistatic possible 
sequence-function relationship compatible with available data. It minimizes the expected 
squared epistatic coefficient for random pairs of mutations across genetic backgrounds.

```python
from fitness_landscape.models.minimum_epistasis import MinimumEpistasisInterpolation

# Initialize and fit the model
mei = MinimumEpistasisInterpolation(sequence_length=4, alphabet_size=2)
mei.fit(landscape)

# Predict fitness for new sequences
predicted_fitness = mei.predict(["1111"])

# Get epistatic coefficients
coeffs = mei.get_epistatic_coefficients()
```

### Diffusion Axes Visualization
Based on David McCandlish's work, this method creates low-dimensional representations 
of fitness landscapes using diffusion maps, which plot genotypes in a manner that 
captures important features of the landscape.

```python
from fitness_landscape.utils.visualization import DiffusionAxesVisualization

# Initialize and fit the model
diffusion_viz = DiffusionAxesVisualization(sequence_length=4, n_components=3)
diffusion_coords = diffusion_viz.fit_transform(landscape)

# Visualize the landscape
import matplotlib.pyplot as plt
ax = diffusion_viz.plot_landscape(dimensions=[0, 1], fitness_values=landscape.genotype_to_fitness)
plt.show()
```

### Sparsity Analysis
Based on Jennifer Listgarten's work, these methods leverage the observation that 
empirical fitness functions display substantial sparsity when represented in terms 
of epistatic interactions, using Compressed Sensing theory to provide scaling laws 
for sample requirements.

```python
from fitness_landscape.analysis.sparsity import SparsityAnalysis

# Initialize the analyzer
sparsity_analyzer = SparsityAnalysis(sequence_length=4, alphabet_size=2)

# Estimate sparsity
sparsity = sparsity_analyzer.estimate_sparsity(landscape)

# Calculate sample complexity
samples_needed = sparsity_analyzer.calculate_sample_complexity(sparsity)

# Generate and analyze GNK models
gnk_model = sparsity_analyzer.generate_gnk_model(interaction_structure='adjacent')
comparison = sparsity_analyzer.compare_empirical_vs_gnk(landscape, gnk_model)
```

Examples
--------
See the `examples` directory for detailed examples of each feature:

* `basic_analysis.py`: Basic fitness landscape analysis
* `walsh_transform_analysis.py`: Walsh-Hadamard transform for epistasis analysis
* `graph_spectral_analysis.py`: Graph Fourier transforms and eigenmode decomposition
* `evolutionary_paths.py`: Path analysis and evolutionary trajectory simulation
* `minimum_epistasis_example.py`: Using Minimum Epistasis Interpolation
* `diffusion_axes_example.py`: Using Diffusion Axes Visualization
* `sparsity_analysis_example.py`: Using Sparsity Analysis Methods

License
-------
MIT License
"""
