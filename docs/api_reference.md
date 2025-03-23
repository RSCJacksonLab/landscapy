"""
API Reference for Fitness Landscape Package
==========================================

This document provides a comprehensive reference for all modules, classes, and functions
in the fitness landscape package.

Core Module
----------

### Sequence Module

```python
from fitness_landscape.core.sequence import Sequence, BinarySequence, MultialleleSequence
```

**Sequence**: Base class for all sequence types
- `__init__(self, sequence)`: Initialize with sequence string
- `distance(self, other)`: Calculate distance to another sequence
- `__str__(self)`: String representation
- `__len__(self)`: Sequence length

**BinarySequence**: Optimized for binary sequences
- `__init__(self, sequence)`: Initialize with binary sequence string
- `hamming_distance(self, other)`: Calculate Hamming distance
- `neighbors(self)`: Get all sequences that differ by one position

**MultialleleSequence**: For sequences with multiple possible values at each position
- `__init__(self, sequence, alphabet_size)`: Initialize with sequence string and alphabet size
- `hamming_distance(self, other)`: Calculate Hamming distance
- `neighbors(self)`: Get all sequences that differ by one position

### Landscape Module

```python
from fitness_landscape.core.landscape import FitnessLandscape
```

**FitnessLandscape**: Maps sequences to fitness values
- `__init__(self, genotype_to_fitness)`: Initialize with dictionary mapping sequences to fitness
- `get_fitness(self, sequence)`: Get fitness for a sequence
- `get_all_fitnesses(self)`: Get all fitness values
- `get_all_sequences(self)`: Get all sequences
- `to_graph(self, graph_type='hamming')`: Convert to graph representation
- `visualize(self, method='network')`: Visualize the landscape

### Graph Module

```python
from fitness_landscape.core.graph import create_hamming_graph, create_knn_graph, graph_properties
```

**create_hamming_graph(sequences)**: Create graph connecting sequences that differ by one position

**create_knn_graph(sequences, k=5)**: Create graph connecting each sequence to its k nearest neighbors

**graph_properties(graph)**: Calculate relevant graph properties for fitness landscapes

Transforms Module
---------------

### Walsh-Hadamard Transform

```python
from fitness_landscape.transforms.walsh_hadamard import walsh_transform, inverse_walsh_transform, multiallelic_walsh_transform
```

**walsh_transform(fitness_values)**: Compute Walsh-Hadamard transform of fitness values

**inverse_walsh_transform(walsh_coefficients)**: Compute inverse Walsh-Hadamard transform

**multiallelic_walsh_transform(fitness_values, alphabet_size)**: Extended Walsh-Hadamard transform for multiallelic sequences

### Graph Fourier Transform

```python
from fitness_landscape.transforms.graph_fourier import graph_fourier_transform, inverse_graph_fourier_transform
```

**graph_fourier_transform(graph, signal)**: Compute Graph Fourier transform

**inverse_graph_fourier_transform(graph, fourier_coefficients)**: Compute inverse Graph Fourier transform

### Eigenmode Decomposition

```python
from fitness_landscape.transforms.eigenmode import eigenmode_decomposition, reconstruct_from_eigenmodes
```

**eigenmode_decomposition(graph, n_components=None)**: Compute eigenmode decomposition of graph

**reconstruct_from_eigenmodes(eigenmodes, coefficients)**: Reconstruct graph from eigenmodes

Analysis Module
-------------

### Epistasis Analysis

```python
from fitness_landscape.analysis.epistasis import calculate_epistasis, decompose_fitness
```

**calculate_epistasis(landscape, method='walsh')**: Calculate epistasis using specified method
- Methods: 'walsh', 'regression', 'ensemble', 'reference_free'

**decompose_fitness(landscape, order=None)**: Decompose fitness into epistatic components

### Ruggedness Analysis

```python
from fitness_landscape.analysis.ruggedness import calculate_ruggedness, adaptive_walk
```

**calculate_ruggedness(landscape, method='autocorrelation')**: Calculate ruggedness using specified method
- Methods: 'autocorrelation', 'fdc', 'local_optima', 'roughness'

**adaptive_walk(landscape, start_sequence, steps=100)**: Simulate adaptive walk on landscape

### Path Analysis

```python
from fitness_landscape.analysis.path import find_paths, path_accessibility
```

**find_paths(landscape, start, end, method='shortest')**: Find paths between sequences
- Methods: 'shortest', 'accessible', 'all'

**path_accessibility(landscape, paths)**: Calculate path accessibility metrics

### Statistical Analysis

```python
from fitness_landscape.analysis.statistics import fitness_distribution, correlation_analysis
```

**fitness_distribution(landscape)**: Analyze fitness distribution

**correlation_analysis(landscape, features)**: Analyze correlation between features and fitness

### Sparsity Analysis

```python
from fitness_landscape.analysis.sparsity import SparsityAnalysis
```

**SparsityAnalysis**: Analyze sparsity of fitness landscapes
- `__init__(self, sequence_length, alphabet_size=2)`: Initialize analyzer
- `estimate_sparsity(self, landscape, threshold=0.01)`: Estimate landscape sparsity
- `calculate_sample_complexity(self, sparsity=None)`: Calculate required samples
- `generate_gnk_model(self, interaction_structure='adjacent')`: Generate GNK model
- `compare_empirical_vs_gnk(self, landscape, gnk_model)`: Compare landscapes

Models Module
-----------

### Minimum Epistasis Interpolation

```python
from fitness_landscape.models.minimum_epistasis import MinimumEpistasisInterpolation
```

**MinimumEpistasisInterpolation**: Infer least epistatic sequence-function relationship
- `__init__(self, sequence_length, alphabet_size=2)`: Initialize model
- `fit(self, landscape)`: Fit model to observed landscape
- `predict(self, sequences)`: Predict fitness for new sequences
- `get_epistatic_coefficients(self)`: Get epistatic coefficients
- `get_epistasis_statistics(self)`: Get statistics about epistasis

Utils Module
----------

### Visualization

```python
from fitness_landscape.utils.visualization import DiffusionAxesVisualization
```

**DiffusionAxesVisualization**: Create low-dimensional representations of landscapes
- `__init__(self, sequence_length, alphabet_size=2, n_components=3)`: Initialize model
- `fit_transform(self, landscape, alpha=1.0, t=1.0)`: Compute diffusion coordinates
- `transform(self, sequences)`: Project new sequences onto diffusion axes
- `plot_landscape(self, ax=None, dimensions=[0, 1])`: Visualize landscape
- `plot_paths(self, paths, ax=None, dimensions=[0, 1])`: Visualize paths
- `get_eigenvalues(self)`: Get eigenvalues of diffusion operator
- `get_eigenvectors(self)`: Get eigenvectors of diffusion operator
"""
