# User Guide

## Introduction

The `fitness_landscape` package provides tools for analyzing fitness landscapes modeled as network graphs. It includes efficient implementations of Walsh-Hadamard transformations, graph Fourier transforms, and eigenmode decomposition, as well as various analysis methods for studying epistasis, ruggedness, and evolutionary paths.

This guide will walk you through the basic concepts and usage of the package.

## Installation

```bash
pip install fitness-landscape
```

## Basic Concepts

### Sequences

Sequences are the fundamental building blocks of fitness landscapes. The package provides several sequence classes:

- `Sequence`: Base class for all sequence types
- `BinarySequence`: Optimized for binary sequences (0s and 1s)
- `MultialleleSequence`: Supports sequences with multiple possible values at each position

Example:

```python
from fitness_landscape import BinarySequence, MultialleleSequence

# Create a binary sequence
binary_seq = BinarySequence([0, 1, 0, 1])

# Create a multiallelic sequence
multi_seq = MultialleleSequence([0, 2, 1, 3])
```

You can also generate sequences using the `generate_sequences` function:

```python
from fitness_landscape import generate_sequences

# Generate all possible binary sequences of length 3
binary_seqs = generate_sequences(3, [0, 1], strategy='complete')

# Generate 10 random DNA sequences of length 5
dna_seqs = generate_sequences(5, ['A', 'C', 'G', 'T'], strategy='random', n_sequences=10)
```

### Fitness Landscapes

A fitness landscape maps sequences to fitness values. The `FitnessLandscape` class represents this mapping and provides methods for analysis.

```python
from fitness_landscape import FitnessLandscape

# Create a fitness landscape
landscape = FitnessLandscape(sequences, fitness_values)

# Get fitness of a sequence
fitness = landscape.get_fitness(sequences[0])
```

### Graph Representations

Fitness landscapes can be represented as graphs, where nodes are sequences and edges connect related sequences. The package provides functions for creating different types of graphs:

```python
from fitness_landscape import create_hamming_graph, create_knn_graph

# Create a Hamming graph (edges connect sequences that differ by one position)
hamming_graph = create_hamming_graph(sequences, fitness_values)

# Create a k-nearest neighbor graph
knn_graph = create_knn_graph(sequences, fitness_values, k=5)

# Assign graph to landscape
landscape.graph = hamming_graph
```

## Mathematical Transformations

### Walsh-Hadamard Transform

The Walsh-Hadamard transform is a powerful tool for analyzing epistasis in fitness landscapes.

```python
from fitness_landscape.transforms import walsh_transform, walsh_coefficients

# Compute Walsh transform
coeffs = walsh_transform(landscape)

# Extract Walsh coefficients by order
epistasis = walsh_coefficients(landscape, order=2)
```

For multiallelic sequences, use the `MultialleleWalshTransform` class:

```python
from fitness_landscape.transforms import MultialleleWalshTransform

# Create transform for sequences with specified alphabet sizes
transform = MultialleleWalshTransform([3, 2, 4])  # 3 values at pos 0, 2 at pos 1, 4 at pos 2

# Compute transform
coeffs = transform.transform(landscape)
```

### Graph Fourier Transform

The graph Fourier transform allows spectral analysis of functions defined on graphs.

```python
from fitness_landscape.transforms import graph_fourier_transform, inverse_graph_fourier_transform

# Compute graph Fourier transform
eigenvectors, eigenvalues, coefficients = graph_fourier_transform(landscape)

# Reconstruct signal
reconstructed = inverse_graph_fourier_transform(eigenvectors, coefficients)
```

### Eigenmode Decomposition

Eigenmode decomposition helps analyze the fundamental patterns in network structures.

```python
from fitness_landscape.transforms import eigenmode_decomposition, eigenmode_analysis

# Compute eigenmode decomposition
eigenvalues, eigenvectors = eigenmode_decomposition(landscape)

# Analyze eigenmodes
analysis = eigenmode_analysis(landscape)
```

## Analysis Methods

### Epistasis Analysis

Epistasis refers to interactions between genetic elements that affect fitness.

```python
from fitness_landscape.analysis import calculate_epistasis, epistasis_decomposition

# Calculate epistasis using Walsh transform
epistasis = calculate_epistasis(landscape, order=2, method='walsh')

# Decompose fitness into epistatic components
decomposition = epistasis_decomposition(landscape)
```

### Ruggedness Analysis

Ruggedness measures how difficult it is to navigate the fitness landscape.

```python
from fitness_landscape.analysis import calculate_ruggedness, adaptive_walk

# Calculate ruggedness using autocorrelation
ruggedness = calculate_ruggedness(landscape, method='autocorrelation')

# Perform adaptive walk
walk = adaptive_walk(landscape, max_steps=100)
```

### Path Analysis

Path analysis examines evolutionary trajectories through the fitness landscape.

```python
from fitness_landscape.analysis import find_accessible_paths, find_evolutionary_trajectories

# Find accessible paths between two sequences
paths = find_accessible_paths(landscape, start_seq, end_seq)

# Simulate evolutionary trajectories
trajectories = find_evolutionary_trajectories(landscape, start_seq, n_trajectories=10)
```

### Statistical Analysis

The package provides various statistical methods for analyzing fitness landscapes.

```python
from fitness_landscape.analysis import analyze_fitness_distribution, correlation_analysis

# Analyze fitness distribution
distribution = analyze_fitness_distribution(landscape)

# Analyze correlations between features and fitness
correlations = correlation_analysis(landscape, features)
```

## Advanced Usage

### Using PyTorch Backend

Most transformation functions support both NumPy and PyTorch backends:

```python
# Use PyTorch backend for Walsh transform
coeffs = walsh_transform(landscape, backend='torch')

# Use PyTorch backend for graph Fourier transform
eigenvectors, eigenvalues, coefficients = graph_fourier_transform(landscape, backend='torch')
```

### Custom Graph Types

You can use custom NetworkX graphs with the package:

```python
import networkx as nx

# Create custom graph
custom_graph = nx.Graph()
# Add nodes and edges...

# Create landscape with custom graph
landscape = FitnessLandscape(sequences, fitness_values, graph=custom_graph)
```

## Further Reading

For a complete reference of all functions and classes, see the [API Reference](api_reference.md).
