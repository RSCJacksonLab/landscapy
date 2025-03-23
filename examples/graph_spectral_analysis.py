"""
Example of graph Fourier transforms and eigenmode decomposition.

This example demonstrates how to use graph Fourier transforms and eigenmode decomposition
to analyze fitness landscapes as network graphs.
"""

import numpy as np
import matplotlib.pyplot as plt
from fitness_landscape import (
    BinarySequence,
    FitnessLandscape,
    create_hamming_graph,
    generate_sequences
)
from fitness_landscape.transforms import (
    graph_fourier_transform,
    inverse_graph_fourier_transform,
    eigenmode_decomposition,
    eigenmode_analysis
)

# Generate all binary sequences of length 4
sequences = generate_sequences(4, [0, 1], strategy='complete')
print(f"Generated {len(sequences)} binary sequences")

# Create a fitness landscape with a smooth pattern
# We'll use a fitness function that depends on the number of 1s (Hamming weight)
def fitness_function(seq):
    # Count number of 1s
    hamming_weight = sum(seq.to_array())
    
    # Create a smooth fitness function
    fitness = np.sin(np.pi * hamming_weight / 4) + 1
    
    return fitness

# Calculate fitness for each sequence
fitness_values = [fitness_function(seq) for seq in sequences]

# Create fitness landscape
landscape = FitnessLandscape(sequences, fitness_values)

# Create Hamming graph
landscape.graph = create_hamming_graph(sequences, fitness_values)

print("\nPart 1: Graph Fourier Transform")
print("------------------------------")

# Perform graph Fourier transform
eigenvectors, eigenvalues, coefficients = graph_fourier_transform(landscape)

print(f"Graph has {len(eigenvalues)} eigenvalues")
print(f"First 5 eigenvalues: {eigenvalues[:5]}")
print(f"First 5 Fourier coefficients: {coefficients[:5]}")

# Reconstruct signal using inverse transform
reconstructed = inverse_graph_fourier_transform(eigenvectors, coefficients)

# Calculate reconstruction error
error = np.mean((np.array(fitness_values) - reconstructed)**2)
print(f"Reconstruction mean squared error: {error:.6f}")

# Plot original vs reconstructed fitness
plt.figure(figsize=(10, 6))
plt.scatter(range(len(fitness_values)), fitness_values, label='Original', alpha=0.7)
plt.scatter(range(len(reconstructed)), reconstructed, label='Reconstructed', alpha=0.7)
plt.xlabel('Sequence Index')
plt.ylabel('Fitness')
plt.title('Original vs Reconstructed Fitness')
plt.legend()
plt.grid(True)
plt.savefig('fourier_reconstruction.png')

# Plot Fourier coefficients
plt.figure(figsize=(10, 6))
plt.stem(range(len(coefficients)), np.abs(coefficients))
plt.xlabel('Coefficient Index')
plt.ylabel('Magnitude')
plt.title('Graph Fourier Coefficients')
plt.grid(True)
plt.savefig('fourier_coefficients.png')

print("\nPart 2: Eigenmode Decomposition")
print("------------------------------")

# Perform eigenmode decomposition
eigenvalues, eigenvectors = eigenmode_decomposition(landscape)

print(f"First 5 eigenvalues from eigenmode decomposition: {eigenvalues[:5]}")

# Analyze eigenmodes
eigenmode_results = eigenmode_analysis(landscape)

print("\nEigenmode analysis results:")
print(f"Spectral gap: {eigenmode_results['spectral_gap']:.4f}")
print("\nParticipation ratios (first 5 modes):")
for i in range(5):
    print(f"  Mode {i}: {eigenmode_results['participation_ratios'][i]:.4f}")

print("\nLocalization (first 5 modes):")
for i in range(5):
    print(f"  Mode {i}: {eigenmode_results['localization'][i]:.4f}")

# Plot participation ratios
plt.figure(figsize=(10, 6))
plt.plot(eigenmode_results['participation_ratios'], 'o-')
plt.xlabel('Mode Index')
plt.ylabel('Participation Ratio')
plt.title('Eigenmode Participation Ratios')
plt.grid(True)
plt.savefig('participation_ratios.png')

# Create a new fitness landscape with a rugged pattern
def rugged_fitness_function(seq):
    # Convert sequence to array
    arr = seq.to_array()
    
    # Base fitness from Hamming weight
    hamming_weight = sum(arr)
    base_fitness = np.sin(np.pi * hamming_weight / 4) + 1
    
    # Add ruggedness with interactions
    ruggedness = 0.3 * arr[0] * arr[1] - 0.2 * arr[1] * arr[2] + 0.4 * arr[0] * arr[3]
    
    # Add some random noise
    noise = np.random.normal(0, 0.1)
    
    return base_fitness + ruggedness + noise

# Calculate rugged fitness for each sequence
rugged_fitness = [rugged_fitness_function(seq) for seq in sequences]

# Create rugged fitness landscape
rugged_landscape = FitnessLandscape(sequences, rugged_fitness)
rugged_landscape.graph = create_hamming_graph(sequences, rugged_fitness)

# Compare eigenmode analysis between smooth and rugged landscapes
smooth_analysis = eigenmode_analysis(landscape)
rugged_analysis = eigenmode_analysis(rugged_landscape)

# Plot comparison of participation ratios
plt.figure(figsize=(10, 6))
plt.plot(smooth_analysis['participation_ratios'][:10], 'o-', label='Smooth Landscape')
plt.plot(rugged_analysis['participation_ratios'][:10], 's-', label='Rugged Landscape')
plt.xlabel('Mode Index')
plt.ylabel('Participation Ratio')
plt.title('Comparison of Eigenmode Participation Ratios')
plt.legend()
plt.grid(True)
plt.savefig('landscape_comparison.png')

print("\nExample completed. Check the generated images for visualizations.")
