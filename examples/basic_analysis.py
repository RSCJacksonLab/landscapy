"""
Example of creating and analyzing a fitness landscape.

This example demonstrates how to create a fitness landscape from sequences and fitness values,
and perform basic analysis using the fitness_landscape package.
"""

import numpy as np
import matplotlib.pyplot as plt
from fitness_landscape import (
    BinarySequence, 
    FitnessLandscape, 
    create_hamming_graph
)
from fitness_landscape.transforms import walsh_transform, walsh_coefficients
from fitness_landscape.analysis import calculate_epistasis, calculate_ruggedness

# Create sequences and fitness values
# We'll create all possible binary sequences of length 3
sequences = [
    BinarySequence([0, 0, 0]), 
    BinarySequence([0, 0, 1]),
    BinarySequence([0, 1, 0]),
    BinarySequence([0, 1, 1]),
    BinarySequence([1, 0, 0]),
    BinarySequence([1, 0, 1]),
    BinarySequence([1, 1, 0]),
    BinarySequence([1, 1, 1])
]

# Define fitness values with epistatic interactions
# This landscape has positive epistasis between positions 0 and 1
fitness_values = [0.1, 0.2, 0.3, 0.7, 0.4, 0.5, 0.6, 1.0]

# Create fitness landscape
landscape = FitnessLandscape(sequences, fitness_values)

# Create Hamming graph
landscape.graph = create_hamming_graph(sequences, fitness_values)

# Calculate epistasis using Walsh transform
epistasis = calculate_epistasis(landscape, order=3, method='walsh')

print("Epistasis analysis:")
print("-------------------")
for order, coeffs in epistasis['by_order'].items():
    print(f"Order {order} coefficients:")
    for term, value in coeffs.items():
        print(f"  {term}: {value:.4f}")

# Calculate ruggedness using autocorrelation
ruggedness = calculate_ruggedness(landscape, method='autocorrelation')

print("\nRuggedness analysis:")
print("-------------------")
print(f"Correlation length: {ruggedness['correlation_length']}")
print(f"Autocorrelation: {ruggedness['autocorrelation']}")

# Plot autocorrelation
plt.figure(figsize=(8, 5))
plt.plot(ruggedness['autocorrelation'], 'o-')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Fitness Landscape Autocorrelation')
plt.grid(True)
plt.savefig('autocorrelation.png')

# Calculate ruggedness using local optima
local_optima = calculate_ruggedness(landscape, method='local_optima')

print("\nLocal optima analysis:")
print("---------------------")
print(f"Number of local optima: {local_optima['local_optima_count']}")
print(f"Density of local optima: {local_optima['local_optima_density']:.4f}")
print(f"Local optima indices: {local_optima['local_optima_indices']}")

# Visualize the fitness landscape as a network
try:
    import networkx as nx
    
    # Get the graph
    G = landscape.graph
    
    # Set node colors based on fitness
    node_colors = [landscape.get_fitness(sequences[i]) for i in G.nodes()]
    
    # Set node labels
    node_labels = {i: str(sequences[i]) for i in G.nodes()}
    
    # Create layout
    pos = nx.spring_layout(G, seed=42)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Draw graph
    nx.draw(G, pos, node_color=node_colors, cmap=plt.cm.viridis, 
            with_labels=True, labels=node_labels, node_size=500, 
            font_size=10, font_color='white', edge_color='gray')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(min(node_colors), max(node_colors)))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Fitness')
    
    plt.title('Fitness Landscape as Hamming Graph')
    plt.savefig('fitness_landscape_graph.png')
    
except ImportError:
    print("NetworkX or matplotlib not available for visualization")

print("\nExample completed. Check the generated images for visualizations.")
