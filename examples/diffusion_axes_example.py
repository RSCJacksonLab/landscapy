# Example: Using Diffusion Axes Visualization

"""
This example demonstrates how to use the Diffusion Axes Visualization method
to create low-dimensional representations of fitness landscapes.

The method plots genotypes in a manner that captures important features of the landscape
using diffusion maps, as described by McCandlish.
"""

import numpy as np
import matplotlib.pyplot as plt
from fitness_landscape.core.sequence import BinarySequence
from fitness_landscape.core.landscape import FitnessLandscape
from fitness_landscape.utils.visualization import DiffusionAxesVisualization

# Create a fitness landscape for visualization
# We'll use a 5-bit binary landscape
sequence_length = 5
alphabet_size = 2

# Generate all possible sequences
all_sequences = [format(i, f'0{sequence_length}b') for i in range(2**sequence_length)]

# Create fitness values with some structure
# We'll make positions 0 and 2 contribute more to fitness
fitnesses = []
for seq in all_sequences:
    base_fitness = 0
    # Add contribution from each position
    for i, bit in enumerate(seq):
        if bit == "1":
            if i in [0, 2]:  # Positions 0 and 2 have higher contribution
                base_fitness += 2.0
            else:
                base_fitness += 0.5
    
    # Add epistatic interaction between position 0 and 2
    if seq[0] == "1" and seq[2] == "1":
        base_fitness += 3.0
    
    # Add some noise
    base_fitness += np.random.normal(0, 0.2)
    
    fitnesses.append(base_fitness)

# Create a dictionary mapping sequences to fitness values
landscape_dict = {seq: fit for seq, fit in zip(all_sequences, fitnesses)}

# Create a FitnessLandscape object
landscape = FitnessLandscape(landscape_dict)

print(f"Created landscape with {len(landscape_dict)} sequences")

# Initialize the DiffusionAxesVisualization model
diffusion_viz = DiffusionAxesVisualization(
    sequence_length=sequence_length, 
    alphabet_size=alphabet_size,
    n_components=3  # We'll compute 3 diffusion components
)

# Compute diffusion coordinates
diffusion_coords = diffusion_viz.fit_transform(landscape)

print(f"Computed diffusion coordinates for {len(diffusion_coords)} sequences")

# Create 2D visualization
plt.figure(figsize=(10, 8))
ax = diffusion_viz.plot_landscape(
    dimensions=[0, 1],  # Use first two diffusion axes
    fitness_values=landscape_dict,
    title="Fitness Landscape - Diffusion Axes Visualization (2D)",
    colormap="viridis"
)
plt.savefig('diffusion_axes_2d.png')
plt.close()

# Create 3D visualization
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
ax = diffusion_viz.plot_landscape(
    ax=ax,
    dimensions=[0, 1, 2],  # Use first three diffusion axes
    fitness_values=landscape_dict,
    title="Fitness Landscape - Diffusion Axes Visualization (3D)",
    colormap="viridis"
)
plt.savefig('diffusion_axes_3d.png')
plt.close()

# Analyze eigenvalues to understand the importance of each diffusion component
eigenvalues = diffusion_viz.get_eigenvalues()
plt.figure(figsize=(10, 6))
plt.bar(range(len(eigenvalues)), eigenvalues)
plt.xlabel('Component Index')
plt.ylabel('Eigenvalue')
plt.title('Eigenvalues of Diffusion Operator')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('diffusion_eigenvalues.png')
plt.close()

# Visualize evolutionary paths
# Let's create some paths from low fitness to high fitness sequences
# First, find sequences with lowest and highest fitness
sorted_seqs = sorted(landscape_dict.items(), key=lambda x: x[1])
lowest_seq = sorted_seqs[0][0]
highest_seq = sorted_seqs[-1][0]

# Create a simple path by flipping one bit at a time
def hamming_neighbors(seq):
    """Get all sequences that differ by one bit"""
    neighbors = []
    for i in range(len(seq)):
        neighbor = seq[:i] + ('1' if seq[i] == '0' else '0') + seq[i+1:]
        neighbors.append(neighbor)
    return neighbors

# Find a greedy path from lowest to highest
current_seq = lowest_seq
path = [current_seq]

while current_seq != highest_seq:
    neighbors = hamming_neighbors(current_seq)
    # Find neighbor with highest fitness
    best_neighbor = max(neighbors, key=lambda seq: landscape_dict.get(seq, -float('inf')))
    current_seq = best_neighbor
    path.append(current_seq)
    
    # Avoid infinite loops
    if len(path) > 10:
        break

print(f"Created evolutionary path with {len(path)} steps:")
for i, seq in enumerate(path):
    print(f"Step {i}: {seq} (Fitness: {landscape_dict[seq]:.2f})")

# Visualize the path on diffusion axes
plt.figure(figsize=(10, 8))
ax = diffusion_viz.plot_landscape(
    dimensions=[0, 1],
    fitness_values=landscape_dict,
    title="Evolutionary Path on Diffusion Axes",
    colormap="viridis"
)
diffusion_viz.plot_paths([path], ax=ax, labels=["Greedy Path"])
plt.savefig('evolutionary_path.png')
plt.close()

print("\nVisualization complete. Images saved to:")
print("- diffusion_axes_2d.png")
print("- diffusion_axes_3d.png")
print("- diffusion_eigenvalues.png")
print("- evolutionary_path.png")
