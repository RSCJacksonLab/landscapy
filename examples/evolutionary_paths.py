"""
Example of evolutionary path analysis in fitness landscapes.

This example demonstrates how to analyze evolutionary paths, accessible trajectories,
and basins of attraction in fitness landscapes.
"""

import numpy as np
import matplotlib.pyplot as plt
from fitness_landscape import (
    BinarySequence,
    FitnessLandscape,
    create_hamming_graph,
    generate_sequences
)
from fitness_landscape.analysis import (
    find_accessible_paths,
    find_shortest_paths,
    analyze_path_accessibility,
    find_evolutionary_trajectories,
    calculate_basin_of_attraction
)

# Generate all binary sequences of length 4
sequences = generate_sequences(4, [0, 1], strategy='complete')
print(f"Generated {len(sequences)} binary sequences")

# Create a fitness landscape with multiple peaks
def fitness_function(seq):
    # Convert sequence to array
    arr = seq.to_array()
    
    # Create two peaks: one at 0000 and one at 1111
    if sum(arr) == 0:  # 0000
        return 0.9
    elif sum(arr) == 4:  # 1111
        return 1.0
    else:
        # Fitness increases with Hamming weight, but with a valley in the middle
        hamming_weight = sum(arr)
        if hamming_weight == 2:
            return 0.3  # Create a valley
        else:
            return 0.4 + 0.1 * hamming_weight
    
# Calculate fitness for each sequence
fitness_values = [fitness_function(seq) for seq in sequences]

# Create fitness landscape
landscape = FitnessLandscape(sequences, fitness_values)

# Create Hamming graph
landscape.graph = create_hamming_graph(sequences, fitness_values)

print("\nPart 1: Accessible Paths Analysis")
print("-------------------------------")

# Find accessible paths between two sequences
start_seq = sequences[0]  # 0000
end_seq = sequences[-1]   # 1111

accessible_paths = find_accessible_paths(landscape, start_seq, end_seq)

print(f"Number of accessible paths from {start_seq} to {end_seq}: {accessible_paths['path_count']}")
if accessible_paths['path_count'] > 0:
    print(f"Mean path length: {accessible_paths['mean_path_length']:.2f}")
    print(f"Shortest path length: {accessible_paths['min_path_length']}")
    print(f"Longest path length: {accessible_paths['max_path_length']}")
    
    # Print the first path
    if accessible_paths['paths']:
        print("\nFirst accessible path:")
        path = accessible_paths['paths'][0]
        for i, (seq, fitness) in enumerate(zip(path['sequences'], path['fitness'])):
            print(f"  Step {i}: {seq} (fitness: {fitness:.4f})")
else:
    print("No accessible paths found")

# Find shortest paths
shortest_paths = find_shortest_paths(landscape, start_seq, end_seq)

print(f"\nNumber of shortest paths: {shortest_paths['path_count']}")
print(f"Shortest path length: {shortest_paths['path_length']}")
print(f"Number of accessible shortest paths: {shortest_paths['accessible_count']}")
print(f"Fraction of accessible shortest paths: {shortest_paths['accessible_fraction']:.4f}")

print("\nPart 2: Path Accessibility Analysis")
print("--------------------------------")

# Analyze path accessibility across the landscape
accessibility = analyze_path_accessibility(landscape)

print(f"Number of local minima: {accessibility['minima_count']}")
print(f"Number of local maxima: {accessibility['maxima_count']}")
print(f"Overall accessibility: {accessibility['accessibility']:.4f}")
print(f"Accessible pairs: {accessibility['accessible_pairs']} out of {accessibility['total_pairs']}")

# Plot accessibility matrix
if accessibility['minima_count'] > 0 and accessibility['maxima_count'] > 0:
    # Create accessibility matrix
    minima = accessibility['local_minima']
    maxima = accessibility['local_maxima']
    
    matrix = np.zeros((len(minima), len(maxima)))
    
    for i, min_idx in enumerate(minima):
        for j, max_idx in enumerate(maxima):
            if max_idx in accessibility['paths_to_maxima'].get(min_idx, {}):
                matrix[i, j] = accessibility['paths_to_maxima'][min_idx][max_idx]
    
    # Plot matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap='viridis')
    plt.colorbar(label='Number of Accessible Paths')
    plt.xlabel('Local Maxima')
    plt.ylabel('Local Minima')
    plt.title('Accessibility Matrix')
    
    # Add labels
    plt.xticks(range(len(maxima)), [str(sequences[idx]) for idx in maxima], rotation=90)
    plt.yticks(range(len(minima)), [str(sequences[idx]) for idx in minima])
    
    plt.tight_layout()
    plt.savefig('accessibility_matrix.png')

print("\nPart 3: Evolutionary Trajectories")
print("------------------------------")

# Simulate evolutionary trajectories
trajectories = find_evolutionary_trajectories(
    landscape, 
    start_seq=sequences[5],  # Start from an intermediate sequence
    max_steps=10, 
    n_trajectories=20
)

print(f"Simulated {trajectories['n_trajectories']} trajectories")
print(f"Mean trajectory length: {trajectories['mean_length']:.2f}")
print(f"Mean fitness gain: {trajectories['mean_fitness_gain']:.4f}")
print(f"Fraction reaching local optimum: {trajectories['optimum_fraction']:.4f}")

# Plot fitness trajectories
plt.figure(figsize=(10, 6))
for i, traj in enumerate(trajectories['trajectories']):
    if i < 10:  # Plot first 10 trajectories for clarity
        plt.plot(traj['fitness'], 'o-', alpha=0.7, label=f'Trajectory {i+1}')
plt.xlabel('Step')
plt.ylabel('Fitness')
plt.title('Evolutionary Trajectories')
plt.grid(True)
plt.legend()
plt.savefig('evolutionary_trajectories.png')

print("\nPart 4: Basin of Attraction Analysis")
print("---------------------------------")

# Find the global optimum
global_opt_idx = np.argmax(fitness_values)
global_optimum = sequences[global_opt_idx]

# Calculate basin of attraction for global optimum
basin = calculate_basin_of_attraction(landscape, global_optimum)

print(f"Global optimum: {global_optimum}")
print(f"Basin size: {basin['basin_size']} sequences")
print(f"Basin fraction: {basin['basin_fraction']:.4f}")

# Create a new landscape with multiple equal peaks
def multi_peak_fitness(seq):
    # Convert sequence to array
    arr = seq.to_array()
    
    # Create three equal peaks
    if np.array_equal(arr, [0, 0, 0, 0]):  # 0000
        return 1.0
    elif np.array_equal(arr, [1, 1, 1, 1]):  # 1111
        return 1.0
    elif np.array_equal(arr, [0, 1, 0, 1]):  # 0101
        return 1.0
    else:
        # Base fitness depends on distance to nearest peak
        dist_to_0000 = sum(arr)
        dist_to_1111 = 4 - sum(arr)
        dist_to_0101 = sum(abs(arr - np.array([0, 1, 0, 1])))
        min_dist = min(dist_to_0000, dist_to_1111, dist_to_0101)
        return 0.9 - 0.2 * min_dist

# Calculate fitness for multi-peak landscape
multi_peak_fitness_values = [multi_peak_fitness(seq) for seq in sequences]

# Create multi-peak fitness landscape
multi_peak_landscape = FitnessLandscape(sequences, multi_peak_fitness_values)
multi_peak_landscape.graph = create_hamming_graph(sequences, multi_peak_fitness_values)

# Find all local optima
multi_peak_optima = []
for i, seq in enumerate(sequences):
    # Get fitness
    fitness = multi_peak_landscape.get_fitness(seq)
    
    # Get neighbors
    neighbors = list(multi_peak_landscape.graph.neighbors(i))
    
    # Check if local optimum
    is_optimum = True
    for neighbor in neighbors:
        if multi_peak_landscape.get_fitness(sequences[neighbor]) > fitness:
            is_optimum = False
            break
    
    if is_optimum:
        multi_peak_optima.append(i)

print(f"\nMulti-peak landscape has {len(multi_peak_optima)} local optima")

# Calculate basin for each optimum
basins = {}
for opt_idx in multi_peak_optima:
    optimum = sequences[opt_idx]
    basin = calculate_basin_of_attraction(multi_peak_landscape, optimum)
    basins[opt_idx] = basin
    print(f"Optimum {optimum}: Basin size = {basin['basin_size']} ({basin['basin_fraction']:.4f})")

print("\nExample completed. Check the generated images for visualizations.")
