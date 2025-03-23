# Example: Using Minimum Epistasis Interpolation

"""
This example demonstrates how to use the Minimum Epistasis Interpolation method
to infer the least epistatic possible sequence-function relationship compatible
with available data.

The method works by minimizing the expected squared epistatic coefficient for
random pairs of mutations across genetic backgrounds.
"""

import numpy as np
import matplotlib.pyplot as plt
from fitness_landscape.core.sequence import BinarySequence
from fitness_landscape.core.landscape import FitnessLandscape
from fitness_landscape.models.minimum_epistasis import MinimumEpistasisInterpolation

# Create a simple fitness landscape with some missing data
# We'll use a 4-bit binary landscape
sequence_length = 4
alphabet_size = 2

# Generate some sequences and fitness values
sequences = [
    "0000", "0001", "0010", "0011", 
    "0100", "0101", "0110", "0111",
    "1000", "1001", "1010"  # Note: We're missing 5 sequences
]

# Create fitness values with some epistasis
# We'll make position 0 and 2 interact
fitnesses = []
for seq in sequences:
    base_fitness = 0
    # Add contribution from each position
    for i, bit in enumerate(seq):
        if bit == "1":
            base_fitness += (i + 1) * 0.5
    
    # Add epistatic interaction between position 0 and 2
    if seq[0] == "1" and seq[2] == "1":
        base_fitness += 2.0
    
    fitnesses.append(base_fitness)

# Create a dictionary mapping sequences to fitness values
observed_landscape = {seq: fit for seq, fit in zip(sequences, fitnesses)}

print("Observed landscape:")
for seq, fit in observed_landscape.items():
    print(f"{seq}: {fit:.2f}")

# Create a FitnessLandscape object
landscape = FitnessLandscape(observed_landscape)

# Initialize and fit the Minimum Epistasis Interpolation model
mei = MinimumEpistasisInterpolation(sequence_length=sequence_length, alphabet_size=alphabet_size)
mei.fit(landscape)

# Get the missing sequences
all_sequences = [format(i, f'0{sequence_length}b') for i in range(2**sequence_length)]
missing_sequences = [seq for seq in all_sequences if seq not in sequences]

print("\nPredicted fitness for missing sequences:")
for seq in missing_sequences:
    predicted_fitness = mei.predict([seq])[0]
    print(f"{seq}: {predicted_fitness:.2f}")

# Get epistatic coefficients
epistatic_coeffs = mei.get_epistatic_coefficients()

print("\nEpistatic coefficients:")
for pattern, coeff in sorted(epistatic_coeffs.items(), key=lambda x: bin(int(x[0], 2)).count('1')):
    if abs(coeff) > 0.1:  # Only show significant coefficients
        order = bin(int(pattern, 2)).count('1')
        print(f"Pattern {pattern} (Order {order}): {coeff:.4f}")

# Get epistasis statistics
stats = mei.get_epistasis_statistics()

print("\nEpistasis statistics:")
print(f"Mean coefficient: {stats['mean']:.4f}")
print(f"Standard deviation: {stats['std']:.4f}")
print(f"Min coefficient: {stats['min']:.4f}")
print(f"Max coefficient: {stats['max']:.4f}")

print("\nStatistics by order:")
for order, order_stats in stats['by_order'].items():
    print(f"Order {order}:")
    print(f"  Mean: {order_stats['mean']:.4f}")
    print(f"  Std: {order_stats['std']:.4f}")
    print(f"  Count: {order_stats['count']}")

# Visualize the epistatic coefficients
orders = range(1, sequence_length + 1)
mean_by_order = [stats['by_order'].get(order, {'mean': 0})['mean'] for order in orders]
std_by_order = [stats['by_order'].get(order, {'std': 0})['std'] for order in orders]

plt.figure(figsize=(10, 6))
plt.bar(orders, mean_by_order, yerr=std_by_order, capsize=10)
plt.xlabel('Interaction Order')
plt.ylabel('Mean Coefficient Magnitude')
plt.title('Epistatic Coefficients by Order')
plt.xticks(orders)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('epistasis_by_order.png')
plt.close()

# Compare the complete landscape with the interpolated one
complete_landscape = {}
for seq in all_sequences:
    if seq in observed_landscape:
        complete_landscape[seq] = observed_landscape[seq]
    else:
        complete_landscape[seq] = mei.predict([seq])[0]

# Plot the complete landscape
fitness_values = [complete_landscape[seq] for seq in all_sequences]

plt.figure(figsize=(12, 6))
plt.bar(range(len(all_sequences)), fitness_values)
plt.xlabel('Sequence Index')
plt.ylabel('Fitness')
plt.title('Complete Fitness Landscape with Minimum Epistasis Interpolation')
plt.xticks(range(len(all_sequences)), all_sequences, rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('complete_landscape.png')
plt.close()

print("\nInterpolation complete. Visualizations saved to 'epistasis_by_order.png' and 'complete_landscape.png'")
