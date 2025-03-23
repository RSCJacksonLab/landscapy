"""
Example of using Walsh-Hadamard transforms for epistasis analysis.

This example demonstrates how to use Walsh-Hadamard transforms to analyze
epistasis in fitness landscapes, including multiallelic sequences.
"""

import numpy as np
import matplotlib.pyplot as plt
from fitness_landscape import (
    BinarySequence,
    MultialleleSequence,
    FitnessLandscape,
    generate_sequences
)
from fitness_landscape.transforms import (
    walsh_transform,
    walsh_coefficients,
    MultialleleWalshTransform
)
from fitness_landscape.analysis import (
    calculate_epistasis,
    epistasis_decomposition
)

# Part 1: Binary sequences with Walsh transform
print("Part 1: Binary sequences with Walsh transform")
print("--------------------------------------------")

# Generate all binary sequences of length 4
binary_sequences = generate_sequences(4, [0, 1], strategy='complete')
print(f"Generated {len(binary_sequences)} binary sequences")

# Create fitness function with known epistasis
def fitness_function(seq):
    # Convert sequence to array
    arr = seq.to_array()
    
    # Baseline fitness
    fitness = 0.5
    
    # First-order effects (additive)
    fitness += 0.1 * arr[0]
    fitness += 0.2 * arr[1]
    fitness += 0.15 * arr[2]
    fitness += 0.05 * arr[3]
    
    # Second-order effects (pairwise epistasis)
    fitness += 0.3 * arr[0] * arr[1]  # Strong interaction between pos 0 and 1
    fitness += 0.1 * arr[1] * arr[2]  # Weak interaction between pos 1 and 2
    
    # Third-order effect (higher-order epistasis)
    fitness += 0.2 * arr[0] * arr[1] * arr[3]
    
    return fitness

# Calculate fitness for each sequence
binary_fitness = [fitness_function(seq) for seq in binary_sequences]

# Create fitness landscape
binary_landscape = FitnessLandscape(binary_sequences, binary_fitness)

# Calculate epistasis using Walsh transform
epistasis = calculate_epistasis(binary_landscape, order=4, method='walsh')

print("\nEpistasis coefficients:")
for order, coeffs in epistasis['by_order'].items():
    print(f"\nOrder {order} coefficients:")
    for term, value in coeffs.items():
        print(f"  {term}: {value:.4f}")

# Decompose fitness into epistatic components
decomposition = epistasis_decomposition(binary_landscape, method='walsh', order=4)

print("\nVariance explained by each order:")
for order, variance in decomposition['variance_explained'].items():
    print(f"  Order {order}: {variance:.4f} ({variance*100:.1f}%)")

# Plot variance explained
orders = list(decomposition['variance_explained'].keys())
variances = [decomposition['variance_explained'][o] for o in orders]

plt.figure(figsize=(8, 5))
plt.bar(orders, variances)
plt.xlabel('Order')
plt.ylabel('Variance Explained')
plt.title('Variance Explained by Epistatic Order')
plt.xticks(orders)
plt.grid(True, axis='y')
plt.savefig('variance_explained.png')

# Part 2: Multiallelic sequences with extended Walsh transform
print("\n\nPart 2: Multiallelic sequences with extended Walsh transform")
print("----------------------------------------------------------")

# Generate sequences with multiple possible values at each position
# Position 0: 3 possible values (0, 1, 2)
# Position 1: 2 possible values (0, 1)
# Position 2: 4 possible values (0, 1, 2, 3)
multi_sequences = []

for val0 in range(3):
    for val1 in range(2):
        for val2 in range(4):
            multi_sequences.append(MultialleleSequence([val0, val1, val2]))

print(f"Generated {len(multi_sequences)} multiallelic sequences")

# Create fitness function for multiallelic sequences
def multi_fitness_function(seq):
    # Convert sequence to array
    arr = seq.to_array()
    
    # Baseline fitness
    fitness = 0.5
    
    # First-order effects
    fitness += 0.1 * arr[0]  # Linear effect of position 0
    fitness += 0.2 * (arr[1] == 1)  # Binary effect of position 1
    
    # Position 2 has non-linear effect
    if arr[2] == 0:
        fitness += 0.0
    elif arr[2] == 1:
        fitness += 0.1
    elif arr[2] == 2:
        fitness += 0.3
    else:  # arr[2] == 3
        fitness += 0.2
    
    # Interaction between positions 0 and 1
    if arr[0] == 2 and arr[1] == 1:
        fitness += 0.4  # Strong epistasis for specific combination
    
    # Add some noise
    fitness += np.random.normal(0, 0.01)
    
    return fitness

# Calculate fitness for each sequence
multi_fitness = [multi_fitness_function(seq) for seq in multi_sequences]

# Create fitness landscape
multi_landscape = FitnessLandscape(multi_sequences, multi_fitness)

# Create multiallelic Walsh transform
alphabet_sizes = [3, 2, 4]  # Number of possible values at each position
transform = MultialleleWalshTransform(alphabet_sizes)

# Compute transform
coefficients = transform.transform(multi_landscape)

# Print top coefficients by magnitude
print("\nTop 10 multiallelic Walsh coefficients by magnitude:")
coef_magnitudes = np.abs(coefficients)
top_indices = np.argsort(coef_magnitudes)[::-1][:10]

for i, idx in enumerate(top_indices):
    print(f"  {i+1}. Coefficient {idx}: {coefficients[idx]:.4f}")

print("\nExample completed. Check the generated images for visualizations.")
