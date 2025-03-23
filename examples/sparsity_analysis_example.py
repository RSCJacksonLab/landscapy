# Example: Using Sparsity Analysis Methods

"""
This example demonstrates how to use the Sparsity Analysis methods
based on Jennifer Listgarten's work to analyze the sparsity of fitness landscapes
and its implications for learning.

The methods leverage the observation that empirical fitness functions display
substantial sparsity when represented in terms of epistatic interactions.
"""

import numpy as np
import matplotlib.pyplot as plt
from fitness_landscape.core.sequence import BinarySequence
from fitness_landscape.core.landscape import FitnessLandscape
from fitness_landscape.analysis.sparsity import SparsityAnalysis

# Create a fitness landscape for sparsity analysis
# We'll use a 6-bit binary landscape
sequence_length = 6
alphabet_size = 2

# Initialize the SparsityAnalysis model
sparsity_analyzer = SparsityAnalysis(sequence_length=sequence_length, alphabet_size=alphabet_size)

# Generate a GNK model landscape with adjacent interactions
print("Generating GNK model landscape with adjacent interactions...")
gnk_adjacent = sparsity_analyzer.generate_gnk_model(
    interaction_structure='adjacent',
    random_seed=42
)

# Generate a GNK model landscape with random interactions
print("Generating GNK model landscape with random interactions...")
gnk_random = sparsity_analyzer.generate_gnk_model(
    interaction_structure='random',
    random_seed=42
)

# Create a custom landscape with known sparsity properties
# We'll make only a few specific interactions matter
print("Creating custom sparse landscape...")
all_sequences = [format(i, f'0{sequence_length}b') for i in range(2**sequence_length)]
custom_landscape = {}

for seq in all_sequences:
    fitness = 0.0
    
    # Add contribution from each position
    for i, bit in enumerate(seq):
        if bit == '1':
            fitness += 0.5
    
    # Add specific epistatic interactions
    # Only positions 0-1, 2-3, and 4-5 interact
    if seq[0] == '1' and seq[1] == '1':
        fitness += 2.0
    if seq[2] == '1' and seq[3] == '1':
        fitness += 1.5
    if seq[4] == '1' and seq[5] == '1':
        fitness += 1.0
    
    # Add some noise
    fitness += np.random.normal(0, 0.1, 1)[0]
    
    custom_landscape[seq] = fitness

# Estimate sparsity for each landscape
print("\nEstimating sparsity for each landscape...")
sparsity_gnk_adjacent = sparsity_analyzer.estimate_sparsity(gnk_adjacent)
print(f"GNK Adjacent Interactions Sparsity: {sparsity_gnk_adjacent:.4f}")

# Reset sparsity analyzer for next landscape
sparsity_analyzer = SparsityAnalysis(sequence_length=sequence_length, alphabet_size=alphabet_size)
sparsity_gnk_random = sparsity_analyzer.estimate_sparsity(gnk_random)
print(f"GNK Random Interactions Sparsity: {sparsity_gnk_random:.4f}")

# Reset sparsity analyzer for next landscape
sparsity_analyzer = SparsityAnalysis(sequence_length=sequence_length, alphabet_size=alphabet_size)
sparsity_custom = sparsity_analyzer.estimate_sparsity(custom_landscape)
print(f"Custom Sparse Landscape Sparsity: {sparsity_custom:.4f}")

# Calculate sample complexity for each landscape
print("\nCalculating sample complexity for each landscape...")
samples_gnk_adjacent = sparsity_analyzer.calculate_sample_complexity(sparsity_gnk_adjacent)
samples_gnk_random = sparsity_analyzer.calculate_sample_complexity(sparsity_gnk_random)
samples_custom = sparsity_analyzer.calculate_sample_complexity(sparsity_custom)

print(f"GNK Adjacent: {samples_gnk_adjacent} samples needed (out of {2**sequence_length} possible)")
print(f"GNK Random: {samples_gnk_random} samples needed (out of {2**sequence_length} possible)")
print(f"Custom Sparse: {samples_custom} samples needed (out of {2**sequence_length} possible)")

# Compare empirical (custom) landscape with GNK model
print("\nComparing custom landscape with GNK model...")
comparison = sparsity_analyzer.compare_empirical_vs_gnk(
    custom_landscape, 
    gnk_adjacent,
    plot=True
)

print("\nComparison metrics:")
print(f"Correlation: {comparison['correlation']:.4f}")
print(f"Mean Squared Error: {comparison['mse']:.4f}")
print(f"Empirical Sparsity: {comparison['empirical_sparsity']:.4f}")
print(f"GNK Model Sparsity: {comparison['gnk_sparsity']:.4f}")
print(f"Sample Complexity (Empirical): {comparison['sample_complexity_empirical']}")
print(f"Sample Complexity (GNK): {comparison['sample_complexity_gnk']}")

# Save the comparison figure
plt.savefig('sparsity_comparison.png')
plt.close()

# Visualize the distribution of significant Walsh coefficients
significant_coeffs = sparsity_analyzer.significant_coefficients
orders = {}

# Group coefficients by order (number of 1s in the binary pattern)
for pattern, coeff in significant_coeffs.items():
    order = bin(int(pattern, 2)).count('1')
    if order not in orders:
        orders[order] = []
    orders[order].append(abs(coeff))

# Plot distribution by order
plt.figure(figsize=(10, 6))
order_labels = sorted(orders.keys())
order_values = [np.mean(orders[order]) for order in order_labels]
order_errors = [np.std(orders[order]) for order in order_labels]

plt.bar(order_labels, order_values, yerr=order_errors, capsize=10)
plt.xlabel('Interaction Order')
plt.ylabel('Mean Coefficient Magnitude')
plt.title('Significant Walsh Coefficients by Order')
plt.xticks(order_labels)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('walsh_coefficients_by_order.png')
plt.close()

# Demonstrate subsampling and recovery
print("\nDemonstrating subsampling and recovery...")

# Take a random subset of the custom landscape
np.random.seed(42)
sample_sizes = [10, 20, 50, 100, 200]
recovery_errors = []

for sample_size in sample_sizes:
    # Sample sequences
    sampled_seqs = np.random.choice(all_sequences, size=sample_size, replace=False)
    sampled_landscape = {seq: custom_landscape[seq] for seq in sampled_seqs}
    
    # Create a new sparsity analyzer
    subsample_analyzer = SparsityAnalysis(sequence_length=sequence_length, alphabet_size=alphabet_size)
    
    # Estimate sparsity from the subsample
    subsample_sparsity = subsample_analyzer.estimate_sparsity(sampled_landscape)
    
    # Calculate error in recovered landscape
    errors = []
    for seq in all_sequences:
        if seq not in sampled_seqs:
            # Predict fitness using Walsh coefficients
            true_fitness = custom_landscape[seq]
            
            # Reconstruct fitness from significant Walsh coefficients
            reconstructed = 0
            for pattern, coeff in subsample_analyzer.significant_coefficients.items():
                # Check if pattern matches sequence
                match = True
                for i in range(sequence_length):
                    if pattern[i] == '1' and seq[i] == '0':
                        match = False
                        break
                
                if match:
                    reconstructed += coeff
            
            errors.append((true_fitness - reconstructed) ** 2)
    
    mse = np.mean(errors)
    recovery_errors.append(mse)
    
    print(f"Sample size: {sample_size}, Sparsity: {subsample_sparsity:.4f}, MSE: {mse:.4f}")

# Plot recovery error vs sample size
plt.figure(figsize=(10, 6))
plt.plot(sample_sizes, recovery_errors, 'o-', linewidth=2)
plt.xlabel('Sample Size')
plt.ylabel('Mean Squared Error')
plt.title('Recovery Error vs Sample Size')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('recovery_error.png')
plt.close()

print("\nSparsity analysis complete. Visualizations saved to:")
print("- sparsity_comparison.png")
print("- walsh_coefficients_by_order.png")
print("- recovery_error.png")
