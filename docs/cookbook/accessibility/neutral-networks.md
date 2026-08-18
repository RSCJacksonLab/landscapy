# Analyze neutral networks

A neutral network is a connected component formed by represented edges whose
endpoint fitness difference is no larger than a declared threshold. Neutral
networks connect robustness and access to new variation; see
[Huynen, Stadler, and Fontana (1996)](https://doi.org/10.1073/PNAS.93.1.397),
[Wagner (2008)](https://doi.org/10.1098/rspb.2007.1137), and
[Romero and Arnold (2009)](https://doi.org/10.1038/nrm2805).

## Input

Choose the threshold from measurement resolution, replicate variability, or a
predeclared scientific equivalence margin—not from a visually convenient graph.

## Worked example

```python
# cookbook: test
from fitness_landscape import BinarySequence
from fitness_landscape.analysis import neutral_network_analysis
from fitness_landscape.core import FitnessLandscape, NumericFitness

sequences = [BinarySequence(f"{value:03b}", sequence_id=f"g{value}") for value in range(8)]
fitness = [0.00, 0.01, 0.02, 0.50, 0.00, 0.48, 0.52, 0.50]
landscape = FitnessLandscape.build(
    sequences,
    graph="hamming",
    fitness_layers={"assay": NumericFitness.from_scalars("assay", fitness)},
)
landscape.view("assay")

sensitivity = {}
for threshold in [0.0, 0.015, 0.03]:
    result = neutral_network_analysis(landscape, threshold=threshold)
    sensitivity[threshold] = {
        "network_count": result["network_count"],
        "largest_size": result["largest_network_size"],
        "largest_fraction": result["largest_network_fraction"],
        "sizes": [network["size"] for network in result["networks"]],
        "nodes": [network["sequence_indices"] for network in result["networks"]],
    }

assert sensitivity[0.0]["network_count"] == 6
assert sensitivity[0.015]["largest_size"] == 3
assert sensitivity[0.03]["sizes"] == [4, 4]
assert all(sum(item["sizes"]) == 8 for item in sensitivity.values())
print(sensitivity)
```

The component partition changes sharply between thresholds, which is the point
of reporting sensitivity. Relate neutral components to local optima or basins
only after preserving their distinct definitions and denominators.
