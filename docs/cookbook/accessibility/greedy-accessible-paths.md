# Enumerate greedy accessible paths

Greedy accessible paths move only along represented edges with strictly
increasing active fitness. Accessible mutational paths are central to analyses
of historical contingency and epistasis; see
[Weinreich et al. (2006)](https://doi.org/10.1126/science.1123539),
[Poelwijk et al. (2007)](https://doi.org/10.1038/nature05451), and
[Franke et al. (2011)](https://doi.org/10.1371/journal.pcbi.1002134).

## Input

Start and end variants must be unique landscape rows in the same represented
component. State the improvement direction and how equal-fitness edges are
handled; the current function excludes ties.

## Worked example

```python
# cookbook: test
import numpy as np

from fitness_landscape import BinarySequence
from fitness_landscape.analysis import analyze_path_accessibility, find_greedy_accessible_paths
from fitness_landscape.core import FitnessLandscape, NumericFitness

sequences = [BinarySequence(f"{value:03b}", sequence_id=f"g{value}") for value in range(8)]
fitness = [sum(int(site) for site in f"{value:03b}") for value in range(8)]
landscape = FitnessLandscape.build(
    sequences,
    graph="hamming",
    fitness_layers={"additive": NumericFitness.from_scalars("additive", fitness)},
)
landscape.view("additive")

paths = find_greedy_accessible_paths(landscape, sequences[0], sequences[7])
summary = analyze_path_accessibility(landscape)

assert paths["start_sequence"].id == "g0" and paths["end_sequence"].id == "g7"
assert paths["start_fitness"] < paths["end_fitness"]
assert paths["path_count"] == 6
assert paths["min_path_length"] == paths["max_path_length"] == 3
for path in paths["paths"]:
    assert np.all(np.diff(path["fitness"]) > 0)
    assert path["nodes"][0] == paths["start_node"]
    assert path["nodes"][-1] == paths["end_node"]

assert summary["minima_count"] == summary["maxima_count"] == 1
assert summary["accessible_pairs"] == summary["total_pairs"] == 1
print(paths["path_count"], paths["mean_path_length"], summary["accessibility"])
```

The complete three-site additive cube has `3! = 6` strictly increasing
shortest paths from `000` to `111`.
